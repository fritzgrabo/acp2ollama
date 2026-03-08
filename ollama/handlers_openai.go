package ollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/fritzgrabo/acp2ollama/acp"
)

// handleV1Chat handles POST /v1/chat/completions.
func (s *Server) handleV1Chat(w http.ResponseWriter, r *http.Request) {
	var req V1ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages are required")
		return
	}

	msgs := toACPMessages(req.Messages)

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	model := modelName(req.Model, s.defaultModel)

	sess, err := s.chatPool.Acquire(msgs)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}

	sess.SwitchModelIfNeeded(r.Context(), model)

	chunkCh, err := sess.Prompt(r.Context(), msgs)
	if err != nil {
		s.chatPool.Release(sess, nil)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var ok bool
	if req.Stream == nil || *req.Stream {
		ok = streamV1Chat(w, chunkCh, model, id)
	} else {
		ok = nonStreamV1Chat(w, chunkCh, model, id)
	}

	var history []acp.Message
	if ok {
		history = msgs
	}
	s.chatPool.Release(sess, history)
}

func streamV1Chat(w http.ResponseWriter, chunkCh <-chan string, model, id string) bool {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	created := time.Now().Unix()
	stop := "stop"

	emit := func(v any) {
		data, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", data) //nolint:errcheck
		if flusher != nil {
			flusher.Flush()
		}
	}

	if !streamChunks(chunkCh,
		func(chunk string) {
			emit(V1ChatChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []V1ChatChunkChoice{
					{Delta: V1ChatDelta{Content: chunk}, Index: 0, FinishReason: nil},
				},
			})
		},
		func(errMsg string) { emit(ErrorResponse{Error: errMsg}) },
	) {
		return false
	}

	// Final chunk: empty delta, finish_reason="stop".
	emit(V1ChatChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []V1ChatChunkChoice{
			{Delta: V1ChatDelta{}, Index: 0, FinishReason: &stop},
		},
	})
	fmt.Fprint(w, "data: [DONE]\n\n") //nolint:errcheck
	if flusher != nil {
		flusher.Flush()
	}
	return true
}

func nonStreamV1Chat(w http.ResponseWriter, chunkCh <-chan string, model, id string) bool {
	text, err := collectChunks(chunkCh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return false
	}

	writeJSON(w, http.StatusOK, V1ChatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []V1ChatChoice{
			{Message: ChatMessage{Role: "assistant", Content: text}, Index: 0, FinishReason: "stop"},
		},
	})
	return true
}

// handleV1Completions handles POST /v1/completions.
//
// The prompt and optional suffix are combined into a fill-in-the-middle (FIM)
// format: the agent sees the text before the cursor, the cursor marker, and
// the text after. A system prompt instructs it to output only the insertion.
func (s *Server) handleV1Completions(w http.ResponseWriter, r *http.Request) {
	var req V1CompletionsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	userContent := req.Prompt + "<cursor>" + req.Suffix
	msgs := []acp.Message{
		{
			Role:    "system",
			Content: "You are a code completion engine. Complete the code at the position marked <cursor>. Output ONLY the raw text to insert at the cursor — no explanation, no markdown, no code fences, no surrounding context.",
		},
		{Role: "user", Content: userContent},
	}

	id := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	model := modelName(req.Model, s.defaultModel)

	sess, err := s.genPool.Acquire(r.Context())
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err.Error())
		return
	}

	sess.SwitchModelIfNeeded(r.Context(), model)

	chunkCh, err := sess.PromptDelta(r.Context(), msgs)
	if err != nil {
		s.genPool.Release(sess)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if req.Stream == nil || *req.Stream {
		streamV1Completions(w, chunkCh, model, id)
	} else {
		nonStreamV1Completions(w, chunkCh, model, id)
	}

	s.genPool.Release(sess)
}

func streamV1Completions(w http.ResponseWriter, chunkCh <-chan string, model, id string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	created := time.Now().Unix()
	stop := "stop"

	emit := func(v any) {
		data, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", data) //nolint:errcheck
		if flusher != nil {
			flusher.Flush()
		}
	}

	if !streamChunks(chunkCh,
		func(chunk string) {
			emit(V1CompletionsChunk{
				ID:      id,
				Object:  "text_completion",
				Created: created,
				Model:   model,
				Choices: []V1CompletionsChunkChoice{
					{Text: chunk, Index: 0, FinishReason: nil},
				},
			})
		},
		func(errMsg string) { emit(ErrorResponse{Error: errMsg}) },
	) {
		return
	}

	emit(V1CompletionsChunk{
		ID:      id,
		Object:  "text_completion",
		Created: created,
		Model:   model,
		Choices: []V1CompletionsChunkChoice{
			{Text: "", Index: 0, FinishReason: &stop},
		},
	})
	fmt.Fprint(w, "data: [DONE]\n\n") //nolint:errcheck
	if flusher != nil {
		flusher.Flush()
	}
}

func nonStreamV1Completions(w http.ResponseWriter, chunkCh <-chan string, model, id string) {
	text, err := collectChunks(chunkCh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, V1CompletionsResponse{
		ID:      id,
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []V1CompletionsChoice{
			{Text: text, Index: 0, FinishReason: "stop"},
		},
	})
}
