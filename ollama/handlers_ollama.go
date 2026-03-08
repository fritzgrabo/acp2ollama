package ollama

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/fritzgrabo/acp2ollama/acp"
)

// handleChat handles POST /api/chat.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if isPresent(req.Format) {
		writeError(w, http.StatusNotImplemented, "format is not supported by acp2ollama")
		return
	}

	model := modelName(req.Model, s.defaultModel)

	// Empty messages = load/unload no-op (per Ollama spec).
	if len(req.Messages) == 0 {
		doneReason := "load"
		if isUnload(req.KeepAlive) {
			doneReason = "unload"
		}
		writeJSON(w, http.StatusOK, ChatResponseChunk{
			Model:      model,
			CreatedAt:  time.Now(),
			Message:    ChatMessage{Role: "assistant"},
			DoneReason: doneReason,
			Done:       true,
		})
		return
	}

	msgs := toACPMessages(req.Messages)

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

	start := time.Now()
	var ok bool
	if req.Stream == nil || *req.Stream {
		ok = streamChat(w, chunkCh, model, start)
	} else {
		ok = nonStreamChat(w, chunkCh, model, start)
	}

	var history []acp.Message
	if ok {
		history = msgs
	}
	s.chatPool.Release(sess, history)
}

func streamChat(w http.ResponseWriter, chunkCh <-chan string, model string, start time.Time) bool {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)
	enc := json.NewEncoder(w)
	flush := func() {
		if canFlush {
			flusher.Flush()
		}
	}

	if !streamChunks(chunkCh,
		func(chunk string) {
			enc.Encode(ChatResponseChunk{ //nolint:errcheck
				Model:     model,
				CreatedAt: time.Now(),
				Message:   ChatMessage{Role: "assistant", Content: chunk},
				Done:      false,
			})
			flush()
		},
		func(errMsg string) {
			enc.Encode(ErrorResponse{Error: errMsg}) //nolint:errcheck
			flush()
		},
	) {
		return false
	}

	enc.Encode(ChatResponseChunk{ //nolint:errcheck
		Model:         model,
		CreatedAt:     time.Now(),
		Message:       ChatMessage{Role: "assistant"},
		Done:          true,
		DoneReason:    "stop",
		TotalDuration: time.Since(start).Nanoseconds(),
	})
	flush()
	return true
}

func nonStreamChat(w http.ResponseWriter, chunkCh <-chan string, model string, start time.Time) bool {
	text, err := collectChunks(chunkCh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return false
	}

	writeJSON(w, http.StatusOK, ChatResponseChunk{
		Model:         model,
		CreatedAt:     time.Now(),
		Message:       ChatMessage{Role: "assistant", Content: text},
		Done:          true,
		DoneReason:    "stop",
		TotalDuration: time.Since(start).Nanoseconds(),
	})
	return true
}

// handleGenerate handles POST /api/generate.
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if isPresent(req.Format) {
		writeError(w, http.StatusNotImplemented, "format is not supported by acp2ollama")
		return
	}

	model := modelName(req.Model, s.defaultModel)

	// Empty prompt = load/unload no-op.
	if req.Prompt == "" {
		doneReason := "load"
		if isUnload(req.KeepAlive) {
			doneReason = "unload"
		}
		writeJSON(w, http.StatusOK, GenerateResponseChunk{
			Model:      model,
			CreatedAt:  time.Now(),
			DoneReason: doneReason,
			Done:       true,
		})
		return
	}

	var msgs []acp.Message
	if req.System != "" {
		msgs = append(msgs, acp.Message{Role: "system", Content: req.System})
	}
	msgs = append(msgs, acp.Message{Role: "user", Content: req.Prompt})

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

	emptyCtx := []int{}
	start := time.Now()
	if req.Stream == nil || *req.Stream {
		streamGenerate(w, chunkCh, model, start, &emptyCtx)
	} else {
		nonStreamGenerate(w, chunkCh, model, start, &emptyCtx)
	}

	s.genPool.Release(sess)
}

func streamGenerate(w http.ResponseWriter, chunkCh <-chan string, model string, start time.Time, finalCtx *[]int) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)
	enc := json.NewEncoder(w)
	flush := func() {
		if canFlush {
			flusher.Flush()
		}
	}

	if !streamChunks(chunkCh,
		func(chunk string) {
			enc.Encode(GenerateResponseChunk{ //nolint:errcheck
				Model:     model,
				CreatedAt: time.Now(),
				Response:  chunk,
				Done:      false,
			})
			flush()
		},
		func(errMsg string) {
			enc.Encode(ErrorResponse{Error: errMsg}) //nolint:errcheck
			flush()
		},
	) {
		return
	}

	enc.Encode(GenerateResponseChunk{ //nolint:errcheck
		Model:         model,
		CreatedAt:     time.Now(),
		Done:          true,
		DoneReason:    "stop",
		Context:       finalCtx,
		TotalDuration: time.Since(start).Nanoseconds(),
	})
	flush()
}

func nonStreamGenerate(w http.ResponseWriter, chunkCh <-chan string, model string, start time.Time, finalCtx *[]int) {
	text, err := collectChunks(chunkCh)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, GenerateResponseChunk{
		Model:         model,
		CreatedAt:     time.Now(),
		Response:      text,
		Done:          true,
		DoneReason:    "stop",
		Context:       finalCtx,
		TotalDuration: time.Since(start).Nanoseconds(),
	})
}

// handleTags handles GET /api/tags.
func (s *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	name := modelTag(s.defaultModel)
	writeJSON(w, http.StatusOK, TagsResponse{
		Models: []ModelInfo{
			{
				Name:       name,
				Model:      name,
				ModifiedAt: s.startTime,
				Details: ModelDetails{
					Format:            "acp",
					Family:            "acp",
					Families:          []string{"acp"},
					ParameterSize:     "unknown",
					QuantizationLevel: "unknown",
				},
			},
		},
	})
}

// handleVersion handles GET /api/version.
func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, VersionResponse{Version: s.version})
}

// handlePS handles GET /api/ps.
func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	name := modelTag(s.defaultModel)
	writeJSON(w, http.StatusOK, PSResponse{
		Models: []PSModelInfo{
			{
				Name:      name,
				Model:     name,
				ExpiresAt: "9999-12-31T23:59:59Z",
			},
		},
	})
}

// handleNotImplemented is the catch-all for unrecognised paths.
func (s *Server) handleNotImplemented(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented, "not supported by acp2ollama")
}

// --- Helpers ---

// writeJSON writes v as a JSON response with the given HTTP status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

// writeError writes a JSON error response.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, ErrorResponse{Error: msg})
}

// modelTag appends ":latest" to a model name.
func modelTag(name string) string { return name + ":latest" }

// modelName returns req if non-empty, otherwise fallback.
func modelName(req, fallback string) string {
	if req != "" {
		return req
	}
	return fallback
}

// toACPMessages converts a slice of ChatMessages to acp.Messages.
func toACPMessages(ms []ChatMessage) []acp.Message {
	out := make([]acp.Message, len(ms))
	for i, m := range ms {
		out[i] = acp.Message{Role: m.Role, Content: m.Content}
	}
	return out
}

// isPresent reports whether raw is a non-empty, non-null JSON value.
// Used to detect optional request fields that trigger 501 responses.
func isPresent(raw json.RawMessage) bool {
	s := bytes.TrimSpace(raw)
	return len(s) > 0 && string(s) != "null"
}

// isUnload reports whether a keep_alive value means "unload" (0 or "0").
func isUnload(raw json.RawMessage) bool {
	s := bytes.TrimSpace(raw)
	return string(s) == "0" || string(s) == `"0"`
}

// collectChunks accumulates all content chunks from chunkCh into a string.
// On an error chunk it drains the channel and returns the error.
func collectChunks(chunkCh <-chan string) (string, error) {
	var sb strings.Builder
	for chunk := range chunkCh {
		if errMsg, isErr := acp.IsErrorChunk(chunk); isErr {
			drain(chunkCh)
			return "", errors.New(errMsg)
		}
		sb.WriteString(chunk)
	}
	return sb.String(), nil
}

// streamChunks ranges over chunkCh, calling emit for each content chunk.
// On an error chunk it calls onErr with the message, drains the channel, and
// returns false. Returns true when the channel closes normally.
func streamChunks(chunkCh <-chan string, emit func(string), onErr func(string)) bool {
	for chunk := range chunkCh {
		if errMsg, isErr := acp.IsErrorChunk(chunk); isErr {
			onErr(errMsg)
			drain(chunkCh)
			return false
		}
		emit(chunk)
	}
	return true
}

// drain discards all remaining values from ch. Called after an error chunk is
// received to ensure the goroutine in startPrompt has fully exited before the
// session is released back to the pool.
func drain(ch <-chan string) {
	for range ch {
	}
}
