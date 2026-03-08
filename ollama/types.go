// Package ollama implements an Ollama-compatible HTTP server backed by ACP sessions.
package ollama

import (
	"encoding/json"
	"time"
)

// ChatRequest is the body of POST /api/chat.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []ChatMessage   `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	Tools     json.RawMessage `json:"tools,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

// ChatMessage is a single message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponseChunk is one NDJSON object in a streaming /api/chat response.
// For non-streaming responses, it is the sole response body.
type ChatResponseChunk struct {
	Model         string      `json:"model"`
	CreatedAt     time.Time   `json:"created_at"`
	Message       ChatMessage `json:"message"`
	Done          bool        `json:"done"`
	DoneReason    string      `json:"done_reason,omitempty"`
	TotalDuration int64       `json:"total_duration,omitempty"`
}

// GenerateRequest is the body of POST /api/generate.
type GenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	System    string          `json:"system,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

// GenerateResponseChunk is one NDJSON object in a streaming /api/generate response.
// For non-streaming, it is the sole response body.
type GenerateResponseChunk struct {
	Model         string `json:"model"`
	CreatedAt     time.Time `json:"created_at"`
	Response      string `json:"response"`
	Done          bool   `json:"done"`
	DoneReason    string `json:"done_reason,omitempty"`
	// Context uses *[]int (pointer to slice) so that nil → omitted but &[]int{} → [].
	// A plain []int with omitempty would omit even a non-nil empty slice.
	Context       *[]int `json:"context,omitempty"`
	TotalDuration int64  `json:"total_duration,omitempty"`
}

// TagsResponse is the body of GET /api/tags.
type TagsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelInfo is one entry in a TagsResponse.
type ModelInfo struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	ModifiedAt time.Time    `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

// ModelDetails is the details sub-object in a ModelInfo.
type ModelDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// PSResponse is the body of GET /api/ps.
type PSResponse struct {
	Models []PSModelInfo `json:"models"`
}

// PSModelInfo is one entry in a PSResponse.
type PSModelInfo struct {
	Name      string `json:"name"`
	Model     string `json:"model"`
	Size      int64  `json:"size"`
	Digest    string `json:"digest"`
	ExpiresAt string `json:"expires_at"`
	SizeVram  int64  `json:"size_vram"`
}

// VersionResponse is the body of GET /api/version.
type VersionResponse struct {
	Version string `json:"version"`
}

// ErrorResponse is the body of all error responses.
type ErrorResponse struct {
	Error string `json:"error"`
}

// ── OpenAI-compatible /v1/chat/completions ────────────────────────────────────

// V1ChatRequest is the body of POST /v1/chat/completions.
type V1ChatRequest struct {
	Model    string          `json:"model"`
	Messages []ChatMessage   `json:"messages"`
	Stream   *bool           `json:"stream,omitempty"`
	Tools    json.RawMessage `json:"tools,omitempty"`
}

// V1ChatChunk is one SSE event in a streaming /v1/chat/completions response.
type V1ChatChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []V1ChatChunkChoice `json:"choices"`
}

// V1ChatChunkChoice is one choice in a V1ChatChunk.
type V1ChatChunkChoice struct {
	Delta        V1ChatDelta `json:"delta"`
	Index        int         `json:"index"`
	FinishReason *string     `json:"finish_reason"` // null for intermediate chunks, "stop" for final
}

// V1ChatDelta carries the incremental content for a streaming chunk.
// Content is omitted in the final chunk (finish_reason="stop", empty delta).
type V1ChatDelta struct {
	Content string `json:"content,omitempty"`
}

// V1ChatResponse is the body of a non-streaming /v1/chat/completions response.
type V1ChatResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []V1ChatChoice `json:"choices"`
}

// V1ChatChoice is one choice in a V1ChatResponse.
type V1ChatChoice struct {
	Message      ChatMessage `json:"message"`
	Index        int         `json:"index"`
	FinishReason string      `json:"finish_reason"`
}

// ── OpenAI-compatible /v1/completions ────────────────────────────────────────

// V1CompletionsRequest is the body of POST /v1/completions.
type V1CompletionsRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Suffix string `json:"suffix,omitempty"`
	Stream *bool  `json:"stream,omitempty"`
}

// V1CompletionsChunk is one SSE event in a streaming /v1/completions response.
type V1CompletionsChunk struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []V1CompletionsChunkChoice `json:"choices"`
}

// V1CompletionsChunkChoice is one choice in a V1CompletionsChunk.
type V1CompletionsChunkChoice struct {
	Text         string  `json:"text"`
	Index        int     `json:"index"`
	FinishReason *string `json:"finish_reason"`
}

// V1CompletionsResponse is the body of a non-streaming /v1/completions response.
type V1CompletionsResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []V1CompletionsChoice `json:"choices"`
}

// V1CompletionsChoice is one choice in a V1CompletionsResponse.
type V1CompletionsChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}
