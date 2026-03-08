// Package acp wraps the ACP Go SDK, managing agent subprocess lifecycles and
// translating ACP session/update notifications into streamable text chunks.
package acp

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	acp "github.com/coder/acp-go-sdk"
)

// proxyClient implements acp.Client. It receives SDK callbacks and routes
// incoming text chunks to the channel registered by the active Prompt call.
type proxyClient struct {
	logVerbose bool // log chunk/thought/tool events at debug level
	logger     *slog.Logger

	mu      sync.Mutex
	chunkCh chan string
	ctx     context.Context
}

// setChunkCh replaces the active chunk channel. Called before and after each
// Prompt call to attach and detach the current request's channel.
func (c *proxyClient) setChunkCh(ctx context.Context, ch chan string) {
	c.mu.Lock()
	c.chunkCh = ch
	c.ctx = ctx
	c.mu.Unlock()
}

// send delivers a text chunk to the active channel. It blocks until the channel
// accepts the chunk or the request context is cancelled. This applies backpressure
// to the agent process if the HTTP client is slow.
func (c *proxyClient) send(text string) {
	c.mu.Lock()
	// Hold the lock until we select, to prevent the channel from being closed
	// (in setChunkCh) while we are trying to send.
	defer c.mu.Unlock()

	ch := c.chunkCh
	ctx := c.ctx
	if ch == nil {
		return
	}

	select {
	case ch <- text:
	case <-ctx.Done():
		// Request cancelled; drop the chunk.
	}
}

// SessionUpdate is called by interceptSessionUpdates for each session/update
// notification. Because it is called sequentially from a single goroutine,
// chunks always arrive in order — no additional locking is needed here.
func (c *proxyClient) SessionUpdate(_ context.Context, params acp.SessionNotification) error {
	u := params.Update
	switch {
	case u.AgentMessageChunk != nil:
		if u.AgentMessageChunk.Content.Text != nil {
			text := u.AgentMessageChunk.Content.Text.Text
			if c.logVerbose {
				c.logger.Debug("ACP agent chunk", "text", text)
			}
			c.send(text)
		}
	case u.AgentThoughtChunk != nil:
		if c.logVerbose && u.AgentThoughtChunk.Content.Text != nil {
			c.logger.Debug("ACP agent thought", "text", u.AgentThoughtChunk.Content.Text.Text)
		}
	case u.ToolCall != nil:
		if c.logVerbose {
			c.logger.Debug("ACP tool call", "title", u.ToolCall.Title, "status", u.ToolCall.Status)
		}
	case u.ToolCallUpdate != nil:
		if c.logVerbose {
			c.logger.Debug("ACP tool call update", "tool_call_id", u.ToolCallUpdate.ToolCallId)
		}
	}
	return nil
}

// RequestPermission handles agent permission prompts. We always cancel — the
// agent should be configured via --session-mode to allow the operations it
// needs without requiring per-call prompts.
func (c *proxyClient) RequestPermission(_ context.Context, _ acp.RequestPermissionRequest) (acp.RequestPermissionResponse, error) {
	return acp.RequestPermissionResponse{
		Outcome: acp.RequestPermissionOutcome{
			Cancelled: &acp.RequestPermissionOutcomeCancelled{},
		},
	}, nil
}

// Filesystem and terminal capabilities are not supported. The fs methods
// return errors; the terminal methods return empty success responses to avoid
// panics in the SDK if it ever calls them.

func (c *proxyClient) ReadTextFile(_ context.Context, _ acp.ReadTextFileRequest) (acp.ReadTextFileResponse, error) {
	return acp.ReadTextFileResponse{}, fmt.Errorf("acp2ollama: fs capability not supported")
}

func (c *proxyClient) WriteTextFile(_ context.Context, _ acp.WriteTextFileRequest) (acp.WriteTextFileResponse, error) {
	return acp.WriteTextFileResponse{}, fmt.Errorf("acp2ollama: fs capability not supported")
}

func (c *proxyClient) CreateTerminal(_ context.Context, _ acp.CreateTerminalRequest) (acp.CreateTerminalResponse, error) {
	return acp.CreateTerminalResponse{}, nil
}

func (c *proxyClient) KillTerminalCommand(_ context.Context, _ acp.KillTerminalCommandRequest) (acp.KillTerminalCommandResponse, error) {
	return acp.KillTerminalCommandResponse{}, nil
}

func (c *proxyClient) TerminalOutput(_ context.Context, _ acp.TerminalOutputRequest) (acp.TerminalOutputResponse, error) {
	return acp.TerminalOutputResponse{}, nil
}

func (c *proxyClient) ReleaseTerminal(_ context.Context, _ acp.ReleaseTerminalRequest) (acp.ReleaseTerminalResponse, error) {
	return acp.ReleaseTerminalResponse{}, nil
}

func (c *proxyClient) WaitForTerminalExit(_ context.Context, _ acp.WaitForTerminalExitRequest) (acp.WaitForTerminalExitResponse, error) {
	return acp.WaitForTerminalExitResponse{}, nil
}
