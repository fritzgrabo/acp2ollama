package acp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	acp "github.com/coder/acp-go-sdk"
)

// Message is an Ollama-style chat message. Role is "system", "user", or "assistant".
type Message struct {
	Role    string
	Content string
}

// Session manages a single long-lived ACP agent subprocess. It serializes
// prompts so only one is in flight at a time.
type Session struct {
	conn      *acp.ClientSideConnection
	sessionId acp.SessionId
	client    *proxyClient
	cmd       *exec.Cmd
	logger    *slog.Logger

	mu     sync.Mutex
	cursor int    // number of messages already acknowledged by ACP
	busy   bool   // true while a Prompt call is in flight
	model  string // current model name (normalized, without :latest suffix)
}

// Spawn starts an agent subprocess, runs the ACP handshake, and returns a
// ready Session. cfg.AgentArgs[0] is the executable; the rest are its arguments.
// modelFlag is sent to the agent via session/set_model; it may differ from
// cfg.Model when a per-request model override is in effect.
func Spawn(ctx context.Context, cfg SpawnConfig, modelFlag string) (*Session, error) {
	logger := cfg.Logger
	logger.Debug("Spawning agent process", "command", cfg.AgentArgs)

	t0 := time.Now()

	// os/exec.Command builds an *exec.Cmd. CommandContext ties the process
	// lifetime to the context: if ctx is cancelled, the process is killed.
	cmd := exec.CommandContext(ctx, cfg.AgentArgs[0], cfg.AgentArgs[1:]...)
	// Forward agent stderr to our stderr only when --log-requests is on;
	// otherwise discard it to avoid noisy agent output in normal use.
	if cfg.LogAgentOutput {
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stderr = io.Discard
	}

	// StdinPipe and StdoutPipe return io.WriteCloser / io.ReadCloser connected
	// to the child process. We must call them before cmd.Start().
	agentStdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stdin pipe: %w", err)
	}
	agentStdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stdout pipe: %w", err)
	}

	tExec := time.Now()
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting agent process: %w", err)
	}
	logger.Debug("spawn: process forked", "fork_ms", time.Since(tExec).Round(time.Millisecond))

	client := &proxyClient{logVerbose: cfg.LogAgentOutput, logger: logger}

	// --- The ordering trick (see agents.md) ---
	//
	// The SDK dispatches every incoming JSON-RPC message with "go handleInbound()",
	// which spawns a new goroutine per message. For session/update notifications
	// this causes non-deterministic chunk ordering.
	// Tracked upstream: https://github.com/coder/acp-go-sdk/issues/20
	//
	// Fix: insert an io.Pipe between agentStdout and the SDK. The goroutine
	// below reads from agentStdout, handles session/update directly and
	// sequentially, and forwards everything else to the SDK via pw.
	pr, pw := io.Pipe()
	go interceptSessionUpdates(ctx, agentStdout, pw, client, logger)

	// The SDK reads from pr (not agentStdout directly) and writes to agentStdin.
	conn := acp.NewClientSideConnection(client, agentStdin, pr)

	tInit := time.Now()
	initResp, err := conn.Initialize(ctx, acp.InitializeRequest{
		ProtocolVersion:    acp.ProtocolVersionNumber,
		ClientCapabilities: acp.ClientCapabilities{},
		ClientInfo:         &acp.Implementation{Name: "acp2ollama", Version: cfg.Version},
	})
	if err != nil {
		cmd.Process.Kill() //nolint:errcheck
		return nil, fmt.Errorf("ACP initialize: %w", err)
	}
	logger.Debug("spawn: initialize complete",
		"protocol_version", initResp.ProtocolVersion,
		"initialize_ms", time.Since(tInit).Round(time.Millisecond),
		"cumulative_ms", time.Since(t0).Round(time.Millisecond),
	)

	tNewSession := time.Now()
	sessResp, err := conn.NewSession(ctx, acp.NewSessionRequest{
		Cwd:        cfg.Cwd,
		McpServers: []acp.McpServer{},
	})
	if err != nil {
		cmd.Process.Kill() //nolint:errcheck
		return nil, fmt.Errorf("ACP new session: %w", err)
	}
	// Log available modes and models if the agent advertises them.
	logArgs := []any{
		"session_id", sessResp.SessionId,
		"new_session_ms", time.Since(tNewSession).Round(time.Millisecond),
		"cumulative_ms", time.Since(t0).Round(time.Millisecond),
	}
	if m := sessResp.Modes; m != nil {
		ids := make([]string, len(m.AvailableModes))
		for i, mode := range m.AvailableModes {
			ids[i] = string(mode.Id)
		}
		logArgs = append(logArgs, "available_modes", ids, "current_mode", string(m.CurrentModeId))
	}
	if m := sessResp.Models; m != nil {
		ids := make([]string, len(m.AvailableModels))
		for i, model := range m.AvailableModels {
			ids[i] = string(model.ModelId)
		}
		logArgs = append(logArgs, "available_models", ids, "current_model", string(m.CurrentModelId))
	}
	logger.Debug("spawn: session/new complete", logArgs...)

	// If a session mode was requested, activate it now.
	if cfg.SessionMode != "" {
		tMode := time.Now()
		if _, err := conn.SetSessionMode(ctx, acp.SetSessionModeRequest{
			SessionId: sessResp.SessionId,
			ModeId:    acp.SessionModeId(cfg.SessionMode),
		}); err != nil {
			logger.Warn("failed to set session mode", "mode", cfg.SessionMode, "error", err)
		} else {
			logger.Debug("spawn: session/set_mode complete",
				"mode", cfg.SessionMode,
				"set_mode_ms", time.Since(tMode).Round(time.Millisecond),
				"cumulative_ms", time.Since(t0).Round(time.Millisecond),
			)
		}
	}

	// Always call session/set_model so the agent knows which model to use.
	tModel := time.Now()
	if _, err := conn.SetSessionModel(ctx, acp.SetSessionModelRequest{
		SessionId: sessResp.SessionId,
		ModelId:   acp.ModelId(modelFlag),
	}); err != nil {
		logger.Warn("failed to set session model", "model", modelFlag, "error", err)
	} else {
		logger.Debug("spawn: session/set_model complete",
			"model", modelFlag,
			"set_model_ms", time.Since(tModel).Round(time.Millisecond),
			"cumulative_ms", time.Since(t0).Round(time.Millisecond),
		)
	}

	logger.Debug("spawn: ready",
		"total_ms", time.Since(t0).Round(time.Millisecond),
	)

	return &Session{
		conn:      conn,
		sessionId: sessResp.SessionId,
		client:    client,
		cmd:       cmd,
		logger:    logger,
		model:     normalizeModel(modelFlag),
	}, nil
}

// Done returns a channel that is closed when the agent subprocess exits.
// Callers can select on this to detect unexpected agent death.
func (s *Session) Done() <-chan struct{} { return s.conn.Done() }

// Close terminates the agent subprocess and waits for it to exit, releasing
// OS resources. Safe to call multiple times (Kill is idempotent).
func (s *Session) Close() {
	s.cmd.Process.Kill() //nolint:errcheck
	s.cmd.Wait()         //nolint:errcheck
}

// Prompt formats the new messages (those after the session's cursor position)
// as plain text and sends them to ACP. It returns a channel that yields text
// chunks as they arrive and is closed when the agent turn completes.
//
// Only messages beyond the cursor are sent — the ACP session already has the
// prior context in its own history.
func (s *Session) Prompt(ctx context.Context, messages []Message) (<-chan string, error) {
	s.mu.Lock()
	if s.busy {
		s.mu.Unlock()
		return nil, fmt.Errorf("session is busy")
	}
	if len(messages) <= s.cursor {
		s.mu.Unlock()
		return nil, fmt.Errorf("message history shorter than cursor (pool routing error)")
	}

	newMessages := messages[s.cursor:]
	totalLen := len(messages)
	s.busy = true
	s.mu.Unlock()

	s.logger.Debug("ACP prompt", "new_messages", len(newMessages), "cursor_before", s.cursor)
	return s.startPrompt(ctx, buildPromptText(newMessages), func(success bool) {
		s.mu.Lock()
		if success {
			s.cursor = totalLen
		}
		s.busy = false
		s.mu.Unlock()
	})
}

// PromptDelta sends newMessages to ACP without cursor-diff logic. Used by
// /api/generate where the caller always provides the complete set of messages
// to send (no reuse across turns).
func (s *Session) PromptDelta(ctx context.Context, newMessages []Message) (<-chan string, error) {
	s.mu.Lock()
	if s.busy {
		s.mu.Unlock()
		return nil, fmt.Errorf("session is busy")
	}
	if len(newMessages) == 0 {
		s.mu.Unlock()
		return nil, fmt.Errorf("no messages to send")
	}
	s.busy = true
	s.mu.Unlock()

	s.logger.Debug("ACP prompt (delta)", "messages", len(newMessages))
	return s.startPrompt(ctx, buildPromptText(newMessages), func(bool) {
		s.mu.Lock()
		s.busy = false
		s.mu.Unlock()
	})
}

// startPrompt is the shared implementation of Prompt and PromptDelta.
// It wires up the chunk channel, fires off conn.Prompt in a goroutine, and
// calls onDone when the prompt completes (used to reset cursor and busy flag).
func (s *Session) startPrompt(ctx context.Context, promptText string, onDone func(bool)) (<-chan string, error) {
	// Buffer of 64 so SessionUpdate can keep writing without blocking while
	// the HTTP handler is draining the channel into the response.
	chunkCh := make(chan string, 64)
	s.client.setChunkCh(ctx, chunkCh)

	go func() {
		// Always clear the chunk channel and call onDone when we exit,
		// even if there's an error. defer runs in LIFO order so the close
		// happens before onDone — callers can rely on the channel being
		// closed before busy becomes false.
		var err error
		defer func() {
			s.client.setChunkCh(nil, nil)
			close(chunkCh)
			onDone(err == nil)
		}()

		_, err = s.conn.Prompt(ctx, acp.PromptRequest{
			SessionId: s.sessionId,
			Prompt:    []acp.ContentBlock{acp.TextBlock(promptText)},
		})
		if err != nil {
			// Send the error as a sentinel chunk so the HTTP handler can
			// detect it while draining the channel.
			select {
			case chunkCh <- encodeErrorChunk(err.Error()):
			default:
			}
		}
	}()

	return chunkCh, nil
}

// encodeErrorChunk wraps an error message in a sentinel string that
// IsErrorChunk can distinguish from real text content.
func encodeErrorChunk(msg string) string { return errorPrefix + msg }

const errorPrefix = "\x00ERR:"

// IsErrorChunk reports whether chunk is an error sentinel from the SDK goroutine.
// If so, the error message is returned as the second value.
func IsErrorChunk(chunk string) (string, bool) {
	if strings.HasPrefix(chunk, errorPrefix) {
		return chunk[len(errorPrefix):], true
	}
	return "", false
}

// interceptSessionUpdates reads agentOut line-by-line, handles session/update
// notifications directly (preserving sequential order), and forwards all other
// messages to pw for the SDK to process.
//
// This goroutine is the only reader of agentOut and the only writer to pw,
// which ensures that the SDK sees messages in arrival order and that
// session/update chunks are delivered to the client sequentially.
func interceptSessionUpdates(ctx context.Context, agentOut io.Reader, pw *io.PipeWriter, client *proxyClient, logger *slog.Logger) {
	// The agent may emit large lines (e.g. file contents). We allow up to 10 MB.
	const (
		initialBuf = 1024 * 1024
		maxBuf     = 10 * 1024 * 1024
	)
	scanner := bufio.NewScanner(agentOut)
	buf := make([]byte, 0, initialBuf)
	scanner.Buffer(buf, maxBuf)

	forward := func(line []byte) {
		pw.Write(append(bytes.Clone(line), '\n')) //nolint:errcheck
	}

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}

		// Parse just enough of the JSON-RPC envelope to know the method and
		// whether this is a notification (no id field, or id: null).
		var envelope struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(line, &envelope); err != nil {
			// Unparseable: forward as-is and let the SDK handle it.
			forward(line)
			continue
		}

		isNotification := len(envelope.ID) == 0 || string(envelope.ID) == "null"
		if isNotification && envelope.Method == acp.ClientMethodSessionUpdate {
			var params acp.SessionNotification
			if err := json.Unmarshal(envelope.Params, &params); err != nil {
				logger.Error("failed to parse session/update params", "error", err)
				continue
			}
			if err := client.SessionUpdate(ctx, params); err != nil {
				logger.Error("session/update handler error", "error", err)
			}
			continue
		}

		// Forward responses, other requests, and non-session/update notifications.
		forward(line)
	}

	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			logger.Error("agent output line too long", "max_buf", maxBuf)
		}
		pw.CloseWithError(err) //nolint:errcheck
	} else {
		pw.Close() //nolint:errcheck
	}
}

// SwitchModelIfNeeded calls session/set_model when the requested model differs
// from the session's current model. On failure it logs a warning and continues;
// the agent keeps its previous model. The model name is normalised (":latest"
// stripped) before comparison and before being sent to the agent.
func (s *Session) SwitchModelIfNeeded(ctx context.Context, requestedModel string) {
	normalized := normalizeModel(requestedModel)

	s.mu.Lock()
	current := s.model
	s.mu.Unlock()

	if normalized == current {
		return
	}

	if _, err := s.conn.SetSessionModel(ctx, acp.SetSessionModelRequest{
		SessionId: s.sessionId,
		ModelId:   acp.ModelId(normalized),
	}); err != nil {
		s.logger.Warn("Failed to switch session model", "from", current, "to", normalized, "error", err)
		return
	}

	s.mu.Lock()
	s.model = normalized
	s.mu.Unlock()
	s.logger.Debug("Session model switched", "from", current, "to", normalized)
}

// normalizeModel strips a trailing ":latest" suffix so that "foo" and
// "foo:latest" are treated as the same model identifier.
func normalizeModel(m string) string {
	return strings.TrimSuffix(m, ":latest")
}

// buildPromptText formats a slice of messages as the plain-text ACP prompt
// format described in the spec.
func buildPromptText(messages []Message) string {
	var sb strings.Builder
	for i, m := range messages {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		fmt.Fprintf(&sb, "[%s]\n%s", m.Role, m.Content)
	}
	return sb.String()
}
