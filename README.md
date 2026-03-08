# acp2ollama

Command line tool that wraps an [ACP](https://agentclientprotocol.com/) agent and exposes it as an Ollama-compatible HTTP server. Any tool that speaks the [Ollama API](https://docs.ollama.com/api/) can use it to talk to the agent.

> **Experimental:** This is a proof of concept, not suitable for daily use.
>
> **Important:** Check your ACP agent's terms of service before connecting it to a third-party client.

## Caveats/Architecture

### A semantic mismatch

Ollama clients and ACP agents have opposite assumptions about conversation state:

- **Ollama API is stateless.** The client resends the *full* message history on every request.
- **ACP agents are stateful.** The agent accumulates context across turns and only receives *new* messages.

`acp2ollama` bridges this by maintaining pools of long-lived ACP sessions. Each incoming request is matched to the session that already holds the right conversation context (by fingerprinting the message history), so only the new messages need to be sent.

### On tool use

The two protocols handle tool use in fundamentally opposite ways:

- **Ollama API uses a client-side model.** The client declares available tools in the request, and the LLM asks the client to call them.
- **ACP agents use a server-side model.** The agent ships with its own tools and executes them autonomously.

`acp2ollama` silently ignores incoming tool declarations, and it never asks to execute tools on the client side. ACP agents may execute tools internally and only stream text to the client. Use `--session-mode` to activate a session mode that pre-approves the operations your agent needs. Without a session mode set, all permission requests are silently denied and the agent will be unable to perform any operation that requires approval.

### Resources

`acp2ollama` maintains two session pools:

- **Chat pool** (`/api/chat`, `/v1/chat/completions`): sessions are stateful and reused across turns. The pool maintains two distinct sets:
  - **Fresh sessions** (controlled by `--chat-pool-warmup`): pre-warmed empty sessions held in a ready queue for brand-new conversations. Acquiring one is O(1) — no spawn wait. The queue is replenished immediately when a session is consumed (not only after the request completes), so back-to-back unrelated requests are each served from the queue rather than blocking on a cold spawn.
  - **Active sessions**: sessions that have sent at least one prompt, matched to continuation requests by history fingerprint so only new messages need to be forwarded.

  When the pool is full, idle active sessions are evicted (LRU) before fresh ones are touched. Fresh sessions are exempt from TTL reaping. `503` if all sessions are in use.

- **Generate pool** (`/api/generate`, `/v1/completions`): sessions are single-use — each request gets a fresh session, which is killed after use. `--generate-pool-idle-target` controls how many idle sessions to keep ready: they are spawned at startup and replenished after each use. Requests queue when the pool is full.

## Build

Requires Go 1.24+.

```bash
go build -o acp2ollama .
```

## Usage

```bash
acp2ollama [flags] -- <agent-command> [agent-args...]
```

```bash
# Minimal
acp2ollama -- /usr/local/bin/my-acp-agent

# With model and working directory
acp2ollama --model my-model-id --cwd ~/myproject -- /usr/local/bin/my-acp-agent

# Custom port, session mode, debug logging
acp2ollama --port 8080 --session-mode bypass --log-level debug -- /usr/local/bin/my-acp-agent
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `11434` | TCP port |
| `--host` | `0.0.0.0` | Host/interface to bind |
| `--cwd` | current dir | Working directory for ACP sessions |
| `--model` | `acp-agent` | Model ID for ACP sessions, sent via `session/set_model` on spawn. Clients can override it per-request. Also advertised in `/api/tags` and `/api/ps` — configure your Ollama client to use the same name. Run your agent directly once to discover which models it provides. |
| `--session-mode` | (unset) | ACP session mode to activate on spawn (e.g. `bypass`, `auto`). Session modes let the agent approve certain tool calls without per-call permission prompts. When unset (the default), no mode is activated and all permission requests are silently denied — the safest option. Run your agent directly once to discover which modes it exposes. |
| `--log-level` | `info` | `debug`, `info`, `warn`, `error` |
| `--log-requests` | `false` | Enable verbose logging. When set: request bodies and non-streaming response bodies are logged at debug level (truncated to 2000 bytes); streaming responses log a summary (chunk count and byte count); and the agent subprocess's own stderr output (its thoughts, tool use, etc.) is forwarded to acp2ollama's stderr. Without this flag, agent stderr is suppressed. |
| `--chat-pool-size` | `10` | Max chat sessions |
| `--chat-pool-warmup` | `3` | Number of pre-warmed empty sessions to keep ready for new conversations. Replenished immediately when consumed, not just after each release. |
| `--generate-pool-size` | `3` | Max generate sessions |
| `--generate-pool-idle-target` | `0` | Idle generate sessions to keep ready (spawned at startup and after each use) |
| `--session-ttl` | `0` | Evict idle chat sessions after this duration (e.g. `30m`). `0` = disabled. |

## Supported API endpoints

All unrecognised paths return `501 Not Implemented`.

### Ollama API

**`POST /api/chat`**
- Uses the chat pool (stateful sessions, reused across turns)
- Streaming and non-streaming
- Honoured: `model`, `messages`, `stream`
- `tools` is silently ignored — the ACP agent manages its own tools
- `format` returns 501
- Empty `messages` = load/unload no-op (`keep_alive: 0` or `"0"` = unload; anything else = load)
- `total_duration` is set; token counts are not available via ACP

**`POST /api/generate`**
- Uses the generate pool (single-use sessions)
- Streaming and non-streaming
- Honoured: `model`, `prompt`, `system`, `stream`
- `format` returns 501
- `context` is always `[]`
- Empty `prompt` = load/unload no-op (`keep_alive: 0` or `"0"` = unload; anything else = load)
- `total_duration` is set; token counts are not available via ACP

**`GET /api/tags`**, **`GET /api/ps`**, **`GET /api/version`**
- Return static responses based on `--model`

### OpenAI-compatible API

**`POST /v1/chat/completions`**
- Same behaviour as `/api/chat`, with SSE streaming
- `tools` silently ignored

**`POST /v1/completions`**
- Same behaviour as `/api/generate`, with SSE streaming
- The `prompt` and `suffix` are combined into a fill-in-the-middle format with a code-completion system prompt

> **Note:** Completions via an ACP agent are barely usable in practice — ACP agents are not optimised for fill-in-the-middle tasks. Expect seconds, not milliseconds.

## License

Apache 2.0
