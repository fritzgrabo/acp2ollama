package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/fritzgrabo/acp2ollama/acp"
	"github.com/fritzgrabo/acp2ollama/ollama"
)

const version = "0.1.0"

// config holds all command-line configuration and is passed around the
// application rather than relying on package-level globals.
type config struct {
	host                   string
	port                   int
	cwd                    string
	model                  string
	sessionMode            string
	chatPoolSize           int
	chatPoolFreshTarget    int
	generatePoolSize       int
	generatePoolIdleTarget int
	sessionTTL             time.Duration
	logRequests            bool
	agentCmd               []string
}

func main() {
	// os.Args looks like: acp2ollama [our-flags...] -- <agent> [agent-args...]
	// We split on "--" before calling flag.Parse so the flag package never
	// sees the agent command or its arguments.
	var ourArgs, agentCmd []string
	for i, arg := range os.Args[1:] {
		if arg == "--" {
			ourArgs = os.Args[1 : i+1]
			agentCmd = os.Args[i+2:]
			break
		}
	}
	if agentCmd == nil {
		// No "--" found; treat everything as our flags and report an error below.
		ourArgs = os.Args[1:]
	}

	// --- Flag parsing ---
	fs := flag.NewFlagSet("acp2ollama", flag.ExitOnError)

	port := fs.Int("port", 11434, "TCP port to bind the HTTP server on")
	host := fs.String("host", "0.0.0.0", "Host/interface to bind")
	cwd := fs.String("cwd", "", "Working directory for ACP sessions (defaults to current directory)")
	model := fs.String("model", "acp-agent", "Model ID to use for ACP sessions. Also advertised in /api/tags and /api/ps — configure your Ollama client to use the same name.")
	sessionMode := fs.String("session-mode", "", "ACP session mode to activate on spawn (e.g. \"bypass\"); empty = safest default")
	logLevel := fs.String("log-level", "info", "Log verbosity: debug, info, warn, error")
	chatPoolSize := fs.Int("chat-pool-size", 10, "Maximum concurrent ACP sessions for the chat pool")
	chatPoolWarmup := fs.Int("chat-pool-warmup", 3, "Number of pre-warmed empty sessions to keep ready for new conversations")
	generatePoolSize := fs.Int("generate-pool-size", 3, "Maximum concurrent ACP sessions for the generate pool")
	generatePoolIdleTarget := fs.Int("generate-pool-idle-target", 0, "Number of idle generate sessions to keep ready (spawned at startup and after each use)")
	sessionTTL := fs.Duration("session-ttl", 0, "Evict idle chat sessions after this duration (e.g. 30m); 0 disables")
	logRequests := fs.Bool("log-requests", false, "Log request and response bodies at debug level")

	fs.Parse(ourArgs) //nolint:errcheck // ExitOnError means this never returns an error

	if len(agentCmd) == 0 {
		fmt.Fprintln(os.Stderr, "usage: acp2ollama [flags] -- <agent-command> [agent-args...]")
		os.Exit(1)
	}

	// --- Logger setup ---
	var level slog.Level
	if err := level.UnmarshalText([]byte(*logLevel)); err != nil {
		fmt.Fprintf(os.Stderr, "invalid --log-level %q: %v\n", *logLevel, err)
		os.Exit(1)
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	// --- Working directory ---
	if *cwd == "" {
		var err error
		*cwd, err = os.Getwd()
		if err != nil {
			logger.Error("failed to get current directory", "error", err)
			os.Exit(1)
		}
	}

	cfg := &config{
		host:                   *host,
		port:                   *port,
		cwd:                    *cwd,
		model:                  *model,
		sessionMode:            *sessionMode,
		chatPoolSize:           *chatPoolSize,
		chatPoolFreshTarget:    *chatPoolWarmup,
		generatePoolSize:       *generatePoolSize,
		generatePoolIdleTarget: *generatePoolIdleTarget,
		sessionTTL:             *sessionTTL,
		logRequests:            *logRequests,
		agentCmd:               agentCmd,
	}

	// --- Context and signal handling ---
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigs
		logger.Info("received signal, shutting down", "signal", sig)
		cancel()
	}()

	// --- Session pools ---
	spawnCfg := acp.SpawnConfig{
		AgentArgs:      cfg.agentCmd,
		Cwd:            cfg.cwd,
		SessionMode:    cfg.sessionMode,
		Model:          cfg.model,
		Version:        version,
		Logger:         logger,
		LogAgentOutput: cfg.logRequests,
	}
	chatPool := acp.NewChatPool(ctx, spawnCfg, cfg.chatPoolSize, cfg.chatPoolFreshTarget, cfg.sessionTTL)
	genPool := acp.NewGeneratePool(ctx, spawnCfg, cfg.generatePoolSize, cfg.generatePoolIdleTarget)

	// --- HTTP server ---
	addr := net.JoinHostPort(cfg.host, strconv.Itoa(cfg.port))

	logger.Info("acp2ollama listening",
		"addr", "http://"+addr,
		"version", version,
		"chat_pool_size", cfg.chatPoolSize,
		"generate_pool_size", cfg.generatePoolSize,
	)

	if err := ollama.ListenAndServe(ctx, addr, chatPool, genPool, cfg.model, version, logger, cfg.logRequests); err != nil {
		logger.Error("server error", "error", err)
		os.Exit(1)
	}
}
