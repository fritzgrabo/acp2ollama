package ollama

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/fritzgrabo/acp2ollama/acp"
)

// Server is the Ollama-compatible HTTP server.
type Server struct {
	chatPool     *acp.ChatPool
	genPool      *acp.GeneratePool
	defaultModel string
	version      string
	startTime    time.Time
	logger       *slog.Logger
	logRequests  bool
	mux          *http.ServeMux
}

// NewServer wires up routes and returns a ready Server.
func NewServer(chatPool *acp.ChatPool, genPool *acp.GeneratePool, defaultModel, version string, logger *slog.Logger, logRequests bool) *Server {
	s := &Server{
		chatPool:     chatPool,
		genPool:      genPool,
		defaultModel: defaultModel,
		version:      version,
		startTime:    time.Now(),
		logger:       logger,
		logRequests:  logRequests,
		mux:          http.NewServeMux(),
	}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	// Ollama API
	s.mux.HandleFunc("POST /api/chat", s.handleChat)
	s.mux.HandleFunc("POST /api/generate", s.handleGenerate)
	s.mux.HandleFunc("GET /api/tags", s.handleTags)
	s.mux.HandleFunc("GET /api/version", s.handleVersion)
	s.mux.HandleFunc("GET /api/ps", s.handlePS)

	// OpenAI-compatible API
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleV1Chat)
	s.mux.HandleFunc("POST /v1/completions", s.handleV1Completions)

	// Catch-all: 501 for anything unrecognised.
	s.mux.HandleFunc("/", s.handleNotImplemented)
}

// ListenAndServe starts the HTTP server on addr and blocks until ctx is cancelled,
// then performs a graceful shutdown.
func ListenAndServe(ctx context.Context, addr string, chatPool *acp.ChatPool, genPool *acp.GeneratePool, defaultModel, version string, logger *slog.Logger, logRequests bool) error {
	srv := NewServer(chatPool, genPool, defaultModel, version, logger, logRequests)
	httpSrv := &http.Server{
		Addr:    addr,
		Handler: srv.loggingMiddleware(srv.mux),
	}

	errCh := make(chan error, 1)
	go func() {
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutCtx)
	}
}
