package ollama

import (
	"bytes"
	"io"
	"net/http"
	"time"
)

const maxBodyLog = 2000

// captureWriter wraps an http.ResponseWriter to observe the response status,
// body (for non-streaming), and chunk/byte counts (for streaming).
//
// It detects streaming by checking the Content-Type header at WriteHeader time.
// For streaming responses, Write calls are counted as chunks. For non-streaming,
// the first maxBodyLog bytes of the body are captured for logging.
//
// captureWriter also implements http.Flusher so streaming handlers work normally.
type captureWriter struct {
	http.ResponseWriter
	status     int
	headerSent bool
	streaming  bool
	chunks     int
	totalBytes int
	buf        bytes.Buffer
}

func (cw *captureWriter) WriteHeader(code int) {
	if cw.headerSent {
		return
	}
	cw.headerSent = true
	cw.status = code
	ct := cw.Header().Get("Content-Type")
	cw.streaming = ct == "text/event-stream" || ct == "application/x-ndjson"
	cw.ResponseWriter.WriteHeader(code)
}

func (cw *captureWriter) Write(p []byte) (int, error) {
	if !cw.headerSent {
		cw.WriteHeader(http.StatusOK)
	}
	n, err := cw.ResponseWriter.Write(p)
	cw.totalBytes += n
	if cw.streaming {
		cw.chunks++
	} else if cw.buf.Len() < maxBodyLog {
		cw.buf.Write(p[:min(len(p), maxBodyLog-cw.buf.Len())])
	}
	return n, err
}

// Flush forwards to the underlying ResponseWriter if it implements http.Flusher.
func (cw *captureWriter) Flush() {
	if f, ok := cw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// loggingMiddleware wraps h and:
//   - always logs "Request complete" at debug with duration_ms
//   - when s.logRequests: logs the request body ("Request") before the handler
//     runs, and logs either "Response" (non-streaming) or "Response stream"
//     (streaming) after the handler returns
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		if s.logRequests {
			bodyBytes, _ := io.ReadAll(r.Body)
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			s.logger.Debug("Request",
				"method", r.Method,
				"path", r.URL.Path,
				"body", truncateBody(bodyBytes),
			)
		}

		cw := &captureWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(cw, r)

		durationMs := time.Since(start).Milliseconds()

		if s.logRequests {
			if cw.streaming {
				s.logger.Debug("Response stream",
					"path", r.URL.Path,
					"status", cw.status,
					"chunks", cw.chunks,
					"bytes", cw.totalBytes,
				)
			} else {
				s.logger.Debug("Response",
					"path", r.URL.Path,
					"status", cw.status,
					"body", truncateBody(cw.buf.Bytes()),
				)
			}
		}

		s.logger.Debug("Request complete",
			"path", r.URL.Path,
			"duration_ms", durationMs,
		)
	})
}

// truncateBody returns the body as a string, truncated to maxBodyLog bytes
// with a "...(truncated)" suffix appended if it was cut short.
func truncateBody(b []byte) string {
	if len(b) <= maxBodyLog {
		return string(b)
	}
	return string(b[:maxBodyLog]) + "...(truncated)"
}
