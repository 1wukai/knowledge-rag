// Command gateway 是面向客户端的接入网关。
//
// P0 只提供健康检查与到 Python API 的流式反向代理占位；
// 鉴权、限流、SSE 透传细化与 trace 传播在 P3 补齐。
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"
)

type config struct {
	addr       string
	apiUpstream string
}

func loadConfig() config {
	return config{
		addr:        envOr("GATEWAY_ADDR", ":8080"),
		apiUpstream: envOr("API_UPSTREAM", "http://localhost:8000"),
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func newProxy(target string) (*httputil.ReverseProxy, error) {
	upstream, err := url.Parse(target)
	if err != nil {
		return nil, err
	}
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	// 立即 flush，保证 SSE 流式透传不被缓冲。
	proxy.FlushInterval = -1
	return proxy, nil
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	cfg := loadConfig()
	proxy, err := newProxy(cfg.apiUpstream)
	if err != nil {
		logger.Error("invalid api upstream", "error", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health/live", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, `{"status":"ok"}`)
	})
	mux.HandleFunc("/health/ready", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, `{"status":"ok","dependencies_checked":false}`)
	})
	mux.Handle("/", proxy)

	server := &http.Server{
		Addr:              cfg.addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		logger.Info("gateway listening", "addr", cfg.addr, "upstream", cfg.apiUpstream)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("gateway stopped unexpectedly", "error", err)
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
	}
	logger.Info("gateway stopped")
}

func writeJSON(w http.ResponseWriter, status int, body string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(body))
}
