package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/version"
)

func zstdCompress(t *testing.T, data []byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	w, err := zstd.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestStatusHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}
	w := createRequest(t, s.StatusHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if !resp.Cloud.Disabled {
		t.Fatalf("expected cloud.disabled true, got false")
	}
	if resp.Cloud.Source != "env" {
		t.Fatalf("expected cloud.source env, got %q", resp.Cloud.Source)
	}
}

func TestCloudDisabledBlocksRemoteOperations(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := Server{}

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test-cloud",
		RemoteHost: "example.com",
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("chat remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-cloud",
			Messages: []api.Message{{Role: "user", Content: "hi"}},
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"`+internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable)+`"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("generate remote blocked", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-cloud",
			Prompt: "hi",
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"`+internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable)+`"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})

	t.Run("show remote blocked", func(t *testing.T) {
		w := createRequest(t, s.ShowHandler, api.ShowRequest{
			Model: "test-cloud",
		})
		if w.Code != http.StatusForbidden {
			t.Fatalf("expected status 403, got %d", w.Code)
		}
		if got := w.Body.String(); got != `{"error":"`+internalcloud.DisabledError(cloudErrRemoteModelDetailsUnavailable)+`"}` {
			t.Fatalf("unexpected response: %s", got)
		}
	})
}

func TestDeleteHandlerNormalizesExplicitSourceSuffixes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	s := Server{}

	tests := []string{
		"gpt-oss:20b:local",
		"gpt-oss:20b:cloud",
		"qwen3:cloud",
	}

	for _, modelName := range tests {
		t.Run(modelName, func(t *testing.T) {
			w := createRequest(t, s.DeleteHandler, api.DeleteRequest{
				Model: modelName,
			})
			if w.Code != http.StatusNotFound {
				t.Fatalf("expected status 404, got %d (%s)", w.Code, w.Body.String())
			}

			var resp map[string]string
			if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
				t.Fatal(err)
			}
			want := "model '" + modelName + "' not found"
			if resp["error"] != want {
				t.Fatalf("unexpected error: got %q, want %q", resp["error"], want)
			}
		})
	}
}

func TestExplicitCloudPassthroughAPIAndV1(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path   string
		body   string
		header http.Header
	}

	newUpstream := func(t *testing.T, responseBody string) (*httptest.Server, *upstreamCapture) {
		t.Helper()
		capture := &upstreamCapture{}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			payload, _ := io.ReadAll(r.Body)
			capture.path = r.URL.Path
			capture.body = string(payload)
			capture.header = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(responseBody))
		}))

		return srv, capture
	}

	t.Run("api generate", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"ok":"api"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "api-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/generate" {
			t.Fatalf("expected upstream path /api/generate, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if got := capture.header.Get("X-Test-Header"); got != "api-header" {
			t.Fatalf("expected forwarded X-Test-Header=api-header, got %q", got)
		}
		if got := capture.header.Get(cloudProxyClientVersionHeader); got != version.Version {
			t.Fatalf("expected %s=%q, got %q", cloudProxyClientVersionHeader, version.Version, got)
		}
	})

	t.Run("api chat", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"message":{"role":"assistant","content":"ok"},"done":true}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hello"}],"stream":false}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/chat", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/chat" {
			t.Fatalf("expected upstream path /api/chat, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("api embed", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"model":"kimi-k2.5:cloud","embeddings":[[0.1,0.2]]}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","input":"hello"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/embed", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/embed" {
			t.Fatalf("expected upstream path /api/embed, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("api embeddings", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"embedding":[0.1,0.2]}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/embeddings", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/embeddings" {
			t.Fatalf("expected upstream path /api/embeddings, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("api show", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"details":{"format":"gguf"}}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud"}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/show", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/show" {
			t.Fatalf("expected upstream path /api/show, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}
	})

	t.Run("v1 chat completions bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"chatcmpl_test","object":"chat.completion"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"gpt-oss:120b:cloud","messages":[{"role":"user","content":"hi"}],"max_tokens":7}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "v1-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/chat/completions" {
			t.Fatalf("expected upstream path /v1/chat/completions, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"max_tokens":7`) {
			t.Fatalf("expected original OpenAI request body, got %q", capture.body)
		}

		if !strings.Contains(capture.body, `"model":"gpt-oss:120b"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if strings.Contains(capture.body, `"options"`) {
			t.Fatalf("expected no converted Ollama options in upstream body, got %q", capture.body)
		}

		if got := capture.header.Get("X-Test-Header"); got != "v1-header" {
			t.Fatalf("expected forwarded X-Test-Header=v1-header, got %q", got)
		}
	})

	t.Run("v1 chat completions bypasses conversion with legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"chatcmpl_test","object":"chat.completion"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"gpt-oss:120b-cloud","messages":[{"role":"user","content":"hi"}],"max_tokens":7}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Test-Header", "v1-legacy-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/chat/completions" {
			t.Fatalf("expected upstream path /v1/chat/completions, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"max_tokens":7`) {
			t.Fatalf("expected original OpenAI request body, got %q", capture.body)
		}

		if !strings.Contains(capture.body, `"model":"gpt-oss:120b"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if strings.Contains(capture.body, `"options"`) {
			t.Fatalf("expected no converted Ollama options in upstream body, got %q", capture.body)
		}

		if got := capture.header.Get("X-Test-Header"); got != "v1-legacy-header" {
			t.Fatalf("expected forwarded X-Test-Header=v1-legacy-header, got %q", got)
		}
	})

	t.Run("v1 messages bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"msg_1","type":"message"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:cloud","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/messages" {
			t.Fatalf("expected upstream path /v1/messages, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"max_tokens":10`) {
			t.Fatalf("expected original Anthropic request body, got %q", capture.body)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if strings.Contains(capture.body, `"options"`) {
			t.Fatalf("expected no converted Ollama options in upstream body, got %q", capture.body)
		}
	})

	t.Run("v1 messages bypasses conversion with legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"msg_1","type":"message"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"kimi-k2.5:latest-cloud","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/messages" {
			t.Fatalf("expected upstream path /v1/messages, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"max_tokens":10`) {
			t.Fatalf("expected original Anthropic request body, got %q", capture.body)
		}

		if !strings.Contains(capture.body, `"model":"kimi-k2.5:latest"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if strings.Contains(capture.body, `"options"`) {
			t.Fatalf("expected no converted Ollama options in upstream body, got %q", capture.body)
		}
	})

	t.Run("v1 messages web_search fallback uses legacy cloud /api/chat path", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"model":"gpt-oss:120b","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"hello"},"done":true}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
				"model":"gpt-oss:120b-cloud",
				"max_tokens":10,
				"messages":[{"role":"user","content":"search the web"}],
				"tools":[{"type":"web_search_20250305","name":"web_search"}],
				"stream":false
			}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages?beta=true", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/api/chat" {
			t.Fatalf("expected upstream path /api/chat for web_search fallback, got %q", capture.path)
		}

		if !strings.Contains(capture.body, `"model":"gpt-oss:120b"`) {
			t.Fatalf("expected normalized model in upstream body, got %q", capture.body)
		}

		if !strings.Contains(capture.body, `"num_predict":10`) {
			t.Fatalf("expected converted ollama options in upstream body, got %q", capture.body)
		}
	})

	t.Run("v1 model retrieve bypasses conversion", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"kimi-k2.5:cloud","object":"model","created":1,"owned_by":"ollama"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, local.URL+"/v1/models/kimi-k2.5:cloud", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Test-Header", "v1-model-header")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/models/kimi-k2.5" {
			t.Fatalf("expected upstream path /v1/models/kimi-k2.5, got %q", capture.path)
		}

		if capture.body != "" {
			t.Fatalf("expected empty request body, got %q", capture.body)
		}

		if got := capture.header.Get("X-Test-Header"); got != "v1-model-header" {
			t.Fatalf("expected forwarded X-Test-Header=v1-model-header, got %q", got)
		}
	})

	t.Run("v1 model retrieve normalizes legacy cloud suffix", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"kimi-k2.5:latest","object":"model","created":1,"owned_by":"ollama"}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, local.URL+"/v1/models/kimi-k2.5:latest-cloud", nil)
		if err != nil {
			t.Fatal(err)
		}

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		if capture.path != "/v1/models/kimi-k2.5:latest" {
			t.Fatalf("expected upstream path /v1/models/kimi-k2.5:latest, got %q", capture.path)
		}
	})

	t.Run("v1 responses strips encrypted content from json", func(t *testing.T) {
		upstream, capture := newUpstream(t, `{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[{"id":"rs_1","type":"reasoning","encrypted_content":"plain text","summary":[{"type":"summary_text","text":"plain text"}]},{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}`)
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"minimax-m2.5:cloud","input":"hello","stream":false}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		if bytes.Contains(body, []byte(`"encrypted_content"`)) {
			t.Fatalf("expected encrypted_content to be stripped, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"model":"minimax-m2.5:cloud"`)) {
			t.Fatalf("expected model alias to be restored, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"phase":"final_answer"`)) {
			t.Fatalf("expected assistant phase to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"reasoning":{"effort":"none","summary":null}`)) {
			t.Fatalf("expected reasoning metadata to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"verbosity":"medium"`)) {
			t.Fatalf("expected text verbosity to be set, got %s", string(body))
		}
	})

	t.Run("v1 responses routes zstd request bodies through cloud passthrough", func(t *testing.T) {
		type upstreamCapture struct {
			path            string
			contentEncoding string
			body            []byte
		}
		capture := &upstreamCapture{}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			capture.contentEncoding = r.Header.Get("Content-Encoding")
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`))
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := []byte(`{"model":"minimax-m2.5:cloud","input":"hello","stream":false}`)
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewReader(zstdCompress(t, reqBody)))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Content-Encoding", "zstd")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		if capture.contentEncoding != "" {
			t.Fatalf("expected content-encoding to be stripped, got %q", capture.contentEncoding)
		}
		if !bytes.Contains(capture.body, []byte(`"model":"minimax-m2.5"`)) {
			t.Fatalf("expected normalized model in upstream body, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses normalizes custom tool history for cloud upstream", func(t *testing.T) {
		type upstreamCapture struct {
			body []byte
		}
		capture := &upstreamCapture{}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`))
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"custom_tool_call","call_id":"call_patch","name":"apply_patch","input":"*** Begin Patch"},
				{"type":"custom_tool_call_output","call_id":"call_patch","output":"Patch applied"}
			],
			"stream":false
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if bytes.Contains(capture.body, []byte(`"type":"custom_tool_call"`)) {
			t.Fatalf("expected custom_tool_call to be normalized, got %s", string(capture.body))
		}
		if bytes.Contains(capture.body, []byte(`"type":"custom_tool_call_output"`)) {
			t.Fatalf("expected custom_tool_call_output to be normalized, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"type":"function_call"`)) {
			t.Fatalf("expected function_call in upstream body, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected function_call_output in upstream body, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"arguments":"{\"input\":\"*** Begin Patch\"}"`)) {
			t.Fatalf("expected wrapped raw custom tool input, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses expands compaction history for cloud upstream", func(t *testing.T) {
		var capture struct {
			path string
			body []byte
		}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`)
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":"before"},
				{"type":"compaction","message":"","replacement_history":[
					{"type":"message","role":"assistant","content":[{"type":"output_text","text":"summary"}]},
					{"type":"custom_tool_call","call_id":"call_patch","name":"apply_patch","input":"*** Begin Patch"},
					{"type":"custom_tool_call_output","call_id":"call_patch","output":"Patch applied"}
				]},
				{"type":"message","role":"user","content":"after"}
			],
			"stream":false
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		if bytes.Contains(capture.body, []byte(`"type":"compaction"`)) {
			t.Fatalf("expected compaction to be expanded, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"text":"summary"`)) {
			t.Fatalf("expected replacement history message in upstream body, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"type":"function_call"`)) {
			t.Fatalf("expected nested custom tool call to be normalized, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected nested custom tool output to be normalized, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"content":"before"`)) || !bytes.Contains(capture.body, []byte(`"content":"after"`)) {
			t.Fatalf("expected surrounding history to be preserved, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses drops web search call history for cloud upstream", func(t *testing.T) {
		var capture struct {
			path string
			body []byte
		}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`)
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":"before"},
				{"type":"web_search_call","status":"completed","action":{"type":"search","query":"hello"}},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"after"}]}
			],
			"stream":false
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		if bytes.Contains(capture.body, []byte(`"type":"web_search_call"`)) {
			t.Fatalf("expected web_search_call to be dropped, got %s", string(capture.body))
		}
		if !bytes.Contains(capture.body, []byte(`"content":"before"`)) || !bytes.Contains(capture.body, []byte(`"text":"after"`)) {
			t.Fatalf("expected surrounding history to be preserved, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses compacts oversized normalized history before cloud upstream", func(t *testing.T) {
		var capture struct {
			path string
			body []byte
		}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`)
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		var b strings.Builder
		b.WriteString(`{"model":"minimax-m2.5:cloud","input":[`)
		for i := 0; i < 32; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(`{"type":"message","role":"user","content":[{"type":"input_text","text":"`)
			b.WriteString(strings.Repeat("very long message ", 80))
			b.WriteString(`"}]}`)
		}
		b.WriteString(`],"stream":false}`)

		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(b.String()))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		var capturePayload struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.Unmarshal(capture.body, &capturePayload); err != nil {
			t.Fatalf("unmarshal compacted upstream body: %v", err)
		}
		threshold := cloudResponsesInputCompactThreshold("minimax-m2.5")
		if got := estimateResponsesInputTokens(capturePayload.Input); got > threshold {
			t.Fatalf("expected compacted upstream input estimate <= %d tokens, got %d", threshold, got)
		}
		if !bytes.Contains(capture.body, []byte(`Previous conversation history was compacted by Ollama.`)) {
			t.Fatalf("expected synthetic compacted summary, got %s", string(capture.body))
		}
		if bytes.Count(capture.body, []byte(`"type":"function_call"`)) > 1 || bytes.Count(capture.body, []byte(`"type":"function_call_output"`)) > 1 {
			t.Fatalf("expected at most one recent tool exchange to remain, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses keeps low-token json-heavy cloud input uncompact", func(t *testing.T) {
		var capture struct {
			path string
			body []byte
		}
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capture.path = r.URL.Path
			capture.body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","model":"minimax-m2.5","output":[]}`)
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		var b strings.Builder
		b.WriteString(`{"model":"minimax-m2.5:cloud","input":[`)
		for i := 0; i < 150; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `{"type":"function_call","call_id":"call_%03d","name":"noop","arguments":"{}"}`, i)
		}
		b.WriteString(`],"stream":false}`)

		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(b.String()))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if capture.path != "/v1/responses" {
			t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
		}
		if bytes.Contains(capture.body, []byte(`Previous conversation history was compacted by Ollama.`)) {
			t.Fatalf("expected low-token cloud input to pass through without compaction, got %s", string(capture.body))
		}
	})

	t.Run("v1 responses uses model-aware cloud compact threshold", func(t *testing.T) {
		run := func(t *testing.T, model string) []byte {
			t.Helper()

			var capture struct {
				path string
				body []byte
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capture.path = r.URL.Path
				capture.body, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","model":"`+strings.TrimSuffix(model, `:cloud`)+`","output":[]}`)
			}))
			defer upstream.Close()

			original := cloudProxyBaseURL
			cloudProxyBaseURL = upstream.URL
			t.Cleanup(func() { cloudProxyBaseURL = original })

			s := &Server{}
			router, err := s.GenerateRoutes(nil)
			if err != nil {
				t.Fatal(err)
			}
			local := httptest.NewServer(router)
			defer local.Close()

			var b strings.Builder
			b.WriteString(`{"model":"`)
			b.WriteString(model)
			b.WriteString(`","input":[`)
			for i := 0; i < 36; i++ {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(`{"type":"message","role":"user","content":[{"type":"input_text","text":"`)
				b.WriteString(strings.Repeat("mid size ", 50))
				b.WriteString(`"}]}`)
			}
			b.WriteString(`],"stream":false}`)

			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(b.String()))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := local.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
			}
			if capture.path != "/v1/responses" {
				t.Fatalf("expected upstream path /v1/responses, got %q", capture.path)
			}
			return capture.body
		}

		minimaxBody := run(t, "minimax-m2.5:cloud")
		if bytes.Contains(minimaxBody, []byte(`Previous conversation history was compacted by Ollama.`)) {
			t.Fatalf("expected larger-context model to avoid compaction for mid-sized input, got %s", string(minimaxBody))
		}

		gptOSSBody := run(t, "gpt-oss:20b:cloud")
		if !bytes.Contains(gptOSSBody, []byte(`Previous conversation history was compacted by Ollama.`)) {
			t.Fatalf("expected smaller-context model to compact the same input, got %s", string(gptOSSBody))
		}
	})

	t.Run("v1 responses compact returns input-compatible output", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := []byte(`{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"system","content":[{"type":"input_text","text":"system rules"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"keep this request"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"previous assistant answer"}]},
				{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"docs\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"search result payload"}
			]
		}`)
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewReader(zstdCompress(t, reqBody)))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Content-Encoding", "zstd")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		var payload struct {
			Object string `json:"object"`
			Model  string `json:"model"`
			Output []struct {
				Type    string `json:"type"`
				Role    string `json:"role"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"output"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		if payload.Object != "response.compaction" {
			t.Fatalf("expected response.compaction object, got %q", payload.Object)
		}
		if payload.Model != "minimax-m2.5:cloud" {
			t.Fatalf("expected aliased model in response, got %q", payload.Model)
		}
		if len(payload.Output) < 4 {
			t.Fatalf("expected preserved compacted output, got %+v", payload.Output)
		}
		if payload.Output[0].Role != "system" || payload.Output[1].Role != "user" {
			t.Fatalf("expected preserved system and user messages, got %+v", payload.Output)
		}
		if payload.Output[2].Role != "assistant" || payload.Output[2].Content[0].Text != "previous assistant answer" {
			t.Fatalf("expected latest assistant message to be preserved, got %+v", payload.Output)
		}
		last := payload.Output[len(payload.Output)-1]
		if last.Type == "function_call_output" {
			return
		}
		if last.Role != "assistant" || len(last.Content) == 0 {
			t.Fatalf("expected assistant summary message or preserved tool output, got %+v", last)
		}
		if !strings.Contains(last.Content[0].Text, "search result payload") {
			t.Fatalf("expected summary to include tool output, got %q", last.Content[0].Text)
		}
	})

	t.Run("v1 responses model-aware compaction preserves larger recent tail", func(t *testing.T) {
		makeInput := func() json.RawMessage {
			input := []byte(`[
				{"type":"message","role":"user","content":[{"type":"input_text","text":"older 1"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older 2"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"older 3"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older 4"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"older 5"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older 6"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 5"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 6"}]}
			]`)
			return json.RawMessage(input)
		}

		largeOutput, err := compactResponsesInputForModel(makeInput(), "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		smallOutput, err := compactResponsesInputForModel(makeInput(), "gpt-oss:20b")
		if err != nil {
			t.Fatal(err)
		}

		largeJSON, err := json.Marshal(largeOutput)
		if err != nil {
			t.Fatal(err)
		}
		smallJSON, err := json.Marshal(smallOutput)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(largeJSON, []byte(`"text":"recent 2"`)) {
			t.Fatalf("expected larger-context model compaction to preserve a larger recent tail, got %s", string(largeJSON))
		}
		if bytes.Contains(smallJSON, []byte(`"text":"recent 2"`)) {
			t.Fatalf("expected smaller-context model compaction to preserve a smaller recent tail, got %s", string(smallJSON))
		}
	})

	t.Run("v1 responses compaction preserves one older structured tool exchange", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"older docs\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"older result"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older explanation"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		output, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		body, err := json.Marshal(output)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(body, []byte(`"type":"function_call"`)) || !bytes.Contains(body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected one older tool exchange to remain as structured history, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"text":"older explanation"`)) {
			t.Fatalf("expected assistant explanation to remain represented after preserving structured tool exchange, got %s", string(body))
		}
	})

	t.Run("v1 responses compaction preserves older structured tool exchange with assistant for larger models", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"older docs\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"older result"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older explanation"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		largeOutput, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		smallOutput, err := compactResponsesInputForModel(raw, "gpt-oss:20b")
		if err != nil {
			t.Fatal(err)
		}

		largeJSON, err := json.Marshal(largeOutput)
		if err != nil {
			t.Fatal(err)
		}
		smallJSON, err := json.Marshal(smallOutput)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(largeJSON, []byte(`"type":"function_call"`)) || !bytes.Contains(largeJSON, []byte(`"type":"function_call_output"`)) || !bytes.Contains(largeJSON, []byte(`"text":"older explanation"`)) {
			t.Fatalf("expected larger-context model to preserve older structured tool triple, got %s", string(largeJSON))
		}
		if bytes.Contains(smallJSON, []byte(`"type":"function_call"`)) && bytes.Contains(smallJSON, []byte(`"text":"older explanation"`)) {
			t.Fatalf("expected smaller-context model to summarize at least part of the older tool triple, got %s", string(smallJSON))
		}
	})

	t.Run("v1 responses compaction preserves one older structured message pair for larger models", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"older question"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older answer"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		largeOutput, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		smallOutput, err := compactResponsesInputForModel(raw, "gpt-oss:20b")
		if err != nil {
			t.Fatal(err)
		}

		largeJSON, err := json.Marshal(largeOutput)
		if err != nil {
			t.Fatal(err)
		}
		smallJSON, err := json.Marshal(smallOutput)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(largeJSON, []byte(`"text":"older question"`)) || !bytes.Contains(largeJSON, []byte(`"text":"older answer"`)) {
			t.Fatalf("expected larger-context model to preserve one older structured message pair, got %s", string(largeJSON))
		}
		if bytes.Contains(smallJSON, []byte(`"text":"older question"`)) && bytes.Contains(smallJSON, []byte(`"text":"older answer"`)) {
			t.Fatalf("expected smaller-context model to summarize the older message pair, got %s", string(smallJSON))
		}
	})

	t.Run("v1 responses compaction preserves older user tool triple for larger models", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"older question"}]},
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"older docs\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"older result"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older explanation"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		largeOutput, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		smallOutput, err := compactResponsesInputForModel(raw, "gpt-oss:20b")
		if err != nil {
			t.Fatal(err)
		}

		largeJSON, err := json.Marshal(largeOutput)
		if err != nil {
			t.Fatal(err)
		}
		smallJSON, err := json.Marshal(smallOutput)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(largeJSON, []byte(`"text":"older question"`)) || !bytes.Contains(largeJSON, []byte(`"type":"function_call"`)) || !bytes.Contains(largeJSON, []byte(`"text":"older explanation"`)) {
			t.Fatalf("expected larger-context model to preserve older user/tool triple, got %s", string(largeJSON))
		}
		if bytes.Contains(smallJSON, []byte(`"text":"older question"`)) && bytes.Contains(smallJSON, []byte(`"type":"function_call"`)) && bytes.Contains(smallJSON, []byte(`"text":"older explanation"`)) {
			t.Fatalf("expected smaller-context model to summarize at least part of the older user/tool triple, got %s", string(smallJSON))
		}
	})

	t.Run("v1 responses structured candidate selection prefers more recent equal-score chunk", func(t *testing.T) {
		compactedHead := []map[string]any{
			{"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{\"query\":\"older\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "older output"},
			{"type": "function_call", "call_id": "call_2", "name": "search", "arguments": "{\"query\":\"newer\"}"},
			{"type": "function_call_output", "call_id": "call_2", "output": "newer output"},
		}

		candidate, ok := selectStructuredCompactionCandidate(compactedHead, nil, "minimax-m2.5")
		if !ok {
			t.Fatal("expected candidate")
		}
		body, err := json.Marshal(candidate.structured)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`newer output`)) {
			t.Fatalf("expected more recent equal-score tool chunk to win, got %s", string(body))
		}
	})

	t.Run("v1 responses structured candidate scoring prefers richer chunk over bare pair", func(t *testing.T) {
		compactedHead := []map[string]any{
			{"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{\"query\":\"pair\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "pair output"},
			{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "richer question"}}},
			{"type": "function_call", "call_id": "call_2", "name": "search", "arguments": "{\"query\":\"richer\"}"},
			{"type": "function_call_output", "call_id": "call_2", "output": "richer output"},
			{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "richer explanation"}}},
		}

		candidate, ok := selectStructuredCompactionCandidate(compactedHead, nil, "minimax-m2.5")
		if !ok {
			t.Fatal("expected candidate")
		}
		body, err := json.Marshal(candidate.structured)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`richer explanation`)) {
			t.Fatalf("expected richer structured chunk to beat bare tool pair, got %s", string(body))
		}
	})

	t.Run("v1 responses compaction preserves two structured chunks for larger models", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"older question"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older answer"}]},
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"older docs\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"older result"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older explanation"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		largeOutput, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		smallOutput, err := compactResponsesInputForModel(raw, "gpt-oss:20b")
		if err != nil {
			t.Fatal(err)
		}

		largeJSON, err := json.Marshal(largeOutput)
		if err != nil {
			t.Fatal(err)
		}
		smallJSON, err := json.Marshal(smallOutput)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Contains(largeJSON, []byte(`"text":"older answer"`)) || !bytes.Contains(largeJSON, []byte(`"type":"function_call"`)) {
			t.Fatalf("expected larger-context model to preserve two structured chunks, got %s", string(largeJSON))
		}
		if bytes.Contains(smallJSON, []byte(`"text":"older answer"`)) && bytes.Contains(smallJSON, []byte(`"type":"function_call"`)) {
			t.Fatalf("expected smaller-context model to preserve fewer structured chunks, got %s", string(smallJSON))
		}
	})

	t.Run("v1 responses preserved structured chunks stay chronological", func(t *testing.T) {
		raw := json.RawMessage(`[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"older question"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older answer"}]},
			{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"middle docs\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"middle result"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"middle explanation"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 1"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 2"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"recent 3"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent 4"}]}
		]`)

		output, err := compactResponsesInputForModel(raw, "minimax-m2.5")
		if err != nil {
			t.Fatal(err)
		}
		body, err := json.Marshal(output)
		if err != nil {
			t.Fatal(err)
		}

		olderIdx := bytes.Index(body, []byte(`older answer`))
		middleIdx := bytes.Index(body, []byte(`middle result`))
		if olderIdx == -1 || middleIdx == -1 {
			t.Fatalf("expected both structured chunks in output, got %s", string(body))
		}
		if olderIdx > middleIdx {
			t.Fatalf("expected structured chunks to remain chronological, got %s", string(body))
		}
	})

	t.Run("v1 responses compact drops older user messages", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u1"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u2"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u3"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u4"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u5"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u6"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"u7"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"assistant context"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(body, []byte(`"text":"u1"`)) {
			t.Fatalf("expected oldest user message to be compacted, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"text":"u7"`)) {
			t.Fatalf("expected newest user message to remain, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`User: u1 / u2`)) {
			t.Fatalf("expected dropped user messages to be summarized together, got %s", string(body))
		}
	})

	t.Run("v1 responses compact keeps most recent assistant message", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"oldest assistant"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older assistant"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"latest assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"older user"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"latest user"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`"text":"latest assistant"`)) {
			t.Fatalf("expected latest assistant message to remain, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"text":"oldest assistant"`)) {
			t.Fatalf("expected oldest assistant message to be compacted, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"text":"older assistant"`)) {
			t.Fatalf("expected older assistant message to be compacted once tail budget is exceeded, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`Assistant: oldest assistant / older assistant`)) {
			t.Fatalf("expected older assistant messages to be summarized together, got %s", string(body))
		}
	})

	t.Run("v1 responses compact preserves recent tool tail", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older assistant"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"middle assistant"}]},
				{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"recent\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"recent output"},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"latest user"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`"type":"function_call"`)) || !bytes.Contains(body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected recent tool call pair to remain in compacted output, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`Assistant: older assistant`)) {
			t.Fatalf("expected older assistant content to be summarized, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"text":"middle assistant"`)) {
			t.Fatalf("expected more recent assistant message to remain in tail, got %s", string(body))
		}
	})

	t.Run("v1 responses compact combines older tool exchange summary", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"function_call","call_id":"call_old","name":"search","arguments":"{\"query\":\"archived\"}"},
				{"type":"function_call_output","call_id":"call_old","output":"archived output"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`"type":"function_call"`)) || !bytes.Contains(body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected one older tool exchange to remain structured, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`archived output`)) {
			t.Fatalf("expected preserved older tool output, got %s", string(body))
		}
	})

	t.Run("v1 responses compact combines older message run summary", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":[{"type":"input_text","text":"older question"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older answer"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`User: older question | Assistant: older answer`)) {
			t.Fatalf("expected older message run to be summarized as one line, got %s", string(body))
		}
	})

	t.Run("v1 responses compact combines older same-role message run summary", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older assistant one"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"older assistant two"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`Assistant: older assistant one / older assistant two`)) {
			t.Fatalf("expected older same-role assistant run to be summarized as one line, got %s", string(body))
		}
	})

	t.Run("v1 responses compact keeps three short same-role messages together", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"one"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"two"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"three"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`Assistant: one / two / three`)) {
			t.Fatalf("expected three short assistant messages to be summarized together, got %s", string(body))
		}
	})

	t.Run("v1 responses compact deduplicates repeated same-role summary text", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"repeat"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"repeat"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"repeat"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"distinct"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`Assistant: repeat / distinct`)) {
			t.Fatalf("expected repeated assistant text to be deduplicated in summary, got %s", string(body))
		}
	})

	t.Run("v1 responses compact deduplicates repeated alternating summary text", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":[{"type":"input_text","text":"same"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"same"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"different"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"different answer"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`User: same | User: different | Assistant: different answer`)) {
			t.Fatalf("expected repeated alternating text to be deduplicated in summary, got %s", string(body))
		}
	})

	t.Run("v1 responses compact combines older tool exchange with assistant follow-up", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"release note\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"found release note"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The release note confirms the fix shipped."}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`"type":"function_call"`)) || !bytes.Contains(body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected one older tool exchange to remain structured, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`Assistant: The release note confirms the fix shipped.`)) {
			t.Fatalf("expected assistant follow-up to remain represented after structured tool preservation, got %s", string(body))
		}
	})

	t.Run("v1 responses compact combines older user tool assistant run", func(t *testing.T) {
		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{
			"model":"minimax-m2.5:cloud",
			"input":[
				{"type":"message","role":"user","content":[{"type":"input_text","text":"Check whether the release note mentions the fix."}]},
				{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"release note\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"found release note"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The release note confirms the fix shipped."}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"recent user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recent assistant"}]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"newest user"}]},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"newest assistant"}]}
			]
		}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses/compact", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte(`"type":"function_call"`)) || !bytes.Contains(body, []byte(`"type":"function_call_output"`)) {
			t.Fatalf("expected older tool exchange to remain structured, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`User: Check whether the release note mentions the fix. | Assistant: The release note confirms the fix shipped.`)) {
			t.Fatalf("expected older user/assistant context to remain represented in summary, got %s", string(body))
		}
	})

	t.Run("v1 responses strips encrypted content from event stream", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "event: response.output_item.done\n")
			_, _ = io.WriteString(w, "data: {\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},\"type\":\"response.output_item.done\"}\n\n")
			_, _ = io.WriteString(w, "event: response.completed\n")
			_, _ = io.WriteString(w, "data: {\"response\":{\"id\":\"resp_1\",\"model\":\"minimax-m2.5\",\"output\":[{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]},\"type\":\"response.completed\"}\n\n")
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"minimax-m2.5:cloud","input":"hello","stream":true}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
			t.Fatalf("expected event-stream content type, got %q", resp.Header.Get("Content-Type"))
		}
		if bytes.Contains(body, []byte("response.reasoning_summary_text")) {
			t.Fatalf("expected reasoning summary events to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"type":"reasoning"`)) {
			t.Fatalf("expected reasoning output items to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte("event: response.output_item.added\nevent:")) ||
			bytes.Contains(body, []byte("event: response.output_item.done\nevent:")) {
			t.Fatalf("expected no orphaned event lines, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"encrypted_content"`)) {
			t.Fatalf("expected encrypted_content to be stripped, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"model":"minimax-m2.5:cloud"`)) {
			t.Fatalf("expected model alias to be restored, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"phase":"final_answer"`)) {
			t.Fatalf("expected assistant phase to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte("event: response.output_item.done")) ||
			!bytes.Contains(body, []byte(`"type":"response.output_item.done"`)) {
			t.Fatalf("expected assistant output_item.done to be present, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"output_index":1`)) {
			t.Fatalf("expected output_index to be normalized, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"reasoning":{"effort":"none","summary":null}`)) {
			t.Fatalf("expected reasoning metadata to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"verbosity":"medium"`)) {
			t.Fatalf("expected text verbosity to be set, got %s", string(body))
		}
	})

	t.Run("v1 responses detects event stream despite json content type", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "event: response.output_item.done\n")
			_, _ = io.WriteString(w, "data: {\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},\"type\":\"response.output_item.done\"}\n\n")
			_, _ = io.WriteString(w, "event: response.completed\n")
			_, _ = io.WriteString(w, "data: {\"response\":{\"id\":\"resp_1\",\"model\":\"minimax-m2.5\",\"output\":[{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]},\"type\":\"response.completed\"}\n\n")
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"minimax-m2.5:cloud","input":"hello","stream":true}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if !bytes.Contains(body, []byte("event: response.completed")) {
			t.Fatalf("expected completed event in body, got %s", string(body))
		}
		if bytes.Contains(body, []byte("response.reasoning_summary_text")) {
			t.Fatalf("expected reasoning summary events to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"type":"reasoning"`)) {
			t.Fatalf("expected reasoning output items to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte("event: response.output_item.added\nevent:")) ||
			bytes.Contains(body, []byte("event: response.output_item.done\nevent:")) {
			t.Fatalf("expected no orphaned event lines, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"encrypted_content"`)) {
			t.Fatalf("expected encrypted_content to be stripped for JSON-classified responses, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"model":"minimax-m2.5:cloud"`)) {
			t.Fatalf("expected model alias to be restored, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"phase":"final_answer"`)) {
			t.Fatalf("expected assistant phase to be set, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"output_index":1`)) {
			t.Fatalf("expected output_index to be normalized, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"reasoning":{"effort":"none","summary":null}`)) {
			t.Fatalf("expected reasoning metadata to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"verbosity":"medium"`)) {
			t.Fatalf("expected text verbosity to be set, got %s", string(body))
		}
	})

	t.Run("v1 responses passes through non-json data frames", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "event: response.created\n")
			_, _ = io.WriteString(w, "data: ephemeral-control-frame\n\n")
			_, _ = io.WriteString(w, "event: response.output_item.done\n")
			_, _ = io.WriteString(w, "data: {\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},\"type\":\"response.output_item.done\"}\n\n")
			_, _ = io.WriteString(w, "event: response.completed\n")
			_, _ = io.WriteString(w, "data: {\"response\":{\"id\":\"resp_1\",\"model\":\"minimax-m2.5\",\"output\":[{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]},\"type\":\"response.completed\"}\n\n")
		}))
		defer upstream.Close()

		original := cloudProxyBaseURL
		cloudProxyBaseURL = upstream.URL
		t.Cleanup(func() { cloudProxyBaseURL = original })

		s := &Server{}
		router, err := s.GenerateRoutes(nil)
		if err != nil {
			t.Fatal(err)
		}
		local := httptest.NewServer(router)
		defer local.Close()

		reqBody := `{"model":"minimax-m2.5:cloud","input":"hello","stream":true}`
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/responses", bytes.NewBufferString(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
		}
		if !bytes.Contains(body, []byte("data: ephemeral-control-frame")) {
			t.Fatalf("expected non-json data frame to pass through, got %s", string(body))
		}
		if !bytes.Contains(body, []byte("event: response.completed")) {
			t.Fatalf("expected completed event in body, got %s", string(body))
		}
		if bytes.Contains(body, []byte("response.reasoning_summary_text")) {
			t.Fatalf("expected reasoning summary events to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"type":"reasoning"`)) {
			t.Fatalf("expected reasoning output items to be removed, got %s", string(body))
		}
		if bytes.Contains(body, []byte("event: response.output_item.added\nevent:")) ||
			bytes.Contains(body, []byte("event: response.output_item.done\nevent:")) {
			t.Fatalf("expected no orphaned event lines, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"encrypted_content"`)) {
			t.Fatalf("expected encrypted_content to be stripped, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"model":"minimax-m2.5:cloud"`)) {
			t.Fatalf("expected model alias to be restored, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"phase":"final_answer"`)) {
			t.Fatalf("expected assistant phase to be set, got %s", string(body))
		}
		if bytes.Contains(body, []byte(`"output_index":1`)) {
			t.Fatalf("expected output_index to be normalized, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"reasoning":{"effort":"none","summary":null}`)) {
			t.Fatalf("expected reasoning metadata to be set, got %s", string(body))
		}
		if !bytes.Contains(body, []byte(`"verbosity":"medium"`)) {
			t.Fatalf("expected text verbosity to be set, got %s", string(body))
		}
	})
}

func TestCopySanitizedResponsesJSONFallsBackToEventStream(t *testing.T) {
	rec := httptest.NewRecorder()
	body := "event: response.output_item.done\n" +
		"data: {\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},\"type\":\"response.output_item.done\"}\n\n" +
		"event: response.completed\n" +
		"data: {\"response\":{\"id\":\"resp_1\",\"model\":\"minimax-m2.5\",\"output\":[{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"plain text\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"plain text\"}]},{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]},\"type\":\"response.completed\"}\n\n"

	if err := copySanitizedResponsesJSON(rec, strings.NewReader(body), "minimax-m2.5:cloud", "minimax-m2.5"); err != nil {
		t.Fatal(err)
	}

	got := rec.Body.Bytes()
	if !bytes.Contains(got, []byte("event: response.completed")) {
		t.Fatalf("expected completed event in body, got %s", string(got))
	}
	if bytes.Contains(got, []byte("response.reasoning_summary_text")) {
		t.Fatalf("expected reasoning summary events to be removed, got %s", string(got))
	}
	if bytes.Contains(got, []byte(`"type":"reasoning"`)) {
		t.Fatalf("expected reasoning output items to be removed, got %s", string(got))
	}
	if bytes.Contains(got, []byte("event: response.output_item.added\nevent:")) ||
		bytes.Contains(got, []byte("event: response.output_item.done\nevent:")) {
		t.Fatalf("expected no orphaned event lines, got %s", string(got))
	}
	if bytes.Contains(got, []byte(`"encrypted_content"`)) {
		t.Fatalf("expected encrypted_content to be stripped, got %s", string(got))
	}
	if !bytes.Contains(got, []byte(`"model":"minimax-m2.5:cloud"`)) {
		t.Fatalf("expected model alias to be restored, got %s", string(got))
	}
	if !bytes.Contains(got, []byte(`"phase":"final_answer"`)) {
		t.Fatalf("expected assistant phase to be set, got %s", string(got))
	}
	if !bytes.Contains(got, []byte("event: response.output_item.done")) ||
		!bytes.Contains(got, []byte(`"type":"response.output_item.done"`)) {
		t.Fatalf("expected assistant output_item.done to be present, got %s", string(got))
	}
	if bytes.Contains(got, []byte(`"output_index":1`)) {
		t.Fatalf("expected output_index to be normalized, got %s", string(got))
	}
	if !bytes.Contains(got, []byte(`"reasoning":{"effort":"none","summary":null}`)) {
		t.Fatalf("expected reasoning metadata to be set, got %s", string(got))
	}
	if !bytes.Contains(got, []byte(`"verbosity":"medium"`)) {
		t.Fatalf("expected text verbosity to be set, got %s", string(got))
	}
}

func TestCloudDisabledBlocksExplicitCloudPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/chat/completions", bytes.NewBufferString(`{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]string
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}

	if got["error"] != internalcloud.DisabledError(cloudErrRemoteInferenceUnavailable) {
		t.Fatalf("unexpected error message: %q", got["error"])
	}
}

func TestCloudPassthroughStreamsPromptly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("upstream writer is not a flusher")
		}

		_, _ = w.Write([]byte(`{"response":"first"}` + "\n"))
		flusher.Flush()

		time.Sleep(700 * time.Millisecond)

		_, _ = w.Write([]byte(`{"response":"second"}` + "\n"))
		flusher.Flush()
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/chat", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
	}

	reader := bufio.NewReader(resp.Body)

	start := time.Now()
	firstLine, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("failed reading first streamed line: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 400*time.Millisecond {
		t.Fatalf("first streamed line arrived too late (%s), likely not flushing", elapsed)
	}
	if !strings.Contains(firstLine, `"first"`) {
		t.Fatalf("expected first line to contain first chunk, got %q", firstLine)
	}

	secondLine, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("failed reading second streamed line: %v", err)
	}
	if !strings.Contains(secondLine, `"second"`) {
		t.Fatalf("expected second line to contain second chunk, got %q", secondLine)
	}
}

func TestCloudPassthroughSkipsAnthropicWebSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path string
	}
	capture := &upstreamCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message"}`))
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	router := gin.New()
	router.POST(
		"/v1/messages",
		cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable),
		middleware.AnthropicMessagesMiddleware(),
		func(c *gin.Context) { c.Status(http.StatusTeapot) },
	)

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{
		"model":"kimi-k2.5:cloud",
		"max_tokens":10,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTeapot {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected local middleware path status %d, got %d (%s)", http.StatusTeapot, resp.StatusCode, string(body))
	}

	if capture.path != "" {
		t.Fatalf("expected no passthrough for web_search requests, got upstream path %q", capture.path)
	}
}

func TestCloudPassthroughSkipsAnthropicWebSearchLegacySuffix(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	type upstreamCapture struct {
		path string
	}
	capture := &upstreamCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capture.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message"}`))
	}))
	defer upstream.Close()

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	router := gin.New()
	router.POST(
		"/v1/messages",
		cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable),
		middleware.AnthropicMessagesMiddleware(),
		func(c *gin.Context) { c.Status(http.StatusTeapot) },
	)

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{
		"model":"kimi-k2.5:latest-cloud",
		"max_tokens":10,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/v1/messages", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusTeapot {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected local middleware path status %d, got %d (%s)", http.StatusTeapot, resp.StatusCode, string(body))
	}

	if capture.path != "" {
		t.Fatalf("expected no passthrough for web_search requests, got upstream path %q", capture.path)
	}
}

func TestCloudPassthroughSigningFailureReturnsUnauthorized(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "https://ollama.com/signin/example", nil
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}

	if got["error"] != "unauthorized" {
		t.Fatalf("unexpected error message: %v", got["error"])
	}

	if got["signin_url"] != "https://ollama.com/signin/example" {
		t.Fatalf("unexpected signin_url: %v", got["signin_url"])
	}
}

func TestCloudPassthroughSigningFailureWithoutSigninURL(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "", errors.New("key missing")
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	reqBody := `{"model":"kimi-k2.5:cloud","prompt":"hello","stream":false}`
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/generate", bytes.NewBufferString(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}

	if got["error"] != "unauthorized" {
		t.Fatalf("unexpected error message: %v", got["error"])
	}

	if _, ok := got["signin_url"]; ok {
		t.Fatalf("did not expect signin_url when helper fails, got %v", got["signin_url"])
	}
}
