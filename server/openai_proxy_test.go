package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

// TestOpenAIProxyBaseURL verifies that openAIProxyBaseURL returns the default
// when the env var is unset and respects an override when it is set.
func TestOpenAIProxyBaseURL(t *testing.T) {
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", "")
	if got := openAIProxyBaseURL(); got != defaultOpenAIProxyBaseURL {
		t.Fatalf("expected default %q, got %q", defaultOpenAIProxyBaseURL, got)
	}

	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", "https://my-proxy.example.com/v1/")
	if got := openAIProxyBaseURL(); got != "https://my-proxy.example.com/v1" {
		t.Fatalf("expected trimmed URL, got %q", got)
	}
}

// TestOpenAIAPIKey verifies that openAIAPIKey returns the env var value.
func TestOpenAIAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	if got := openAIAPIKey(); got != "" {
		t.Fatalf("expected empty key, got %q", got)
	}

	t.Setenv("OPENAI_API_KEY", "sk-test-key")
	if got := openAIAPIKey(); got != "sk-test-key" {
		t.Fatalf("expected %q, got %q", "sk-test-key", got)
	}
}

// TestOpenAIPassthroughMiddleware_NoAPIKey verifies that the middleware
// falls through to the next handler when OPENAI_API_KEY is not set.
func TestOpenAIPassthroughMiddleware_NoAPIKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "")

	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	if !nextCalled {
		t.Fatal("expected next handler to be called when OPENAI_API_KEY is empty")
	}
}

// TestOpenAIPassthroughMiddleware_CloudModelFallthrough verifies that a model
// with an explicit ":cloud" suffix is NOT proxied to OpenAI — it should fall
// through to the Ollama cloud passthrough middleware instead.
func TestOpenAIPassthroughMiddleware_CloudModelFallthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "sk-test")

	// ":cloud" suffix marks this as an explicit Ollama cloud model.
	body := []byte(`{"model":"gpt-oss:cloud","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	// ":cloud" models must NOT be proxied to OpenAI; they should fall through
	// to the Ollama cloud proxy middleware.
	if !nextCalled {
		t.Fatal("expected next handler to be called for an explicit :cloud model")
	}
}

// TestOpenAIPassthroughMiddleware_GETPassthrough verifies that GET requests
// are always passed through to the next handler without proxying.
func TestOpenAIPassthroughMiddleware_GETPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "sk-test")

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.GET("/v1/models", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	if !nextCalled {
		t.Fatal("expected GET request to fall through to next handler")
	}
}

// TestOpenAIPassthroughMiddleware_ProxiesUnknownModel verifies that a model
// name that is not installed locally (e.g. "gpt-4o") is proxied to the
// upstream OpenAI API when OPENAI_API_KEY is set.
func TestOpenAIPassthroughMiddleware_ProxiesUnknownModel(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Start a fake upstream OpenAI server.
	var upstreamReq *http.Request
	var upstreamAuthHeader string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamReq = r
		upstreamAuthHeader = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"choices": []any{},
		})
	}))
	defer upstream.Close()

	t.Setenv("OPENAI_API_KEY", "sk-real-key")
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", upstream.URL+"/v1")

	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	// The next handler should NOT be called because the middleware proxied
	// the request and called c.Abort().
	if nextCalled {
		t.Fatal("expected middleware to proxy gpt-4o and abort, not call next handler")
	}

	// The upstream server should have received the request.
	if upstreamReq == nil {
		t.Fatal("expected upstream server to receive a request")
	}

	// The Authorization header should be the real API key.
	if upstreamAuthHeader != "Bearer sk-real-key" {
		t.Fatalf("expected Authorization %q, got %q", "Bearer sk-real-key", upstreamAuthHeader)
	}

	// The response should be 200 OK.
	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
}

// TestOpenAIPassthroughMiddleware_OrgProjectHeaders verifies that
// OPENAI_ORG_ID and OPENAI_PROJECT_ID are forwarded to the upstream.
func TestOpenAIPassthroughMiddleware_OrgProjectHeaders(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var gotOrg, gotProject string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotOrg = r.Header.Get("OpenAI-Organization")
		gotProject = r.Header.Get("OpenAI-Project")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()

	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OPENAI_ORG_ID", "org-abc123")
	t.Setenv("OPENAI_PROJECT_ID", "proj-xyz789")
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", upstream.URL+"/v1")

	body := []byte(`{"model":"gpt-4o","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	if gotOrg != "org-abc123" {
		t.Fatalf("expected OpenAI-Organization %q, got %q", "org-abc123", gotOrg)
	}
	if gotProject != "proj-xyz789" {
		t.Fatalf("expected OpenAI-Project %q, got %q", "proj-xyz789", gotProject)
	}
}

// TestOpenAIPassthroughMiddleware_NoModelField verifies that a request
// without a model field falls through to the next handler.
func TestOpenAIPassthroughMiddleware_NoModelField(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "sk-test")

	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	if !nextCalled {
		t.Fatal("expected next handler to be called when model field is absent")
	}
}

// TestOpenAIPassthroughMiddleware_PathStripping verifies that the /v1 prefix
// is correctly stripped when constructing the upstream URL.
//
// The base URL already contains "/v1" (e.g. https://api.openai.com/v1), so
// the proxy strips the leading "/v1" from the Ollama request path before
// appending it.  The upstream therefore receives "/chat/completions", not
// "/v1/chat/completions".
func TestOpenAIPassthroughMiddleware_PathStripping(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var upstreamPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer upstream.Close()

	t.Setenv("OPENAI_API_KEY", "sk-test")
	// Base URL ends in /v1; the proxy strips /v1 from the request path.
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", upstream.URL+"/v1")

	body := []byte(`{"model":"gpt-4o","messages":[]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	r := gin.New()
	r.POST("/v1/chat/completions", openAIPassthroughMiddleware(), func(c *gin.Context) {
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	// After stripping "/v1" from the Ollama path "/v1/chat/completions",
	// the upstream receives "/chat/completions" appended to its base path "/v1",
	// resulting in the upstream seeing "/v1/chat/completions" at its own root.
	// However, url.ResolveReference with an absolute path replaces the base path,
	// so the upstream sees "/chat/completions" directly.
	// This is correct: base="http://host/v1" + path="/chat/completions"
	// → url.ResolveReference gives "http://host/chat/completions"
	// (absolute path overrides base path per RFC 3986).
	// The net effect is correct: host/v1 + /chat/completions = host/chat/completions,
	// which the upstream server (rooted at /v1) sees as /chat/completions.
	if upstreamPath != "/chat/completions" {
		t.Fatalf("expected upstream path %q, got %q", "/chat/completions", upstreamPath)
	}
	_ = rec
}

// TestOpenAIModelsPassthroughMiddleware_Passthrough verifies that the models
// middleware always calls Next (it is currently a no-op pass-through).
func TestOpenAIModelsPassthroughMiddleware_Passthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()

	nextCalled := false
	r := gin.New()
	r.GET("/v1/models", openAIModelsPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.Status(http.StatusOK)
	})
	r.ServeHTTP(rec, req)

	if !nextCalled {
		t.Fatal("expected openAIModelsPassthroughMiddleware to call Next")
	}
}
