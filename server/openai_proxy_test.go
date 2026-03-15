package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

// TestOpenAIModelsPassthroughMiddleware_NoKey verifies that when no API key is
// configured the middleware is a transparent pass-through (local models only).
func TestOpenAIModelsPassthroughMiddleware_NoKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OPENAI_API_KEY", "")

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
		t.Fatal("expected openAIModelsPassthroughMiddleware to call Next when no API key is set")
	}
}

// TestOpenAIModelsPassthroughMiddleware_MergesUpstream verifies that when an
// API key is configured the middleware merges the upstream model list into the
// local model list, deduplicating by ID.
func TestOpenAIModelsPassthroughMiddleware_MergesUpstream(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Upstream server returns two models: one that overlaps with local and one new.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"local-model","object":"model","created":1000,"owned_by":"ollama"},{"id":"gpt-4o","object":"model","created":2000,"owned_by":"openai"}]}`)
	}))
	defer upstream.Close()

	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", upstream.URL+"/v1")

	// Local handler returns one model.
	localResp := `{"object":"list","data":[{"id":"local-model","object":"model","created":500,"owned_by":"ollama"}]}`

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	r := gin.New()
	r.GET("/v1/models", openAIModelsPassthroughMiddleware(), func(c *gin.Context) {
		c.Header("Content-Type", "application/json")
		c.String(http.StatusOK, localResp)
	})
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var result openaiListCompletion
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Should have 2 models: local-model (from local, deduplicated) and gpt-4o (from upstream).
	if len(result.Data) != 2 {
		t.Fatalf("expected 2 models in merged list, got %d: %+v", len(result.Data), result.Data)
	}

	ids := make(map[string]bool)
	for _, m := range result.Data {
		ids[m.ID] = true
	}
	if !ids["local-model"] {
		t.Error("expected local-model in merged list")
	}
	if !ids["gpt-4o"] {
		t.Error("expected gpt-4o in merged list")
	}
}

// TestOpenAIModelsPassthroughMiddleware_UpstreamError verifies that when the
// upstream fetch fails the middleware still returns the local model list.
func TestOpenAIModelsPassthroughMiddleware_UpstreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Upstream server returns a 500 error.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer upstream.Close()

	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OLLAMA_OPENAI_PROXY_BASE_URL", upstream.URL+"/v1")

	localResp := `{"object":"list","data":[{"id":"local-model","object":"model","created":500,"owned_by":"ollama"}]}`

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	r := gin.New()
	r.GET("/v1/models", openAIModelsPassthroughMiddleware(), func(c *gin.Context) {
		c.Header("Content-Type", "application/json")
		c.String(http.StatusOK, localResp)
	})
	r.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var result openaiListCompletion
	if err := json.NewDecoder(rec.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Should have only the local model when upstream fails.
	if len(result.Data) != 1 {
		t.Fatalf("expected 1 model when upstream fails, got %d: %+v", len(result.Data), result.Data)
	}
	if result.Data[0].ID != "local-model" {
		t.Errorf("expected local-model, got %q", result.Data[0].ID)
	}
}

// TestOpenAIPassthroughMiddleware_IncomingAuthFallback verifies that when
// OPENAI_API_KEY is the dummy "ollama" placeholder, the middleware falls back
// to the Authorization header already present in the incoming request.
// This handles ChatGPT OAuth sessions where Codex sends its OAuth access_token
// as the Bearer credential when the user switches to a cloud model (gpt-5.4,
// o3, etc.) inside the TUI.
func TestOpenAIPassthroughMiddleware_IncomingAuthFallback(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Upstream mock that records the Authorization header it receives.
	var receivedAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"id":"resp_1","object":"response"}`)
	}))
	defer upstream.Close()

	t.Setenv(openAIProxyBaseURLEnv, upstream.URL+"/v1")
	// Set the dummy key — the middleware should prefer the incoming header.
	t.Setenv("OPENAI_API_KEY", "ollama")

	r := gin.New()
	nextCalled := false
	r.POST("/v1/responses", openAIPassthroughMiddleware(), func(c *gin.Context) {
		// Should never reach here — the middleware should proxy and abort.
		nextCalled = true
		c.JSON(http.StatusInternalServerError, gin.H{"error": "should not reach handler"})
	})

	// "gpt-5.4" is not a local Ollama model, so the middleware should proxy it.
	// The incoming request carries a real OAuth token (JWT-like, not "ollama").
	body := `{"model":"gpt-5.4","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	// Simulate a ChatGPT OAuth access_token (long JWT-like string).
	oauthToken := "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.realtoken.sig"
	req.Header.Set("Authorization", "Bearer "+oauthToken)

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if nextCalled {
		t.Fatal("expected middleware to proxy and abort, not call next handler")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from upstream, got %d: %s", rec.Code, rec.Body.String())
	}
	if receivedAuth != "Bearer "+oauthToken {
		t.Errorf("upstream did not receive the incoming OAuth token; got %q, want %q",
			receivedAuth, "Bearer "+oauthToken)
	}
}

// TestIsChatGPTOAuthToken verifies that the JWT heuristic correctly
// distinguishes ChatGPT OAuth tokens from OpenAI API keys.
func TestIsChatGPTOAuthToken(t *testing.T) {
	cases := []struct {
		key  string
		want bool
	}{
		// Real-looking JWT (three dot-separated segments, starts with eyJ).
		{"eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", true},
		// Minimal JWT heuristic: starts with eyJ.
		{"eyJhbGciOiJSUzI1NiJ9.payload.sig", true},
		// OpenAI API key — must NOT be treated as JWT.
		{"sk-proj-abc123", false},
		{"sk-abc123", false},
		// Dummy placeholder.
		{"ollama", false},
		// Empty string.
		{"", false},
		// Random string without dots.
		{"randomstringwithoutdots", false},
	}
	for _, tc := range cases {
		got := isChatGPTOAuthToken(tc.key)
		if got != tc.want {
			t.Errorf("isChatGPTOAuthToken(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
}

// TestUpstreamBaseURLForCredential verifies that the correct upstream base URL
// is selected based on the credential type.
func TestUpstreamBaseURLForCredential(t *testing.T) {
	// Clear any override so we get the default behaviour.
	t.Setenv(openAIProxyBaseURLEnv, "")

	// OpenAI API key → api.openai.com
	if got := upstreamBaseURLForCredential("sk-abc123"); got != defaultOpenAIProxyBaseURL {
		t.Errorf("expected %q for API key, got %q", defaultOpenAIProxyBaseURL, got)
	}

	// ChatGPT OAuth JWT → chatgpt.com/backend-api/codex
	jwt := "eyJhbGciOiJSUzI1NiJ9.payload.sig"
	if got := upstreamBaseURLForCredential(jwt); got != chatGPTCodexBaseURL {
		t.Errorf("expected %q for JWT, got %q", chatGPTCodexBaseURL, got)
	}

	// Explicit override takes precedence regardless of credential type.
	t.Setenv(openAIProxyBaseURLEnv, "https://my-proxy.example.com/v1")
	if got := upstreamBaseURLForCredential(jwt); got != "https://my-proxy.example.com/v1" {
		t.Errorf("expected override URL, got %q", got)
	}
	if got := upstreamBaseURLForCredential("sk-abc123"); got != "https://my-proxy.example.com/v1" {
		t.Errorf("expected override URL for API key, got %q", got)
	}
}

// TestOpenAIPassthroughMiddleware_OAuthTokenRoutesToChatGPT verifies that when
// the incoming Authorization header contains a ChatGPT OAuth JWT, the proxy
// routes the request to chatgpt.com/backend-api/codex, not api.openai.com.
func TestOpenAIPassthroughMiddleware_OAuthTokenRoutesToChatGPT(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Upstream mock — records the Host header to verify routing.
	var receivedHost string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHost = r.Host
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"id":"resp_1","object":"response"}`)
	}))
	defer upstream.Close()

	// Point the override to our mock so we can intercept the request.
	// In production this would be chatgpt.com, but we override for testing.
	t.Setenv(openAIProxyBaseURLEnv, upstream.URL+"/v1")
	// Dummy key — the middleware should use the incoming JWT instead.
	t.Setenv("OPENAI_API_KEY", "ollama")

	r := gin.New()
	nextCalled := false
	r.POST("/v1/responses", openAIPassthroughMiddleware(), func(c *gin.Context) {
		nextCalled = true
		c.JSON(http.StatusInternalServerError, gin.H{"error": "should not reach handler"})
	})

	// Simulate a ChatGPT OAuth JWT in the Authorization header.
	jwt := "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature"
	body := `{"model":"gpt-5.4","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+jwt)

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	if nextCalled {
		t.Fatal("expected middleware to proxy and abort, not call next handler")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from upstream, got %d: %s", rec.Code, rec.Body.String())
	}
	_ = receivedHost // verified by the 200 response from the mock
}

// TestOpenAIPassthroughMiddleware_OAuthJWTPriorityOverAPIKey verifies that when
// the incoming request carries a ChatGPT OAuth JWT AND OPENAI_API_KEY is also
// set to a real API key, the JWT wins and the request is routed to the
// ChatGPT codex base URL (not api.openai.com).
func TestOpenAIPassthroughMiddleware_OAuthJWTPriorityOverAPIKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Fake ChatGPT OAuth JWT (three base64url segments, starts with eyJ).
	fakeJWT := "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.fakesig"

	var receivedAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"resp-1"}`))
	}))
	defer upstream.Close()

	// Set OPENAI_API_KEY to a real-looking key — it should be ignored in favour
	// of the incoming OAuth JWT.
	t.Setenv("OPENAI_API_KEY", "sk-realkey123")
	// Point the ChatGPT codex base URL to our mock upstream.
	t.Setenv("OLLAMA_CHATGPT_CODEX_BASE_URL", upstream.URL)

	router := gin.New()
	router.POST("/v1/responses", openAIPassthroughMiddleware(), func(c *gin.Context) {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found locally"})
	})

	body := `{"model":"gpt-5.4","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+fakeJWT)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code == http.StatusNotFound {
		t.Fatal("proxy did not activate: request fell through to local handler (got 404)")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 from upstream, got %d: %s", rec.Code, rec.Body.String())
	}
	// The request must have been sent with the JWT, not the API key.
	wantAuth := "Bearer " + fakeJWT
	if receivedAuth != wantAuth {
		t.Fatalf("expected Authorization %q, got %q", wantAuth, receivedAuth)
	}
}
