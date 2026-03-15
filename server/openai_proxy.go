package server

// openai_proxy.go — transparent passthrough proxy to the real OpenAI API.
//
// When Ollama receives a request for a model that is neither locally installed
// nor a cloud (ollama.com) model, it forwards the request verbatim to the
// upstream OpenAI-compatible API.  This lets a single Ollama endpoint serve
// both local models and OpenAI models without the caller needing to know which
// is which.
//
// Configuration (all optional):
//
//	OLLAMA_OPENAI_PROXY_BASE_URL  – upstream base URL (default: https://api.openai.com/v1)
//	OPENAI_API_KEY                – Bearer token forwarded to upstream
//	OPENAI_ORG_ID                 – forwarded as OpenAI-Organization header
//	OPENAI_PROJECT_ID             – forwarded as OpenAI-Project header
//
// The middleware is inserted before cloudPassthroughMiddleware in the chain so
// that explicit ":cloud" models still go through the Ollama cloud proxy.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

const (
	// defaultOpenAIProxyBaseURL is the upstream used when
	// OLLAMA_OPENAI_PROXY_BASE_URL is not set and the credential is an
	// OpenAI API key (sk-...).
	defaultOpenAIProxyBaseURL = "https://api.openai.com/v1"

	// chatGPTCodexBaseURL is the upstream used when the credential is a
	// ChatGPT OAuth JWT token.  This matches the base_url that the Codex
	// Rust client uses when auth_mode == AuthMode::Chatgpt.
	chatGPTCodexBaseURL = "https://chatgpt.com/backend-api/codex"

	// openAIProxyBaseURLEnv is the environment variable that overrides the
	// upstream OpenAI base URL.
	openAIProxyBaseURLEnv = "OLLAMA_OPENAI_PROXY_BASE_URL"
)

// openAIProxyBaseURL returns the upstream base URL for the OpenAI proxy,
// trimming any trailing slash so we can append paths cleanly.
func openAIProxyBaseURL() string {
	if v := strings.TrimSpace(os.Getenv(openAIProxyBaseURLEnv)); v != "" {
		return strings.TrimRight(v, "/")
	}
	return defaultOpenAIProxyBaseURL
}

// openAIAPIKey returns the API key to use when proxying to OpenAI.
// Priority order:
//  1. OPENAI_API_KEY environment variable
//  2. Key stored in ~/.codex/auth.json (written by `ollama openai login`)
//
// Returns an empty string when no key is available, which will cause the
// upstream to return a 401 that is surfaced cleanly to the caller.
func openAIAPIKey() string {
	if v := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); v != "" {
		return v
	}
	if v := loadStoredOpenAIAPIKey(); v != "" {
		return v
	}
	return ""
}

// loadStoredOpenAIAPIKey reads the OPENAI_API_KEY field from
// ~/.codex/auth.json.  Errors are silently ignored so the proxy never fails
// to start due to a missing or malformed credentials file.
func loadStoredOpenAIAPIKey() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(home + "/.codex/auth.json")
	if err != nil {
		return ""
	}
	var auth struct {
		OpenAIAPIKey string `json:"OPENAI_API_KEY"`
	}
	if err := json.Unmarshal(data, &auth); err != nil {
		return ""
	}
	return strings.TrimSpace(auth.OpenAIAPIKey)
}

// isModelLocalOrCloud returns true when the model name refers to a locally
// installed Ollama model or to an explicit Ollama cloud model (":cloud"
// suffix).  In either case the request should NOT be proxied to OpenAI.
func isModelLocalOrCloud(modelName string) bool {
	// Parse the model reference to detect an explicit ":cloud" source tag.
	ref, err := parseAndValidateModelRef(modelName)
	if err != nil {
		// Unparseable name — let the normal handler deal with it.
		return true
	}
	if ref.Source == modelSourceCloud {
		return true
	}

	// Check whether the model is locally installed by looking for its
	// manifest on disk.  An os.IsNotExist error means it is not local.
	_, manifestErr := manifest.ParseNamedManifest(ref.Name)
	if manifestErr == nil {
		// Manifest found — model is local.
		return true
	}
	// Any error other than "not found" (e.g. permission denied, corrupt
	// manifest) is treated conservatively as "local" so we don't
	// accidentally proxy a request that should stay local.
	if !os.IsNotExist(manifestErr) {
		return true
	}
	return false
}

// openAIPassthroughMiddleware returns a gin.HandlerFunc that transparently
// proxies requests to the upstream OpenAI API when the requested model is
// neither a local Ollama model nor an explicit cloud model.
//
// The middleware is designed to be placed first in the handler chain for
// OpenAI-compatible endpoints:
//
//	r.POST("/v1/chat/completions",
//	    openAIPassthroughMiddleware(),
//	    cloudPassthroughMiddleware(...),
//	    middleware.ChatMiddleware(),
//	    s.ChatHandler)
func openAIPassthroughMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Only intercept POST requests; GET requests (e.g. /v1/models) are
		// handled by a separate path.
		if c.Request.Method != http.MethodPost {
			c.Next()
			return
		}

		// Determine the credential to use for proxying.
		//
		// Priority:
		//  1. If the incoming request carries a ChatGPT OAuth JWT (eyJ…), use
		//     that — it must be routed to chatgpt.com, not api.openai.com.
		//     This handles the case where the user switches to a ChatGPT-only
		//     model (gpt-5.4, o3, etc.) inside Codex while logged in via OAuth.
		//  2. OPENAI_API_KEY env var or stored key in ~/.codex/auth.json.
		//  3. Any other real credential in the incoming Authorization header.
		var apiKey string
		if incomingAuth := c.GetHeader("Authorization"); isRealCredential(incomingAuth) {
			incomingToken := strings.TrimSpace(strings.TrimPrefix(incomingAuth, "Bearer "))
			if isChatGPTOAuthToken(incomingToken) {
				// Always prefer the OAuth JWT so it routes to chatgpt.com.
				apiKey = incomingToken
			}
		}
		if apiKey == "" {
			apiKey = openAIAPIKey()
		}
		if (apiKey == "" || apiKey == "ollama") {
			// Last resort: use whatever real credential Codex sent.
			if incomingAuth := c.GetHeader("Authorization"); isRealCredential(incomingAuth) {
				apiKey = strings.TrimSpace(strings.TrimPrefix(incomingAuth, "Bearer "))
			}
		}
		if apiKey == "" {
			// No API key configured — skip proxy and let the local handler
			// attempt to serve the request (it will fail gracefully if the
			// model is not found locally).
			c.Next()
			return
		}

		// Buffer the request body so we can inspect the model field without
		// consuming the stream.  readRequestBody also handles zstd
		// decompression and restores c.Request.Body for downstream handlers.
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		modelName, ok := extractModelField(body)
		if !ok {
			// No model field — let the normal handler return the appropriate
			// error.
			c.Next()
			return
		}

		if isModelLocalOrCloud(modelName) {
			// Local or explicit cloud model — do not proxy to OpenAI.
			c.Next()
			return
		}

		// The model is not local and not an explicit cloud model.  Proxy to
		// the upstream OpenAI API.
		proxyToOpenAI(c, body, apiKey)
		c.Abort()
	}
}

// openAIModelsPassthroughMiddleware merges the upstream OpenAI /v1/models list
// into the local Ollama model list.  It is used on the GET /v1/models route.
//
// When OPENAI_API_KEY (or a stored credential) is available the middleware:
//  1. Lets the local handler run and captures its response (local models).
//  2. Fetches the upstream /v1/models list from the OpenAI-compatible API.
//  3. Merges the two lists, deduplicating by model ID.
//  4. Writes the merged list back to the client.
//
// If no API key is configured the middleware is a transparent pass-through so
// only local models are returned.
func openAIModelsPassthroughMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		apiKey := openAIAPIKey()
		if apiKey == "" {
			// No upstream key — serve only local models.
			c.Next()
			return
		}

		// Intercept the response written by the downstream handler so we can
		// merge the upstream model list into it.
		bw := &modelsBufferedWriter{ResponseWriter: c.Writer}
		c.Writer = bw
		c.Next()
		c.Writer = bw.ResponseWriter // restore the original gin.ResponseWriter

		// Only merge on a successful local response.
		if bw.status != 0 && bw.status != http.StatusOK {
			// Write the original (error) response unchanged.
			c.Writer.WriteHeader(bw.status)
			_, _ = c.Writer.Write(bw.body)
			return
		}

		// Parse the local model list.
		var local openaiListCompletion
		if err := json.Unmarshal(bw.body, &local); err != nil {
			// Cannot parse — return the original response unchanged.
			if bw.status != 0 {
				c.Writer.WriteHeader(bw.status)
			}
			_, _ = c.Writer.Write(bw.body)
			return
		}

		// Fetch the upstream model list with a short timeout.
		upstreamModels := fetchOpenAIModels(c.Request.Context(), apiKey)

		// Merge: start with local models, then append upstream models that are
		// not already present locally (deduplicate by ID).
		merged := local.Data
		localIDs := make(map[string]struct{}, len(merged))
		for _, m := range merged {
			localIDs[m.ID] = struct{}{}
		}
		for _, m := range upstreamModels {
			if _, exists := localIDs[m.ID]; !exists {
				merged = append(merged, m)
			}
		}

		result := openaiListCompletion{Object: "list", Data: merged}
		data, err := json.Marshal(result)
		if err != nil {
			// Fallback to original response on marshal error.
			if bw.status != 0 {
				c.Writer.WriteHeader(bw.status)
			}
			_, _ = c.Writer.Write(bw.body)
			return
		}

		c.Writer.Header().Set("Content-Type", "application/json")
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	}
}

// openaiModelEntry is a minimal OpenAI-compatible model object used for
// merging the upstream model list with local Ollama models.
type openaiModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// openaiListCompletion is the OpenAI /v1/models response envelope.
type openaiListCompletion struct {
	Object string             `json:"object"`
	Data   []openaiModelEntry `json:"data"`
}

// modelsBufferedWriter captures the response written by a downstream handler
// so the middleware can inspect and modify it before sending to the client.
// It embeds gin.ResponseWriter so it satisfies the full gin.ResponseWriter
// interface while overriding only WriteHeader and Write.
type modelsBufferedWriter struct {
	gin.ResponseWriter
	body   []byte
	status int
}

func (w *modelsBufferedWriter) WriteHeader(code int) {
	w.status = code
}

func (w *modelsBufferedWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	w.body = append(w.body, b...)
	return len(b), nil
}

// fetchOpenAIModels fetches the model list from the upstream OpenAI-compatible
// API.  Returns an empty slice on any error so the caller can still serve the
// local model list.
func fetchOpenAIModels(ctx context.Context, apiKey string) []openaiModelEntry {
	baseURL := openAIProxyBaseURL()
	targetURL := strings.TrimRight(baseURL, "/") + "/models"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		slog.Debug("openai proxy: failed to build models request", "error", err)
		return nil
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	if org := strings.TrimSpace(os.Getenv("OPENAI_ORG_ID")); org != "" {
		req.Header.Set("OpenAI-Organization", org)
	}
	if proj := strings.TrimSpace(os.Getenv("OPENAI_PROJECT_ID")); proj != "" {
		req.Header.Set("OpenAI-Project", proj)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		slog.Debug("openai proxy: models fetch failed", "error", err)
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		slog.Debug("openai proxy: models fetch returned non-200", "status", resp.StatusCode)
		return nil
	}

	var result openaiListCompletion
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		slog.Debug("openai proxy: failed to decode models response", "error", err)
		return nil
	}
	return result.Data
}

// isChatGPTOAuthToken reports whether the credential looks like a ChatGPT
// OAuth JWT access token rather than an OpenAI API key.
//
// ChatGPT OAuth tokens are JWTs: three base64url-encoded segments separated by
// dots, where the first segment decodes to a JSON header beginning with
// {"alg":...}.  In practice they always start with "eyJ" (base64url of '{').
// OpenAI API keys always start with "sk-".
func isChatGPTOAuthToken(key string) bool {
	// Cheap heuristic: JWTs start with "eyJ" (base64url({"alg"...)).
	// OpenAI API keys start with "sk-".  Anything else is treated as an
	// API key and sent to api.openai.com.
	return strings.HasPrefix(key, "eyJ") || strings.Count(key, ".") >= 2
}

// chatGPTCodexBaseURLEnv is an optional env var that overrides the ChatGPT
// codex base URL.  It is intended for testing only.
const chatGPTCodexBaseURLEnv = "OLLAMA_CHATGPT_CODEX_BASE_URL"

// upstreamBaseURLForCredential returns the correct upstream base URL for the
// given credential.  ChatGPT OAuth JWTs use chatgpt.com/backend-api/codex;
// OpenAI API keys use the configured OLLAMA_OPENAI_PROXY_BASE_URL (default:
// https://api.openai.com/v1).
func upstreamBaseURLForCredential(apiKey string) string {
	// Allow explicit override regardless of credential type.
	if v := strings.TrimSpace(os.Getenv(openAIProxyBaseURLEnv)); v != "" {
		return strings.TrimRight(v, "/")
	}
	if isChatGPTOAuthToken(apiKey) {
		// Allow test override of the ChatGPT codex base URL.
		if v := strings.TrimSpace(os.Getenv(chatGPTCodexBaseURLEnv)); v != "" {
			return strings.TrimRight(v, "/")
		}
		return chatGPTCodexBaseURL
	}
	return defaultOpenAIProxyBaseURL
}

// proxyToOpenAI forwards the current request to the upstream OpenAI API and
// streams the response back to the caller.
func proxyToOpenAI(c *gin.Context, body []byte, apiKey string) {
	baseURL := upstreamBaseURLForCredential(apiKey)

	base, err := url.Parse(baseURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "invalid OpenAI proxy base URL: " + err.Error()})
		return
	}

	// Resolve the full target URL by appending the request path and query.
	// Strip the leading "/v1" prefix from the Ollama path because both
	// https://api.openai.com/v1 and https://chatgpt.com/backend-api/codex
	// already include their own path prefix.
	upstreamPath := c.Request.URL.Path
	if strings.HasPrefix(upstreamPath, "/v1") {
		upstreamPath = upstreamPath[len("/v1"):]
	}
	targetURL := base.ResolveReference(&url.URL{
		Path:     upstreamPath,
		RawQuery: c.Request.URL.RawQuery,
	})

	outReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		targetURL.String(),
		bytes.NewReader(body),
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Copy safe request headers from the incoming request.
	copyProxyRequestHeaders(outReq.Header, c.Request.Header)

	// Override / set the Authorization header with the real OpenAI key.
	outReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Forward optional organisation and project headers.
	if org := strings.TrimSpace(os.Getenv("OPENAI_ORG_ID")); org != "" {
		outReq.Header.Set("OpenAI-Organization", org)
	}
	if proj := strings.TrimSpace(os.Getenv("OPENAI_PROJECT_ID")); proj != "" {
		outReq.Header.Set("OpenAI-Project", proj)
	}

	if outReq.Header.Get("Content-Type") == "" && len(body) > 0 {
		outReq.Header.Set("Content-Type", "application/json")
	}

	slog.Debug("openai proxy: forwarding request",
		"path", c.Request.URL.Path,
		"target", targetURL.String(),
	)

	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "OpenAI proxy error: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	if err := copyProxyStream(c.Writer, resp.Body); err != nil {
		ctxErr := c.Request.Context().Err()
		if errors.Is(err, context.Canceled) && errors.Is(ctxErr, context.Canceled) {
			slog.Debug("openai proxy: stream closed by client",
				"path", c.Request.URL.Path,
				"status", resp.StatusCode,
			)
			return
		}
		slog.Warn("openai proxy: response copy failed",
			"path", c.Request.URL.Path,
			"status", resp.StatusCode,
			"error", err,
		)
	}
}

// openAIProxyModelName extracts the model name from a request body and
// returns the fully-qualified Ollama model.Name for local existence checks.
// Returns the zero Name and false if the model field is absent or unparseable.
func openAIProxyModelName(body []byte) (model.Name, bool) {
	raw, ok := extractModelField(body)
	if !ok {
		return model.Name{}, false
	}
	ref, err := parseAndValidateModelRef(raw)
	if err != nil {
		return model.Name{}, false
	}
	return ref.Name, true
}
