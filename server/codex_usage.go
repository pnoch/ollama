package server

// codex_usage.go — implements the Codex usage / rate-limit endpoints so that
// the Codex TUI's /status slash-command can display usage information for both
// local Ollama models and OpenAI cloud models.
//
// Background
// ----------
// The Codex backend-client calls one of two URLs depending on the path style
// it detects from chatgpt_base_url:
//
//   PathStyle::CodexApi  → GET {base_url}/api/codex/usage
//   PathStyle::ChatGptApi → GET {base_url}/wham/usage
//
// PathStyle is determined by whether chatgpt_base_url contains "/backend-api".
// The Ollama launcher injects:
//
//   -c chatgpt_base_url=http://localhost:11434/backend-api/
//
// so the client picks PathStyle::ChatGptApi and calls:
//
//   GET http://localhost:11434/backend-api/wham/usage
//
// The client also sends:
//   Authorization: Bearer <token>   (ChatGPT OAuth token or OpenAI API key)
//   ChatGPT-Account-Id: <id>        (optional, only for OAuth sessions)
//
// Behaviour
// ---------
//  • If the Authorization header carries a real credential (anything other
//    than the dummy "Bearer ollama" value injected for local-only sessions),
//    this handler proxies the request verbatim to the real ChatGPT backend
//    (https://chatgpt.com/backend-api/wham/usage) and streams the response
//    back.  This gives the /status overlay real rate-limit bars for cloud
//    model sessions (gpt-5.4, o3, etc.).
//
//  • If no real credential is present (dummy key or no key at all), the
//    handler returns a synthetic "unlimited local inference" payload so the
//    /status overlay renders cleanly for pure Ollama sessions.
//
// Both the /api/codex/usage path (CodexApi style) and the /backend-api/…
// paths are registered in routes.go.

import (
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	// chatGPTBackendBase is the upstream ChatGPT backend URL.
	chatGPTBackendBase = "https://chatgpt.com/backend-api"

	// codexUsagePathCodexAPI is the path used by PathStyle::CodexApi.
	codexUsagePathCodexAPI = "/api/codex/usage"

	// codexUsagePathChatGPT is the path used by PathStyle::ChatGptApi.
	codexUsagePathChatGPT = "/wham/usage"

	// dummyOllamaToken is the placeholder API key injected when no real key
	// is available.  Requests carrying only this token are served locally.
	dummyOllamaToken = "ollama"
)

// codexRateLimitWindowSnapshot mirrors the OpenAPI model of the same name.
type codexRateLimitWindowSnapshot struct {
	// UsedPercent is the percentage (0-100) of the window that has been used.
	UsedPercent int `json:"used_percent"`
	// LimitWindowSeconds is the rolling window duration in seconds.
	LimitWindowSeconds int `json:"limit_window_seconds"`
	// ResetAfterSeconds is the number of seconds until the window resets.
	ResetAfterSeconds int `json:"reset_after_seconds"`
	// ResetAt is the Unix timestamp (seconds) when the window resets.
	ResetAt int64 `json:"reset_at"`
}

// codexRateLimitStatusDetails mirrors the OpenAPI model of the same name.
type codexRateLimitStatusDetails struct {
	// Allowed indicates whether requests are currently allowed.
	Allowed bool `json:"allowed"`
	// LimitReached indicates whether the rate limit has been reached.
	LimitReached bool `json:"limit_reached"`
	// PrimaryWindow is the primary (short) rate-limit window snapshot.
	PrimaryWindow *codexRateLimitWindowSnapshot `json:"primary_window,omitempty"`
	// SecondaryWindow is the secondary (long) rate-limit window snapshot.
	SecondaryWindow *codexRateLimitWindowSnapshot `json:"secondary_window,omitempty"`
}

// codexCreditStatusDetails mirrors the OpenAPI model of the same name.
type codexCreditStatusDetails struct {
	// HasCredits indicates whether the account has credits available.
	HasCredits bool `json:"has_credits"`
	// Unlimited indicates whether the account has unlimited credits.
	Unlimited bool `json:"unlimited"`
	// Balance is the current credit balance, if applicable.
	Balance *string `json:"balance,omitempty"`
}

// codexRateLimitStatusPayload mirrors the OpenAPI model of the same name.
// It is the response body for GET /api/codex/usage and GET /wham/usage.
type codexRateLimitStatusPayload struct {
	// PlanType is the account plan type.
	PlanType string `json:"plan_type"`
	// RateLimit is the primary Codex rate-limit details.  nil means no limit.
	RateLimit *codexRateLimitStatusDetails `json:"rate_limit,omitempty"`
	// Credits is the credit status.  nil means not applicable.
	Credits *codexCreditStatusDetails `json:"credits,omitempty"`
	// AdditionalRateLimits lists any extra named rate limits.
	AdditionalRateLimits []any `json:"additional_rate_limits,omitempty"`
}

// buildCodexUsagePayload constructs a RateLimitStatusPayload that represents
// "unlimited local inference" — 0% used, no rate limits, credits unlimited.
// The reset timestamp is set to one hour from now so that the Codex TUI
// renders a valid (non-zero) reset time in the /status overlay.
func buildCodexUsagePayload() codexRateLimitStatusPayload {
	resetAt := time.Now().Add(time.Hour).Unix()
	primaryWindow := &codexRateLimitWindowSnapshot{
		UsedPercent:        0,
		LimitWindowSeconds: 5 * 60 * 60, // 5 hours
		ResetAfterSeconds:  int(time.Until(time.Unix(resetAt, 0)).Seconds()),
		ResetAt:            resetAt,
	}
	return codexRateLimitStatusPayload{
		PlanType: "free",
		RateLimit: &codexRateLimitStatusDetails{
			Allowed:         true,
			LimitReached:    false,
			PrimaryWindow:   primaryWindow,
			SecondaryWindow: nil,
		},
		Credits: &codexCreditStatusDetails{
			HasCredits: true,
			Unlimited:  true,
		},
		AdditionalRateLimits: []any{},
	}
}

// isRealCredential reports whether the Authorization header value carries a
// real credential (not the dummy "ollama" placeholder).
//
// The Codex backend-client always sends "Bearer <token>".  We consider the
// credential real if:
//   - It is non-empty.
//   - The token part (after "Bearer ") is not the dummy "ollama" value.
//   - The token part does not look like a local Ollama dummy key.
func isRealCredential(authHeader string) bool {
	if authHeader == "" {
		return false
	}
	token := strings.TrimPrefix(authHeader, "Bearer ")
	token = strings.TrimSpace(token)
	if token == "" || strings.EqualFold(token, dummyOllamaToken) {
		return false
	}
	return true
}

// proxyCodexUsageToUpstream forwards the usage request to the real ChatGPT
// backend and streams the response back to the client.  It is called when a
// real credential is present (cloud model session).
//
// upstreamPath is the path to append to chatGPTBackendBase, e.g. "/wham/usage"
// or "/api/codex/usage".
func proxyCodexUsageToUpstream(c *gin.Context, upstreamPath string) {
	targetURL := chatGPTBackendBase + upstreamPath

	outReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, targetURL, nil)
	if err != nil {
		slog.Warn("codex usage proxy: failed to build upstream request", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Forward the Authorization header verbatim — it carries the ChatGPT
	// OAuth token or OpenAI API key that the Codex TUI obtained.
	if auth := c.GetHeader("Authorization"); auth != "" {
		outReq.Header.Set("Authorization", auth)
	}

	// Forward the optional ChatGPT-Account-Id header used by OAuth sessions.
	if accountID := c.GetHeader("ChatGPT-Account-Id"); accountID != "" {
		outReq.Header.Set("ChatGPT-Account-Id", accountID)
	}

	// Forward the User-Agent so ChatGPT can identify the Codex client.
	if ua := c.GetHeader("User-Agent"); ua != "" {
		outReq.Header.Set("User-Agent", ua)
	}

	outReq.Header.Set("Accept", "application/json")

	slog.Debug("codex usage proxy: forwarding to upstream", "url", targetURL)

	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		slog.Warn("codex usage proxy: upstream request failed", "error", err)
		// Fall back to the local "unlimited" payload so /status still works.
		c.JSON(http.StatusOK, buildCodexUsagePayload())
		return
	}
	defer resp.Body.Close()

	// Copy response headers and status code.
	for k, vs := range resp.Header {
		for _, v := range vs {
			c.Header(k, v)
		}
	}
	c.Status(resp.StatusCode)

	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		slog.Debug("codex usage proxy: response copy interrupted", "error", err)
	}
}

// CodexUsageHandler handles GET /api/codex/usage and GET /backend-api/wham/usage.
//
// When a real credential is present in the Authorization header, the request
// is proxied to the real ChatGPT backend so the /status overlay shows live
// rate-limit data for cloud model sessions.  Otherwise, a synthetic
// "unlimited local inference" payload is returned for pure Ollama sessions.
func (s *Server) CodexUsageHandler(c *gin.Context) {
	authHeader := c.GetHeader("Authorization")

	if isRealCredential(authHeader) {
		// Determine which upstream path to use based on the request path.
		// Requests arriving at /backend-api/wham/usage use the ChatGPT path;
		// requests at /api/codex/usage use the CodexApi path.
		upstreamPath := codexUsagePathCodexAPI
		if strings.Contains(c.Request.URL.Path, "wham") {
			upstreamPath = codexUsagePathChatGPT
		}
		proxyCodexUsageToUpstream(c, upstreamPath)
		return
	}

	// No real credential — return the local "unlimited" payload.
	c.JSON(http.StatusOK, buildCodexUsagePayload())
}
