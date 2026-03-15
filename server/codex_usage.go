package server

// codex_usage.go — implements GET /api/codex/usage (and the ChatGPT-style
// alias GET /backend-api/api/codex/usage) so that the Codex TUI's /status
// slash-command can display usage information when using Ollama as the local
// inference provider.
//
// The Codex backend-client calls:
//
//	GET {chatgpt_base_url}/api/codex/usage
//
// and expects a RateLimitStatusPayload JSON response.  When Ollama is the
// provider there are no rate limits, so we return a minimal payload that
// signals "unlimited local inference" — this causes the /status overlay to
// render the rate-limit bars as 0% used with no reset time.
//
// The chatgpt_base_url is set to http://localhost:11434/backend-api/ by the
// Ollama launcher (cmd/launch/codex.go) via the -c flag so that Codex routes
// these requests to the local Ollama server instead of chatgpt.com.

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
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
// It is the response body for GET /api/codex/usage.
type codexRateLimitStatusPayload struct {
	// PlanType is the account plan type.  We report "free" for local Ollama.
	PlanType string `json:"plan_type"`
	// RateLimit is the primary Codex rate-limit details.  nil means no limit.
	RateLimit *codexRateLimitStatusDetails `json:"rate_limit,omitempty"`
	// Credits is the credit status.  nil means not applicable.
	Credits *codexCreditStatusDetails `json:"credits,omitempty"`
	// AdditionalRateLimits lists any extra named rate limits.
	AdditionalRateLimits []any `json:"additional_rate_limits,omitempty"`
}

// buildCodexUsagePayload constructs a RateLimitStatusPayload that represents
// "unlimited local inference" — 0% used, no rate limits, credits not
// applicable.  The reset timestamp is set to one hour from now so that the
// Codex TUI renders a valid (non-zero) reset time in the /status overlay.
func buildCodexUsagePayload() codexRateLimitStatusPayload {
	resetAt := time.Now().Add(time.Hour).Unix()
	// A 5-hour window at 0% used, resetting in one hour.
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

// CodexUsageHandler handles GET /api/codex/usage.
//
// It returns a RateLimitStatusPayload JSON that signals "unlimited local
// inference" to the Codex TUI's /status slash-command.
func (s *Server) CodexUsageHandler(c *gin.Context) {
	c.JSON(http.StatusOK, buildCodexUsagePayload())
}
