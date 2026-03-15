package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

// TestBuildCodexUsagePayload verifies that buildCodexUsagePayload returns a
// well-formed RateLimitStatusPayload that signals "unlimited local inference".
func TestBuildCodexUsagePayload(t *testing.T) {
	before := time.Now()
	p := buildCodexUsagePayload()
	after := time.Now()

	if p.PlanType != "free" {
		t.Errorf("PlanType = %q, want %q", p.PlanType, "free")
	}

	if p.RateLimit == nil {
		t.Fatal("RateLimit is nil, want non-nil")
	}
	if !p.RateLimit.Allowed {
		t.Error("RateLimit.Allowed = false, want true")
	}
	if p.RateLimit.LimitReached {
		t.Error("RateLimit.LimitReached = true, want false")
	}
	if p.RateLimit.PrimaryWindow == nil {
		t.Fatal("RateLimit.PrimaryWindow is nil, want non-nil")
	}
	if p.RateLimit.PrimaryWindow.UsedPercent != 0 {
		t.Errorf("PrimaryWindow.UsedPercent = %d, want 0", p.RateLimit.PrimaryWindow.UsedPercent)
	}
	if p.RateLimit.PrimaryWindow.LimitWindowSeconds != 5*60*60 {
		t.Errorf("PrimaryWindow.LimitWindowSeconds = %d, want %d", p.RateLimit.PrimaryWindow.LimitWindowSeconds, 5*60*60)
	}
	// ResetAt should be approximately one hour from now.
	resetAt := p.RateLimit.PrimaryWindow.ResetAt
	wantMin := before.Add(59 * time.Minute).Unix()
	wantMax := after.Add(61 * time.Minute).Unix()
	if resetAt < wantMin || resetAt > wantMax {
		t.Errorf("PrimaryWindow.ResetAt = %d, want in [%d, %d]", resetAt, wantMin, wantMax)
	}

	if p.Credits == nil {
		t.Fatal("Credits is nil, want non-nil")
	}
	if !p.Credits.HasCredits {
		t.Error("Credits.HasCredits = false, want true")
	}
	if !p.Credits.Unlimited {
		t.Error("Credits.Unlimited = false, want true")
	}
}

// TestCodexUsageHandler_JSON verifies that the handler returns HTTP 200 with a
// valid JSON body that can be decoded into a codexRateLimitStatusPayload.
func TestCodexUsageHandler_JSON(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	s := &Server{}
	router.GET("/api/codex/usage", s.CodexUsageHandler)
	router.GET("/backend-api/api/codex/usage", s.CodexUsageHandler)

	for _, path := range []string{"/api/codex/usage", "/backend-api/api/codex/usage"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
			}

			ct := rec.Header().Get("Content-Type")
			if ct == "" {
				t.Error("Content-Type header is empty")
			}

			var payload codexRateLimitStatusPayload
			if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
				t.Fatalf("failed to decode response body: %v", err)
			}
			if payload.PlanType != "free" {
				t.Errorf("PlanType = %q, want %q", payload.PlanType, "free")
			}
			if payload.RateLimit == nil {
				t.Error("RateLimit is nil in JSON response")
			}
			if payload.Credits == nil {
				t.Error("Credits is nil in JSON response")
			}
		})
	}
}

// TestCodexUsageHandler_MethodNotAllowed verifies that POST to the usage
// endpoint returns 405 Method Not Allowed (not 404 or 200).
func TestCodexUsageHandler_MethodNotAllowed(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.HandleMethodNotAllowed = true
	s := &Server{}
	router.GET("/api/codex/usage", s.CodexUsageHandler)

	req := httptest.NewRequest(http.MethodPost, "/api/codex/usage", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}
