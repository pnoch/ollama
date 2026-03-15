package launch

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ─── PKCE helpers ─────────────────────────────────────────────────────────────

func TestGeneratePKCE(t *testing.T) {
	verifier, challenge, err := generatePKCE()
	if err != nil {
		t.Fatalf("generatePKCE: %v", err)
	}
	if len(verifier) == 0 {
		t.Fatal("verifier is empty")
	}
	if len(challenge) == 0 {
		t.Fatal("challenge is empty")
	}
	// Verifier must be valid URL-safe base64 without padding.
	if _, err := base64.RawURLEncoding.DecodeString(verifier); err != nil {
		t.Fatalf("verifier is not valid base64url: %v", err)
	}
	// Challenge must be valid URL-safe base64 without padding.
	if _, err := base64.RawURLEncoding.DecodeString(challenge); err != nil {
		t.Fatalf("challenge is not valid base64url: %v", err)
	}
	// Two calls must produce different values.
	v2, c2, _ := generatePKCE()
	if verifier == v2 {
		t.Error("generatePKCE returned the same verifier twice")
	}
	if challenge == c2 {
		t.Error("generatePKCE returned the same challenge twice")
	}
}

func TestGenerateState(t *testing.T) {
	s1, err := generateState()
	if err != nil {
		t.Fatalf("generateState: %v", err)
	}
	s2, err := generateState()
	if err != nil {
		t.Fatalf("generateState: %v", err)
	}
	if s1 == s2 {
		t.Error("generateState returned the same state twice")
	}
}

// ─── buildAuthorizeURL ────────────────────────────────────────────────────────

func TestBuildAuthorizeURL(t *testing.T) {
	u := buildAuthorizeURL(
		"https://auth.openai.com",
		"test-client-id",
		"http://localhost:1455/auth/callback",
		"challenge123",
		"state456",
	)
	for _, want := range []string{
		"https://auth.openai.com/oauth/authorize",
		"response_type=code",
		"client_id=test-client-id",
		"redirect_uri=",
		"scope=",
		"code_challenge=challenge123",
		"code_challenge_method=S256",
		"state=state456",
	} {
		if !strings.Contains(u, want) {
			t.Errorf("authorize URL missing %q\n  got: %s", want, u)
		}
	}
}

// ─── auth.json storage ────────────────────────────────────────────────────────

func TestSaveAndLoadAuthJSON(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	now := time.Now().UTC().Truncate(time.Second)
	a := openAIAuthJSON{
		OpenAIAPIKey: "sk-test-key",
		Tokens: &openAITokenData{
			IDToken:      "id.tok.en",
			AccessToken:  "access.tok.en",
			RefreshToken: "refresh.tok.en",
		},
		LastRefresh: &now,
	}
	if err := saveAuthJSON(a); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}

	got, err := loadAuthJSON()
	if err != nil {
		t.Fatalf("loadAuthJSON: %v", err)
	}
	if got.OpenAIAPIKey != a.OpenAIAPIKey {
		t.Errorf("OpenAIAPIKey: got %q, want %q", got.OpenAIAPIKey, a.OpenAIAPIKey)
	}
	if got.Tokens == nil {
		t.Fatal("Tokens is nil after round-trip")
	}
	if got.Tokens.IDToken != a.Tokens.IDToken {
		t.Errorf("IDToken: got %q, want %q", got.Tokens.IDToken, a.Tokens.IDToken)
	}
}

func TestLoadAuthJSON_Missing(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	got, err := loadAuthJSON()
	if err != nil {
		t.Fatalf("loadAuthJSON on missing file: %v", err)
	}
	if got != (openAIAuthJSON{}) {
		t.Errorf("expected zero value for missing file, got %+v", got)
	}
}

func TestAuthJSONMode(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	if err := saveAuthJSON(openAIAuthJSON{OpenAIAPIKey: "sk-x"}); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}
	path := filepath.Join(dir, ".codex", "auth.json")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("auth.json mode: got %o, want 0600", mode)
	}
}

// ─── LoadStoredOpenAIAPIKey ───────────────────────────────────────────────────

func TestLoadStoredOpenAIAPIKey(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// No file → empty string.
	if got := LoadStoredOpenAIAPIKey(); got != "" {
		t.Errorf("expected empty string for missing file, got %q", got)
	}

	// Write a key.
	if err := saveAuthJSON(openAIAuthJSON{OpenAIAPIKey: "sk-stored"}); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}
	if got := LoadStoredOpenAIAPIKey(); got != "sk-stored" {
		t.Errorf("LoadStoredOpenAIAPIKey: got %q, want %q", got, "sk-stored")
	}
}

// ─── SaveOpenAIAPIKey ─────────────────────────────────────────────────────────

func TestSaveOpenAIAPIKey(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	if err := SaveOpenAIAPIKey("sk-direct"); err != nil {
		t.Fatalf("SaveOpenAIAPIKey: %v", err)
	}
	got, err := loadAuthJSON()
	if err != nil {
		t.Fatalf("loadAuthJSON: %v", err)
	}
	if got.OpenAIAPIKey != "sk-direct" {
		t.Errorf("got %q, want %q", got.OpenAIAPIKey, "sk-direct")
	}
	if got.Tokens != nil {
		t.Error("Tokens should be nil after SaveOpenAIAPIKey")
	}
}

func TestSaveOpenAIAPIKey_Empty(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	if err := SaveOpenAIAPIKey(""); err == nil {
		t.Error("expected error for empty API key")
	}
	if err := SaveOpenAIAPIKey("   "); err == nil {
		t.Error("expected error for whitespace-only API key")
	}
}

// ─── LogoutOpenAI ─────────────────────────────────────────────────────────────

func TestLogoutOpenAI(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Write credentials first.
	if err := saveAuthJSON(openAIAuthJSON{OpenAIAPIKey: "sk-x"}); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}
	if err := LogoutOpenAI(); err != nil {
		t.Fatalf("LogoutOpenAI: %v", err)
	}
	// File should be gone (empty struct → deleted).
	path := filepath.Join(dir, ".codex", "auth.json")
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("auth.json should be deleted after logout")
	}
}

func TestLogoutOpenAI_NoFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Should not error when there's nothing to log out of.
	if err := LogoutOpenAI(); err != nil {
		t.Fatalf("LogoutOpenAI on missing file: %v", err)
	}
}

// ─── WhoamiOpenAI ─────────────────────────────────────────────────────────────

func TestWhoamiOpenAI_NotSignedIn(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	_, err := WhoamiOpenAI()
	if err == nil {
		t.Error("expected error when not signed in")
	}
}

func TestWhoamiOpenAI_APIKeyOnly(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	if err := saveAuthJSON(openAIAuthJSON{OpenAIAPIKey: "sk-abcdefghijklmnop"}); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}
	info, err := WhoamiOpenAI()
	if err != nil {
		t.Fatalf("WhoamiOpenAI: %v", err)
	}
	// Should show masked key (first 8 chars of "sk-abcdefghijklmnop" = "sk-abcde").
	if !strings.Contains(info, "sk-abcde") {
		t.Errorf("expected masked key in output, got %q", info)
	}
}

func TestWhoamiOpenAI_WithIDToken(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	// Build a minimal JWT with an email claim.
	claims := map[string]string{"email": "user@example.com"}
	claimsJSON, _ := json.Marshal(claims)
	payload := base64.RawURLEncoding.EncodeToString(claimsJSON)
	idToken := "header." + payload + ".signature"

	if err := saveAuthJSON(openAIAuthJSON{
		OpenAIAPIKey: "sk-x",
		Tokens: &openAITokenData{
			IDToken:     idToken,
			AccessToken: "acc",
		},
	}); err != nil {
		t.Fatalf("saveAuthJSON: %v", err)
	}

	info, err := WhoamiOpenAI()
	if err != nil {
		t.Fatalf("WhoamiOpenAI: %v", err)
	}
	if info != "user@example.com" {
		t.Errorf("got %q, want %q", info, "user@example.com")
	}
}

// ─── emailFromIDToken ─────────────────────────────────────────────────────────

func TestEmailFromIDToken(t *testing.T) {
	claims := map[string]string{"email": "alice@example.com"}
	claimsJSON, _ := json.Marshal(claims)
	payload := base64.RawURLEncoding.EncodeToString(claimsJSON)
	token := "hdr." + payload + ".sig"

	email, err := emailFromIDToken(token)
	if err != nil {
		t.Fatalf("emailFromIDToken: %v", err)
	}
	if email != "alice@example.com" {
		t.Errorf("got %q, want %q", email, "alice@example.com")
	}
}

func TestEmailFromIDToken_Invalid(t *testing.T) {
	if _, err := emailFromIDToken("notajwt"); err == nil {
		t.Error("expected error for non-JWT string")
	}
}

// ─── bindCallbackServer ───────────────────────────────────────────────────────

func TestBindCallbackServer_DefaultPort(t *testing.T) {
	ln, port, err := bindCallbackServer(0) // port 0 = OS-assigned
	if err != nil {
		t.Fatalf("bindCallbackServer: %v", err)
	}
	defer ln.Close()
	if port == 0 {
		t.Error("expected non-zero port")
	}
}

func TestBindCallbackServer_FallbackOnBusy(t *testing.T) {
	// Bind a listener on a known port, then try to bind again on the same
	// port — the function should fall back to a random port.
	first, port1, err := bindCallbackServer(0)
	if err != nil {
		t.Fatalf("first bind: %v", err)
	}
	defer first.Close()

	second, port2, err := bindCallbackServer(port1)
	if err != nil {
		t.Fatalf("second bind: %v", err)
	}
	defer second.Close()

	if port2 == port1 {
		t.Error("expected fallback to a different port")
	}
}

// ─── token exchange (mocked) ──────────────────────────────────────────────────

func TestExchangeCodeForTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if r.FormValue("grant_type") != "authorization_code" {
			http.Error(w, "wrong grant_type", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"id_token":      "id.tok.en",
			"access_token":  "access.tok.en",
			"refresh_token": "refresh.tok.en",
		})
	}))
	defer srv.Close()

	tokens, err := exchangeCodeForTokens(
		context.Background(),
		srv.URL,
		"client-id",
		"http://localhost:1455/auth/callback",
		"verifier",
		"code123",
	)
	if err != nil {
		t.Fatalf("exchangeCodeForTokens: %v", err)
	}
	if tokens.IDToken != "id.tok.en" {
		t.Errorf("IDToken: got %q", tokens.IDToken)
	}
	if tokens.AccessToken != "access.tok.en" {
		t.Errorf("AccessToken: got %q", tokens.AccessToken)
	}
	if tokens.RefreshToken != "refresh.tok.en" {
		t.Errorf("RefreshToken: got %q", tokens.RefreshToken)
	}
}

func TestExchangeCodeForTokens_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	_, err := exchangeCodeForTokens(context.Background(), srv.URL, "c", "r", "v", "bad-code")
	if err == nil {
		t.Error("expected error for 400 response")
	}
}

func TestExchangeIDTokenForAPIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if r.FormValue("grant_type") != "urn:ietf:params:oauth:grant-type:token-exchange" {
			http.Error(w, "wrong grant_type", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"access_token": "sk-exchanged-api-key",
		})
	}))
	defer srv.Close()

	key, err := exchangeIDTokenForAPIKey(context.Background(), srv.URL, "client-id", "id.tok.en")
	if err != nil {
		t.Fatalf("exchangeIDTokenForAPIKey: %v", err)
	}
	if key != "sk-exchanged-api-key" {
		t.Errorf("got %q, want %q", key, "sk-exchanged-api-key")
	}
}
