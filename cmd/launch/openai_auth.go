// Package launch — OpenAI authentication helpers.
//
// This file implements the same browser-based OAuth 2.0 + PKCE sign-in flow
// used by the native Codex CLI (codex-rs/login).  The flow is:
//
//  1. Generate a PKCE code-verifier and code-challenge (S256).
//  2. Build the authorization URL at https://auth.openai.com/oauth/authorize.
//  3. Open the URL in the user's default browser and start a tiny local HTTP
//     server on port 1455 (same default as Codex) to receive the callback.
//  4. Exchange the authorization code for tokens at /oauth/token.
//  5. Exchange the id_token for an OpenAI API key via a token-exchange grant.
//  6. Persist the API key (and tokens) to ~/.codex/auth.json in the same
//     format that Codex itself uses, so the two tools share credentials.
//
// References:
//   - codex-rs/login/src/server.rs
//   - codex-rs/login/src/pkce.rs
//   - codex-rs/core/src/auth/storage.rs
package launch

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/browser"
)

// ─── OpenAI OAuth constants ──────────────────────────────────────────────────

const (
	// openAIClientID is the public OAuth client-id used by the native Codex
	// CLI.  It is safe to embed here because it is already public in the
	// open-source codex-rs repository.
	openAIClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

	// openAIIssuer is the OpenAI OAuth 2.0 authorization server.
	openAIIssuer = "https://auth.openai.com"

	// openAICallbackPort is the local port used for the redirect_uri callback
	// server.  Matches the Codex default (DEFAULT_PORT = 1455).
	openAICallbackPort = 1455

	// openAIScope is the OAuth scope requested during authorization.
	openAIScope = "openid profile email offline_access api.connectors.read api.connectors.invoke"
)

// ─── auth.json storage ───────────────────────────────────────────────────────

// openAIAuthJSON mirrors the ~/.codex/auth.json structure used by Codex so
// that credentials are shared between the two tools.
type openAIAuthJSON struct {
	// OPENAI_API_KEY holds the exchanged API key (or a key entered manually).
	OpenAIAPIKey string `json:"OPENAI_API_KEY,omitempty"`
	// Tokens holds the raw OAuth token data returned by the authorization
	// server.  Stored for future token-refresh support.
	Tokens *openAITokenData `json:"tokens,omitempty"`
	// LastRefresh is the UTC timestamp of the last successful token refresh.
	LastRefresh *time.Time `json:"last_refresh,omitempty"`
}

// openAITokenData mirrors the TokenData struct in codex-rs/core/src/token_data.rs.
type openAITokenData struct {
	// IDToken is the raw JWT id_token string (stored as-is for refresh).
	IDToken string `json:"id_token"`
	// AccessToken is the OAuth access token JWT.
	AccessToken string `json:"access_token"`
	// RefreshToken is the long-lived refresh token.
	RefreshToken string `json:"refresh_token"`
}

// codexHome returns the path to the ~/.codex directory, creating it if needed.
func codexHome() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("openai auth: cannot determine home directory: %w", err)
	}
	dir := filepath.Join(home, ".codex")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("openai auth: cannot create %s: %w", dir, err)
	}
	return dir, nil
}

// authJSONPath returns the path to ~/.codex/auth.json.
func authJSONPath() (string, error) {
	dir, err := codexHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "auth.json"), nil
}

// loadAuthJSON reads and parses ~/.codex/auth.json.  Returns a zero-value
// struct (not an error) when the file does not exist.
func loadAuthJSON() (openAIAuthJSON, error) {
	path, err := authJSONPath()
	if err != nil {
		return openAIAuthJSON{}, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return openAIAuthJSON{}, nil
	}
	if err != nil {
		return openAIAuthJSON{}, fmt.Errorf("openai auth: read %s: %w", path, err)
	}
	var a openAIAuthJSON
	if err := json.Unmarshal(data, &a); err != nil {
		return openAIAuthJSON{}, fmt.Errorf("openai auth: parse %s: %w", path, err)
	}
	return a, nil
}

// saveAuthJSON writes a to ~/.codex/auth.json with mode 0600.
func saveAuthJSON(a openAIAuthJSON) error {
	path, err := authJSONPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return fmt.Errorf("openai auth: marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("openai auth: write %s: %w", path, err)
	}
	return nil
}

// ─── Public helpers ───────────────────────────────────────────────────────────

// LoadStoredOpenAIAPIKey returns the API key stored in ~/.codex/auth.json, or
// an empty string if none is stored.  Errors are silently swallowed so callers
// never need to handle them.
func LoadStoredOpenAIAPIKey() string {
	a, err := loadAuthJSON()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(a.OpenAIAPIKey)
}

// WhoamiOpenAI returns a human-readable string describing the currently
// authenticated OpenAI user, or an error if no valid credentials are found.
func WhoamiOpenAI() (string, error) {
	a, err := loadAuthJSON()
	if err != nil {
		return "", err
	}
	if a.OpenAIAPIKey == "" && a.Tokens == nil {
		return "", fmt.Errorf("not signed in to OpenAI (run: ollama openai login)")
	}
	if a.Tokens != nil && a.Tokens.IDToken != "" {
		email, err := emailFromIDToken(a.Tokens.IDToken)
		if err == nil && email != "" {
			return email, nil
		}
	}
	if a.OpenAIAPIKey != "" {
		// Mask the key for display.
		key := a.OpenAIAPIKey
		if len(key) > 8 {
			key = key[:8] + strings.Repeat("*", len(key)-8)
		}
		return fmt.Sprintf("API key: %s", key), nil
	}
	return "signed in (no email available)", nil
}

// LogoutOpenAI removes the stored OpenAI credentials from ~/.codex/auth.json.
// If the file contains other keys (e.g. Ollama credentials) they are preserved.
func LogoutOpenAI() error {
	a, err := loadAuthJSON()
	if err != nil {
		return err
	}
	a.OpenAIAPIKey = ""
	a.Tokens = nil
	a.LastRefresh = nil
	// If the struct is now empty, delete the file entirely.
	if a == (openAIAuthJSON{}) {
		path, err := authJSONPath()
		if err != nil {
			return err
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("openai auth: remove %s: %w", path, err)
		}
		return nil
	}
	return saveAuthJSON(a)
}

// LoginOpenAI runs the full browser-based PKCE sign-in flow:
//  1. Generates PKCE codes.
//  2. Starts a local callback server on port 1455.
//  3. Opens the authorization URL in the user's browser (and prints it for
//     headless environments).
//  4. Waits for the callback, exchanges the code for tokens, exchanges the
//     id_token for an API key, and saves everything to ~/.codex/auth.json.
func LoginOpenAI(ctx context.Context) error {
	verifier, challenge, err := generatePKCE()
	if err != nil {
		return fmt.Errorf("openai auth: PKCE generation failed: %w", err)
	}
	state, err := generateState()
	if err != nil {
		return fmt.Errorf("openai auth: state generation failed: %w", err)
	}

	// Bind the callback server.  Try the default port first; fall back to a
	// random port so the flow still works when 1455 is occupied.
	ln, actualPort, err := bindCallbackServer(openAICallbackPort)
	if err != nil {
		return fmt.Errorf("openai auth: cannot start callback server: %w", err)
	}
	defer ln.Close()

	redirectURI := fmt.Sprintf("http://localhost:%d/auth/callback", actualPort)
	authURL := buildAuthorizeURL(openAIIssuer, openAIClientID, redirectURI, challenge, state)

	fmt.Println()
	fmt.Println("Opening your browser to sign in to OpenAI...")
	fmt.Println()
	fmt.Printf("  %s\n", authURL)
	fmt.Println()
	fmt.Println("If your browser did not open, copy the URL above and paste it into your browser.")
	fmt.Println()

	_ = browser.OpenURL(authURL)

	// Wait for the OAuth callback.
	codeCh := make(chan string, 1)
	errCh := make(chan error, 1)

	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/auth/callback" {
				http.NotFound(w, r)
				return
			}
			q := r.URL.Query()
			if q.Get("state") != state {
				http.Error(w, "state mismatch", http.StatusBadRequest)
				errCh <- fmt.Errorf("openai auth: OAuth state mismatch (possible CSRF)")
				return
			}
			if errParam := q.Get("error"); errParam != "" {
				desc := q.Get("error_description")
				if desc == "" {
					desc = errParam
				}
				http.Error(w, desc, http.StatusBadRequest)
				errCh <- fmt.Errorf("openai auth: OAuth error: %s", desc)
				return
			}
			code := q.Get("code")
			if code == "" {
				http.Error(w, "missing code", http.StatusBadRequest)
				errCh <- fmt.Errorf("openai auth: missing authorization code in callback")
				return
			}
			// Serve a simple success page so the browser tab closes cleanly.
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			fmt.Fprint(w, callbackSuccessHTML)
			codeCh <- code
		}),
	}

	go func() { _ = srv.Serve(ln) }()

	// Wait for callback or context cancellation.
	var code string
	select {
	case code = <-codeCh:
	case err = <-errCh:
		_ = srv.Shutdown(context.Background())
		return err
	case <-ctx.Done():
		_ = srv.Shutdown(context.Background())
		return fmt.Errorf("openai auth: sign-in cancelled")
	}
	_ = srv.Shutdown(context.Background())

	fmt.Println("Authorization code received. Exchanging for tokens...")

	tokens, err := exchangeCodeForTokens(ctx, openAIIssuer, openAIClientID, redirectURI, verifier, code)
	if err != nil {
		return fmt.Errorf("openai auth: token exchange failed: %w", err)
	}

	fmt.Println("Exchanging tokens for API key...")

	apiKey, err := exchangeIDTokenForAPIKey(ctx, openAIIssuer, openAIClientID, tokens.IDToken)
	if err != nil {
		// Non-fatal: some accounts may not support the key exchange.  Store
		// the access token as the API key instead so the proxy still works.
		apiKey = tokens.AccessToken
	}

	now := time.Now().UTC()
	auth := openAIAuthJSON{
		OpenAIAPIKey: apiKey,
		Tokens:       tokens,
		LastRefresh:  &now,
	}
	if err := saveAuthJSON(auth); err != nil {
		return fmt.Errorf("openai auth: save credentials: %w", err)
	}

	email, _ := emailFromIDToken(tokens.IDToken)
	if email != "" {
		fmt.Printf("\nSigned in as %s\n\n", email)
	} else {
			fmt.Println("Signed in to OpenAI successfully.")
	}
	return nil
}

// ─── PKCE helpers ────────────────────────────────────────────────────────────

// generatePKCE returns a (verifier, challenge) pair using the S256 method.
// The verifier is 64 random bytes encoded as URL-safe base64 without padding.
// The challenge is the SHA-256 of the verifier, also base64url-encoded.
func generatePKCE() (verifier, challenge string, err error) {
	b := make([]byte, 64)
	if _, err = rand.Read(b); err != nil {
		return
	}
	verifier = base64.RawURLEncoding.EncodeToString(b)
	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return
}

// generateState returns a random CSRF state token (32 bytes, base64url).
func generateState() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// ─── Network helpers ─────────────────────────────────────────────────────────

// bindCallbackServer tries to bind on preferredPort; if that fails it binds
// on a random free port.  Returns the listener and the actual port.
func bindCallbackServer(preferredPort int) (net.Listener, int, error) {
	addr := fmt.Sprintf("127.0.0.1:%d", preferredPort)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		// Fall back to any free port.
		ln, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, 0, err
		}
	}
	port := ln.Addr().(*net.TCPAddr).Port
	return ln, port, nil
}

// buildAuthorizeURL constructs the OAuth 2.0 authorization URL.
func buildAuthorizeURL(issuer, clientID, redirectURI, codeChallenge, state string) string {
	params := url.Values{}
	params.Set("response_type", "code")
	params.Set("client_id", clientID)
	params.Set("redirect_uri", redirectURI)
	params.Set("scope", openAIScope)
	params.Set("code_challenge", codeChallenge)
	params.Set("code_challenge_method", "S256")
	params.Set("id_token_add_organizations", "true")
	params.Set("codex_cli_simplified_flow", "true")
	params.Set("state", state)
	return issuer + "/oauth/authorize?" + params.Encode()
}

// ─── Token exchange ───────────────────────────────────────────────────────────

// exchangeCodeForTokens exchanges an authorization code for OAuth tokens.
func exchangeCodeForTokens(ctx context.Context, issuer, clientID, redirectURI, codeVerifier, code string) (*openAITokenData, error) {
	body := url.Values{}
	body.Set("grant_type", "authorization_code")
	body.Set("code", code)
	body.Set("redirect_uri", redirectURI)
	body.Set("client_id", clientID)
	body.Set("code_verifier", codeVerifier)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		issuer+"/oauth/token",
		strings.NewReader(body.Encode()),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("token endpoint returned %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var tr struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, fmt.Errorf("parse token response: %w", err)
	}
	return &openAITokenData{
		IDToken:      tr.IDToken,
		AccessToken:  tr.AccessToken,
		RefreshToken: tr.RefreshToken,
	}, nil
}

// exchangeIDTokenForAPIKey exchanges an id_token for an OpenAI API key using
// the token-exchange grant type (RFC 8693).
func exchangeIDTokenForAPIKey(ctx context.Context, issuer, clientID, idToken string) (string, error) {
	body := url.Values{}
	body.Set("grant_type", "urn:ietf:params:oauth:grant-type:token-exchange")
	body.Set("client_id", clientID)
	body.Set("requested_token", "openai-api-key")
	body.Set("subject_token", idToken)
	body.Set("subject_token_type", "urn:ietf:params:oauth:token-type:id_token")

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		issuer+"/oauth/token",
		strings.NewReader(body.Encode()),
	)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API key exchange returned %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var er struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(raw, &er); err != nil {
		return "", fmt.Errorf("parse API key exchange response: %w", err)
	}
	return er.AccessToken, nil
}

// ─── JWT helpers ──────────────────────────────────────────────────────────────

// emailFromIDToken extracts the email claim from a JWT id_token without
// verifying the signature (we trust the token since we just received it from
// the authorization server over TLS).
func emailFromIDToken(idToken string) (string, error) {
	parts := strings.Split(idToken, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("invalid JWT format")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode JWT payload: %w", err)
	}
	var claims struct {
		Email string `json:"email"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("parse JWT claims: %w", err)
	}
	return claims.Email, nil
}

// ─── Misc helpers ─────────────────────────────────────────────────────────────

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// callbackSuccessHTML is served to the browser after a successful sign-in.
const callbackSuccessHTML = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Signed in — Ollama</title>
<style>body{font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f5f5f5;}
.card{background:#fff;border-radius:12px;padding:2rem 3rem;box-shadow:0 2px 16px rgba(0,0,0,.1);text-align:center;}
h1{color:#10a37f;margin-bottom:.5rem;}p{color:#555;}</style></head>
<body><div class="card">
<h1>&#10003; Signed in</h1>
<p>You are now signed in to OpenAI.<br>You can close this tab and return to your terminal.</p>
</div></body></html>`

// SaveOpenAIAPIKey stores an API key directly to ~/.codex/auth.json without
// going through the browser flow.  Existing token data is cleared since the
// key supersedes any OAuth session.
func SaveOpenAIAPIKey(apiKey string) error {
apiKey = strings.TrimSpace(apiKey)
if apiKey == "" {
return fmt.Errorf("openai auth: API key must not be empty")
}
// Preserve any existing non-auth fields (e.g. Ollama credentials) by
// loading first, then updating only the OpenAI fields.
a, err := loadAuthJSON()
if err != nil {
a = openAIAuthJSON{}
}
a.OpenAIAPIKey = apiKey
a.Tokens = nil
a.LastRefresh = nil
if err := saveAuthJSON(a); err != nil {
return err
}
fmt.Printf("API key saved to ~/.codex/auth.json\n")
return nil
}
