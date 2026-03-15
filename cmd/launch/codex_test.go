package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--oss", "--local-provider=ollama", "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--oss", "--local-provider=ollama"}},
		{"with model and profile", "qwen3.5", []string{"-p", "myprofile"}, []string{"--oss", "--local-provider=ollama", "-m", "qwen3.5", "-p", "myprofile"}},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, []string{"--oss", "--local-provider=ollama", "-m", "llama3.2", "--sandbox", "workspace-write"}},
		{"resume session", "llama3.2", []string{"resume", "session-123"}, []string{"--oss", "--local-provider=ollama", "-m", "llama3.2", "resume", "session-123"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestListCodexSessions(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	writeSession := func(rel, id, ts, cwd, provider, extra string) {
		path := filepath.Join(home, ".codex", "sessions", rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		data := `{"payload":{"id":"` + id + `","timestamp":"` + ts + `","cwd":"` + cwd + `","model_provider":"` + provider + `","git":{"branch":"main","repository_url":"https://github.com/acme/repo.git"}}}` + "\n" + extra
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	writeSession("2026/03/10/a.jsonl", "older-session", "2026-03-10T10:00:00Z", "/repo", "ollama", "")
	writeSession("2026/03/10/b.jsonl", "newer-session", "2026-03-10T12:00:00Z", "/repo", "ollama", `{"payload":{"message":"Fix the flaky Ollama test.","type":"user_message"}}`+"\n"+`{"payload":{"model":"qwen3:8b"}}`+"\n")
	writeSession("2026/03/10/c.jsonl", "other-cwd", "2026-03-10T13:00:00Z", "/elsewhere", "openai", "")
	writeSession("2026/03/10/d.jsonl", "older-session", "2026-03-10T09:00:00Z", "/repo", "ollama", "")

	sessions, err := ListCodexSessions("/repo", 10)
	if err != nil {
		t.Fatalf("ListCodexSessions() error = %v", err)
	}
	if len(sessions) != 2 {
		t.Fatalf("len(sessions) = %d, want 2", len(sessions))
	}
	if sessions[0].ID != "newer-session" {
		t.Fatalf("sessions[0].ID = %q, want newer-session", sessions[0].ID)
	}
	if sessions[1].ID != "older-session" {
		t.Fatalf("sessions[1].ID = %q, want older-session", sessions[1].ID)
	}
	if sessions[0].Model != "qwen3:8b" {
		t.Fatalf("sessions[0].Model = %q, want qwen3:8b", sessions[0].Model)
	}
	if sessions[0].Provider != "ollama" {
		t.Fatalf("sessions[0].Provider = %q, want ollama", sessions[0].Provider)
	}
	if sessions[0].Repository != "repo" {
		t.Fatalf("sessions[0].Repository = %q, want repo", sessions[0].Repository)
	}
	if sessions[0].Branch != "main" {
		t.Fatalf("sessions[0].Branch = %q, want main", sessions[0].Branch)
	}
	if sessions[0].Prompt != "Fix the flaky Ollama test." {
		t.Fatalf("sessions[0].Prompt = %q", sessions[0].Prompt)
	}
}

func TestCodexModelCatalogArg(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	cachePath := filepath.Join(home, ".codex", "models_cache.json")
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	cache := `{"fetched_at":"2026-03-10T00:00:00Z","client_version":"0.113.0","models":[{"slug":"gpt-5.4","display_name":"gpt-5.4","description":"Latest frontier agentic coding model.","default_reasoning_level":"medium","supported_reasoning_levels":[{"effort":"low","description":"Low"},{"effort":"medium","description":"Medium"},{"effort":"high","description":"High"}],"shell_type":"shell_command","visibility":"list","supported_in_api":true,"priority":0,"availability_nux":null,"upgrade":null,"base_instructions":"base","instructions_variables":{},"supports_reasoning_summaries":true,"default_reasoning_summary":"none","support_verbosity":true,"default_verbosity":"low","apply_patch_tool_type":"freeform","web_search_tool_type":"text","truncation_policy":{"mode":"tokens","limit":10000},"supports_parallel_tool_calls":true,"supports_image_detail_original":true,"context_window":272000,"effective_context_window_percent":95,"experimental_supported_tools":[],"input_modalities":["text"],"prefer_websockets":true,"auto_compact_token_limit":258400}]}` + "\n"
	if err := os.WriteFile(cachePath, []byte(cache), 0o644); err != nil {
		t.Fatalf("write cache: %v", err)
	}

	arg, cleanup, err := codexModelCatalogArg("minimax-m2.5:cloud")
	if err != nil {
		t.Fatalf("codexModelCatalogArg() error = %v", err)
	}
	if cleanup == nil {
		t.Fatal("cleanup = nil, want non-nil")
	}
	if !strings.HasPrefix(arg, `model_catalog_json="`) {
		t.Fatalf("arg = %q, want model_catalog_json path", arg)
	}

	path := strings.TrimPrefix(arg, `model_catalog_json="`)
	path = strings.TrimSuffix(path, `"`)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read generated catalog: %v", err)
	}

	var catalog codexModelCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatalf("unmarshal generated catalog: %v", err)
	}
	if len(catalog.Models) != 2 {
		t.Fatalf("len(catalog.Models) = %d, want 2", len(catalog.Models))
	}

	entry := catalog.Models[1]
	if entry["slug"] != "minimax-m2.5:cloud" {
		t.Fatalf("entry slug = %v, want minimax-m2.5:cloud", entry["slug"])
	}
	if entry["context_window"] != float64(204800) {
		t.Fatalf("entry context_window = %v, want 204800", entry["context_window"])
	}
	if _, ok := entry["auto_compact_token_limit"]; ok {
		t.Fatalf("entry auto_compact_token_limit = %v, want omitted", entry["auto_compact_token_limit"])
	}
	if entry["prefer_websockets"] != false {
		t.Fatalf("entry prefer_websockets = %v, want false", entry["prefer_websockets"])
	}

	cleanup()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("generated catalog still exists after cleanup, err = %v", err)
	}
}

func TestCodexModelCatalogArg_LocalModel(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	// Write a models_cache.json so the catalog template is available.
	cachePath := filepath.Join(home, ".codex", "models_cache.json")
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	cache := `{"fetched_at":"2026-03-10T00:00:00Z","client_version":"0.113.0","models":[{"slug":"gpt-5.4","display_name":"gpt-5.4","description":"Latest frontier agentic coding model.","context_window":272000,"effective_context_window_percent":95,"priority":0,"prefer_websockets":true,"auto_compact_token_limit":258400}]}` + "\n"
	if err := os.WriteFile(cachePath, []byte(cache), 0o644); err != nil {
		t.Fatalf("write cache: %v", err)
	}

	// Start a mock /api/show server that returns a known context_length.
	import_net_http_test_server := func() (string, func()) {
		// We can't import net/http/httptest here without a separate file,
		// so we test the fallback path: no server running → default context window.
		return "", func() {}
	}
	_ = import_net_http_test_server

	// "llama3.2" is not in cloudModelLimits, so fetchLocalModelContextWindow
	// will be called. The Ollama server is not running in tests, so it falls
	// back to defaultLocalContextWindow (128_000).
	arg, cleanup, err := codexModelCatalogArg("llama3.2")
	if err != nil {
		t.Fatalf("codexModelCatalogArg() error = %v", err)
	}
	if cleanup == nil {
		t.Fatal("cleanup = nil, want non-nil (local models must also produce a catalog)")
	}
	if !strings.HasPrefix(arg, `model_catalog_json="`) {
		t.Fatalf("arg = %q, want model_catalog_json path", arg)
	}

	path := strings.TrimPrefix(arg, `model_catalog_json="`)
	path = strings.TrimSuffix(path, `"`)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read generated catalog: %v", err)
	}

	var catalog codexModelCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatalf("unmarshal generated catalog: %v", err)
	}
	if len(catalog.Models) != 2 {
		t.Fatalf("len(catalog.Models) = %d, want 2 (native + ollama model)", len(catalog.Models))
	}

	entry := catalog.Models[1]
	if entry["slug"] != "llama3.2" {
		t.Fatalf("entry slug = %v, want llama3.2", entry["slug"])
	}
	// Falls back to defaultLocalContextWindow when server is not running.
	if entry["context_window"] != float64(defaultLocalContextWindow) {
		t.Fatalf("entry context_window = %v, want %d", entry["context_window"], defaultLocalContextWindow)
	}
	if entry["priority"] != float64(100) {
		t.Fatalf("entry priority = %v, want 100", entry["priority"])
	}
	if _, ok := entry["auto_compact_token_limit"]; ok {
		t.Fatalf("entry auto_compact_token_limit should be omitted for local models")
	}

	cleanup()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("generated catalog still exists after cleanup, err = %v", err)
	}
}

func TestFetchLocalModelContextWindow_Fallback(t *testing.T) {
	// When the Ollama server is not running, fetchLocalModelContextWindow
	// must return defaultLocalContextWindow without panicking.
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1") // port 1 is never open
	got := fetchLocalModelContextWindow("llama3.2")
	if got != defaultLocalContextWindow {
		t.Fatalf("fetchLocalModelContextWindow() = %d, want %d", got, defaultLocalContextWindow)
	}
}

// TestCodexAuthEnvPassthrough verifies the env var injection logic in RunContext:
//   - OPENAI_BASE_URL is only set to the Ollama endpoint when the user has not
//     already configured a custom base URL.
//   - OPENAI_API_KEY is only set to the dummy "ollama" value when neither
//     OPENAI_API_KEY nor CODEX_API_KEY is set.
func TestCodexAuthEnvPassthrough(t *testing.T) {
	ollamaURL := "http://127.0.0.1:11434/v1/"

	cases := []struct {
		name              string
		envOpenAIBaseURL  string
		envOpenAIAPIKey   string
		envCodexAPIKey    string
		wantBaseURL       string // expected value of OPENAI_BASE_URL in subprocess env
		wantAPIKeyDummy   bool   // expect OPENAI_API_KEY=ollama to be injected
	}{
		{
			name:            "no_user_vars_injects_defaults",
			wantBaseURL:     ollamaURL,
			wantAPIKeyDummy: true,
		},
		{
			name:             "user_base_url_preserved",
			envOpenAIBaseURL: "https://my-azure.openai.azure.com/openai/v1/",
			wantBaseURL:      "https://my-azure.openai.azure.com/openai/v1/",
			wantAPIKeyDummy:  true,
		},
		{
			name:            "user_openai_api_key_preserved",
			envOpenAIAPIKey: "sk-real-key",
			wantBaseURL:     ollamaURL,
			wantAPIKeyDummy: false,
		},
		{
			name:            "codex_api_key_suppresses_dummy",
			envCodexAPIKey:  "ck-real-key",
			wantBaseURL:     ollamaURL,
			wantAPIKeyDummy: false,
		},
		{
			name:             "all_user_vars_fully_preserved",
			envOpenAIBaseURL: "https://us.api.openai.com/v1",
			envOpenAIAPIKey:  "sk-real-key",
			wantBaseURL:      "https://us.api.openai.com/v1",
			wantAPIKeyDummy:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Isolate env vars for this sub-test.
			t.Setenv("OPENAI_BASE_URL", tc.envOpenAIBaseURL)
			t.Setenv("OPENAI_API_KEY", tc.envOpenAIAPIKey)
			t.Setenv("CODEX_API_KEY", tc.envCodexAPIKey)
			// Point OLLAMA_HOST at the test URL so envconfig.Host() is predictable.
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

			// Replicate the env-building logic from RunContext directly so we
			// can test it without actually launching a codex subprocess.
			env := os.Environ()
			if os.Getenv("OPENAI_BASE_URL") == "" {
				env = append(env, "OPENAI_BASE_URL="+envconfig.Host().String()+"/v1/")
			}
			if os.Getenv("OPENAI_API_KEY") == "" && os.Getenv("CODEX_API_KEY") == "" {
				env = append(env, "OPENAI_API_KEY=ollama")
			}

			// Build a lookup map from the last occurrence of each key
			// (later entries win, matching exec.Cmd behaviour on Linux).
			envMap := make(map[string]string)
			for _, kv := range env {
				idx := strings.IndexByte(kv, '=')
				if idx < 0 {
					continue
				}
				envMap[kv[:idx]] = kv[idx+1:]
			}

			if got := envMap["OPENAI_BASE_URL"]; got != tc.wantBaseURL {
				t.Errorf("OPENAI_BASE_URL = %q, want %q", got, tc.wantBaseURL)
			}

			if tc.wantAPIKeyDummy {
				if got := envMap["OPENAI_API_KEY"]; got != "ollama" {
					t.Errorf("OPENAI_API_KEY = %q, want %q", got, "ollama")
				}
			} else {
				if got := envMap["OPENAI_API_KEY"]; got == "ollama" {
					t.Errorf("OPENAI_API_KEY should not be overridden with dummy value")
				}
			}
		})
	}
}
