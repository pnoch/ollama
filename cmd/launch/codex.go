package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"golang.org/x/mod/semver"
)

// Codex implements Runner for Codex integration
type Codex struct{}

func (c *Codex) String() string { return "Codex" }

type CodexSession struct {
	ID          string
	Provider    string
	Model       string
	Branch      string
	Repository  string
	Prompt      string
	Title       string
	Description string
	CWD         string
	Timestamp   time.Time
}

func (c *Codex) args(model string, extra []string) []string {
	args := []string{"--oss", "--local-provider=ollama"}
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Codex) Run(model string, args []string) error {
	return c.RunContext(context.Background(), model, args)
}

func (c *Codex) RunContext(ctx context.Context, model string, args []string) error {
	if err := checkCodexVersion(); err != nil {
		return err
	}

	cmdArgs := c.args(model, args)

	// Add model catalog if available (don't fail if it doesn't work)
	catalogArg, cleanup, err := codexModelCatalogArg(model)
	if err == nil && catalogArg != "" {
		cmdArgs = append(cmdArgs, "-c", catalogArg)
	}
	if cleanup != nil {
		defer cleanup()
	}

	cmd := exec.Command("codex", cmdArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Build the subprocess environment starting from the current process
	// environment so that all user-configured auth vars are inherited:
	//   OPENAI_API_KEY / CODEX_API_KEY  – standard and Codex-specific keys
	//   OPENAI_ORG_ID / OPENAI_PROJECT_ID – organisation / project scoping
	//   AZURE_OPENAI_API_KEY            – Azure OpenAI key
	//   Any custom env_key from ~/.codex/config.toml (MISTRAL_API_KEY, etc.)
	//
	// OPENAI_BASE_URL is only set to the local Ollama endpoint when the user
	// has not already configured a custom base URL (Azure endpoint, proxy,
	// data-residency URL, etc.).  Preserving the user's value lets all
	// provider configs in ~/.codex/config.toml work without interference.
	env := os.Environ()
	if os.Getenv("OPENAI_BASE_URL") == "" {
		env = append(env, "OPENAI_BASE_URL="+envconfig.Host().String()+"/v1/")
	}
	// Only inject the dummy API key when neither OPENAI_API_KEY nor the
	// Codex-specific CODEX_API_KEY is set, so that real keys are never
	// shadowed by the placeholder value.
	if os.Getenv("OPENAI_API_KEY") == "" && os.Getenv("CODEX_API_KEY") == "" {
		env = append(env, "OPENAI_API_KEY=ollama")
	}
	cmd.Env = env
	return cmd.Run()
}

func checkCodexVersion() error {
	if _, err := exec.LookPath("codex"); err != nil {
		return fmt.Errorf("codex is not installed, install with: npm install -g @openai/codex")
	}

	out, err := exec.Command("codex", "--version").Output()
	if err != nil {
		return fmt.Errorf("failed to get codex version: %w", err)
	}

	// Parse output like "codex-cli 0.87.0"
	fields := strings.Fields(strings.TrimSpace(string(out)))
	if len(fields) < 2 {
		return fmt.Errorf("unexpected codex version output: %s", string(out))
	}

	version := "v" + fields[len(fields)-1]
	minVersion := "v0.81.0"

	if semver.Compare(version, minVersion) < 0 {
		return fmt.Errorf("codex version %s is too old, minimum required is %s, update with: npm update -g @openai/codex", fields[len(fields)-1], "0.81.0")
	}

	return nil
}

func ListCodexSessions(cwd string, limit int) ([]CodexSession, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	root := filepath.Join(home, ".codex", "sessions")
	var sessions []CodexSession

	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			slog.Warn("error accessing path during session scan", "path", path, "error", err)
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}

		session, ok := readCodexSessionMeta(path)
		if !ok {
			return nil
		}
		if cwd != "" && session.CWD != cwd {
			return nil
		}
		sessions = append(sessions, session)
		return nil
	})
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	slices.SortFunc(sessions, func(a, b CodexSession) int {
		switch {
		case a.Timestamp.After(b.Timestamp):
			return -1
		case a.Timestamp.Before(b.Timestamp):
			return 1
		default:
			return strings.Compare(a.ID, b.ID)
		}
	})
	sessions = dedupeCodexSessions(sessions)

	if limit > 0 && len(sessions) > limit {
		sessions = sessions[:limit]
	}

	return sessions, nil
}

type codexSessionMeta struct {
	Payload struct {
		ID        string `json:"id"`
		Timestamp string `json:"timestamp"`
		CWD       string `json:"cwd"`
		Provider  string `json:"model_provider"`
		Git       struct {
			Branch        string `json:"branch"`
			RepositoryURL string `json:"repository_url"`
		} `json:"git"`
	} `json:"payload"`
}

func readCodexSessionMeta(path string) (CodexSession, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return CodexSession{}, false
	}

	line, err := readFirstLine(bytes.NewReader(data), 1024*1024)
	if err != nil || len(line) == 0 {
		return CodexSession{}, false
	}

	var meta codexSessionMeta
	if err := json.Unmarshal(line, &meta); err != nil {
		return CodexSession{}, false
	}
	if meta.Payload.ID == "" || meta.Payload.CWD == "" || meta.Payload.Timestamp == "" {
		return CodexSession{}, false
	}

	timestamp, err := time.Parse(time.RFC3339Nano, meta.Payload.Timestamp)
	if err != nil {
		return CodexSession{}, false
	}

	cwdLabel := meta.Payload.CWD
	if base := filepath.Base(meta.Payload.CWD); base != "" && base != "." && base != string(filepath.Separator) {
		cwdLabel = base
	}
	shortID := meta.Payload.ID
	if len(shortID) > 8 {
		shortID = shortID[:8]
	}
	repoLabel := repoName(meta.Payload.Git.RepositoryURL, meta.Payload.CWD)
	model := detectCodexSessionModel(data)
	prompt := detectCodexSessionPrompt(data)

	var detailParts []string
	detailParts = append(detailParts, shortID)
	if model != "" {
		detailParts = append(detailParts, model)
	}
	if repoLabel != "" {
		detailParts = append(detailParts, repoLabel)
	}
	if meta.Payload.Git.Branch != "" {
		detailParts = append(detailParts, meta.Payload.Git.Branch)
	}
	if prompt != "" {
		detailParts = append(detailParts, prompt)
	}

	return CodexSession{
		Provider:    meta.Payload.Provider,
		Model:       model,
		Branch:      meta.Payload.Git.Branch,
		Repository:  repoLabel,
		Prompt:      prompt,
		ID:          meta.Payload.ID,
		Title:       fmt.Sprintf("%s  %s", timestamp.Local().Format("Jan 2 15:04"), cwdLabel),
		Description: strings.Join(detailParts, "  "),
		CWD:         meta.Payload.CWD,
		Timestamp:   timestamp,
	}, true
}

func dedupeCodexSessions(sessions []CodexSession) []CodexSession {
	if len(sessions) == 0 {
		return sessions
	}
	// Use a nil out-slice and lazy append so no allocation occurs when there
	// are no duplicates (the common case). The seen map uses a conservative
	// initial capacity since duplicates are rare.
	seen := make(map[string]struct{}, len(sessions)/2+1)
	var out []CodexSession
	for _, session := range sessions {
		if _, dup := seen[session.ID]; dup {
			continue
		}
		seen[session.ID] = struct{}{}
		out = append(out, session)
	}
	if out == nil {
		return sessions
	}
	return out
}

func readFirstLine(r io.Reader, maxLen int) ([]byte, error) {
	var buf bytes.Buffer
	tmp := make([]byte, 4096)
	for buf.Len() < maxLen {
		n, err := r.Read(tmp)
		if n > 0 {
			if idx := bytes.IndexByte(tmp[:n], '\n'); idx >= 0 {
				buf.Write(tmp[:idx])
				return buf.Bytes(), nil
			}
			buf.Write(tmp[:n])
		}
		if err != nil {
			if err == io.EOF {
				return buf.Bytes(), nil
			}
			return nil, err
		}
	}
	return buf.Bytes(), fmt.Errorf("first line exceeds maximum length of %d bytes", maxLen)
}

func repoName(repositoryURL, cwd string) string {
	repositoryURL = strings.TrimSpace(repositoryURL)
	if repositoryURL != "" {
		base := strings.TrimSuffix(filepath.Base(repositoryURL), ".git")
		if base != "" && base != "." && base != "/" {
			return base
		}
	}
	base := filepath.Base(cwd)
	if base == "." || base == "/" {
		return ""
	}
	return base
}

func detectCodexSessionModel(data []byte) string {
	lines := bytes.Split(data, []byte("\n"))
	maxLines := 10
	if len(lines) < maxLines {
		maxLines = len(lines)
	}
	for i := 0; i < maxLines; i++ {
		line := lines[i]
		if len(line) == 0 {
			continue
		}
		if !bytes.Contains(line, []byte(`"payload"`)) || !bytes.Contains(line, []byte(`"model"`)) {
			continue
		}

		var entry struct {
			Payload struct {
				Model string `json:"model"`
			} `json:"payload"`
		}
		if json.Unmarshal(line, &entry) != nil {
			continue
		}
		if entry.Payload.Model != "" {
			return entry.Payload.Model
		}
	}
	return ""
}

func detectCodexSessionPrompt(data []byte) string {
	lines := bytes.Split(data, []byte("\n"))
	// Only scan the first 50 lines: the initial user prompt always appears
	// near the top of the JSONL file, so scanning the whole file is wasteful
	// for long sessions.
	const maxPromptScanLines = 50
	if len(lines) > maxPromptScanLines {
		lines = lines[:maxPromptScanLines]
	}
	for _, line := range lines {
		if len(line) == 0 || !bytes.Contains(line, []byte(`"user_message"`)) {
			continue
		}

		var entry struct {
			Payload struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"payload"`
		}
		if json.Unmarshal(line, &entry) != nil {
			continue
		}
		if entry.Payload.Type == "user_message" && entry.Payload.Message != "" {
			return compactPrompt(entry.Payload.Message)
		}
	}
	return ""
}

func compactPrompt(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 72 {
		return s[:69] + "..."
	}
	return s
}

type codexModelCatalog struct {
	FetchedAt     string           `json:"fetched_at,omitempty"`
	ETag          string           `json:"etag,omitempty"`
	ClientVersion string           `json:"client_version,omitempty"`
	Models        []map[string]any `json:"models"`
}

// defaultLocalContextWindow is used when the Ollama /api/show endpoint is
// unavailable or returns no context_length (e.g. model not yet pulled).
const defaultLocalContextWindow = 128_000

// fetchLocalModelContextWindow queries the running Ollama server for the
// context window of a local model. It returns defaultLocalContextWindow on
// any error so the caller never needs to handle the failure case.
func fetchLocalModelContextWindow(model string) int {
	type showReq struct {
		Model string `json:"model"`
	}
	type showResp struct {
		ModelInfo map[string]any `json:"model_info"`
	}
	body, _ := json.Marshal(showReq{Model: model})
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Post(
		envconfig.Host().String()+"/api/show",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil || resp.StatusCode != http.StatusOK {
		return defaultLocalContextWindow
	}
	defer resp.Body.Close()
	var show showResp
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return defaultLocalContextWindow
	}
	// ModelInfo keys are "<arch>.context_length" (e.g. "llama.context_length").
	// The /v1/show endpoint also exposes a bare "context_length" key.
	for k, v := range show.ModelInfo {
		if k == "context_length" || strings.HasSuffix(k, ".context_length") {
			if n, ok := v.(float64); ok && n > 0 {
				return int(n)
			}
		}
	}
	return defaultLocalContextWindow
}

func codexModelCatalogArg(model string) (string, func(), error) {
	// Resolve context window: use the hardcoded cloud limit when available,
	// otherwise query the local Ollama server (falls back to a safe default).
	var contextWindow int
	if limit, ok := lookupCloudModelLimit(model); ok {
		contextWindow = limit.Context
	} else {
		contextWindow = fetchLocalModelContextWindow(model)
	}

	catalog, err := readCodexModelCatalogTemplate()
	if err != nil {
		return "", nil, err
	}
	if len(catalog.Models) == 0 {
		return "", nil, nil
	}

	entry, err := cloneCodexCatalogEntry(catalog.Models[0])
	if err != nil {
		return "", nil, err
	}

	entry["slug"] = model
	entry["display_name"] = model
	entry["description"] = codexModelDescription(model)
	entry["priority"] = 100
	entry["availability_nux"] = nil
	entry["upgrade"] = nil
	entry["context_window"] = contextWindow
	entry["effective_context_window_percent"] = 95
	entry["prefer_websockets"] = false
	delete(entry, "auto_compact_token_limit")

	catalog.Models = append(catalog.Models, entry)

	f, err := os.CreateTemp("", "ollama-codex-model-catalog-*.json")
	if err != nil {
		return "", nil, err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(catalog); err != nil {
		_ = os.Remove(f.Name())
		return "", nil, err
	}

	return fmt.Sprintf(`model_catalog_json=%q`, f.Name()), func() {
		_ = os.Remove(f.Name())
	}, nil
}

func readCodexModelCatalogTemplate() (codexModelCatalog, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return codexModelCatalog{}, err
	}

	data, err := os.ReadFile(filepath.Join(home, ".codex", "models_cache.json"))
	if err != nil {
		return codexModelCatalog{}, err
	}

	var catalog codexModelCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return codexModelCatalog{}, err
	}
	return catalog, nil
}

func cloneCodexCatalogEntry(entry map[string]any) (map[string]any, error) {
	data, err := json.Marshal(entry)
	if err != nil {
		return nil, err
	}

	var cloned map[string]any
	if err := json.Unmarshal(data, &cloned); err != nil {
		return nil, err
	}
	return cloned, nil
}

// codexModelDescriptionMap is built once from recommendedModels for O(1) lookup.
var codexModelDescriptionMap = func() map[string]string {
	m := make(map[string]string, len(recommendedModels))
	for _, item := range recommendedModels {
		if item.Description != "" {
			m[item.Name] = item.Description
		}
	}
	return m
}()

func codexModelDescription(model string) string {
	if desc, ok := codexModelDescriptionMap[model]; ok {
		return desc
	}
	return model
}
