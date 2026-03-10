package launch

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
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
	args := []string{"--oss"}
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Codex) Run(model string, args []string) error {
	if err := checkCodexVersion(); err != nil {
		return err
	}

	cmdArgs := c.args(model, args)
	catalogArg, cleanup, err := codexModelCatalogArg(model)
	if err == nil && catalogArg != "" {
		cmdArgs = append(cmdArgs[:len(cmdArgs)-len(args)], append([]string{"-c", catalogArg}, cmdArgs[len(cmdArgs)-len(args):]...)...)
	}
	if cleanup != nil {
		defer cleanup()
	}

	cmd := exec.Command("codex", cmdArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"OPENAI_BASE_URL="+envconfig.Host().String()+"/v1/",
		"OPENAI_API_KEY=ollama",
	)
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

	line, err := readFirstLine(bytes.NewReader(data))
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
	seen := make(map[string]struct{}, len(sessions))
	deduped := make([]CodexSession, 0, len(sessions))
	for _, session := range sessions {
		if _, ok := seen[session.ID]; ok {
			continue
		}
		seen[session.ID] = struct{}{}
		deduped = append(deduped, session)
	}
	return deduped
}

func readFirstLine(r io.Reader) ([]byte, error) {
	var buf bytes.Buffer
	tmp := make([]byte, 4096)
	for {
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
	for _, line := range lines {
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

func codexModelCatalogArg(model string) (string, func(), error) {
	limit, ok := lookupCloudModelLimit(model)
	if !ok {
		return "", nil, nil
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
	entry["context_window"] = limit.Context
	entry["effective_context_window_percent"] = 95
	entry["auto_compact_token_limit"] = limit.Context * 95 / 100
	entry["prefer_websockets"] = false

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

func codexModelDescription(model string) string {
	for _, item := range recommendedModels {
		if item.Name == model {
			return item.Description
		}
	}
	return model
}
