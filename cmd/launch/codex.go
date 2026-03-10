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

	cmd := exec.Command("codex", c.args(model, args)...)
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
