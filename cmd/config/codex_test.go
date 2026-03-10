package config

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--oss", "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--oss"}},
		{"with model and profile", "qwen3-coder", []string{"-p", "myprofile"}, []string{"--oss", "-m", "qwen3-coder", "-p", "myprofile"}},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, []string{"--oss", "-m", "llama3.2", "--sandbox", "workspace-write"}},
		{"resume session", "llama3.2", []string{"resume", "session-123"}, []string{"--oss", "-m", "llama3.2", "resume", "session-123"}},
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

	writeSession := func(rel, id, ts, cwd, extra string) {
		path := filepath.Join(home, ".codex", "sessions", rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		data := `{"payload":{"id":"` + id + `","timestamp":"` + ts + `","cwd":"` + cwd + `","git":{"branch":"main","repository_url":"https://github.com/acme/repo.git"}}}` + "\n" + extra
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	writeSession("2026/03/10/a.jsonl", "older-session", "2026-03-10T10:00:00Z", "/repo", "")
	writeSession("2026/03/10/b.jsonl", "newer-session", "2026-03-10T12:00:00Z", "/repo", `{"payload":{"message":"Fix the flaky Ollama test.","type":"user_message"}}`+"\n"+`{"payload":{"model":"qwen3:8b"}}`+"\n")
	writeSession("2026/03/10/c.jsonl", "other-cwd", "2026-03-10T13:00:00Z", "/elsewhere", "")

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
