package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestCodexModelModalRightOpensSessionPicker(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	cwd := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	defer func() {
		_ = os.Chdir(oldwd)
	}()
	if err := os.Chdir(cwd); err != nil {
		t.Fatalf("Chdir: %v", err)
	}

	sessionPath := filepath.Join(home, ".codex", "sessions", "2026", "03", "10", "session.jsonl")
	if err := os.MkdirAll(filepath.Dir(sessionPath), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	data := `{"payload":{"id":"session-123","timestamp":"2026-03-10T12:00:00Z","cwd":"` + cwd + `","model_provider":"ollama","git":{"branch":"main","repository_url":"https://github.com/acme/ollama.git"}}}` + "\n" +
		`{"payload":{"model":"qwen3:8b"}}` + "\n" +
		`{"payload":{"type":"user_message","message":"Resume the previous fix and finish the tests."}}` + "\n"
	if err := os.WriteFile(sessionPath, []byte(data), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	otherSessionPath := filepath.Join(home, ".codex", "sessions", "2026", "03", "09", "session-other.jsonl")
	if err := os.MkdirAll(filepath.Dir(otherSessionPath), 0o755); err != nil {
		t.Fatalf("MkdirAll other: %v", err)
	}
	otherCWD := filepath.Join(home, "other-project")
	otherData := `{"payload":{"id":"session-999","timestamp":"2026-03-10T11:00:00Z","cwd":"` + otherCWD + `","model_provider":"openai","git":{"branch":"feature","repository_url":"https://github.com/acme/ollama.git"}}}` + "\n" +
		`{"payload":{"model":"glm-5:cloud"}}` + "\n"
	if err := os.WriteFile(otherSessionPath, []byte(otherData), 0o644); err != nil {
		t.Fatalf("WriteFile other: %v", err)
	}

	m := model{
		items:        []menuItem{{integration: "codex"}},
		showingModal: true,
		modalSelector: selectorModel{
			items: []SelectItem{{Name: "qwen3:8b", Value: "qwen3:8b"}},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRight})
	fm := updated.(model)

	if !fm.showingSessionModal {
		t.Fatal("expected showingSessionModal=true")
	}
	if fm.showingModal {
		t.Fatal("expected showingModal=false")
	}
	if fm.modalSelector.selected != "qwen3:8b" {
		t.Fatalf("selected model = %q, want qwen3:8b", fm.modalSelector.selected)
	}
	if len(fm.sessionSelector.items) != 2 {
		t.Fatalf("len(sessionSelector.items) = %d, want 2", len(fm.sessionSelector.items))
	}
	if fm.sessionSelector.items[0].Value != "session-123" {
		t.Fatalf("session value = %q, want session-123", fm.sessionSelector.items[0].Value)
	}
	if !fm.sessionSelector.items[0].Recommended {
		t.Fatal("expected matching-model session to be pinned as recommended")
	}
	if !strings.Contains(fm.sessionSelector.items[0].Description, "qwen3:8b") {
		t.Fatalf("session description = %q, want model metadata", fm.sessionSelector.items[0].Description)
	}
	if fm.sessionSelector.items[1].Value != "session-999" {
		t.Fatalf("session value = %q, want session-999", fm.sessionSelector.items[1].Value)
	}
	if fm.sessionSelector.items[1].Recommended {
		t.Fatal("expected non-matching session to stay in other sessions")
	}
	if fm.sessionSelector.recommendedHeader != "Matching Model" {
		t.Fatalf("recommendedHeader = %q, want Matching Model", fm.sessionSelector.recommendedHeader)
	}
	if fm.sessionSelector.otherHeader != "Other Sessions" {
		t.Fatalf("otherHeader = %q, want Other Sessions", fm.sessionSelector.otherHeader)
	}
}

func TestResultFromModelForCodexResume(t *testing.T) {
	result := resultFromModel(model{
		items:       []menuItem{{integration: "codex"}},
		changeModel: true,
		modalSelector: selectorModel{
			selected: "qwen3:8b",
		},
		selectedSessionID: "session-123",
	})

	if result.Selection != SelectionChangeIntegration {
		t.Fatalf("Selection = %v, want %v", result.Selection, SelectionChangeIntegration)
	}
	if result.Integration != "codex" {
		t.Fatalf("Integration = %q, want codex", result.Integration)
	}
	if result.Model != "qwen3:8b" {
		t.Fatalf("Model = %q, want qwen3:8b", result.Model)
	}
	if result.SessionID != "session-123" {
		t.Fatalf("SessionID = %q, want session-123", result.SessionID)
	}
}

func TestViewShowsCodexResumeHint(t *testing.T) {
	m := model{
		items: []menuItem{
			{integration: "codex", title: "Launch Codex", description: "OpenAI's open-source coding agent"},
		},
		availableModels: map[string]bool{"qwen3:8b": true},
	}

	view := m.View()
	if !strings.Contains(view, "Press →, then → again on a model, to resume a past session") {
		t.Fatalf("view missing Codex resume hint: %q", view)
	}
}
