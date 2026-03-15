package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/cmd/launch"
)

func launcherTestState() *launch.LauncherState {
	return &launch.LauncherState{
		LastSelection: "run",
		RunModel:      "qwen3:8b",
		Integrations: map[string]launch.LauncherIntegrationState{
			"claude": {
				Name:         "claude",
				DisplayName:  "Claude Code",
				Description:  "Anthropic's coding tool with subagents",
				Selectable:   true,
				Changeable:   true,
				CurrentModel: "glm-5:cloud",
			},
			"codex": {
				Name:        "codex",
				DisplayName: "Codex",
				Description: "OpenAI's open-source coding agent",
				Selectable:  true,
				Changeable:  true,
			},
			"openclaw": {
				Name:            "openclaw",
				DisplayName:     "OpenClaw",
				Description:     "Personal AI with 100+ skills",
				Selectable:      true,
				Changeable:      true,
				AutoInstallable: true,
			},
			"droid": {
				Name:        "droid",
				DisplayName: "Droid",
				Description: "Factory's coding agent across terminal and IDEs",
				Selectable:  true,
				Changeable:  true,
			},
			"pi": {
				Name:        "pi",
				DisplayName: "Pi",
				Description: "Minimal AI agent toolkit with plugin support",
				Selectable:  true,
				Changeable:  true,
			},
		},
	}
}

func TestMenuRendersPinnedItemsAndMore(t *testing.T) {
	view := newModel(launcherTestState()).View()
	for _, want := range []string{"Run a model", "Launch Claude Code", "Launch Codex", "Launch OpenClaw", "More..."} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected menu view to contain %q\n%s", want, view)
		}
	}
}

func TestMenuExpandsOthersFromLastSelection(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "pi"

	menu := newModel(state)
	if !menu.showOthers {
		t.Fatal("expected others section to expand when last selection is in the overflow list")
	}
	view := menu.View()
	if !strings.Contains(view, "Launch Pi") {
		t.Fatalf("expected expanded view to contain overflow integration\n%s", view)
	}
	if strings.Contains(view, "More...") {
		t.Fatalf("expected expanded view to replace More... item\n%s", view)
	}
}

func TestMenuEnterOnRunSelectsRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on run to select run action, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnRunSelectsChangeRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel, ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on run to select change-run action, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuEnterOnIntegrationSelectsLaunch(t *testing.T) {
	menu := newModel(launcherTestState())
	menu.cursor = 1
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "claude"}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on integration to launch, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnIntegrationSelectsConfigure(t *testing.T) {
	menu := newModel(launcherTestState())
	menu.cursor = 1
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "claude", ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on integration to configure, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuIgnoresDisabledActions(t *testing.T) {
	state := launcherTestState()
	claude := state.Integrations["claude"]
	claude.Selectable = false
	claude.Changeable = false
	state.Integrations["claude"] = claude

	menu := newModel(state)
	menu.cursor = 1

	updatedEnter, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if updatedEnter.(model).selected {
		t.Fatal("expected non-selectable integration to ignore enter")
	}

	updatedRight, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	if updatedRight.(model).selected {
		t.Fatal("expected non-changeable integration to ignore right")
	}
}

func TestMenuShowsCurrentModelSuffixes(t *testing.T) {
	menu := newModel(launcherTestState())
	runView := menu.View()
	if !strings.Contains(runView, "(qwen3:8b)") {
		t.Fatalf("expected run row to show current model suffix\n%s", runView)
	}

	menu.cursor = 1
	integrationView := menu.View()
	if !strings.Contains(integrationView, "(glm-5:cloud)") {
		t.Fatalf("expected integration row to show current model suffix\n%s", integrationView)
	}
}

func TestMenuShowsInstallStatusAndHint(t *testing.T) {
	state := launcherTestState()
	codex := state.Integrations["codex"]
	codex.Installed = false
	codex.Selectable = false
	codex.Changeable = false
	codex.InstallHint = "Install from https://example.com/codex"
	state.Integrations["codex"] = codex

	menu := newModel(state)
	menu.cursor = 2
	view := menu.View()
	if !strings.Contains(view, "(not installed)") {
		t.Fatalf("expected not-installed marker\n%s", view)
	}
	if !strings.Contains(view, codex.InstallHint) {
		t.Fatalf("expected install hint in description\n%s", view)
	}
}

// ---- Session resume tests ----

func TestCodexRightOpensSessionPicker(t *testing.T) {
	// Create a fake codex session so ListCodexSessions returns something.
	home := t.TempDir()
	t.Setenv("HOME", home)
	sessionPath := filepath.Join(home, ".codex", "sessions", "2026", "03", "10", "session.jsonl")
	if err := os.MkdirAll(filepath.Dir(sessionPath), 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	cwd := t.TempDir()
	// Change to cwd so that os.Getwd() in openCodexSessionModal matches the session's cwd.
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if err := os.Chdir(cwd); err != nil {
		t.Fatalf("Chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(origDir) })
	data := `{"payload":{"id":"session-123","timestamp":"2026-03-10T12:00:00Z","cwd":"` + cwd + `","model_provider":"ollama","git":{"branch":"main","repository_url":"https://github.com/acme/ollama.git"}}}` + "\n" +
		`{"payload":{"model":"qwen3:8b"}}` + "\n" +
		`{"payload":{"type":"user_message","message":"Fix the bug."}}` + "\n"
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

	state := launcherTestState()
	codex := state.Integrations["codex"]
	codex.CurrentModel = "qwen3:8b"
	state.Integrations["codex"] = codex

	menu := newModel(state)
	// cursor 2 = codex in the default menu
	menu.cursor = 2

	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)

	if !got.showingSessionModal {
		t.Fatal("expected showingSessionModal=true after right on codex")
	}
	// Both the Ollama session and the cross-provider OpenAI session should now
	// appear in the picker (2 items total). The OpenAI session is placed after
	// the Ollama session in the "Other Sessions (cross-provider)" section.
	if len(got.sessionSelector.items) != 2 {
		t.Fatalf("len(sessionSelector.items) = %d, want 2 (ollama + cross-provider session)", len(got.sessionSelector.items))
	}
	if got.sessionSelector.items[0].Value != "session-123" {
		t.Fatalf("session value = %q, want session-123", got.sessionSelector.items[0].Value)
	}
	if !got.sessionSelector.items[0].Recommended {
		t.Fatal("expected matching-model session to be pinned as recommended")
	}
	if !strings.Contains(got.sessionSelector.items[0].Description, "qwen3:8b") {
		t.Fatalf("session description = %q, want model metadata", got.sessionSelector.items[0].Description)
	}
	if got.sessionSelector.recommendedHeader != "Matching Model" {
		t.Fatalf("recommendedHeader = %q, want Matching Model", got.sessionSelector.recommendedHeader)
	}
	if got.sessionSelector.otherHeader != "Other Sessions (cross-provider)" {
		t.Fatalf("otherHeader = %q, want \"Other Sessions (cross-provider)\"", got.sessionSelector.otherHeader)
	}
	// The second item should be the cross-provider OpenAI session.
	if got.sessionSelector.items[1].Value != "session-999" {
		t.Fatalf("items[1].Value = %q, want session-999 (cross-provider session)", got.sessionSelector.items[1].Value)
	}
	if got.sessionSelector.items[1].Recommended {
		t.Fatal("cross-provider session should not be marked as recommended")
	}
	// The cross-provider session description should contain the original provider
	// and a note that the local model will be used on resume.
	if !strings.Contains(got.sessionSelector.items[1].Description, "originally openai") {
		t.Fatalf("cross-provider description = %q, want 'originally openai'", got.sessionSelector.items[1].Description)
	}
	if !strings.Contains(got.sessionSelector.items[1].Description, "will use qwen3:8b") {
		t.Fatalf("cross-provider description = %q, want 'will use qwen3:8b'", got.sessionSelector.items[1].Description)
	}
	_ = cwd // used in session data
}

func TestCodexSessionPickerEnterSelectsSession(t *testing.T) {
	menu := model{
		state: launcherTestState(),
		items: []menuItem{{integration: "codex"}},
		showingSessionModal: true,
		sessionSelector: selectorModel{
			items: ReorderItems([]SelectItem{
				{Name: "Fix the bug  [qwen3:8b]", Value: "session-123", Recommended: true},
			}),
		},
	}

	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)

	if !got.selected {
		t.Fatal("expected selected=true after enter in session picker")
	}
	if got.action.Kind != TUIActionLaunchIntegration {
		t.Fatalf("action.Kind = %v, want TUIActionLaunchIntegration", got.action.Kind)
	}
	if got.action.Integration != "codex" {
		t.Fatalf("action.Integration = %q, want codex", got.action.Integration)
	}
	if got.action.SessionID != "session-123" {
		t.Fatalf("action.SessionID = %q, want session-123", got.action.SessionID)
	}
}

func TestIntegrationLaunchRequestWithSessionID(t *testing.T) {
	action := TUIAction{
		Kind:        TUIActionLaunchIntegration,
		Integration: "codex",
		SessionID:   "session-abc",
	}
	req := action.IntegrationLaunchRequest()
	if len(req.ExtraArgs) != 2 || req.ExtraArgs[0] != "resume" || req.ExtraArgs[1] != "session-abc" {
		t.Fatalf("ExtraArgs = %v, want [resume session-abc]", req.ExtraArgs)
	}
}

func TestSessionPickerEscClosesModal(t *testing.T) {
	menu := model{
		state: launcherTestState(),
		items: []menuItem{{integration: "codex"}},
		showingSessionModal: true,
		sessionSelector: selectorModel{
			items: []SelectItem{{Name: "session", Value: "s1"}},
		},
	}
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEsc})
	got := updated.(model)
	if got.showingSessionModal {
		t.Fatal("expected showingSessionModal=false after esc")
	}
	if got.selected {
		t.Fatal("expected selected=false after esc")
	}
}
