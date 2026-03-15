package tui

import (
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

// TestCodexRightExitsTUIWithForceConfigure verifies that pressing right arrow
// on the Codex menu item exits the TUI with ForceConfigure=true, which triggers
// the two-level model+session picker via DefaultCodexPicker.
func TestCodexRightExitsTUIWithForceConfigure(t *testing.T) {
	state := launcherTestState()
	codex := state.Integrations["codex"]
	codex.CurrentModel = "qwen3:8b"
	state.Integrations["codex"] = codex

	menu := newModel(state)
	// cursor 2 = codex in the default menu
	menu.cursor = 2

	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)

	// Right arrow on Codex should exit the TUI with ForceConfigure=true.
	if !got.quitting {
		t.Fatal("expected quitting=true after right on codex")
	}
	if !got.selected {
		t.Fatal("expected selected=true after right on codex")
	}
	if !got.action.ForceConfigure {
		t.Fatal("expected action.ForceConfigure=true after right on codex")
	}
	if got.action.Integration != "codex" {
		t.Fatalf("action.Integration = %q, want codex", got.action.Integration)
	}
	// The inline session modal should NOT be shown.
	if got.showingSessionModal {
		t.Fatal("expected showingSessionModal=false: session picker is now handled by DefaultCodexPicker")
	}
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
