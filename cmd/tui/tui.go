package tui

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/version"
)

var (
	versionStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "243", Dark: "250"})

	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2)

	menuSelectedItemStyle = lipgloss.NewStyle().
				Bold(true).
				Background(lipgloss.AdaptiveColor{Light: "254", Dark: "236"})

	menuDescStyle = selectorDescStyle.
			PaddingLeft(4)

	greyedStyle = menuItemStyle.
			Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	greyedSelectedStyle = menuSelectedItemStyle.
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	modelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "243", Dark: "250"})

	notInstalledStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"}).
				Italic(true)
)

type menuItem struct {
	title       string
	description string
	integration string
	isRunModel  bool
	isOthers    bool
}

var mainMenuItems = []menuItem{
	{
		title:       "Run a model",
		description: "Start an interactive chat with a model",
		isRunModel:  true,
	},
	{
		integration: "claude",
	},
	{
		integration: "codex",
	},
	{
		integration: "openclaw",
	},
}

var othersMenuItem = menuItem{
	title:       "More...",
	description: "Show additional integrations",
	isOthers:    true,
}

type model struct {
	state               *launch.LauncherState
	items               []menuItem
	cursor              int
	showOthers          bool
	width               int
	quitting            bool
	selected            bool
	action              TUIAction
	// session resume state
	showingSessionModal bool
	sessionSelector     selectorModel
	selectedSessionID   string
	statusMsg           string // temporary status message shown near help text
}

func newModel(state *launch.LauncherState) model {
	m := model{
		state: state,
	}
	m.showOthers = shouldExpandOthers(state)
	m.items = buildMenuItems(state, m.showOthers)
	m.cursor = initialCursor(state, m.items)
	return m
}

func shouldExpandOthers(state *launch.LauncherState) bool {
	if state == nil {
		return false
	}
	for _, item := range otherIntegrationItems(state) {
		if item.integration == state.LastSelection {
			return true
		}
	}
	return false
}

func buildMenuItems(state *launch.LauncherState, showOthers bool) []menuItem {
	items := make([]menuItem, 0, len(mainMenuItems)+1)
	for _, item := range mainMenuItems {
		if item.integration == "" {
			items = append(items, item)
			continue
		}
		if integrationState, ok := state.Integrations[item.integration]; ok {
			items = append(items, integrationMenuItem(integrationState))
		}
	}

	if showOthers {
		items = append(items, otherIntegrationItems(state)...)
	} else {
		items = append(items, othersMenuItem)
	}

	return items
}

func integrationMenuItem(state launch.LauncherIntegrationState) menuItem {
	description := state.Description
	if description == "" {
		description = "Open " + state.DisplayName + " integration"
	}
	return menuItem{
		title:       "Launch " + state.DisplayName,
		description: description,
		integration: state.Name,
	}
}

func otherIntegrationItems(state *launch.LauncherState) []menuItem {
	pinned := map[string]bool{
		"claude":   true,
		"codex":    true,
		"openclaw": true,
	}

	var items []menuItem
	for _, info := range launch.ListIntegrationInfos() {
		if pinned[info.Name] {
			continue
		}
		integrationState, ok := state.Integrations[info.Name]
		if !ok {
			continue
		}
		items = append(items, integrationMenuItem(integrationState))
	}
	return items
}

func initialCursor(state *launch.LauncherState, items []menuItem) int {
	if state == nil || state.LastSelection == "" {
		return 0
	}
	for i, item := range items {
		if state.LastSelection == "run" && item.isRunModel {
			return i
		}
		if item.integration == state.LastSelection {
			return i
		}
	}
	return 0
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// Session modal takes priority
	if m.showingSessionModal {
		if kmsg, ok := msg.(tea.KeyMsg); ok {
			switch kmsg.Type {
			case tea.KeyCtrlC, tea.KeyEsc, tea.KeyLeft:
				m.showingSessionModal = false
				return m, nil
			case tea.KeyEnter:
				filtered := m.sessionSelector.filteredItems()
				if len(filtered) > 0 && m.sessionSelector.cursor < len(filtered) {
					m.selectedSessionID = filtered[m.sessionSelector.cursor].selectedValue()
				}
				if m.selectedSessionID != "" {
					item := m.items[m.cursor]
					integrationState := m.state.Integrations[item.integration]
					m.selected = true
					m.action = TUIAction{
						Kind:          TUIActionLaunchIntegration,
						Integration:   item.integration,
						ModelOverride: integrationState.CurrentModel,
						SessionID:     m.selectedSessionID,
					}
					m.quitting = true
					return m, tea.Quit
				}
				return m, nil
			default:
				m.sessionSelector.updateNavigation(kmsg)
			}
		}
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			m.quitting = true
			return m, tea.Quit

		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
			if m.showOthers && m.cursor < len(mainMenuItems) {
				m.showOthers = false
				m.items = buildMenuItems(m.state, false)
				m.cursor = min(m.cursor, len(m.items)-1)
			}
			return m, nil

		case "down", "j":
			if m.cursor < len(m.items)-1 {
				m.cursor++
			}
			if m.cursor < len(m.items) && m.items[m.cursor].isOthers && !m.showOthers {
				m.showOthers = true
				m.items = buildMenuItems(m.state, true)
			}
			return m, nil

		case "enter", " ":
			if m.selectableItem(m.items[m.cursor]) {
				m.selected = true
				m.action = actionForMenuItem(m.items[m.cursor], false)
				m.quitting = true
				return m, tea.Quit
			}
			return m, nil

		case "right", "l":
			item := m.items[m.cursor]
			if item.integration == "codex" && m.changeableItem(item) {
				// For Codex, right opens session picker
				integrationState := m.state.Integrations[item.integration]
				m.openCodexSessionModal(integrationState.CurrentModel)
				return m, nil
			}
			if item.isRunModel || m.changeableItem(item) {
				m.selected = true
				m.action = actionForMenuItem(item, true)
				m.quitting = true
				return m, tea.Quit
			}
			return m, nil
		}
	}
	return m, nil
}

// openCodexSessionModal opens the session picker for the given Codex model.
func (m *model) openCodexSessionModal(selectedModel string) {
	cwd, err := os.Getwd()
	if err != nil {
		m.statusMsg = "Unable to determine current directory for Codex sessions."
		return
	}

	sessions, err := launch.ListCodexSessions(cwd, 30)
	if err != nil {
		m.statusMsg = "Unable to read Codex sessions."
		return
	}
	if len(sessions) == 0 {
		m.statusMsg = fmt.Sprintf("No Codex sessions found in %s", cwd)
		return
	}


	items := make([]SelectItem, len(sessions))
	var count int
	for _, session := range sessions {
		// Skip sessions from non-Ollama providers (e.g. openai) since they
		// cannot be resumed against a local Ollama model.
		if session.Provider != "" && session.Provider != "ollama" {
			continue
		}
		recommended := selectedModel != "" && session.Model != "" && session.Model == selectedModel
		name := session.Title
		if session.Model != "" {
			name = fmt.Sprintf("%s  [%s]", name, session.Model)
		}
		items[count] = SelectItem{
			Name:        name,
			Value:       session.ID,
			Description: session.Description,
			Recommended: recommended,
		}
		count++
	}
	items = items[:count]
	if len(items) == 0 {
		m.statusMsg = "No Codex sessions found."
		return
	}
	title := "Resume Codex session:"
	if selectedModel != "" {
		title = fmt.Sprintf("Resume Codex session for %s:", selectedModel)
	}
	m.sessionSelector = selectorModel{
		title:             title,
		items:             ReorderItems(items),
		helpText:          "\u2191/\u2193 navigate \u2022 enter resume \u2022 matching model sessions pinned first \u2022 \u2190 back",
		recommendedHeader: "Matching Model",
		otherHeader:       "Other Sessions",
	}
	m.showingSessionModal = true
}

func (m model) selectableItem(item menuItem) bool {
	if item.isRunModel {
		return true
	}
	if item.integration == "" || item.isOthers {
		return false
	}
	state, ok := m.state.Integrations[item.integration]
	return ok && state.Selectable
}

func (m model) changeableItem(item menuItem) bool {
	if item.integration == "" || item.isOthers {
		return false
	}
	state, ok := m.state.Integrations[item.integration]
	return ok && state.Changeable
}

func (m model) View() string {
	if m.quitting {
		return ""
	}

	if m.showingSessionModal {
		content := m.sessionSelector.renderContent()
		if m.width > 0 {
			return lipgloss.NewStyle().MaxWidth(m.width).Render(content)
		}
		return content
	}

	s := selectorTitleStyle.Render("Ollama "+versionStyle.Render(version.Version)) + "\n\n"

	for i, item := range m.items {
		s += m.renderMenuItem(i, item)
	}

	s += "\n" + selectorHelpStyle.Render("↑/↓ navigate • enter launch • → configure • esc quit")

	if m.statusMsg != "" {
		s += "\n" + lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "124", Dark: "210"}).Render(m.statusMsg) + "\n"
	}

	if m.width > 0 {
		return lipgloss.NewStyle().MaxWidth(m.width).Render(s)
	}
	return s
}

func (m model) renderMenuItem(index int, item menuItem) string {
	cursor := ""
	style := menuItemStyle
	title := item.title
	description := item.description
	modelSuffix := ""

	if m.cursor == index {
		cursor = "▸ "
	}

	if item.isRunModel {
		if m.cursor == index && m.state.RunModel != "" {
			modelSuffix = " " + modelStyle.Render("("+m.state.RunModel+")")
		}
		if m.cursor == index {
			style = menuSelectedItemStyle
		}
	} else if item.isOthers {
		if m.cursor == index {
			style = menuSelectedItemStyle
		}
	} else {
		integrationState := m.state.Integrations[item.integration]
		if !integrationState.Selectable {
			if m.cursor == index {
				style = greyedSelectedStyle
			} else {
				style = greyedStyle
			}
		} else if m.cursor == index {
			style = menuSelectedItemStyle
		}

		if m.cursor == index && integrationState.CurrentModel != "" {
			modelSuffix = " " + modelStyle.Render("("+integrationState.CurrentModel+")")
		}

		if !integrationState.Installed {
			if integrationState.AutoInstallable {
				title += " " + notInstalledStyle.Render("(install)")
			} else {
				title += " " + notInstalledStyle.Render("(not installed)")
			}
			if m.cursor == index {
				if integrationState.AutoInstallable {
					description = "Press enter to install"
				} else if integrationState.InstallHint != "" {
					description = integrationState.InstallHint
				} else {
					description = "not installed"
				}
			}
		}
	}

	return style.Render(cursor+title) + modelSuffix + "\n" + menuDescStyle.Render(description) + "\n\n"
}

type TUIActionKind int

const (
	TUIActionNone TUIActionKind = iota
	TUIActionRunModel
	TUIActionLaunchIntegration
)

type TUIAction struct {
	Kind           TUIActionKind
	Integration    string
	ForceConfigure bool
	ModelOverride  string // model to use when resuming
	SessionID      string // session id if resuming a Codex session
}

func (a TUIAction) LastSelection() string {
	switch a.Kind {
	case TUIActionRunModel:
		return "run"
	case TUIActionLaunchIntegration:
		return a.Integration
	default:
		return ""
	}
}

func (a TUIAction) RunModelRequest() launch.RunModelRequest {
	return launch.RunModelRequest{ForcePicker: a.ForceConfigure}
}

func (a TUIAction) IntegrationLaunchRequest() launch.IntegrationLaunchRequest {
	req := launch.IntegrationLaunchRequest{
		Name:           a.Integration,
		ModelOverride:  a.ModelOverride,
		ForceConfigure: a.ForceConfigure,
	}
	if a.SessionID != "" {
		req.ExtraArgs = []string{"resume", a.SessionID}
	}
	return req
}

func actionForMenuItem(item menuItem, forceConfigure bool) TUIAction {
	switch {
	case item.isRunModel:
		return TUIAction{Kind: TUIActionRunModel, ForceConfigure: forceConfigure}
	case item.integration != "":
		return TUIAction{Kind: TUIActionLaunchIntegration, Integration: item.integration, ForceConfigure: forceConfigure}
	default:
		return TUIAction{Kind: TUIActionNone}
	}
}

func RunMenu(state *launch.LauncherState) (TUIAction, error) {
	menu := newModel(state)
	program := tea.NewProgram(menu)

	finalModel, err := program.Run()
	if err != nil {
		return TUIAction{Kind: TUIActionNone}, fmt.Errorf("error running TUI: %w", err)
	}

	finalMenu := finalModel.(model)
	if !finalMenu.selected {
		return TUIAction{Kind: TUIActionNone}, nil
	}

	return finalMenu.action, nil
}
