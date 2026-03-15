package tui

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/cmd/launch"
)

// CodexPickerResult is the result of RunCodexPicker.
type CodexPickerResult struct {
	// Model is the model the user selected.
	Model string
	// SessionID is the session the user chose to resume, or empty for a new session.
	SessionID string
	// Cancelled is true if the user pressed Esc/Ctrl+C without making a selection.
	Cancelled bool
}

// codexPickerState tracks which layer of the two-level picker is active.
type codexPickerState int

const (
	codexPickerStateModel   codexPickerState = iota // showing model picker
	codexPickerStateSession                         // showing session picker
)

// codexPickerModel is the bubbletea model for the two-level Codex picker.
type codexPickerModel struct {
	// model picker
	modelSelector selectorModel
	modelItems    []SelectItem

	// session picker
	sessionSelector selectorModel
	sessionItems    []SelectItem

	state codexPickerState

	// result
	selectedModel     string
	selectedSessionID string
	cancelled         bool
	done              bool

	width int
}

func newCodexPickerModel(modelItems []SelectItem, currentModel string) codexPickerModel {
	ms := selectorModelWithCurrent(
		"Select model for Codex:",
		modelItems,
		currentModel,
	)
	ms.helpText = "↑/↓ navigate • enter new session • → session picker • esc cancel"
	return codexPickerModel{
		modelSelector: ms,
		modelItems:    modelItems,
		state:         codexPickerStateModel,
	}
}

func (m codexPickerModel) Init() tea.Cmd {
	return nil
}

func (m codexPickerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if wmsg, ok := msg.(tea.WindowSizeMsg); ok {
		m.width = wmsg.Width
		m.modelSelector.width = wmsg.Width
		m.sessionSelector.width = wmsg.Width
		return m, nil
	}

	switch m.state {
	case codexPickerStateModel:
		return m.updateModelPicker(msg)
	case codexPickerStateSession:
		return m.updateSessionPicker(msg)
	}
	return m, nil
}

func (m codexPickerModel) updateModelPicker(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}

	switch keyMsg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.cancelled = true
		m.done = true
		return m, tea.Quit

	case tea.KeyEnter:
		// Launch new session with selected model.
		filtered := m.modelSelector.filteredItems()
		if len(filtered) > 0 && m.modelSelector.cursor < len(filtered) {
			m.selectedModel = filtered[m.modelSelector.cursor].selectedValue()
		}
		if m.selectedModel != "" {
			m.done = true
			return m, tea.Quit
		}
		return m, nil

	case tea.KeyRight:
		// Open session picker for the highlighted model.
		filtered := m.modelSelector.filteredItems()
		if len(filtered) > 0 && m.modelSelector.cursor < len(filtered) {
			m.selectedModel = filtered[m.modelSelector.cursor].selectedValue()
		}
		if m.selectedModel == "" {
			return m, nil
		}
		// Build session items for this model.
		items, title, err := buildCodexSessionItems(m.selectedModel)
		if err != nil || len(items) == 0 {
			// No sessions — just launch a new session directly.
			m.done = true
			return m, tea.Quit
		}
		ss := selectorModelWithCurrent(title, items, "")
		ss.helpText = "↑/↓ navigate • enter resume • ← back to model picker"
		if m.width > 0 {
			ss.width = m.width
		}
		m.sessionSelector = ss
		m.sessionItems = items
		m.state = codexPickerStateSession
		return m, nil

	default:
		m.modelSelector.updateNavigation(keyMsg)
		return m, nil
	}
}

func (m codexPickerModel) updateSessionPicker(msg tea.Msg) (tea.Model, tea.Cmd) {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return m, nil
	}

	switch keyMsg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.cancelled = true
		m.done = true
		return m, tea.Quit

	case tea.KeyLeft:
		// Go back to model picker.
		m.selectedModel = ""
		m.state = codexPickerStateModel
		return m, nil

	case tea.KeyEnter:
		filtered := m.sessionSelector.filteredItems()
		if len(filtered) > 0 && m.sessionSelector.cursor < len(filtered) {
			m.selectedSessionID = filtered[m.sessionSelector.cursor].selectedValue()
		}
		m.done = true
		return m, tea.Quit

	default:
		m.sessionSelector.updateNavigation(keyMsg)
		return m, nil
	}
}

func (m codexPickerModel) View() string {
	if m.done || m.cancelled {
		return ""
	}
	switch m.state {
	case codexPickerStateModel:
		return m.modelSelector.renderContent()
	case codexPickerStateSession:
		return m.sessionSelector.renderContent()
	}
	return ""
}

// RunCodexPicker runs the two-level Codex model+session picker and returns the
// result. The caller should check result.Cancelled before using the other fields.
func RunCodexPicker(modelItems []SelectItem, currentModel string) (CodexPickerResult, error) {
	if len(modelItems) == 0 {
		return CodexPickerResult{Cancelled: true}, fmt.Errorf("no models available")
	}

	m := newCodexPickerModel(modelItems, currentModel)
	p := tea.NewProgram(m)
	finalModel, err := p.Run()
	if err != nil {
		return CodexPickerResult{Cancelled: true}, fmt.Errorf("error running Codex picker: %w", err)
	}

	fm := finalModel.(codexPickerModel)
	if fm.cancelled {
		return CodexPickerResult{Cancelled: true}, nil
	}
	return CodexPickerResult{
		Model:     fm.selectedModel,
		SessionID: fm.selectedSessionID,
	}, nil
}

// SelectCodexSession shows a standalone session picker for the given model and
// returns the session ID the user selected, or an empty string to start a new
// session. Returns ErrCancelled if the user pressed Esc.
func SelectCodexSession(model string) (string, error) {
	items, title, err := buildCodexSessionItems(model)
	if err != nil {
		return "", err
	}
	if len(items) == 0 {
		// No sessions available — start a new session.
		return "", nil
	}

	m := selectorModelWithCurrent(title, items, "")
	m.helpText = "↑/↓ navigate • enter resume • ← back / new session"

	p := tea.NewProgram(m)
	finalModel, runErr := p.Run()
	if runErr != nil {
		return "", fmt.Errorf("error running session picker: %w", runErr)
	}

	fm := finalModel.(selectorModel)
	if fm.cancelled {
		return "", ErrCancelled
	}
	return fm.selected, nil
}

// buildCodexSessionItems builds the SelectItem list and title for the session
// picker.
func buildCodexSessionItems(selectedModel string) ([]SelectItem, string, error) {
	allSessions, err := launch.ListCodexSessions("", 60)
	if err != nil {
		return nil, "", fmt.Errorf("unable to read Codex sessions: %w", err)
	}

	var ollamaItems, otherItems []SelectItem
	for _, session := range allSessions {
		if session.Provider == "" || session.Provider == "ollama" {
			recommended := selectedModel != "" && session.Model != "" && session.Model == selectedModel
			name := session.Title
			if session.Model != "" {
				name = fmt.Sprintf("%s  [%s]", name, session.Model)
			}
			ollamaItems = append(ollamaItems, SelectItem{
				Name:        name,
				Value:       session.ID,
				Description: session.Description,
				Recommended: recommended,
			})
		} else {
			name := session.Title
			if session.Model != "" {
				name = fmt.Sprintf("%s  [%s]", name, session.Model)
			}
			desc := session.Description
			providerNote := fmt.Sprintf("originally %s", session.Provider)
			if selectedModel != "" {
				providerNote = fmt.Sprintf("originally %s • will use %s", session.Provider, selectedModel)
			}
			if desc != "" {
				desc = desc + "  (" + providerNote + ")"
			} else {
				desc = providerNote
			}
			otherItems = append(otherItems, SelectItem{
				Name:        name,
				Value:       session.ID,
				Description: desc,
				Recommended: false,
			})
		}
	}

	allItems := append(ReorderItems(ollamaItems), otherItems...)

	title := "Resume Codex session:"
	if selectedModel != "" {
		title = fmt.Sprintf("Resume Codex session for %s:", selectedModel)
	}

	return allItems, title, nil
}
