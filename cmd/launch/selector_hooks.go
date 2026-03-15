package launch

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/term"
)

// ANSI escape sequences for terminal formatting.
const (
	ansiBold   = "\033[1m"
	ansiReset  = "\033[0m"
	ansiGray   = "\033[37m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
)

// ErrCancelled is returned when the user cancels a selection.
var ErrCancelled = errors.New("cancelled")

// errCancelled is kept as an internal alias for existing call sites.
var errCancelled = ErrCancelled

// ErrRightArrow is returned by a SingleSelector when the user pressed the right
// arrow key on a model, signalling that a sub-picker (e.g. session picker) should
// be shown for that model. The selected model name is returned alongside this error.
var ErrRightArrow = errors.New("right arrow")

// DefaultConfirmPrompt provides a TUI-based confirmation prompt.
// When set, ConfirmPrompt delegates to it instead of using raw terminal I/O.
var DefaultConfirmPrompt func(prompt string) (bool, error)

// SingleSelector is a function type for single item selection.
// current is the name of the previously selected item to highlight; empty means no pre-selection.
type SingleSelector func(title string, items []ModelItem, current string) (string, error)

// MultiSelector is a function type for multi item selection.
type MultiSelector func(title string, items []ModelItem, preChecked []string) ([]string, error)

// DefaultSingleSelector is the default single-select implementation.
var DefaultSingleSelector SingleSelector

// DefaultMultiSelector is the default multi-select implementation.
var DefaultMultiSelector MultiSelector

// DefaultSignIn provides a TUI-based sign-in flow.
// When set, ensureAuth uses it instead of plain text prompts.
// Returns the signed-in username or an error.
var DefaultSignIn func(modelName, signInURL string) (string, error)

// DefaultCodexSessionSelector is an optional hook for showing a session picker
// after the user presses right arrow on a model in the Codex model picker.
// It receives the selected model name and returns the session ID to resume, or
// an empty string to start a new session, or ErrCancelled to abort.
var DefaultCodexSessionSelector func(model string) (sessionID string, err error)

// CodexPickerSelection is the result of DefaultCodexPicker.
type CodexPickerSelection struct {
	// Model is the model the user selected.
	Model string
	// SessionID is the session to resume, or empty for a new session.
	SessionID string
}

// DefaultCodexPicker is an optional hook that replaces the standard
// DefaultSingleSelector for Codex. It shows a two-level model+session picker
// and returns the selected model and optional session ID to resume.
// Returns ErrCancelled if the user cancels.
var DefaultCodexPicker func(title string, items []ModelItem, current string) (CodexPickerSelection, error)

type launchConfirmPolicy struct {
	yes               bool
	requireYesMessage bool
}

var currentLaunchConfirmPolicy launchConfirmPolicy

func withLaunchConfirmPolicy(policy launchConfirmPolicy) func() {
	old := currentLaunchConfirmPolicy
	currentLaunchConfirmPolicy = policy
	return func() {
		currentLaunchConfirmPolicy = old
	}
}

// ConfirmPrompt is the shared confirmation gate for launch flows (integration
// edits, missing-model pulls, sign-in prompts, OpenClaw install/security, etc).
// Behavior is controlled by currentLaunchConfirmPolicy, typically scoped by
// withLaunchConfirmPolicy in LaunchCmd (e.g. auto-approve with --yes).
func ConfirmPrompt(prompt string) (bool, error) {
	if currentLaunchConfirmPolicy.yes {
		return true, nil
	}
	if currentLaunchConfirmPolicy.requireYesMessage {
		return false, fmt.Errorf("%s requires confirmation; re-run with --yes to continue", prompt)
	}

	if DefaultConfirmPrompt != nil {
		return DefaultConfirmPrompt(prompt)
	}

	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return false, err
	}
	defer term.Restore(fd, oldState)

	fmt.Fprintf(os.Stderr, "%s (\033[1my\033[0m/n) ", prompt)

	buf := make([]byte, 1)
	for {
		if _, err := os.Stdin.Read(buf); err != nil {
			return false, err
		}

		switch buf[0] {
		case 'Y', 'y', 13:
			fmt.Fprintf(os.Stderr, "yes\r\n")
			return true, nil
		case 'N', 'n', 27, 3:
			fmt.Fprintf(os.Stderr, "no\r\n")
			return false, nil
		}
	}
}
