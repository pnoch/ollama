//go:build !windows

package launch

import (
	"os/exec"
	"syscall"
)

// setCodexSysProcAttr configures the command to run in its own process group
// so that when the Ollama process exits (or the context is cancelled), the
// entire codex process tree is terminated rather than leaving orphaned children.
func setCodexSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}
}
