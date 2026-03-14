//go:build windows

package launch

import "os/exec"

// setCodexSysProcAttr is a no-op on Windows. Process group management via
// Setpgid is a POSIX concept; on Windows the process will be cleaned up
// through the normal context-cancellation mechanism.
func setCodexSysProcAttr(cmd *exec.Cmd) {}
