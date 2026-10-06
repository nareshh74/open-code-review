//go:build !windows

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"errors"
	"os/exec"
	"syscall"
)

type acpProcessTree struct {
	pid int
}

func configureACPCommand(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func attachACPProcess(cmd *exec.Cmd) (*acpProcessTree, error) {
	return &acpProcessTree{pid: cmd.Process.Pid}, nil
}

func (p *acpProcessTree) terminate() error {
	err := syscall.Kill(-p.pid, syscall.SIGKILL)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

func (p *acpProcessTree) close() error {
	return p.terminate()
}
