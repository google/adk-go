// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package codeexecution

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"google.golang.org/adk/agent"
)

// A code executor that runs locally.
//
// The code executor allows running code using local python installation (if available).
type unsafeLocalCodeExecutor struct {
	cfg *CodeExecutorConfig
}

func NewUnsafeLocalCodeExecutor(ctx context.Context, opts ...ConfigOption) (CodeExecutor, error) {
	return &unsafeLocalCodeExecutor{newCodeExecutionConfig(opts...)}, nil
}

func (e *unsafeLocalCodeExecutor) ExecuteCode(ctx agent.InvocationContext, req *CodeExecutionInput) (*CodeExecutionResult, error) {
	var pythonCmd string
	for _, py := range []string{"python3", "python"} {
		if _, err := exec.LookPath(py); err == nil {
			pythonCmd = py
			break
		}
	}
	if pythonCmd == "" {
		return nil, fmt.Errorf("valid python installation not found")
	}
	cmd := exec.CommandContext(ctx, pythonCmd)
	cmd.Stdin = strings.NewReader(req.Code)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			return nil, fmt.Errorf("failed to run process: %w", err)
		}
	}

	return &CodeExecutionResult{
		Stdout: stdout.String(),
		Stderr: stderr.String(),
	}, nil
}

func (e *unsafeLocalCodeExecutor) Config() *CodeExecutorConfig {
	return e.cfg
}

var _ CodeExecutor = (*unsafeLocalCodeExecutor)(nil)
