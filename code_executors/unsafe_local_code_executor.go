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

package code_executors

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"time"

	"google.golang.org/adk/agent"
)

const DEFAULT_SCRIPT_TIMEOUT = 300 * time.Second

type UnsafeLocalCodeExecutor struct {
	Timeout          time.Duration
	BaseCodeExecutor *BaseCodeExecutor
}

func (s *UnsafeLocalCodeExecutor) ExecuteCode(ctx agent.InvocationContext, input CodeExecutionInput) (CodeExecutionResult, error) {
	scriptPath := input.ScriptPath
	ext := ""
	if i := strings.LastIndex(scriptPath, "."); i >= 0 {
		ext = strings.ToLower(scriptPath[i+1:])
	}

	if ext != "py" && ext != "sh" && ext != "bash" {
		extMsg := "(no extension)"
		if ext != "" {
			extMsg = "." + ext
		}
		return CodeExecutionResult{
			StdErr: fmt.Sprintf("UNSUPPORTED_SCRIPT_TYPE: Unsupported script type '%s'. Supported types: .py, .sh, .bash", extMsg),
		}, nil
	}

	var cmd *exec.Cmd
	excCtx := context.Background()
	if s.Timeout > 0 {
		var cancel context.CancelFunc
		excCtx, cancel = context.WithTimeout(context.Background(), s.Timeout)
		defer cancel()
	}

	argv := []string{scriptPath}
	if input.Args != nil {
		switch args := input.Args.(type) {
		case []string:
			argv = append(argv, args...)
		case []interface{}:
			for _, v := range args {
				argv = append(argv, fmt.Sprint(v))
			}
		case map[string]string:
			for k, v := range args {
				argv = append(argv, "--"+k, v)
			}
		case map[string]interface{}:
			for k, v := range args {
				argv = append(argv, "--"+k, fmt.Sprint(v))
			}
		default:
			return CodeExecutionResult{
				StdErr: fmt.Sprintf("INVALID_ARGS: Unsupported args type: %T. Expected list or map.", input.Args),
			}, nil
		}
	}

	if ext == "py" {
		cmd = exec.CommandContext(excCtx, "python3", argv...)
	} else {
		cmd = exec.CommandContext(excCtx, "bash", argv...)
	}

	log.Printf("UnsafeLocalCodeExecutor cmd  is %s", cmd.String())

	var stdoutBuf, stderrBuf strings.Builder
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	runErr := cmd.Run()

	stdout := stdoutBuf.String()
	stderr := stderrBuf.String()
	rc := 0

	if runErr != nil {
		log.Printf("UnsafeLocalCodeExecutor cmd executed error: %s", runErr.Error())
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			rc = ee.ExitCode()
			if rc != 0 && stderr == "" {
				stderr += "\n" + fmt.Sprintf("Exit code %d", rc)
			}
		} else {
			stderr += "\n" + fmt.Sprintf("cmd {%s} run error:%s", cmd.String(), runErr.Error())
		}
	}

	log.Printf("UnsafeLocalCodeExecutor result: %s", stdout)

	return CodeExecutionResult{
		StdOut: stdout,
		StdErr: stderr,
	}, nil
}

func NewUnsafeLocalCodeExecutor(timeout time.Duration) *UnsafeLocalCodeExecutor {
	if timeout == 0 {
		timeout = DEFAULT_SCRIPT_TIMEOUT
	}
	return &UnsafeLocalCodeExecutor{
		Timeout:          timeout,
		BaseCodeExecutor: DefaultBaseCodeExecutor(),
	}
}
