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
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestUnsafeLocalCodeExecutor(t *testing.T) {
	// Setup paths
	cwd, _ := os.Getwd()
	scriptsDir := filepath.Join(cwd, "test_scripts")
	input := CodeExecutionInput{
		ScriptPath: filepath.Join(scriptsDir, "multiply.py"),
		Args:       []string{"2", "3", "4"},
	}
	executor := NewUnsafeLocalCodeExecutor(300 * time.Second)
	// Invoke
	result, err := executor.ExecuteCode(nil, input)
	if err != nil {
		t.Errorf("UnsafeLocalCodeExecutor should not return an error: %s", err.Error())
		return
	}
	t.Logf("UnsafeLocalCodeExecutor result: %s", result.StdOut)
}

func TestSkillScriptExecutor_ExecuteCode(t *testing.T) {
	// Setup paths
	cwd, _ := os.Getwd()
	scriptsDir := filepath.Join(cwd, "test_scripts")

	tests := []struct {
		name           string
		scriptPath     string
		args           any
		timeout        time.Duration
		expectedStdout string
		expectedStderr string
		expectError    bool
	}{
		{
			name:           "Python Hello",
			scriptPath:     filepath.Join(scriptsDir, "hello.py"),
			expectedStdout: "Hello from Python",
		},
		{
			name:           "Bash Hello",
			scriptPath:     filepath.Join(scriptsDir, "hello.sh"),
			expectedStdout: "Hello from Bash",
		},
		{
			name:           "Python Args List",
			scriptPath:     filepath.Join(scriptsDir, "args.py"),
			args:           []string{"arg1", "arg2"},
			expectedStdout: "Arguments: ['arg1', 'arg2']",
		},
		{
			name:           "Python Args Interface List",
			scriptPath:     filepath.Join(scriptsDir, "args.py"),
			args:           []interface{}{"arg1", 123},
			expectedStdout: "Arguments: ['arg1', '123']",
		},
		{
			name:           "Python Args Map",
			scriptPath:     filepath.Join(scriptsDir, "args.py"),
			args:           map[string]string{"key": "value"},
			expectedStdout: "Arguments: ['--key', 'value']",
		},
		{
			name:           "Fail Script",
			scriptPath:     filepath.Join(scriptsDir, "fail.sh"),
			expectedStderr: "This is an error",
		},
		{
			name:       "Timeout Script",
			scriptPath: filepath.Join(scriptsDir, "timeout.sh"),
			timeout:    1 * time.Second,
		},
		{
			name:           "Unsupported Extension",
			scriptPath:     "test.txt",
			expectedStderr: "UNSUPPORTED_SCRIPT_TYPE: Unsupported script type '.txt'. Supported types: .py, .sh, .bash",
		},
		{
			name:           "Args List",
			scriptPath:     filepath.Join(scriptsDir, "multiply.py"),
			args:           []string{"2", "3", "4"},
			expectedStdout: "24.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executor := NewUnsafeLocalCodeExecutor(tt.timeout)
			input := CodeExecutionInput{
				ScriptPath: tt.scriptPath,
				Args:       tt.args,
			}

			// Invoke
			result, err := executor.ExecuteCode(nil, input)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			if tt.expectedStdout != "" {
				assert.Contains(t, result.StdOut, tt.expectedStdout)
			}
			if tt.expectedStderr != "" {
				assert.Contains(t, result.StdErr, tt.expectedStderr)
			}
		})
	}
}
