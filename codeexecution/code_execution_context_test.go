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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestGetStateDelta(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         map[string]any
	}{
		{
			name: "no error count present",
			initialState: &mockState{
				data: map[string]any{
					contextKey: map[string]any{
						sessionIdKey:          "session-id-1",
						processedFileNamesKey: []string{"file.txt"},
					},
				},
			},
			want: map[string]any{
				contextKey: map[string]any{
					sessionIdKey:          "session-id-1",
					processedFileNamesKey: []string{"file.txt"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			got := c.GetStateDelta()
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GetStateDelta() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetExecutionId(t *testing.T) {
	tests := []struct {
		name  string
		state *mockState
		want  string
	}{
		{
			name: "execution id present",
			state: &mockState{
				data: map[string]any{
					contextKey: map[string]any{
						sessionIdKey: "session-id-1",
					},
				},
			},
			want: "session-id-1",
		},
		{
			name: "execution id not present",
			state: &mockState{
				data: map[string]any{
					contextKey: map[string]any{},
				},
			},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.state)
			got := c.GetExecutionId()
			if got != tt.want {
				t.Errorf("expected executionId %s, got %s", tt.want, got)
			}
		})
	}
}

func TestSetExecutionId(t *testing.T) {
	tests := []struct {
		name      string
		state     *mockState
		want      map[string]any
		sessionId string
	}{
		{
			name:      "basic",
			state:     &mockState{},
			sessionId: "execution-id-1",
			want: map[string]any{
				sessionIdKey: "execution-id-1",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.state)
			c.SetExecutionId(tt.sessionId)
			if diff := cmp.Diff(c.contextState, tt.want); diff != "" {
				t.Errorf("SetExecutionId() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetProcessedFileNames(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         []string
	}{
		{
			name: "files present",
			initialState: &mockState{
				data: map[string]any{
					contextKey: map[string]any{
						processedFileNamesKey: []string{"file.txt"},
					},
				},
			},
			want: []string{"file.txt"},
		},
		{
			name: "no files present",
			initialState: &mockState{
				data: map[string]any{},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			got := c.GetProcessedFileNames()
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GetProcessedFileNames() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAddProcessedFileNames(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         map[string]any
	}{
		{
			name: "add file with none existing",
			initialState: &mockState{
				data: map[string]any{},
			},
			want: map[string]any{
				processedFileNamesKey: []string{"newfile.txt"},
			},
		},
		{
			name: "add file with existing",
			initialState: &mockState{
				data: map[string]any{
					contextKey: map[string]any{
						processedFileNamesKey: []string{"file.txt"},
					},
				},
			},
			want: map[string]any{
				processedFileNamesKey: []string{"file.txt", "newfile.txt"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			c.AddProcessedFileNames([]string{"newfile.txt"})
			if diff := cmp.Diff(c.contextState, tt.want); diff != "" {
				t.Errorf("AddProcessedFileNames() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetInputFiles(t *testing.T) {
	tests := []struct {
		name  string
		state *mockState
		want  []*File
	}{
		{
			name: "files present",
			state: &mockState{
				data: map[string]any{
					inputFileKey: []*File{
						{
							Name:     "file.txt",
							Content:  []byte("Some content"),
							MimeType: "text/plain",
						},
					},
				},
			},
			want: []*File{
				{
					Name:     "file.txt",
					Content:  []byte("Some content"),
					MimeType: "text/plain",
				},
			},
		},
		{
			name: "no files present",
			state: &mockState{
				data: map[string]any{},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.state)
			got := c.GetInputFiles()
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GetInputFiles() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAddInputFiles(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         []*File
	}{
		{
			name: "add file with none existing",
			initialState: &mockState{
				data: map[string]any{},
			},
			want: []*File{
				{
					Name:     "file.txt",
					Content:  []byte("Some content"),
					MimeType: "text/plain",
				},
			},
		},
		{
			name: "add file with existing",
			initialState: &mockState{
				data: map[string]any{
					inputFileKey: []*File{
						{
							Name:     "existing.txt",
							Content:  []byte("Some content"),
							MimeType: "text/plain",
						},
					},
				},
			},
			want: []*File{
				{
					Name:     "existing.txt",
					Content:  []byte("Some content"),
					MimeType: "text/plain",
				},
				{
					Name:     "file.txt",
					Content:  []byte("Some content"),
					MimeType: "text/plain",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			err := c.AddInputFiles([]*File{
				{
					Name:     "file.txt",
					Content:  []byte("Some content"),
					MimeType: "text/plain",
				},
			})
			if err != nil {
				t.Errorf("unexpected error = %v", err)
				return
			}
			files, err := c.sessionState.Get(inputFileKey)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if diff := cmp.Diff(files, tt.want); diff != "" {
				t.Errorf("AddInputFiles() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestClearInputFiles(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         []File
	}{
		{
			name: "clear with existing",
			initialState: &mockState{
				data: map[string]any{
					inputFileKey: []File{
						{
							Name:     "existing.txt",
							Content:  []byte("Some content"),
							MimeType: "text/plain",
						},
					},
					contextKey: map[string]any{
						inputFileKey: []File{
							{
								Name:     "existing.txt",
								Content:  []byte("Some content"),
								MimeType: "text/plain",
							},
						},
					},
				},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			err := c.ClearInputFiles()
			if err != nil {
				t.Errorf("unexpected error = %v", err)
				return
			}
			sessionStateRaw, err := c.sessionState.Get(inputFileKey)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			sessionStateFiles, _ := sessionStateRaw.([]File)
			if diff := cmp.Diff(sessionStateFiles, tt.want); diff != "" {
				t.Errorf("ClearInputFiles() mismatch (-want +got):\n%s", diff)
				return
			}
			contextStateFiles, _ := c.contextState[inputFileKey].([]File)
			if diff := cmp.Diff(contextStateFiles, tt.want); diff != "" {
				t.Errorf("ClearInputFiles() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetErrorCount(t *testing.T) {
	invocationId := "invocation-1"
	tests := []struct {
		name        string
		state       *mockState
		expectError bool
		want        int32
		validate    func(t *testing.T, got, want int32)
	}{
		{
			name: "error count present",
			state: &mockState{
				data: map[string]any{
					errorCountKey: map[string]int32{
						invocationId: 1,
					},
				},
			},
			want: 1,
		},
		{
			name: "error count not present",
			state: &mockState{
				data: map[string]any{},
			},
			want: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.state)
			got := c.GetErrorCount(invocationId)
			if got != tt.want {
				t.Errorf("expected error count=%d, got %d", tt.want, got)
			}
		})
	}
}

func TestResetErrorCount(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         map[string]int32
	}{
		{
			name: "error count present",
			initialState: &mockState{
				data: map[string]any{
					errorCountKey: map[string]int32{
						"invocation-1": 1,
					},
				},
			},
			want: map[string]int32{},
		},
		{
			name: "multiple error counts present",
			initialState: &mockState{
				data: map[string]any{
					errorCountKey: map[string]int32{
						"invocation-1": 1,
						"invocation-2": 1,
					},
				},
			},
			want: map[string]int32{
				"invocation-2": 1,
			},
		},
		{
			name: "no error counts present",
			initialState: &mockState{
				data: map[string]any{},
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			err := c.ResetErrorCount("invocation-1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			errCount, err := c.sessionState.Get(errorCountKey)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			errCountMap, _ := errCount.(map[string]int32)
			if diff := cmp.Diff(errCountMap, tt.want); diff != "" {
				t.Errorf("ResetErrorCount() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestIncrementErrorCount(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         map[string]int32
	}{
		{
			name: "no error count present",
			initialState: &mockState{
				data: map[string]any{},
			},
			want: map[string]int32{
				"invocation-1": 1,
			},
		},
		{
			name: "error counts present",
			initialState: &mockState{
				data: map[string]any{
					errorCountKey: map[string]int32{
						"invocation-1": 1,
					},
				},
			},
			want: map[string]int32{
				"invocation-1": 2,
			},
		},
		{
			name: "multiple error counts present",
			initialState: &mockState{
				data: map[string]any{
					errorCountKey: map[string]int32{
						"invocation-1": 1,
						"invocation-2": 1,
					},
				},
			},
			want: map[string]int32{
				"invocation-1": 2,
				"invocation-2": 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			err := c.IncrementErrorCount("invocation-1")
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			x, err := c.sessionState.Get(errorCountKey)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if diff := cmp.Diff(x, tt.want); diff != "" {
				t.Errorf("IncrementErrorCount() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestUpdateCodeExecutionResult(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		invocationId string
		code         string
		stdout       string
		stderr       string
		want         map[string][]executionResult
	}{
		{
			name: "no existing results",
			initialState: &mockState{
				data: map[string]any{},
			},
			invocationId: "invocation-1",
			code:         "fmt.Println(\"Hello\")",
			stdout:       "Hello\n",
			stderr:       "",
			want: map[string][]executionResult{
				"invocation-1": {
					{
						Code:   "fmt.Println(\"Hello\")",
						Stdout: "Hello\n",
						Stderr: "",
					},
				},
			},
		},
		{
			name: "existing results",
			initialState: &mockState{
				data: map[string]any{
					executionResultsKey: map[string][]executionResult{
						"invocation-1": {
							{
								Code:      "fmt.Println(\"Hello\")", // 1. Start with Hello
								Stdout:    "Hello\n",
								Stderr:    "",
								Timestamp: time.Now().UTC(),
							},
						},
					},
				},
			},
			invocationId: "invocation-1",
			code:         "fmt.Println(\"World\")",
			stdout:       "World\n",
			stderr:       "",
			want: map[string][]executionResult{
				"invocation-1": {
					{
						Code:   "fmt.Println(\"Hello\")",
						Stdout: "Hello\n",
						Stderr: "",
					},
					{
						Code:   "fmt.Println(\"World\")",
						Stdout: "World\n",
						Stderr: "",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCodeExecutorContext(tt.initialState)
			err := c.UpdateCodeExecutionResult(tt.invocationId, tt.code, tt.stdout, tt.stderr)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			result, err := c.sessionState.Get(executionResultsKey)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			resultsMap, _ := result.(map[string][]executionResult)
			if diff := cmp.Diff(resultsMap, tt.want, cmpopts.IgnoreFields(executionResult{}, "Timestamp")); diff != "" {
				t.Errorf("UpdateCodeExecutionResult() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetCodeExecutorContext(t *testing.T) {
	tests := []struct {
		name         string
		initialState *mockState
		want         map[string]any
	}{
		{
			name: "...",
			initialState: &mockState{
				data: map[string]any{
					contextKey: map[string]any{
						sessionIdKey:          "session-id1",
						processedFileNamesKey: []string{"file.txt"},
					},
				},
			},
			want: map[string]any{
				sessionIdKey:          "session-id1",
				processedFileNamesKey: []string{"file.txt"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getCodeExecutorContext(tt.initialState)
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("getCodeExecutorContext() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
