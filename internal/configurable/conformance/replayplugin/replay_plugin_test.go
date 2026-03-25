// Copyright 2026 Google LLC
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

package replayplugin_test

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable/conformance/replayplugin"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// TestReplayPlugin verifies the plugin's callback behavior and replay functionality.
func TestReplayPlugin(t *testing.T) {
	// Setup per test
	setup := func(t *testing.T) (*plugin.Plugin, *MockSession, *MockState) {
		plugin := replayplugin.MustNew("/")
		sessionState := make(map[string]any)
		mockState := &MockState{data: sessionState}
		mockSession := &MockSession{state: mockState}
		return plugin, mockSession, mockState
	}

	t.Run("BeforeModelCallback_WithMatchingRecording_ReturnsRecordedResponse", func(t *testing.T) {
		plugin, mockSession, _ := setup(t)
		tempDir := t.TempDir()

		// 1. Create recording file
		recordingsYaml := `
recordings:
  - user_message_index: 0
    agent_name: "test_agent"
    llm_recording:
      llm_request:
        model: "gemini-2.0-flash"
        contents:
          - role: "user"
            parts:
              - text: "Hello"
      llm_responses:
        - content:
            role: "model"
            parts:
              - text: "Recorded response"
`
		createRecordingsFile(t, tempDir, recordingsYaml)

		// 2. Setup replay config
		err := mockSession.State().Set("_adk_replay_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 3. Load recordings (BeforeRunCallback)
		invContext := &MockInvocationContext{
			session:      mockSession,
			invocationID: "test-invocation",
		}
		_, err = plugin.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 4. Call BeforeModelCallback with matching request
		cbContext := &MockCallbackContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}

		request := &model.LLMRequest{
			Model: "gemini-2.0-flash",
			Contents: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello"}},
				},
			},
		}

		result, err := plugin.BeforeModelCallback()(cbContext, request)
		// 5. Verify
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if result.Content == nil {
			t.Fatal("expected non-nil result.Content")
		}

		if got := result.Content.Parts[0].Text; got != "Recorded response" {
			t.Errorf("expected %q, got %q", "Recorded response", got)
		}
	})

	t.Run("BeforeModelCallback_RequestMismatch_ReturnsEmpty", func(t *testing.T) {
		plugin, mockSession, _ := setup(t)
		tempDir := t.TempDir()

		// 1. Create recording with DIFFERENT model
		recordingsYaml := `
recordings:
  - user_message_index: 0
    agent_name: "test_agent"
    llm_recording:
      llm_request:
        model: "gemini-1.5-pro" 
        contents:
          - role: "user"
            parts:
              - text: "Hello"
`
		createRecordingsFile(t, tempDir, recordingsYaml)

		// 2. Setup config
		err := mockSession.State().Set("_adk_replay_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 3. Load recordings
		invContext := &MockInvocationContext{
			session:      mockSession,
			invocationID: "test-invocation",
		}
		_, err = plugin.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 4. Call with mismatched request (gemini-2.0-flash vs 1.5-pro)
		cbContext := &MockCallbackContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}

		request := &model.LLMRequest{
			Model: "gemini-2.0-flash",
			Contents: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello"}},
				},
			},
		}

		result, err := plugin.BeforeModelCallback()(cbContext, request)
		// 5. Verify result is nil (empty) and error is returned
		if err == nil {
			t.Fatal("expected error due to mismatch")
		}
		if result != nil {
			t.Errorf("expected nil result, got %v", result)
		}
	})

	t.Run("BeforeToolCallback_WithMatchingRecording_ReturnsRecordedResponse", func(t *testing.T) {
		plugin, mockSession, _ := setup(t)
		tempDir := t.TempDir()

		// 1. Create recording with tool call
		recordingsYaml := `
recordings:
  - user_message_index: 0
    agent_name: "test_agent"
    tool_recording:
      tool_call:
        name: "test_tool"
        args:
          param1: "value1"
          param2: 42
      tool_response:
        name: "test_tool"
        response:
          result: "success"
          data: "recorded data"
`
		createRecordingsFile(t, tempDir, recordingsYaml)

		// 2. Setup config
		err := mockSession.State().Set("_adk_replay_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 3. Load recordings
		invContext := &MockInvocationContext{
			session:      mockSession,
			invocationID: "test-invocation",
		}
		_, err = plugin.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 4. Call BeforeToolCallback
		mockTool := &MockTool{NameVal: "test_tool"}
		toolContext := &MockToolContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}
		toolArgs := map[string]any{"param1": "value1", "param2": 42}

		result, err := plugin.BeforeToolCallback()(toolContext, mockTool, toolArgs)
		// 5. Verify
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("expected non-nil result")
		}
		if got := result["result"]; got != "success" {
			t.Errorf("expected %q, got %q", "success", got)
		}
		if got := result["data"]; got != "recorded data" {
			t.Errorf("expected %q, got %q", "recorded data", got)
		}
	})

	t.Run("BeforeToolCallback_ToolNameMismatch_ReturnsEmpty", func(t *testing.T) {
		plugin, mockSession, _ := setup(t)
		tempDir := t.TempDir()

		// 1. Recording expects "expected_tool"
		recordingsYaml := `
recordings:
  - user_message_index: 0
    agent_name: "test_agent"
    tool_recording:
      tool_call:
        name: "expected_tool"
        args:
          param: "value"
`
		createRecordingsFile(t, tempDir, recordingsYaml)

		err := mockSession.State().Set("_adk_replay_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		invContext := &MockInvocationContext{session: mockSession, invocationID: "test-invocation"}
		_, err = plugin.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 2. Call with "actual_tool"
		mockTool := &MockTool{NameVal: "actual_tool"} // Wrong name
		toolContext := &MockToolContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}

		result, err := plugin.BeforeToolCallback()(toolContext, mockTool, map[string]any{"param": "value"})
		// 3. Verify nil result and error
		if err == nil {
			t.Fatal("expected error due to tool name mismatch")
		}
		if result != nil {
			t.Errorf("expected nil result, got %v", result)
		}
	})

	t.Run("BeforeToolCallback_ToolArgsMismatch_ReturnsEmpty", func(t *testing.T) {
		plugin, mockSession, _ := setup(t)
		tempDir := t.TempDir()

		// 1. Recording expects "expected_value"
		recordingsYaml := `
recordings:
  - user_message_index: 0
    agent_name: "test_agent"
    tool_recording:
      tool_call:
        name: "test_tool"
        args:
          param: "expected_value"
`
		createRecordingsFile(t, tempDir, recordingsYaml)

		err := mockSession.State().Set("_adk_replay_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		invContext := &MockInvocationContext{session: mockSession, invocationID: "test-invocation"}
		_, err = plugin.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// 2. Call with "actual_value"
		mockTool := &MockTool{NameVal: "test_tool"}
		toolContext := &MockToolContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}

		result, err := plugin.BeforeToolCallback()(toolContext, mockTool, map[string]any{"param": "actual_value"})
		// 3. Verify nil result and error
		if err == nil {
			t.Fatal("expected error due to tool args mismatch")
		}
		if result != nil {
			t.Errorf("expected nil result, got %v", result)
		}
	})
}

// --- Helpers & Mocks ---

func createRecordingsFile(t *testing.T, dir, content string) {
	path := filepath.Join(dir, "generated-recordings.yaml")
	err := os.WriteFile(path, []byte(content), 0o644)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// Mock interfaces needed to replace Mockito
// These should implement the interfaces defined in your main code.

type MockState struct {
	data map[string]any
}

func (m *MockState) Set(key string, val any) error { m.data[key] = val; return nil }
func (m *MockState) Get(key string) (any, error)   { return m.data[key], nil }
func (m *MockState) All() iter.Seq2[string, any]   { return nil }

// Assuming Session interface has a State() method
type MockSession struct {
	state *MockState
}

func (m *MockSession) ID() string                { return "mock-session-id" }
func (m *MockSession) AppName() string           { return "mock-app" }
func (m *MockSession) UserID() string            { return "mock-user" }
func (m *MockSession) State() session.State      { return m.state }
func (m *MockSession) Events() session.Events    { return nil }
func (m *MockSession) LastUpdateTime() time.Time { return time.Now() }

// MockInvocationContext
type MockInvocationContext struct {
	session      *MockSession
	invocationID string
}

func (m *MockInvocationContext) Session() session.Session                                { return m.session }
func (m *MockInvocationContext) InvocationID() string                                    { return m.invocationID }
func (m *MockInvocationContext) Agent() agent.Agent                                      { return nil }
func (m *MockInvocationContext) Artifacts() agent.Artifacts                              { return nil }
func (m *MockInvocationContext) Memory() agent.Memory                                    { return nil }
func (m *MockInvocationContext) Branch() string                                          { return "" }
func (m *MockInvocationContext) UserContent() *genai.Content                             { return nil }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig                             { return nil } // Use context? No, RunConfig struct.
func (m *MockInvocationContext) EndInvocation()                                          {}
func (m *MockInvocationContext) Ended() bool                                             { return false }
func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext { return m }
func (m *MockInvocationContext) Value(key any) any                                       { return nil }
func (m *MockInvocationContext) Deadline() (deadline time.Time, ok bool)                 { return time.Time{}, false }
func (m *MockInvocationContext) Done() <-chan struct{}                                   { return nil }
func (m *MockInvocationContext) Err() error                                              { return nil }

// MockCallbackContext
type MockCallbackContext struct {
	state        session.State
	invocationID string
	agentName    string
}

func (m *MockCallbackContext) State() session.State                    { return m.state }
func (m *MockCallbackContext) ReadonlyState() session.ReadonlyState    { return m.state }
func (m *MockCallbackContext) InvocationID() string                    { return m.invocationID }
func (m *MockCallbackContext) AgentName() string                       { return m.agentName }
func (m *MockCallbackContext) AppName() string                         { return "mock-app" }
func (m *MockCallbackContext) Branch() string                          { return "" }
func (m *MockCallbackContext) SessionID() string                       { return "mock-session-id" }
func (m *MockCallbackContext) UserID() string                          { return "mock-user" }
func (m *MockCallbackContext) UserContent() *genai.Content             { return nil }
func (m *MockCallbackContext) Artifacts() agent.Artifacts              { return nil }
func (m *MockCallbackContext) Value(key any) any                       { return nil }
func (m *MockCallbackContext) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }
func (m *MockCallbackContext) Done() <-chan struct{}                   { return nil }
func (m *MockCallbackContext) Err() error                              { return nil }

// MockToolContext
type MockToolContext struct {
	state        session.State
	invocationID string
	agentName    string
}

func (m *MockToolContext) State() session.State                 { return m.state }
func (m *MockToolContext) ReadonlyState() session.ReadonlyState { return m.state }
func (m *MockToolContext) InvocationID() string                 { return m.invocationID }
func (m *MockToolContext) AgentName() string                    { return m.agentName }
func (m *MockToolContext) FunctionCallID() string               { return "mock-function-call-id" }
func (m *MockToolContext) Actions() *session.EventActions       { return nil }
func (m *MockToolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, nil
}
func (m *MockToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (m *MockToolContext) RequestConfirmation(hint string, payload any) error   { return nil }
func (m *MockToolContext) AppName() string                                      { return "mock-app" }
func (m *MockToolContext) Branch() string                                       { return "" }
func (m *MockToolContext) SessionID() string                                    { return "mock-session-id" }
func (m *MockToolContext) UserID() string                                       { return "mock-user" }
func (m *MockToolContext) UserContent() *genai.Content                          { return nil }
func (m *MockToolContext) Artifacts() agent.Artifacts                           { return nil }
func (m *MockToolContext) Value(key any) any                                    { return nil }
func (m *MockToolContext) Deadline() (deadline time.Time, ok bool)              { return time.Time{}, false }
func (m *MockToolContext) Done() <-chan struct{}                                { return nil }
func (m *MockToolContext) Err() error                                           { return nil }

// MockTool
type MockTool struct {
	NameVal string
}

func (m *MockTool) Name() string                                        { return m.NameVal }
func (m *MockTool) Description() string                                 { return "mock tool" }
func (m *MockTool) IsLongRunning() bool                                 { return false }
func (m *MockTool) Run(ctx any, args map[string]any, toolCtx any) error { return nil }

func TestReplayPlugin_PathValidation(t *testing.T) {
	// Create a temporary directory structure for testing path validation
	tempDir := t.TempDir()
	safeDir := filepath.Join(tempDir, "safe")
	if err := os.Mkdir(safeDir, 0o755); err != nil {
		t.Fatalf("failed to create safe dir: %v", err)
	}

	// Create a safe recordings file
	createRecordingsFile(t, safeDir, "recordings: []")

	// Initialize plugin with restricted base directory
	plugin := replayplugin.MustNew(safeDir)
	sessionState := make(map[string]any)
	mockState := &MockState{data: sessionState}
	mockSession := &MockSession{state: mockState}
	invContext := &MockInvocationContext{
		session:      mockSession,
		invocationID: "test-invocation",
	}

	tests := []struct {
		name        string
		dir         string
		expectError bool
	}{
		{
			name:        "ValidPath_InsideBaseDir",
			dir:         safeDir,
			expectError: false,
		},
		{
			name:        "InvalidPath_ParentTraversal",
			dir:         filepath.Join(safeDir, ".."),
			expectError: true,
		},
		{
			name:        "InvalidPath_OutsideBaseDir",
			dir:         tempDir, // tempDir is parent of safeDir, so it's outside
			expectError: true,
		},
		{
			name:        "InvalidPath_AbsoluteOutside",
			dir:         "/etc", // outside
			expectError: true,
		},
		{
			name:        "InvalidPath_RelativeTraversal",
			dir:         "../", // Relative path traversing up
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set the config
			err := mockSession.State().Set("_adk_replay_config", map[string]any{
				"dir":                tt.dir,
				"user_message_index": 0,
			})
			if err != nil {
				t.Fatalf("unexpected error setting config: %v", err)
			}

			// Run BeforeRunCallback
			_, err = plugin.BeforeRunCallback()(invContext)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error for dir %q, got nil", tt.dir)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error for dir %q: %v", tt.dir, err)
				}
			}
		})
	}
}
