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

package recordplugin_test

import (
	"context"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable/conformance/recordplugin"
	"google.golang.org/adk/internal/configurable/conformance/replayplugin/recording"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

func TestRecordPlugin(t *testing.T) {
	setup := func(t *testing.T, baseDir string) (*plugin.Plugin, *MockSession, *MockState) {
		p := recordplugin.MustNew(baseDir)
		sessionState := make(map[string]any)
		mockState := &MockState{data: sessionState}
		mockSession := &MockSession{state: mockState}
		return p, mockSession, mockState
	}

	t.Run("RecordLlmAndToolCallsAndSaveYaml", func(t *testing.T) {
		tempDir := t.TempDir()
		p, mockSession, _ := setup(t, tempDir)

		err := mockSession.State().Set("_adk_recordings_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		invContext := &MockInvocationContext{
			session:      mockSession,
			invocationID: "test-invocation",
		}

		// 1. beforeRun
		_, err = p.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("beforeRun failed: %v", err)
		}

		// 2. beforeModel (LLM request)
		cbContext := &MockCallbackContext{
			state:        mockSession.State(),
			invocationID: "test-invocation",
			agentName:    "test_agent",
		}
		req := &model.LLMRequest{
			Model: "gemini-2.0-flash",
			Contents: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello agent"}},
				},
			},
			Config: &genai.GenerateContentConfig{
				HTTPOptions: &genai.HTTPOptions{
					ExtrasRequestProvider: func(body map[string]any) map[string]any {
						return body
					},
				},
			},
			Tools: map[string]any{
				"test_tool": "dummy_val",
			},
		}
		_, err = p.BeforeModelCallback()(cbContext, req)
		if err != nil {
			t.Fatalf("beforeModel failed: %v", err)
		}

		// 3. afterModel (LLM response)
		resp := &model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: "Hello user, calling tool now"}},
			},
			ModelVersion: "gemini-2.0-flash",
			Partial:      false,
		}
		_, err = p.AfterModelCallback()(cbContext, resp, nil)
		if err != nil {
			t.Fatalf("afterModel failed: %v", err)
		}

		// 4. beforeTool
		toolContext := &MockToolContext{
			state:          mockSession.State(),
			invocationID:   "test-invocation",
			agentName:      "test_agent",
			functionCallID: "call-123",
		}
		mockTool := &MockTool{NameVal: "test_tool"}
		toolArgs := map[string]any{"param": "test"}
		_, err = p.BeforeToolCallback()(toolContext, mockTool, toolArgs)
		if err != nil {
			t.Fatalf("beforeTool failed: %v", err)
		}

		// 5. afterTool
		toolResult := map[string]any{"result": "test_success"}
		_, err = p.AfterToolCallback()(toolContext, mockTool, toolArgs, toolResult, nil)
		if err != nil {
			t.Fatalf("afterTool failed: %v", err)
		}

		// 6. afterRun
		p.AfterRunCallback()(invContext)

		// Verify created YAML content
		filePath := filepath.Join(tempDir, "generated-recordings.yaml")
		data, err := os.ReadFile(filePath)
		if err != nil {
			t.Fatalf("failed to read recordings file: %v", err)
		}

		var recordings recording.Recordings
		err = yaml.Unmarshal(data, &recordings)
		if err != nil {
			t.Fatalf("failed to unmarshal yaml recordings: %v", err)
		}

		// Assert that raw YAML string does not contain nulls, empty collections, or default-zero/false properties
		yamlStr := string(data)
		if strings.Contains(yamlStr, "null") {
			t.Errorf("YAML contains unexpected 'null' fields:\n%s", yamlStr)
		}
		if strings.Contains(yamlStr, "candidatecount") {
			t.Errorf("YAML contains unexpected 'candidatecount' field:\n%s", yamlStr)
		}
		if strings.Contains(yamlStr, "thoughtsignature") {
			t.Errorf("YAML contains unexpected 'thoughtsignature' field:\n%s", yamlStr)
		}
		if strings.Contains(yamlStr, "thought: false") {
			t.Errorf("YAML contains unexpected 'thought: false' field:\n%s", yamlStr)
		}
		if strings.Contains(yamlStr, "tools:") {
			t.Errorf("YAML contains unexpected 'tools:' field:\n%s", yamlStr)
		}

		if len(recordings.Recordings) != 2 {
			t.Fatalf("expected exactly 2 recordings, got %d", len(recordings.Recordings))
		}

		// First recording: LLM
		r1 := recordings.Recordings[0]
		if r1.UserMessageIndex != 0 {
			t.Errorf("expected index 0, got %d", r1.UserMessageIndex)
		}
		if r1.AgentName != "test_agent" {
			t.Errorf("expected agent 'test_agent', got %q", r1.AgentName)
		}
		if r1.LLMRecording == nil {
			t.Fatal("expected LLM recording to be present")
		}
		if r1.LLMRecording.LLMRequest.Model != "gemini-2.0-flash" {
			t.Errorf("expected model 'gemini-2.0-flash', got %q", r1.LLMRecording.LLMRequest.Model)
		}

		// Second recording: Tool
		r2 := recordings.Recordings[1]
		if r2.ToolRecording == nil {
			t.Fatal("expected Tool recording to be present")
		}
		if r2.ToolRecording.ToolCall.Name != "test_tool" {
			t.Errorf("expected tool name 'test_tool', got %q", r2.ToolRecording.ToolCall.Name)
		}
		if r2.ToolRecording.ToolResponse.Response["result"] != "test_success" {
			t.Errorf("expected response 'test_success', got %v", r2.ToolRecording.ToolResponse.Response["result"])
		}
	})

	t.Run("PathValidation", func(t *testing.T) {
		tempDir := t.TempDir()
		safeDir := filepath.Join(tempDir, "safe")
		_ = os.Mkdir(safeDir, 0o755)

		p, mockSession, _ := setup(t, safeDir)

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
				name:        "InvalidPath_AbsoluteOutside",
				dir:         "/etc",
				expectError: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mockSession.State().Set("_adk_recordings_config", map[string]any{
					"dir":                tt.dir,
					"user_message_index": 0,
				})
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				invContext := &MockInvocationContext{
					session:      mockSession,
					invocationID: "test-invocation-" + tt.name,
				}

				_, err = p.BeforeRunCallback()(invContext)
				if tt.expectError {
					if err == nil {
						t.Errorf("expected path error for %q, got nil", tt.dir)
					}
				} else {
					if err != nil {
						t.Errorf("unexpected path error for %q: %v", tt.dir, err)
					}
				}
			})
		}
	})

	t.Run("MultiTurnAppendAndDeduplication", func(t *testing.T) {
		tempDir := t.TempDir()
		p, mockSession, _ := setup(t, tempDir)

		// --- Turn 0 ---
		err := mockSession.State().Set("_adk_recordings_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		invContext1 := &MockInvocationContext{session: mockSession, invocationID: "inv-0"}
		_, _ = p.BeforeRunCallback()(invContext1)

		cbContext1 := &MockCallbackContext{state: mockSession.State(), invocationID: "inv-0", agentName: "test_agent"}
		_, _ = p.BeforeModelCallback()(cbContext1, &model.LLMRequest{Model: "model-0"})
		_, _ = p.AfterModelCallback()(cbContext1, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Response 0"}}}, Partial: false}, nil)
		p.AfterRunCallback()(invContext1)

		// --- Turn 1 Append ---
		err = mockSession.State().Set("_adk_recordings_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 1,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		invContext2 := &MockInvocationContext{session: mockSession, invocationID: "inv-1"}
		_, _ = p.BeforeRunCallback()(invContext2)

		cbContext2 := &MockCallbackContext{state: mockSession.State(), invocationID: "inv-1", agentName: "test_agent"}
		_, _ = p.BeforeModelCallback()(cbContext2, &model.LLMRequest{Model: "model-1"})
		_, _ = p.AfterModelCallback()(cbContext2, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Response 1"}}}, Partial: false}, nil)
		p.AfterRunCallback()(invContext2)

		// Read and verify both turns are recorded
		filePath := filepath.Join(tempDir, "generated-recordings.yaml")
		data, err := os.ReadFile(filePath)
		if err != nil {
			t.Fatalf("failed to read recordings file: %v", err)
		}

		var recs recording.Recordings
		_ = yaml.Unmarshal(data, &recs)
		if len(recs.Recordings) != 2 {
			t.Fatalf("expected 2 recordings after multi-turn run, got %d", len(recs.Recordings))
		}
		if recs.Recordings[0].UserMessageIndex != 0 || recs.Recordings[1].UserMessageIndex != 1 {
			t.Errorf("unexpected sequence indexes: turn 0 = %d, turn 1 = %d", recs.Recordings[0].UserMessageIndex, recs.Recordings[1].UserMessageIndex)
		}

		// --- Same-Turn Deduplication/Overwrite ---
		err = mockSession.State().Set("_adk_recordings_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0, // Overwrite turn 0
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		invContext3 := &MockInvocationContext{session: mockSession, invocationID: "inv-0-new"}
		_, _ = p.BeforeRunCallback()(invContext3)

		cbContext3 := &MockCallbackContext{state: mockSession.State(), invocationID: "inv-0-new", agentName: "test_agent"}
		_, _ = p.BeforeModelCallback()(cbContext3, &model.LLMRequest{Model: "model-0"})
		_, _ = p.AfterModelCallback()(cbContext3, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Response 0 Updated"}}}, Partial: false}, nil)
		p.AfterRunCallback()(invContext3)

		data, _ = os.ReadFile(filePath)
		_ = yaml.Unmarshal(data, &recs)

		// Verify we still have exactly 2 recordings, but the turn 0 content is updated!
		if len(recs.Recordings) != 2 {
			t.Fatalf("expected exactly 2 recordings after deduplication, got %d", len(recs.Recordings))
		}
		var foundTurn0 bool
		for _, r := range recs.Recordings {
			if r.UserMessageIndex == 0 {
				foundTurn0 = true
				gotText := r.LLMRecording.LLMResponses[0].Content.Parts[0].Text
				if gotText != "Response 0 Updated" {
					t.Errorf("expected overwritten text 'Response 0 Updated', got %q", gotText)
				}
			}
		}
		if !foundTurn0 {
			t.Error("turn 0 recording was unexpectedly lost during deduplication")
		}
	})

	t.Run("StreamingChunkAccumulation", func(t *testing.T) {
		tempDir := t.TempDir()
		p, mockSession, _ := setup(t, tempDir)

		err := mockSession.State().Set("_adk_recordings_config", map[string]any{
			"dir":                tempDir,
			"user_message_index": 0,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		invContext := &MockInvocationContext{session: mockSession, invocationID: "inv-stream"}
		_, _ = p.BeforeRunCallback()(invContext)

		cbContext := &MockCallbackContext{state: mockSession.State(), invocationID: "inv-stream", agentName: "test_agent"}
		_, _ = p.BeforeModelCallback()(cbContext, &model.LLMRequest{Model: "model-stream"})

		// 3 Partial responses
		_, _ = p.AfterModelCallback()(cbContext, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Chunk 1"}}}, Partial: true}, nil)
		_, _ = p.AfterModelCallback()(cbContext, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Chunk 2"}}}, Partial: true}, nil)
		_, _ = p.AfterModelCallback()(cbContext, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Chunk 3"}}}, Partial: true}, nil)

		// 1 Final non-partial response
		_, _ = p.AfterModelCallback()(cbContext, &model.LLMResponse{Content: &genai.Content{Parts: []*genai.Part{{Text: "Final Chunk"}}}, Partial: false}, nil)
		p.AfterRunCallback()(invContext)

		filePath := filepath.Join(tempDir, "generated-recordings.yaml")
		data, _ := os.ReadFile(filePath)
		var recs recording.Recordings
		_ = yaml.Unmarshal(data, &recs)

		if len(recs.Recordings) != 1 {
			t.Fatalf("expected 1 recording, got %d", len(recs.Recordings))
		}
		llmRec := recs.Recordings[0].LLMRecording
		if len(llmRec.LLMResponses) != 4 {
			t.Fatalf("expected all 4 stream chunks to be accumulated, got %d", len(llmRec.LLMResponses))
		}
		if llmRec.LLMResponses[3].Content.Parts[0].Text != "Final Chunk" {
			t.Errorf("unexpected final response chunk text: %q", llmRec.LLMResponses[3].Content.Parts[0].Text)
		}
	})

	t.Run("DisabledBypass", func(t *testing.T) {
		tempDir := t.TempDir()
		p, mockSession, _ := setup(t, tempDir)

		// We don't set any recording configs in mockSession state

		invContext := &MockInvocationContext{session: mockSession, invocationID: "inv-bypass"}
		_, err := p.BeforeRunCallback()(invContext)
		if err != nil {
			t.Fatalf("beforeRun should not error on empty config, got: %v", err)
		}

		cbContext := &MockCallbackContext{state: mockSession.State(), invocationID: "inv-bypass", agentName: "test_agent"}
		_, err = p.BeforeModelCallback()(cbContext, &model.LLMRequest{Model: "model-bypass"})
		if err != nil {
			t.Fatalf("beforeModel should not error on empty config, got: %v", err)
		}

		p.AfterRunCallback()(invContext)

		filePath := filepath.Join(tempDir, "generated-recordings.yaml")
		if _, err := os.Stat(filePath); err == nil {
			t.Error("recordings file was unexpectedly created even though the plugin was disabled")
		}
	})
}

// --- Mock Implementations ---

type MockState struct {
	data map[string]any
}

func (m *MockState) Set(key string, val any) error { m.data[key] = val; return nil }
func (m *MockState) Get(key string) (any, error)   { return m.data[key], nil }
func (m *MockState) All() iter.Seq2[string, any]   { return nil }

type MockSession struct {
	state *MockState
}

func (m *MockSession) ID() string                { return "mock-session-id" }
func (m *MockSession) AppName() string           { return "mock-app" }
func (m *MockSession) UserID() string            { return "mock-user" }
func (m *MockSession) State() session.State      { return m.state }
func (m *MockSession) Events() session.Events    { return nil }
func (m *MockSession) LastUpdateTime() time.Time { return time.Now() }

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
func (m *MockInvocationContext) RunConfig() *agent.RunConfig                             { return nil }
func (m *MockInvocationContext) EndInvocation()                                          {}
func (m *MockInvocationContext) Ended() bool                                             { return false }
func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext { return m }
func (m *MockInvocationContext) Value(key any) any                                       { return nil }
func (m *MockInvocationContext) ResumedInput(string) (any, bool)                         { return nil, false }
func (m *MockInvocationContext) Deadline() (deadline time.Time, ok bool)                 { return time.Time{}, false }
func (m *MockInvocationContext) Done() <-chan struct{}                                   { return nil }
func (m *MockInvocationContext) Err() error                                              { return nil }

type MockCallbackContext struct {
	state        session.State
	invocationID string
	agentName    string
}

func (m *MockCallbackContext) State() session.State                                 { return m.state }
func (m *MockCallbackContext) ReadonlyState() session.ReadonlyState                 { return m.state }
func (m *MockCallbackContext) InvocationID() string                                 { return m.invocationID }
func (m *MockCallbackContext) AgentName() string                                    { return m.agentName }
func (m *MockCallbackContext) AppName() string                                      { return "mock-app" }
func (m *MockCallbackContext) Branch() string                                       { return "" }
func (m *MockCallbackContext) SessionID() string                                    { return "mock-session-id" }
func (m *MockCallbackContext) UserID() string                                       { return "mock-user" }
func (m *MockCallbackContext) UserContent() *genai.Content                          { return nil }
func (m *MockCallbackContext) Artifacts() agent.Artifacts                           { return nil }
func (m *MockCallbackContext) Value(key any) any                                    { return nil }
func (m *MockCallbackContext) Deadline() (deadline time.Time, ok bool)              { return time.Time{}, false }
func (m *MockCallbackContext) Done() <-chan struct{}                                { return nil }
func (m *MockCallbackContext) Err() error                                           { return nil }
func (m *MockCallbackContext) Actions() *session.EventActions                       { return nil }
func (m *MockCallbackContext) FunctionCallID() string                               { return "" }
func (m *MockCallbackContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (m *MockCallbackContext) RequestConfirmation(hint string, payload any) error {
	return fmt.Errorf("RequestConfirmation() is not supported for MockCallbackContext")
}

func (m *MockCallbackContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, fmt.Errorf("SearchMemory() is not supported for MockCallbackContext")
}

// InvocationContext surface (Context embeds InvocationContext after
// context unification).
func (m *MockCallbackContext) Agent() agent.Agent                         { return nil }
func (m *MockCallbackContext) Memory() agent.Memory                       { return nil }
func (m *MockCallbackContext) Session() session.Session                   { return nil }
func (m *MockCallbackContext) RunConfig() *agent.RunConfig                { return nil }
func (m *MockCallbackContext) Ended() bool                                { return false }
func (m *MockCallbackContext) ResumedInput(string) (any, bool)            { return nil, false }
func (m *MockCallbackContext) Path() string                               { return "" }
func (m *MockCallbackContext) RunID() string                              { return "" }
func (m *MockCallbackContext) NodeScheduler() agent.NodeScheduler         { return nil }
func (m *MockCallbackContext) InvocationContext() agent.InvocationContext { return nil }
func (m *MockCallbackContext) WithContext(context.Context) agent.Context  { return m }

var _ agent.CallbackContext = (*MockCallbackContext)(nil)

type MockToolContext struct {
	state          session.State
	invocationID   string
	agentName      string
	functionCallID string
}

func (m *MockToolContext) State() session.State                 { return m.state }
func (m *MockToolContext) ReadonlyState() session.ReadonlyState { return m.state }
func (m *MockToolContext) InvocationID() string                 { return m.invocationID }
func (m *MockToolContext) AgentName() string                    { return m.agentName }
func (m *MockToolContext) FunctionCallID() string               { return m.functionCallID }
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

// InvocationContext surface (Context embeds InvocationContext after
// context unification).
func (m *MockToolContext) Agent() agent.Agent                         { return nil }
func (m *MockToolContext) Memory() agent.Memory                       { return nil }
func (m *MockToolContext) Session() session.Session                   { return nil }
func (m *MockToolContext) RunConfig() *agent.RunConfig                { return nil }
func (m *MockToolContext) Ended() bool                                { return false }
func (m *MockToolContext) ResumedInput(string) (any, bool)            { return nil, false }
func (m *MockToolContext) Path() string                               { return "" }
func (m *MockToolContext) RunID() string                              { return "" }
func (m *MockToolContext) NodeScheduler() agent.NodeScheduler         { return nil }
func (m *MockToolContext) InvocationContext() agent.InvocationContext { return nil }
func (m *MockToolContext) WithContext(context.Context) agent.Context  { return m }

type MockTool struct {
	NameVal string
}

func (m *MockTool) Name() string        { return m.NameVal }
func (m *MockTool) Description() string { return "mock tool" }
func (m *MockTool) IsLongRunning() bool { return false }
