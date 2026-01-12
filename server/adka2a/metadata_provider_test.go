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

package adka2a_test

import (
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// mockToolContext implements tool.Context for testing
type mockToolContext struct {
	context.Context
	state *mockState
}

var _ tool.Context = (*mockToolContext)(nil)

func (m *mockToolContext) FunctionCallID() string                                               { return "test-function-call-id" }
func (m *mockToolContext) Actions() *session.EventActions                                       { return nil }
func (m *mockToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) { return nil, nil }
func (m *mockToolContext) UserContent() *genai.Content                                          { return nil }
func (m *mockToolContext) InvocationID() string                                                 { return "test-invocation-id" }
func (m *mockToolContext) AgentName() string                                                    { return "test-agent" }
func (m *mockToolContext) ReadonlyState() session.ReadonlyState                                 { return m.state }
func (m *mockToolContext) UserID() string                                                       { return "test-user" }
func (m *mockToolContext) AppName() string                                                      { return "test-app" }
func (m *mockToolContext) SessionID() string                                                    { return "test-session" }
func (m *mockToolContext) Branch() string                                                       { return "" }
func (m *mockToolContext) Artifacts() agent.Artifacts                                           { return &mockArtifacts{} }
func (m *mockToolContext) State() session.State                                                 { return m.state }

type mockArtifacts struct{}

func (m *mockArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	return nil, nil
}
func (m *mockArtifacts) List(context.Context) (*artifact.ListResponse, error)                  { return nil, nil }
func (m *mockArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) { return nil, nil }
func (m *mockArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return nil, nil
}

type mockState struct {
	data map[string]any
}

func (m *mockState) Get(key string) (any, error) {
	if v, ok := m.data[key]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

func (m *mockState) Set(key string, val any) error {
	if m.data == nil {
		m.data = make(map[string]any)
	}
	m.data[key] = val
	return nil
}

func (m *mockState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range m.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

func TestA2AMetadataProvider(t *testing.T) {
	testCases := []struct {
		name        string
		a2aMeta     *adka2a.A2AMetadata
		forwardKeys []string
		want        map[string]any
	}{
		{
			name:    "no a2a metadata in context",
			a2aMeta: nil,
			want:    nil,
		},
		{
			name: "forward all metadata",
			a2aMeta: &adka2a.A2AMetadata{
				TaskID:          "task-123",
				ContextID:       "ctx-456",
				RequestMetadata: map[string]any{"trace_id": "trace-789"},
				MessageMetadata: map[string]any{"correlation_id": "corr-abc"},
			},
			forwardKeys: nil,
			want: map[string]any{
				"a2a:task_id":    "task-123",
				"a2a:context_id": "ctx-456",
				"trace_id":       "trace-789",
				"correlation_id": "corr-abc",
			},
		},
		{
			name: "forward only specific keys",
			a2aMeta: &adka2a.A2AMetadata{
				TaskID:          "task-123",
				ContextID:       "ctx-456",
				RequestMetadata: map[string]any{"trace_id": "trace-789", "ignored": "value"},
				MessageMetadata: map[string]any{"correlation_id": "corr-abc"},
			},
			forwardKeys: []string{"trace_id"},
			want: map[string]any{
				"a2a:task_id":    "task-123",
				"a2a:context_id": "ctx-456",
				"trace_id":       "trace-789",
			},
		},
		{
			name: "task id only",
			a2aMeta: &adka2a.A2AMetadata{
				TaskID: "task-123",
			},
			forwardKeys: nil,
			want: map[string]any{
				"a2a:task_id": "task-123",
			},
		},
		{
			name:        "empty metadata returns nil",
			a2aMeta:     &adka2a.A2AMetadata{},
			forwardKeys: nil,
			want:        nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			if tc.a2aMeta != nil {
				ctx = adka2a.ContextWithA2AMetadata(ctx, tc.a2aMeta)
			}

			mockCtx := &mockToolContext{Context: ctx, state: &mockState{}}
			provider := adka2a.A2AMetadataProvider(tc.forwardKeys)
			got := provider(mockCtx)

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("A2AMetadataProvider() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
