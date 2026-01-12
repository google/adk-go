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

package mcptoolset_test

import (
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
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

func TestSessionStateMetadataProvider(t *testing.T) {
	testCases := []struct {
		name      string
		stateData map[string]any
		stateKeys map[string]string
		want      map[string]any
	}{
		{
			name:      "empty state keys",
			stateKeys: nil,
			want:      nil,
		},
		{
			name: "read from state",
			stateData: map[string]any{
				"temp:trace_id":   "trace-123",
				"temp:request_id": "req-456",
			},
			stateKeys: map[string]string{
				"temp:trace_id":   "x-trace-id",
				"temp:request_id": "x-request-id",
			},
			want: map[string]any{
				"x-trace-id":   "trace-123",
				"x-request-id": "req-456",
			},
		},
		{
			name: "missing state key ignored",
			stateData: map[string]any{
				"temp:trace_id": "trace-123",
			},
			stateKeys: map[string]string{
				"temp:trace_id": "x-trace-id",
				"temp:missing":  "x-missing",
			},
			want: map[string]any{
				"x-trace-id": "trace-123",
			},
		},
		{
			name:      "no matching keys returns nil",
			stateData: map[string]any{},
			stateKeys: map[string]string{
				"temp:missing": "x-missing",
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockCtx := &mockToolContext{
				Context: context.Background(),
				state:   &mockState{data: tc.stateData},
			}

			provider := mcptoolset.SessionStateMetadataProvider(tc.stateKeys)
			got := provider(mockCtx)

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("SessionStateMetadataProvider() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestChainMetadataProviders(t *testing.T) {
	testCases := []struct {
		name      string
		providers []mcptoolset.MetadataProvider
		want      map[string]any
	}{
		{
			name:      "no providers",
			providers: nil,
			want:      nil,
		},
		{
			name: "nil provider in chain",
			providers: []mcptoolset.MetadataProvider{
				nil,
				func(ctx tool.Context) map[string]any {
					return map[string]any{"key": "value"}
				},
			},
			want: map[string]any{"key": "value"},
		},
		{
			name: "later provider overrides earlier",
			providers: []mcptoolset.MetadataProvider{
				func(ctx tool.Context) map[string]any {
					return map[string]any{"key1": "first", "key2": "first"}
				},
				func(ctx tool.Context) map[string]any {
					return map[string]any{"key2": "second", "key3": "second"}
				},
			},
			want: map[string]any{"key1": "first", "key2": "second", "key3": "second"},
		},
		{
			name: "all nil returns nil",
			providers: []mcptoolset.MetadataProvider{
				func(ctx tool.Context) map[string]any { return nil },
				func(ctx tool.Context) map[string]any { return nil },
			},
			want: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockCtx := &mockToolContext{Context: context.Background(), state: &mockState{}}

			provider := mcptoolset.ChainMetadataProviders(tc.providers...)
			got := provider(mockCtx)

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ChainMetadataProviders() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
