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

package vertexai

import (
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TestRawEventWireKeysAreCamelCase pins the key names raw_event carries. The
// field is shared with other ADK runtimes, whose models bind camelCase
// aliases and ignore unknown keys, so a PascalCase key is not a cosmetic
// difference: it decodes to a default value with no error.
func TestRawEventWireKeysAreCamelCase(t *testing.T) {
	event := &session.Event{
		LLMResponse:        model.LLMResponse{Content: genai.NewContentFromText("hi", genai.RoleModel)},
		ID:                 "evt-1",
		Timestamp:          time.Unix(1733040000, 0).UTC(),
		InvocationID:       "inv-1",
		Branch:             "root.planner",
		Author:             "planner",
		IsolationScope:     "approval_subflow",
		LongRunningToolIDs: []string{"tool-1"},
		Routes:             []string{"reviewer"},
		RequestedInput:     &session.RequestInput{InterruptID: "approval-1", Message: "Approve?"},
		Output:             map[string]any{"summary": "ok"},
		NodeInfo:           &session.NodeInfo{Path: "wf/planner", MessageAsOutput: true},
		Actions:            session.EventActions{StateDelta: map[string]any{"stage": "drafted"}},
	}

	raw, err := eventToRawEvent(event)
	if err != nil {
		t.Fatalf("eventToRawEvent() error = %v", err)
	}
	got := make([]string, 0, len(raw.GetFields()))
	for k := range raw.GetFields() {
		got = append(got, k)
	}
	sort.Strings(got)

	want := []string{
		"actions", "author", "branch", "content", "id", "invocationId",
		"isolationScope", "longRunningToolIds", "nodeInfo", "output",
		"requestedInput", "routes", "timestamp",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("raw_event wire keys mismatch (-want +got):\n%s", diff)
	}
}

// pythonRawEvent is a raw_event payload as emitted by adk-python, whose Event
// model dumps camelCase aliases and carries its timestamp as a float. Captured
// from an instrumented run rather than written by hand, so it reflects the
// encoding actually found in shared session corpora.
const pythonRawEvent = `{
  "actions": {"artifactDelta": {"artifact": 1.0}, "escalate": true,
              "requestedAuthConfigs": {}, "requestedToolConfirmations": {},
              "skipSummarization": true, "stateDelta": {"state": "updated"},
              "transferToAgent": "test-agent"},
  "author": "user",
  "branch": "test-branch",
  "content": {"parts": [{"text": "Hello, raw_event!"},
                        {"functionCall": {"args": {"arg1": "value1"},
                                          "id": "test-id", "name": "test-function"}}],
              "role": "user"},
  "customMetadata": {"custom-key": "custom-value"},
  "errorCode": "400",
  "errorMessage": "test error",
  "groundingMetadata": {"retrievalQueries": ["query2"], "webSearchQueries": ["query1"]},
  "id": "test-id",
  "interrupted": true,
  "invocationId": "test-invocation-id",
  "isolationScope": "scope-1",
  "longRunningToolIds": ["test-tool-id-1", "test-tool-id-2"],
  "nodeInfo": {"messageAsOutput": true, "outputFor": ["wf/A@1/B@1", "wf/A@1"],
               "path": "wf/A@1/B@1"},
  "output": {"count": 3.0, "result": "ok"},
  "partial": true,
  "timestamp": 1733040000.0,
  "turnComplete": true,
  "usageMetadata": {"candidatesTokenCount": 50.0, "promptTokenCount": 100.0,
                    "totalTokenCount": 200.0}
}`

// TestEventFromRawEventDecodesOtherRuntimeDump reads a raw_event written by a
// different ADK runtime. Sessions are shared across runtimes, so this is a
// read path that runs in production, not a hypothetical.
func TestEventFromRawEventDecodesOtherRuntimeDump(t *testing.T) {
	raw := &structpb.Struct{}
	if err := raw.UnmarshalJSON([]byte(pythonRawEvent)); err != nil {
		t.Fatalf("test fixture is not valid JSON: %v", err)
	}

	got, err := eventFromRawEvent(raw)
	if err != nil {
		t.Fatalf("eventFromRawEvent() error = %v", err)
	}

	// The timestamp arrives as a float; decoding must not abort on it and
	// leave every later field unset.
	if want := time.Unix(1733040000, 0).UTC(); !got.Timestamp.UTC().Equal(want) {
		t.Errorf("Timestamp = %v, want %v", got.Timestamp.UTC(), want)
	}

	scalars := []struct {
		name string
		got  any
		want any
	}{
		{"Author", got.Author, "user"},
		{"Branch", got.Branch, "test-branch"},
		{"InvocationID", got.InvocationID, "test-invocation-id"},
		{"IsolationScope", got.IsolationScope, "scope-1"},
		{"ErrorCode", got.ErrorCode, "400"},
		{"ErrorMessage", got.ErrorMessage, "test error"},
		{"Partial", got.Partial, true},
		{"TurnComplete", got.TurnComplete, true},
		{"Interrupted", got.Interrupted, true},
		{"Actions.TransferToAgent", got.Actions.TransferToAgent, "test-agent"},
		{"Actions.Escalate", got.Actions.Escalate, true},
		{"Actions.SkipSummarization", got.Actions.SkipSummarization, true},
	}
	for _, s := range scalars {
		if diff := cmp.Diff(s.want, s.got); diff != "" {
			t.Errorf("%s mismatch (-want +got):\n%s", s.name, diff)
		}
	}

	wantNode := &session.NodeInfo{
		Path:            "wf/A@1/B@1",
		MessageAsOutput: true,
		OutputFor:       []string{"wf/A@1/B@1", "wf/A@1"},
	}
	if diff := cmp.Diff(wantNode, got.NodeInfo); diff != "" {
		t.Errorf("NodeInfo mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"count": float64(3), "result": "ok"}, got.Output); diff != "" {
		t.Errorf("Output mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"state": "updated"}, got.Actions.StateDelta); diff != "" {
		t.Errorf("Actions.StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"test-tool-id-1", "test-tool-id-2"}, got.LongRunningToolIDs); diff != "" {
		t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
	}

	if got.Content == nil || len(got.Content.Parts) != 2 {
		t.Fatalf("Content = %v, want 2 parts", got.Content)
	}
	if want := "Hello, raw_event!"; got.Content.Parts[0].Text != want {
		t.Errorf("Content.Parts[0].Text = %q, want %q", got.Content.Parts[0].Text, want)
	}
	fc := got.Content.Parts[1].FunctionCall
	if fc == nil {
		t.Fatal("Content.Parts[1].FunctionCall = nil, want a decoded function call")
	}
	if fc.Name != "test-function" || fc.ID != "test-id" {
		t.Errorf("FunctionCall = {Name:%q ID:%q}, want {Name:%q ID:%q}",
			fc.Name, fc.ID, "test-function", "test-id")
	}
}
