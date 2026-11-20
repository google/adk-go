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

package llminternal

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// An original call without an ID and with an empty args map is written with
// "args": {} and no "id" key, like a call with an ID minus the ID: adk-python
// leaves an unset id out of the dump.
func TestGenerateRequestConfirmationEvent_EmptyArgsWithoutID(t *testing.T) {
	ic := &mockInvocationContext{invocationID: "inv_1", agentName: "agent_1"}
	call := &genai.FunctionCall{Name: "tool_b", Args: map[string]any{}}
	callEvent := &session.Event{LLMResponse: model.LLMResponse{Content: &genai.Content{
		Parts: []*genai.Part{{FunctionCall: call}},
	}}}
	responseEvent := &session.Event{Actions: session.EventActions{
		RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{"": {Hint: "h"}},
	}}

	ev := generateRequestConfirmationEvent(ic, callEvent, responseEvent)
	if ev == nil || ev.Content == nil || len(ev.Content.Parts) != 1 || ev.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("generateRequestConfirmationEvent() = %+v, want one function call part", ev)
	}
	data, err := json.Marshal(ev.Content.Parts[0].FunctionCall.Args["originalFunctionCall"])
	if err != nil {
		t.Fatalf("json.Marshal(originalFunctionCall) error = %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(originalFunctionCall) error = %v", err)
	}
	want := map[string]any{"args": map[string]any{}, "name": "tool_b"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("originalFunctionCall JSON mismatch (-want +got):\n%s", diff)
	}
}
