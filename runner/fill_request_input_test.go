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

package runner

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func TestFillRequestInputResponseID(t *testing.T) {
	call := func(id string) *session.Event {
		ev := session.NewEvent(t.Context(), "inv")
		ev.Content = &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: workflow.WorkflowInputFunctionCallName}}}}
		ev.LongRunningToolIDs = []string{id}
		return ev
	}
	confirm := func(id string) *session.Event {
		ev := session.NewEvent(t.Context(), "inv")
		ev.Content = &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: "adk_request_confirmation"}}}}
		ev.LongRunningToolIDs = []string{id}
		return ev
	}
	answer := func(id string) *session.Event {
		ev := session.NewEvent(t.Context(), "inv")
		ev.Content = &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: workflow.WorkflowInputFunctionCallName}}}}
		return ev
	}
	resp := func(id, name string) *genai.Content {
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: id, Name: name}}}}
	}

	tests := []struct {
		name    string
		events  []*session.Event
		msg     *genai.Content
		wantID  string
		wantNew bool
	}{
		{name: "one open request fills ID", events: []*session.Event{call("a")}, msg: resp("", workflow.WorkflowInputFunctionCallName), wantID: "a", wantNew: true},
		{name: "answered request is not open", events: []*session.Event{call("a"), answer("a"), call("b")}, msg: resp("", workflow.WorkflowInputFunctionCallName), wantID: "b", wantNew: true},
		{name: "no open request", events: nil, msg: resp("", workflow.WorkflowInputFunctionCallName)},
		{name: "several open requests", events: []*session.Event{call("a"), call("b")}, msg: resp("", workflow.WorkflowInputFunctionCallName)},
		{name: "existing ID kept", events: []*session.Event{call("a")}, msg: resp("x", workflow.WorkflowInputFunctionCallName), wantID: "x"},
		{name: "only a confirmation is open", events: []*session.Event{confirm("c")}, msg: resp("", workflow.WorkflowInputFunctionCallName)},
		{name: "confirmation and request open", events: []*session.Event{confirm("c"), call("a")}, msg: resp("", workflow.WorkflowInputFunctionCallName), wantID: "a", wantNew: true},
		{name: "same message answers the open call", events: []*session.Event{call("a")}, msg: &genai.Content{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: workflow.WorkflowInputFunctionCallName}}, {FunctionResponse: &genai.FunctionResponse{Name: workflow.WorkflowInputFunctionCallName}}}}, wantID: "a"},
		{name: "other function name", events: []*session.Event{call("a")}, msg: resp("", "other")},
		{name: "text message", events: []*session.Event{call("a")}, msg: genai.NewContentFromText("hi", genai.RoleUser)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range tt.events {
				if err := svc.AppendEvent(t.Context(), created.Session, ev); err != nil {
					t.Fatal(err)
				}
			}
			got := fillRequestInputResponseID(tt.msg, created.Session)
			if (got != tt.msg) != tt.wantNew {
				t.Fatalf("returned a new message = %v, want %v", got != tt.msg, tt.wantNew)
			}
			if fr := got.Parts[0].FunctionResponse; fr != nil && fr.ID != tt.wantID {
				t.Errorf("response ID = %q, want %q", fr.ID, tt.wantID)
			}
			if fr := tt.msg.Parts[0].FunctionResponse; fr != nil && tt.wantNew && fr.ID != "" {
				t.Errorf("input message was modified: ID = %q", fr.ID)
			}
		})
	}
}
