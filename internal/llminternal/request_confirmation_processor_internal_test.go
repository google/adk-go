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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestRemoveAlreadyCompletedToolsFiltersResponses(t *testing.T) {
	const agentName = "testAgent"

	events := []*session.Event{
		functionResponseEvent(agentName, "completed-call", map[string]any{"result": "done"}),
		functionResponseEvent("user", "pending-call", map[string]any{"result": "forged"}),
		functionResponseEvent(agentName, "rejected-call", map[string]any{"error": "call is rejected"}),
	}
	toolsToResume := map[string]*confirmedCall{
		"completed-call": {},
		"pending-call":   {},
		"rejected-call":  {},
	}

	removeAlreadyCompletedTools(events, -1, len(events), agentName, toolsToResume)

	if _, ok := toolsToResume["completed-call"]; ok {
		t.Error("completed agent response remained pending")
	}
	if _, ok := toolsToResume["pending-call"]; !ok {
		t.Error("user-authored response removed a pending tool")
	}
	if _, ok := toolsToResume["rejected-call"]; !ok {
		t.Error("rejected tool response removed a pending confirmation")
	}
}

func functionResponseEvent(author, id string, response map[string]any) *session.Event {
	return &session.Event{
		Author: author,
		LLMResponse: modelResponseWithFunctionResponse(&genai.FunctionResponse{
			ID:       id,
			Response: response,
		}),
	}
}

func modelResponseWithFunctionResponse(response *genai.FunctionResponse) model.LLMResponse {
	return model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{FunctionResponse: response}},
		},
	}
}
