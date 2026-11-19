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

package runner_test

import (
	"errors"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// Each refusal of a run's message is one adk-python raises as ValueError
// (runners.py), so a caller matches all of them with errors.Is and the
// message keeps its own text.
func TestRun_MessageRefusalsMatchErrInvalidMessage(t *testing.T) {
	tests := []struct {
		name      string
		root      iidRootKind
		resumable bool
		seed      []*session.Event
		msg       *genai.Content
		opts      []runner.RunOption
		wantErr   string
	}{
		{
			name:      "function_response_without_id",
			root:      iidClassicRoot,
			resumable: true,
			seed:      []*session.Event{seedUser("inv_a", "first")},
			msg:       iidResponses(""),
			wantErr:   "function response id is required to resume an invocation",
		},
		{
			name:    "text_and_function_response",
			root:    iidNodeRoot,
			seed:    []*session.Event{seedUser("inv_a", "first"), resumeSeed("inv_a", "root", fnCallContent("fc-a", "tool", nil))},
			msg:     &genai.Content{Role: genai.RoleUser, Parts: append(iidResponses("fc-a").Parts, &genai.Part{Text: "hi"})},
			wantErr: "message cannot contain both function responses and text: function responses resume an existing invocation while text starts a new one",
		},
		{
			name:      "function_call_not_found",
			root:      iidClassicRoot,
			resumable: true,
			seed:      []*session.Event{seedUser("inv_a", "first")},
			msg:       iidResponses("fc-missing"),
			wantErr:   "function call not found for 1 function response ids: each function response ID must match a function call in the session history",
		},
		{
			name:    "responses_of_two_invocations",
			root:    iidLLMRoot,
			seed:    iidTwoInvocations(),
			msg:     iidResponses("fc-a", "fc-b"),
			opts:    []runner.RunOption{runner.WithInvocationID("inv_wrong")},
			wantErr: "function responses resolve to multiple invocations (2): all function responses in a single message must belong to the same invocation",
		},
		{
			name:    "function_call_in_message",
			root:    iidClassicRoot,
			msg:     &genai.Content{Role: genai.RoleUser, Parts: fnCallContent("fc-u", "tool", nil).Parts},
			wantErr: "user message cannot contain function calls",
		},
		{
			name:    "no_parts",
			root:    iidClassicRoot,
			msg:     &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{}},
			wantErr: "no parts in the new_message",
		},
		{
			name:      "resume_in_session_without_events",
			root:      iidClassicRoot,
			resumable: true,
			opts:      []runner.RunOption{runner.WithInvocationID("inv-1")},
			wantErr:   "session s has no events to resume",
		},
		{
			name:      "resume_without_stored_user_message",
			root:      iidClassicRoot,
			resumable: true,
			seed:      []*session.Event{seedUser("inv-other", "hi")},
			opts:      []runner.RunOption{runner.WithInvocationID("inv-1")},
			wantErr:   "no user message available for resuming invocation: inv-1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := session.InMemoryService()
			seedResumeSession(t, svc, tt.seed...)
			r := newIIDRunner(t, tt.root(t), svc, tt.resumable)

			_, errs := collectRun(r.Run(t.Context(), nodeTestUser, nodeTestSession, tt.msg, agent.RunConfig{}, tt.opts...))

			if len(errs) != 1 {
				t.Fatalf("Run() errors = %q, want [%q]", errorTexts(errs), tt.wantErr)
			}
			if got := errs[0].Error(); got != tt.wantErr {
				t.Errorf("Run() error = %q, want %q", got, tt.wantErr)
			}
			if !errors.Is(errs[0], runner.ErrInvalidMessage) {
				t.Errorf("errors.Is(%q, runner.ErrInvalidMessage) = false, want true", errs[0])
			}
		})
	}
}
