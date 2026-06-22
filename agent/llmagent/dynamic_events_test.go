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

package llmagent_test

import (
	"iter"
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/session"
)

func TestSessionEvents_YieldedPresence(t *testing.T) {
	// Create a custom agent that yields an event and then checks the session events.
	customAgent, err := agent.New(agent.Config{
		Name: "test_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// 1. Yield a test event
				testEvent := session.NewEvent(ctx.InvocationID())
				testEvent.Content = genai.NewContentFromText("Initial test event", genai.RoleModel)
				if !yield(testEvent, nil) {
					return
				}

				// 2. Check if the event is in the session
				var found bool
				if ctx.Session() != nil {
					for e := range ctx.Session().Events().All() {
						if e.Content != nil && len(e.Content.Parts) > 0 {
							if e.Content.Parts[0].Text == "Initial test event" {
								found = true
								break
							}
						}
					}
				}

				// 3. Yield the result of the check
				resultEvent := session.NewEvent(ctx.InvocationID())
				if found {
					resultEvent.Content = genai.NewContentFromText("Found initial event in session", genai.RoleModel)
				} else {
					resultEvent.Content = genai.NewContentFromText("Did NOT find initial event in session", genai.RoleModel)
				}
				yield(resultEvent, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("failed to create custom agent: %v", err)
	}

	runner := testutil.NewTestAgentRunner(t, customAgent)

	// Run the agent
	var results []string
	for ev, err := range runner.Run(t, "test_session", "Hello") {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if ev.Content != nil && len(ev.Content.Parts) > 0 {
			results = append(results, ev.Content.Parts[0].Text)
		}
	}

	// Verify the result
	if !slices.Contains(results, "Found initial event in session") {
		t.Errorf("Expected to find initial event in session, but results were: %v", results)
	}
}
