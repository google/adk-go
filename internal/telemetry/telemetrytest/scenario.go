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

package telemetrytest

import (
	"context"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// RunScenario drives rootAgent through a single user turn via a
// real Runner backed by an in-memory session service. Used by
// every functional telemetry test; kept here rather than in a
// _test.go file so callers in any external test package can reuse
// the same scenario harness.
//
// Mirrors functional_test_helpers.run_agent_scenario in adk-python.
func RunScenario(t *testing.T, rootAgent agent.Agent, prompt string) {
	t.Helper()
	ctx := context.Background()

	sessSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          rootAgent,
		SessionService: sessSvc,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	sess, err := sessSvc.Create(ctx, &session.CreateRequest{
		AppName: "test_app",
		UserID:  "test_user",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}

	msg := genai.NewContentFromText(prompt, genai.RoleUser)
	for _, err := range r.Run(ctx, "test_user", sess.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run yielded error: %v", err)
		}
	}
}
