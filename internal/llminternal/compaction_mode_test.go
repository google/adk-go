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

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// The estimate decides when a turn is compacted, so it has to describe the
// prompt the flow will actually build. Both read the mode the agent runs under
// rather than the one it declares: an agent placed single_turn is sent the
// single-turn nudge, and an estimate taken from the bare declaration leaves it
// out. The nudge only reaches a scoped run, which is where a delegated
// single_turn agent lives, so the fixture supplies an isolation scope.
func TestPromptTokenEstimator_UsesTheResolvedMode(t *testing.T) {
	t.Parallel()

	const agentName = "worker"

	newCtx := func(bind bool) agent.InvocationContext {
		stdCtx := t.Context()
		if bind {
			stdCtx = WithBoundMode(stdCtx, agentName, ModeSingleTurn)
		}
		return icontext.NewInvocationContext(stdCtx, icontext.InvocationContextParams{
			Agent: &mockLLMAgent{
				Agent: utils.Must(agent.New(agent.Config{Name: agentName})),
				s:     &State{},
			},
			IsolationScope: "scope-1",
			UserContent:    genai.NewContentFromText("do the thing", "user"),
		})
	}

	// Identical events and an identical declaration on both sides. The only
	// difference is whether a placement resolved single_turn for this agent.
	var events []*session.Event
	placed := promptTokenEstimator(newCtx(true))(events)
	unplaced := promptTokenEstimator(newCtx(false))(events)

	if placed <= unplaced {
		t.Errorf("estimate with a single_turn placement = %d, without one = %d; the placed "+
			"estimate must be the larger of the two, because its prompt carries the "+
			"single-turn nudge and the unplaced one does not", placed, unplaced)
	}
}
