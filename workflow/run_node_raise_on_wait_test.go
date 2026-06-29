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

package workflow

import (
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// longRunningChild emits one FC listed in LongRunningToolIDs, then
// optionally a matching FR. Drives the WithRaiseOnWait scenarios.
type longRunningChild struct {
	BaseNode
	fcName          string
	fcID            string
	yieldMatchingFR bool
}

func newLongRunningChild(name, fcName, fcID string, yieldMatchingFR bool) *longRunningChild {
	return &longRunningChild{
		BaseNode:        NewBaseNode(name, "", NodeConfig{}),
		fcName:          fcName,
		fcID:            fcID,
		yieldMatchingFR: yieldMatchingFR,
	}
}

func (n *longRunningChild) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// FC event: ID in LongRunningToolIDs signals "tool pauses".
		fcEvent := &session.Event{
			Author:             n.Name(),
			LongRunningToolIDs: []string{n.fcID},
		}
		fcEvent.LLMResponse.Content = &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: n.fcName, ID: n.fcID}}},
		}
		if !yield(fcEvent, nil) {
			return
		}
		if !n.yieldMatchingFR {
			return
		}
		// Matching FR clears the pending-LRT set.
		frEvent := &session.Event{Author: n.Name()}
		frEvent.LLMResponse.Content = &genai.Content{
			Role: "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name:     n.fcName,
				ID:       n.fcID,
				Response: map[string]any{"result": "ok"},
			}}},
		}
		yield(frEvent, nil)
	}
}

// Unresolved LRT + WithRaiseOnWait → ErrNodeInterrupted (so callers
// can distinguish "task paused" from "task finished with no output").
func TestRunNode_WithRaiseOnWait_UnresolvedLRT_ReturnsErrNodeInterrupted(t *testing.T) {
	child := newLongRunningChild("paused_tool_caller", "confirmation", "fc-1", false /* no matching FR */)

	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithRaiseOnWait())
	})
	if !errors.Is(err, ErrNodeInterrupted) {
		t.Errorf("err = %v, want errors.Is(err, ErrNodeInterrupted)", err)
	}
}

// In-iteration matching FR clears the pending LRT — run completes
// normally even with WithRaiseOnWait set.
func TestRunNode_WithRaiseOnWait_ResolvedLRT_CompletesNormally(t *testing.T) {
	child := newLongRunningChild("resolves_in_place", "tool", "fc-1", true /* matching FR yielded */)

	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil, WithRaiseOnWait())
	})
	if err != nil {
		t.Errorf("err = %v, want nil (in-iteration FR clears the pending LRT)", err)
	}
}

// Pins the gating as opt-in: the same unresolved-LRT child completes
// normally when WithRaiseOnWait is not set.
func TestRunNode_WithoutRaiseOnWait_UnresolvedLRT_CompletesNormally(t *testing.T) {
	child := newLongRunningChild("paused_tool_caller", "confirmation", "fc-1", false /* no matching FR */)

	_, err := runInOrchestratorWithErr[string](t, func(ctx agent.Context) (string, error) {
		return RunNode[string](ctx, child, nil /* no WithRaiseOnWait */)
	})
	if err != nil {
		t.Errorf("err = %v, want nil (LRT tracking must be opt-in via WithRaiseOnWait)", err)
	}
}
