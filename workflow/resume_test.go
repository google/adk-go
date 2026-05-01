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

package workflow_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/app"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestWorkflow_Resume_SkipsCompletedNodes verifies that on a Resume call,
// nodes that already produced output in the prior invocation do not run
// again — only the not-yet-completed tail re-executes.
func TestWorkflow_Resume_SkipsCompletedNodes(t *testing.T) {
	type val struct {
		N int `json:"n"`
	}
	var aRuns, bRuns, cRuns atomic.Int32

	a := workflow.Func("step_a",
		func(_ *workflow.NodeContext, _ any) (val, error) {
			aRuns.Add(1)
			return val{N: 1}, nil
		})
	bShouldFail := atomic.Bool{}
	bShouldFail.Store(true)
	b := workflow.Func("step_b",
		func(_ *workflow.NodeContext, in val) (val, error) {
			bRuns.Add(1)
			if bShouldFail.Load() {
				return val{}, errors.New("transient")
			}
			return val{N: in.N + 10}, nil
		})
	c := workflow.Func("step_c",
		func(_ *workflow.NodeContext, in val) (val, error) {
			cRuns.Add(1)
			return val{N: in.N * 100}, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "rwf",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, a),
			workflow.Connect(a, b),
			workflow.Connect(b, c),
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, _ := wf.AsAgent()

	sessSvc := session.InMemoryService()
	r, err := runner.New(runner.Config{
		App: &app.App{
			Name:               "test",
			RootAgent:          wfAgent,
			ResumabilityConfig: &app.ResumabilityConfig{IsResumable: true},
		},
		SessionService:    sessSvc,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}

	// First run: a succeeds, b fails. Workflow aborts.
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}}
	for _, err := range r.Run(context.Background(), "u", "sess1", msg, agent.RunConfig{}) {
		_ = err // expect error from b
	}
	if aRuns.Load() != 1 {
		t.Errorf("aRuns after first run = %d, want 1", aRuns.Load())
	}
	if bRuns.Load() != 1 {
		t.Errorf("bRuns after first run = %d, want 1", bRuns.Load())
	}
	if cRuns.Load() != 0 {
		t.Errorf("cRuns after first run = %d, want 0", cRuns.Load())
	}

	// Flip b to succeed and resume. Expect: a does NOT run again (cached
	// output replayed); b runs (no completion event); c runs.
	bShouldFail.Store(false)
	for ev, err := range r.Resume(context.Background(), "u", "sess1", agent.RunConfig{}) {
		if err != nil {
			t.Logf("resume ev: %+v err: %v", ev, err)
		}
	}
	if got := aRuns.Load(); got != 1 {
		t.Errorf("aRuns after resume = %d, want 1 (a should be replayed from cache)", got)
	}
	if got := bRuns.Load(); got != 2 {
		t.Errorf("bRuns after resume = %d, want 2 (b reruns, no prior completion)", got)
	}
	if got := cRuns.Load(); got != 1 {
		t.Errorf("cRuns after resume = %d, want 1", got)
	}
}

// TestWorkflow_Resume_RerunOnResumeForcesRerun verifies that nodes
// configured with WithRerunOnResume(true) re-execute even when they have
// a prior completion event.
func TestWorkflow_Resume_RerunOnResumeForcesRerun(t *testing.T) {
	var aRuns atomic.Int32
	a := workflow.Func("a",
		func(_ *workflow.NodeContext, _ any) (int, error) {
			aRuns.Add(1)
			return 1, nil
		},
		workflow.WithRerunOnResume(true),
	)
	wf, err := workflow.New(workflow.Config{
		Name:  "rerun_wf",
		Edges: []workflow.Edge{workflow.Connect(workflow.START, a)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		App: &app.App{
			Name:               "t",
			RootAgent:          wfAgent,
			ResumabilityConfig: &app.ResumabilityConfig{IsResumable: true},
		},
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "x"}}}
	for range r.Run(context.Background(), "u", "s", msg, agent.RunConfig{}) {
	}
	for range r.Resume(context.Background(), "u", "s", agent.RunConfig{}) {
	}
	if got := aRuns.Load(); got != 2 {
		t.Errorf("aRuns = %d, want 2 (RerunOnResume should force a re-run)", got)
	}
}
