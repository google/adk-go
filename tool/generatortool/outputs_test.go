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

package generatortool_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

const multipleOutputsText = "Output already set. A node can produce at most one output."

func output(v any) *session.Event { return &session.Event{Output: v} }

// runMultipleOutputs runs one call of a tool named gen with handler, and
// checks what every multiple-output failure shares: the run's error, no
// function response, one model request, and the tool's events, which must be
// exactly the first output event.
func runMultipleOutputs(t *testing.T, handler func(agent.Context, queryArgs, func(*session.Event) error) (any, error)) {
	t.Helper()
	gen := mustNew(t, functiontool.Config{Name: "gen"}, handler)
	h := newHarness(t, harnessConfig{tools: []tool.Tool{gen}, responses: []*genai.Content{
		calls(call("call_1", "gen", map[string]any{"query": "q"})), modelText("done"),
	}})

	_, err := h.run(userText("go"))

	if !errors.Is(err, workflow.ErrMultipleOutputs) || err.Error() != multipleOutputsText {
		t.Errorf("Run() error = %v, want %q matching workflow.ErrMultipleOutputs", err, multipleOutputsText)
	}
	events := h.events()
	if got := responses(events, "gen"); len(got) != 0 {
		t.Errorf("function responses = %v, want none", got)
	}
	if got := h.modelRequests(); got != 1 {
		t.Errorf("model requests = %d, want 1", got)
	}
	wantTool := []evSummary{{Author: "gen", Branch: "gen@call_1", Path: "root_agent/gen@call_1", Output: map[string]any{"rows": 1}, NoContent: true}}
	if diff := cmp.Diff(wantTool, summarize(authoredBy(events, "gen"))); diff != "" {
		t.Errorf("tool events mismatch (-want +got):\n%s", diff)
	}
}

// A9. Fails if a second output replaces the first, is joined to it, or is
// answered to the model, or if the error lacks Python's text.
func TestRun_SecondOutputFailsRun(t *testing.T) {
	var closed atomic.Int32
	runMultipleOutputs(t, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		defer closed.Add(1)
		if err := emit(output(map[string]any{"rows": 1})); err != nil {
			return nil, err
		}
		return nil, emit(output(map[string]any{"rows": 2}))
	})
	if got := closed.Load(); got != 1 {
		t.Errorf("handler's defer ran %d times, want 1", got)
	}
}

// A10. Fails if a result returned after an emitted output overwrites it or
// is dropped silently.
func TestRun_ResultAfterEmittedOutputFailsRun(t *testing.T) {
	runMultipleOutputs(t, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(output(map[string]any{"rows": 1})); err != nil {
			return nil, err
		}
		return map[string]any{"rows": 2}, nil
	})
}

// A11. Fails if the failure depends on the handler returning the emit error.
func TestRun_EmitErrorIgnoredStillFails(t *testing.T) {
	runMultipleOutputs(t, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		if err := emit(output(map[string]any{"rows": 1})); err != nil {
			return nil, err
		}
		_ = emit(output(map[string]any{"rows": 2})) // The test is that this error is ignored.
		return nil, nil
	})
}

// A30. Fails if emit accepts events after the violation, or if the body's
// context stays open.
func TestRun_EmitAfterSecondOutputIsRefused(t *testing.T) {
	var e0, e1, e2, c1 error
	runMultipleOutputs(t, func(ctx agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		e0 = emit(output(map[string]any{"rows": 1}))
		e1 = emit(output(map[string]any{"rows": 2}))
		c1 = ctx.Err()
		e2 = emit(message("after"))
		return nil, nil
	})
	if e0 != nil {
		t.Errorf("first emit error = %v, want nil", e0)
	}
	for name, err := range map[string]error{"second output": e1, "message after it": e2} {
		if !errors.Is(err, workflow.ErrMultipleOutputs) || err.Error() != multipleOutputsText {
			t.Errorf("emit of the %s error = %v, want %q matching workflow.ErrMultipleOutputs", name, err, multipleOutputsText)
		}
	}
	if !errors.Is(c1, context.Canceled) {
		t.Errorf("ctx.Err() after the second output = %v, want context.Canceled", c1)
	}
}

// A34. Fails if the output count is not guarded against concurrent emits (the
// race detector reports it), or if two concurrent outputs are both accepted.
func TestRun_EmitConcurrentOutputsOneWins(t *testing.T) {
	const goroutines = 8
	var accepted atomic.Int32
	runMultipleOutputs(t, func(_ agent.Context, _ queryArgs, emit func(*session.Event) error) (any, error) {
		var wg sync.WaitGroup
		for range goroutines {
			wg.Go(func() {
				if err := emit(output(map[string]any{"rows": 1})); err == nil {
					accepted.Add(1)
				}
			})
		}
		wg.Wait()
		return nil, nil
	})
	if got := accepted.Load(); got != 1 {
		t.Errorf("accepted outputs = %d, want 1", got)
	}
}
