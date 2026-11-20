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
	"fmt"
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// staticPath is the path of a node "n" that runs as the only node of
// workflow "wf" under runParent.
const staticPath = "wf@1/n@1"

// integerSchema is an output schema that accepts integers only.
var integerSchema = &jsonschema.Schema{Type: "integer"}

// resolvedIntegerSchema resolves integerSchema.
func resolvedIntegerSchema(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	r, err := integerSchema.Resolve(nil)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestScheduler_OutputSchemaFailureIsRetried checks that a graph node whose
// output its schema refuses is run again as its RetryConfig allows, that
// each failed attempt leaves a failure record and its own state writes, and
// that the refused output is never yielded.
//
// Fails if the refusal is terminal instead of retried, or if the refused
// event loses the attempt's pending writes.
func TestScheduler_OutputSchemaFailureIsRetried(t *testing.T) {
	t.Run("retries_until_valid", func(t *testing.T) {
		calls := 0
		n, err := NewFunctionNodeWithSchema("n", func(ctx agent.Context, _ any) (any, error) {
			calls++
			if err := ctx.State().Set(fmt.Sprintf("attempt_%d", calls), calls); err != nil {
				return nil, err
			}
			if calls < 3 {
				return "not_an_int", nil
			}
			return 42, nil
		}, nil, integerSchema, NodeConfig{RetryConfig: retryNoDelay(3)})
		if err != nil {
			t.Fatal(err)
		}

		res := runParent(t, n)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		if calls != 3 {
			t.Errorf("node ran %d times, want 3", calls)
		}
		want := []string{"state:attempt_1=1", "failed", "state:attempt_2=2", "failed", "output=42 state:attempt_3=3"}
		if diff := cmp.Diff(want, describeAt(res.events, staticPath)); diff != "" {
			t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
		}
		for _, ev := range res.events {
			if ev.Output == "not_an_int" {
				t.Errorf("an event carries the refused output %q", ev.Output)
			}
		}
	})
	t.Run("exhausts_max_attempts", func(t *testing.T) {
		calls := 0
		n, err := NewFunctionNodeWithSchema("n", func(agent.Context, any) (any, error) {
			calls++
			return "not_an_int", nil
		}, nil, integerSchema, NodeConfig{RetryConfig: retryNoDelay(2)})
		if err != nil {
			t.Fatal(err)
		}

		res := runParent(t, n)

		want := `workflow: dynamic child failed: output validation failed for node "n": validating root: type: not_an_int has type "string", want "integer"`
		if res.err == nil || res.err.Error() != want {
			t.Errorf("Run error = %v, want %q", res.err, want)
		}
		if !errors.Is(res.err, ErrNodeFailed) {
			t.Errorf("Run error = %v, want errors.Is ErrNodeFailed", res.err)
		}
		if calls != 2 {
			t.Errorf("node ran %d times, want 2", calls)
		}
	})
}

// TestScheduler_RejectedOutputSendsNoEmptyEvent checks that a refused output
// event that carries nothing once its output is removed is not yielded.
//
// Fails if the scheduler yields the stripped event anyway.
func TestScheduler_RejectedOutputSendsNoEmptyEvent(t *testing.T) {
	calls := 0
	n, err := NewFunctionNodeWithSchema("n", func(agent.Context, any) (any, error) {
		calls++
		if calls < 3 {
			return "not_an_int", nil
		}
		return 42, nil
	}, nil, integerSchema, NodeConfig{RetryConfig: retryNoDelay(3)})
	if err != nil {
		t.Fatal(err)
	}

	res := runParent(t, n)

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	want := []string{"failed", "failed", "output=42"}
	if diff := cmp.Diff(want, describeAt(res.events, staticPath)); diff != "" {
		t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
	}
}

// TestScheduler_MultipleOutputsAreNotRetried checks that a node that yields
// two outputs or two routing events fails at once with ErrMultipleOutputs or
// ErrMultipleRoutingEvents even when a later output (direct or delegated) of
// the same attempt is refused by its schema and RetryConfig retries every
// error.
//
// Fails if the scheduler retries or cancels any attempt that saw a refused
// output when an earlier terminal error was already recorded.
func TestScheduler_MultipleOutputsAreNotRetried(t *testing.T) {
	refusedDirect := func(ctx agent.Context) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "bad"
		return ev
	}
	refusedDelegated := func(ctx agent.Context) *session.Event {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: true, workflowDelegatedOutputKey: "bad"}
		return ev
	}
	tests := []struct {
		name    string
		emit    func(ctx agent.Context, yield func(*session.Event, error) bool)
		wantErr error
	}{
		{
			name: "multiple_outputs_then_refused_output",
			emit: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				for _, out := range []any{1, 2} {
					if !faninEmit(ctx, yield, out) {
						return
					}
				}
				yield(refusedDirect(ctx), nil)
			},
			wantErr: ErrMultipleOutputs,
		},
		{
			name: "multiple_outputs_then_refused_delegated_output",
			emit: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				for _, out := range []any{1, 2} {
					if !faninEmit(ctx, yield, out) {
						return
					}
				}
				yield(refusedDelegated(ctx), nil)
			},
			wantErr: ErrMultipleOutputs,
		},
		{
			name: "multiple_routes_then_refused_output",
			emit: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				for _, route := range []string{"r1", "r2"} {
					if !faninEmit(ctx, yield, nil, route) {
						return
					}
				}
				yield(refusedDirect(ctx), nil)
			},
			wantErr: ErrMultipleRoutingEvents,
		},
		{
			name: "multiple_routes_then_refused_delegated_output",
			emit: func(ctx agent.Context, yield func(*session.Event, error) bool) {
				for _, route := range []string{"r1", "r2"} {
					if !faninEmit(ctx, yield, nil, route) {
						return
					}
				}
				yield(refusedDelegated(ctx), nil)
			},
			wantErr: ErrMultipleRoutingEvents,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			n := &faninNode{
				BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(3)}, nil, resolvedIntegerSchema(t)),
				run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
					calls++
					tc.emit(ctx, yield)
				},
			}

			res := runParent(t, n)

			if !errors.Is(res.err, tc.wantErr) {
				t.Errorf("Run error = %v, want %v", res.err, tc.wantErr)
			}
			if calls != 1 {
				t.Errorf("node ran %d times, want 1", calls)
			}
		})
	}
}

// TestScheduler_RefusedOutputStoresNothingOfItsOwnEvent checks that an event
// whose output the node's schema refuses stores nothing it carried itself,
// while the writes the attempt made through its context are stored.
//
// Fails if the refused event keeps its own state or artifact delta, or if it
// loses the pending writes that admit moved onto it.
func TestScheduler_RefusedOutputStoresNothingOfItsOwnEvent(t *testing.T) {
	calls := 0
	n := &faninNode{
		BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(3)}, nil, resolvedIntegerSchema(t)),
		run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			calls++
			k := calls
			if err := ctx.State().Set(fmt.Sprintf("pending_%d", k), k); err != nil {
				yield(nil, err)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = "bad"
			if k == 3 {
				ev.Output = 42
			}
			ev.Actions.StateDelta = map[string]any{fmt.Sprintf("own_%d", k): k}
			ev.Actions.ArtifactDelta = map[string]int64{fmt.Sprintf("own_art_%d", k): int64(k)}
			yield(ev, nil)
		},
	}

	res := runParent(t, n)

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	wantState := map[string]any{"pending_1": 1, "pending_2": 2, "pending_3": 3, "own_3": 3}
	if diff := cmp.Diff(wantState, storedState(res.events)); diff != "" {
		t.Errorf("stored state mismatch (-want +got):\n%s", diff)
	}
	storedArt := map[string]int64{}
	for _, ev := range res.events {
		if !ev.Partial {
			maps.Copy(storedArt, ev.Actions.ArtifactDelta)
		}
	}
	if diff := cmp.Diff(map[string]int64{"own_art_3": 3}, storedArt); diff != "" {
		t.Errorf("stored artifact delta mismatch (-want +got):\n%s", diff)
	}
	wantEvents := []string{"state:pending_1=1", "failed", "state:pending_2=2", "failed", "output=42 state:own_3=3,pending_3=3"}
	if diff := cmp.Diff(wantEvents, describeAt(res.events, staticPath)); diff != "" {
		t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
	}
}

// TestScheduler_RefusedOutputSendsNoContentOrRoutes checks that the content
// and routes of an event whose output the schema refuses are neither yielded
// nor counted. Each attempt writes state, so the refused event is still sent
// with the attempt's pending writes.
//
// Fails if rejectOutput keeps Content or Routes, or if the scheduler reads
// an event's routes before it validates the event's output.
func TestScheduler_RefusedOutputSendsNoContentOrRoutes(t *testing.T) {
	t.Run("text_output", func(t *testing.T) {
		calls := 0
		n, err := NewEmittingFunctionNodeWithSchema("n", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
			calls++
			if err := ctx.State().Set(fmt.Sprintf("seen_%d", calls), calls); err != nil {
				return nil, err
			}
			if calls == 1 {
				return "not_an_int", nil
			}
			return 42, nil
		}, nil, integerSchema, NodeConfig{RetryConfig: retryNoDelay(2)})
		if err != nil {
			t.Fatal(err)
		}

		res := runParent(t, n)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		for _, ev := range res.events {
			if text, _ := messageText(ev); text == "not_an_int" {
				t.Errorf("event at %v carries the refused text %q", ev.NodeInfo, text)
			}
		}
		if diff := cmp.Diff([]string{"state:seen_1=1", "failed", "output=42 state:seen_2=2"}, describeAt(res.events, staticPath)); diff != "" {
			t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
		}
	})
	t.Run("content_and_routes", func(t *testing.T) {
		calls := 0
		n := &faninNode{
			BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(2)}, nil, resolvedIntegerSchema(t)),
			run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				calls++
				if calls == 2 {
					faninEmit(ctx, yield, 42)
					return
				}
				if err := ctx.State().Set("seen", calls); err != nil {
					yield(nil, err)
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "bad"
				ev.Content = genai.NewContentFromText("refused text", genai.RoleModel)
				ev.Routes = []string{"refused-route"}
				yield(ev, nil)
			},
		}

		res := runParent(t, n)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		if got := storedState(res.events)["seen"]; got != 1 {
			t.Errorf("stored seen = %v, want 1 (the refused event carries the pending write)", got)
		}
		for _, ev := range res.events {
			if text, _ := messageText(ev); text == "refused text" {
				t.Errorf("an event carries the refused content %q", text)
			}
			if len(ev.Routes) > 0 {
				t.Errorf("an event carries the refused routes %v", ev.Routes)
			}
		}
	})
	t.Run("routes_read_after_validation", func(t *testing.T) {
		calls := 0
		n := &faninNode{
			BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(2)}, nil, resolvedIntegerSchema(t)),
			run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				calls++
				if calls == 2 {
					faninEmit(ctx, yield, 42)
					return
				}
				if !faninEmit(ctx, yield, nil, "first") {
					return
				}
				faninEmit(ctx, yield, "bad", "second")
			},
		}

		res := runParent(t, n)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil (the refusal is retried, the refused routes do not count)", res.err)
		}
		if calls != 2 {
			t.Errorf("node ran %d times, want 2", calls)
		}
	})
	t.Run("actions_and_ids", func(t *testing.T) {
		calls := 0
		n := &faninNode{
			BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(2)}, nil, resolvedIntegerSchema(t)),
			run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
				calls++
				if calls == 2 {
					faninEmit(ctx, yield, 42)
					return
				}
				if err := ctx.State().Set("seen", 1); err != nil {
					yield(nil, err)
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "bad"
				ev.Actions.Escalate = true
				ev.Actions.TransferToAgent = "other"
				ev.Actions.SkipSummarization = true
				ev.LongRunningToolIDs = []string{"lr-x"}
				ev.RequestedInput = &session.RequestInput{InterruptID: "q-x", Message: "?"}
				yield(ev, nil)
			},
		}

		res := runParent(t, n)

		if res.err != nil {
			t.Fatalf("Run error = %v, want nil", res.err)
		}
		if diff := cmp.Diff([]string{"state:seen=1", "failed", "output=42"}, describeAt(res.events, staticPath)); diff != "" {
			t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
		}
		for _, ev := range res.events {
			a := ev.Actions
			if a.Escalate || a.TransferToAgent != "" || a.SkipSummarization || len(ev.LongRunningToolIDs) > 0 || ev.RequestedInput != nil {
				t.Errorf("event at %v carries the refused event's escalate=%v transfer=%q skip=%v long-running=%v requested=%v",
					ev.NodeInfo, a.Escalate, a.TransferToAgent, a.SkipSummarization, ev.LongRunningToolIDs, ev.RequestedInput)
			}
		}
	})
}

// TestScheduler_AttemptStopsAtRefusedOutput checks that an attempt ends at
// the event whose output its schema refuses: what the node yields after it
// is neither yielded nor stored.
//
// Fails if the scheduler leaves the node's context alive after a refusal, or
// if it then takes the node's cancellation for a sibling cancellation.
func TestScheduler_AttemptStopsAtRefusedOutput(t *testing.T) {
	tests := []struct {
		name    string
		refused func(ctx agent.Context) *session.Event
	}{
		{
			name: "output",
			refused: func(ctx agent.Context) *session.Event {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "bad"
				return ev
			},
		},
		{
			name: "delegated_output",
			refused: func(ctx agent.Context) *session.Event {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.CustomMetadata = map[string]any{workflowNodeCompletedKey: true, workflowDelegatedOutputKey: "bad"}
				return ev
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			n := &faninNode{
				BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(2)}, nil, resolvedIntegerSchema(t)),
				run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
					calls++
					if calls == 2 {
						faninEmit(ctx, yield, 42)
						return
					}
					if !yield(tc.refused(ctx), nil) {
						return
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Actions.StateDelta = map[string]any{"after_refusal": 1}
					yield(ev, nil)
				},
			}

			res := runParent(t, n)

			if res.err != nil {
				t.Fatalf("Run error = %v, want nil", res.err)
			}
			if calls != 2 {
				t.Errorf("node ran %d times, want 2", calls)
			}
			if got, ok := storedState(res.events)["after_refusal"]; ok {
				t.Errorf("stored state holds after_refusal=%v, want it absent", got)
			}
		})
	}
}

// TestScheduler_RefusedPartialOutputIsRetried checks that an attempt that
// ends with a partial event whose output the schema refuses fails with the
// refusal, so RetryConfig runs the node again.
//
// Fails if the refusal becomes the attempt's error only when the node's
// iterator ended canceled, not when it returned on its own.
func TestScheduler_RefusedPartialOutputIsRetried(t *testing.T) {
	calls := 0
	n := &faninNode{
		BaseNode: NewBaseNodeWithSchemas("n", "", NodeConfig{RetryConfig: retryNoDelay(2)}, nil, resolvedIntegerSchema(t)),
		run: func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			calls++
			if calls == 2 {
				faninEmit(ctx, yield, 42)
				return
			}
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Partial = true
			ev.Output = "bad"
			yield(ev, nil)
		},
	}

	res := runParent(t, n)

	if res.err != nil {
		t.Fatalf("Run error = %v, want nil", res.err)
	}
	if calls != 2 {
		t.Errorf("node ran %d times, want 2", calls)
	}
	if diff := cmp.Diff([]string{"failed", "output=42"}, describeAt(res.events, staticPath)); diff != "" {
		t.Errorf("events at %s mismatch (-want +got):\n%s", staticPath, diff)
	}
}
