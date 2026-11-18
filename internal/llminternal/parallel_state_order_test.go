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

package llminternal_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// psoGuard bounds every wait in these tests, so a broken order fails a test
// instead of hanging it.
const psoGuard = 5 * time.Second

// psoTurn is the outcome of one run whose model calls tools in one turn and
// then answers "done".
type psoTurn struct {
	// responses maps a function call ID to the response it got.
	responses map[string]map[string]any
	// responseEvent is the event that carries the function responses.
	responseEvent *session.Event
	state         session.ReadonlyState
}

// psoRun runs one invocation of an agent with tools whose model first makes
// calls, then answers "done", over a session that starts with initial.
func psoRun(t *testing.T, tools []tool.Tool, initial map[string]any, calls ...*genai.FunctionCall) psoTurn {
	t.Helper()
	parts := make([]*genai.Part, len(calls))
	for i, c := range calls {
		parts[i] = &genai.Part{FunctionCall: c}
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: parts},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: llm, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s", State: initial}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), psoGuard)
	defer cancel()

	turn := psoTurn{responses: map[string]map[string]any{}}
	for ev, err := range r.Run(ctx, "user", "s", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run error: %v", err)
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil {
				turn.responses[p.FunctionResponse.ID] = p.FunctionResponse.Response
				turn.responseEvent = ev
			}
		}
	}
	got, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	turn.state = got.Session.State()
	return turn
}

func psoTool[A any](t *testing.T, name string, fn func(agent.Context, A) (map[string]any, error)) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tl
}

// psoSeqTool is psoTool with Config.Sequential set.
func psoSeqTool[A any](t *testing.T, name string, fn func(agent.Context, A) (map[string]any, error)) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name, Sequential: true}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tl
}

func psoLongRunningTool(t *testing.T, name string, fn func(agent.Context, struct{}) (map[string]any, error)) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: name, IsLongRunning: true}, fn)
	if err != nil {
		t.Fatal(err)
	}
	return tl
}

func psoCall(id, name string, args map[string]any) *genai.FunctionCall {
	if args == nil {
		args = map[string]any{}
	}
	return &genai.FunctionCall{ID: id, Name: name, Args: args}
}

// psoGet returns the state value of key, or zero when the key is absent.
func psoGet[T any](ctx agent.Context, key string, zero T) (T, error) {
	v, err := ctx.State().Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return zero, nil
	}
	if err != nil {
		return zero, err
	}
	typed, ok := v.(T)
	if !ok {
		return zero, fmt.Errorf("state value has type %T", v)
	}
	return typed, nil
}

func psoStored(t *testing.T, s session.ReadonlyState, key string) any {
	t.Helper()
	v, err := s.Get(key)
	if err != nil {
		t.Fatalf("stored state %q: %v", key, err)
	}
	return v
}

// psoWait waits for ch to close, or fails after psoGuard.
func psoWait(ch <-chan struct{}) error {
	select {
	case <-ch:
		return nil
	case <-time.After(psoGuard):
		return errors.New("a sibling call never got there")
	}
}

func TestParallelCalls_LaterCallSeesEveryEarlierCallsWrite(t *testing.T) {
	slowWrite := psoTool(t, "slow_write", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		time.Sleep(100 * time.Millisecond)
		return map[string]any{}, ctx.State().Set("k", "v0")
	})
	noop := psoTool(t, "noop", func(agent.Context, struct{}) (map[string]any, error) {
		return map[string]any{}, nil
	})
	readK := psoSeqTool(t, "read_k", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		v, err := ctx.State().Get("k")
		if err != nil {
			return nil, err
		}
		return map[string]any{"seen": v}, nil
	})

	turn := psoRun(t, []tool.Tool{slowWrite, noop, readK}, nil,
		psoCall("c0", "slow_write", nil), psoCall("c1", "noop", nil), psoCall("c2", "read_k", nil))

	if diff := cmp.Diff(map[string]any{"seen": "v0"}, turn.responses["c2"]); diff != "" {
		t.Errorf("read_k response mismatch (-want +got):\n%s", diff)
	}
}

func TestParallelCalls_BlockingBetweenReadAndWriteLosesNoUpdate(t *testing.T) {
	incSlowly := psoSeqTool(t, "inc_slowly", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		n, err := psoGet(ctx, "n", 0)
		if err != nil {
			return nil, err
		}
		time.Sleep(50 * time.Millisecond)
		return map[string]any{}, ctx.State().Set("n", n+1)
	})

	turn := psoRun(t, []tool.Tool{incSlowly}, nil, psoCall("c0", "inc_slowly", nil), psoCall("c1", "inc_slowly", nil))

	if got := psoStored(t, turn.state, "n"); got != 2 {
		t.Errorf("stored n = %v (%T), want 2", got, got)
	}
}

func TestParallelCalls_StateFreeSiblingIsNotDelayedByStateUser(t *testing.T) {
	siblingDone := make(chan struct{})
	holder := psoTool(t, "holder", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		if err := ctx.State().Set("a", 1); err != nil {
			return nil, err
		}
		return map[string]any{"released": psoWait(siblingDone) == nil}, nil
	})
	free := psoTool(t, "free", func(agent.Context, struct{}) (map[string]any, error) {
		close(siblingDone)
		return map[string]any{}, nil
	})

	turn := psoRun(t, []tool.Tool{holder, free}, nil, psoCall("c0", "holder", nil), psoCall("c1", "free", nil))

	if diff := cmp.Diff(map[string]any{"released": true}, turn.responses["c0"]); diff != "" {
		t.Errorf("holder response mismatch (-want +got):\n%s", diff)
	}
	if got := psoStored(t, turn.state, "a"); got != 1 {
		t.Errorf("stored a = %v, want 1", got)
	}
}

func TestParallelCalls_EarlyReturnAndFailureReleaseTheTurn(t *testing.T) {
	readA := func(t *testing.T) tool.Tool {
		return psoSeqTool(t, "read_a", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			v, err := ctx.State().Get("a")
			if err != nil {
				return nil, err
			}
			return map[string]any{"seen": v}, nil
		})
	}

	t.Run("long-running call returns no response", func(t *testing.T) {
		pending := psoLongRunningTool(t, "pending", func(agent.Context, struct{}) (map[string]any, error) {
			return nil, nil
		})
		inc := psoSeqTool(t, "inc", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			n, err := psoGet(ctx, "counter", 0)
			if err != nil {
				return nil, err
			}
			return map[string]any{"counter": n + 1}, ctx.State().Set("counter", n+1)
		})

		turn := psoRun(t, []tool.Tool{pending, inc}, nil, psoCall("c0", "pending", nil), psoCall("c1", "inc", nil))

		if turn.responseEvent == nil {
			t.Fatal("no function response event")
		}
		if n := len(turn.responseEvent.Content.Parts); n != 1 {
			t.Errorf("function responses in the merged event = %d, want 1", n)
		}
		if _, ok := turn.responses["c1"]; !ok {
			t.Errorf("no response for inc, responses: %v", turn.responses)
		}
		if got := psoStored(t, turn.state, "counter"); got != 1 {
			t.Errorf("stored counter = %v, want 1", got)
		}
	})

	failures := []struct {
		name string
		fail func(agent.Context, struct{}) (map[string]any, error)
	}{
		{name: "earlier call returns an error", fail: func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			time.Sleep(50 * time.Millisecond)
			if err := ctx.State().Set("a", 1); err != nil {
				return nil, err
			}
			return nil, errors.New("boom")
		}},
		{name: "earlier call panics", fail: func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			time.Sleep(50 * time.Millisecond)
			if err := ctx.State().Set("a", 1); err != nil {
				return nil, err
			}
			panic("boom")
		}},
	}
	for _, tt := range failures {
		t.Run(tt.name, func(t *testing.T) {
			fail := psoTool(t, "fail", tt.fail)

			turn := psoRun(t, []tool.Tool{fail, readA(t)}, nil, psoCall("c0", "fail", nil), psoCall("c1", "read_a", nil))

			// A function tool's result reaches the response through JSON.
			if diff := cmp.Diff(map[string]any{"seen": float64(1)}, turn.responses["c1"]); diff != "" {
				t.Errorf("read_a response mismatch (-want +got):\n%s", diff)
			}
			if turn.responseEvent == nil {
				t.Fatal("no function response event")
			}
			if got := turn.responseEvent.Actions.StateDelta["a"]; got != 1 {
				t.Errorf("merged state delta a = %v, want 1", got)
			}
		})
	}
}

func TestParallelCalls_DictReadModifyWrite(t *testing.T) {
	incDoc := psoSeqTool(t, "inc_doc", func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		read, err := psoGet(ctx, "doc", map[string]any{})
		if err != nil {
			return nil, err
		}
		doc := maps.Clone(read)
		n, _ := doc["n"].(int)
		doc["n"] = n + 1
		// Holds the read open, so that without the call order the second
		// call reads before the first writes.
		time.Sleep(20 * time.Millisecond)
		return map[string]any{}, ctx.State().Set("doc", doc)
	})

	turn := psoRun(t, []tool.Tool{incDoc}, nil, psoCall("c0", "inc_doc", nil), psoCall("c1", "inc_doc", nil))

	if diff := cmp.Diff(map[string]any{"n": 2}, psoStored(t, turn.state, "doc")); diff != "" {
		t.Errorf("stored doc mismatch (-want +got):\n%s", diff)
	}
}

// TestParallelCalls_OutOfOrderAppendKeepsEveryItem runs adk-python's
// test_parallel_appends_finishing_out_of_order_keep_every_item and
// test_parallel_snapshot_writes_keep_the_latest_nested_list
// (tests/unittests/flows/llm_flows/tools/test_functions_parallel.py): call 0
// of a concurrent tool waits for call 1's write before it writes.
func TestParallelCalls_OutOfOrderAppendKeepsEveryItem(t *testing.T) {
	type itemArgs struct {
		Item string `json:"item"`
	}
	type labelArgs struct {
		Label string `json:"label"`
	}

	t.Run("appended list", func(t *testing.T) {
		bWritten := make(chan struct{})
		appendItem := psoTool(t, "append_item", func(ctx agent.Context, args itemArgs) (map[string]any, error) {
			if args.Item == "a" {
				if err := psoWait(bWritten); err != nil {
					return nil, err
				}
			}
			items, err := psoGet(ctx, "items", []any{})
			if err != nil {
				return nil, err
			}
			if err := ctx.State().Set("items", append(slices.Clone(items), args.Item)); err != nil {
				return nil, err
			}
			if args.Item == "b" {
				close(bWritten)
			}
			return map[string]any{}, nil
		})

		turn := psoRun(t, []tool.Tool{appendItem}, nil,
			psoCall("c0", "append_item", map[string]any{"item": "a"}), psoCall("c1", "append_item", map[string]any{"item": "b"}))

		if diff := cmp.Diff([]any{"b", "a"}, psoStored(t, turn.state, "items")); diff != "" {
			t.Errorf("stored items mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("snapshot of a nested list", func(t *testing.T) {
		bWritten := make(chan struct{})
		addLabel := psoTool(t, "add_label", func(ctx agent.Context, args labelArgs) (map[string]any, error) {
			if args.Label == "a" {
				if err := psoWait(bWritten); err != nil {
					return nil, err
				}
			}
			doc, err := psoGet(ctx, "doc", map[string]any{})
			if err != nil {
				return nil, err
			}
			if err := ctx.State().Set("doc", psoWithLabel(doc, args.Label)); err != nil {
				return nil, err
			}
			if args.Label == "b" {
				close(bWritten)
			}
			return map[string]any{}, nil
		})
		initial := map[string]any{"doc": map[string]any{"entities": []any{map[string]any{"name": "e1", "labels": []any{}}}}}

		turn := psoRun(t, []tool.Tool{addLabel}, initial,
			psoCall("c0", "add_label", map[string]any{"label": "a"}), psoCall("c1", "add_label", map[string]any{"label": "b"}))

		want := map[string]any{"entities": []any{map[string]any{"name": "e1", "labels": []any{"b", "a"}}}}
		if diff := cmp.Diff(want, psoStored(t, turn.state, "doc")); diff != "" {
			t.Errorf("stored doc mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestParallelCalls_EmptiedListIsTheLatestWrite has two calls each remove one
// entry of a two-entry list, and call 0 reads call 1's write before it writes.
// The list call 0 writes is empty with spare capacity and is the session's
// value, so the state ends empty, not with call 1's one-entry list.
func TestParallelCalls_EmptiedListIsTheLatestWrite(t *testing.T) {
	type itemArgs struct {
		Item string `json:"item"`
	}
	bWritten := make(chan struct{})
	removeItem := psoTool(t, "remove_item", func(ctx agent.Context, args itemArgs) (map[string]any, error) {
		if args.Item == "a" {
			if err := psoWait(bWritten); err != nil {
				return nil, err
			}
		}
		items, err := psoGet(ctx, "items", []any{})
		if err != nil {
			return nil, err
		}
		kept := make([]any, 0, len(items))
		for _, it := range items {
			if it != args.Item {
				kept = append(kept, it)
			}
		}
		if err := ctx.State().Set("items", kept); err != nil {
			return nil, err
		}
		if args.Item == "b" {
			close(bWritten)
		}
		return map[string]any{}, nil
	})
	initial := map[string]any{"items": []any{"a", "b"}}

	turn := psoRun(t, []tool.Tool{removeItem}, initial,
		psoCall("c0", "remove_item", map[string]any{"item": "a"}), psoCall("c1", "remove_item", map[string]any{"item": "b"}))

	if diff := cmp.Diff([]any{}, psoStored(t, turn.state, "items")); diff != "" {
		t.Errorf("stored items mismatch (-want +got):\n%s", diff)
	}
}

// psoWithLabel returns a deep copy of doc ({entities: [{name, labels}]}) with
// label appended to the first entity's labels.
func psoWithLabel(doc map[string]any, label string) map[string]any {
	entities, _ := doc["entities"].([]any)
	first, _ := entities[0].(map[string]any)
	labels, _ := first["labels"].([]any)
	entity := maps.Clone(first)
	entity["labels"] = append(slices.Clone(labels), label)
	out := maps.Clone(doc)
	out["entities"] = append([]any{entity}, entities[1:]...)
	return out
}
