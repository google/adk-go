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
	"reflect"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// streamSeen is what a caller observed while ranging over a run.
type streamSeen struct {
	outputs []any          // Output of every event that has one, in order
	texts   []string       // text of every event with text content, in order
	state   map[string]any // union of the events' state deltas
	err     error          // first error the run yielded
}

// observe runs seq to the end and records what the caller saw.
func observe(seq iter.Seq2[*session.Event, error]) streamSeen {
	var s streamSeen
	for ev, err := range seq {
		if err != nil {
			if s.err == nil {
				s.err = err
			}
			continue
		}
		if ev == nil {
			continue
		}
		for k, v := range ev.Actions.StateDelta {
			if s.state == nil {
				s.state = map[string]any{}
			}
			s.state[k] = v
		}
		if ev.Output != nil {
			s.outputs = append(s.outputs, ev.Output)
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text != "" {
					s.texts = append(s.texts, p.Text)
				}
			}
		}
	}
	return s
}

// modelTextEvent returns a model event with text and the given Partial flag.
func modelTextEvent(ctx agent.Context, text string, partial bool) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Content = genai.NewContentFromText(text, genai.RoleModel)
	ev.Partial = partial
	return ev
}

func TestScheduler_TimedOutNodeThatIgnoresItsContextDeliversNoOutput(t *testing.T) {
	ign := timedNode("ign", 20*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "done", nil
	})
	w := mustNew(t, []Edge{{From: Start, To: ign}})

	got := observe(w.Run(boundedSeededCtx(t)))

	if got.outputs != nil {
		t.Errorf("outputs in the stream = %#v, want none", got.outputs)
	}
	var te *NodeTimeoutError
	if !errors.As(got.err, &te) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", got.err)
	}
	if te.NodeName != "ign" || te.Timeout != 20*time.Millisecond {
		t.Errorf("Run error = {%q, %v}, want {\"ign\", 20ms}", te.NodeName, te.Timeout)
	}
}

func TestScheduler_TimedOutNodeDeliversOnlyTheEventsItYieldedBeforeTheDeadline(t *testing.T) {
	lateTaken := true
	n := newFaninNode("streamer", NodeConfig{Timeout: 200 * time.Millisecond}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		if !yield(modelTextEvent(ctx, "early", false), nil) {
			return
		}
		<-ctx.Done()
		lateTaken = yield(modelTextEvent(ctx, "late partial", true), nil)
	})
	w := mustNew(t, []Edge{{From: Start, To: n}})

	got := observe(w.Run(boundedSeededCtx(t)))

	if want := []string{"early"}; !reflect.DeepEqual(got.texts, want) {
		t.Errorf("texts in the stream = %q, want %q", got.texts, want)
	}
	if lateTaken {
		t.Error("yield of the late partial event = true, want false")
	}
	var te *NodeTimeoutError
	if !errors.As(got.err, &te) {
		t.Fatalf("Run error = %v, want a *NodeTimeoutError", got.err)
	}
	if te.NodeName != "streamer" {
		t.Errorf("NodeTimeoutError.NodeName = %q, want \"streamer\"", te.NodeName)
	}
}

func TestRunNode_TimedOutChildThatIgnoresItsContextDeliversNoOutput(t *testing.T) {
	lateTaken := true
	child := newFaninNode("deaf_child", NodeConfig{Timeout: 20 * time.Millisecond}, func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		ctx.Actions().StateDelta = map[string]any{"k": "v"}
		<-ctx.Done()
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = "late"
		lateTaken = yield(ev, nil)
	})
	var childErr error
	w := dynamicWorkflow(t, defaultNodeConfig, func(ctx agent.Context) (string, error) {
		if _, err := RunNode[string](ctx, child, nil); err != nil {
			childErr = err
			return "fallback", nil
		}
		return "child succeeded", nil
	})

	got := observe(w.Run(boundedSeededCtx(t)))

	if got.err != nil {
		t.Fatalf("Run error = %v, want nil", got.err)
	}
	if want := []any{"fallback"}; !reflect.DeepEqual(got.outputs, want) {
		t.Errorf("outputs in the stream = %#v, want %#v", got.outputs, want)
	}
	if lateTaken {
		t.Error("yield of the late output event = true, want false")
	}
	if want := map[string]any{"k": "v"}; !reflect.DeepEqual(got.state, want) {
		t.Errorf("state deltas in the stream = %v, want %v", got.state, want)
	}
	var te *NodeTimeoutError
	if !errors.As(childErr, &te) {
		t.Fatalf("RunNode error = %v, want a *NodeTimeoutError", childErr)
	}
	if te.NodeName != "deaf_child" || te.Timeout != 20*time.Millisecond {
		t.Errorf("RunNode error = {%q, %v}, want {\"deaf_child\", 20ms}", te.NodeName, te.Timeout)
	}
}

func TestScheduler_TimedOutNodeKeepsTheStateItWroteBeforeItsLateEvent(t *testing.T) {
	ign := timedNode("ign", 20*time.Millisecond, func(ctx agent.Context, _ any) (string, error) {
		ctx.Actions().StateDelta = map[string]any{"k": "v"}
		<-ctx.Done()
		return "done", nil
	})
	w := mustNew(t, []Edge{{From: Start, To: ign}})

	got := observe(w.Run(boundedSeededCtx(t)))

	if want := map[string]any{"k": "v"}; !reflect.DeepEqual(got.state, want) {
		t.Errorf("state deltas in the stream = %v, want %v", got.state, want)
	}
	if got.outputs != nil {
		t.Errorf("outputs in the stream = %#v, want none", got.outputs)
	}
	if _, ok := errors.AsType[*NodeTimeoutError](got.err); !ok {
		t.Errorf("Run error = %v, want a *NodeTimeoutError", got.err)
	}
}
