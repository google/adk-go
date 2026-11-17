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
	"context"
	"fmt"
	"iter"
	"runtime"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// stopTestNodeKey tags an event with the name of the static node it
// belongs to, so a consumer can tell node events from the scheduler's
// own completion events and knows whose event it stopped at.
const stopTestNodeKey = "stop_test_node"

func stopTestEvent(node string) *session.Event {
	return &session.Event{LLMResponse: model.LLMResponse{
		CustomMetadata: map[string]any{stopTestNodeKey: node},
	}}
}

func stopTestNodeOf(ev *session.Event) (string, bool) {
	if ev == nil {
		return "", false
	}
	name, ok := ev.CustomMetadata[stopTestNodeKey].(string)
	return name, ok
}

// wentOnNode yields one tagged, non-partial event and counts the runs
// in which its yield returned true, that is, in which it went on past
// the event.
type wentOnNode struct {
	BaseNode
	wentOn *atomic.Int32
}

func newWentOnNode(name string) *wentOnNode {
	return &wentOnNode{BaseNode: NewBaseNode(name, "", defaultNodeConfig), wentOn: &atomic.Int32{}}
}

func (n *wentOnNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if yield(stopTestEvent(n.Name()), nil) {
			n.wentOn.Add(1)
		}
	}
}

// stopTestKind builds a node named name that sends one tagged event
// through the static scheduler's handshake and adds 1 to wentOn when the
// node's code is told that the event was taken.
type stopTestKind struct {
	name  string
	build func(t *testing.T, name string, wentOn *atomic.Int32) Node
}

var stopTestKinds = []stopTestKind{
	{
		name: "plain node",
		build: func(_ *testing.T, name string, wentOn *atomic.Int32) Node {
			n := newWentOnNode(name)
			n.wentOn = wentOn
			return n
		},
	},
	{
		name: "emitting function node",
		build: func(_ *testing.T, name string, wentOn *atomic.Int32) Node {
			return NewEmittingFunctionNode(name, func(_ agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				if emit(stopTestEvent(name)) == nil {
					wentOn.Add(1)
				}
				return nil, nil
			}, defaultNodeConfig)
		},
	},
	{
		name: "dynamic child",
		build: func(_ *testing.T, name string, wentOn *atomic.Int32) Node {
			child := NewEmittingFunctionNode(name+"_child", func(_ agent.Context, _ any, emit func(*session.Event) error) (any, error) {
				if emit(stopTestEvent(name)) == nil {
					wentOn.Add(1)
				}
				return nil, nil
			}, defaultNodeConfig)
			return NewDynamicNode(name, func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
				_, err := RunNode[any](ctx, child, nil)
				return nil, err
			}, defaultNodeConfig)
		},
	},
	{
		name: "agent node",
		build: func(t *testing.T, name string, wentOn *atomic.Int32) Node {
			a, err := agent.New(agent.Config{
				Name: name,
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if yield(stopTestEvent(name), nil) {
							wentOn.Add(1)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New: %v", err)
			}
			n, err := NewAgentNode(a, defaultNodeConfig)
			if err != nil {
				t.Fatalf("NewAgentNode: %v", err)
			}
			return n
		},
	},
}

// newHandshakeScheduler returns a scheduler over Start -> a, Start -> b
// whose event queue is unbuffered, with Start triggered. With one P,
// the producer that sends second is still blocked in its send when the
// consumer handles the first event, so it reaches its handshake select
// only after the consumer has handled its event and released the
// handshake.
func newHandshakeScheduler(t *testing.T, ctx agent.InvocationContext, a, b Node) *scheduler {
	t.Helper()
	w := mustNew(t, []Edge{{From: Start, To: a}, {From: Start, To: b}})
	s := newScheduler(agent.Promote(ctx), w.graph, 0)
	s.eventQueue = make(chan queueItem)
	s.state.EnsureNode(Start.Name()).Input = nil
	s.scheduleNode(Start, nil, "", "")
	return s
}

// TestScheduler_StoppedConsumerDoesNotLetNodeGoOn checks that a node
// whose event the caller did not take is not told that the event was
// taken, and ends as NodeCancelled, for every kind of node that sends its
// events through the static scheduler's handshake.
func TestScheduler_StoppedConsumerDoesNotLetNodeGoOn(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for _, kind := range stopTestKinds {
		t.Run(kind.name, func(t *testing.T) {
			for i := range 64 {
				wentOn := map[string]*atomic.Int32{"a": {}, "b": {}}
				s := newHandshakeScheduler(t, newMockCtx(t),
					kind.build(t, "a", wentOn["a"]), kind.build(t, "b", wentOn["b"]))

				nodeEvents := 0
				stoppedAt := ""
				s.run(func(ev *session.Event, _ error) bool {
					name, ok := stopTestNodeOf(ev)
					if !ok {
						return true
					}
					nodeEvents++
					if nodeEvents == 1 {
						// Let the other producer block in its send.
						runtime.Gosched()
						return true
					}
					stoppedAt = name
					return false
				}, false)
				s.wg.Wait()

				if stoppedAt == "" {
					t.Fatalf("iteration %d: the consumer saw %d node events, want 2", i, nodeEvents)
				}
				if got := wentOn[stoppedAt].Load(); got != 0 {
					t.Fatalf("iteration %d: node %q went on past the event the consumer did not take %d times, want 0", i, stoppedAt, got)
				}
				if got := s.state.Nodes[stoppedAt].Status; got != NodeCancelled {
					t.Fatalf("iteration %d: node %q status = %v, want NodeCancelled", i, stoppedAt, got)
				}
			}
		})
	}
}

// TestScheduler_CanceledContextDoesNotLetNodeGoOn checks that a node
// whose event was taken while the invocation context was being canceled
// is not told that the event was taken.
func TestScheduler_CanceledContextDoesNotLetNodeGoOn(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for i := range 64 {
		ctx, cancel := context.WithCancel(t.Context())
		a, b := newWentOnNode("a"), newWentOnNode("b")
		wentOn := map[string]*atomic.Int32{"a": a.wentOn, "b": b.wentOn}
		s := newHandshakeScheduler(t, &MockInvocationContext{Context: ctx}, a, b)

		nodeEvents := 0
		canceledAt := ""
		s.run(func(ev *session.Event, _ error) bool {
			name, ok := stopTestNodeOf(ev)
			if !ok {
				return true
			}
			nodeEvents++
			if nodeEvents == 1 {
				runtime.Gosched()
				return true
			}
			// The event's producer was dequeued, so it is past its send
			// and reaches its handshake select after this cancel.
			canceledAt = name
			cancel()
			return true
		}, false)
		s.wg.Wait()
		cancel()

		if canceledAt == "" {
			t.Fatalf("iteration %d: the consumer saw %d node events, want 2", i, nodeEvents)
		}
		if got := wentOn[canceledAt].Load(); got != 0 {
			t.Fatalf("iteration %d: node %q went on past an event taken after the cancel %d times, want 0", i, canceledAt, got)
		}
	}
}

// TestScheduler_StoppedConsumerManyNodesNoneGoOn checks that the node
// whose event the caller did not take is canceled before its handshake
// is released. With many nodes, cancelAll takes long enough that a node
// released first would wake on another P and go on before its own
// cancel.
func TestScheduler_StoppedConsumerManyNodesNoneGoOn(t *testing.T) {
	const nodes = 1500
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(8))

	for i := range 30 {
		edges := make([]Edge, 0, nodes)
		wentOn := make(map[string]*atomic.Int32, nodes)
		for j := range nodes {
			n := newWentOnNode(fmt.Sprintf("n%d", j))
			wentOn[n.Name()] = n.wentOn
			edges = append(edges, Edge{From: Start, To: n})
		}
		w := mustNew(t, edges)
		s := newScheduler(agent.Promote(newMockCtx(t)), w.graph, 0)
		s.state.EnsureNode(Start.Name()).Input = nil
		s.scheduleNode(Start, nil, "", "")

		nodeEvents := 0
		stoppedAt := ""
		s.run(func(ev *session.Event, _ error) bool {
			name, ok := stopTestNodeOf(ev)
			if !ok {
				return true
			}
			nodeEvents++
			if nodeEvents < nodes {
				return true
			}
			stoppedAt = name
			return false
		}, false)
		s.wg.Wait()

		if stoppedAt == "" {
			t.Fatalf("iteration %d: the consumer saw %d node events, want %d", i, nodeEvents, nodes)
		}
		if got := wentOn[stoppedAt].Load(); got != 0 {
			t.Fatalf("iteration %d: node %q went on past the event the consumer did not take %d times, want 0", i, stoppedAt, got)
		}
	}
}

// silentStopTestNode ends without events or output, so the scheduler
// yields a completion event for it.
type silentStopTestNode struct{ BaseNode }

func (n *silentStopTestNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(func(*session.Event, error) bool) {}
}

// gatedWentOnNode waits for release, then yields one tagged, non-partial
// event and counts the runs in which its yield returned true.
type gatedWentOnNode struct {
	BaseNode
	release chan struct{}
	wentOn  atomic.Int32
}

func (n *gatedWentOnNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		select {
		case <-n.release:
		case <-ctx.Done():
			return
		}
		if yield(stopTestEvent(n.Name()), nil) {
			n.wentOn.Add(1)
		}
	}
}

// TestScheduler_StopAtCompletionEventDoesNotLetNodeGoOn checks that a
// caller that stops at a node's completion event stops every other node:
// a node whose event enters the queue after the stop is not told that the
// event was taken, and the event does not reach the caller. It fails if
// the scheduler releases a drained event's handshake without canceling
// the nodes after the caller stops at a completion event.
func TestScheduler_StopAtCompletionEventDoesNotLetNodeGoOn(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	for i := range 64 {
		a := &silentStopTestNode{BaseNode: NewBaseNode("a", "", defaultNodeConfig)}
		b := &gatedWentOnNode{BaseNode: NewBaseNode("b", "", defaultNodeConfig), release: make(chan struct{})}
		s := newHandshakeScheduler(t, newMockCtx(t), a, b)

		stoppedAt := ""
		bEvents := 0
		s.run(func(ev *session.Event, _ error) bool {
			if name, ok := stopTestNodeOf(ev); ok && name == "b" {
				bEvents++
				return true
			}
			if ev == nil || ev.CustomMetadata[workflowNodeCompletedKey] != true || stoppedAt != "" {
				return true
			}
			stoppedAt = ev.NodeInfo.Path
			// Let b send its event while the caller is still inside yield.
			close(b.release)
			runtime.Gosched()
			return false
		}, false)
		s.wg.Wait()

		if stoppedAt != "a@1" {
			t.Fatalf("iteration %d: the consumer stopped at the completion event of %q, want %q", i, stoppedAt, "a@1")
		}
		if bEvents != 0 {
			t.Fatalf("iteration %d: the consumer got %d events from b after it stopped, want 0", i, bEvents)
		}
		if got := b.wentOn.Load(); got != 0 {
			t.Fatalf("iteration %d: node b went on past an event the consumer did not take %d times, want 0", i, got)
		}
	}
}
