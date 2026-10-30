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
	"errors"
	"iter"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

type panicAfterErrorNode struct {
	BaseNode
	node Node
}

func (n *panicAfterErrorNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		defer func() { panic("scripted iterator cleanup panic") }()
		for ev, err := range n.node.Run(ctx, input) {
			if err != nil {
				publishFailureInventory(ctx, n, failureInventory{known: true})
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

func TestFailureInventory_ActivationIsolation(t *testing.T) {
	parentNode := newDummyNode("parent")
	parent, parentRecorder := withFailureInventory(agent.Promote(newMockCtx(t)), parentNode)
	paths := []string{"parent/stable@1"}
	publishFailureInventory(parent, parentNode, failureInventory{known: true, paths: paths})
	paths[0] = "changed"
	if !reflect.DeepEqual(parentRecorder.snapshot().paths, []string{"parent/stable@1"}) {
		t.Fatal("publishing retained a mutable path slice")
	}
	copy := parentRecorder.snapshot()
	copy.paths[0] = "changed"
	if parentRecorder.snapshot().paths[0] != "parent/stable@1" {
		t.Fatal("inventory snapshot aliases the recorder")
	}

	var wg sync.WaitGroup
	for _, path := range []string{"parent/left@1", "parent/right@1"} {
		wg.Go(func() {
			childNode := newDummyNode("child")
			child, recorder := withFailureInventory(parent, childNode)
			if inventory := recorder.snapshot(); inventory.known || len(inventory.paths) != 0 {
				t.Error("new activation inherited its parent's inventory")
			}
			publishFailureInventory(child, childNode, failureInventory{known: true, paths: []string{path}})
			if !reflect.DeepEqual(recorder.snapshot().paths, []string{path}) {
				t.Error("parallel activations shared inventory")
			}
		})
	}
	wg.Wait()
	if !reflect.DeepEqual(parentRecorder.snapshot().paths, []string{"parent/stable@1"}) {
		t.Fatal("child activation overwrote its parent's inventory")
	}
}

func TestScheduler_FailureInventoryExecutors(t *testing.T) {
	for _, mode := range []string{"known_empty", "known_child", "unknown_function", "panic", "panic_after_error"} {
		t.Run(mode, func(t *testing.T) {
			parentNode := newDummyNode("parent")
			parent, recorder := withFailureInventory(agent.Promote(newMockCtx(t)), parentNode)
			publishFailureInventory(parent, parentNode, failureInventory{known: true, paths: []string{"parent/stale@1"}})
			failure := errors.New("scripted body failure")
			var n Node
			n = NewDynamicNode("body", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
				if mode == "known_child" {
					child := NewFunctionNode("leaf", func(agent.Context, any) (any, error) { return nil, failure }, NodeConfig{})
					return RunNode[any](ctx, child, nil)
				}
				if mode == "panic" {
					publishFailureInventory(ctx, n, failureInventory{known: true})
					panic("scripted panic")
				}
				return nil, failure
			}, NodeConfig{})
			switch mode {
			case "unknown_function":
				n = NewFunctionNode("body", func(agent.Context, any) (any, error) { return nil, failure }, NodeConfig{})
			case "panic_after_error":
				n = &panicAfterErrorNode{BaseNode: NewBaseNode("body", "", NodeConfig{}), node: n}
			}
			queue := make(chan queueItem, defaultEventQueueCapacity)
			var wg sync.WaitGroup
			wg.Add(1)
			go runNode(queue, &wg, "body", n, parent, nil)
			var completion completionItem
			for {
				switch item := (<-queue).(type) {
				case eventItem:
					if item.processed != nil {
						close(item.processed)
					}
				case completionItem:
					completion = item
					goto completed
				}
			}
		completed:
			wg.Wait()
			wantKnown := mode == "known_empty" || mode == "known_child"
			if completion.inventory.known != wantKnown {
				t.Fatal("executor lost known inventory or trusted an unknown outcome")
			}
			if mode == "known_child" {
				if !reflect.DeepEqual(completion.inventory.paths, []string{"body/leaf@1"}) || !errors.Is(completion.err, failure) {
					t.Fatalf("executor lost the failed child's path or original cause (paths: %v)", completion.inventory.paths)
				}
			} else if mode != "panic" && mode != "panic_after_error" && completion.err != failure {
				t.Fatal("executor replaced the business error")
			}
			if !reflect.DeepEqual(recorder.snapshot().paths, []string{"parent/stale@1"}) {
				t.Fatal("executor reused the parent's inventory recorder")
			}
		})
	}
}

func TestScheduler_FailureInventoryFirstError(t *testing.T) {
	for _, mode := range []string{"known", "unknown", "accumulator_error", "consumer_gone"} {
		t.Run(mode, func(t *testing.T) {
			first, second := newDummyNode("first"), newDummyNode("second")
			nested, err := NewWorkflowNode("nested", []Edge{{From: Start, To: first}, {From: Start, To: second}})
			if err != nil {
				t.Fatal("nested workflow construction failed")
			}
			ctx, recorder := withFailureInventory(agent.Promote(newMockCtx(t)), nested)
			s := newScheduler(ctx, nested.subWorkflow.graph, 0)
			s.runsByName["first"] = &nodeRun{nodePath: "first"}
			s.runsByName["second"] = &nodeRun{nodePath: "second"}
			failure := errors.New("scripted first failure")
			item := completionItem{nodeName: "first", err: failure}
			switch mode {
			case "known":
				item.inventory = failureInventory{known: true, paths: []string{"first/leaf@1"}}
			case "accumulator_error":
				item.err = nil
				item.inventory = failureInventory{known: true, paths: []string{"unrelated"}}
				s.runsByName["first"].err = failure
			}
			s.eventQueue <- item
			s.eventQueue <- completionItem{nodeName: "second", err: context.Canceled, inventory: failureInventory{known: true, paths: []string{"second/leaf@1"}}}
			var got error
			yield := func(_ *session.Event, err error) bool { got = err; return true }
			if mode == "consumer_gone" {
				s.run(nil, true)
			} else {
				s.run(yield, false)
				if got != failure {
					t.Fatal("first error was replaced while draining")
				}
			}
			inventory := recorder.snapshot()
			if inventory.known != (mode == "known") {
				t.Fatal("unknown or departed-consumer outcome acquired another node's inventory")
			}
			if mode == "known" && !reflect.DeepEqual(inventory.paths, []string{"first/leaf@1"}) {
				t.Fatal("sibling cancellation overwrote the first error's inventory")
			}
		})
	}
}

func TestWorkflowNode_FailureInventory(t *testing.T) {
	const innerPath = "parent/inner@stable"
	const leafPath = innerPath + "/body@1/leaf@stable"
	for _, mode := range []string{"known_empty", "leaf_failure", "recovered_leaf"} {
		t.Run(mode, func(t *testing.T) {
			failure := &identityBodyError{}
			var leafCalls atomic.Int32
			leaf := NewFunctionNode("leaf", func(agent.Context, any) (any, error) {
				if leafCalls.Add(1) == 1 {
					return nil, failure
				}
				return "recovered", nil
			}, NodeConfig{})
			body := NewDynamicNode("body", func(ctx agent.Context, _ any, _ func(*session.Event) error) (any, error) {
				if mode != "known_empty" {
					_, err := RunNode[any](ctx, leaf, nil, WithRunID("stable"))
					if mode == "leaf_failure" {
						return nil, err
					}
					if !errors.Is(err, failure) {
						t.Error("leaf fixture did not fail")
					}
					if _, err := RunNode[any](ctx, leaf, nil, WithRunID("stable")); err != nil {
						return nil, err
					}
				}
				return nil, failure
			}, NodeConfig{})
			inner, err := NewWorkflowNode("inner", []Edge{{From: Start, To: body}})
			if err != nil {
				t.Fatal("nested workflow construction failed")
			}
			var marker *session.Event
			sub := newDynamicSubScheduler(agent.Promote(newMockCtx(t)), "parent", func(ev *session.Event) error {
				if ev.NodeInfo != nil && ev.NodeInfo.Path == innerPath && workflowNodeOutcome(ev) != "" {
					marker = ev
				}
				return nil
			}).(*dynamicSubScheduler)
			_, err = sub.runNode(inner, nil, runNodeOptions{customRunID: "stable"})
			if !errors.Is(err, failure) || marker == nil {
				t.Fatal("nested workflow lost its original failure or outcome marker")
			}
			paths, ok := marker.CustomMetadata[workflowFailedChildPathsKey].([]any)
			want := []any{}
			if mode == "leaf_failure" {
				want = append(want, leafPath)
			}
			if !ok || !reflect.DeepEqual(paths, want) {
				t.Fatal("WorkflowNode lost precise inventory or widened it to the containing child")
			}
			inventory := sub.childFailures()
			if !inventory.known || len(inventory.paths) != len(want) {
				t.Fatal("outer scheduler retained a recovered leaf or lost known-empty inventory")
			}
			if mode == "leaf_failure" && inventory.paths[0] != leafPath {
				t.Fatal("outer scheduler did not inherit the precise failed leaf")
			}
		})
	}
}

func TestSubScheduler_PanicAfterErrorInventory(t *testing.T) {
	const childPath = "parent/body@stable"
	inner := NewDynamicNode("body", func(agent.Context, any, func(*session.Event) error) (any, error) {
		return nil, errors.New("scripted body failure")
	}, NodeConfig{})
	child := &panicAfterErrorNode{BaseNode: NewBaseNode("body", "", NodeConfig{}), node: inner}
	var marker *session.Event
	sub := newDynamicSubScheduler(agent.Promote(newMockCtx(t)), "parent", func(ev *session.Event) error {
		if ev.NodeInfo != nil && ev.NodeInfo.Path == childPath && workflowNodeOutcome(ev) != "" {
			marker = ev
		}
		return nil
	}).(*dynamicSubScheduler)
	panicked := false
	func() {
		defer func() { panicked = recover() != nil }()
		_, _ = sub.runNode(child, nil, runNodeOptions{customRunID: "stable"})
	}()
	if !panicked || marker == nil {
		t.Fatal("child panic was swallowed or its failure marker was lost")
	}
	if _, ok := marker.CustomMetadata[workflowFailedChildPathsKey]; ok {
		t.Fatal("iterator cleanup panic trusted the preceding error's inventory")
	}
	if _, ok := sub.failedPaths[childPath]; !ok {
		t.Fatal("panicked child was not conservatively invalidated")
	}
}

func TestFailureInventory_DirectRunsRemainUnknown(t *testing.T) {
	for _, mode := range []string{"handled", "handled_same_sentinel", "same_name", "parallel", "same_producer"} {
		t.Run(mode, func(t *testing.T) {
			failure := errors.New("scripted composite failure")
			childFailure := errors.New("scripted direct child failure")
			if mode == "handled_same_sentinel" {
				childFailure = failure
			}
			var node Node
			body := func(ctx agent.Context, input any) (any, error) {
				if input != nil {
					return nil, childFailure
				}
				child := NewDynamicNode("child", func(agent.Context, any, func(*session.Event) error) (any, error) {
					return nil, childFailure
				}, NodeConfig{})
				switch mode {
				case "same_name":
					child = NewDynamicNode("body", func(agent.Context, any, func(*session.Event) error) (any, error) {
						return nil, childFailure
					}, NodeConfig{})
				case "same_producer":
					child = node
				}
				consume := func() {
					for _, err := range child.Run(ctx, 1) {
						if err != childFailure {
							t.Error("direct child error was replaced")
						}
					}
				}
				if mode == "parallel" || mode == "same_producer" {
					var wg sync.WaitGroup
					wg.Go(consume)
					wg.Go(consume)
					wg.Wait()
				} else {
					consume()
				}
				return nil, failure
			}
			node = NewFunctionNode("body", body, NodeConfig{})
			if mode == "same_producer" {
				node = NewDynamicNode("body", func(ctx agent.Context, input any, _ func(*session.Event) error) (any, error) {
					return body(ctx, input)
				}, NodeConfig{})
			}
			queue := make(chan queueItem, 1)
			var wg sync.WaitGroup
			wg.Add(1)
			go runNode(queue, &wg, "body", node, agent.Promote(newMockCtx(t)), nil)
			completion, ok := (<-queue).(completionItem)
			wg.Wait()
			if !ok || completion.err != failure || completion.inventory.known {
				t.Fatal("direct node composition supplied an unrelated or ambiguous failure inventory")
			}
		})
	}
}
