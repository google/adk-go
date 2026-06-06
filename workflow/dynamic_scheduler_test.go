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
	"strconv"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestValidateCustomRunID(t *testing.T) {
	tests := []struct {
		id      string
		wantErr bool
	}{
		{"", true},    // empty
		{"123", true}, // purely numeric
		{"0", true},   // purely numeric
		{"a/b", true}, // contains /
		{"a@b", true}, // contains @
		{"order-7", false},
		{"abc", false},
		{"v2-attempt-3", false},
		{"7a", false}, // mixed
	}
	for _, tc := range tests {
		t.Run(tc.id, func(t *testing.T) {
			err := validateCustomRunID(tc.id)
			if (err != nil) != tc.wantErr {
				t.Errorf("validateCustomRunID(%q) err = %v, wantErr = %v", tc.id, err, tc.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidRunID) {
				t.Errorf("validateCustomRunID(%q) err = %v, want errors.Is ErrInvalidRunID", tc.id, err)
			}
		})
	}
}

func TestSubScheduler_Counter_AutoIncrementsPerChildName(t *testing.T) {
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "parent", noopEmit)

	for i := 1; i <= 3; i++ {
		got, err := sub.resolveRunID("childA", "")
		if err != nil {
			t.Fatalf("resolveRunID childA #%d: %v", i, err)
		}
		if got != strconv.Itoa(i) {
			t.Errorf("childA #%d got %q, want %q", i, got, strconv.Itoa(i))
		}
	}
	// Independent counter per child name.
	got, _ := sub.resolveRunID("childB", "")
	if got != "1" {
		t.Errorf("childB first invocation got %q, want \"1\"", got)
	}
}

func TestSubScheduler_Counter_ConcurrentSafe(t *testing.T) {
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "parent", noopEmit)
	const goroutines = 64

	var wg sync.WaitGroup
	wg.Add(goroutines)
	seen := sync.Map{}
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			id, err := sub.resolveRunID("worker", "")
			if err != nil {
				t.Errorf("resolveRunID: %v", err)
				return
			}
			if _, dup := seen.LoadOrStore(id, struct{}{}); dup {
				t.Errorf("duplicate run id %q", id)
			}
		}()
	}
	wg.Wait()
}

func TestSubScheduler_Counter_CustomIDDoesNotIncrement(t *testing.T) {
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "parent", noopEmit)

	if _, err := sub.resolveRunID("c", "order-1"); err != nil {
		t.Fatalf("custom id: %v", err)
	}
	got, _ := sub.resolveRunID("c", "")
	if got != "1" {
		t.Errorf("auto id after custom got %q, want \"1\" (custom must not bump counter)", got)
	}
}

func TestSubScheduler_RunNode_FreshExecution(t *testing.T) {
	child := newStubNode("greeter", "hello")
	var forwarded []*session.Event
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "wf", func(ev *session.Event) error {
		forwarded = append(forwarded, ev)
		return nil
	})

	out, err := sub.runNode(child, "world", runNodeOptions{})
	if err != nil {
		t.Fatalf("runNode: %v", err)
	}
	if out != "hello" {
		t.Errorf("output = %v, want \"hello\"", out)
	}
	if len(forwarded) != 1 {
		t.Fatalf("forwarded events = %d, want 1", len(forwarded))
	}
	if forwarded[0].Output != "hello" {
		t.Errorf("forwarded event Output = %v, want \"hello\"", forwarded[0].Output)
	}
}

func TestSubScheduler_RunNode_CustomIDInPath(t *testing.T) {
	child := newStubNode("processor", nil)
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "wf", noopEmit)

	if _, err := sub.runNode(child, nil, runNodeOptions{customRunID: "order-42"}); err != nil {
		t.Fatalf("runNode: %v", err)
	}
	// The child must have observed its NodeContext populated with the
	// composite path; verify via the stub's captured context.
	if got, want := child.lastPath, "wf/processor@order-42"; got != want {
		t.Errorf("child Path() = %q, want %q", got, want)
	}
}

func TestSubScheduler_RunNode_HITLReturnsInterrupted(t *testing.T) {
	child := newRequestInputNode("asker", "approve?")
	var forwarded []*session.Event
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "wf", func(ev *session.Event) error {
		forwarded = append(forwarded, ev)
		return nil
	})

	_, err := sub.runNode(child, nil, runNodeOptions{})
	if !errors.Is(err, ErrNodeInterrupted) {
		t.Fatalf("err = %v, want ErrNodeInterrupted", err)
	}
	var nre *NodeRunError
	if !errors.As(err, &nre) {
		t.Fatalf("err is not *NodeRunError: %v", err)
	}
	if nre.ChildPath != "wf/asker@1" {
		t.Errorf("ChildPath = %q, want %q", nre.ChildPath, "wf/asker@1")
	}
	if len(forwarded) != 1 || forwarded[0].RequestedInput == nil {
		t.Errorf("forwarded events = %+v, want 1 RequestedInput event", forwarded)
	}
}

func TestSubScheduler_RunNode_ErrorWinsOverInterrupt(t *testing.T) {
	child := newInterruptThenFailNode("flaky")
	sub := newDynamicSubScheduler(newTopLevelCtx(t), "wf", noopEmit)

	_, err := sub.runNode(child, nil, runNodeOptions{})
	if !errors.Is(err, ErrNodeFailed) {
		t.Errorf("err = %v, want ErrNodeFailed", err)
	}
	if errors.Is(err, ErrNodeInterrupted) {
		t.Errorf("err = %v; ErrNodeInterrupted must not leak when child fails after RequestInput", err)
	}
}

// =============================================================================
// Test fixtures and helpers
// =============================================================================

func noopEmit(*session.Event) error { return nil }

func newTopLevelCtx(t *testing.T) agent.Context {
	t.Helper()
	return agent.NewNodeContext(newMockCtx(t), "", "", nil, nil, nil)
}

// stubNode emits one Event{Output: out} and exits.
type stubNode struct {
	BaseNode
	out        any
	lastPath   string
	lastBranch string
}

func newStubNode(name string, out any) *stubNode {
	return &stubNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		out:      out,
	}
}

func (n *stubNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	n.lastPath = ctx.Path()
	n.lastBranch = ctx.Branch()
	out := n.out
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{Output: out}, nil)
	}
}

// messageAsOutputNode emits a final model-text event whose content IS
// its output (NodeInfo.MessageAsOutput set, Event.Output nil), like an
// LlmAgent node in single_turn mode.
type messageAsOutputNode struct {
	BaseNode
	text string
}

func newMessageAsOutputNode(name, text string) *messageAsOutputNode {
	return &messageAsOutputNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		text:     text,
	}
}

func (n *messageAsOutputNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	text := n.text
	return func(yield func(*session.Event, error) bool) {
		ev := &session.Event{NodeInfo: &session.NodeInfo{MessageAsOutput: true}}
		ev.LLMResponse.Content = &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: text}},
		}
		yield(ev, nil)
	}
}

// requestInputNode emits one RequestedInput event and exits cleanly.
type requestInputNode struct {
	BaseNode
	message string
}

func newRequestInputNode(name, msg string) *requestInputNode {
	return &requestInputNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		message:  msg,
	}
}

func (n *requestInputNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{
			RequestedInput: &session.RequestInput{
				InterruptID: "iid-1",
				Message:     n.message,
			},
		}, nil)
	}
}

// interruptThenFailNode yields RequestedInput, then yields an error.
type interruptThenFailNode struct{ BaseNode }

func newInterruptThenFailNode(name string) *interruptThenFailNode {
	return &interruptThenFailNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

func (n *interruptThenFailNode) Run(agent.Context, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if !yield(&session.Event{
			RequestedInput: &session.RequestInput{InterruptID: "iid", Message: "ask"},
		}, nil) {
			return
		}
		yield(nil, errors.New("boom"))
	}
}
