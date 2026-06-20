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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// runFunctionNodeOnce drives a FunctionNode for a single input and
// returns the sole emitted event.
func runFunctionNodeOnce[OUT any](t *testing.T, fn func(ctx agent.Context, input any) (OUT, error), input any) *session.Event {
	t.Helper()
	node := NewFunctionNode[any, OUT]("n", fn, defaultNodeConfig)
	exCtx := agent.NewNodeContext(&MockInvocationContext{sess: nil}, nil)

	var got *session.Event
	count := 0
	for ev, err := range node.Run(exCtx, input) {
		if err != nil {
			t.Fatalf("FunctionNode.Run: %v", err)
		}
		got = ev
		count++
	}
	if count != 1 {
		t.Fatalf("FunctionNode.Run emitted %d events, want 1", count)
	}
	return got
}

// TestFunctionNode_ContentOutputGoesToContent is the regression guard
// for the genai.Content branch: when a FunctionNode returns a
// *genai.Content (or genai.Content), the value must populate
// event.Content (a renderable message), not event.Output (an opaque
// any). Mirrors adk-python _function_node.py which maps a Content
// return to Event(content=...). Before the fix the Content landed in
// event.Output and the message did not render.
func TestFunctionNode_ContentOutputGoesToContent(t *testing.T) {
	t.Parallel()

	want := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{genai.NewPartFromText("hello from content")},
	}

	t.Run("pointer *genai.Content", func(t *testing.T) {
		t.Parallel()
		ev := runFunctionNodeOnce(t, func(ctx agent.Context, _ any) (*genai.Content, error) {
			return want, nil
		}, nil)

		if ev.Output != nil {
			t.Errorf("event.Output = %v, want nil; Content must not go to Output", ev.Output)
		}
		if diff := cmp.Diff(want, ev.Content); diff != "" {
			t.Errorf("event.Content mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("value genai.Content", func(t *testing.T) {
		t.Parallel()
		ev := runFunctionNodeOnce(t, func(ctx agent.Context, _ any) (genai.Content, error) {
			return *want, nil
		}, nil)

		if ev.Output != nil {
			t.Errorf("event.Output = %v, want nil; Content must not go to Output", ev.Output)
		}
		if diff := cmp.Diff(want, ev.Content); diff != "" {
			t.Errorf("event.Content mismatch (-want +got):\n%s", diff)
		}
	})
}

// TestFunctionNode_NonContentOutputGoesToOutput pins the complementary
// case: a non-Content return value populates event.Output and leaves
// event.Content nil.
func TestFunctionNode_NonContentOutputGoesToOutput(t *testing.T) {
	t.Parallel()

	want := map[string]any{"result": "ok"}
	ev := runFunctionNodeOnce(t, func(ctx agent.Context, _ any) (map[string]any, error) {
		return want, nil
	}, nil)

	if ev.Content != nil {
		t.Errorf("event.Content = %v, want nil; non-Content output must not set Content", ev.Content)
	}
	if diff := cmp.Diff(want, ev.Output); diff != "" {
		t.Errorf("event.Output mismatch (-want +got):\n%s", diff)
	}
}
