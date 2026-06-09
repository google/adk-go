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
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// sliceEvents adapts a []*session.Event to session.Events for tests.
type sliceEvents []*session.Event

func (e sliceEvents) Len() int                { return len(e) }
func (e sliceEvents) At(i int) *session.Event { return e[i] }
func (e sliceEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

func modelEvent(path, text string, messageAsOutput bool) *session.Event {
	ev := &session.Event{
		NodeInfo: &session.NodeInfo{Path: path, MessageAsOutput: messageAsOutput},
	}
	ev.LLMResponse.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: text}},
	}
	return ev
}

// Resume derives output from the model message when an event is
// flagged MessageAsOutput with no explicit Output (adk-python parity).
func TestCollectNodeOutputs_MessageAsOutput(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", true)}

	outputs, completed := collectNodeOutputs(events, nodes)

	if got, want := outputs["talky"], "Hello, world!"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
	if !completed["talky"] {
		t.Errorf("completed[talky] = false, want true")
	}
}

func TestCollectNodeOutputs_MessageNotFlagged(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", false)}

	outputs, _ := collectNodeOutputs(events, nodes)

	if _, ok := outputs["talky"]; ok {
		t.Errorf("outputs[talky] = %#v, want absent", outputs["talky"])
	}
}

func TestCollectNodeOutputs_ExplicitOutputWins(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	ev := modelEvent("talky", "from message", true)
	ev.Output = "explicit"
	events := sliceEvents{ev}

	outputs, _ := collectNodeOutputs(events, nodes)

	if got, want := outputs["talky"], "explicit"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
}

// A delegated child's output is attributed on resume to the static
// owners of every path in OutputFor, so a delegating ancestor recovers
// it without re-emitting (adk-python output_for parity).
func TestCollectNodeOutputs_OutputForAttributesAncestors(t *testing.T) {
	nodes := map[string]Node{
		"child": newDummyNode("child"),
		"outer": newDummyNode("outer"),
	}

	ev := &session.Event{
		Output: "delegated",
		NodeInfo: &session.NodeInfo{
			Path:      "child/gc@1",
			OutputFor: []string{"child/gc@1", "outer/child@1"},
		},
	}

	outputs, _ := collectNodeOutputs(sliceEvents{ev}, nodes)

	if got, want := outputs["child"], "delegated"; got != want {
		t.Errorf("outputs[child] = %#v, want %q", got, want)
	}
	if got, want := outputs["outer"], "delegated"; got != want {
		t.Errorf("outputs[outer] = %#v, want %q (ancestor not attributed)", got, want)
	}
}
