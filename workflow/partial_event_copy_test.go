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
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// partialThenFinalNode yields one partial event and then one non-partial
// event, and keeps both pointers so the test can inspect what the node
// still holds.
type partialThenFinalNode struct {
	BaseNode
	partial, final *session.Event
}

func (n *partialThenFinalNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.partial = session.NewEvent(ctx, ctx.InvocationID())
		n.partial.LLMResponse = model.LLMResponse{
			Partial: true,
			Content: &genai.Content{Parts: []*genai.Part{{Text: "chunk"}}},
		}
		n.partial.NodeInfo = &session.NodeInfo{}
		if !yield(n.partial, nil) {
			return
		}
		n.final = session.NewEvent(ctx, ctx.InvocationID())
		n.final.LLMResponse = model.LLMResponse{
			Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "done"}}},
		}
		yield(n.final, nil)
	}
}

// TestRunNode_PartialEventIsCopiedForConsumer checks that the consumer
// receives its own copy of a partial event, so neither the scheduler nor
// the consumer writes to the event the node yielded, while a non-partial
// event keeps its pointer identity.
func TestRunNode_PartialEventIsCopiedForConsumer(t *testing.T) {
	node := &partialThenFinalNode{BaseNode: NewBaseNode("streamer", "", defaultNodeConfig)}
	w := mustNew(t, Chain(Start, node))

	var gotPartial, gotFinal *session.Event
	for ev, err := range w.Run(newSeededMockCtx(t)) {
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if ev.LLMResponse.Partial {
			gotPartial = ev
			ev.Author = "consumer"
			ev.Timestamp = time.Unix(1, 0)
			ev.Actions.Compaction = &session.EventCompaction{}
		} else if gotFinal == nil {
			// The first non-partial event is the node's; the workflow
			// emits its own events after it.
			gotFinal = ev
		}
	}

	if gotPartial == nil || gotFinal == nil {
		t.Fatalf("got partial %t, final %t; want both", gotPartial != nil, gotFinal != nil)
	}
	if gotPartial == node.partial {
		t.Errorf("received partial is the pointer the node yielded; want a copy")
	}
	if gotFinal != node.final {
		t.Errorf("received non-partial is a different pointer; want the one the node yielded")
	}
	if node.partial.Author == "consumer" || node.partial.Timestamp.Equal(time.Unix(1, 0)) || node.partial.Actions.Compaction != nil {
		t.Errorf("consumer write reached the node's partial: Author %q, Timestamp %v, Compaction %v",
			node.partial.Author, node.partial.Timestamp, node.partial.Actions.Compaction)
	}
	if node.partial.Content.Role != "" {
		t.Errorf("node's partial Content.Role = %q, want it left empty; the scheduler must set it on the copy", node.partial.Content.Role)
	}
	if gotPartial.Content.Role == "" {
		t.Errorf("received partial Content.Role is empty; want the scheduler default")
	}
	if node.partial.NodeInfo.Path != "" {
		t.Errorf("node's partial NodeInfo.Path = %q, want it left empty; the scheduler must set it on the copy", node.partial.NodeInfo.Path)
	}
}
