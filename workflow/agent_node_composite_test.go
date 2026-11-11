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

	"google.golang.org/adk/v2/agent"
)

// resumedCtx answers ResumedInput for the interrupt IDs in resumed only.
type resumedCtx struct {
	agent.ContextMock
	userContent *genai.Content
	resumed     map[string]bool
}

func (c *resumedCtx) UserContent() *genai.Content { return c.userContent }

func (c *resumedCtx) ResumedInput(id string) (any, bool) { return nil, c.resumed[id] }

func TestWithResumedResponses(t *testing.T) {
	response := func(id string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: WorkflowInputFunctionCallName}}
	}
	ctx := &resumedCtx{
		userContent: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{response("mine"), response("other node's"), {Text: "note"}}},
		resumed:     map[string]bool{"mine": true},
	}
	input := genai.NewContentFromText("node input", genai.RoleUser)

	got := withResumedResponses(ctx, input)

	want := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "node input"}, response("mine")}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("content mismatch (-want +got):\n%s", diff)
	}
	if len(input.Parts) != 1 {
		t.Error("withResumedResponses modified the node input")
	}
	if got := withResumedResponses(&resumedCtx{userContent: ctx.userContent}, input); got != input {
		t.Error("withResumedResponses changed the input of a node that is not resumed")
	}
}
