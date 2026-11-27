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

package agenttool_test

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// A tool that panics two agents down reaches the parent's flow wrapped twice:
// the workflow's ToolNode wraps what the tool returned, and this package wraps
// what the sub-agent run returned. The parent's function response must still
// be the generic message, because the panic is recognized at any wrapping
// depth. Everywhere else the flow sees the panic wrapped at most once, so
// without this the depth the message promises is not exercised.
func TestAgentTool_PanicInNestedWorkflowToolStaysOutOfTheResponse(t *testing.T) {
	const panicValue = "secret-user-data-123"

	inner, err := functiontool.New(functiontool.Config{Name: "inner"}, func(agent.Context, map[string]any) (map[string]any, error) {
		panic(panicValue)
	})
	if err != nil {
		t.Fatal(err)
	}
	node, err := workflow.NewToolNode(inner, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	// The sub-agent is reached with the request text, which the tool node
	// cannot take as its arguments.
	conv := workflow.NewFunctionNode("conv", func(_ agent.Context, in any) (map[string]any, error) {
		return map[string]any{"in": fmt.Sprint(in)}, nil
	}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{
		Name:        "wf",
		Description: "Runs the inner tool.",
		Edges:       workflow.Chain(workflow.Start, conv, node),
	})
	if err != nil {
		t.Fatal(err)
	}
	m := &testutil.MockModel{Responses: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "wf", Args: map[string]any{"request": "go"}}}}},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	root, err := llmagent.New(llmagent.Config{Name: "root", Model: m, Tools: []tool.Tool{agenttool.New(wf, nil)}})
	if err != nil {
		t.Fatal(err)
	}

	var events []*session.Event
	var responses []map[string]any
	for ev, err := range testutil.NewTestAgentRunner(t, root).Run(t, "s1", "hi") {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, part := range ev.LLMResponse.Content.Parts {
			if part.FunctionResponse != nil && part.FunctionResponse.Name == "wf" {
				responses = append(responses, part.FunctionResponse.Response)
			}
		}
	}

	if len(responses) != 1 {
		t.Fatalf("the run produced %d function responses for the sub-agent, want 1", len(responses))
	}
	want := map[string]any{"error": "Invoking `wf()` failed with an internal error."}
	if diff := cmp.Diff(want, responses[0]); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	// Nothing the run records may carry the panic value or the stack: the
	// model, the session and REST clients all read these events. The whole
	// event is serialized rather than its content formatted, because fmt
	// prints the parts behind a *genai.Content as addresses, which would make
	// this scan pass whatever they hold.
	var scanned strings.Builder
	for _, ev := range events {
		encoded, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("marshaling an event of the run failed: %v", err)
		}
		scanned.Write(encoded)
	}
	text := scanned.String()
	// What the scan must find keeps it honest: a scan that reads none of the
	// part content would pass the leak check below for free.
	if !strings.Contains(text, want["error"].(string)) {
		t.Fatalf("the scanned events do not contain the function response, so the leak check below reads nothing")
	}
	for _, leaked := range []string{panicValue, "goroutine", "PanicError"} {
		if strings.Contains(text, leaked) {
			t.Errorf("the events of the run leak %q", leaked)
		}
	}
}
