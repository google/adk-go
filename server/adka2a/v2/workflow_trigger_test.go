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

package adka2a

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// TestExecutor_WorkflowPauseKeepsPredecessorOutput checks that pausing a
// workflow node does not send the output of the node that triggered it to the
// A2A client. That output only lives in Event.Output, which adka2a does not
// forward. The scheduler's trigger record on the pause names the node and
// carries nothing else.
func TestExecutor_WorkflowPauseKeepsPredecessorOutput(t *testing.T) {
	const output = "lookup-output-stays-on-the-server"
	lookup := workflow.NewFunctionNode("lookup", func(agent.Context, any) (string, error) { return output, nil }, workflow.NodeConfig{})
	// ask pauses the way a model reply does: some text and a long-running
	// call in one event. The text goes out as an artifact that carries the
	// event's custom metadata.
	ask := workflow.NewEmittingFunctionNode("ask", func(ctx agent.Context, _ string, emit func(*session.Event) error) (any, error) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Content = genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromText("Please approve the lookup."),
			{FunctionCall: &genai.FunctionCall{ID: "approval", Name: "approve"}},
		}, genai.RoleModel)
		ev.LongRunningToolIDs = []string{"approval"}
		return nil, emit(ev)
	}, workflow.NodeConfig{RerunOnResume: new(true)})
	wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: []workflow.Edge{
		{From: workflow.Start, To: lookup},
		{From: lookup, To: ask},
	}})
	if err != nil {
		t.Fatal("workflow construction failed")
	}
	executor := NewExecutor(ExecutorConfig{RunnerConfig: runner.Config{AppName: "wf", Agent: wf, SessionService: session.InMemoryService()}})
	req := &a2asrv.ExecutorContext{TaskID: a2a.NewTaskID(), ContextID: a2a.NewContextID(), Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("start"))}
	var sent strings.Builder
	for event, err := range executor.Execute(t.Context(), req) {
		if err != nil {
			t.Fatal("executor failed to dispatch workflow events")
		}
		data, err := json.Marshal(event)
		if err != nil {
			t.Fatal("A2A event does not encode as JSON")
		}
		sent.Write(data)
	}
	if strings.Contains(sent.String(), output) {
		t.Error("an A2A event carries the output of the node that triggered the paused node")
	}
	if !strings.Contains(sent.String(), `"adk.workflow.trigger.wf":"lookup"`) {
		t.Error("the pause sent to the A2A client has no trigger record naming lookup, so this test checks nothing")
	}
}
