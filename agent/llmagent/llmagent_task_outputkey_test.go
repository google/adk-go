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

package llmagent_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// runCoordinatorWithTask runs a chat coordinator whose only turn delegates to
// taskAgent, and returns the session state after the run.
func runCoordinatorWithTask(t *testing.T, taskAgent agent.Agent) map[string]any {
	t.Helper()
	coordLLM := &scriptedLLM{
		turns: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{
					ID:   "fc-task",
					Name: taskAgent.Name(),
					Args: map[string]any{"request": "do it"},
				},
			}}}},
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{genai.NewPartFromText("ok")}}},
		},
	}
	coord, err := llmagent.New(llmagent.Config{
		Name:      "coord",
		Model:     coordLLM,
		Mode:      llmagent.ModeChat,
		SubAgents: []agent.Agent{taskAgent},
	})
	if err != nil {
		t.Fatalf("llmagent.New(coord): %v", err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{Agent: coord, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s",
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"},
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	state := map[string]any{}
	for k, v := range resp.Session.State().All() {
		state[k] = v
	}
	return state
}

// A task agent delivers its result through finish_task. A text turn before
// that (a clarification question, say) is not the result and must not be
// written to OutputKey.
func TestTaskAgent_TextTurnDoesNotWriteOutputKey(t *testing.T) {
	t.Parallel()
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{genai.NewPartFromText("Which language do you want?")},
		}}},
	}
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		withOutputKey("task_out"),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)

	state := runCoordinatorWithTask(t, taskAgent)

	if got, ok := state["task_out"]; ok {
		t.Errorf("state[%q] = %v after a task agent's text turn, want it unset", "task_out", got)
	}
}

// When the task completes, OutputKey holds the finish_task output, the same
// value as the terminal event's Output: the value under the wrapper key for a
// wrapped primitive, otherwise the args map.
func TestTaskAgent_FinishTaskWritesOutputKey(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		schema *genai.Schema
		args   map[string]any
		want   any
	}{
		{
			name: "object output",
			args: map[string]any{"result": "summarized"},
			want: map[string]any{"result": "summarized"},
		},
		{
			name:   "wrapped primitive",
			schema: &genai.Schema{Type: genai.TypeString},
			args:   map[string]any{"result": "summarized"},
			want:   "summarized",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			taskLLM := &scriptedLLM{
				turns: []*model.LLMResponse{{Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID:   "finish-1",
						Name: "finish_task",
						Args: tc.args,
					}}},
				}}},
			}
			opts := []func(*llmagent.Config){
				withMode(llmagent.ModeTask),
				withOutputKey("task_out"),
				func(c *llmagent.Config) { c.Model = taskLLM },
			}
			if tc.schema != nil {
				opts = append(opts, withOutputSchema(tc.schema))
			}
			taskAgent := makeLLMAgent(t, "doer", opts...)

			state := runCoordinatorWithTask(t, taskAgent)

			got, ok := state["task_out"]
			if !ok {
				t.Fatalf("state[%q] is unset after finish_task, want the finish_task output", "task_out")
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("state[%q] mismatch (-want +got):\n%s", "task_out", diff)
			}
		})
	}
}

// Without an OutputKey, finishing a task writes nothing to state.
func TestTaskAgent_FinishTaskWithoutOutputKeyWritesNoState(t *testing.T) {
	t.Parallel()
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{{Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "finish-1",
				Name: "finish_task",
				Args: map[string]any{"result": "summarized"},
			}}},
		}}},
	}
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)

	state := runCoordinatorWithTask(t, taskAgent)

	if len(state) != 0 {
		t.Errorf("session state = %v, want it empty", state)
	}
}

// Content returned by a before-agent callback is not a task result either, so
// it must not reach OutputKey on a task agent.
func TestTaskAgent_BeforeAgentCallbackContentDoesNotWriteOutputKey(t *testing.T) {
	t.Parallel()
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		withOutputKey("task_out"),
		func(c *llmagent.Config) {
			c.BeforeAgentCallbacks = []agent.BeforeAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return genai.NewContentFromText("cached answer", genai.RoleModel), nil
				},
			}
		},
	)

	state := runCoordinatorWithTask(t, taskAgent)

	if got, ok := state["task_out"]; ok {
		t.Errorf("state[%q] = %v after a before-agent callback short-circuit on a task agent, want it unset", "task_out", got)
	}
}

// A finish_task call that fails validation is not a result. OutputKey must stay
// unset until a call is accepted, so the write belongs on the success response,
// not on the call.
func TestTaskAgent_RejectedFinishTaskDoesNotWriteOutputKey(t *testing.T) {
	t.Parallel()
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
		Required:   []string{"answer"},
	}
	taskLLM := &scriptedLLM{
		turns: []*model.LLMResponse{{Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "finish-bad",
				Name: "finish_task",
				Args: map[string]any{"wrong_field": "oops"},
			}}},
		}}},
	}
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		withOutputKey("task_out"),
		withOutputSchema(schema),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)

	state := runCoordinatorWithTask(t, taskAgent)

	if got, ok := state["task_out"]; ok {
		t.Errorf("state[%q] = %v after only a rejected finish_task, want it unset", "task_out", got)
	}
}

// A SequentialAgent calls Run on its children, not RunNode, so a task agent
// under one does not go through runTask and nothing writes the finish_task
// output. OutputKey stays unset, and the text turn after finish_task does not
// fill it either. adk-python behaves the same way. This pins the limit of the
// OutputKey behaviour for task agents run through Run.
func TestSequentialParentTaskChildLeavesOutputKeyUnset(t *testing.T) {
	t.Parallel()
	taskLLM := &scriptedLLM{turns: []*model.LLMResponse{
		{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "finish-1", Name: "finish_task", Args: map[string]any{"result": "summarized"},
		}}}}},
		{Content: &genai.Content{Role: "model", Parts: []*genai.Part{genai.NewPartFromText("all done")}}},
	}}
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		withOutputKey("task_out"),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{taskAgent}}})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{Agent: seq, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"}, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	if v, err := resp.Session.State().Get("task_out"); err == nil {
		t.Errorf("state[%q] = %v under a sequential parent, want it unset", "task_out", v)
	}
}

// A task agent cannot sit on the edges of a workflowagent graph directly; it is
// dispatched with workflow.RunNode from inside another node. That enters the
// task agent through its RunNode method, so runTask runs and OutputKey holds the
// finish_task output, unlike under a SequentialAgent.
func TestWorkflowAgentDispatchedTaskWritesOutputKey(t *testing.T) {
	t.Parallel()
	taskLLM := &scriptedLLM{turns: []*model.LLMResponse{
		{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "finish-1", Name: "finish_task", Args: map[string]any{"result": "summarized"},
		}}}}},
	}}
	taskAgent := makeLLMAgent(t, "doer",
		withMode(llmagent.ModeTask),
		withOutputKey("task_out"),
		func(c *llmagent.Config) { c.Model = taskLLM },
	)
	taskNode, err := workflow.NewAgentNode(taskAgent, workflow.NodeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	router := workflow.NewDynamicNode("router",
		func(ctx agent.Context, in string, _ func(*session.Event) error) (string, error) {
			_, err := workflow.RunNode[any](ctx, taskNode, in)
			return "routed", err
		}, workflow.NodeConfig{})
	wf, err := workflowagent.New(workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, router)})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{Agent: wf, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(t.Context(), "u", "s", &genai.Content{Parts: []*genai.Part{genai.NewPartFromText("go")}, Role: "user"}, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run: %v", err)
		}
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := resp.Session.State().Get("task_out")
	if err != nil {
		t.Fatalf("state[%q] is unset after a dispatched task agent's finish_task: %v", "task_out", err)
	}
	if diff := cmp.Diff(map[string]any{"result": "summarized"}, got); diff != "" {
		t.Errorf("state[%q] mismatch (-want +got):\n%s", "task_out", diff)
	}
}
