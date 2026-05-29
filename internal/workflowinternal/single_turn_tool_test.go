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

package workflowinternal_test

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/internal/workflowinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

const (
	agentName = "math_agent"
	agentDesc = "Solves math problems."
)

var sampleInputSchema = &genai.Schema{
	Type: "OBJECT",
	Properties: map[string]*genai.Schema{
		"is_magic": {Type: "BOOLEAN"},
		"name":     {Type: "STRING"},
	},
	Required: []string{"is_magic", "name"},
}

var defaultDecl = &genai.FunctionDeclaration{
	Name:        agentName,
	Description: agentDesc,
	Parameters: &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"request": {Type: "STRING"},
		},
		Required: []string{"request"},
	},
}

func TestSingleTurnTool_Metadata(t *testing.T) {
	a := newLLMAgent(t, agentName, agentDesc, nil)
	st := newSingleTurnTool(t, a)

	if got, want := st.Name(), agentName; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	if got, want := st.Description(), agentDesc; got != want {
		t.Errorf("Description() = %q, want %q", got, want)
	}
	if st.IsLongRunning() {
		t.Errorf("IsLongRunning() = true, want false")
	}
}

func TestSingleTurnTool_Declaration(t *testing.T) {
	tests := []struct {
		name  string
		agent agent.Agent
		want  *genai.FunctionDeclaration
	}{
		{
			name:  "no input schema falls back to {request: STRING}",
			agent: newLLMAgent(t, agentName, agentDesc, nil),
			want:  defaultDecl,
		},
		{
			name:  "uses wrapped agent's input schema as is",
			agent: newLLMAgent(t, agentName, agentDesc, sampleInputSchema),
			want: &genai.FunctionDeclaration{
				Name:        agentName,
				Description: agentDesc,
				Parameters:  sampleInputSchema,
			},
		},
		{
			name: "composite agent recurses into first sub-agent's schema",
			agent: newCompositeAgent(t, "parent_no_schema", "Outer composite.",
				newLLMAgent(t, "child_with_schema", "Inner agent.", sampleInputSchema)),
			want: &genai.FunctionDeclaration{
				Name:        "parent_no_schema",
				Description: "Outer composite.",
				Parameters:  sampleInputSchema,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := newSingleTurnTool(t, tc.agent).Declaration()
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Declaration() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSingleTurnTool_Run_Failures(t *testing.T) {
	tests := []struct {
		name        string
		inputSchema *genai.Schema
		args        any
		wantSubstr  []string
	}{
		{
			name:       "args is not a map",
			args:       "not a map",
			wantSubstr: []string{"map[string]any"},
		},
		{
			name:        "schema: extra field rejected",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true, "name": "x", "name_invalid": "y"},
			wantSubstr:  []string{"argument validation failed", agentName, "name_invalid"},
		},
		{
			name:        "schema: missing required field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true},
			wantSubstr:  []string{"argument validation failed", agentName, "name"},
		},
		{
			name:        "schema: wrong type for boolean field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": "not a bool", "name": "x"},
			wantSubstr:  []string{"argument validation failed", agentName, "is_magic"},
		},
		{
			name:        "schema: wrong type for string field",
			inputSchema: sampleInputSchema,
			args:        map[string]any{"is_magic": true, "name": 42.0},
			wantSubstr:  []string{"argument validation failed", agentName, "name"},
		},
		{
			name:       "default schema: empty args missing required \"request\"",
			args:       map[string]any{},
			wantSubstr: []string{"argument validation failed", "request"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newLLMAgent(t, agentName, agentDesc, tc.inputSchema)
			st := newSingleTurnTool(t, a)

			got, err := st.Run(nil, tc.args)
			if err == nil {
				t.Fatalf("Run(%v) err = nil, want non-nil", tc.args)
			}
			if got != nil {
				t.Errorf("Run(%v) result = %v, want nil", tc.args, got)
			}
			for _, sub := range tc.wantSubstr {
				if !strings.Contains(err.Error(), sub) {
					t.Errorf("Run(%v) err = %q, want substring %q", tc.args, err.Error(), sub)
				}
			}
		})
	}
}

func TestSingleTurnTool_ProcessRequest_RegistersTool(t *testing.T) {
	st := newSingleTurnTool(t, newLLMAgent(t, agentName, agentDesc, nil))

	req := &model.LLMRequest{}
	if err := st.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() err = %v", err)
	}

	if _, ok := req.Tools[agentName]; !ok {
		t.Errorf("req.Tools missing %q; got keys %v", agentName, sortedKeys(req.Tools))
	}
	decls := utils.FunctionDecls(req.Config)
	if !slices.ContainsFunc(decls, func(d *genai.FunctionDeclaration) bool { return d.Name == agentName }) {
		var names []string
		for _, d := range decls {
			if d != nil {
				names = append(names, d.Name)
			}
		}
		t.Errorf("req.Config function declarations missing %q; got %v", agentName, names)
	}
}

func TestSingleTurnTool_Run_HappyPath(t *testing.T) {
	const wantOutput = "world"

	a, err := agent.New(agent.Config{
		Name:        "echo_agent",
		Description: "Yields a single fixed-output event.",
		Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(&session.Event{Output: wantOutput}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	st := newSingleTurnTool(t, a)

	var (
		gotResult map[string]any
		gotErr    error
	)
	orchestrator := workflow.NewDynamicNode("orchestrator",
		func(ctx workflow.NodeContext, _ string, _ func(*session.Event) error) (any, error) {
			ic := ctx.WithContext(workflow.WithNodeContext(ctx, ctx))
			toolCtx := toolinternal.NewToolContext(ic, "fc-id", &session.EventActions{}, nil)
			gotResult, gotErr = st.Run(toolCtx, map[string]any{"request": "hello"})
			return nil, gotErr
		},
		workflow.NodeConfig{},
	)

	w, err := workflow.New("root", workflow.Chain(workflow.Start, orchestrator))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	sessResp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: "test_session",
	})
	if err != nil {
		t.Fatalf("session.Create: %v", err)
	}

	ic := &fakeInvocationContext{Context: t.Context(), sess: sessResp.Session}
	for _, err := range w.Run(ic) {
		if err != nil {
			t.Fatalf("workflow.Run error: %v", err)
		}
	}

	if gotErr != nil {
		t.Fatalf("SingleTurnTool.Run err = %v, want nil", gotErr)
	}
	want := map[string]any{"result": wantOutput}
	if diff := cmp.Diff(want, gotResult); diff != "" {
		t.Errorf("SingleTurnTool.Run result diff (-want +got):\n%s", diff)
	}
}

func newSingleTurnTool(t *testing.T, a agent.Agent) *workflowinternal.SingleTurnTool {
	t.Helper()
	tt, err := workflowinternal.NewSingleTurnTool(a)
	if err != nil {
		t.Fatalf("NewSingleTurnTool: %v", err)
	}
	st, ok := tt.(*workflowinternal.SingleTurnTool)
	if !ok {
		t.Fatalf("NewSingleTurnTool returned %T, want *workflowinternal.SingleTurnTool", tt)
	}
	return st
}

func newLLMAgent(t *testing.T, name, description string, inputSchema *genai.Schema) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", name, err)
	}
	return a
}

func newCompositeAgent(t *testing.T, name, description string, subAgents ...agent.Agent) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:        name,
		Description: description,
		SubAgents:   subAgents,
		Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, errors.New("custom agent Run should not be invoked in unit tests"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q): %v", name, err)
	}
	return a
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

type fakeInvocationContext struct {
	context.Context
	sess session.Session
}

func (c *fakeInvocationContext) Agent() agent.Agent              { return nil }
func (c *fakeInvocationContext) Artifacts() agent.Artifacts      { return nil }
func (c *fakeInvocationContext) Memory() agent.Memory            { return nil }
func (c *fakeInvocationContext) Session() session.Session        { return c.sess }
func (c *fakeInvocationContext) InvocationID() string            { return "test-invocation-id" }
func (c *fakeInvocationContext) Branch() string                  { return "" }
func (c *fakeInvocationContext) UserContent() *genai.Content     { return nil }
func (c *fakeInvocationContext) RunConfig() *agent.RunConfig     { return nil }
func (c *fakeInvocationContext) EndInvocation()                  {}
func (c *fakeInvocationContext) Ended() bool                     { return false }
func (c *fakeInvocationContext) ResumedInput(string) (any, bool) { return nil, false }
func (c *fakeInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	cp := *c
	cp.Context = ctx
	return &cp
}
