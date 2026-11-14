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

package llminternal

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/codeexecutor"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// countingCodeExecutor is a codeexecutor.CodeExecutor that is not a
// codeexecutor.BuiltIn. It counts its calls and returns stdout "ok".
type countingCodeExecutor struct{ calls atomic.Int32 }

func (*countingCodeExecutor) Settings() codeexecutor.Settings { return codeexecutor.DefaultSettings() }

func (e *countingCodeExecutor) ExecuteCode(context.Context, agent.InvocationContext, *codeexecutor.Input) (*codeexecutor.Result, error) {
	e.calls.Add(1)
	return &codeexecutor.Result{Stdout: "ok"}, nil
}

// newCodeExecutionContext returns an invocation context of an LLM agent with
// exec, a session and an artifact service, so that a processor that does not
// stop early can run code and yield its events.
func newCodeExecutionContext(t *testing.T, exec codeexecutor.CodeExecutor) agent.InvocationContext {
	t.Helper()
	base, err := agent.New(agent.Config{Name: "test_agent"})
	if err != nil {
		t.Fatal(err)
	}
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatal(err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:     &mockLLMAgent{Agent: base, s: &State{CodeExecutor: exec}},
		Session:   created.Session,
		Artifacts: &artifactinternal.Artifacts{Service: artifact.InMemoryService(), AppName: "app", UserID: "user", SessionID: "s1"},
	})
}

func codeAndImageResponse() *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{Text: "```python\nx = 1\n```"},
		{InlineData: &genai.Blob{Data: []byte("png"), MIMEType: "image/png"}},
	}}}
}

func TestCodeExecutionResponseProcessor_NoOpInputs(t *testing.T) {
	partial := codeAndImageResponse()
	partial.Partial = true
	tests := []struct {
		name      string
		resp      *model.LLMResponse
		noBuiltIn bool // the case has no executor at all
	}{
		{name: "nil_response"},
		{name: "partial_response", resp: partial},
		{name: "nil_content", resp: &model.LLMResponse{ErrorCode: "X"}},
		{name: "no_executor", resp: codeAndImageResponse(), noBuiltIn: true},
	}
	for _, tc := range tests {
		executors := map[string]func() codeexecutor.CodeExecutor{
			"not_built_in": func() codeexecutor.CodeExecutor { return &countingCodeExecutor{} },
			"built_in":     func() codeexecutor.CodeExecutor { return codeexecutor.NewBuiltInExecutor() },
		}
		if tc.noBuiltIn {
			executors = map[string]func() codeexecutor.CodeExecutor{"none": func() codeexecutor.CodeExecutor { return nil }}
		}
		for execName, newExec := range executors {
			t.Run(tc.name+"/"+execName, func(t *testing.T) {
				exec := newExec()
				ctx := newCodeExecutionContext(t, exec)
				var resp *model.LLMResponse
				if tc.resp != nil {
					c := *tc.resp
					c.Content = clone(tc.resp.Content)
					resp = &c
				}

				for ev, err := range codeExecutionResponseProcessor(ctx, &model.LLMRequest{}, resp) {
					t.Errorf("codeExecutionResponseProcessor yielded (%v, %v), want nothing", ev, err)
				}

				if diff := cmp.Diff(tc.resp, resp); diff != "" {
					t.Errorf("response changed (-before +after):\n%s", diff)
				}
				if c, ok := exec.(*countingCodeExecutor); ok && c.calls.Load() != 0 {
					t.Errorf("executor calls = %d, want 0", c.calls.Load())
				}
			})
		}
	}
}

func TestCodeExecutionResponseProcessor_StopsWhenConsumerStops(t *testing.T) {
	exec := &countingCodeExecutor{}
	ctx := newCodeExecutionContext(t, exec)
	resp := &model.LLMResponse{Content: genai.NewContentFromText("lead\n```python\nx = 1\n```", genai.RoleModel)}

	var events int
	for ev, err := range codeExecutionResponseProcessor(ctx, &model.LLMRequest{}, resp) {
		if err != nil {
			t.Fatalf("codeExecutionResponseProcessor error: %v", err)
		}
		if ev != nil {
			events++
		}
		break
	}

	if events != 1 {
		t.Errorf("events before the consumer stopped = %d, want 1", events)
	}
	if got := exec.calls.Load(); got != 0 {
		t.Errorf("executor calls = %d, want 0", got)
	}
	// The code event's prefix part is a copy: the response keeps its text.
	if diff := cmp.Diff(genai.NewContentFromText("lead\n```python\nx = 1\n```", genai.RoleModel), resp.Content); diff != "" {
		t.Errorf("response Content changed (-want +got):\n%s", diff)
	}
}

func TestCodeExecutionRequestProcessor_LeavesReplacedContentUnchanged(t *testing.T) {
	ctx := newCodeExecutionContext(t, &countingCodeExecutor{})
	code := genai.NewPartFromExecutableCode("y()", genai.LanguagePython)
	parts := []*genai.Part{{Text: "t"}, code}
	stored := &genai.Content{Role: genai.RoleModel, Parts: parts}
	req := &model.LLMRequest{Contents: []*genai.Content{stored}}

	for ev, err := range codeExecutionRequestProcessor(ctx, req, &Flow{}) {
		t.Fatalf("codeExecutionRequestProcessor yielded (%v, %v), want nothing", ev, err)
	}

	want := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "t"}, {Text: "```tool_code\ny()\n```"}}}
	if diff := cmp.Diff(want, req.Contents[0]); diff != "" {
		t.Errorf("request content mismatch (-want +got):\n%s", diff)
	}
	if parts[1] != code {
		t.Errorf("stored content's last part = %+v, want the executable code part", parts[1])
	}
}
