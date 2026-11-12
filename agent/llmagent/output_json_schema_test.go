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
	"context"
	"encoding/json"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

// schemaModel is a Gemini API model that answers its n-th request with
// replies[n], repeating the last reply once they run out, and keeps every
// request. Its name is not a Gemini model's, so it does not get the native
// output-schema-and-tools path. Safe for concurrent use.
type schemaModel struct {
	replies []*genai.Content
	// beforeFirstReply, when set, runs before the first reply, with the
	// request's context.
	beforeFirstReply func(context.Context)

	mu   sync.Mutex
	reqs []*model.LLMRequest
}

func (*schemaModel) Name() string { return "recording-model" }

func (*schemaModel) GetGoogleLLMVariant() genai.Backend { return genai.BackendGeminiAPI }

func (m *schemaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	n := len(m.reqs)
	m.reqs = append(m.reqs, req)
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		if n == 0 && m.beforeFirstReply != nil {
			m.beforeFirstReply(ctx)
		}
		reply := m.replies[min(n, len(m.replies)-1)]
		yield(&model.LLMResponse{Content: reply, FinishReason: genai.FinishReasonStop}, nil)
	}
}

func (m *schemaModel) requests() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*model.LLMRequest(nil), m.reqs...)
}

// capableSchemaModel declares that it takes an output schema and tools in
// one request.
type capableSchemaModel struct{ *schemaModel }

func (capableSchemaModel) Capabilities() model.Capabilities {
	return model.Capabilities{OutputSchemaAndTools: true}
}

func textReply(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleModel) }

func callReply(name string, args map[string]any) *genai.Content {
	return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-" + name, Name: name, Args: args}}}}
}

// otherTool is a tool the model never needs; its presence is what makes an
// agent with an output schema offer set_model_response.
func otherTool(t *testing.T) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{Name: "other_tool", Description: "Does nothing."},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{}, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return ft
}

// mapSchema is dict[str, int] as a JSON schema.
func mapSchema() *jsonschema.Schema {
	return &jsonschema.Schema{Type: "object", AdditionalProperties: &jsonschema.Schema{Type: "integer"}}
}

func newTestRunner(t *testing.T, a agent.Agent) (*runner.Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r, svc
}

// newNodeRunner returns a runner whose root is a workflow agent with one node
// over a, the placement a single_turn agent runs in.
func newNodeRunner(t *testing.T, a agent.Agent) (*runner.Runner, session.Service) {
	t.Helper()
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("workflow.NewAgentNode: %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name:  "wf",
		Edges: []workflow.Edge{{From: workflow.Start, To: node}},
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: wf, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r, svc
}

// runOnce runs r once in a new session sessionID with ctx and returns the
// events up to the first error, and that error.
func runOnce(ctx context.Context, t *testing.T, r *runner.Runner, svc session.Service, sessionID string) ([]*session.Event, error) {
	t.Helper()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: sessionID}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	var events []*session.Event
	for ev, err := range r.Run(ctx, "u", sessionID, genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

func declarationNamed(req *model.LLMRequest, name string) *genai.FunctionDeclaration {
	if req.Config == nil {
		return nil
	}
	for _, gt := range req.Config.Tools {
		if gt == nil {
			continue
		}
		for _, d := range gt.FunctionDeclarations {
			if d != nil && d.Name == name {
				return d
			}
		}
	}
	return nil
}

func jsonOf(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%T): %v", v, err)
	}
	return parseJSON(t, string(b))
}

func parseJSON(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("json.Unmarshal(%s): %v", s, err)
	}
	return v
}

// T16. A map output schema with tools reaches the model inside
// set_model_response, with its value type.
// Fails when the processor offers no tool for an agent with only
// OutputJsonSchema, or declares "response" without additionalProperties.
func TestOutputJSONSchema_ToolsPathDeclaration(t *testing.T) {
	m := &schemaModel{replies: []*genai.Content{textReply("done")}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{otherTool(t)}, OutputJsonSchema: mapSchema()})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newTestRunner(t, a)

	if _, err := runOnce(t.Context(), t, r, svc, "s"); err != nil {
		t.Fatalf("run: %v", err)
	}

	req := m.requests()[0]
	decl := declarationNamed(req, "set_model_response")
	if decl == nil {
		t.Fatal("first request has no set_model_response declaration")
	}
	want := parseJSON(t, `{"type":"object","properties":{"response":{"type":"object","additionalProperties":{"type":"integer"}}},"required":["response"]}`)
	if diff := cmp.Diff(want, jsonOf(t, decl.ParametersJsonSchema)); diff != "" {
		t.Errorf("set_model_response ParametersJsonSchema mismatch (-want +got):\n%s", diff)
	}
	if req.Config.ResponseJsonSchema != nil || req.Config.ResponseSchema != nil {
		t.Errorf("ResponseJsonSchema = %v, ResponseSchema = %v; want both nil", req.Config.ResponseJsonSchema, req.Config.ResponseSchema)
	}
}

// T17. Without tools, a map output schema goes to the request config as a JSON
// schema with the JSON MIME type, on the Gemini API too. adk-python fails in
// the genai SDK here (additionalProperties is rejected for the Gemini API);
// ADK sends it and lets the backend decide.
// Fails when the MIME type is missing, ResponseSchema is set as well, or the
// tool is offered without tools.
func TestOutputJSONSchema_NoToolsPath(t *testing.T) {
	m := &schemaModel{replies: []*genai.Content{textReply("done")}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, OutputJsonSchema: mapSchema()})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newTestRunner(t, a)

	if _, err := runOnce(t.Context(), t, r, svc, "s"); err != nil {
		t.Fatalf("run: %v", err)
	}

	req := m.requests()[0]
	want := parseJSON(t, `{"type":"object","additionalProperties":{"type":"integer"}}`)
	if diff := cmp.Diff(want, jsonOf(t, req.Config.ResponseJsonSchema)); diff != "" {
		t.Errorf("ResponseJsonSchema mismatch (-want +got):\n%s", diff)
	}
	if req.Config.ResponseMIMEType != "application/json" {
		t.Errorf("ResponseMIMEType = %q, want %q", req.Config.ResponseMIMEType, "application/json")
	}
	if req.Config.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %v, want nil", req.Config.ResponseSchema)
	}
	if decl := declarationNamed(req, "set_model_response"); decl != nil {
		t.Errorf("request declares set_model_response without tools: %v", decl)
	}
}

// T18. A model that takes an output schema and tools together gets the JSON
// schema natively.
// Fails when the native path ignores the model's capability for
// OutputJsonSchema.
func TestOutputJSONSchema_CapableModel(t *testing.T) {
	m := capableSchemaModel{&schemaModel{replies: []*genai.Content{textReply("done")}}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{otherTool(t)}, OutputJsonSchema: mapSchema()})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newTestRunner(t, a)

	if _, err := runOnce(t.Context(), t, r, svc, "s"); err != nil {
		t.Fatalf("run: %v", err)
	}

	req := m.requests()[0]
	want := parseJSON(t, `{"type":"object","additionalProperties":{"type":"integer"}}`)
	if diff := cmp.Diff(want, jsonOf(t, req.Config.ResponseJsonSchema)); diff != "" {
		t.Errorf("ResponseJsonSchema mismatch (-want +got):\n%s", diff)
	}
	if req.Config.ResponseMIMEType != "application/json" {
		t.Errorf("ResponseMIMEType = %q, want %q", req.Config.ResponseMIMEType, "application/json")
	}
	if decl := declarationNamed(req, "set_model_response"); decl != nil {
		t.Errorf("request declares set_model_response for a capable model: %v", decl)
	}
}

// T19. A single_turn agent's final text is parsed as JSON, after a markdown
// code fence is removed, as adk-python's validate_schema does
// (utils/_schema_utils.py), and the parsed value is what OutputKey stores.
// Fails when the text is kept as a string or the fence is not removed.
func TestOutputJSONSchema_FinalTextIsParsed(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply string
	}{
		{"plain", `{"a": 1}`},
		{"fenced", "```json\n{\"a\": 1}\n```"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &schemaModel{replies: []*genai.Content{textReply(tc.reply)}}
			a, err := llmagent.New(llmagent.Config{
				Name:             "a",
				Model:            m,
				Mode:             llmagent.ModeSingleTurn,
				OutputKey:        "out",
				OutputJsonSchema: mapSchema(),
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			r, svc := newNodeRunner(t, a)

			events, err := runOnce(t.Context(), t, r, svc, "s")
			if err != nil {
				t.Fatalf("run: %v", err)
			}

			want := map[string]any{"a": float64(1)}
			var delta any
			for _, ev := range events {
				if v, ok := ev.Actions.StateDelta["out"]; ok {
					delta = v
				}
			}
			if diff := cmp.Diff(want, delta); diff != "" {
				t.Errorf("state delta %q mismatch (-want +got):\n%s", "out", diff)
			}
			resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
			if err != nil {
				t.Fatalf("session.Get: %v", err)
			}
			got, err := resp.Session.State().Get("out")
			if err != nil {
				t.Fatalf("state %q: %v", "out", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("state %q mismatch (-want +got):\n%s", "out", diff)
			}
		})
	}
}

// An all-whitespace final text gives no output: no error, and OutputKey
// stores nothing.
// Fails when the whitespace check is skipped, so the text is parsed as JSON
// and the run fails.
func TestOutputJSONSchema_WhitespaceFinalTextGivesNoOutput(t *testing.T) {
	m := &schemaModel{replies: []*genai.Content{textReply("   ")}}
	a, err := llmagent.New(llmagent.Config{
		Name:             "a",
		Model:            m,
		Mode:             llmagent.ModeSingleTurn,
		OutputKey:        "out",
		OutputJsonSchema: mapSchema(),
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newNodeRunner(t, a)

	events, err := runOnce(t.Context(), t, r, svc, "s")
	if err != nil {
		t.Errorf("run error = %v, want nil", err)
	}
	for _, ev := range events {
		if v, ok := ev.Actions.StateDelta["out"]; ok {
			t.Errorf("event %s sets state %q to %v, want no value", ev.ID, "out", v)
		}
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("session.Get: %v", err)
	}
	if v, err := resp.Session.State().Get("out"); err == nil {
		t.Errorf("state %q = %v, want no value", "out", v)
	}
}

// T20. A final text that is not JSON is an error, and OutputKey stores
// nothing.
// Fails when the parse error is ignored and the raw text kept.
func TestOutputJSONSchema_MalformedFinalText(t *testing.T) {
	m := &schemaModel{replies: []*genai.Content{textReply("not json")}}
	a, err := llmagent.New(llmagent.Config{
		Name:             "a",
		Model:            m,
		Mode:             llmagent.ModeSingleTurn,
		OutputKey:        "out",
		OutputJsonSchema: mapSchema(),
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newNodeRunner(t, a)

	events, err := runOnce(t.Context(), t, r, svc, "s")

	if err == nil || !strings.Contains(err.Error(), `LlmAgent "a"`) {
		t.Errorf("run error = %v, want one that contains %q", err, `LlmAgent "a"`)
	}
	for _, ev := range events {
		if v, ok := ev.Actions.StateDelta["out"]; ok {
			t.Errorf("event %s sets state %q to %v, want no value", ev.ID, "out", v)
		}
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("session.Get: %v", err)
	}
	if v, err := resp.Session.State().Get("out"); err == nil {
		t.Errorf("state %q = %v, want no value", "out", v)
	}
}
