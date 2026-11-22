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

package workflow_test

import (
	"context"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

type weatherOut struct {
	City string `json:"city"`
	Temp int    `json:"temp"`
}

var weatherSchema = &genai.Schema{
	Type: genai.TypeObject,
	Properties: map[string]*genai.Schema{
		"city": {Type: genai.TypeString},
		"temp": {Type: genai.TypeInteger},
	},
	Required: []string{"city", "temp"},
}

// requestRecorder records every request it receives. On a Gemini API name
// with tools it calls set_model_response once when that tool is offered,
// the way a model following the output-schema instruction would.
type requestRecorder struct {
	name     string
	backend  genai.Backend
	reqs     []*model.LLMRequest
	reply    string // text answer; defaults to a weatherOut value
	setReply map[string]any
	// setNoArgs makes the set_model_response call carry no args at all.
	setNoArgs bool
}

func (r *requestRecorder) Name() string                       { return r.name }
func (r *requestRecorder) GetGoogleLLMVariant() genai.Backend { return r.backend }

func (r *requestRecorder) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	r.reqs = append(r.reqs, req)
	text := r.reply
	if text == "" {
		text = `{"city":"Paris","temp":20}`
	}
	part := &genai.Part{Text: text}
	if (r.setReply != nil || r.setNoArgs) && len(r.reqs) == 1 && offersTool(req, "set_model_response") {
		part = &genai.Part{FunctionCall: &genai.FunctionCall{Name: "set_model_response", Args: r.setReply}}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}}, nil)
	}
}

func offersTool(req *model.LLMRequest, name string) bool {
	return toolDecl(req, name) != nil
}

func toolDecl(req *model.LLMRequest, name string) *genai.FunctionDeclaration {
	if req.Config == nil {
		return nil
	}
	for _, tl := range req.Config.Tools {
		for _, d := range tl.FunctionDeclarations {
			if d.Name == name {
				return d
			}
		}
	}
	return nil
}

type lookupArgs struct {
	Q string `json:"q"`
}

func lookupTool(t *testing.T) tool.Tool {
	t.Helper()
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks a city up"},
		func(_ agent.Context, in lookupArgs) (map[string]any, error) { return map[string]any{"q": in.Q}, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return lookup
}

// firstRequest returns the first request llm received.
func firstRequest(t *testing.T, llm *requestRecorder) *model.LLMRequest {
	t.Helper()
	if len(llm.reqs) == 0 {
		t.Fatal("the model was never called")
	}
	return llm.reqs[0]
}

// untypedAndTypedRequests runs one fresh agent built from cfg under an untyped
// node and another under the node build returns, and returns the first
// request each sent. cfg.Model must be nil; newLLM supplies a fresh one per run.
func untypedAndTypedRequests(t *testing.T, cfg llmagent.Config, newLLM func() *requestRecorder, build func(agent.Agent) (*workflow.AgentNode, error)) (untyped, typed *model.LLMRequest) {
	t.Helper()
	request := func(build func(agent.Agent) (*workflow.AgentNode, error)) *model.LLMRequest {
		llm := newLLM()
		cfg := cfg
		cfg.Model = llm
		a, err := llmagent.New(cfg)
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		node, err := build(a)
		if err != nil {
			t.Fatalf("constructing node: %v", err)
		}
		runNode(t, node, a)
		return firstRequest(t, llm)
	}
	untyped = request(func(a agent.Agent) (*workflow.AgentNode, error) {
		return workflow.NewAgentNode(a, workflow.NodeConfig{})
	})
	return untyped, request(build)
}

// runNode runs node once and returns the last output it emitted.
func runNode(t *testing.T, node *workflow.AgentNode, a agent.Agent) any {
	t.Helper()
	var out any
	for ev, err := range node.Run(newRunnableNodeContext(t, a), "weather in Paris?") {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev.Output != nil {
			out = ev.Output
		}
	}
	return out
}

func TestAgentNode_TypedObjectOutputSetsResponseSchema(t *testing.T) {
	llm := &requestRecorder{name: "recorder"}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	runNode(t, node, a)

	cfg := firstRequest(t, llm).Config
	if diff := cmp.Diff(weatherSchema, cfg.ResponseSchema); diff != "" {
		t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
	}
	if cfg.ResponseMIMEType != "application/json" {
		t.Errorf("ResponseMIMEType = %q, want %q", cfg.ResponseMIMEType, "application/json")
	}
}

// A Gemini API model cannot combine tools with a response schema, so the
// schema goes out as set_model_response and the node returns its arguments.
func TestAgentNode_TypedObjectOutputWithToolsUsesSetModelResponse(t *testing.T) {
	llm := &requestRecorder{
		name:    "gemini-2.5-flash",
		backend: genai.BackendGeminiAPI,
		// Differs from the default text reply, so the output can only come from
		// the set_model_response call. JSON numbers decode as float64.
		setReply: map[string]any{"city": "Paris", "temp": float64(31)},
	}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm, Tools: []tool.Tool{lookupTool(t)}})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	out := runNode(t, node, a)

	first := firstRequest(t, llm)
	decl := toolDecl(first, "set_model_response")
	if decl == nil {
		t.Fatal("set_model_response was not offered to the model")
	}
	if diff := cmp.Diff(weatherSchema, decl.ParametersJsonSchema); diff != "" {
		t.Errorf("set_model_response parameters mismatch (-want +got):\n%s", diff)
	}
	if first.Config.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %v alongside tools, want nil", first.Config.ResponseSchema)
	}
	got, err := node.ValidateOutput(out)
	if err != nil {
		t.Fatalf("ValidateOutput(%v): %v", out, err)
	}
	want := map[string]any{"city": "Paris", "temp": float64(31)}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("node output mismatch (-want +got):\n%s", diff)
	}
	// The set_model_response result ends the turn: no second model call.
	if len(llm.reqs) != 1 {
		t.Errorf("model was called %d times, want 1", len(llm.reqs))
	}
}

// An OutputSchema the agent declares is what the model sees, whatever the
// node's Output type.
func TestAgentNode_DeclaredOutputSchemaWins(t *testing.T) {
	declared := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
	}
	llm := &requestRecorder{name: "recorder", reply: `{"answer":"sunny"}`}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm, OutputSchema: declared})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	runNode(t, node, a)

	if diff := cmp.Diff(declared, firstRequest(t, llm).Config.ResponseSchema); diff != "" {
		t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
	}
}

type cityOnly struct {
	City string `json:"city"`
}

// One agent can back several nodes. Each node's schema reaches only its own
// runs, and the agent's declared OutputSchema is never written.
func TestAgentNode_OneAgentTwoTypedNodes(t *testing.T) {
	llm := &requestRecorder{name: "recorder"}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	weatherNode, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped[weatherOut]: %v", err)
	}
	cityNode, err := workflow.NewAgentNodeTyped[string, cityOnly](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped[cityOnly]: %v", err)
	}
	citySchema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}

	runNode(t, weatherNode, a)
	runNode(t, cityNode, a)
	runNode(t, weatherNode, a)

	if len(llm.reqs) != 3 {
		t.Fatalf("got %d model requests, want 3", len(llm.reqs))
	}
	for i, want := range []*genai.Schema{weatherSchema, citySchema, weatherSchema} {
		if diff := cmp.Diff(want, llm.reqs[i].Config.ResponseSchema); diff != "" {
			t.Errorf("run %d ResponseSchema mismatch (-want +got):\n%s", i, diff)
		}
	}
	if got := llminternal.Reveal(a.(llminternal.Agent)).OutputSchema; got != nil {
		t.Errorf("agent's declared OutputSchema = %v after the runs, want nil", got)
	}
}

// A task agent returns its result through finish_task, so a typed node adds
// neither a response schema nor set_model_response to its requests.
// Chat and task agents talk to the user between replies, so the node's shape
// is not forced on them.
func TestAgentNode_ChatAndTaskAgentsGetNoOutputSchema(t *testing.T) {
	for _, tc := range []struct {
		name  string
		mode  llmagent.Mode
		llm   *requestRecorder
		tools []tool.Tool
	}{
		{name: "task, no tools", mode: llmagent.ModeTask, llm: &requestRecorder{name: "recorder"}},
		{name: "task, gemini api with tools", mode: llmagent.ModeTask, llm: &requestRecorder{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}, tools: []tool.Tool{lookupTool(t)}},
		{name: "chat, no tools", mode: llmagent.ModeChat, llm: &requestRecorder{name: "recorder"}},
		{name: "chat, gemini api with tools", mode: llmagent.ModeChat, llm: &requestRecorder{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}, tools: []tool.Tool{lookupTool(t)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := llmagent.New(llmagent.Config{Name: "weather", Model: tc.llm, Mode: tc.mode, Tools: tc.tools})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
			if err != nil {
				t.Fatalf("NewAgentNodeTyped: %v", err)
			}

			runNode(t, node, a)

			req := firstRequest(t, tc.llm)
			if req.Config.ResponseSchema != nil || req.Config.ResponseMIMEType != "" {
				t.Errorf("%s agent request has ResponseSchema %v, ResponseMIMEType %q; want neither", tc.mode, req.Config.ResponseSchema, req.Config.ResponseMIMEType)
			}
			if offersTool(req, "set_model_response") {
				t.Errorf("%s agent was offered set_model_response", tc.mode)
			}
		})
	}
}

type withAnyField struct {
	Summary string `json:"summary"`
	Details any    `json:"details"`
}

type withMapField struct {
	Counts map[string]int `json:"counts"`
}

type empty struct{}

// Outputs that are not a closed, non-null object with properties, or that
// genai.Schema cannot describe, must construct and send exactly the request
// an untyped node sends, with and without tools.
func TestAgentNode_OutputSchemaLeavesRequestUnchanged(t *testing.T) {
	intEnum := &jsonschema.Schema{
		Type:                 "object",
		Properties:           map[string]*jsonschema.Schema{"level": {Type: "integer", Enum: []any{1, 2, 3}}},
		AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
	}
	// No additionalProperties: the node accepts extra keys, which
	// set_model_response would reject.
	openObject := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}},
	}
	newNodes := map[string]func(agent.Agent) (*workflow.AgentNode, error){
		"string": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, string](a, workflow.NodeConfig{})
		},
		"int": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, int](a, workflow.NodeConfig{})
		},
		"slice of objects": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, []weatherOut](a, workflow.NodeConfig{})
		},
		"map": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, map[string]int](a, workflow.NodeConfig{})
		},
		"empty struct": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, empty](a, workflow.NodeConfig{})
		},
		"struct with an any field": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, withAnyField](a, workflow.NodeConfig{})
		},
		"struct with a map field": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, withMapField](a, workflow.NodeConfig{})
		},
		"explicit schema with an integer enum": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeWithSchemas(a, nil, intEnum, workflow.NodeConfig{})
		},
		"explicit open object schema": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeWithSchemas(a, nil, openObject, workflow.NodeConfig{})
		},
		"pointer to struct": func(a agent.Agent) (*workflow.AgentNode, error) {
			return workflow.NewAgentNodeTyped[string, *weatherOut](a, workflow.NodeConfig{})
		},
	}
	models := map[string]struct {
		newLLM    func() *requestRecorder
		withTools bool
	}{
		"no tools": {newLLM: func() *requestRecorder { return &requestRecorder{name: "recorder"} }},
		"gemini api with tools": {
			newLLM: func() *requestRecorder {
				return &requestRecorder{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}
			},
			withTools: true,
		},
	}
	for modelName, m := range models {
		for nodeName, newNode := range newNodes {
			t.Run(modelName+"/"+nodeName, func(t *testing.T) {
				cfg := llmagent.Config{Name: "weather"}
				if m.withTools {
					cfg.Tools = []tool.Tool{lookupTool(t)}
				}

				untyped, typed := untypedAndTypedRequests(t, cfg, m.newLLM, newNode)

				if diff := cmp.Diff(untyped.Config, typed.Config); diff != "" {
					t.Errorf("request config differs from an untyped node's (-untyped +typed):\n%s", diff)
				}
			})
		}
	}
}

// An agent whose own setup makes a response schema unsafe to add sends the same
// request under a typed node as under an untyped one: a schema set through
// GenerateContentConfig is the agent's declaration, and an agent that can
// transfer declares transfer_to_agent, a function tool, which a Gemini API
// model cannot combine with a response schema.
func TestAgentNode_AgentSetupKeepsItsRequest(t *testing.T) {
	declared := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
	}
	sub, err := llmagent.New(llmagent.Config{Name: "helper", Model: &requestRecorder{name: "recorder"}})
	if err != nil {
		t.Fatalf("llmagent.New(helper): %v", err)
	}
	for _, tc := range []struct {
		name   string
		cfg    llmagent.Config
		newLLM func() *requestRecorder
	}{
		{
			name: "GenerateContentConfig.ResponseSchema",
			cfg: llmagent.Config{GenerateContentConfig: &genai.GenerateContentConfig{
				ResponseSchema: declared,
			}},
			newLLM: func() *requestRecorder { return &requestRecorder{name: "recorder", reply: `{"answer":"sunny"}`} },
		},
		{
			name: "GenerateContentConfig.ResponseJsonSchema",
			cfg: llmagent.Config{GenerateContentConfig: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}},
			}},
			newLLM: func() *requestRecorder { return &requestRecorder{name: "recorder", reply: `{"answer":"sunny"}`} },
		},
		{
			name: "GenerateContentConfig.ResponseMIMEType",
			cfg: llmagent.Config{GenerateContentConfig: &genai.GenerateContentConfig{
				ResponseMIMEType: "text/plain",
			}},
			newLLM: func() *requestRecorder { return &requestRecorder{name: "recorder", reply: "sunny"} },
		},
		{
			name: "sub-agent on a gemini api model",
			cfg:  llmagent.Config{SubAgents: []agent.Agent{sub}},
			newLLM: func() *requestRecorder {
				return &requestRecorder{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			cfg.Name = "weather"

			untyped, typed := untypedAndTypedRequests(t, cfg, tc.newLLM, func(a agent.Agent) (*workflow.AgentNode, error) {
				return workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
			})

			if diff := cmp.Diff(untyped.Config, typed.Config); diff != "" {
				t.Errorf("request config differs from an untyped node's (-untyped +typed):\n%s", diff)
			}
		})
	}
}

// A model callback that edits the request's schema changes that request only;
// the node's next run is sent the node's schema unchanged.
func TestAgentNode_CallbackEditDoesNotReachNextRun(t *testing.T) {
	llm := &requestRecorder{name: "recorder"}
	edited := false
	a, err := llmagent.New(llmagent.Config{
		Name:  "weather",
		Model: llm,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				if !edited && req.Config.ResponseSchema != nil {
					edited = true
					req.Config.ResponseSchema.Properties["injected"] = &genai.Schema{Type: genai.TypeString}
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	runNode(t, node, a)
	runNode(t, node, a)

	if len(llm.reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(llm.reqs))
	}
	if diff := cmp.Diff(weatherSchema, llm.reqs[1].Config.ResponseSchema); diff != "" {
		t.Errorf("second run's ResponseSchema mismatch (-want +got):\n%s", diff)
	}
}

// statelessLLM answers every call with the same text and records nothing, so
// it is safe to call from concurrent runs.
type statelessLLM struct{}

func (statelessLLM) Name() string { return "stateless" }

func (statelessLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: `{"city":"Paris","temp":20}`}}}}, nil)
	}
}

// Concurrent runs of one node, as under a fan-out, each get their own schema,
// so a callback that writes to it does not race with the other run (also
// checked under -race).
func TestAgentNode_ConcurrentRunsDoNotShareSchema(t *testing.T) {
	var (
		mu   sync.Mutex
		seen []*genai.Schema
	)
	a, err := llmagent.New(llmagent.Config{
		Name:  "weather",
		Model: statelessLLM{},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				schema := req.Config.ResponseSchema
				mu.Lock()
				seen = append(seen, schema)
				mu.Unlock()
				if schema != nil {
					schema.Description = "set by callback" // races under -race if shared
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	var wg sync.WaitGroup
	for range 2 {
		ctx := newRunnableNodeContext(t, a)
		wg.Go(func() {
			for _, err := range node.Run(ctx, "weather in Paris?") {
				if err != nil {
					t.Errorf("Run: %v", err)
					return
				}
			}
		})
	}
	wg.Wait()

	if len(seen) != 2 || seen[0] == nil || seen[1] == nil {
		t.Fatalf("callback saw schemas %v, want two non-nil", seen)
	}
	if seen[0] == seen[1] {
		t.Error("both runs were handed the same *genai.Schema")
	}
}

// An agent that declares single_turn itself is treated like one that runs
// single_turn by default at a node.
func TestAgentNode_ExplicitSingleTurnAgentGetsOutputSchema(t *testing.T) {
	llm := &requestRecorder{name: "recorder"}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm, Mode: llmagent.ModeSingleTurn})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, weatherOut](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	runNode(t, node, a)

	if diff := cmp.Diff(weatherSchema, firstRequest(t, llm).Config.ResponseSchema); diff != "" {
		t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
	}
}

// An untyped placement of an agent, running inside a typed placement of the
// same agent, clears the outer schema rather than inheriting it.
func TestAgentNode_InnerUntypedPlacementClearsOuterSchema(t *testing.T) {
	llm := &requestRecorder{name: "recorder"}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	inner, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}
	state := llminternal.Reveal(a.(llminternal.Agent))
	outer := llminternal.WithBoundOutputSchema(
		llminternal.WithBoundMode(t.Context(), a.Name(), state, llminternal.ModeSingleTurn),
		a.Name(), state, weatherSchema)

	for _, err := range inner.Run(nodeContextFrom(t, outer, a), "weather in Paris?") {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	}

	if got := firstRequest(t, llm).Config.ResponseSchema; got != nil {
		t.Errorf("ResponseSchema = %v under an untyped inner placement, want nil", got)
	}
}

// nodeContextFrom is newRunnableNodeContext built on base, so a test can
// start a node inside bindings an outer placement made.
func nodeContextFrom(t *testing.T, base context.Context, a agent.Agent) agent.Context {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	ctx := runconfig.ToContext(base, &runconfig.RunConfig{StreamingMode: runconfig.StreamingModeNone})
	return agent.NewContext(icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Agent:        a,
		Session:      resp.Session,
		InvocationID: "inv-test",
	}))
}

// A hand-written object schema that allows extra keys is not sent: the model's
// answer with an extra key reaches the node as it does on an untyped node,
// rather than set_model_response rejecting it and the error becoming the
// output.
func TestAgentNode_OpenObjectSchemaKeepsModelAnswer(t *testing.T) {
	answer := map[string]any{"city": "Paris", "country": "FR"}
	llm := &requestRecorder{
		name:     "gemini-2.5-flash",
		backend:  genai.BackendGeminiAPI,
		reply:    `{"city":"Paris","country":"FR"}`,
		setReply: answer, // used only if set_model_response is offered
	}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm, Tools: []tool.Tool{lookupTool(t)}})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	open := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"city": {Type: "string"}}}
	node, err := workflow.NewAgentNodeWithSchemas(a, nil, open, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeWithSchemas: %v", err)
	}

	out := runNode(t, node, a)

	got, err := node.ValidateOutput(out)
	if err != nil {
		t.Fatalf("ValidateOutput(%v): %v", out, err)
	}
	if diff := cmp.Diff(answer, got); diff != "" {
		t.Errorf("node output mismatch (-want +got):\n%s", diff)
	}
}

type cityWithNote struct {
	City string  `json:"city"`
	Note *string `json:"note"`
}

// A pointer field goes out as nullable, so set_model_response accepts a null
// for it, as the node's own schema does, and the null reaches the output
// instead of the tool's error.
func TestAgentNode_NullFieldThroughSetModelResponse(t *testing.T) {
	answer := map[string]any{"city": "Paris", "note": nil}
	llm := &requestRecorder{
		name:     "gemini-2.5-flash",
		backend:  genai.BackendGeminiAPI,
		reply:    `{"city":"Lyon","note":"text"}`, // differs, so a pass needs the tool call
		setReply: answer,
	}
	a, err := llmagent.New(llmagent.Config{Name: "weather", Model: llm, Tools: []tool.Tool{lookupTool(t)}})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[string, cityWithNote](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	out := runNode(t, node, a)

	if !offersTool(firstRequest(t, llm), "set_model_response") {
		t.Fatal("set_model_response was not offered to the model")
	}
	got, err := node.ValidateOutput(out)
	if err != nil {
		t.Fatalf("ValidateOutput(%v): %v", out, err)
	}
	if diff := cmp.Diff(answer, got); diff != "" {
		t.Errorf("node output mismatch (-want +got):\n%s", diff)
	}
}

// A set_model_response call without args is the empty object, as a call with
// empty args is, so it passes a schema whose fields are all optional.
func TestAgentNode_SetModelResponseWithoutArgs(t *testing.T) {
	llm := &requestRecorder{
		name:      "gemini-2.5-flash",
		backend:   genai.BackendGeminiAPI,
		reply:     `{"note":"text"}`, // differs, so a pass needs the tool call
		setNoArgs: true,
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "a",
		Model: llm,
		Tools: []tool.Tool{lookupTool(t)},
		OutputSchema: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{"note": {Type: genai.TypeString}},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	out := runNode(t, node, a)

	if !offersTool(firstRequest(t, llm), "set_model_response") {
		t.Fatal("set_model_response was not offered to the model")
	}
	if diff := cmp.Diff(map[string]any{}, out); diff != "" {
		t.Errorf("node output mismatch (-want +got):\n%s", diff)
	}
}
