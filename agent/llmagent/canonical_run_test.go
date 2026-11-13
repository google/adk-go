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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type lookupArgs struct {
	Query string `json:"query"`
}

type lookupResult struct {
	Answer string `json:"answer"`
}

func lookup(agent.Context, lookupArgs) (lookupResult, error) { return lookupResult{}, nil }

func TestRun_ResolvedModelDecidesOutputSchemaTool(t *testing.T) {
	lookupTool, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks up."}, lookup)
	if err != nil {
		t.Fatal(err)
	}
	child := newLLMAgent(t, llmagent.Config{
		Name:         "child",
		Tools:        []tool.Tool{lookupTool},
		OutputSchema: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}}},
	})
	// A Gemini API model takes no output schema together with tools, so the
	// child's request carries the set_model_response tool instead.
	parentModel := &canonLLM{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}
	newLLMAgent(t, llmagent.Config{Name: "parent", Model: parentModel, SubAgents: []agent.Agent{child}})

	runAgent(t, child)

	if len(parentModel.requests) == 0 {
		t.Fatal("the inherited model got no request")
	}
	req := parentModel.requests[0]
	if _, ok := req.Tools["set_model_response"]; !ok {
		t.Errorf("request tools = %v, want set_model_response among them", toolNames(req.Tools))
	}
	if req.Config.ResponseSchema != nil {
		t.Errorf("request ResponseSchema = %+v, want nil (the schema goes through set_model_response)", req.Config.ResponseSchema)
	}
}

// A step resolves the model once (amendment A6): a default that changes
// during the step, here from the instruction provider, reaches neither the
// later request processors nor the model call of that step.
func TestRun_DefaultChangedDuringStepKeepsTheStepModel(t *testing.T) {
	restoreDefaults(t)
	lookupTool, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks up."}, lookup)
	if err != nil {
		t.Fatal(err)
	}
	// The Gemini API takes no output schema together with tools, so a request
	// built for stepModel carries set_model_response; Vertex AI takes both,
	// so one built for laterModel would carry a ResponseSchema instead.
	stepModel := &canonLLM{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}
	laterModel := &canonLLM{name: "gemini-2.5-flash", backend: genai.BackendVertexAI}
	if err := llmagent.SetDefaultModel(stepModel); err != nil {
		t.Fatal(err)
	}
	a := newLLMAgent(t, llmagent.Config{
		Name:         "a",
		Tools:        []tool.Tool{lookupTool},
		OutputSchema: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}}},
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return "answer", llmagent.SetDefaultModel(laterModel)
		},
	})

	runAgent(t, a)

	if got := len(laterModel.requests); got != 0 {
		t.Errorf("the default set during the step got %d requests, want 0", got)
	}
	if len(stepModel.requests) != 1 {
		t.Fatalf("the step's model got %d requests, want 1", len(stepModel.requests))
	}
	req := stepModel.requests[0]
	if _, ok := req.Tools["set_model_response"]; !ok {
		t.Errorf("request tools = %v, want set_model_response among them (built for the step's Gemini API model)", toolNames(req.Tools))
	}
	if req.Config.ResponseSchema != nil {
		t.Errorf("request ResponseSchema = %+v, want nil (built for the step's Gemini API model)", req.Config.ResponseSchema)
	}
}

func toolNames(tools map[string]any) []string {
	var names []string
	for name := range tools {
		names = append(names, name)
	}
	return names
}

func TestRun_ResolvedModelDecidesDisplayNameStripping(t *testing.T) {
	child := newLLMAgent(t, llmagent.Config{Name: "child"})
	parentModel := &canonLLM{name: "gemini-2.5-flash", backend: genai.BackendGeminiAPI}
	newLLMAgent(t, llmagent.Config{Name: "parent", Model: parentModel, SubAgents: []agent.Agent{child}})
	content := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "see the file"},
		{InlineData: &genai.Blob{MIMEType: "text/plain", Data: []byte("x"), DisplayName: "notes.txt"}},
	}}

	runAgentWithContent(t, child, content)

	if len(parentModel.requests) == 0 {
		t.Fatal("the inherited model got no request")
	}
	var displayNames []string
	for _, c := range parentModel.requests[0].Contents {
		for _, p := range c.Parts {
			if p.InlineData != nil {
				displayNames = append(displayNames, p.InlineData.DisplayName)
			}
		}
	}
	if len(displayNames) != 1 || displayNames[0] != "" {
		t.Errorf("inline data display names in the request = %q, want one empty name (the Gemini API takes none)", displayNames)
	}
}

// A live run falls through to the default live model, not the normal
// default: both defaults are Gemini models without a client here, and the
// error names the one the run resolved.
func TestRunLive_NilClientIsAnError(t *testing.T) {
	for _, key := range []string{"GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"} {
		t.Setenv(key, "")
	}
	restoreDefaults(t)
	// A fresh resolution of the built-in live default, so its client is
	// created with the variables above cleared.
	if err := llmagent.SetDefaultLiveModelByName(llmagent.DefaultLiveModel); err != nil {
		t.Fatal(err)
	}
	if err := llmagent.SetDefaultModelByName("gemini-2.5-flash"); err != nil {
		t.Fatal(err)
	}
	a := newLLMAgent(t, llmagent.Config{Name: "a"})
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}

	sess, _, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})

	if err == nil {
		_ = sess.Close()
		t.Fatal("RunLive with a Gemini model that has no client = nil error, want an error")
	}
	if !strings.Contains(err.Error(), `"gemini-live-2.5-flash-native-audio"`) {
		t.Errorf("RunLive() error = %q, want it to name the default live model \"gemini-live-2.5-flash-native-audio\"", err)
	}
}
