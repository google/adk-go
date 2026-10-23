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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/workflow"
)

// schemaCapturingLLM records the last request's Config and replies with
// replyJSON, which callers set to satisfy whatever schema is in play.
type schemaCapturingLLM struct {
	lastConfig *genai.GenerateContentConfig
	replyJSON  string
}

func (*schemaCapturingLLM) Name() string { return "capturing-mock" }

func (c *schemaCapturingLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	c.lastConfig = req.Config
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: []*genai.Part{{Text: c.replyJSON}},
			},
		}, nil)
	}
}

var _ model.LLM = (*schemaCapturingLLM)(nil)

// reportOutput is the workflow node's typed Output in these tests: a plain
// struct so NewAgentNodeTyped has something concrete to derive a schema from.
type reportOutput struct {
	Summary string `json:"summary"`
	Score   int    `json:"score"`
}

// TestAgentNode_TypedOutput_ConstrainsModelResponseSchema is the regression
// test for google/adk-go#1597: a typed workflow node never asked the model
// to conform to its Output type.
func TestAgentNode_TypedOutput_ConstrainsModelResponseSchema(t *testing.T) {
	t.Parallel()

	llm := &schemaCapturingLLM{replyJSON: `{"summary":"ok","score":1}`}
	a, err := llmagent.New(llmagent.Config{Name: "reporter", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[any, reportOutput](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	ctx := newRunnableNodeContext(t, a)
	for _, err := range node.Run(ctx, "describe the report") {
		if err != nil {
			t.Fatalf("node.Run yielded err: %v", err)
		}
	}

	if llm.lastConfig == nil || llm.lastConfig.ResponseSchema == nil {
		t.Fatal("model never received a ResponseSchema; output type is not constraining the prompt")
	}
	schema := llm.lastConfig.ResponseSchema
	if schema.Type != genai.TypeObject {
		t.Errorf("ResponseSchema.Type = %q, want %q", schema.Type, genai.TypeObject)
	}
	if got := schema.Properties["summary"]; got == nil || got.Type != genai.TypeString {
		t.Errorf("ResponseSchema.Properties[summary] = %+v, want type STRING", got)
	}
	if got := schema.Properties["score"]; got == nil || got.Type != genai.TypeInteger {
		t.Errorf("ResponseSchema.Properties[score] = %+v, want type INTEGER", got)
	}
	if llm.lastConfig.ResponseMIMEType != "application/json" {
		t.Errorf("ResponseMIMEType = %q, want application/json", llm.lastConfig.ResponseMIMEType)
	}
}

// TestAgentNode_PlainNewAgentNode_LeavesResponseSchemaUnconstrained is the
// control: an Output of `any` (NewAgentNode's default) carries no real
// schema, so it must not force a constraint the caller never asked for.
func TestAgentNode_PlainNewAgentNode_LeavesResponseSchemaUnconstrained(t *testing.T) {
	t.Parallel()

	llm := &schemaCapturingLLM{replyJSON: `anything`}
	a, err := llmagent.New(llmagent.Config{Name: "freeform", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNode: %v", err)
	}

	ctx := newRunnableNodeContext(t, a)
	for _, err := range node.Run(ctx, "say anything") {
		if err != nil {
			t.Fatalf("node.Run yielded err: %v", err)
		}
	}

	if llm.lastConfig != nil && llm.lastConfig.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %+v, want nil for an untyped (any) node", llm.lastConfig.ResponseSchema)
	}
}

// otherReportOutput has a different shape than reportOutput: a placement's
// derived schema must be scoped to the node that derived it, not to the
// agent it wraps.
type otherReportOutput struct {
	Verdict string `json:"verdict"`
}

// TestAgentNode_SameAgentInstance_TwoTypedNodesDoNotCrossContaminate is the
// regression test for the design this fix had to avoid: writing the derived
// schema onto the wrapped agent's shared State would let one agent.Agent
// instance, wrapped by two nodes with different Output types, apply
// whichever node ran last to both. nodeB is constructed after nodeA and run
// first; nodeA must still see its own schema afterwards, not nodeB's.
func TestAgentNode_SameAgentInstance_TwoTypedNodesDoNotCrossContaminate(t *testing.T) {
	t.Parallel()

	llm := &schemaCapturingLLM{replyJSON: `{"summary":"ok","score":1,"verdict":"ok"}`}
	shared, err := llmagent.New(llmagent.Config{Name: "shared", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	nodeA, err := workflow.NewAgentNodeTyped[any, reportOutput](shared, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped (A): %v", err)
	}
	// Constructed AFTER nodeA, wrapping the SAME agent instance with a
	// different Output type. If the schema were written onto shared State,
	// constructing or running nodeB would overwrite nodeA's.
	nodeB, err := workflow.NewAgentNodeTyped[any, otherReportOutput](shared, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped (B): %v", err)
	}

	// Run nodeB first so a shared-state write would already have landed
	// before nodeA runs.
	for _, runErr := range nodeB.Run(newRunnableNodeContext(t, shared), "describe the report") {
		if runErr != nil {
			t.Fatalf("nodeB.Run yielded err: %v", runErr)
		}
	}
	if llm.lastConfig == nil || llm.lastConfig.ResponseSchema == nil {
		t.Fatal("nodeB: model never received a ResponseSchema")
	}
	if _, ok := llm.lastConfig.ResponseSchema.Properties["verdict"]; !ok {
		t.Errorf("nodeB: ResponseSchema.Properties = %+v, want \"verdict\"", llm.lastConfig.ResponseSchema.Properties)
	}

	for _, runErr := range nodeA.Run(newRunnableNodeContext(t, shared), "describe the report") {
		if runErr != nil {
			t.Fatalf("nodeA.Run yielded err: %v", runErr)
		}
	}
	if llm.lastConfig == nil || llm.lastConfig.ResponseSchema == nil {
		t.Fatal("nodeA: model never received a ResponseSchema")
	}
	schemaA := llm.lastConfig.ResponseSchema
	if _, ok := schemaA.Properties["summary"]; !ok {
		t.Errorf("nodeA: ResponseSchema.Properties = %+v, want \"summary\" (nodeA's own schema), not nodeB's \"verdict\"", schemaA.Properties)
	}
	if _, ok := schemaA.Properties["verdict"]; ok {
		t.Errorf("nodeA: ResponseSchema.Properties = %+v carries nodeB's \"verdict\": schema leaked across nodes sharing one agent instance", schemaA.Properties)
	}
}

// TestAgentNode_TypedOutput_DoesNotOverrideExplicitOutputSchema confirms a
// caller-supplied llmagent.Config.OutputSchema always wins: the derived
// schema fills a gap, it never replaces an explicit one.
func TestAgentNode_TypedOutput_DoesNotOverrideExplicitOutputSchema(t *testing.T) {
	t.Parallel()

	explicit := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"verdict": {Type: genai.TypeString}},
	}
	llm := &schemaCapturingLLM{replyJSON: `{"verdict":"ok"}`}
	a, err := llmagent.New(llmagent.Config{Name: "explicit_schema", Model: llm, OutputSchema: explicit})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	// A different Output type than what OutputSchema above describes: if the
	// wiring ever stopped respecting the explicit config, this would replace
	// "verdict" with "summary"/"score" and the test would catch it.
	node, err := workflow.NewAgentNodeTyped[any, reportOutput](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	ctx := newRunnableNodeContext(t, a)
	for _, err := range node.Run(ctx, "describe the report") {
		if err != nil {
			t.Fatalf("node.Run yielded err: %v", err)
		}
	}

	if llm.lastConfig == nil || llm.lastConfig.ResponseSchema == nil {
		t.Fatal("model never received a ResponseSchema")
	}
	if _, ok := llm.lastConfig.ResponseSchema.Properties["verdict"]; !ok {
		t.Errorf("ResponseSchema.Properties = %+v, want the explicit schema's \"verdict\" untouched", llm.lastConfig.ResponseSchema.Properties)
	}
}

// reportWithOptionalOutput has a pointer field, so jsonschema-go resolves its
// "note" property to a JSON Schema type ARRAY (["null","string"]), not a bare
// string. genai.Schema.Type can only hold one value.
type reportWithOptionalOutput struct {
	Summary string  `json:"summary"`
	Note    *string `json:"note,omitempty"`
}

// TestAgentNode_TypedOutput_OptionalFieldStillConstrainsSchema is the
// regression test for the type-array case: "note" must collapse to
// Nullable: true plus a single concrete Type, and every other property
// must still reach the model.
func TestAgentNode_TypedOutput_OptionalFieldStillConstrainsSchema(t *testing.T) {
	t.Parallel()

	llm := &schemaCapturingLLM{replyJSON: `{"summary":"ok"}`}
	a, err := llmagent.New(llmagent.Config{Name: "optional_field", Model: llm})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNodeTyped[any, reportWithOptionalOutput](a, workflow.NodeConfig{})
	if err != nil {
		t.Fatalf("NewAgentNodeTyped: %v", err)
	}

	ctx := newRunnableNodeContext(t, a)
	for _, err := range node.Run(ctx, "describe the report") {
		if err != nil {
			t.Fatalf("node.Run yielded err: %v", err)
		}
	}

	if llm.lastConfig == nil || llm.lastConfig.ResponseSchema == nil {
		t.Fatal("model never received a ResponseSchema; a nullable field must not drop the whole schema")
	}
	schema := llm.lastConfig.ResponseSchema
	if got := schema.Properties["summary"]; got == nil || got.Type != genai.TypeString {
		t.Errorf("ResponseSchema.Properties[summary] = %+v, want type STRING", got)
	}
	note := schema.Properties["note"]
	if note == nil {
		t.Fatal("ResponseSchema.Properties[note] missing")
	}
	if note.Type != genai.TypeString {
		t.Errorf("ResponseSchema.Properties[note].Type = %q, want %q", note.Type, genai.TypeString)
	}
	if note.Nullable == nil || !*note.Nullable {
		t.Errorf("ResponseSchema.Properties[note].Nullable = %v, want true", note.Nullable)
	}
}
