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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
)

type ss3BadParams struct {
	P string `state:"unknown_param"`
}

type ss3OrderParams struct {
	Zeta  string `state:"zeta"`
	Alpha string `state:"alpha"`
}

type ss3OwnerParams struct {
	P string `state:"adk_owner:key"`
}

type ss3TypoScopeParams struct {
	P string `state:"usr:name"`
}

type ss3AppParams struct {
	P string `state:"app:theme"`
}

func ss3Fn[P any](agent.InvocationContext, P) (string, error) { return "ok", nil }

// ss3PipelineSchema declares counter, name and optional_field under the title
// PipelineSchema.
func ss3PipelineSchema(t *testing.T) *jsonschema.Resolved {
	t.Helper()
	s, err := (&jsonschema.Schema{
		Type:  "object",
		Title: "PipelineSchema",
		Properties: map[string]*jsonschema.Schema{
			"counter":        {Type: "integer"},
			"name":           {Type: "string"},
			"optional_field": {Types: []string{"string", "null"}},
		},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	return s
}

// ss3BuildError returns the error New reports for a workflow of the one node n
// under schema, which must be a *StateSchemaError returned as it is.
func ss3BuildError(t *testing.T, n Node, schema *jsonschema.Resolved) *StateSchemaError {
	t.Helper()
	w, err := New("wf", Chain(Start, n), WithStateSchema(schema))
	if w != nil {
		t.Errorf("New() workflow = %v, want nil", w)
	}
	se, ok := err.(*StateSchemaError)
	if !ok {
		t.Fatalf("New() error = %#v, want a *StateSchemaError returned as it is", err)
	}
	return se
}

func TestNew_StateBindingUndeclared_TypedError(t *testing.T) {
	n, err := NewFunctionNodeFromState("node_with_bad_param", ss3Fn[ss3BadParams], NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState() error = %v", err)
	}

	se := ss3BuildError(t, n, ss3PipelineSchema(t))

	want := &StateSchemaError{
		Reason:   StateBindingUndeclared,
		Node:     "node_with_bad_param",
		Key:      "unknown_param",
		Schema:   "PipelineSchema",
		Declared: []string{"counter", "name", "optional_field"},
	}
	if diff := cmp.Diff(want, se); diff != "" {
		t.Errorf("New() error mismatch (-want +got):\n%s", diff)
	}
	const wantText = `node "node_with_bad_param" references state field "unknown_param" which is not declared in StateSchema (declared: [counter name optional_field])`
	if got := se.Error(); got != wantText {
		t.Errorf("Error() = %q, want %q", got, wantText)
	}
}

// Of a node's undeclared bindings, the first in field order is reported.
func TestNew_StateBindingUndeclared_ReportsFirstBindingInFieldOrder(t *testing.T) {
	schema, err := (&jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"other": {Type: "string"}},
	}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	n, err := NewFunctionNodeFromState("n", ss3Fn[ss3OrderParams], NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState() error = %v", err)
	}

	se := ss3BuildError(t, n, schema)

	if se.Key != "zeta" {
		t.Errorf("Key = %q, want %q", se.Key, "zeta")
	}
}

// A schema that declares no property, including a zero *jsonschema.Resolved,
// refuses every binding without a colon, with an empty title and an empty
// Declared.
func TestNew_StateBindingUndeclared_NoPanicOnEmptySchema(t *testing.T) {
	noProperties, err := (&jsonschema.Schema{Type: "object"}).Resolve(nil)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	tests := []struct {
		name   string
		schema *jsonschema.Resolved
	}{
		{name: "zero Resolved", schema: &jsonschema.Resolved{}},
		{name: "object without properties", schema: noProperties},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n, err := NewFunctionNodeFromState("n", dummyFnInvalid, NodeConfig{})
			if err != nil {
				t.Fatalf("NewFunctionNodeFromState() error = %v", err)
			}

			se := ss3BuildError(t, n, tc.schema)

			if se.Reason != StateBindingUndeclared || se.Key != "foo" || se.Schema != "" || len(se.Declared) != 0 {
				t.Errorf("New() error = %+v, want Reason StateBindingUndeclared, Key foo, no Schema, empty Declared", se)
			}
			if !strings.HasSuffix(se.Error(), "(declared: [])") {
				t.Errorf("Error() = %q, want it to end with %q", se.Error(), "(declared: [])")
			}
		})
	}
}

// A binding whose key contains a colon is not checked against the schema.
func TestNew_StateBindingWithColonBuilds(t *testing.T) {
	tests := []struct {
		key     string
		newNode func() (*FunctionNode, error)
	}{
		{key: "adk_owner:key", newNode: func() (*FunctionNode, error) {
			return NewFunctionNodeFromState("n", ss3Fn[ss3OwnerParams], NodeConfig{})
		}},
		{key: "usr:name", newNode: func() (*FunctionNode, error) {
			return NewFunctionNodeFromState("n", ss3Fn[ss3TypoScopeParams], NodeConfig{})
		}},
		{key: "app:theme", newNode: func() (*FunctionNode, error) {
			return NewFunctionNodeFromState("n", ss3Fn[ss3AppParams], NodeConfig{})
		}},
	}
	for _, tc := range tests {
		t.Run(tc.key, func(t *testing.T) {
			n, err := tc.newNode()
			if err != nil {
				t.Fatalf("NewFunctionNodeFromState() error = %v", err)
			}

			w, err := New("wf", Chain(Start, n), WithStateSchema(ss3PipelineSchema(t)))

			if w == nil || err != nil {
				t.Errorf("New() = %v, %v, want a workflow and nil", w, err)
			}
		})
	}
}

// The check covers the workflow's own nodes: a node inside a nested workflow
// is not checked against the outer schema.
func TestNew_NestedWorkflowBindingIsNotChecked(t *testing.T) {
	n, err := NewFunctionNodeFromState("n", dummyFnInvalid, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState() error = %v", err)
	}
	inner, err := NewWorkflowNode("inner", Chain(Start, n))
	if err != nil {
		t.Fatalf("NewWorkflowNode() error = %v", err)
	}

	w, err := New("wf", Chain(Start, inner), WithStateSchema(ss3PipelineSchema(t)))

	if w == nil || err != nil {
		t.Errorf("New() = %v, %v, want a workflow and nil", w, err)
	}
}
