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

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/adk/agent"
)

func TestNewFunctionNodeWithSchema(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type Output struct {
		Result string `json:"result"`
	}

	upperFn := func(ctx agent.InvocationContext, input Input) (map[string]any, error) {
		return map[string]any{"result": strings.ToUpper(input.Value)}, nil
	}

	ischema, err := jsonschema.For[Input](nil)
	if err != nil {
		t.Fatalf("jsonschema.For[Input] failed: %v", err)
	}
	oschema, err := jsonschema.For[Output](nil)
	if err != nil {
		t.Fatalf("jsonschema.For[Output] failed: %v", err)
	}

	node, err := NewFunctionNodeWithSchema[Input, map[string]any]("upper", upperFn, ischema, oschema, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}

	if node.Name() != "upper" {
		t.Errorf("expected name 'upper', got %s", node.Name())
	}

	// Test execution with valid input
	mockCtx := &MockInvocationContext{sess: nil}
	events := node.Run(mockCtx, Input{Value: "hello"})

	count := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++
		output, ok := ev.Actions.StateDelta["output"]
		if !ok {
			t.Fatal("expected output in state delta")
		}
		typedOutput, ok := output.(map[string]any)
		if !ok {
			t.Fatalf("expected map[string]any type, got %T", output)
		}
		if typedOutput["result"] != "HELLO" {
			t.Errorf("expected 'HELLO', got %s", typedOutput["result"])
		}
	}
	if count != 1 {
		t.Errorf("expected 1 event, got %d", count)
	}
}

func TestNewFunctionNodeWithSchema_ValidationError(t *testing.T) {
	type Input struct {
		Value string `json:"value"`
	}
	type TargetOutput struct {
		Result int `json:"result"`
	}

	fn := func(ctx agent.InvocationContext, input Input) (map[string]any, error) {
		return map[string]any{"result": "not-an-int"}, nil
	}

	ischema, _ := jsonschema.For[Input](nil)
	oschema, _ := jsonschema.For[TargetOutput](nil)

	node, err := NewFunctionNodeWithSchema[Input, map[string]any]("test", fn, ischema, oschema, defaultNodeConfig)
	if err != nil {
		t.Fatalf("NewFunctionNodeWithSchema failed: %v", err)
	}

	mockCtx := &MockInvocationContext{sess: nil}
	events := node.Run(mockCtx, Input{Value: "hello"})

	for _, err := range events {
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "validation failed for output") {
			t.Errorf("expected validation error, got: %v", err)
		}
		return // We expect error on first event/iteration
	}
	t.Error("expected at least one event/error")
}
