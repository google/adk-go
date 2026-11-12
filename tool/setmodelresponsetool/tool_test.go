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

package setmodelresponsetool_test

import (
	"strconv"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

// functionTool asserts that a constructor succeeded and returned a tool the
// flow can run.
func functionTool(t *testing.T, tl tool.Tool, err error) toolinternal.FunctionTool {
	t.Helper()
	if err != nil {
		t.Fatalf("constructor returned error: %v", err)
	}
	ft, ok := tl.(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("tool %T does not implement toolinternal.FunctionTool", tl)
	}
	return ft
}

func newTool(t *testing.T, schema *genai.Schema) toolinternal.FunctionTool {
	t.Helper()
	tl, err := setmodelresponsetool.New(schema)
	return functionTool(t, tl, err)
}

func newRawTool(t *testing.T, schema *jsonschema.Schema) toolinternal.FunctionTool {
	t.Helper()
	tl, err := setmodelresponsetool.NewFromJSONSchema(schema)
	return functionTool(t, tl, err)
}

func newToolContext(t *testing.T) agent.Context {
	t.Helper()
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	return agent.NewToolContext(invCtx, "call-1", &session.EventActions{}, nil)
}

func TestNew_NilSchemaErrors(t *testing.T) {
	if setmodelresponsetool.Name != "set_model_response" {
		t.Errorf("Name = %q, want %q", setmodelresponsetool.Name, "set_model_response")
	}
	const wantErr = "set_model_response: schema is nil"

	tl, err := setmodelresponsetool.New(nil)
	if tl != nil || err == nil || err.Error() != wantErr {
		t.Errorf("New(nil) = (%v, %v), want (nil, %q)", tl, err, wantErr)
	}

	tl, err = setmodelresponsetool.NewFromJSONSchema(nil)
	if tl != nil || err == nil || err.Error() != wantErr {
		t.Errorf("NewFromJSONSchema(nil) = (%v, %v), want (nil, %q)", tl, err, wantErr)
	}
}

func TestTool_ConcurrentUse(t *testing.T) {
	tl := newTool(t, &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"id":   {Type: genai.TypeInteger},
			"role": {Type: genai.TypeString, Default: "user"},
			"tags": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
		},
		Required: []string{"id"},
	})
	const n = 16
	ctxs := make([]agent.Context, n)
	for i := range ctxs {
		ctxs[i] = newToolContext(t)
	}

	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			_ = tl.Declaration()
			if _, err := tl.Run(ctxs[i], map[string]any{"id": strconv.Itoa(i), "tags": []any{"t"}}); err != nil {
				t.Errorf("Run() in goroutine %d: %v", i, err)
			}
		})
	}
	wg.Wait()

	for i, ctx := range ctxs {
		want := map[string]any{"id": float64(i), "role": "user", "tags": []any{"t"}}
		if diff := cmp.Diff(want, ctx.Actions().SetModelResponse); diff != "" {
			t.Errorf("goroutine %d: SetModelResponse mismatch (-want +got):\n%s", i, diff)
		}
	}
}
