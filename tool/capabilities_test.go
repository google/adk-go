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

package tool_test

import (
	"errors"
	"fmt"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type transparentTool struct{ tool.Tool }

func (t transparentTool) Unwrap() tool.Tool { return t.Tool }

// This capability is deliberately unknown to the framework, like a capability
// added after a third-party wrapper was written.
type futureCapability interface {
	Enabled() bool
}

type futureTool struct {
	tool.Tool
	values []string
}

func (futureTool) Enabled() bool { return true }

type disabledTool struct{ transparentTool }

func (disabledTool) Enabled() bool { return false }

func TestAs(t *testing.T) {
	base := &publicCapabilityTool{}
	future := futureTool{Tool: base, values: []string{}}
	disabled := disabledTool{transparentTool{future}}
	for _, tc := range []struct {
		name    string
		tool    tool.Tool
		found   bool
		enabled bool
	}{
		{name: "nil"},
		{name: "missing capability", tool: base},
		{name: "non-comparable tool", tool: future, found: true, enabled: true},
		{name: "wrapped capability", tool: transparentTool{future}, found: true, enabled: true},
		{name: "nested wrappers", tool: transparentTool{transparentTool{future}}, found: true, enabled: true},
		{name: "wrapped missing capability", tool: transparentTool{base}},
		{name: "nil inner tool", tool: transparentTool{}},
		{name: "outer override", tool: disabled, found: true},
		{name: "nested override", tool: transparentTool{disabled}, found: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tool.As[futureCapability](tc.tool)
			if ok != tc.found {
				t.Fatalf("As() found = %t, want %t", ok, tc.found)
			}
			if !ok {
				if got != nil {
					t.Fatal("As() returned a nonzero value for an absent capability")
				}
				return
			}
			if got.Enabled() != tc.enabled {
				t.Fatal("As() resolved the wrong capability implementation")
			}
		})
	}
}

type runOnlyTool struct{ transparentTool }

func (runOnlyTool) Run(agent.Context, any) (map[string]any, error) {
	return nil, errors.New("incomplete function capability")
}

func TestAsRequiresCompleteCapability(t *testing.T) {
	base := &publicCapabilityTool{}
	wrapped := runOnlyTool{transparentTool{base}}
	got, ok := tool.As[tool.FunctionTool](wrapped)
	if !ok || got != base {
		t.Fatal("As() did not resolve the inner complete FunctionTool")
	}
}

type publicCapabilityTool struct{}

var (
	_ tool.FunctionTool                     = (*publicCapabilityTool)(nil)
	_ tool.StreamingFunctionTool            = (*publicCapabilityTool)(nil)
	_ tool.RequestProcessor                 = (*publicCapabilityTool)(nil)
	_ tool.ResponseDeferrer                 = (*publicCapabilityTool)(nil)
	_ tool.SkipSummarizationResultDisplayer = (*publicCapabilityTool)(nil)
	_ tool.Wrapper                          = transparentTool{}
)

func (*publicCapabilityTool) Name() string        { return "custom" }
func (*publicCapabilityTool) Description() string { return "A custom tool." }
func (*publicCapabilityTool) IsLongRunning() bool { return false }
func (t *publicCapabilityTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name()}
}
func (*publicCapabilityTool) Run(agent.Context, any) (map[string]any, error) { return nil, nil }
func (*publicCapabilityTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(func(string, error) bool) {}
}
func (*publicCapabilityTool) ProcessRequest(agent.Context, *model.LLMRequest) error { return nil }
func (*publicCapabilityTool) DefersResponse() bool                                  { return true }
func (*publicCapabilityTool) DisplayResultOnSkipSummarization() bool                { return true }

// A retry decorator overrides one complete capability and uses Unwrap to
// preserve every other capability on its inner tool.
type retryTool struct {
	tool.FunctionTool
}

var (
	_ tool.FunctionTool = (*retryTool)(nil)
	_ tool.Wrapper      = (*retryTool)(nil)
)

func (t *retryTool) Unwrap() tool.Tool { return t.FunctionTool }

func (t *retryTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	result, err := t.FunctionTool.Run(ctx, args)
	if err != nil {
		return t.FunctionTool.Run(ctx, args)
	}
	return result, nil
}

type flakyTool struct {
	attempts int
}

func (*flakyTool) Name() string        { return "flaky" }
func (*flakyTool) Description() string { return "A tool with a temporary failure." }
func (*flakyTool) IsLongRunning() bool { return false }
func (t *flakyTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: t.Name()}
}

func (t *flakyTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}
func (*flakyTool) DefersResponse() bool { return true }

func (t *flakyTool) Run(agent.Context, any) (map[string]any, error) {
	t.attempts++
	if t.attempts == 1 {
		return nil, errors.New("temporary failure")
	}
	return map[string]any{"attempts": t.attempts}, nil
}

func ExampleAs() {
	var wrapped tool.Tool = &retryTool{FunctionTool: &flakyTool{}}
	callable, ok := tool.As[tool.FunctionTool](wrapped)
	if !ok {
		return
	}
	result, err := callable.Run(nil, nil)
	if err != nil {
		return
	}
	fmt.Println("Attempts:", result["attempts"])
	_, preserved := tool.As[tool.ResponseDeferrer](wrapped)
	fmt.Println("Deferred response capability:", preserved)
	// Output:
	// Attempts: 2
	// Deferred response capability: true
}

// Exposing execution methods alone does not satisfy the public tool contracts.
type executionOnlyTool struct{ tool.Tool }

func (*executionOnlyTool) Declaration() *genai.FunctionDeclaration        { return nil }
func (*executionOnlyTool) Run(agent.Context, any) (map[string]any, error) { return nil, nil }
func (*executionOnlyTool) RunStream(agent.Context, any) iter.Seq2[string, error] {
	return func(func(string, error) bool) {}
}

func TestCallableContractsRequireRequestProcessor(t *testing.T) {
	var executionOnly tool.Tool = &executionOnlyTool{Tool: &publicCapabilityTool{}}
	if _, ok := executionOnly.(tool.FunctionTool); ok {
		t.Error("FunctionTool accepted a tool without ProcessRequest")
	}
	if _, ok := executionOnly.(tool.StreamingFunctionTool); ok {
		t.Error("StreamingFunctionTool accepted a tool without ProcessRequest")
	}
}
