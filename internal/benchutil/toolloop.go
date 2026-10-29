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

package benchutil

import (
	"context"
	"iter"
	"sync/atomic"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// ToolLoopModel drives a tool loop: each turn it answers the first Calls
// model calls with one function call to ToolName and the next one with text,
// so a turn costs Calls+1 model calls. A turn starts whenever the request ends
// on a user message that carries no function response, so one instance serves
// any number of sequential turns. It is not safe for concurrent use.
type ToolLoopModel struct {
	Calls    int
	ToolName string
	Args     map[string]any

	// ModelCalls counts GenerateContent calls.
	ModelCalls atomic.Int64

	step int
}

func (m *ToolLoopModel) Name() string { return "tool-loop" }

func (m *ToolLoopModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.ModelCalls.Add(1)
	if n := len(req.Contents); n == 0 || !hasFunctionResponse(req.Contents[n-1]) {
		m.step = 0
	}
	m.step++
	var content *genai.Content
	if m.step <= m.Calls {
		content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{Name: m.ToolName, Args: m.Args},
		}}}
	} else {
		content = genai.NewContentFromText(BenchText, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop, TurnComplete: true}, nil)
	}
}

func hasFunctionResponse(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// NoopArgs is the argument type of NoopTool.
type NoopArgs struct {
	Q string `json:"q,omitempty"`
}

// NoopResult is the result type of NoopTool.
type NoopResult struct {
	OK bool `json:"ok"`
}

// NoopTool returns a function tool that does nothing, so a tool loop measures
// the framework rather than the tool.
func NoopTool(name string) tool.Tool {
	t, err := functiontool.New(functiontool.Config{Name: name, Description: "Does nothing."},
		func(agent.Context, NoopArgs) (NoopResult, error) { return NoopResult{OK: true}, nil })
	if err != nil {
		panic(err)
	}
	return t
}

// CountingToolset is a static Toolset that counts how many times the flow
// resolves it.
type CountingToolset struct {
	ToolsetName string
	ToolList    []tool.Tool
	Resolutions atomic.Int64
}

func (c *CountingToolset) Name() string { return c.ToolsetName }

func (c *CountingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	c.Resolutions.Add(1)
	return c.ToolList, nil
}
