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

// Package main demonstrates recovering from a function tool error in an ADK
// agent.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

var errInventoryUnavailable = errors.New("inventory service unavailable")

type lookupArgs struct {
	Item string `json:"item"`
}

type lookupResult struct {
	Status    string `json:"status"`
	Retryable bool   `json:"retryable"`
}

func lookupInventory(_ agent.Context, args lookupArgs) (lookupResult, error) {
	if args.Item == "sold-out" {
		return lookupResult{}, fmt.Errorf("%w for %q", errInventoryUnavailable, args.Item)
	}
	return lookupResult{Status: "in stock"}, nil
}

func recoverInventoryError(out io.Writer) llmagent.OnToolErrorCallback {
	return func(_ agent.Context, calledTool tool.Tool, _ map[string]any, err error) (map[string]any, error) {
		// Recover only the failure this application understands. Unknown failures
		// remain errors instead of being accidentally hidden.
		if !errors.Is(err, errInventoryUnavailable) {
			return nil, err
		}

		fmt.Fprintf(out, "Tool %s failed: %v\n", calledTool.Name(), err)
		return map[string]any{
			"status":    "temporarily unavailable",
			"retryable": true,
		}, nil
	}
}

func run(ctx context.Context, out io.Writer) error {
	inventoryTool, err := functiontool.New(functiontool.Config{
		Name:        "inventory_lookup",
		Description: "Checks whether an item is in stock.",
	}, lookupInventory)
	if err != nil {
		return fmt.Errorf("create inventory tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:                 "inventory_agent",
		Description:          "Answers inventory questions.",
		Model:                inventoryDemoModel{},
		Tools:                []tool.Tool{inventoryTool},
		OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{recoverInventoryError(out)},
	})
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}

	r, err := runner.NewInMemory("tool_error_example", a)
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	message := genai.NewContentFromText("Is the sold-out item available?", genai.RoleUser)
	for event, err := range r.Run(ctx, "example-user", "example-session", message, agent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("run agent: %w", err)
		}
		if event == nil || event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				fmt.Fprintf(out, "Agent: %s\n", part.Text)
			}
		}
	}
	return nil
}

// inventoryDemoModel makes the example deterministic and credential-free. Its
// first response calls the tool; its second turns the callback's fallback into
// the final agent response.
type inventoryDemoModel struct{}

func (inventoryDemoModel) Name() string { return "inventory-demo-model" }

func (inventoryDemoModel) GenerateContent(ctx context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		toolResponse := latestFunctionResponse(req.Contents)
		if toolResponse == nil {
			yield(&model.LLMResponse{Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "inventory-call-1",
					Name: "inventory_lookup",
					Args: map[string]any{"item": "sold-out"},
				}}},
			}}, nil)
			return
		}

		status, ok := toolResponse.Response["status"].(string)
		if !ok {
			yield(nil, errors.New("inventory tool response has no status"))
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText(
			fmt.Sprintf("Inventory is %s; please try again.", status),
			genai.RoleModel,
		)}, nil)
	}
}

func latestFunctionResponse(contents []*genai.Content) *genai.FunctionResponse {
	for i := len(contents) - 1; i >= 0; i-- {
		for j := len(contents[i].Parts) - 1; j >= 0; j-- {
			if response := contents[i].Parts[j].FunctionResponse; response != nil {
				return response
			}
		}
	}
	return nil
}

func main() {
	log.SetFlags(0)
	if err := run(context.Background(), os.Stdout); err != nil {
		log.Fatal(err)
	}
}
