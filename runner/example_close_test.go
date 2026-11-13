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

package runner_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// fixedReplyModel stands in for a real model such as Gemini.
type fixedReplyModel struct{}

func (fixedReplyModel) Name() string { return "fixed-reply" }

func (fixedReplyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel)}, nil)
	}
}

// connectionToolset stands in for a toolset that holds a resource, such as
// an MCP toolset with an open session. It implements io.Closer.
type connectionToolset struct{}

func (connectionToolset) Name() string                                     { return "connection" }
func (connectionToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }

func (connectionToolset) Close() error {
	fmt.Println("closed")
	return nil
}

// Deferring Close releases the toolsets of the runner's agents, and then its
// plugins, when the application is done with the runner.
func ExampleRunner_Close() {
	a, err := llmagent.New(llmagent.Config{
		Name:     "assistant",
		Model:    fixedReplyModel{},
		Toolsets: []tool.Toolset{connectionToolset{}},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "example",
		Agent:          a,
		SessionService: session.InMemoryService(),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			log.Print(err)
		}
	}()

	// Run the agent with r.Run here.

	// Output:
	// closed
}
