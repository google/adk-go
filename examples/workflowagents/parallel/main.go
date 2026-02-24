// Copyright 2025 Google LLC
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

// Package demonstrates a workflow agent that runs sub-agents in parallel.
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	rand "math/rand/v2"
	"os"
	"time"

	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/parallelagent"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/util/modelfactory"
)

func main() {
	ctx := context.Background()

	subAgent1, err := agent.New(agent.Config{
		Name:        "my_custom_agent_1",
		Description: "A custom agent that responds with a greeting.",
		Run:         myAgent{id: 1}.Run,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	subAgent2, err := agent.New(agent.Config{
		Name:        "my_custom_agent_2",
		Description: "A custom agent that responds with a greeting.",
		Run:         myAgent{id: 2}.Run,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "parallel_agent",
			Description: "A parallel agent that runs sub-agents",
			SubAgents:   []agent.Agent{subAgent1, subAgent2},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(parallelAgent),
	}

	l := full.NewLauncher()
	// 过滤掉-model和-model-name参数，避免与launcher参数冲突
	launcherArgs := modelfactory.ExtractLauncherArgs(os.Args[1:])
	if err = l.Execute(ctx, config, launcherArgs); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

type myAgent struct {
	id int
}

func (a myAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for range 3 {
			if !yield(&session.Event{
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{
							{
								Text: fmt.Sprintf("Hello from MyAgent id: %v!\n", a.id),
							},
						},
					},
				},
			}, nil) {
				return
			}

			r := 1 + rand.IntN(5)
			time.Sleep(time.Duration(r) * time.Second)
		}
	}
}
