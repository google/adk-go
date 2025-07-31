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

package main

import (
	"context"
	"iter"

	"github.com/google/adk-go"
	"github.com/google/adk-go/agent"
	"github.com/google/adk-go/examples"
	"google.golang.org/genai"
)

type MyAgent struct {
	agentSpec *adk.AgentSpec
}

func (a *MyAgent) Spec() *adk.AgentSpec {
	return a.agentSpec
}

func (a *MyAgent) Run(ctx context.Context, ictx *adk.InvocationContext) iter.Seq2[*adk.Event, error] {
	return func(yield func(*adk.Event, error) bool) {
		yield(&adk.Event{
			LLMResponse: &adk.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							Text: "Hello from MyAgent!\n",
						},
					},
				},
			},
		}, nil)
	}
}

var _ adk.Agent = (*MyAgent)(nil)

func NewMyAgent() *MyAgent {
	return &MyAgent{}
}

func main() {
	ctx := context.Background()

	myAgent := NewMyAgent()
	myAgent.agentSpec = &adk.AgentSpec{
		Name:        "my_custom_agent",
		Description: "A custom agent that responds with a greeting.",
	}
	myAgent.agentSpec.Init(myAgent)

	loopAgent, err := agent.NewLoopAgent("loop_agent", 3,
		agent.WithDescription("A loop agent that runs sub-agents"),
		agent.WithSubAgents(myAgent))
	if err != nil {
		panic(err)
	}

	examples.Run(ctx, loopAgent)
}
