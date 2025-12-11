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

package adka2a

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/adk/tool/loadartifactstool"
)

func must[T agent.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

func TestGetAgentSkills_LLMAgent(t *testing.T) {
	googleSearch, loadArtifacts := geminitool.GoogleSearch{}, loadartifactstool.New()

	testCases := []struct {
		name  string
		agent agent.Agent
		want  []a2a.AgentSkill
	}{
		{
			name:  "custom agent",
			agent: must(agent.New(agent.Config{Name: "Test", Description: "Test test"})),
			want: []a2a.AgentSkill{{
				ID:          "Test",
				Description: "Test test",
				Name:        "custom",
				Tags:        []string{"custom_agent"},
			}},
		},
		{
			name: "llm with instruction",
			agent: must(llmagent.New(llmagent.Config{
				Name:        "Test LLM",
				Description: "Test llm.",
				Instruction: "You're a helpful agent, only respond with useful information.",
			})),
			want: []a2a.AgentSkill{{
				ID:          "Test LLM",
				Description: "Test llm. I am a helpful agent, only respond with useful information.",
				Name:        "model",
				Tags:        []string{"llm"},
			}},
		},
		{
			name: "llm with tools",
			agent: must(llmagent.New(llmagent.Config{
				Name:        "Test LLM",
				Description: "Test llm.",
				Tools:       []tool.Tool{loadArtifacts, googleSearch},
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test LLM",
					Description: "Test llm.",
					Name:        "model",
					Tags:        []string{"llm"},
				},
				{
					ID:          "Test LLM-" + loadArtifacts.Name(),
					Name:        loadArtifacts.Name(),
					Description: loadArtifacts.Description(),
					Tags:        []string{"llm", "tools"},
				},
				{
					ID:          "Test LLM-" + googleSearch.Name(),
					Name:        googleSearch.Name(),
					Description: googleSearch.Description(),
					Tags:        []string{"llm", "tools"},
				},
			},
		},
		{
			name: "empty loop agent",
			agent: must(loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{Name: "Test", Description: "Test test."},
			})),
			want: []a2a.AgentSkill{
				{ID: "Test", Description: "Test test.", Name: "workflow", Tags: []string{"loop_workflow"}},
			},
		},
		{
			name: "loop agent",
			agent: must(loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{
					Name:        "Test",
					Description: "Test test.",
					SubAgents: []agent.Agent{
						must(agent.New(agent.Config{Name: "Inner 1", Description: "Inner 1 description"})),
						must(agent.New(agent.Config{Name: "Inner 2", Description: "Inner 2 description"})),
					},
				},
				MaxIterations: 5,
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test",
					Description: "Test test. This agent will Inner 1 description and Inner 2 description in a loop (max 5 iterations).",
					Name:        "workflow",
					Tags:        []string{"loop_workflow"},
				},
				{
					ID:          "Test-sub-agents",
					Description: "Orchestrates: Inner 1 description; Inner 2 description",
					Name:        "sub-agents",
					Tags:        []string{"loop_workflow", "orchestration"},
				},
				{
					ID:          "Inner 1_Inner 1",
					Description: "Inner 1 description",
					Name:        "Inner 1: custom",
					Tags:        []string{"sub_agent:Inner 1", "custom_agent"},
				},
				{
					ID:          "Inner 2_Inner 2",
					Description: "Inner 2 description",
					Name:        "Inner 2: custom",
					Tags:        []string{"sub_agent:Inner 2", "custom_agent"},
				},
			},
		},
		{
			name: "unlimited loop agent",
			agent: must(loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{
					Name:        "Test",
					Description: "Test test.",
					SubAgents: []agent.Agent{
						must(agent.New(agent.Config{Name: "Inner 1", Description: "Inner 1 description"})),
					},
				},
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test",
					Description: "Test test. This agent will Inner 1 description in a loop (max unlimited iterations).",
					Name:        "workflow",
					Tags:        []string{"loop_workflow"},
				},
				{
					ID:          "Test-sub-agents",
					Description: "Orchestrates: Inner 1 description",
					Name:        "sub-agents",
					Tags:        []string{"loop_workflow", "orchestration"},
				},
				{
					ID:          "Inner 1_Inner 1",
					Description: "Inner 1 description",
					Name:        "Inner 1: custom",
					Tags:        []string{"sub_agent:Inner 1", "custom_agent"},
				},
			},
		},

		{
			name: "empty sequential agent",
			agent: must(sequentialagent.New(sequentialagent.Config{
				AgentConfig: agent.Config{Name: "Test", Description: "Test test."},
			})),
			want: []a2a.AgentSkill{
				{ID: "Test", Description: "Test test.", Name: "workflow", Tags: []string{"sequential_workflow"}},
			},
		},
		{
			name: "sequential agent",
			agent: must(sequentialagent.New(sequentialagent.Config{
				AgentConfig: agent.Config{
					Name:        "Test",
					Description: "Test test.",
					SubAgents: []agent.Agent{
						must(agent.New(agent.Config{Name: "Inner 1", Description: "Inner 1 description"})),
						must(agent.New(agent.Config{Name: "Inner 2", Description: "Inner 2 description"})),
					},
				},
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test",
					Description: "Test test. First, this agent will Inner 1 description. Finally, this agent will Inner 2 description.",
					Name:        "workflow",
					Tags:        []string{"sequential_workflow"},
				},
				{
					ID:          "Test-sub-agents",
					Description: "Orchestrates: Inner 1 description; Inner 2 description",
					Name:        "sub-agents",
					Tags:        []string{"sequential_workflow", "orchestration"},
				},
				{
					ID:          "Inner 1_Inner 1",
					Description: "Inner 1 description",
					Name:        "Inner 1: custom",
					Tags:        []string{"sub_agent:Inner 1", "custom_agent"},
				},
				{
					ID:          "Inner 2_Inner 2",
					Description: "Inner 2 description",
					Name:        "Inner 2: custom",
					Tags:        []string{"sub_agent:Inner 2", "custom_agent"},
				},
			},
		},
		{
			name: "empty parallel agent",
			agent: must(parallelagent.New(parallelagent.Config{
				AgentConfig: agent.Config{Name: "Test", Description: "Test test."},
			})),
			want: []a2a.AgentSkill{
				{ID: "Test", Description: "Test test.", Name: "workflow", Tags: []string{"parallel_workflow"}},
			},
		},
		{
			name: "parallel agent",
			agent: must(parallelagent.New(parallelagent.Config{
				AgentConfig: agent.Config{
					Name:        "Test",
					Description: "Test test.",
					SubAgents: []agent.Agent{
						must(agent.New(agent.Config{Name: "Inner 1", Description: "Inner 1 description"})),
						must(agent.New(agent.Config{Name: "Inner 2", Description: "Inner 2 description"})),
					},
				},
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test",
					Description: "Test test. This agent will Inner 1 description and Inner 2 description simultaneously.",
					Name:        "workflow",
					Tags:        []string{"parallel_workflow"},
				},
				{
					ID:          "Test-sub-agents",
					Description: "Orchestrates: Inner 1 description; Inner 2 description",
					Name:        "sub-agents",
					Tags:        []string{"parallel_workflow", "orchestration"},
				},
				{
					ID:          "Inner 1_Inner 1",
					Description: "Inner 1 description",
					Name:        "Inner 1: custom",
					Tags:        []string{"sub_agent:Inner 1", "custom_agent"},
				},
				{
					ID:          "Inner 2_Inner 2",
					Description: "Inner 2 description",
					Name:        "Inner 2: custom",
					Tags:        []string{"sub_agent:Inner 2", "custom_agent"},
				},
			},
		},
		{
			name: "deep subagents",
			agent: must(parallelagent.New(parallelagent.Config{
				AgentConfig: agent.Config{
					Name:        "Test",
					Description: "Test test.",
					SubAgents: []agent.Agent{
						must(loopagent.New(loopagent.Config{
							AgentConfig: agent.Config{
								Name:        "Nested",
								Description: "Nested loop",
								SubAgents: []agent.Agent{
									must(llmagent.New(llmagent.Config{
										Name:        "Test LLM",
										Description: "Test llm",
										Tools:       []tool.Tool{loadArtifacts},
									})),
									must(sequentialagent.New(sequentialagent.Config{
										AgentConfig: agent.Config{
											Name:        "Leaf",
											Description: "Leaf agent",
											SubAgents: []agent.Agent{
												must(agent.New(agent.Config{Name: "Leaf", Description: "leaf"})),
											},
										},
									})),
								},
							},
						})),
					},
				},
			})),
			want: []a2a.AgentSkill{
				{
					ID:          "Test",
					Description: "Test test. This agent will Nested loop simultaneously.",
					Name:        "workflow",
					Tags:        []string{"parallel_workflow"},
				},
				{
					ID:          "Test-sub-agents",
					Description: "Orchestrates: Nested loop",
					Name:        "sub-agents",
					Tags:        []string{"parallel_workflow", "orchestration"},
				},
				{
					ID:          "Nested_Nested",
					Description: "Nested loop This agent will Test llm and Leaf agent in a loop (max unlimited iterations).",
					Name:        "Nested: workflow",
					Tags:        []string{"sub_agent:Nested", "loop_workflow"},
				},
				{
					Description: "Orchestrates: Test llm; Leaf agent",
					ID:          "Nested_Nested-sub-agents",
					Name:        "Nested: sub-agents",
					Tags:        []string{"sub_agent:Nested", "loop_workflow", "orchestration"},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildAgentSkills(tc.agent)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("BuildAgentSkills() wrong result (+got,-want)\ngot = %+v\nwant = %+v\ndiff = %s", got, tc.want, diff)
			}
		})
	}
}

func TestReplacePronouns(t *testing.T) {
	testCases := []struct {
		input string
		want  string
	}{
		{
			input: "you are an agent. you were an agent, you're an agent, you've tasks, your tasks",
			want:  "I am an agent. I was an agent, I am an agent, I have tasks, my tasks",
		},
		{
			input: "You should do your work and it will be yours.",
			want:  "I should do my work and it will be mine.",
		},
		{
			input: "YOU should do YOUR work and it will be YOURS.",
			want:  "I should do my work and it will be mine.",
		},
		{
			input: "You should do Your work and it will be Yours.",
			want:  "I should do my work and it will be mine.",
		},
		{
			input: "This is a test message without pronouns.",
			want:  "This is a test message without pronouns.",
		},
		{
			input: "youth, yourself, yourname",
			want:  "youth, yourself, yourname",
		},
		{
			input: "You are a helpful chatbot",
			want:  "I am a helpful chatbot",
		},
		{
			input: "Your task is to be helpful",
			want:  "my task is to be helpful",
		},
		{
			input: "you you you",
			want:  "I I I",
		},
	}
	for _, tc := range testCases {
		got := replacePronouns(tc.input)
		if got != tc.want {
			t.Errorf("replacePronouns(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
