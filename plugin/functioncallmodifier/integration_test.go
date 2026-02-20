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

package functioncallmodifier_test

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/plugin/functioncallmodifier"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/functiontool"
)

//go:generate go test -v -httprecord=testdata/.*\.httprr

func TestPluginCallbackIntegration(t *testing.T) {
	functionTool, err := functiontool.New(functiontool.Config{
		Name: "other_tool",
	}, okFunc)
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}

	testCases := []struct {
		name                     string
		tools                    func(agent.Agent) []tool.Tool
		wantSkillStateValue      string
		wantRationaleStateValue  string
		shouldHaveSkillState     bool
		shouldHaveRationaleState bool
	}{
		{
			name:                     "no relevant tools",
			tools:                    func(a agent.Agent) []tool.Tool { return []tool.Tool{functionTool} },
			shouldHaveSkillState:     false,
			shouldHaveRationaleState: false,
		},
		{
			name: "agent tool default schema",
			tools: func(a agent.Agent) []tool.Tool {
				agentToolDefault := agenttool.New(a, nil)
				return []tool.Tool{agentToolDefault}
			},
			wantSkillStateValue:      "add",
			wantRationaleStateValue:  "The user is asking to add two numbers, and the calculator tool with the add skill can perform this operation.",
			shouldHaveSkillState:     true,
			shouldHaveRationaleState: true,
		},
		{
			name:                     "transfer to agent tool",
			tools:                    func(a agent.Agent) []tool.Tool { return []tool.Tool{} },
			wantSkillStateValue:      "add",
			wantRationaleStateValue:  "The user is asking to add two numbers, and the calculator agent has an add skill to perform this operation.",
			shouldHaveSkillState:     true,
			shouldHaveRationaleState: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")

			model, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", testutil.NewGeminiTestClientConfig(t, httpRecordFilename))
			if err != nil {
				t.Fatalf("gemini.NewModel failed: %v", err)
			}

			calc, err := llmagent.New(llmagent.Config{
				Name:        "calculator",
				Description: "calculator agent\n Skills: add, subtract, multiply, divide",
				Instruction: "You are a calculator agent. You can calculate numbers.",
				Model:       model,
			})
			if err != nil {
				t.Fatalf("NewLLMAgent calculator failed: %v", err)
			}

			tools := tc.tools(calc)
			subAgents := []agent.Agent{}
			if len(tools) == 0 {
				subAgents = append(subAgents, calc)
			}

			a, err := llmagent.New(llmagent.Config{
				Name:        "transfer_agent",
				Description: "transfer agent",
				Instruction: "You are a transfer agent. You can transfer to other agents using your tools.",
				Model:       model,
				Tools:       tools,
				SubAgents:   subAgents,
			})
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}

			functionCallModifierPlugin, err := functioncallmodifier.NewPlugin(functioncallmodifier.FunctionCallModifierConfig{
				Predicate: func(toolName string) bool {
					return toolName == "transfer_to_agent" || toolName == "calculator"
				},
				Args: map[string]*genai.Schema{
					"skill_id": {
						Description: "The specific skill to be utilized by the agent.",
						Type:        "STRING",
					},
					"rationale": {
						Description: "The reasoning behind selecting this agent and skill.",
						Type:        "STRING",
					},
				},
				OverrideDescription: func(originalDescription string) string {
					return fmt.Sprintf("This tool can now optionally accept skill_id and rationale parameters to guide skill-based orchestration. %s", originalDescription)
				},
			})
			if err != nil {
				t.Fatalf("New plugin failed: %v", err)
			}

			pluginConfig := runner.PluginConfig{
				Plugins: []*plugin.Plugin{functionCallModifierPlugin},
			}

			appName := "test_app"
			sessionService := session.InMemoryService()

			_, err = sessionService.Create(t.Context(), &session.CreateRequest{
				AppName:   appName,
				UserID:    "id",
				SessionID: "test_session",
			})
			if err != nil {
				t.Fatalf("sessionService.Create failed: %v", err)
			}

			r, err := runner.New(runner.Config{
				AppName:        appName,
				Agent:          a,
				SessionService: sessionService,
				PluginConfig:   pluginConfig,
			})
			if err != nil {
				t.Fatalf("NewRunner failed: %v", err)
			}

			msg := genai.NewContentFromText("Can you add 2 and 2?", "user")

			stream := r.Run(t.Context(), "id", "test_session", msg, agent.RunConfig{StreamingMode: agent.StreamingModeNone})
			events, err := testutil.CollectEvents(stream)
			if err != nil {
				t.Fatalf("CollectEvents failed: %v", err)
			}

			fcId := ""

			// check function calls don't include skill_id and rationale
			for _, event := range events {
				for _, part := range event.Content.Parts {
					if part.FunctionCall != nil {
						fc := part.FunctionCall
						if fc.Args["skill_id"] != nil || fc.Args["rationale"] != nil {
							t.Errorf("function call includes skill_id or rationale: %v", fc)
						}
						fcId = fc.ID
					}
				}
			}

			// check if state includes skill_id and rationale
			resp, err := sessionService.Get(t.Context(), &session.GetRequest{
				AppName:   appName,
				UserID:    "id",
				SessionID: "test_session",
			})
			if err != nil {
				t.Fatalf("sessionService.Get failed: %v", err)
			}

			skillIdKey := fmt.Sprintf("%s/skill_id", fcId)
			skillId, err := resp.Session.State().Get(skillIdKey)
			if tc.shouldHaveSkillState {
				if err != nil {
					t.Fatalf("State().Get(%q) unexpected error: %v", skillIdKey, err)
				}
				if skillId != tc.wantSkillStateValue {
					t.Errorf("want skill_id %q, got %q", tc.wantSkillStateValue, skillId)
				}
			} else {
				if err == nil {
					t.Errorf("unexpectedly found skill_id in state with value: %q", skillId)
				} else if err != session.ErrStateKeyNotExist {
					t.Fatalf("State().Get(%q) unexpected error when expecting key not to exist: %v", skillIdKey, err)
				}
			}

			rationaleKey := fmt.Sprintf("%s/rationale", fcId)
			rationale, err := resp.Session.State().Get(rationaleKey)
			if tc.shouldHaveRationaleState {
				if err != nil {
					t.Fatalf("State().Get(%q) unexpected error: %v", rationaleKey, err)
				}
				if rationale != tc.wantRationaleStateValue {
					t.Errorf("want rationale %q, got %q", tc.wantRationaleStateValue, rationale)
				}
			} else {
				if err == nil {
					t.Errorf("unexpectedly found rationale in state with value: %q", rationale)
				} else if err != session.ErrStateKeyNotExist {
					t.Fatalf("State().Get(%q) unexpected error when expecting key not to exist: %v", rationaleKey, err)
				}
			}
		})
	}
}
