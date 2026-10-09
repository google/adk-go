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

package replayplugin_test

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/configurable/conformance/replayplugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type replayWrapper struct{ tool.Tool }

func (t replayWrapper) Unwrap() tool.Tool { return t.Tool }

type replayCountingTool struct {
	tool.FunctionTool
	calls int
}

func (t *replayCountingTool) Unwrap() tool.Tool { return t.FunctionTool }
func (t *replayCountingTool) Run(agent.Context, any) (map[string]any, error) {
	t.calls++
	return nil, nil
}

func TestReplayWrappedTools(t *testing.T) {
	for _, isAgent := range []bool{false, true} {
		name := "function"
		if isAgent {
			name = "agent"
		}
		t.Run(name, func(t *testing.T) {
			var inner tool.Tool
			if isAgent {
				a, err := llmagent.New(llmagent.Config{Name: "test_tool"})
				if err != nil {
					t.Fatal("create agent")
				}
				inner = agenttool.New(a, nil)
			} else {
				var err error
				inner, err = functiontool.New(functiontool.Config{Name: "test_tool"}, func(agent.Context, struct{}) (struct{}, error) {
					t.Error("replay bypassed the outer Run override")
					return struct{}{}, nil
				})
				if err != nil {
					t.Fatal("create function tool")
				}
			}
			fn, ok := tool.As[tool.FunctionTool](inner)
			if !ok {
				t.Fatal("missing function capability")
			}
			counted := &replayCountingTool{FunctionTool: fn}
			wrapped := replayWrapper{Tool: replayWrapper{Tool: counted}}
			dir := t.TempDir()
			createRecordingsFile(t, dir, `recordings:
  - user_message_index: 0
    agent_name: test_agent
    tool_recording:
      tool_call:
        name: test_tool
        args: {}
      tool_response:
        name: test_tool
        response:
          result: recorded
`)
			state := &MockState{data: map[string]any{"_adk_replay_config": map[string]any{
				"dir": dir, "user_message_index": 0,
			}}}
			p := replayplugin.MustNew(dir)
			if _, err := p.BeforeRunCallback()(&MockInvocationContext{
				session: &MockSession{state: state}, invocationID: "wrapped-replay",
			}); err != nil {
				t.Fatal("load replay recording")
			}
			ctx := &MockToolContext{state: state, invocationID: "wrapped-replay", agentName: "test_agent"}
			result, err := p.BeforeToolCallback()(ctx, wrapped, map[string]any{})
			if err != nil || result["result"] != "recorded" {
				t.Fatal("replay did not return recorded response")
			}
			wantCalls := 1
			if isAgent {
				wantCalls = 0
			}
			if counted.calls != wantCalls {
				t.Fatalf("Run called %d times, want %d", counted.calls, wantCalls)
			}
		})
	}
}
