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

package workflowinternal

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

// taskAgentTool wraps a task-mode sub-agent so that a parent LLMAgent can
// delegate to it via a function call. The presence of a taskAgentTool in the
// parent's tools is also the marker used by the workflow-mode LLM agent
// wrapper to detect task-delegation function calls.
//
// This is the Go counterpart of adk-python's
// `google.adk.tools.agent_tool._TaskAgentTool`. The full delegation logic
// (dispatching the sub-agent as a workflow node, isolation scopes, etc.) is
// not yet wired in adk-go; for now the tool exposes the sub-agent's input
// schema and returns a stub response so that the LlmAgent constructor can
// register it.
type taskAgentTool struct {
	agent agent.Agent
	tool.Tool
}

func NewTaskAgentTool(a agent.Agent) (tool.Tool, error) {
	// TODO: implement
	return nil, nil
}
