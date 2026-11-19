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

package llminternal

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/tool/geminitool"
)

// Python class names of the built-in search tools the guard recognizes. The
// error text names them as Python does.
const (
	googleSearchToolName        = "GoogleSearchTool"
	enterpriseWebSearchToolName = "EnterpriseWebSearchTool"
)

// builtinSearchGuard ports Python's built-in search check of the agent transfer step
// (flows/llm_flows/extensions/_agent_transfer.py:58-69 and _get_incompatible_builtin_tool_error,
// :298-354). cur is the agent whose request is being built and targets are the transfer
// targets transferTargets computed for it (len(targets) > 0).
//
//   - (false, nil): cur has no incompatible built-in search tool for the step's model. The caller
//     goes on and adds the tool and the instruction.
//   - (true, nil): cur has one, and none of targets is one of cur's own sub-agents. The caller adds
//     neither the tool nor the instruction.
//   - (false, err): cur has one, and at least one target is one of cur's own sub-agents. err
//     names cur and the tool and the caller yields it as the step's error.
//
// "Incompatible" means: cur's own Tools hold a built-in search tool that Python treats as not
// combinable with function calling (geminitool.GoogleSearch, geminitool.EnterpriseWebSearch, by
// value or by pointer), and the step's model, processorModel(ctx, f).Name() ("" when no model
// resolves), is not Gemini 3 or later (Python's _supports_builtin_tools_with_function_calling:
// after extract_model_name, the name starts with "gemini-" and the text up to the next "-" parses
// as a version whose major is at least 3). Toolsets are not looked into, as in Python.
// f may be nil. The model is read only after an incompatible tool is found, so an agent
// without one never touches its model (several existing tests run the processor with a
// model double whose Name panics).
func builtinSearchGuard(ctx agent.InvocationContext, f *Flow, cur agent.Agent, targets []agent.Agent) (skip bool, err error) {
	searchTool := incompatibleSearchTool(cur)
	if searchTool == "" {
		return false, nil
	}
	var modelName string
	if m := processorModel(ctx, f); m != nil {
		modelName = m.Name()
	}
	if supportsBuiltinToolsWithFunctionCalling(modelName) {
		return false, nil
	}
	if !hasSubAgentTarget(cur, targets) {
		return true, nil
	}
	return false, incompatibleSearchToolError(cur.Name(), searchTool)
}

// incompatibleSearchTool returns the Python class name of the first tool in
// cur's own Tools that Python's check treats as a built-in search tool, or ""
// when there is none or cur is not an LLM agent. adk-go has no Vertex AI
// Search tool type and no bypass_multi_tools_limit option, so every
// GoogleSearch counts as Python's GoogleSearchTool without the bypass.
func incompatibleSearchTool(cur agent.Agent) string {
	a := asLLMAgent(cur)
	if a == nil {
		return ""
	}
	for _, t := range a.internal().Tools {
		switch t.(type) {
		case geminitool.GoogleSearch, *geminitool.GoogleSearch:
			return googleSearchToolName
		case geminitool.EnterpriseWebSearch, *geminitool.EnterpriseWebSearch:
			return enterpriseWebSearchToolName
		}
	}
	return ""
}

// supportsBuiltinToolsWithFunctionCalling reports whether model accepts
// built-in tools in the same request as function declarations, which Gemini 3
// and later do. Follows adk-python's _supports_builtin_tools_with_function_calling
// (utils/model_name_utils.py:199-227); an empty or non-Gemini name does not.
func supportsBuiltinToolsWithFunctionCalling(model string) bool {
	if !googlellm.IsGeminiModelExact(model) {
		return false
	}
	// Python reads the version from extract_model_name's result. For a Gemini
	// name that result is model's last path segment, unless the result itself
	// holds a "/", which no real model name does.
	name := model[strings.LastIndex(model, "/")+1:]
	rest, ok := strings.CutPrefix(name, "gemini-")
	if !ok {
		return false
	}
	version, _, _ := strings.Cut(rest, "-")
	major, ok := majorVersion(version)
	return ok && major >= 3
}

// majorVersion returns the first number of a version made of dot-separated
// decimal numbers ("3", "2.5", "3.1"), and false for anything else ("pro",
// "exp", "", "3.").
func majorVersion(version string) (int, bool) {
	parts := strings.Split(version, ".")
	if slices.ContainsFunc(parts, func(p string) bool {
		return p == "" || strings.Trim(p, "0123456789") != ""
	}) {
		return 0, false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, false
	}
	return major, true
}

// hasSubAgentTarget reports whether one of targets is one of cur's own
// sub-agents. Agents are compared by name, which is unique in a tree, because
// == on an agent.Agent panics when its dynamic type is not comparable.
func hasSubAgentTarget(cur agent.Agent, targets []agent.Agent) bool {
	for _, sub := range cur.SubAgents() {
		if slices.ContainsFunc(targets, func(t agent.Agent) bool { return t.Name() == sub.Name() }) {
			return true
		}
	}
	return false
}

// incompatibleSearchToolError returns the error Python raises for an agent
// with sub-agent transfer targets and a built-in search tool on a model
// before Gemini 3. The text is Python's up to the tool name; the remedies are
// the ones adk-go offers (it has no bypass_multi_tools_limit option).
func incompatibleSearchToolError(agentName, searchTool string) error {
	if searchTool == enterpriseWebSearchToolName {
		return fmt.Errorf("Agent '%s' has sub-agent transfer targets but is configured with EnterpriseWebSearchTool."+
			" Gemini models before 3.0 do not allow EnterpriseWebSearchTool to be combined with function calling"+
			" (agent delegation). To enable both, use a Gemini 3+ model or move EnterpriseWebSearchTool into a"+
			" single_turn sub-agent", agentName)
	}
	return fmt.Errorf("Agent '%s' has sub-agent transfer targets but is configured with %s."+
		" Gemini models before 3.0 do not allow built-in search tools to be combined with function calling"+
		" (agent delegation). To enable both search and sub-agent delegation, use a Gemini 3+ model", agentName, searchTool)
}
