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

// Package agentinfo describes an agent tree without running it: the
// instruction, tools and children of every agent reachable from a root.
//
// It backs the /apps/{app_name}/app-info endpoint, which external tooling
// (notably the Agents CLI evaluation flow) uses to learn an app's shape.
package agentinfo

import (
	"context"
	"log"
	"slices"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/tool"
)

// toolsetTimeout bounds the time spent resolving an agent's toolsets. Toolsets
// may reach out over the network (an MCP server, for example), and describing
// an app must not hang on one.
const toolsetTimeout = 10 * time.Second

// declarer is implemented by tools the model calls as functions.
type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

// genaiToolProvider is implemented by tools with a native [genai.Tool]
// representation, such as Google Search, which the model executes itself rather
// than calling as a function.
type genaiToolProvider interface {
	GenaiTool() *genai.Tool
}

// Collect walks the agent tree rooted at root and returns one
// [models.AgentInfo] per agent, keyed by agent name.
//
// Every kind of agent is included. Agents that are not LLM agents report an
// empty instruction and no tools, because they have neither; their children are
// still walked, so the returned map always describes the whole tree.
//
// An agent reachable by more than one path is reported once, which also stops
// a cyclic tree from recursing forever.
func Collect(ctx context.Context, appName string, root agent.Agent) map[string]*models.AgentInfo {
	if root == nil {
		return nil
	}

	agents := make(map[string]*models.AgentInfo)

	var walk func(a agent.Agent)
	walk = func(a agent.Agent) {
		if a == nil {
			return
		}
		if _, seen := agents[a.Name()]; seen {
			return
		}

		info := &models.AgentInfo{
			Name:        a.Name(),
			Description: a.Description(),
			// Never nil: the wire contract requires the field to be present,
			// and a nil slice would marshal to null.
			Tools: []*genai.Tool{},
		}
		// Record before recursing so that a cycle terminates here.
		agents[a.Name()] = info

		if llmAgent, ok := a.(llminternal.Agent); ok {
			state := llminternal.Reveal(llmAgent)
			// An agent whose instruction comes from an InstructionProvider
			// reports an empty instruction: resolving it needs session state
			// that does not exist outside of an invocation.
			info.Instruction = state.Instruction
			info.Tools = toolsInfo(ctx, appName, a.Name(), state)
		}

		for _, sub := range a.SubAgents() {
			if sub == nil {
				continue
			}
			info.SubAgents = append(info.SubAgents, sub.Name())
			walk(sub)
		}
	}
	walk(root)

	return agents
}

// toolsInfo describes the tools an LLM agent exposes to the model.
//
// Tools that are neither function-callable nor native Gemini tools are omitted:
// they shape the request (by injecting memory or examples into the prompt, for
// example) rather than offering the model anything to call.
func toolsInfo(ctx context.Context, appName, agentName string, state *llminternal.State) []*genai.Tool {
	tools := resolveTools(ctx, appName, agentName, state)

	infos := make([]*genai.Tool, 0, len(tools))
	for _, t := range tools {
		switch v := t.(type) {
		case declarer:
			if decl := v.Declaration(); decl != nil {
				infos = append(infos, &genai.Tool{
					FunctionDeclarations: []*genai.FunctionDeclaration{decl},
				})
			}
		case genaiToolProvider:
			if gt := v.GenaiTool(); gt != nil {
				infos = append(infos, gt)
			}
		}
	}
	return infos
}

// resolveTools returns an agent's static tools followed by the tools of each of
// its toolsets. A toolset that fails to resolve is logged and skipped: a
// description of the rest of the app is more useful than no description at all.
func resolveTools(ctx context.Context, appName, agentName string, state *llminternal.State) []tool.Tool {
	tools := slices.Clone(state.Tools)
	if len(state.Toolsets) == 0 {
		return tools
	}

	ctx, cancel := context.WithTimeout(ctx, toolsetTimeout)
	defer cancel()
	toolsetCtx := newStaticContext(ctx, appName, agentName)

	for _, ts := range state.Toolsets {
		if ts == nil {
			continue
		}
		tsTools, err := ts.Tools(toolsetCtx)
		if err != nil {
			log.Printf("agentinfo: agent %q: skipping toolset %q: %v", agentName, ts.Name(), err)
			continue
		}
		tools = append(tools, tsTools...)
	}
	return tools
}
