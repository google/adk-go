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

package services

import (
	"context"
	"iter"
	"log"
	"slices"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// toolsetResolveTimeout bounds the time spent resolving an agent's toolsets.
// Toolsets may reach out over the network (an MCP server, for example), and
// describing an app must not hang on one.
const toolsetResolveTimeout = 10 * time.Second

// declarer is implemented by tools the model calls as functions.
type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

// GetAppInfo describes an app without running it: its root agent, and every
// LLM agent reachable from that root with its instruction, tools and children.
//
// RootAgentName names the app's entry agent whatever its kind, so it is absent
// from Agents when the root is not an LLM agent.
func GetAppInfo(ctx context.Context, appName string, root agent.Agent) *models.AppInfo {
	if root == nil {
		return nil
	}
	return &models.AppInfo{
		Name:          appName,
		RootAgentName: root.Name(),
		Description:   root.Description(),
		Language:      models.LanguageGo,
		Agents:        collectAgents(ctx, appName, root),
	}
}

// collectAgents walks the agent tree rooted at root, returning one
// [models.AgentInfo] per LLM agent, keyed by agent name.
//
// Only LLM agents are reported, per the wire contract, but the walk does not
// stop at an agent that is not one: a SequentialAgent between two LLM agents is
// stepped over rather than cutting off everything below it. So the names in
// [models.AgentInfo.SubAgents] are the nearest LLM agents below that parent,
// and every one of them is a key of the returned map.
func collectAgents(ctx context.Context, appName string, root agent.Agent) map[string]*models.AgentInfo {
	agents := make(map[string]*models.AgentInfo)
	// resolved caches what each agent contributes to its parent's sub-agent
	// list, so an agent reachable from two parents is described once and still
	// linked from both.
	resolved := make(map[string][]string)
	// inProgress holds the agents on the current path, so a tree that is not
	// one -- an agent that is its own ancestor -- terminates.
	inProgress := make(map[string]bool)

	// nearestLLMAgents describes the nearest LLM agents at or below a, adding
	// each to agents, and returns their names.
	var nearestLLMAgents func(a agent.Agent) []string
	nearestLLMAgents = func(a agent.Agent) []string {
		if a == nil {
			return nil
		}
		name := a.Name()
		if names, done := resolved[name]; done {
			return names
		}
		if inProgress[name] {
			return nil
		}
		inProgress[name] = true
		defer delete(inProgress, name)

		llmAgent, isLLM := a.(llminternal.Agent)
		if !isLLM {
			// Not reported itself. The LLM agents below it stand in for it, so
			// its parent links to them directly.
			var nested []string
			for _, sub := range a.SubAgents() {
				nested = append(nested, nearestLLMAgents(sub)...)
			}
			resolved[name] = dedupe(nested)
			return resolved[name]
		}

		state := llminternal.Reveal(llmAgent)
		info := &models.AgentInfo{
			Name:        name,
			Description: a.Description(),
			// An agent whose instruction comes from an InstructionProvider
			// reports an empty instruction: resolving it needs session state
			// that does not exist outside of an invocation.
			Instruction: state.Instruction,
			Tools:       agentTools(ctx, appName, name, state),
		}
		agents[name] = info
		// Recorded before recursing, so a descendant that reaches back here
		// links to this agent rather than to nothing.
		resolved[name] = []string{name}

		var subAgents []string
		for _, sub := range a.SubAgents() {
			subAgents = append(subAgents, nearestLLMAgents(sub)...)
		}
		info.SubAgents = dedupe(subAgents)

		return resolved[name]
	}
	nearestLLMAgents(root)

	return agents
}

// dedupe removes repeated names, keeping the first of each and the order they
// arrived in. Two children can lead to the same LLM agent, and a parent should
// list it once.
func dedupe(names []string) []string {
	if len(names) < 2 {
		return names
	}
	seen := make(map[string]bool, len(names))
	out := make([]string, 0, len(names))
	for _, n := range names {
		if seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	return out
}

// agentTools describes the tools an LLM agent exposes to the model, as function
// declarations.
//
// Tools without a declaration are omitted.
func agentTools(ctx context.Context, appName, agentName string, state *llminternal.State) []*genai.Tool {
	tools := resolveTools(ctx, appName, agentName, state)

	infos := make([]*genai.Tool, 0, len(tools))
	for _, t := range tools {
		d, ok := t.(declarer)
		if !ok {
			continue
		}
		if decl := d.Declaration(); decl != nil {
			infos = append(infos, &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{decl},
			})
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

	ctx, cancel := context.WithTimeout(ctx, toolsetResolveTimeout)
	defer cancel()
	toolsetCtx := appInfoContext{Context: ctx, appName: appName, agentName: agentName}

	for _, ts := range state.Toolsets {
		if ts == nil {
			continue
		}
		tsTools, err := ts.Tools(toolsetCtx)
		if err != nil {
			log.Printf("app-info: agent %q: skipping toolset %q: %v", agentName, ts.Name(), err)
			continue
		}
		tools = append(tools, tsTools...)
	}
	return tools
}

// appInfoContext is a minimal [agent.ReadonlyContext] for resolving toolsets
// outside of an invocation. Describing an app runs no agent, so there is no
// user content, no session and no invocation to expose.
type appInfoContext struct {
	context.Context

	appName   string
	agentName string
}

func (c appInfoContext) AppName() string   { return c.appName }
func (c appInfoContext) AgentName() string { return c.agentName }

func (appInfoContext) UserContent() *genai.Content { return nil }
func (appInfoContext) InvocationID() string        { return "" }
func (appInfoContext) UserID() string              { return "" }
func (appInfoContext) SessionID() string           { return "" }
func (appInfoContext) Branch() string              { return "" }

func (appInfoContext) ReadonlyState() session.ReadonlyState { return emptyState{} }

// emptyState is a [session.ReadonlyState] that holds nothing.
type emptyState struct{}

func (emptyState) Get(string) (any, error) { return nil, session.ErrStateKeyNotExist }

func (emptyState) All() iter.Seq2[string, any] {
	return func(func(string, any) bool) {}
}
