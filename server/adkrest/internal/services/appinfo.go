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
	"fmt"
	"iter"
	"log"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// toolsetResolveBudget bounds the time one request spends resolving toolsets,
// across every agent in the app rather than per agent. Toolsets may reach out
// over the network (an MCP server, for example), and describing an app must not
// hang on one.
//
// A budget per agent would multiply by the number of agents holding a toolset,
// which outlives the server's own write timeout: the client would see a broken
// response while the handler kept opening outbound connections. Once this is
// spent, the remaining Tools calls return the context error at once and are
// logged and skipped like any other failing toolset.
const toolsetResolveBudget = 10 * time.Second

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
	// One budget for the whole walk, not one per agent.
	ctx, cancel := context.WithTimeout(ctx, toolsetResolveBudget)
	defer cancel()

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
//
// An agent's children are its sub-agents plus the agents its workflow graph
// runs, since a workflow agent holds those in its edges rather than in
// SubAgents.
//
// An agent reachable only as an agent tool is deliberately left out. Such an
// agent runs under its own runner and session inside the tool call, so none of
// its events reach the stream and no event is ever attributed to it. Evaluation
// keys an agent by the author of the events it produced, so an agent that
// authors none has nothing to match. It is still reported as a function
// declaration in its caller's Tools.
func collectAgents(ctx context.Context, appName string, root agent.Agent) map[string]*models.AgentInfo {
	agents := make(map[string]*models.AgentInfo)
	// resolved caches what each agent contributes to its parent's sub-agent
	// list, so an agent reachable from two parents is described once and still
	// linked from both.
	resolved := make(map[string][]string)
	// inProgress holds the agents on the current path, so a tree that is not
	// one -- an agent that is its own ancestor -- terminates.
	inProgress := make(map[string]bool)
	// truncations counts how often the walk turned back at an agent already on
	// the path. It only ever moves in a graph that loops.
	truncations := 0

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
			truncations++
			return nil
		}
		inProgress[name] = true
		defer delete(inProgress, name)

		llmAgent, isLLM := a.(llminternal.Agent)
		if !isLLM {
			// Not reported itself. The LLM agents below it stand in for it, so
			// its parent links to them directly.
			before := truncations
			var nested []string
			for _, sub := range children(a) {
				nested = append(nested, nearestLLMAgents(sub)...)
			}
			out := dedupe(nested)
			// Cache only a result the walk reached the bottom of. A loop makes
			// this agent's answer depend on where the walk entered it, and
			// caching the truncated one would hand it to a later parent that
			// could have seen the whole subtree. An LLM agent is immune: it
			// contributes its own name whatever the path.
			if truncations == before {
				resolved[name] = out
			}
			return out
		}

		state := llminternal.Reveal(llmAgent)
		tools := resolveTools(ctx, appName, name, state)
		info := &models.AgentInfo{
			Name:        name,
			Description: a.Description(),
			Instruction: instruction(state),
			Tools:       toolDeclarations(tools),
		}
		agents[name] = info
		// Recorded before recursing, so a descendant that reaches back here
		// links to this agent rather than to nothing.
		resolved[name] = []string{name}

		// Starts non-nil: an agent with no LLM agent below it must report []
		// and not null, which the contract forbids.
		subAgents := []string{}
		for _, sub := range children(a) {
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

// children returns the agents a walk should descend into from a: its
// sub-agents, followed by the agents its workflow graph runs.
//
// A workflow agent keeps the agents of its graph in its edges, not in
// SubAgents, so following SubAgents alone reports nothing for a graph-rooted
// app.
func children(a agent.Agent) []agent.Agent {
	subAgents := a.SubAgents()
	edges := workflowEdges(a)
	if len(edges) == 0 {
		return subAgents
	}

	out := slices.Clone(subAgents)
	seen := make(map[workflow.Node]bool)
	var walkEdges func(edges []workflow.Edge)
	walkEdges = func(edges []workflow.Edge) {
		for _, e := range edges {
			for _, n := range []workflow.Node{e.From, e.To} {
				if n == nil || seen[n] {
					continue
				}
				seen[n] = true
				switch node := n.(type) {
				case *workflow.AgentNode:
					out = append(out, node.Agent())
				case *workflow.WorkflowNode:
					walkEdges(node.Workflow().Edges())
				}
			}
		}
	}
	walkEdges(edges)
	return out
}

// workflowEdges returns the edges of a's workflow graph, or nil when a is not
// a workflow agent.
func workflowEdges(a agent.Agent) []workflow.Edge {
	internalAgent, ok := a.(agentinternal.Agent)
	if !ok {
		return nil
	}
	cfg, ok := agentinternal.Reveal(internalAgent).Config.(workflowagent.Config)
	if !ok {
		return nil
	}
	return cfg.Edges
}

// instruction is the agent's system instruction as app-info reports it.
//
// An instruction that comes from a provider is named rather than resolved:
// resolving it needs session state that does not exist outside an invocation.
// Reporting it empty would read as an agent with no instruction at all.
// adk-python reports the same placeholder.
func instruction(state *llminternal.State) string {
	if state.Instruction != "" || state.InstructionProvider == nil {
		return state.Instruction
	}
	return fmt.Sprintf("<InstructionProvider: %s>", providerName(state.InstructionProvider))
}

// providerName is the declared name of an instruction provider, trimmed of its
// package path.
func providerName(p llminternal.InstructionProvider) string {
	fn := runtime.FuncForPC(reflect.ValueOf(p).Pointer())
	if fn == nil {
		return "unknown"
	}
	name := fn.Name()
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	return name
}

// toolDeclarations describes the tools an LLM agent exposes to the model, as
// function declarations. Tools without a declaration are omitted.
func toolDeclarations(tools []tool.Tool) []*genai.Tool {
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
//
// ctx already carries the request's toolset budget, shared with every other
// agent in the walk.
func resolveTools(ctx context.Context, appName, agentName string, state *llminternal.State) []tool.Tool {
	tools := slices.Clone(state.Tools)
	if len(state.Toolsets) == 0 {
		return tools
	}

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
