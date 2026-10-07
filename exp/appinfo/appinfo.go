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

// Package appinfo implements GET /apps/{app_name}/app-info, which describes an
// ADK app without running it: its root agent, and every LLM agent reachable
// from that root with its description, instruction, tools and sub-agents.
//
// The endpoint is experimental. Its response may change, or the endpoint may be
// removed, in a later version without notice, and [Handler] logs a warning
// saying so the first time it serves a request.
//
// It is served only when asked for, since it hands out every agent's
// instruction and tool declarations. [NewLauncher] returns the web sublauncher
// that serves it. Pass it to web.NewLauncher before the api sublauncher, and
// name it on the command line:
//
//	l := universal.NewLauncher(
//		console.NewLauncher(),
//		web.NewLauncher(webui.NewLauncher(), appinfo.NewLauncher(), api.NewLauncher()),
//	)
//
//	go run . web api -path_prefix / appinfo
//
// A server built some other way can mount [Handler] itself.
//
// # What is reported
//
// Only LLM agents are described, but the walk passes through agents of every
// kind to find them: the sub-agents of a SequentialAgent, and the agents a
// workflow graph runs, nested sub-workflows and ParallelWorkers included. Some
// agents are not found:
//
//   - An agent wrapped in an agent tool. It runs under its own runner inside the
//     tool call, so none of its events reach the event stream. It is reported
//     as a tool of the LLM agent that has the tool, and not at all when a
//     workflow ToolNode runs the tool.
//   - An agent that a dynamic workflow node runs. Its body is Go code, so which
//     agents it runs is known only once it runs.
//
// Tools are reported as function declarations. Every request resolves each
// agent's toolsets, which for an MCP toolset can start a session with its
// server. They are resolved outside any invocation, with no user, session or
// state, so a toolset that picks its tools per user reports the tools it offers
// such an anonymous caller.
package appinfo

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"slices"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/workflowwalk"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
)

// AppInfo describes an ADK app and the LLM agents it contains. It is the
// response body of GET /apps/{app_name}/app-info.
//
// Agents is flat, keyed by agent name, and holds only LLM agents. So
// RootAgentName is not one of its keys when the root agent is of another kind.
//
// Agents carries no omitempty. Evaluation reads the key on every response, so
// an app with no LLM agent at all reports an empty object rather than nothing.
type AppInfo struct {
	Name          string                `json:"name"`
	RootAgentName string                `json:"rootAgentName"`
	Description   string                `json:"description"`
	Language      string                `json:"language"`
	Agents        map[string]*AgentInfo `json:"agents"`
}

// AgentInfo describes one LLM agent within an app.
//
// Instruction is the agent's instruction as written, placeholders included.
// For an agent whose instruction comes from an InstructionProvider it is
// "<InstructionProvider>", since resolving the provider needs an invocation.
//
// SubAgents names the agent's own sub-agents that are LLM agents. A sub-agent
// of another kind is not listed, and neither are the LLM agents below it: they
// are described in [AppInfo.Agents], but this agent hands off to the agent in
// between rather than to them.
//
// Tools and SubAgents are present on every agent, as [] when there is nothing
// to report, so a client can read them without checking that the key exists.
type AgentInfo struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Instruction string        `json:"instruction"`
	Tools       []*genai.Tool `json:"tools"`
	SubAgents   []string      `json:"subAgents"`
}

// toolsetResolveBudget is the deadline one request gives toolset resolution,
// across every agent in the app rather than per agent. Toolsets may reach out
// over the network (an MCP server, for example), and describing an app should
// not wait on one indefinitely.
//
// A budget per agent would multiply by the number of agents holding a toolset,
// which outlives the server's own write timeout: the client would see a broken
// response while the handler kept opening outbound connections. The deadline
// binds only toolsets that honor their context. Once it passes, those return
// the context error at once and are logged and skipped like any other failing
// toolset.
const toolsetResolveBudget = 10 * time.Second

const languageGo string = "go"

// declarer is implemented by tools the model calls as functions.
type declarer interface {
	Declaration() *genai.FunctionDeclaration
}

// build describes the app called appName whose root agent is root.
func build(ctx context.Context, appName string, root agent.Agent) *AppInfo {
	// One budget for the whole walk, not one per agent.
	ctx, cancel := context.WithTimeout(ctx, toolsetResolveBudget)
	defer cancel()

	return &AppInfo{
		Name:          appName,
		RootAgentName: root.Name(),
		Description:   root.Description(),
		Language:      languageGo,
		Agents:        collectAgents(ctx, appName, root),
	}
}

// collectAgents walks every agent reachable from root and describes each LLM
// agent, keyed by name.
//
// The walk descends into agents of every kind, so a SequentialAgent between two
// LLM agents does not hide the one below it. An agent's children are its
// sub-agents followed by the agents its workflow graph runs.
//
// Agents are told apart by identity rather than by name, so an agent reachable
// along two paths is described once and a loop terminates. Two different
// agents sharing a name cannot both be keys of the map: the first one found is
// described, the clash is logged, and the walk still descends into the other.
func collectAgents(ctx context.Context, appName string, root agent.Agent) map[string]*AgentInfo {
	agents := make(map[string]*AgentInfo)
	visited := make(map[any]bool)

	var walk func(a agent.Agent)
	walk = func(a agent.Agent) {
		if a == nil {
			return
		}
		id := identity(a)
		if visited[id] {
			return
		}
		visited[id] = true

		if llmAgent, ok := a.(llminternal.Agent); ok {
			name := a.Name()
			if _, clash := agents[name]; clash {
				log.Printf("app-info: two different agents are named %q; describing the first one found", name)
			} else {
				agents[name] = describe(ctx, appName, a, llminternal.Reveal(llmAgent))
			}
		}
		for _, child := range children(a) {
			walk(child)
		}
	}
	walk(root)

	return agents
}

// identity is the key the walk tells agents apart by: the agent's internal
// state, which every agent the ADK constructors build holds and no two share.
// An agent built some other way, such as a struct embedding agent.Agent, is
// keyed by its name instead. The agent itself cannot be the key, since a map
// key of a type that is not comparable panics.
func identity(a agent.Agent) any {
	if internalAgent, ok := a.(agentinternal.Agent); ok {
		return agentinternal.Reveal(internalAgent)
	}
	return a.Name()
}

// describe reports the LLM agent a, whose internal state is state.
func describe(ctx context.Context, appName string, a agent.Agent, state *llminternal.State) *AgentInfo {
	return &AgentInfo{
		Name:        a.Name(),
		Description: a.Description(),
		Instruction: instruction(state),
		Tools:       toolDeclarations(resolveTools(ctx, appName, a.Name(), state)),
		SubAgents:   llmSubAgents(a),
	}
}

// llmSubAgents names the sub-agents of a that are LLM agents, in order. It is
// never nil, so an agent with none reports [] rather than null.
func llmSubAgents(a agent.Agent) []string {
	names := []string{}
	for _, sub := range a.SubAgents() {
		if _, ok := sub.(llminternal.Agent); ok {
			names = append(names, sub.Name())
		}
	}
	return names
}

// children returns the agents a walk should descend into from a: its
// sub-agents, followed by the agents its workflow graph runs.
//
// A workflow agent keeps the agents of its graph in its edges, not in
// SubAgents, so following SubAgents alone reports nothing for a graph-rooted
// app.
func children(a agent.Agent) []agent.Agent {
	// The graph's agents come in no particular order, so if two different
	// agents share a name and one is in a nested workflow, which one is
	// described can change from request to request. Sort them here if that
	// matters.
	return slices.Concat(a.SubAgents(), workflowwalk.WalkAgents(workflowEdges(a)))
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

// providerInstruction is what app-info reports for an instruction that comes
// from an InstructionProvider.
const providerInstruction = "<InstructionProvider>"

// instruction is the agent's instruction as app-info reports it.
//
// A provider takes over from Instruction when both are set, as it does when the
// agent runs, so it is checked first. It is not resolved, since resolving it
// needs session state that exists only during an invocation.
func instruction(state *llminternal.State) string {
	if state.InstructionProvider == nil {
		return state.Instruction
	}
	return providerInstruction
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
			log.Printf("app-info: agent %q: skipping toolset %q, which could not list its tools (%s)", agentName, ts.Name(), errorShape(err))
			continue
		}
		tools = append(tools, tsTools...)
	}
	return tools
}

// errorShape describes err for a log line without its text, which can carry a
// server URL with a token in its query, a credential or a response body.
func errorShape(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "timed out"
	case errors.Is(err, context.Canceled):
		return "request canceled"
	default:
		return fmt.Sprintf("%T", err)
	}
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
