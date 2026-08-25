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
	llmagentinternal "google.golang.org/adk/v2/internal/llminternal"
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

// genaiToolProvider is implemented by tools with a native [genai.Tool]
// representation, such as Google Search, which the model executes itself rather
// than calling as a function.
type genaiToolProvider interface {
	GenaiTool() *genai.Tool
}

// GetAppInfo describes an app without running it: its root agent, and every
// agent reachable from that root with its instruction, tools and children.
//
// Every kind of agent is included. Agents that are not LLM agents report an
// empty instruction and no tools, because they have neither; their children are
// still walked, so the result always describes the whole tree.
//
// An agent reachable by more than one path is reported once, which also stops a
// cyclic tree from recursing forever.
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
// [models.AgentInfo] per agent, keyed by agent name.
func collectAgents(ctx context.Context, appName string, root agent.Agent) map[string]*models.AgentInfo {
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

		if llmAgent, ok := a.(llmagentinternal.Agent); ok {
			state := llmagentinternal.Reveal(llmAgent)
			// An agent whose instruction comes from an InstructionProvider
			// reports an empty instruction: resolving it needs session state
			// that does not exist outside of an invocation.
			info.Instruction = state.Instruction
			info.Tools = agentTools(ctx, appName, a.Name(), state)
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

// agentTools describes the tools an LLM agent exposes to the model.
//
// Tools that are neither function-callable nor native Gemini tools are omitted:
// they shape the request (by injecting memory or examples into the prompt, for
// example) rather than offering the model anything to call.
func agentTools(ctx context.Context, appName, agentName string, state *llmagentinternal.State) []*genai.Tool {
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
func resolveTools(ctx context.Context, appName, agentName string, state *llmagentinternal.State) []tool.Tool {
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
// user content, no session and no invocation to expose; a toolset that consults
// any of them sees empty values rather than a nil dereference.
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
