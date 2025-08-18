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

package runner

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func New(appName string, rootAgent agent.Agent, sessionService sessionservice.Service) (*Runner, error) {
	parents, err := parentmap.New(rootAgent)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent tree: %w", err)
	}

	return &Runner{
		AppName:        appName,
		RootAgent:      rootAgent,
		SessionService: sessionService,
		parents:        parents,
	}, nil
}

type Runner struct {
	AppName        string
	RootAgent      agent.Agent
	SessionService sessionservice.Service

	parents parentmap.Map
}

// Run runs the agent.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *types.AgentRunConfig) iter.Seq2[*session.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*session.Event, error) bool) {
		session, err := r.SessionService.Get(ctx, &sessionservice.GetRequest{
			ID: session.ID{
				AppName:   r.AppName,
				UserID:    userID,
				SessionID: sessionID,
			},
		})
		if err != nil {
			yield(nil, err)
			return
		}

		agentToRun, err := r.findAgentToRun(session)
		if err != nil {
			yield(nil, err)
			return
		}

		if cfg != nil && cfg.SupportCFC {
			if err := r.setupCFC(agentToRun); err != nil {
				yield(nil, fmt.Errorf("failed to setup CFC: %w", err))
				return
			}
		}

		ctx = parentmap.ToContext(ctx, r.parents)

		ctx := agent.NewContext(ctx, agentToRun, msg, &mutableSession{
			service:       r.SessionService,
			storedSession: session,
		}, "")

		if err := r.appendMessageToSession(ctx, session, msg); err != nil {
			yield(nil, err)
			return
		}

		for event, err := range agentToRun.Run(ctx) {
			if err != nil {
				if !yield(event, err) {
					return
				}
				continue
			}

			// only commit non-partial event to a session service
			if !(event.LLMResponse != nil && event.LLMResponse.Partial) {

				// TODO: update session state & delta

				if err := r.SessionService.AppendEvent(ctx, session, event); err != nil {
					yield(nil, fmt.Errorf("failed to add event to session: %w", err))
					return
				}
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}

func (r *Runner) setupCFC(curAgent agent.Agent) error {
	llmAgent, ok := curAgent.(llminternal.Agent)
	if !ok {
		return fmt.Errorf("agent %v is not an LLMAgent", curAgent.Name())
	}

	model := llminternal.Reveal(llmAgent).Model

	if model == nil {
		return fmt.Errorf("LLMAgent has no model")
	}

	if !strings.HasPrefix(model.Name(), "gemini-2") {
		return fmt.Errorf("CFC is not supported for model: %v", model.Name())
	}

	// TODO: handle CFC setup for LLMAgent, e.g. setting code_executor
	return nil
}

func (r *Runner) appendMessageToSession(ctx agent.Context, storedSession sessionservice.StoredSession, msg *genai.Content) error {
	if msg == nil {
		return nil
	}
	event := session.NewEvent(ctx.InvocationID())

	event.Author = "user"
	event.LLMResponse = &llm.Response{
		Content: msg,
	}

	if err := r.SessionService.AppendEvent(ctx, storedSession, event); err != nil {
		return fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return nil
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session sessionservice.StoredSession) (agent.Agent, error) {
	events := session.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)

		// TODO: findMatchingFunctionCall.

		if event.Author == "user" {
			continue
		}

		subAgent := findAgent(r.RootAgent, event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID)
			continue
		}

		if r.isTransferableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.RootAgent, nil
}

// checks if the agent and its parent chain allow transfer up the tree.
func (r *Runner) isTransferableAcrossAgentTree(agentToRun agent.Agent) bool {
	for curAgent := agentToRun; curAgent != nil; curAgent = r.parents[curAgent.Name()] {
		llmAgent, ok := agentToRun.(llminternal.Agent)
		if !ok {
			return false
		}

		if llminternal.Reveal(llmAgent).DisallowTransferToParent {
			return false
		}
	}

	return true
}

func findAgent(curAgent agent.Agent, targetName string) agent.Agent {
	if curAgent == nil || curAgent.Name() == targetName {
		return curAgent
	}

	for _, subAgent := range curAgent.SubAgents() {
		if agent := findAgent(subAgent, targetName); agent != nil {
			return agent
		}
	}
	return nil
}
