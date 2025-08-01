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
	internalRunner "google.golang.org/adk/internal/runner"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func NewRunner(appName string, rootAgent types.Agent, sessionService types.SessionService) *Runner {
	return &Runner{
		AppName:        appName,
		RootAgent:      rootAgent,
		SessionService: sessionService,
	}
}

type Runner struct {
	AppName        string
	RootAgent      types.Agent
	SessionService types.SessionService
}

// Run runs the agent.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *types.AgentRunConfig) iter.Seq2[*types.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*types.Event, error) bool) {
		session, err := r.SessionService.Get(ctx, &types.SessionGetRequest{
			AppName:   r.AppName,
			UserID:    userID,
			SessionID: sessionID,
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

		ctx, ictx, err := r.newInvocationContext(ctx, session, agentToRun, cfg, msg)
		if err != nil {
			yield(nil, err)
			return
		}

		if err := r.appendMessageToSession(ctx, ictx, session, msg); err != nil {
			yield(nil, err)
			return
		}

		for event, err := range internalRunner.RunAgent(ctx, ictx, agentToRun) {
			if !yield(event, err) {
				return
			}
		}
	}
}

func (r *Runner) newInvocationContext(ctx context.Context, session *types.Session, agent types.Agent, runConfig *types.AgentRunConfig, msg *genai.Content) (context.Context, *types.InvocationContext, error) {
	if runConfig != nil && runConfig.SupportCFC {
		if err := r.setupCFC(agent); err != nil {
			return nil, nil, fmt.Errorf("failed to setup CFC: %w", err)
		}
	}

	ctx, ictx := types.NewInvocationContext(ctx, agent, r.SessionService, session, runConfig, msg)
	return ctx, ictx, nil
}

func (r *Runner) setupCFC(curAgent types.Agent) error {
	llmAgent, ok := curAgent.(*agent.LLMAgent)
	if !ok {
		return fmt.Errorf("cannot setup cfc for non-LLMAgent")
	}

	if llmAgent.Model == nil {
		return fmt.Errorf("LLMAgent has no model")
	}

	if !strings.HasPrefix(llmAgent.Model.Name(), "gemini-2") {
		return fmt.Errorf("CFC is not supported for model: %v", llmAgent.Model.Name())
	}

	// TODO: handle CFC setup for LLMAgent, e.g. setting code_executor
	return nil
}

func (r *Runner) appendMessageToSession(ctx context.Context, ictx *types.InvocationContext, session *types.Session, msg *genai.Content) error {
	event := types.NewEvent(ictx.InvocationID)

	event.Author = "user"
	event.LLMResponse = &types.LLMResponse{
		Content: msg,
	}

	if err := r.SessionService.AppendEvent(ctx, session, event); err != nil {
		return fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return nil
}

// findAgentToRun returns the agent that should handle the next request based on
// session history.
func (r *Runner) findAgentToRun(session *types.Session) (types.Agent, error) {
	for i := len(session.Events) - 1; i >= 0; i-- {
		event := session.Events[i]

		if event.Author == "user" {
			continue
		}

		subAgent := findAgent(r.RootAgent, event.Author)
		// Agent not found, continue looking for the other event.
		if subAgent == nil {
			log.Printf("Event from an unknown agent: %s, event id: %s", event.Author, event.ID)
			continue
		}

		if isTransferrableAcrossAgentTree(subAgent) {
			return subAgent, nil
		}
	}

	// Falls back to root agent if no suitable agents are found in the session.
	return r.RootAgent, nil
}

// checks if the agent and its parent chain allow transfer up the tree.
func isTransferrableAcrossAgentTree(agentToRun types.Agent) bool {
	for curAgent := agentToRun; curAgent != nil; curAgent = curAgent.Spec().Parent() {
		// TODO: properly verify if agent is or embeds LLMAgent
		llmAgent, ok := agentToRun.(*agent.LLMAgent)
		if !ok {
			return false
		}

		if llmAgent.DisallowTransferToParent {
			return false
		}
	}

	return true
}

func findAgent(curAgent types.Agent, targetName string) types.Agent {
	if curAgent == nil || curAgent.Spec() == nil || curAgent.Spec().Name == targetName {
		return curAgent
	}

	for _, subAgent := range curAgent.Spec().SubAgents {
		if agent := findAgent(subAgent, targetName); agent != nil {
			return agent
		}
	}
	return nil
}
