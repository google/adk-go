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
	"strings"

	"github.com/google/adk-go"

	"google.golang.org/genai"
)

func NewRunner(appName string, rootAgent adk.Agent, sessionService adk.SessionService) *Runner {
	// TODO: potentially validate the whole agent tree here.
	return &Runner{
		AppName:        appName,
		RootAgent:      rootAgent,
		SessionService: sessionService,
	}
}

type Runner struct {
	AppName        string
	RootAgent      adk.Agent
	SessionService adk.SessionService
}

// Run runs the agent.
func (r *Runner) RunAsync(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *adk.AgentRunConfig) iter.Seq2[*adk.Event, error] {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	// TODO: setup tracer.
	return func(yield func(*adk.Event, error) bool) {
		session, err := r.SessionService.Get(ctx, &adk.SessionGetRequest{
			AppName:   r.AppName,
			UserID:    userID,
			SessionID: sessionID,
		})
		if err != nil {
			yield(nil, err)
			return
		}

		ctx, ictx, err := r.newInvocationContext(ctx, session, cfg, msg)
		if err != nil {
			yield(nil, err)
			return
		}

		if err := r.appendMessageToSession(ctx, ictx, session, msg); err != nil {
			yield(nil, err)
			return
		}

		// TODO: run _find_agent_to_run(session, root_agent). For now just run root.

		for event, err := range r.RootAgent.Run(ctx, ictx) {
			if !yield(event, err) {
				return
			}
		}
	}
}

func (r *Runner) newInvocationContext(ctx context.Context, session *adk.Session, cfg *adk.AgentRunConfig, msg *genai.Content) (context.Context, *adk.InvocationContext, error) {
	if cfg != nil && cfg.SupportCFC {
		if err := r.setupCFC(r.RootAgent); err != nil {
			return nil, nil, fmt.Errorf("failed to setup CFC: %w", err)
		}
	}

	ctx, ictx := adk.NewInvocationContext(ctx, r.RootAgent, r.SessionService, session)
	return ctx, ictx, nil
}

func (r *Runner) setupCFC(rootAgent adk.Agent) error {
	spec := rootAgent.Spec()
	if spec.LLMAgent == nil {
		return fmt.Errorf("root agent %s has no LLMAgent spec", spec.Name)
	}

	if spec.LLMAgent.Model == nil {
		return fmt.Errorf("LLMAgent has no model")
	}

	if !strings.HasPrefix(spec.LLMAgent.Model.Name(), "gemini-2") {
		return fmt.Errorf("CFC is not supported for model: %v", spec.LLMAgent.Model.Name())
	}

	// TODO: handle CFC setup for LLMAgent, e.g. setting code_executor
	return nil
}

func (r *Runner) appendMessageToSession(ctx context.Context, ictx *adk.InvocationContext, session *adk.Session, msg *genai.Content) error {
	event := adk.NewEvent(ictx.InvocationID)

	event.Author = "user"
	event.LLMResponse = &adk.LLMResponse{
		Content: msg,
	}

	if err := r.SessionService.AppendEvent(ctx, session, event); err != nil {
		return fmt.Errorf("failed to append event to sessionService: %w", err)
	}
	return nil
}
