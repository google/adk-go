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
	"iter"

	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

// RunAgent is called by adk internally wrapping extra logic on top of agent's Run.
func RunAgent(ctx context.Context, ictx *types.InvocationContext, agent types.Agent) iter.Seq2[*types.Event, error] {
	callbackContext := &types.CallbackContext{
		InvocationContext: ictx,
	}

	return func(yield func(*types.Event, error) bool) {
		if event := runBeforeAgentCallbacks(ctx, callbackContext, agent); event != nil {
			yield(event, nil)
			return
		}

		for event, err := range agent.Run(ctx, ictx) {
			if event.Author == "" {
				event.Author = getAuthorForEvent(ictx, event)
			}

			event = runAfterAgentCallbacks(ctx, callbackContext, event, agent)

			if !yield(event, err) {
				return
			}
		}
	}
}

// runBeforeAgentCallbacks checks if any beforeAgentCallback returns non-nil content
// then it skips agent run and returns callback result.
func runBeforeAgentCallbacks(ctx context.Context, callbackContext *types.CallbackContext, agent types.Agent) *types.Event {
	for _, callback := range agent.Spec().BeforeAgentCallbacks {
		content := callback(ctx, callbackContext)
		if content == nil {
			continue
		}

		event := types.NewEvent(callbackContext.InvocationContext.InvocationID)
		event.LLMResponse = &types.LLMResponse{
			Content: content,
		}
		event.Author = agent.Spec().Name
		event.Branch = callbackContext.InvocationContext.Branch
		event.Actions = callbackContext.EventActions

		// TODO: set ictx.end_invocation

		return event
	}

	return nil
}

// runAfterAgentCallbacks checks if any afterAgentCallback returns non-nil content
// then it replaces the event content with a value from the callback.
func runAfterAgentCallbacks(ctx context.Context, callbackContext *types.CallbackContext, event *types.Event, agent types.Agent) *types.Event {
	if event == nil {
		return event
	}

	for _, callback := range agent.Spec().AfterAgentCallbacks {
		newContent := callback(ctx, callbackContext, event.LLMResponse.Content)
		if newContent == nil {
			continue
		}

		event.LLMResponse.Content = newContent
		return event
	}

	return event
}

func getAuthorForEvent(ictx *types.InvocationContext, event *types.Event) string {
	if event.LLMResponse != nil && event.LLMResponse.Content != nil && event.LLMResponse.Content.Role == genai.RoleUser {
		return genai.RoleUser
	}

	return ictx.Agent.Spec().Name
}
