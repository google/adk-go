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

package conformance

import (
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable"
	"google.golang.org/adk/session"
)

func beforeAgentCallback1(ctx agent.CallbackContext) (*genai.Content, error) {
	err := ctx.State().Set("before_agent_callback_state_key", "value1")
	return nil, err
}

func beforeAgentCallback2(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("before_agent_callback_state_key")
	if err != nil {
		return nil, err
	}
	s, ok := val.(string)
	if !ok {
		return nil, fmt.Errorf("state value for 'before_agent_callback_state_key' is not a string, but %T", val)
	}
	err = ctx.State().Set("before_agent_callback_state_key", s+"+value2")
	return nil, err
}

func shortcutAgentExecution(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("conversation_limit_reached")
	if err != nil {
		if !errors.Is(err, session.ErrStateKeyNotExist) {
			return nil, err
		}
		err = ctx.State().Set("conversation_limit_reached", true)
		return nil, err
	}
	if limitReached, ok := val.(bool); ok && limitReached {
		return &genai.Content{
			Parts: []*genai.Part{
				{Text: "Sorry, you have reached the limit of the conversation."},
			},
			Role: "model",
		}, nil
	}
	return nil, nil
}

func afterAgentCallback1(ctx agent.CallbackContext) (*genai.Content, error) {
	err := ctx.State().Set("after_agent_callback_state_key", "value1")
	return nil, err
}

func afterAgentCallback2(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("after_agent_callback_state_key")
	if err != nil {
		return nil, err
	}
	s, ok := val.(string)
	if !ok {
		return nil, fmt.Errorf("state value for 'after_agent_callback_state_key' is not a string, but %T", val)
	}
	err = ctx.State().Set("after_agent_callback_state_key", s+"+value2")
	return nil, err
}

func RegisterCallbacks() error {
	err := configurable.RegisterCallback("callback_agent_001.callbacks.before_agent_callback1", agent.BeforeAgentCallback(beforeAgentCallback1))
	if err != nil {
		return fmt.Errorf("error registering before agent callback 1: %w", err)
	}
	err = configurable.RegisterCallback("callback_agent_001.callbacks.before_agent_callback2", agent.BeforeAgentCallback(beforeAgentCallback2))
	if err != nil {
		return fmt.Errorf("error registering before agent callback 2: %w", err)
	}
	err = configurable.RegisterCallback("callback_agent_002.callbacks.shortcut_agent_execution", agent.BeforeAgentCallback(shortcutAgentExecution))
	if err != nil {
		return fmt.Errorf("error registering shortcut agent execution: %w", err)
	}
	err = configurable.RegisterCallback("callback_agent_003.callbacks.after_agent_callback1", agent.AfterAgentCallback(afterAgentCallback1))
	if err != nil {
		return fmt.Errorf("error registering after agent callback 1: %w", err)
	}
	err = configurable.RegisterCallback("callback_agent_003.callbacks.after_agent_callback2", agent.AfterAgentCallback(afterAgentCallback2))
	if err != nil {
		return fmt.Errorf("error registering after agent callback 2: %w", err)
	}
	return nil
}
