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

package main

import (
	"errors"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/configurable"
	"google.golang.org/adk/session"
)

func beforeAgentCallback1(ctx agent.CallbackContext) (*genai.Content, error) {
	ctx.State().Set("before_agent_callback_state_key", "value1")
	return nil, nil
}

func beforeAgentCallback2(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("before_agent_callback_state_key")
	if err != nil {
		return nil, err
	}
	ctx.State().Set("before_agent_callback_state_key", val.(string)+"+value2")
	return nil, nil
}

func shortcutAgentExecution(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("conversation_limit_reached")
	if err != nil {
		if !errors.Is(err, session.ErrStateKeyNotExist) {
			return nil, err
		}
		ctx.State().Set("conversation_limit_reached", "True")
		return nil, nil
	}
	if val.(string) == "True" {
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
	ctx.State().Set("after_agent_callback_state_key", "value1")
	return nil, nil
}

func afterAgentCallback2(ctx agent.CallbackContext) (*genai.Content, error) {
	val, err := ctx.State().Get("after_agent_callback_state_key")
	if err != nil {
		return nil, err
	}
	ctx.State().Set("after_agent_callback_state_key", val.(string)+"+value2")
	return nil, nil
}

func RegisterCallbacks() {
	configurable.RegisterCallback("callback_agent_001.callbacks.before_agent_callback1", agent.BeforeAgentCallback(beforeAgentCallback1))
	configurable.RegisterCallback("callback_agent_001.callbacks.before_agent_callback2", agent.BeforeAgentCallback(beforeAgentCallback2))
	configurable.RegisterCallback("callback_agent_002.callbacks.shortcut_agent_execution", agent.BeforeAgentCallback(shortcutAgentExecution))
	configurable.RegisterCallback("callback_agent_003.callbacks.after_agent_callback1", agent.AfterAgentCallback(afterAgentCallback1))
	configurable.RegisterCallback("callback_agent_003.callbacks.after_agent_callback2", agent.AfterAgentCallback(afterAgentCallback2))
}
