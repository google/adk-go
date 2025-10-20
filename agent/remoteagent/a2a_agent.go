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

package remoteagent

import (
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"strings"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2aclient/agentcard"
	"google.golang.org/adk/adka2a"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/session"
)

type A2AConfig struct {
	Name        string
	Description string

	AgentCard          *a2a.AgentCard
	AgentCardSource    string
	CardResolveOptions []agentcard.ResolveOption

	ClientFactory     *a2aclient.Factory
	MessageSendConfig *a2a.MessageSendConfig
}

func New(cfg A2AConfig) (agent.Agent, error) {
	if cfg.AgentCard == nil && cfg.AgentCardSource == "" {
		return nil, fmt.Errorf("either AgentCard, or AgentCardPath, or AgentCardURL must be provided")
	}

	a2aAgent, err := agent.New(agent.Config{
		Name:        cfg.Name,
		Description: cfg.Description,
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return runRemoteA2A(ic, cfg)
		},
	})
	if err != nil {
		return nil, err
	}
	return a2aAgent, err
}

func runRemoteA2A(ctx agent.InvocationContext, cfg A2AConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		card, err := resolveAgentCard(ctx, cfg)
		if err != nil {
			yield(toErrorEvent(ctx, fmt.Errorf("agent card resolution failed: %w", err)), nil)
			return
		}

		var client *a2aclient.Client
		if cfg.ClientFactory != nil {
			client, err = cfg.ClientFactory.CreateFromCard(ctx, card)
		} else {
			client, err = a2aclient.NewFromCard(ctx, card)
		}
		if err != nil {
			yield(toErrorEvent(ctx, fmt.Errorf("client creation failed: %w", err)), nil)
			return
		}
		defer destroy(client)

		msg, err := newMessage(ctx)
		if err != nil {
			yield(toErrorEvent(ctx, fmt.Errorf("message creation failed: %w", err)), nil)
			return
		}

		if len(msg.Parts) == 0 {
			yield(adka2a.NewRemoteAgentEvent(ctx), nil)
			return
		}

		req := &a2a.MessageSendParams{Message: msg, Config: cfg.MessageSendConfig}
		for a2aEvent, err := range client.SendStreamingMessage(ctx, req) {
			if err != nil {
				event := toErrorEvent(ctx, err)
				updateCustomMetadata(event, req, nil)
				yield(event, nil)
				return
			}
			event, err := adka2a.ToSessionEvent(ctx, a2aEvent)
			if err != nil {
				event := toErrorEvent(ctx, fmt.Errorf("failed to convert a2aEvent: %w", err))
				updateCustomMetadata(event, req, nil)
				yield(event, nil)
				return
			}
			if event == nil {
				continue
			}
			updateCustomMetadata(event, req, a2aEvent)
			yield(event, nil)
		}
	}
}

func resolveAgentCard(ctx agent.InvocationContext, cfg A2AConfig) (*a2a.AgentCard, error) {
	if cfg.AgentCard != nil {
		return cfg.AgentCard, nil
	}

	if strings.HasPrefix(cfg.AgentCardSource, "http://") || strings.HasPrefix(cfg.AgentCardSource, "https://") {
		resolver := agentcard.Resolver{BaseURL: cfg.AgentCardSource}
		return resolver.Resolve(ctx, cfg.CardResolveOptions...)
	}

	fileBytes, err := os.ReadFile(cfg.AgentCardSource)
	if err != nil {
		return nil, err
	}

	var card *a2a.AgentCard
	if err := json.Unmarshal(fileBytes, card); err != nil {
		return nil, err
	}

	return card, nil
}

func newMessage(ctx agent.InvocationContext) (*a2a.Message, error) {
	events := ctx.Session().Events()
	if userFnCall := getUserFunctionCallAt(events, events.Len()-1); userFnCall != nil {
		msg, err := adka2a.EventToMessage(ctx, userFnCall.event)
		if err != nil {
			return nil, err
		}
		msg.TaskID = userFnCall.taskID
		msg.ContextID = userFnCall.contextID
		return msg, nil
	}

	parts, contextID := toMissingRemoteSessionParts(ctx, events)
	msg := a2a.NewMessage(a2a.MessageRoleUser, parts...)
	msg.ContextID = contextID
	return msg, nil
}

func toErrorEvent(ctx agent.InvocationContext, err error) *session.Event {
	event := adka2a.NewRemoteAgentEvent(ctx)
	event.ErrorMessage = err.Error()
	event.CustomMetadata = map[string]any{
		adka2a.ToADKMetaKey("error"): err.Error(),
	}
	return event
}

func updateCustomMetadata(event *session.Event, request *a2a.MessageSendParams, response a2a.Event) {
	if request == nil && response == nil {
		return
	}
	if event.CustomMetadata == nil {
		event.CustomMetadata = map[string]any{}
	}
	for k, v := range map[string]any{"request": request, "response": response} {
		if v == nil {
			continue
		}
		payload, err := utils.ToMapStructure(request)
		if err == nil {
			event.CustomMetadata[adka2a.ToADKMetaKey(k)] = payload
		} else {
			event.CustomMetadata[adka2a.ToADKMetaKey(k+"_codec_error")] = err.Error()
		}
	}
}

func destroy(client *a2aclient.Client) {
	// TODO(yarolegovich): log ignored error
	_ = client.Destroy()
}
