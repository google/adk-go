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

// Package remoteagent allows to use a remote agent via A2A protocol.
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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

// A2AConfig is used to describe and configure a remote agent.
type A2AConfig struct {
	Name        string
	Description string

	// AgentCardSource can be either an http(s) URL or a local file path. If a2a.AgentCard
	// is not provided, the source is used to resolve the card during the first agent invocation.
	AgentCard       *a2a.AgentCard
	AgentCardSource string
	// CardResolveOptions can be used to provide a set of agencard.Resolver configurations.
	CardResolveOptions []agentcard.ResolveOption

	// ClientFactory can be used to provide a set of a2aclient.Client configurations.
	ClientFactory *a2aclient.Factory
	// MessageSendConfig is attached to a2a.MessageSendParams sent on every agent invocation.
	MessageSendConfig *a2a.MessageSendConfig
}

// NewA2A creates a remote A2A agent. A2A (Agent-To-Agent) protocol is used for communication with an
// agent which can run in a different process or on a different host.
func NewA2A(cfg A2AConfig) (agent.Agent, error) {
	if cfg.AgentCard == nil && cfg.AgentCardSource == "" {
		return nil, fmt.Errorf("either AgentCard or AgentCardSource must be provided")
	}

	remoteAgent := &a2aAgent{resolvedCard: cfg.AgentCard}
	return agent.New(agent.Config{
		Name:        cfg.Name,
		Description: cfg.Description,
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return remoteAgent.run(ic, cfg)
		},
	})
}

type a2aAgent struct {
	resolvedCard *a2a.AgentCard
}

func (a *a2aAgent) run(ctx agent.InvocationContext, cfg A2AConfig) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		card, err := resolveAgentCard(ctx, cfg)
		if err != nil {
			yield(toErrorEvent(ctx, fmt.Errorf("agent card resolution failed: %w", err)), nil)
			return
		}
		a.resolvedCard = card

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
			if !yield(event, nil) {
				break
			}
		}
	}
}

func resolveAgentCard(ctx agent.InvocationContext, cfg A2AConfig) (*a2a.AgentCard, error) {
	if cfg.AgentCard != nil {
		return cfg.AgentCard, nil
	}

	if strings.HasPrefix(cfg.AgentCardSource, "http://") || strings.HasPrefix(cfg.AgentCardSource, "https://") {
		card, err := agentcard.DefaultResolver.Resolve(ctx, cfg.AgentCardSource, cfg.CardResolveOptions...)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch an agent card: %w", err)
		}
		return card, nil
	}

	fileBytes, err := os.ReadFile(cfg.AgentCardSource)
	if err != nil {
		return nil, fmt.Errorf("failed to read agent card from %q: %w", cfg.AgentCardSource, err)
	}

	var card *a2a.AgentCard
	if err := json.Unmarshal(fileBytes, card); err != nil {
		return nil, fmt.Errorf("failed to unmarshal an agent card: %w", err)
	}

	return card, nil
}

func newMessage(ctx agent.InvocationContext) (*a2a.Message, error) {
	events := ctx.Session().Events()
	if userFnCall := getUserFunctionCallAt(events, events.Len()-1); userFnCall != nil {
		msg, err := adka2a.EventToMessage(userFnCall.event)
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
		payload, err := converters.ToMapStructure(request)
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
