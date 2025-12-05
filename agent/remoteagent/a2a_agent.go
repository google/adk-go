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

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

// BeforeA2ARequestCallback is called before sending a request to the remote agent.
//
// If it returns non-nil result or error, the actual call is skipped and the returned value is used
// as the agent invocation result.
type BeforeA2ARequestCallback func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error)

// A2AEventConverter can be used to provide a custom implementation of A2A event transformation logic.
type A2AEventConverter func(ctx agent.ReadonlyContext, req *a2a.MessageSendParams, event a2a.Event, err error) (*session.Event, error)

// AfterA2ARequestCallback is called after receiving a response from the remote agent and converting it to a session.Event.
// In streaming responses the callback is invoked for every request. Session event parameter might be nil if conversion logic
// decides to not emit an A2A event.
//
// If it returns non-nil result or error, it gets emitted instead of the original result.
type AfterA2ARequestCallback func(ctx agent.CallbackContext, req *a2a.MessageSendParams, resp *session.Event, err error) (*session.Event, error)

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

	// BeforeAgentCallbacks is a list of callbacks that are called sequentially
	// before the agent starts its run.
	//
	// If any callback returns non-nil content or error, then the agent run and
	// the remaining callbacks will be skipped, and a new event will be created
	// from the content or error of that callback.
	BeforeAgentCallbacks []agent.BeforeAgentCallback
	// BeforeRequestCallbacks will be called in the order they are provided until
	// there's a callback that returns a non-nil result or error. Then the
	// actual request is skipped, and the returned response/error is used.
	//
	// This provides an opportunity to inspect, log, or modify the request object.
	// It can also be used to implement caching by returning a cached
	// response, which would skip the actual remote agent call.
	BeforeRequestCallbacks []BeforeA2ARequestCallback
	// Converter is used to convert a2a.Event to session.Event. If not provided, adka2a.ToSessionEvent
	// is used as the default implementation and errors are converted to events with error payload.
	Converter A2AEventConverter
	// AfterRequestCallbacks will be called in the order they are provided until
	// there's a callback that returns a non-nil result or error. Then
	// the actual remote agent event is replaced with the returned result/error.
	//
	// This is the ideal place to log agent responses, collect metrics on token or perform
	// pre-processing of events before a mapper is invoked.
	AfterRequestCallbacks []AfterA2ARequestCallback
	// AfterAgentCallbacks is a list of callbacks that are called sequentially
	// after the agent has completed its run.
	//
	// If any callback returns non-nil content or error, then a new event will be
	// created from the content or error of that callback and the remaining
	// callbacks will be skipped.
	AfterAgentCallbacks []agent.AfterAgentCallback

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
		Name:                 cfg.Name,
		Description:          cfg.Description,
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
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

		req := &a2a.MessageSendParams{Message: msg, Config: cfg.MessageSendConfig}

		if bcbResp, bcbErr := runBeforeA2ARequestCallbacks(ctx, cfg, req); bcbResp != nil || bcbErr != nil {
			if acbResp, acbErr := runAfterA2ARequestCallbacks(ctx, cfg, req, bcbResp, bcbErr); acbResp != nil || acbErr != nil {
				yield(acbResp, acbErr)
			} else {
				yield(bcbResp, bcbErr)
			}
			return
		}

		if len(msg.Parts) == 0 {
			resp := adka2a.NewRemoteAgentEvent(ctx)
			if cbResp, cbErr := runAfterA2ARequestCallbacks(ctx, cfg, req, resp, err); cbResp != nil || cbErr != nil {
				yield(cbResp, cbErr)
			} else {
				yield(resp, nil)
			}
			return
		}

		for a2aEvent, a2aErr := range client.SendStreamingMessage(ctx, req) {
			var err error
			var event *session.Event
			if cfg.Converter != nil {
				event, err = cfg.Converter(icontext.NewReadonlyContext(ctx), req, a2aEvent, a2aErr)
			} else {
				event, err = convertToSessionEvent(ctx, req, a2aEvent, a2aErr)
			}

			if cbResp, cbErr := runAfterA2ARequestCallbacks(ctx, cfg, req, event, err); cbResp != nil || cbErr != nil {
				if cbErr != nil {
					yield(nil, cbErr)
					return
				}
				if !yield(cbResp, nil) {
					return
				}
				continue
			}

			if err != nil {
				yield(nil, err)
				return
			}

			if event != nil { // an event might be skipped
				if !yield(event, nil) {
					return
				}
			}
		}
	}
}

// Converts A2A client SendStreamingMessage result to a session event. Returns nil if nothing should be emitted.
func convertToSessionEvent(ctx agent.InvocationContext, req *a2a.MessageSendParams, a2aEvent a2a.Event, err error) (*session.Event, error) {
	if err != nil {
		event := toErrorEvent(ctx, err)
		updateCustomMetadata(event, req, nil)
		return event, nil
	}

	event, err := adka2a.ToSessionEvent(ctx, a2aEvent)
	if err != nil {
		event := toErrorEvent(ctx, fmt.Errorf("failed to convert a2aEvent: %w", err))
		updateCustomMetadata(event, req, nil)
		return event, nil
	}

	if event != nil {
		updateCustomMetadata(event, req, a2aEvent)
	}

	return event, nil
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

func runBeforeA2ARequestCallbacks(ctx agent.InvocationContext, cfg A2AConfig, req *a2a.MessageSendParams) (*session.Event, error) {
	cctx := icontext.NewCallbackContext(ctx)
	for _, callback := range cfg.BeforeRequestCallbacks {
		if cbResp, cbErr := callback(cctx, req); cbResp != nil || cbErr != nil {
			return cbResp, cbErr
		}
	}
	return nil, nil
}

func runAfterA2ARequestCallbacks(ctx agent.InvocationContext, cfg A2AConfig, req *a2a.MessageSendParams, resp *session.Event, err error) (*session.Event, error) {
	cctx := icontext.NewCallbackContext(ctx)
	for _, callback := range cfg.AfterRequestCallbacks {
		if cbEvent, cbErr := callback(cctx, req, resp, err); cbEvent != nil || cbErr != nil {
			return cbEvent, cbErr
		}
	}
	return nil, nil
}
