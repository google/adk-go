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

// Package remoteagent allows using a remote ADK agents.
//
// Deprecated: Use google.golang.org/adk/agent/remoteagent/v2 instead.
package remoteagent

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2aclient/agentcard"
	"github.com/a2aproject/a2a-go/log"
	v2a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	v2 "google.golang.org/adk/agent/remoteagent/v2"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

// BeforeA2ARequestCallback is called before sending a request to the remote agent.
//
// If it returns non-nil result or error, the actual call is skipped and the returned value is used
// as the agent invocation result.
type BeforeA2ARequestCallback func(ctx agent.CallbackContext, req *a2a.MessageSendParams) (*session.Event, error)

// A2AEventConverter can be used to provide a custom implementation of A2A event transformation logic.
type A2AEventConverter func(ctx agent.InvocationContext, req *a2a.MessageSendParams, event a2a.Event, err error) (*session.Event, error)

// AfterA2ARequestCallback is called after receiving a response from the remote agent and converting it to a session.Event.
// In streaming responses the callback is invoked for every request. Session event parameter might be nil if conversion logic
// decides to not emit an A2A event.
//
// If it returns non-nil result or error, it gets emitted instead of the original result.
type AfterA2ARequestCallback func(ctx agent.CallbackContext, req *a2a.MessageSendParams, resp *session.Event, err error) (*session.Event, error)

// A2ARemoteTaskCleanupCallback is called if Run exited before a terminal event was received from the remote A2A server.
type A2ARemoteTaskCleanupCallback func(ctx context.Context, card *a2a.AgentCard, client *a2aclient.Client, taskInfo a2a.TaskInfo, cause error)

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

	// A2APartConverter is a custom converter for converting A2A parts to GenAI parts.
	// Implementations should generally remember to leverage adka2a.ToGenAiPart for default conversions
	// nil returns are considered intentionally dropped parts.
	A2APartConverter adka2a.A2APartConverter

	// GenAIPartConverter is a custom converter for converting GenAI parts to A2A parts.
	// Implementations should generally remember to leverage adka2a.ToA2APart for default conversions
	// nil returns are considered intentionally dropped parts.
	GenAIPartConverter adka2a.GenAIPartConverter

	// ClientFactory can be used to provide a set of a2aclient.Client configurations.
	ClientFactory *a2aclient.Factory
	// MessageSendConfig is attached to a2a.MessageSendParams sent on every agent invocation.
	MessageSendConfig *a2a.MessageSendConfig

	// RemoteTaskCleanupCallback is called if Run exited before a terminal event was received from the remote A2A server.
	// If Run exited due to an error including context cancellation it will be passed as cause.
	// The context passed to this callback is the original context, but with Err() removed by context.WithoutCancel.
	// If no callback is provided the default behavior is to make a cancel RPC request with 5 second timeout.
	RemoteTaskCleanupCallback A2ARemoteTaskCleanupCallback
}

// NewA2A creates a remote A2A agent. A2A (Agent-To-Agent) protocol is used for communication with an
// agent which can run in a different process or on a different host.
func NewA2A(cfg A2AConfig) (agent.Agent, error) {
	if cfg.AgentCard == nil && cfg.AgentCardSource == "" {
		return nil, fmt.Errorf("either AgentCard or AgentCardSource must be provided")
	}

	v1Cfg := v2.A2AConfig{
		Name:                 cfg.Name,
		Description:          cfg.Description,
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
	}

	if cfg.AgentCard != nil {
		v1Cfg.AgentCard = a2av0.ToV1AgentCard(cfg.AgentCard)
	} else if cfg.AgentCardSource != "" {
		source := cfg.AgentCardSource
		resolveOpts := cfg.CardResolveOptions
		v1Cfg.AgentCardProvider = func(ctx context.Context) (*v2a2a.AgentCard, error) {
			v0Card, err := agentcard.DefaultResolver.Resolve(ctx, source, resolveOpts...)
			if err != nil {
				return nil, err
			}
			return a2av0.ToV1AgentCard(v0Card), nil
		}
	}

	if cfg.MessageSendConfig != nil {
		req, err := a2av0.ToV1SendMessageRequest(&a2a.MessageSendParams{Config: cfg.MessageSendConfig})
		if err != nil {
			return nil, fmt.Errorf("MessageSendConfig conversion failed: %w", err)
		}
		v1Cfg.MessageSendConfig = req.Config
	}

	if cfg.ClientFactory != nil {
		v1Cfg.ClientProvider = func(ctx context.Context, card *v2a2a.AgentCard) (v2.A2AClient, error) {
			legacyCard := a2av0.FromV1AgentCard(card)
			client, err := cfg.ClientFactory.CreateFromCard(ctx, legacyCard)
			if err != nil {
				return nil, err
			}
			return &compatClient{client: client}, nil
		}
	}

	if cfg.Converter != nil {
		v1Cfg.Converter = func(ctx agent.InvocationContext, req *v2a2a.SendMessageRequest, event v2a2a.Event, err error) (*session.Event, error) {
			legacyReq := a2av0.FromV1SendMessageRequest(req)
			var legacyEvent a2a.Event
			if event != nil {
				var convErr error
				legacyEvent, convErr = a2av0.FromV1Event(event)
				if convErr != nil {
					return nil, errors.Join(fmt.Errorf("a2a event conversion failed: %w", convErr), err)
				}
			}
			return cfg.Converter(ctx, legacyReq, legacyEvent, err)
		}
	}

	if cfg.BeforeRequestCallbacks != nil {
		v1Cfg.BeforeRequestCallbacks = make([]v2.BeforeA2ARequestCallback, 0, len(cfg.BeforeRequestCallbacks))
		for _, cb := range cfg.BeforeRequestCallbacks {
			v1Cfg.BeforeRequestCallbacks = append(v1Cfg.BeforeRequestCallbacks, func(ctx agent.CallbackContext, req *v2a2a.SendMessageRequest) (*session.Event, error) {
				legacyReq := a2av0.FromV1SendMessageRequest(req)
				resp, err := cb(ctx, legacyReq)
				if resp != nil || err != nil { // short-circuit, no need to convert the request back
					return resp, err
				}
				// callback pass-through request modifications
				v1Req, convErr := a2av0.ToV1SendMessageRequest(legacyReq)
				if convErr != nil {
					return nil, convErr
				}
				*req = *v1Req
				return nil, nil
			})
		}
	}

	if cfg.AfterRequestCallbacks != nil {
		v1Cfg.AfterRequestCallbacks = make([]v2.AfterA2ARequestCallback, 0, len(cfg.AfterRequestCallbacks))
		for _, cb := range cfg.AfterRequestCallbacks {
			v1Cfg.AfterRequestCallbacks = append(v1Cfg.AfterRequestCallbacks, func(ctx agent.CallbackContext, req *v2a2a.SendMessageRequest, resp *session.Event, err error) (*session.Event, error) {
				legacyReq := a2av0.FromV1SendMessageRequest(req)
				newResp, newErr := cb(ctx, legacyReq, resp, err)
				if newResp != nil || newErr != nil { // short-circuit, no need to convert the request back
					return newResp, newErr
				}
				// callback pass-through request modifications
				v1Req, convErr := a2av0.ToV1SendMessageRequest(legacyReq)
				if convErr != nil {
					return nil, convErr
				}
				*req = *v1Req
				return nil, nil
			})
		}
	}

	if cfg.A2APartConverter != nil {
		v1Cfg.A2APartConverter = func(ctx context.Context, a2aEvent v2a2a.Event, part *v2a2a.Part) (*genai.Part, error) {
			legacyEvent, convErr := a2av0.FromV1Event(a2aEvent)
			if convErr != nil {
				return nil, convErr
			}
			return cfg.A2APartConverter(ctx, legacyEvent, a2av0.FromV1Part(part))
		}
	}

	if cfg.GenAIPartConverter != nil {
		v1Cfg.GenAIPartConverter = func(ctx context.Context, adkEvent *session.Event, part *genai.Part) (*v2a2a.Part, error) {
			legacyPart, err := cfg.GenAIPartConverter(ctx, adkEvent, part)
			if err != nil {
				return nil, err
			}
			return a2av0.ToV1Part(legacyPart), nil
		}
	}

	if cfg.RemoteTaskCleanupCallback != nil {
		v1Cfg.RemoteTaskCleanupCallback = func(ctx context.Context, card *v2a2a.AgentCard, client v2.A2AClient, taskInfo v2a2a.TaskInfo, cause error) {
			legacyCard := a2av0.FromV1AgentCard(card)
			legacyTaskInfo := a2a.TaskInfo{TaskID: a2a.TaskID(taskInfo.TaskID), ContextID: taskInfo.ContextID}

			if cc, ok := client.(*compatClient); ok {
				cfg.RemoteTaskCleanupCallback(ctx, legacyCard, cc.client, legacyTaskInfo, cause)
				return
			}

			log.Warn(ctx, "client is not an instance of compatClient, fallback to creating a new client", "type", fmt.Sprintf("%T", client))

			factory := cfg.ClientFactory
			if factory == nil {
				factory = a2aclient.NewFactory()
			}
			legacyClient, err := factory.CreateFromCard(ctx, legacyCard)
			if err != nil {
				log.Warn(ctx, "RemoteTaskCleanupCallback: failed to create legacy client", "error", err)
				return
			}
			defer func() {
				if err := legacyClient.Destroy(); err != nil {
					log.Warn(ctx, "RemoteTaskCleanupCallback: failed to destroy a legacy client", "error", err)
				}
			}()
			cfg.RemoteTaskCleanupCallback(ctx, legacyCard, legacyClient, legacyTaskInfo, cause)
		}
	}

	return v2.NewA2A(v1Cfg)
}

type compatClient struct {
	client *a2aclient.Client
}

func (s *compatClient) SendMessage(ctx context.Context, req *v2a2a.SendMessageRequest) (v2a2a.SendMessageResult, error) {
	legacyResp, err := s.client.SendMessage(ctx, a2av0.FromV1SendMessageRequest(req))
	if err != nil {
		return nil, err
	}
	v1Event, err := a2av0.ToV1Event(legacyResp)
	if err != nil {
		return nil, err
	}
	res, ok := v1Event.(v2a2a.SendMessageResult)
	if !ok {
		return nil, fmt.Errorf("converted event does not implement SendMessageResult: %T", v1Event)
	}
	return res, nil
}

func (s *compatClient) SendStreamingMessage(ctx context.Context, req *v2a2a.SendMessageRequest) iter.Seq2[v2a2a.Event, error] {
	return func(yield func(v2a2a.Event, error) bool) {
		for legacyEvent, err := range s.client.SendStreamingMessage(ctx, a2av0.FromV1SendMessageRequest(req)) {
			if err != nil {
				yield(nil, err)
				return
			}
			v1Event, convErr := a2av0.ToV1Event(legacyEvent)
			if convErr != nil {
				yield(nil, convErr)
				return
			}
			if !yield(v1Event, nil) {
				return
			}
		}
	}
}

func (s *compatClient) CancelTask(ctx context.Context, req *v2a2a.CancelTaskRequest) (*v2a2a.Task, error) {
	legacyResp, err := s.client.CancelTask(ctx, a2av0.FromV1CancelTaskRequest(req))
	if err != nil {
		return nil, err
	}
	return a2av0.ToV1Task(legacyResp)
}

func (s *compatClient) Destroy() error {
	return s.client.Destroy()
}
