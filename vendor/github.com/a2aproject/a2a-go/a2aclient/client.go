// Copyright 2025 The A2A Authors
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

package a2aclient

import (
	"context"
	"iter"
	"sync/atomic"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/utils"
)

// Config exposes options for customizing [Client] behavior.
type Config struct {
	// PushConfig specifies the default push notification configuration to apply for every Task.
	PushConfig *a2a.PushConfig
	// AcceptedOutputModes are MIME types passed with every Client message and might be used by an agent
	// to decide on the result format.
	// For example, an Agent might declare a skill with OutputModes: ["application/json", "image/png"]
	// and a Client that doesn't support images will pass AcceptedOutputModes: ["application/json"]
	// to get a result in the desired format.
	AcceptedOutputModes []string
	// PreferredTransports is used for selecting the most appropriate communication protocol.
	// The first transport from the list which is also supported by the server is going to be used
	// to establish a connection. If no preference is provided the server ordering will be used.
	// If there's no overlap in supported Transport Factory will return an error on Client
	// creation attempt.
	PreferredTransports []a2a.TransportProtocol
	// Whether client prefers to poll for task updates instead of blocking until a terminal state is reached.
	// If set to true, non-streaming send message result might be a Message or a Task in any (including non-terminal) state.
	// Callers are responsible for running the polling loop. This configuration does not apply to streaming requests.
	Polling bool
}

// Client represents a transport-agnostic implementation of A2A client.
// The actual call is delegated to a specific [Transport] implementation.
// [CallInterceptor]-s are applied before and after every protocol call.
type Client struct {
	config       Config
	transport    Transport
	interceptors []CallInterceptor
	baseURL      string

	card atomic.Pointer[a2a.AgentCard]
}

// AddCallInterceptor allows to attach a [CallInterceptor] to the client after creation.
func (c *Client) AddCallInterceptor(ci CallInterceptor) {
	c.interceptors = append(c.interceptors, ci)
}

// A2A protocol methods

func (c *Client) GetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error) {
	method := "GetTask"

	ctx, err := c.interceptBefore(ctx, method, query)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.GetTask(ctx, query)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) CancelTask(ctx context.Context, id *a2a.TaskIDParams) (*a2a.Task, error) {
	method := "CancelTask"

	ctx, err := c.interceptBefore(ctx, method, id)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.CancelTask(ctx, id)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) SendMessage(ctx context.Context, message *a2a.MessageSendParams) (a2a.SendMessageResult, error) {
	method := "SendMessage"

	message = c.withDefaultSendConfig(message, blocking(!c.config.Polling))

	ctx, err := c.interceptBefore(ctx, method, message)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.SendMessage(ctx, message)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) SendStreamingMessage(ctx context.Context, message *a2a.MessageSendParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		method := "SendStreamingMessage"

		message = c.withDefaultSendConfig(message, blocking(true))

		ctx, err := c.interceptBefore(ctx, method, message)
		if err != nil {
			yield(nil, err)
			return
		}

		if card := c.card.Load(); card != nil && !card.Capabilities.Streaming {
			resp, err := c.transport.SendMessage(ctx, message)
			if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
				yield(nil, errOverride)
				return
			}
			yield(resp, err)
			return
		}

		for resp, err := range c.transport.SendStreamingMessage(ctx, message) {
			if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
				yield(nil, errOverride)
				return
			}

			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(resp, nil) {
				return
			}
		}
	}
}

func (c *Client) ResubscribeToTask(ctx context.Context, id *a2a.TaskIDParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		method := "ResubscribeToTask"

		ctx, err := c.interceptBefore(ctx, method, id)
		if err != nil {
			yield(nil, err)
			return
		}

		for resp, err := range c.transport.ResubscribeToTask(ctx, id) {
			if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
				yield(nil, errOverride)
				return
			}

			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(resp, nil) {
				return
			}
		}
	}
}

func (c *Client) GetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error) {
	method := "GetTaskPushConfig"

	ctx, err := c.interceptBefore(ctx, method, params)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.GetTaskPushConfig(ctx, params)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) ListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error) {
	method := "ListTaskPushConfig"

	ctx, err := c.interceptBefore(ctx, method, params)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.ListTaskPushConfig(ctx, params)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) SetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error) {
	method := "SetTaskPushConfig"

	ctx, err := c.interceptBefore(ctx, method, params)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.SetTaskPushConfig(ctx, params)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	return resp, err
}

func (c *Client) DeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error {
	method := "DeleteTaskPushConfig"

	ctx, err := c.interceptBefore(ctx, method, params)
	if err != nil {
		return err
	}

	err = c.transport.DeleteTaskPushConfig(ctx, params)
	if errOverride := c.interceptAfter(ctx, method, nil, err); errOverride != nil {
		return errOverride
	}

	return err
}

func (c *Client) GetAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	if card := c.card.Load(); card != nil && !card.SupportsAuthenticatedExtendedCard {
		return card, nil
	}

	method := "GetAgentCard"

	ctx, err := c.interceptBefore(ctx, method, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.transport.GetAgentCard(ctx)
	if errOverride := c.interceptAfter(ctx, method, resp, err); errOverride != nil {
		return nil, errOverride
	}

	if err == nil {
		c.card.Store(resp)
	}

	return resp, err
}

func (c *Client) Destroy() error {
	return c.transport.Destroy()
}

type blocking bool

func (c *Client) withDefaultSendConfig(message *a2a.MessageSendParams, blocking blocking) *a2a.MessageSendParams {
	if c.config.PushConfig == nil && c.config.AcceptedOutputModes == nil && blocking {
		return message
	}
	result := *message
	if result.Config == nil {
		result.Config = &a2a.MessageSendConfig{}
	} else {
		configCopy := *result.Config
		result.Config = &configCopy
	}
	if result.Config.PushConfig == nil {
		result.Config.PushConfig = c.config.PushConfig
	}
	if result.Config.AcceptedOutputModes == nil {
		result.Config.AcceptedOutputModes = c.config.AcceptedOutputModes
	}
	result.Config.Blocking = utils.Ptr(bool(blocking))
	return &result
}

func (c *Client) interceptBefore(ctx context.Context, method string, payload any) (context.Context, error) {
	req := Request{
		Method:  method,
		BaseURL: c.baseURL,
		Meta:    CallMeta{},
		Card:    c.card.Load(),
		Payload: payload,
	}

	if payload == nil { // set interface to nil if method does not take any parameters
		req.Payload = nil
	}

	for _, interceptor := range c.interceptors {
		localCtx, err := interceptor.Before(ctx, &req)
		if err != nil {
			return ctx, err
		}
		ctx = localCtx
	}

	return withCallMeta(ctx, req.Meta), nil
}

func (c *Client) interceptAfter(ctx context.Context, method string, payload any, err error) error {
	meta, ok := CallMetaFrom(ctx)
	if !ok {
		meta = CallMeta{}
	}

	resp := Response{
		BaseURL: c.baseURL,
		Method:  method,
		Meta:    meta,
		Payload: payload,
		Card:    c.card.Load(),
		Err:     err,
	}
	if payload == nil { // set interface to nil if method does not return any value
		resp.Payload = nil
	}

	for _, interceptor := range c.interceptors {
		if err := interceptor.After(ctx, &resp); err != nil {
			return err
		}
	}

	return resp.Err
}
