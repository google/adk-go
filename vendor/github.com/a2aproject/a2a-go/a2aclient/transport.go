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
	"errors"
	"iter"

	"github.com/a2aproject/a2a-go/a2a"
)

// A2AClient defines a transport-agnostic interface for making A2A requests.
// Transport implementations are a translation layer between a2a core types and wire formats.
type Transport interface {
	// GetTask calls the 'tasks/get' protocol method.
	GetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error)

	// CancelTask calls the 'tasks/cancel' protocol method.
	CancelTask(ctx context.Context, id *a2a.TaskIDParams) (*a2a.Task, error)

	// SendMessage calls the 'message/send' protocol method (non-streaming).
	SendMessage(ctx context.Context, message *a2a.MessageSendParams) (a2a.SendMessageResult, error)

	// ResubscribeToTask calls the `tasks/resubscribe` protocol method.
	ResubscribeToTask(ctx context.Context, id *a2a.TaskIDParams) iter.Seq2[a2a.Event, error]

	// SendStreamingMessage calls the 'message/stream' protocol method (streaming).
	SendStreamingMessage(ctx context.Context, message *a2a.MessageSendParams) iter.Seq2[a2a.Event, error]

	// GetTaskPushNotificationConfig calls the `tasks/pushNotificationConfig/get` protocol method.
	GetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error)

	// ListTaskPushNotificationConfig calls the `tasks/pushNotificationConfig/list` protocol method.
	ListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error)

	// SetTaskPushConfig calls the `tasks/pushNotificationConfig/set` protocol method.
	SetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error)

	// DeleteTaskPushNotificationConfig calls the `tasks/pushNotificationConfig/delete` protocol method.
	DeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error

	// GetAgentCard resolves the AgentCard.
	// If extended card is supported calls the 'agent/getAuthenticatedExtendedCard' protocol method.
	GetAgentCard(ctx context.Context) (*a2a.AgentCard, error)

	// Clean up resources associated with the transport (eg. close a gRPC channel).
	Destroy() error
}

// TransportFactory creates an A2A protocol connection to the provided URL.
type TransportFactory interface {
	Create(ctx context.Context, url string, card *a2a.AgentCard) (Transport, error)
}

// TransportFactoryFn implements TransportFactory.
type TransportFactoryFn func(ctx context.Context, url string, card *a2a.AgentCard) (Transport, error)

func (fn TransportFactoryFn) Create(ctx context.Context, url string, card *a2a.AgentCard) (Transport, error) {
	return fn(ctx, url, card)
}

var errNotImplemented = errors.New("not implemented")

type unimplementedTransport struct{}

func (unimplementedTransport) GetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) CancelTask(ctx context.Context, id *a2a.TaskIDParams) (*a2a.Task, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) SendMessage(ctx context.Context, message *a2a.MessageSendParams) (a2a.SendMessageResult, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) ResubscribeToTask(ctx context.Context, id *a2a.TaskIDParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(nil, errNotImplemented)
	}
}

func (unimplementedTransport) SendStreamingMessage(ctx context.Context, message *a2a.MessageSendParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(nil, errNotImplemented)
	}
}

func (unimplementedTransport) GetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) ListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) SetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) DeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error {
	return errNotImplemented
}

func (unimplementedTransport) GetAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	return nil, errNotImplemented
}

func (unimplementedTransport) Destroy() error {
	return nil
}
