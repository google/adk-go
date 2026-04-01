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
	"context"
	"iter"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"google.golang.org/adk/agent"
)

// A2AMessageSender is used to send messages to a remote agent.
type A2AClient interface {
	// SendMessage sends a message to the remote agent and returns the result.
	SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error)
	// SendStreamingMessage sends a message to the remote agent and returns a stream of events.
	SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error]
	// SendStreamingMessage sends a message to the remote agent and returns a stream of events.
	CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error)
	// Destroy is called in the end of agent invocation.
	Destroy() error
}

// A2AMessageSenderProvider is a function that creates an A2AMessageSender.
type A2AMessageSenderProvider func(agent.InvocationContext, *a2a.AgentCard) (A2AClient, error)

// NewA2AMessageSenderProvider creates a default A2AMessageSenderProvider from the configured factory.
func NewA2AMessageSenderProvider(factory *a2aclient.Factory) A2AMessageSenderProvider {
	return func(ctx agent.InvocationContext, card *a2a.AgentCard) (A2AClient, error) {
		var client *a2aclient.Client
		var err error
		if factory != nil {
			client, err = factory.CreateFromCard(ctx, card)
		} else {
			client, err = a2aclient.NewFromCard(ctx, card)
		}
		if err != nil {
			return nil, err
		}
		return &defaultA2AMessageSender{client: client}, nil
	}
}

type defaultA2AMessageSender struct {
	client *a2aclient.Client
}

func (s *defaultA2AMessageSender) SendMessage(ctx context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	return s.client.SendMessage(ctx, req)
}

func (s *defaultA2AMessageSender) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return s.client.SendStreamingMessage(ctx, req)
}

func (s *defaultA2AMessageSender) CancelTask(ctx context.Context, req *a2a.CancelTaskRequest) (*a2a.Task, error) {
	return s.client.CancelTask(ctx, req)
}

func (s *defaultA2AMessageSender) Destroy() error {
	return s.client.Destroy()
}
