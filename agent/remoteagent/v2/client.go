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

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"

	iremoteagent "google.golang.org/adk/internal/agent/remoteagent"
)

// A2AClient is used to send messages to a remote agent.
type A2AClient iremoteagent.A2AClient

// A2AClientProvider is a function that creates an A2AMessageSender.
type A2AClientProvider func(context.Context, *a2a.AgentCard) (A2AClient, error)

// CreateClient implements iremoteagent.A2AClientProvider.
func (fn A2AClientProvider) CreateClient(ctx context.Context, card *a2a.AgentCard) (iremoteagent.A2AClient, error) {
	return fn(ctx, card)
}

// NewA2AClientProvider creates a default A2AClientProvider from the configured factory.
func NewA2AClientProvider(factory *a2aclient.Factory) A2AClientProvider {
	return func(ctx context.Context, card *a2a.AgentCard) (A2AClient, error) {
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
		return client, nil
	}
}
