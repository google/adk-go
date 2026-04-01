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

package remoteagent

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"strings"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
)

// A2AClient abstracts a2a-go client so that we can use different SDK versions.
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

// A2AClientProvider creates an [A2AClient].
type A2AClientProvider interface {
	// CreateClient creates an [A2AClient].
	CreateClient(context.Context, *a2a.AgentCard) (A2AClient, error)
}

// RemoteAgentState holds the internal state of a remote agent.
type RemoteAgentState struct {
	// A2A holds the A2A configuration if remote agent is an A2A agent.
	A2A *A2AServerConfig
}

// A2AServerConfig is used to describe and configure a remote agent.
type A2AServerConfig struct {
	// AgentCardSource can be either an http(s) URL or a local file path. If a2a.AgentCard
	// is not provided, the source is used to resolve the card during the first agent invocation.
	AgentCard       *a2a.AgentCard
	AgentCardSource string
	// CardResolveOptions can be used to provide a set of agencard.Resolver configurations.
	CardResolveOptions []agentcard.ResolveOption
	// ClientProvider is used to create an [A2AClient] implementation.
	ClientProvider A2AClientProvider
}

func CreateA2AClient(ctx context.Context, cfg *A2AServerConfig) (*a2a.AgentCard, A2AClient, error) {
	card, err := resolveAgentCard(ctx, cfg)
	if err != nil {
		return nil, nil, fmt.Errorf("agent card resolution failed: %w", err)
	}

	var client A2AClient
	if cfg.ClientProvider != nil {
		client, err = cfg.ClientProvider.CreateClient(ctx, card)
	} else {
		client, err = a2aclient.NewFromCard(ctx, card)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("client creation failed: %w", err)
	}
	return card, client, nil
}

func resolveAgentCard(ctx context.Context, cfg *A2AServerConfig) (*a2a.AgentCard, error) {
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

	var card a2a.AgentCard
	if err := json.Unmarshal(fileBytes, &card); err != nil {
		return nil, fmt.Errorf("failed to unmarshal an agent card: %w", err)
	}
	return &card, nil
}
