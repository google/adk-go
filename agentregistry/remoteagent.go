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

package agentregistry

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"

	"google.golang.org/adk/v2/agent"
	remoteagent "google.golang.org/adk/v2/agent/remoteagent/v2"
)

// cardTypeA2AAgentCard is the Card.Type value for an embedded A2A agent card.
const cardTypeA2AAgentCard = "A2A_AGENT_CARD"

// defaultA2AProtocolVersion is used when the registry does not report one.
const defaultA2AProtocolVersion = "0.3.0"

// RemoteAgent resolves a registered A2A agent into an [agent.Agent] usable as a
// sub-agent. name is the full agent resource name.
//
// The agent card is taken from the registry's embedded card when present, and
// otherwise synthesized from the agent's discrete fields. Egress auth is left to
// the caller: pass [WithA2AHTTPClient] (and/or [WithA2AHeaders]) to authenticate
// requests to the remote agent.
func (c *Client) RemoteAgent(ctx context.Context, name string, opts ...RemoteAgentOption) (agent.Agent, error) {
	info, err := c.GetAgent(ctx, name)
	if err != nil {
		return nil, err
	}

	card, agentName, description, err := agentCard(info, name)
	if err != nil {
		return nil, err
	}

	ec := applyRemoteAgentOptions(opts)
	egress := c.egressClient(cardURL(card), ec, false)

	return remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:           agentName,
		Description:    description,
		AgentCard:      card,
		ClientProvider: remoteagent.NewA2AClientProvider(a2aClientFactory(card, egress)),
	})
}

// a2aClientFactory builds an A2A client factory for the given card and HTTP
// client. Besides the SDK's current-version transports, it registers a
// compatibility transport for each (binding, protocolVersion) the card
// advertises. Agent Registry commonly reports an older A2A protocol version
// (e.g. 0.3.0) than the SDK's current one, and the a2a-go factory matches
// transports by (protocol, version); without a matching compat transport,
// client creation fails with "no compatible transports found".
func a2aClientFactory(card *a2a.AgentCard, httpClient *http.Client) *a2aclient.Factory {
	opts := []a2aclient.FactoryOption{
		a2aclient.WithJSONRPCTransport(httpClient),
		a2aclient.WithRESTTransport(httpClient),
	}
	seen := make(map[string]bool)
	for _, iface := range card.SupportedInterfaces {
		// The current-version transports above already cover a2a.Version.
		if iface == nil || iface.ProtocolVersion == "" || iface.ProtocolVersion == a2a.Version {
			continue
		}
		var factory a2aclient.TransportFactory
		switch iface.ProtocolBinding {
		case a2a.TransportProtocolJSONRPC:
			factory = jsonrpcTransportFactory(httpClient)
		case a2a.TransportProtocolHTTPJSON:
			factory = restTransportFactory(httpClient)
		default:
			continue
		}
		key := string(iface.ProtocolBinding) + "@" + string(iface.ProtocolVersion)
		if seen[key] {
			continue
		}
		seen[key] = true
		opts = append(opts, a2aclient.WithCompatTransport(iface.ProtocolVersion, iface.ProtocolBinding, factory))
	}
	return a2aclient.NewFactory(opts...)
}

func restTransportFactory(httpClient *http.Client) a2aclient.TransportFactory {
	return a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
		u, err := url.Parse(iface.URL)
		if err != nil {
			return nil, fmt.Errorf("agentregistry: parsing endpoint URL %q: %w", iface.URL, err)
		}
		return a2aclient.NewRESTTransport(u, httpClient), nil
	})
}

func jsonrpcTransportFactory(httpClient *http.Client) a2aclient.TransportFactory {
	return a2aclient.TransportFactoryFn(func(_ context.Context, _ *a2a.AgentCard, iface *a2a.AgentInterface) (a2aclient.Transport, error) {
		return a2aclient.NewJSONRPCTransport(iface.URL, httpClient), nil
	})
}

// agentCard resolves an [a2a.AgentCard] plus the cleaned agent name and
// description for a registered agent, using the embedded card when available and
// otherwise synthesizing one. resourceName is used as a fallback name.
func agentCard(info *Agent, resourceName string) (card *a2a.AgentCard, name, description string, err error) {
	if info.Card != nil && info.Card.Type == cardTypeA2AAgentCard && len(info.Card.Content) > 0 {
		var embedded a2a.AgentCard
		if err := json.Unmarshal(info.Card.Content, &embedded); err != nil {
			return nil, "", "", fmt.Errorf("agentregistry: decoding embedded agent card for %q: %w", resourceName, err)
		}
		agentName := cleanName(embedded.Name)
		if agentName == "" {
			agentName = cleanName(resourceName)
		}
		return &embedded, agentName, embedded.Description, nil
	}

	url, version, binding, ok := connectionURI(info.Protocols, nil, protocolTypeA2AAgent, "")
	if !ok {
		return nil, "", "", fmt.Errorf("agentregistry: A2A connection URI not found for agent %q", resourceName)
	}

	displayName := info.DisplayName
	if displayName == "" {
		displayName = resourceName
	}
	agentName := cleanName(displayName)

	transport, ok := transportBinding(binding)
	if !ok {
		transport = a2a.TransportProtocolHTTPJSON
	}
	protocolVersion := version
	if protocolVersion == "" {
		protocolVersion = defaultA2AProtocolVersion
	}

	card = &a2a.AgentCard{
		Name:        agentName,
		Description: info.Description,
		Version:     info.Version,
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL:             url,
			ProtocolBinding: transport,
			ProtocolVersion: a2a.ProtocolVersion(protocolVersion),
		}},
		Skills:             toA2ASkills(info.Skills),
		Capabilities:       a2a.AgentCapabilities{Streaming: false},
		DefaultInputModes:  []string{"text"},
		DefaultOutputModes: []string{"text"},
	}
	return card, agentName, info.Description, nil
}

// cardURL returns the first supported-interface URL of a card, or "".
func cardURL(card *a2a.AgentCard) string {
	for _, iface := range card.SupportedInterfaces {
		if iface != nil && iface.URL != "" {
			return iface.URL
		}
	}
	return ""
}

// toA2ASkills converts registry skills to A2A skills.
func toA2ASkills(skills []Skill) []a2a.AgentSkill {
	if len(skills) == 0 {
		return nil
	}
	out := make([]a2a.AgentSkill, 0, len(skills))
	for _, s := range skills {
		out = append(out, a2a.AgentSkill{
			ID:          s.ID,
			Name:        s.Name,
			Description: s.Description,
			Tags:        s.Tags,
			Examples:    s.Examples,
		})
	}
	return out
}
