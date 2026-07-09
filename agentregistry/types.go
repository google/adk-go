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
	"encoding/json"
	"net/url"
	"strconv"
)

// Interface describes a single connection interface (endpoint URL + binding)
// for a protocol.
type Interface struct {
	URL             string `json:"url,omitempty"`
	ProtocolBinding string `json:"protocolBinding,omitempty"`
}

// Protocol describes a protocol a resource speaks together with its interfaces.
type Protocol struct {
	Type            string      `json:"type,omitempty"`
	ProtocolVersion string      `json:"protocolVersion,omitempty"`
	Interfaces      []Interface `json:"interfaces,omitempty"`
}

// Skill describes an A2A agent skill.
type Skill struct {
	ID          string   `json:"id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Examples    []string `json:"examples,omitempty"`
}

// Card carries an embedded agent card returned by the registry. Content holds
// the raw card JSON (e.g. an A2A AgentCard) when Type is "A2A_AGENT_CARD".
type Card struct {
	Type    string          `json:"type,omitempty"`
	Content json.RawMessage `json:"content,omitempty"`
}

// Agent is a registered A2A agent.
type Agent struct {
	Name        string     `json:"name,omitempty"`
	DisplayName string     `json:"displayName,omitempty"`
	Description string     `json:"description,omitempty"`
	Version     string     `json:"version,omitempty"`
	Protocols   []Protocol `json:"protocols,omitempty"`
	Skills      []Skill    `json:"skills,omitempty"`
	Card        *Card      `json:"card,omitempty"`
}

// MCPServer is a registered MCP server.
type MCPServer struct {
	Name        string      `json:"name,omitempty"`
	MCPServerID string      `json:"mcpServerId,omitempty"`
	DisplayName string      `json:"displayName,omitempty"`
	Description string      `json:"description,omitempty"`
	Protocols   []Protocol  `json:"protocols,omitempty"`
	Interfaces  []Interface `json:"interfaces,omitempty"`
}

// Endpoint is a registered model endpoint.
type Endpoint struct {
	Name        string         `json:"name,omitempty"`
	EndpointID  string         `json:"endpointId,omitempty"`
	DisplayName string         `json:"displayName,omitempty"`
	Description string         `json:"description,omitempty"`
	Interfaces  []Interface    `json:"interfaces,omitempty"`
	CreateTime  string         `json:"createTime,omitempty"`
	UpdateTime  string         `json:"updateTime,omitempty"`
	Attributes  map[string]any `json:"attributes,omitempty"`
}

// ListAgentsResponse is one page of a [Client.ListAgents] response.
type ListAgentsResponse struct {
	Agents        []Agent `json:"agents,omitempty"`
	NextPageToken string  `json:"nextPageToken,omitempty"`
}

// ListMCPServersResponse is one page of a [Client.ListMCPServers] response.
type ListMCPServersResponse struct {
	MCPServers    []MCPServer `json:"mcpServers,omitempty"`
	NextPageToken string      `json:"nextPageToken,omitempty"`
}

// ListEndpointsResponse is one page of a [Client.ListEndpoints] response.
type ListEndpointsResponse struct {
	Endpoints     []Endpoint `json:"endpoints,omitempty"`
	NextPageToken string     `json:"nextPageToken,omitempty"`
}

// ListOption customizes a list request (filter and pagination).
type ListOption func(url.Values)

// WithFilter sets the list filter expression.
func WithFilter(filter string) ListOption {
	return func(v url.Values) {
		if filter != "" {
			v.Set("filter", filter)
		}
	}
}

// WithPageSize sets the maximum number of results per page.
func WithPageSize(size int) ListOption {
	return func(v url.Values) {
		if size > 0 {
			v.Set("pageSize", strconv.Itoa(size))
		}
	}
}

// WithPageToken sets the page token used to continue a previous list call.
func WithPageToken(token string) ListOption {
	return func(v url.Values) {
		if token != "" {
			v.Set("pageToken", token)
		}
	}
}

// listValues collapses list options into url.Values.
func listValues(opts ...ListOption) url.Values {
	v := url.Values{}
	for _, opt := range opts {
		opt(v)
	}
	return v
}
