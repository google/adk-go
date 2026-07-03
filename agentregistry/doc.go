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

// Package agentregistry provides a client for the Google Cloud Agent Registry
// (agentregistry.googleapis.com), a governed catalog of A2A agents, MCP
// servers, and model endpoints.
//
// The client exposes two kinds of operations: discovery — List/Get (and
// auto-paging All*) for agents, MCP servers, and endpoints — and factories that
// turn registry metadata into ready-to-use ADK objects: [Client.RemoteAgent]
// returns an agent.Agent usable as a sub-agent, and [Client.McpToolset] returns
// a tool.Toolset.
//
// # Usage
//
//	ctx := context.Background()
//	client, err := agentregistry.New(ctx, agentregistry.Config{
//		ProjectID: "my-project",
//		Location:  "us-central1",
//	})
//	// handle err
//
//	// Discover agents, following pages on demand.
//	for a, err := range client.AllAgents(ctx) {
//		// handle err; use a
//	}
//
//	// Build a sub-agent and a toolset from the registry.
//	sub, err := client.RemoteAgent(ctx,
//		"projects/my-project/locations/us-central1/agents/summarizer")
//	tools, err := client.McpToolset(ctx,
//		"projects/my-project/locations/us-central1/mcpServers/user-data")
//
// # Authentication
//
// By default [New] authenticates to the registry with Application Default
// Credentials and selects the mTLS endpoint according to the
// GOOGLE_API_USE_MTLS_ENDPOINT and GOOGLE_API_USE_CLIENT_CERTIFICATE
// environment variables. Provide Config.HTTPClient to supply a custom (for
// example, pre-authenticated) client.
//
// For egress to the resolved endpoints, [Client.McpToolset] reuses the
// registry's credentials for *.googleapis.com targets and is otherwise
// unauthenticated, while [Client.RemoteAgent] leaves egress auth to the caller.
// Both factories accept a caller-supplied HTTP client and static headers via
// options (the WithMcp* and WithA2A* functions).
//
// # Parity with adk-python
//
// This client mirrors google.adk.integrations.agent_registry. Implemented:
//   - ListAgents/GetAgent, ListMcpServers/GetMcpServer, ListEndpoints/GetEndpoint
//   - ModelName endpoint parsing
//   - RemoteAgent (embedded agent-card fast path and field synthesis)
//   - McpToolset (JSONRPC-preferred endpoint resolution; ADC for Google APIs)
//   - mTLS endpoint selection and the x-goog-user-project quota header
//
// Deferred:
//   - bindings-based GcpAuthProvider resolution and managed OAuth, which depend
//     on the ADK Go authentication subsystem;
//   - MCP tool-name prefixing and the gcp.mcp.server.destination.id telemetry
//     attribute, which require tool-level metadata support;
//   - keyword/semantic search.
package agentregistry
