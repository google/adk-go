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
// This package provides the client foundation: configuration ([Config]), an
// authenticated REST transport to the third-party (public) endpoint using
// Application Default Credentials with mTLS endpoint selection, typed errors
// ([APIError]), and the wire types returned by the service. Discovery methods
// and the RemoteAgent/MCPToolset factory helpers build on this foundation.
package agentregistry
