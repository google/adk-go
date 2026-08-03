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

// Package auth provides a small, idiomatic authentication layer for ADK Go.
//
// It models the two things a consumer actually needs: a resolved [Credential]
// that knows how to write itself onto an outbound HTTP request, and a
// [CredentialProvider] seam that resolves a credential for the current
// invocation. Built-in providers cover the common cases — a static bearer
// token ([StaticToken]), a header API key ([APIKey]), any
// [golang.org/x/oauth2.TokenSource] ([TokenSourceProvider]), Application
// Default Credentials ([ADC]), and service accounts ([ServiceAccount]) — and a
// context-aware [Transport] applies a provider per outgoing request.
//
// Token exchange and refresh are delegated to golang.org/x/oauth2 rather than
// reimplemented here. Heavier, provider-specific integrations (for example GCP
// agent identity) live in subpackages so this package stays dependency-light.
package auth
