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

// Package context provides backward-compatibility aliases for context
// constructors that have moved to the agent package.
//
// New code should import google.golang.org/adk/agent and use
// agent.NewInvocationContext / agent.InvocationContextParams directly.
// These aliases exist to keep existing call sites compiling during the
// migration; they will be removed in a future release.
package context

import "google.golang.org/adk/agent"

// InvocationContextParams aliases agent.InvocationContextParams.
//
// Deprecated: import google.golang.org/adk/agent and use
// agent.InvocationContextParams instead.
type InvocationContextParams = agent.InvocationContextParams

// NewInvocationContext is an alias for agent.NewInvocationContext.
//
// Deprecated: import google.golang.org/adk/agent and call
// agent.NewInvocationContext directly.
var NewInvocationContext = agent.NewInvocationContext
