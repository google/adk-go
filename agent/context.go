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

package agent

import (
	"context"

	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type InvocationContext interface {
	context.Context

	Artifacts() Artifacts
	Memory() Memory
	Session() session.Session

	InvocationID() string
	Branch() string
	Agent() Agent
	UserContent() *genai.Content
	RunConfig() *RunConfig

	EndInvocation()
	Ended() bool
}

type ReadonlyContext interface {
	context.Context

	UserContent() *genai.Content
	InvocationID() string
	AgentName() string
	ReadonlyState() session.ReadonlyState

	UserID() string
	AppName() string
	SessionID() string
	Branch() string
}

type CallbackContext interface {
	ReadonlyContext

	Artifacts() Artifacts
	State() session.State
}
