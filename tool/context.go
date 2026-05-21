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

package tool

import (
	"github.com/google/uuid"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// NewToolContext constructs a Context for a tool execution.
//
// If functionCallID is empty a new UUID is generated. If actions is nil a
// fresh session.EventActions with empty StateDelta and ArtifactDelta is
// allocated. The returned Context is backed by the same *callbackContext
// implementation used for CallbackContext, so all callback-context semantics
// (state delta tracking, artifact delta tracking, etc.) apply.
func NewToolContext(ic agent.InvocationContext, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) Context {
	if functionCallID == "" {
		functionCallID = uuid.NewString()
	}
	// agent.NewToolCallbackContext returns *agent.callbackContext (an
	// unexported type). Assigning it to the Context interface here is the
	// compile-time assertion that the method set satisfies tool.Context.
	return agent.NewToolCallbackContext(ic, functionCallID, actions, confirmation)
}
