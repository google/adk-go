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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// NewToolContext constructs a ToolContext for a tool execution.
//
// Deprecated: use agent.NewToolContext directly. This wrapper exists only
// to minimize churn during the migration and will be removed in a future
// release.
func NewToolContext(ic agent.InvocationContext, functionCallID string, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) ToolContext {
	return agent.NewToolContext(ic, functionCallID, actions, confirmation)
}
