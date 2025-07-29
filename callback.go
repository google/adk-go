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

package adk

import (
	"context"

	"google.golang.org/genai"
)

// BeforeAgentCallback is a callback function called before agent's run.
// When the content is present, the agent run will be **skipped** and the
// provided content will be returned to user.
type BeforeAgentCallback func(ctx context.Context, callbackCtx *CallbackContext) *genai.Content

// AfterAgentCallback is a callback function called after agent's run.
// When the content is present, the provided content will be used as agent
// response and appended to event history as agent response.
type AfterAgentCallback func(ctx context.Context, callbackCtx *CallbackContext, content *genai.Content) *genai.Content

// CallbackContext is the context for an agent callback invocation.
type CallbackContext struct {
	// The invocation context of the callback call.
	InvocationContext *InvocationContext

	// The event actions of the current callback call.
	EventActions *EventActions
}
