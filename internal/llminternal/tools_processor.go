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

package llminternal

import (
	"fmt"
	"iter"
	"slices"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// toolProcessor populates f.Tools on every step. A Toolset is resolved once
// per invocation and its tools reused for the invocation's later steps,
// unless it opts out through toolinternal.InvocationCacheDisabler because
// its tools depend on state that an earlier step of the same Run() may have
// changed; such a toolset is re-evaluated before each model call.
func toolProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent, ok := ctx.Agent().(Agent)
		if !ok {
			yield(nil, fmt.Errorf("agent %v is not an LLMAgent", ctx.Agent().Name()))
			return
		}
		toolsets := Reveal(llmAgent).Toolsets
		if f.toolsets.invocationID != ctx.InvocationID() || len(f.toolsets.entries) != len(toolsets) {
			f.toolsets = toolsetCache{invocationID: ctx.InvocationID(), entries: make([]toolsetCacheEntry, len(toolsets))}
		}
		tools := slices.Clip(Reveal(llmAgent).Tools)
		for i, toolSet := range toolsets {
			entry := &f.toolsets.entries[i]
			if !entry.resolved {
				tsTools, err := toolSet.Tools(icontext.NewReadonlyContext(ctx))
				if err != nil {
					yield(nil, fmt.Errorf("failed to extract tools from the tool set %q: %w", toolSet.Name(), err))
					return
				}
				entry.tools = tsTools
				entry.resolved = !toolinternal.DisablesInvocationCache(toolSet)
			}
			tools = append(tools, entry.tools...)
		}
		f.Tools = tools
	}
}
