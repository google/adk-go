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

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// ContentRequestProcessor populates the LLMRequest's Contents based on
// the InvocationContext that includes the previous events.
func toolProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if f.Tools != nil {
			return
		}
		llmAgent, ok := ctx.Agent().(Agent)
		if !ok {
			yield(nil, fmt.Errorf("agent %v is not an LLMAgent", ctx.Agent().Name()))
			return
		}
		tools := Reveal(llmAgent).Tools
		for _, toolSet := range Reveal(llmAgent).Toolsets {
			tsTools, err := toolSet.Tools(icontext.NewReadonlyContext(ctx))
			if err != nil {
				yield(nil, fmt.Errorf("failed to extract tools from the tool set %q: %w", toolSet.Name(), err))
				return
			}

			tools = append(tools, tsTools...)
		}
		f.Tools = tools
	}
}
