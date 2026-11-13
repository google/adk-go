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
	"iter"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// LiveInjection holds additions that apply to one live invocation. A live
// caller passes them to [LiveRunnerWithInjection] instead of writing them to
// the agent's shared [State], so a concurrent non-live run cannot observe a
// partial write and the additions do not leak into later runs.
type LiveInjection struct {
	// Tool is added to the agent's tools for this invocation. Nil adds no tool.
	Tool tool.Tool
	// InstructionSuffix is appended to the agent's instruction for this
	// invocation. Empty adds no suffix.
	InstructionSuffix string
}

// LiveRunnerWithInjection is implemented by LLM agents that can apply
// per-invocation additions to a live run.
type LiveRunnerWithInjection interface {
	RunLiveWithInjection(ctx agent.InvocationContext, injection LiveInjection) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}
