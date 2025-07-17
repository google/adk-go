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

package runner

import (
	"context"
	"iter"

	"github.com/google/adk-go"

	"google.golang.org/genai"
)

type Runner struct {
	AppName        string
	Agent          adk.Agent
	SessionService adk.SessionService
}

// Run runs the agent.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *adk.AgentRunConfig) (iter.Seq2[*adk.Event, error], error) {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	//
	// For example, support_cfc requires Agent to be LLMAgent.
	// Note that checking that directly in this package results in circular dependency.
	// Options to consider:
	//     - Move Runner to a separate package (runner imports adk, agent. agent imports adk).
	//     - Require Agent.Validate method.
	//     - Wait until Agent.Run is called.
	/*
		// TODO: setup tracer.
		session, err := r.SessionService.Create(r.AppName, userID, nil)
		if err != nil {
			return nil, err
		}
		invocationCtx := r.newInvocationContext(ctx, session, msg, cfg)
		...
	*/
	panic("unimplemened")
}
