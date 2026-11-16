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

package runner

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
)

// checkCFCModel refuses a run whose RunConfig has SupportCFC when the runner's
// root agent is an LLM agent whose resolved model name does not start with
// "gemini-2". It is adk-python's check in Runner._new_invocation_context
// (runners.py:2158-2167), ported as it is: only the root agent is checked, a
// root that is not an LLM agent is not, and the test is a plain prefix test,
// so "models/gemini-2.0-flash" and "gemini-3-pro" fail.
//
// It returns nil when the check passes or does not apply. A root model that
// cannot be resolved gives an error wrapping llminternal.ErrModelNotConfigured.
func (r *Runner) checkCFCModel(ctx context.Context, cfg agent.RunConfig) error {
	if !cfg.SupportCFC {
		return nil
	}
	if _, ok := r.rootAgent.(llminternal.Agent); !ok {
		return nil
	}
	m, err := llminternal.ResolveModel(ctx, r.rootAgent)
	if err != nil {
		return fmt.Errorf("resolve the model of the root agent for CFC: %w", err)
	}
	if !strings.HasPrefix(m.Name(), "gemini-2") {
		return fmt.Errorf("CFC is not supported for model: %s in agent: %s", m.Name(), r.rootAgent.Name())
	}
	return nil
}
