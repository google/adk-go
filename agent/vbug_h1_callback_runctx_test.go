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

// FINDING H1 — NewCallbackContext is misused as a run context, dropping the agent.
//
// Bug: when a plugin's OnUserMessage modifies the user message, the runner
// rebuilds an InvocationContext carrying the real Agent (and Session/RunConfig)
// and then wraps it with NewCallbackContext, using the *result* as the run
// context. But NewCallbackContext returns a callbackContextWrapper whose
// run-critical accessors (Agent(), Session(), RunConfig(), IsolationScope(),
// ...) ignore the underlying context and simply log and return nil/"". So even
// though the underlying InvocationContext has a real agent, the run context's
// Agent() is nil, and downstream code such as ctx.Agent().Name() nil-derefs.
//
// Expected: a context derived from an InvocationContext that has a real agent
// must expose that agent (Agent() != nil), not silently drop it.
//
// Note: this is asserted directly on the wrapper because a white-box
// (package agent) test cannot import the runner's internal context package
// (it imports agent, which would create an import cycle). The wrapper is the
// root cause, so we feed it a real agent and check it is preserved.
//
// This test currently FAILS, demonstrating the bug.

package agent

import (
	"testing"
)

// vbugH1MockIC is a minimal InvocationContext that exposes a real agent and
// RunConfig, mirroring what the runner passes to NewCallbackContext. All other
// methods are inherited from ContextMock (returning zero values).
type vbugH1MockIC struct {
	*ContextMock
	agent     Agent
	runConfig *RunConfig
}

func (m *vbugH1MockIC) Agent() Agent          { return m.agent }
func (m *vbugH1MockIC) RunConfig() *RunConfig { return m.runConfig }

func TestVbugH1_CallbackContextPreservesRunContext(t *testing.T) {
	realAgent, err := New(Config{Name: "root_agent"})
	if err != nil {
		t.Fatalf("New agent: %v", err)
	}

	ic := &vbugH1MockIC{
		ContextMock: &ContextMock{},
		agent:       realAgent,
		runConfig:   &RunConfig{},
	}

	// Sanity check: the underlying InvocationContext really does carry the agent
	// and run config that the runner placed there.
	if ic.Agent() == nil {
		t.Fatalf("test setup wrong: underlying ic.Agent() is nil")
	}
	if ic.RunConfig() == nil {
		t.Fatalf("test setup wrong: underlying ic.RunConfig() is nil")
	}

	ctx := NewCallbackContext(ic, nil)

	if ctx.Agent() == nil {
		t.Errorf("NewCallbackContext used as run context dropped the agent: Agent() = nil, want the underlying real agent")
	}
	if ctx.RunConfig() == nil {
		t.Errorf("NewCallbackContext used as run context dropped the run config: RunConfig() = nil, want the underlying *RunConfig")
	}
}
