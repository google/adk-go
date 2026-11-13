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

package runner_test

import (
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
)

// closeCountingToolset counts its Close calls. The runner closes a toolset in
// a goroutine of its own, so the count is atomic.
type closeCountingToolset struct {
	closes atomic.Int32
}

func (c *closeCountingToolset) Name() string                                     { return "counting" }
func (c *closeCountingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }
func (c *closeCountingToolset) Close() error                                     { c.closes.Add(1); return nil }

// newAgentToolParent returns an LLM agent whose only tool is an agent tool
// over a child LLM agent holding ts.
func newAgentToolParent(t *testing.T, ts tool.Toolset) (parent, child agent.Agent) {
	t.Helper()
	child, err := llmagent.New(llmagent.Config{Name: "child", Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatalf("llmagent.New(child) error = %v", err)
	}
	parent, err = llmagent.New(llmagent.Config{Name: "parent", Tools: []tool.Tool{agenttool.New(child, nil)}})
	if err != nil {
		t.Fatalf("llmagent.New(parent) error = %v", err)
	}
	return parent, child
}

func TestRunner_Close_ReachesAgentToolChild(t *testing.T) {
	ts := &closeCountingToolset{}
	parent, _ := newAgentToolParent(t, ts)
	r, err := runner.New(runner.Config{AppName: "close_test", Agent: parent, SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if got := ts.closes.Load(); got != 0 {
		t.Fatalf("closes of the child's toolset before Close = %d, want 0", got)
	}

	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if got := ts.closes.Load(); got != 1 {
		t.Errorf("closes of the child's toolset = %d, want 1", got)
	}
}

// TestRunner_Close_KeepsInheritedPlugins checks that closing a runner built
// the way agenttool builds its sub-runner (no PluginConfig) leaves the calling
// runner's plugins open. adk-python sets skip_closing_plugins on such a
// sub-runner (tools/agent_tool.py).
func TestRunner_Close_KeepsInheritedPlugins(t *testing.T) {
	var pluginCloses atomic.Int32
	p, err := plugin.New(plugin.Config{Name: "p", CloseFunc: func() error { pluginCloses.Add(1); return nil }})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	ts := &closeCountingToolset{}
	parentAgent, child := newAgentToolParent(t, ts)
	parent, err := runner.New(runner.Config{
		AppName:        "close_test",
		Agent:          parentAgent,
		SessionService: session.InMemoryService(),
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New(parent) error = %v", err)
	}
	sub, err := runner.New(runner.Config{AppName: child.Name(), Agent: child, SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("runner.New(sub) error = %v", err)
	}

	if err := sub.Close(); err != nil {
		t.Fatalf("sub.Close() error = %v", err)
	}
	if got := pluginCloses.Load(); got != 0 {
		t.Errorf("plugin closes after sub.Close() = %d, want 0", got)
	}
	if got := ts.closes.Load(); got != 1 {
		t.Errorf("closes of the child's toolset after sub.Close() = %d, want 1", got)
	}

	if err := parent.Close(); err != nil {
		t.Fatalf("parent.Close() error = %v", err)
	}
	if got := pluginCloses.Load(); got != 1 {
		t.Errorf("plugin closes after parent.Close() = %d, want 1", got)
	}
}
