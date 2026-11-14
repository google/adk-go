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
	"slices"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
)

// closeOrder records the names of closed toolsets in the order Close reached
// them.
type closeOrder struct {
	mu    sync.Mutex
	names []string
}

func (o *closeOrder) add(name string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.names = append(o.names, name)
}

func (o *closeOrder) get() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Clone(o.names)
}

// orderRecordingToolset adds its name to order when closed.
type orderRecordingToolset struct {
	name  string
	order *closeOrder
}

func (r *orderRecordingToolset) Name() string                                     { return r.name }
func (r *orderRecordingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) { return nil, nil }

func (r *orderRecordingToolset) Close() error { r.order.add(r.name); return nil }

// newCloseAgent returns an LLM agent named name with the given toolsets.
func newCloseAgent(t *testing.T, name string, toolsets ...tool.Toolset) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: name, Toolsets: toolsets})
	if err != nil {
		t.Fatalf("llmagent.New(%q) error = %v", name, err)
	}
	return a
}

// closeRunnerOf builds a runner over root and closes it.
func closeRunnerOf(t *testing.T, root agent.Agent) {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "close_test", Agent: root, SessionService: session.InMemoryService()})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestRunner_Close_ReachesAgentToolChildPlacedThroughAsToolset(t *testing.T) {
	own, childOwn := &closeCountingToolset{}, &closeCountingToolset{}
	child := newCloseAgent(t, "child", childOwn)
	root := newCloseAgent(t, "root", own, tool.AsToolset(agenttool.New(child, nil)))

	closeRunnerOf(t, root)

	if got := childOwn.closes.Load(); got != 1 {
		t.Errorf("closes of the agent tool child's toolset = %d, want 1", got)
	}
	if got := own.closes.Load(); got != 1 {
		t.Errorf("closes of the root's own toolset = %d, want 1", got)
	}
}

func TestRunner_Close_ClosesOwnToolsetsBeforeAnAgentToolChild(t *testing.T) {
	order := &closeOrder{}
	child := newCloseAgent(t, "child", &orderRecordingToolset{name: "child_own", order: order})
	// The agent tool comes first, so a walk that descends into it while
	// looping over the root's toolsets closes the child's toolset first.
	root := newCloseAgent(t, "root",
		tool.AsToolset(agenttool.New(child, nil)),
		&orderRecordingToolset{name: "root_own", order: order},
	)

	closeRunnerOf(t, root)

	if got, want := order.get(), []string{"root_own", "child_own"}; !slices.Equal(got, want) {
		t.Errorf("close order = %v, want %v", got, want)
	}
}

func TestRunner_Close_ReachesAgentToolChildThroughWrappedAsToolset(t *testing.T) {
	childOwn := &closeCountingToolset{}
	child := newCloseAgent(t, "child", childOwn)
	root := newCloseAgent(t, "root", tool.FilterToolset(
		tool.AsToolset(agenttool.New(child, nil)),
		tool.AllowedToolsPredicate([]string{"child"}),
	))

	closeRunnerOf(t, root)

	if got := childOwn.closes.Load(); got != 1 {
		t.Errorf("closes of the agent tool child's toolset = %d, want 1", got)
	}
}
