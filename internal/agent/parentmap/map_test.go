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

package parentmap_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

func TestNew(t *testing.T) {
	child1_1 := utils.Must(agent.New(agent.Config{
		Name: "child1_1",
	}))

	child1 := utils.Must(agent.New(agent.Config{
		Name:      "child1",
		SubAgents: []agent.Agent{child1_1},
	}))

	child2 := utils.Must(agent.New(agent.Config{
		Name: "child2",
	}))

	root := utils.Must(agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{child1, child2},
	}))

	got, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}
	want := parentmap.Map{
		child1_1.Name(): child1,
		child1.Name():   root,
		child2.Name():   root,
	}

	agentNames := cmp.Transformer("agentNames", func(m parentmap.Map) map[string]string {
		if m == nil {
			return nil
		}
		res := make(map[string]string)
		for k, v := range m {
			res[k] = v.Name()
		}
		return res
	})

	if diff := cmp.Diff(want, got, agentNames); diff != "" {
		t.Errorf("New() = %v, got %v diff (-want/+got): %v", got, want, diff)
	}
}

func TestMap_RootAgent(t *testing.T) {
	model := struct {
		model.LLM
	}{}

	nonLLM := utils.Must(agent.New(agent.Config{
		Name: "mock",
	}))
	b := utils.Must(llmagent.New(llmagent.Config{
		Name:      "b",
		Model:     model,
		SubAgents: []agent.Agent{nonLLM},
	}))
	a := utils.Must(llmagent.New(llmagent.Config{
		Name:      "a",
		Model:     model,
		SubAgents: []agent.Agent{b},
	}))
	root := utils.Must(llmagent.New(llmagent.Config{
		Name:      "root",
		Model:     model,
		SubAgents: []agent.Agent{a},
	}))

	agentName := func(a agent.Agent) string {
		if a == nil {
			return "nil"
		}
		return a.Name()
	}

	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		agent agent.Agent
		want  agent.Agent
	}{
		{root, root},
		{a, root},
		{b, root},
		{nonLLM, root},
		{nil, nil},
	} {
		t.Run("agent="+agentName(tc.agent), func(t *testing.T) {
			gotRoot := parents.RootAgent(tc.agent)
			if got, want := agentName(gotRoot), agentName(tc.want); got != want {
				t.Errorf("rootAgent(%q) = %q, want %q", agentName(tc.agent), got, want)
			}
		})
	}
}

type cyclicParentMapCtx struct {
	parent context.Context
}

func (c *cyclicParentMapCtx) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cyclicParentMapCtx) Done() <-chan struct{}       { return nil }
func (c *cyclicParentMapCtx) Err() error                  { return nil }
func (c *cyclicParentMapCtx) Value(key any) any {
	if v, ok := adkcontext.ParentMapStep(key, c.parent); ok {
		return v
	}
	if c.parent == nil {
		return nil
	}
	return c.parent.Value(key)
}

func TestFromContext_SurvivesDeepADKDerivations(t *testing.T) {
	child := utils.Must(agent.New(agent.Config{Name: "child"}))
	root := utils.Must(agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{child},
	}))
	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatalf("parentmap.New: %v", err)
	}

	type unrelatedKey struct{}

	for _, hops := range []int{1, 63, 64, 65, 200} {
		t.Run(fmt.Sprintf("adk_hops_%d", hops), func(t *testing.T) {
			ctx := parentmap.ToContext(t.Context(), parents)
			for i := range hops {
				ctx = context.WithValue(ctx, unrelatedKey{}, i)
				ctx = icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: root})
			}
			got := parentmap.FromContext(ctx)
			if got == nil {
				t.Fatalf("parentmap.FromContext after %d ADK hops = nil, want map", hops)
			}
			if got[child.Name()] != root {
				t.Fatalf("got[%q] = %v, want %q", child.Name(), got[child.Name()], root.Name())
			}
			// Verify adkcontext.ReadParentMap remains hop-bounded at 64 ADK hops
			// (ToContext attaches &parentMapCtx inside context.WithValue, which adds
			// 1 step on ParentMapKey, so hops <= 64 reach it and hops >= 65 stop).
			analyticsRaw := adkcontext.ReadParentMap(ctx)
			if hops <= 64 {
				if analyticsRaw == nil {
					t.Fatalf("adkcontext.ReadParentMap after %d ADK hops = nil, want map", hops)
				}
			} else if analyticsRaw != nil {
				t.Fatalf("adkcontext.ReadParentMap after %d ADK hops = %v, want nil (hop budget exceeded)", hops, analyticsRaw)
			}
		})
	}

	t.Run("analytics_reader_terminates_on_adk_cycle", func(t *testing.T) {
		cyc := &cyclicParentMapCtx{}
		cyc.parent = cyc
		if got := adkcontext.ReadParentMap(cyc); got != nil {
			t.Fatalf("adkcontext.ReadParentMap(cyclic) = %v, want nil", got)
		}
	})
}
