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
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// orderTool records "tool:<name>" in rec when its request hook runs, then
// packs itself.
type orderTool struct {
	name string
	rec  *[]string
}

func (o *orderTool) Name() string        { return o.name }
func (o *orderTool) Description() string { return "records its request hook" }
func (o *orderTool) IsLongRunning() bool { return false }
func (o *orderTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: o.name}
}

func (o *orderTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	*o.rec = append(*o.rec, "tool:"+o.name)
	return toolutils.PackTool(req, o)
}

// orderToolset records "hook:<name>" in rec when its request hook runs and
// returns err from it. Its n-th listing is listings[n], and the last entry
// once the steps outnumber the listings.
type orderToolset struct {
	name     string
	rec      *[]string
	listings [][]tool.Tool
	err      error
	calls    int
}

func (o *orderToolset) Name() string { return o.name }

func (o *orderToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	i := min(o.calls, len(o.listings)-1)
	o.calls++
	return o.listings[i], nil
}

func (o *orderToolset) ProcessRequest(agent.Context, *model.LLMRequest) error {
	*o.rec = append(*o.rec, "hook:"+o.name)
	return o.err
}

// newOrderStep returns a function that runs one step's request processing,
// toolProcessor and then the tool and toolset hooks, on one Flow for an agent
// with tools and toolsets, and returns the step's first error.
func newOrderStep(t *testing.T, tools []tool.Tool, toolsets []tool.Toolset) func() error {
	t.Helper()
	a := &mockLLMAgent{
		Agent: utils.Must(agent.New(agent.Config{Name: "order_agent"})),
		s:     &State{Tools: tools, Toolsets: toolsets},
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	f := &Flow{RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{toolProcessor}}
	return func() error {
		for _, err := range f.preprocess(ctx, &model.LLMRequest{}) {
			if err != nil {
				return err
			}
		}
		return nil
	}
}

func TestPreprocess_ToolsetHookRunsBeforeItsOwnTools(t *testing.T) {
	var rec []string
	a := &orderToolset{name: "A", rec: &rec, listings: [][]tool.Tool{{&orderTool{name: "a1", rec: &rec}, &orderTool{name: "a2", rec: &rec}}}}
	b := &orderToolset{name: "B", rec: &rec, listings: [][]tool.Tool{{&orderTool{name: "b1", rec: &rec}}}}
	step := newOrderStep(t, []tool.Tool{&orderTool{name: "own", rec: &rec}}, []tool.Toolset{a, b})

	if err := step(); err != nil {
		t.Fatalf("preprocess() error = %v", err)
	}

	want := []string{"tool:own", "hook:A", "tool:a1", "tool:a2", "hook:B", "tool:b1"}
	if diff := cmp.Diff(want, rec); diff != "" {
		t.Errorf("request hook order mismatch (-want +got):\n%s", diff)
	}
}

func TestPreprocess_ToolsetWithEmptyListingKeepsItsGroup(t *testing.T) {
	var rec []string
	a := &orderToolset{name: "A", rec: &rec, listings: [][]tool.Tool{nil}}
	b := &orderToolset{name: "B", rec: &rec, listings: [][]tool.Tool{{&orderTool{name: "b1", rec: &rec}}}}
	step := newOrderStep(t, nil, []tool.Toolset{a, b})

	if err := step(); err != nil {
		t.Fatalf("preprocess() error = %v", err)
	}

	want := []string{"hook:A", "hook:B", "tool:b1"}
	if diff := cmp.Diff(want, rec); diff != "" {
		t.Errorf("request hook order mismatch (-want +got):\n%s", diff)
	}
}

// TestPreprocess_ToolsetGroupsFollowTheCurrentStep checks that the groups of
// one step are rebuilt from that step's listings, so a tool a toolset starts
// listing in step 2 is processed after that toolset's hook.
func TestPreprocess_ToolsetGroupsFollowTheCurrentStep(t *testing.T) {
	var rec []string
	ts := &orderToolset{name: "T", rec: &rec, listings: [][]tool.Tool{nil, {&orderTool{name: "x", rec: &rec}}}}
	step := newOrderStep(t, nil, []tool.Toolset{ts})

	if err := step(); err != nil {
		t.Fatalf("step 1: preprocess() error = %v", err)
	}
	if diff := cmp.Diff([]string{"hook:T"}, rec); diff != "" {
		t.Errorf("step 1: request hook order mismatch (-want +got):\n%s", diff)
	}

	rec = nil
	if err := step(); err != nil {
		t.Fatalf("step 2: preprocess() error = %v", err)
	}
	if diff := cmp.Diff([]string{"hook:T", "tool:x"}, rec); diff != "" {
		t.Errorf("step 2: request hook order mismatch (-want +got):\n%s", diff)
	}
}

func TestPreprocess_ToolsetHookErrorStopsLaterGroups(t *testing.T) {
	var rec []string
	boom := errors.New("boom")
	a := &orderToolset{name: "A", rec: &rec, err: boom, listings: [][]tool.Tool{{&orderTool{name: "a1", rec: &rec}}}}
	b := &orderToolset{name: "B", rec: &rec, listings: [][]tool.Tool{{&orderTool{name: "b1", rec: &rec}}}}
	step := newOrderStep(t, []tool.Tool{&orderTool{name: "own", rec: &rec}}, []tool.Toolset{a, b})

	err := step()

	if err == nil || err.Error() != `process request by toolset "A": boom` {
		t.Errorf("preprocess() error = %v, want %q", err, `process request by toolset "A": boom`)
	}
	if !errors.Is(err, boom) {
		t.Errorf("errors.Is(%v, boom) = false, want true", err)
	}
	if diff := cmp.Diff([]string{"tool:own", "hook:A"}, rec); diff != "" {
		t.Errorf("request hook order mismatch (-want +got):\n%s", diff)
	}
}

// failingOrderTool records "tool:<name>" in rec when its request hook runs and
// returns err from it.
type failingOrderTool struct {
	orderTool
	err error
}

func (f *failingOrderTool) ProcessRequest(agent.Context, *model.LLMRequest) error {
	*f.rec = append(*f.rec, "tool:"+f.name)
	return f.err
}

func TestPreprocess_OwnToolHookErrorStopsTheStep(t *testing.T) {
	var rec []string
	boom := errors.New("boom")
	own := &failingOrderTool{orderTool: orderTool{name: "own", rec: &rec}, err: boom}
	a := &orderToolset{name: "A", rec: &rec, listings: [][]tool.Tool{{&orderTool{name: "a1", rec: &rec}}}}
	step := newOrderStep(t, []tool.Tool{own}, []tool.Toolset{a})

	err := step()

	if !errors.Is(err, boom) || err.Error() != "boom" {
		t.Errorf("preprocess() error = %v, want %v unwrapped", err, boom)
	}
	if diff := cmp.Diff([]string{"tool:own"}, rec); diff != "" {
		t.Errorf("request hook order mismatch (-want +got):\n%s", diff)
	}
}

// TestPreprocess_SameToolAsOwnAndFromToolsetStillFailsAsDuplicate checks that
// a tool that is both one of the agent's own tools and listed by a toolset is
// processed in both groups, so the second registration fails.
func TestPreprocess_SameToolAsOwnAndFromToolsetStillFailsAsDuplicate(t *testing.T) {
	var rec []string
	shared := &orderTool{name: "t", rec: &rec}
	ts := &orderToolset{name: "T", rec: &rec, listings: [][]tool.Tool{{shared}}}
	step := newOrderStep(t, []tool.Tool{shared}, []tool.Toolset{ts})

	err := step()

	if err == nil || !strings.Contains(err.Error(), `duplicate tool: "t"`) {
		t.Errorf("preprocess() error = %v, want one containing %q", err, `duplicate tool: "t"`)
	}
}
