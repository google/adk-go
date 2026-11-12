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

package llmagent_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// orderSchema is a record with a titled nested record, so its default
// declaration is a JSON schema with $defs.
func orderSchema() *genai.Schema {
	return &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Order",
		Properties: map[string]*genai.Schema{
			"customer": {
				Type:       genai.TypeObject,
				Title:      "Customer",
				Properties: map[string]*genai.Schema{"email": {Type: genai.TypeString}},
			},
		},
		Required: []string{"customer"},
	}
}

// declarationForm names the form of the set_model_response declaration in
// req: "typed" for Parameters only, "json" for ParametersJsonSchema only.
func declarationForm(t *testing.T, req *model.LLMRequest) string {
	t.Helper()
	decl := declarationNamed(req, "set_model_response")
	switch {
	case decl == nil:
		t.Fatal("request has no set_model_response declaration")
	case decl.Parameters != nil && decl.ParametersJsonSchema == nil:
		return "typed"
	case decl.Parameters == nil && decl.ParametersJsonSchema != nil:
		return "json"
	}
	t.Fatalf("set_model_response declaration sets Parameters = %v and ParametersJsonSchema = %v", decl.Parameters, decl.ParametersJsonSchema)
	return ""
}

// T13. The form follows the context of each run, both ways, on one agent and
// one runner, as adk-python reads its flag for every request (u4590).
// Fails when the form is fixed when the agent or its first request is built,
// or read from a context other than the run's.
func TestLegacySwitch_ReadEveryRequest(t *testing.T) {
	m := &schemaModel{replies: []*genai.Content{textReply("done")}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{otherTool(t)}, OutputSchema: orderSchema()})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newTestRunner(t, a)
	bg := t.Context()
	runs := []struct {
		session string
		ctx     context.Context
	}{
		{"unset", bg},
		{"on", feature.WithLegacyFunctionDeclarationSchema(bg, true)},
		{"off", feature.WithLegacyFunctionDeclarationSchema(bg, false)},
	}

	for _, run := range runs {
		if _, err := runOnce(run.ctx, t, r, svc, run.session); err != nil {
			t.Fatalf("run %s: %v", run.session, err)
		}
	}

	reqs := m.requests()
	if len(reqs) != 3 {
		t.Fatalf("model got %d requests, want 3", len(reqs))
	}
	var got []string
	for _, req := range reqs {
		got = append(got, declarationForm(t, req))
	}
	if diff := cmp.Diff([]string{"json", "typed", "json"}, got); diff != "" {
		t.Errorf("declaration form per run mismatch (-want +got):\n%s", diff)
	}
}

// T14. Two runs at once, one with the switch and one without, each keep their
// own form. Each model holds its first reply until both runs have reached it,
// so both runs have started before either sends its second request.
// Fails when the switch lives anywhere shared between runs.
func TestLegacySwitch_ConcurrentRunsKeepTheirOwnValue(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	var (
		mu      sync.Mutex
		arrived int
		allIn   = make(chan struct{})
	)
	waitForBoth := func(ctx context.Context) {
		mu.Lock()
		arrived++
		if arrived == 2 {
			close(allIn)
		}
		mu.Unlock()
		select {
		case <-allIn:
		case <-ctx.Done():
			t.Error("the two runs never overlapped")
		}
	}
	newModel := func() *schemaModel {
		return &schemaModel{
			replies:          []*genai.Content{callReply("other_tool", map[string]any{}), textReply("done")},
			beforeFirstReply: waitForBoth,
		}
	}
	legacyModel, jsonModel := newModel(), newModel()
	run := func(m *schemaModel, ctx context.Context) error {
		a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{otherTool(t)}, OutputSchema: orderSchema()})
		if err != nil {
			return err
		}
		r, svc := newTestRunner(t, a)
		_, err = runOnce(ctx, t, r, svc, "s")
		return err
	}

	var wg sync.WaitGroup
	errs := make([]error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		errs[0] = run(legacyModel, feature.WithLegacyFunctionDeclarationSchema(ctx, true))
	}()
	go func() {
		defer wg.Done()
		errs[1] = run(jsonModel, ctx)
	}()
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
	}
	for _, tc := range []struct {
		name string
		m    *schemaModel
		want []string
	}{
		{"switch on", legacyModel, []string{"typed", "typed"}},
		{"switch unset", jsonModel, []string{"json", "json"}},
	} {
		var got []string
		for _, req := range tc.m.requests() {
			got = append(got, declarationForm(t, req))
		}
		if diff := cmp.Diff(tc.want, got); diff != "" {
			t.Errorf("%s: declaration form per request mismatch (-want +got):\n%s", tc.name, diff)
		}
	}
}

// T15. In the typed form the model can still call set_model_response, and the
// call ends the turn with its answer.
// Fails when the request registers a tool the flow cannot run, or one whose
// Run does not record the answer.
func TestLegacySwitch_SetModelResponseStillFinalizes(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"count": {Type: genai.TypeInteger}},
		Required:   []string{"count"},
	}
	m := &schemaModel{replies: []*genai.Content{
		callReply("set_model_response", map[string]any{"count": float64(10)}),
		textReply("unexpected second turn"),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "a", Model: m, Tools: []tool.Tool{otherTool(t)}, OutputSchema: schema})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r, svc := newTestRunner(t, a)

	events, err := runOnce(feature.WithLegacyFunctionDeclarationSchema(t.Context(), true), t, r, svc, "s")
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if got := declarationForm(t, m.requests()[0]); got != "typed" {
		t.Fatalf("declaration form = %q, want %q", got, "typed")
	}
	var stored any
	for _, ev := range events {
		if ev.Actions.SetModelResponse != nil {
			stored = ev.Actions.SetModelResponse
		}
	}
	if diff := cmp.Diff(map[string]any{"count": float64(10)}, stored); diff != "" {
		t.Errorf("Actions.SetModelResponse mismatch (-want +got):\n%s", diff)
	}
	if len(events) == 0 {
		t.Fatal("run yielded no events")
	}
	want := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: `{"count":10}`}}}
	if diff := cmp.Diff(want, events[len(events)-1].Content); diff != "" {
		t.Errorf("last event content mismatch (-want +got):\n%s", diff)
	}
}
