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
	"errors"
	"slices"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// ctHookToolset is a ctToolset with a request hook that appends the
// instruction "hook ran".
type ctHookToolset struct {
	ctToolset
}

func (s *ctHookToolset) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.SystemInstruction == nil {
		req.Config.SystemInstruction = &genai.Content{Role: genai.RoleUser}
	}
	req.Config.SystemInstruction.Parts = append(req.Config.SystemInstruction.Parts, genai.NewPartFromText("hook ran"))
	return nil
}

func (m *canonLLM) recordedRequests() []*model.LLMRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.requests)
}

// declaredNames returns the names of every function declaration of req, in
// order.
func declaredNames(req *model.LLMRequest) []string {
	names := []string{}
	if req.Config == nil {
		return names
	}
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

func onlyRequest(t *testing.T, m *canonLLM) *model.LLMRequest {
	t.Helper()
	reqs := m.recordedRequests()
	if len(reqs) != 1 {
		t.Fatalf("the model was called %d times, want 1", len(reqs))
	}
	return reqs[0]
}

func TestRun_DeclaredOrderWithAsToolset(t *testing.T) {
	m := &canonLLM{name: "m"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: m, Toolsets: declaredOrderToolsets(t)})

	events := runAgent(t, a)

	if got, want := declaredNames(onlyRequest(t, m)), []string{"plain_1", "ts1_a", "ts1_b", "ts2_a", "plain_2"}; !slices.Equal(got, want) {
		t.Errorf("the model was offered %v, want %v", got, want)
	}
	if got := finalModelText(events); got != "ok" {
		t.Errorf("final text = %q, want %q", got, "ok")
	}
}

func TestRun_SkipsFailingToolsetAndFinishes(t *testing.T) {
	m := &canonLLM{name: "m"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: m, Toolsets: []tool.Toolset{
		&ctToolset{name: "failing", err: errors.New("mcp down")},
		&ctToolset{name: "working", tools: []tool.Tool{ctTool(t, "good_tool")}},
	}})

	events := runAgent(t, a)

	for _, ev := range events {
		if ev.ErrorCode != "" || ev.ErrorMessage != "" {
			t.Errorf("event with error %q: %q, want no error event", ev.ErrorCode, ev.ErrorMessage)
		}
	}
	if got := declaredNames(onlyRequest(t, m)); !slices.Equal(got, []string{"good_tool"}) {
		t.Errorf("the model was offered %v, want [good_tool]", got)
	}
	if got := finalModelText(events); got != "ok" {
		t.Errorf("final text = %q, want %q", got, "ok")
	}
}

func TestRun_SkippedToolsetStillRunsItsRequestHook(t *testing.T) {
	m := &canonLLM{name: "m"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: m, Toolsets: []tool.Toolset{
		&ctHookToolset{ctToolset{name: "failing_with_hook", err: errors.New("mcp down")}},
		&ctToolset{name: "working", tools: []tool.Tool{ctTool(t, "good_tool")}},
	}})

	runAgent(t, a)

	req := onlyRequest(t, m)
	var instruction strings.Builder
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, p := range req.Config.SystemInstruction.Parts {
			instruction.WriteString(p.Text)
		}
	}
	if n := strings.Count(instruction.String(), "hook ran"); n != 1 {
		t.Errorf("the system instruction %q holds %q %d times, want once", instruction.String(), "hook ran", n)
	}
	if got := declaredNames(req); !slices.Equal(got, []string{"good_tool"}) {
		t.Errorf("the model was offered %v, want [good_tool]", got)
	}
}

func TestRun_CancelledContextEndsTheStepWhenAToolsetFails(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	m := &canonLLM{name: "m"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: m, Toolsets: []tool.Toolset{
		&ctToolset{name: "cancelling", err: errors.New("listing broke"), onList: func(agent.ReadonlyContext) { cancel() }},
	}})
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatal(err)
	}

	var runErr error
	for _, err := range r.Run(ctx, "u", resp.Session.ID(), genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil && runErr == nil {
			runErr = err
		}
	}

	if !errors.Is(runErr, context.Canceled) {
		t.Errorf("Run error = %v, want one wrapping context.Canceled", runErr)
	}
	if n := len(m.recordedRequests()); n != 0 {
		t.Errorf("the model was called %d times, want 0", n)
	}
}

func TestRun_ListsToolsetsConcurrently(t *testing.T) {
	first, second := ctBarrier(t)
	m := &canonLLM{name: "m"}
	a := newLLMAgent(t, llmagent.Config{Name: "a", Model: m, Toolsets: []tool.Toolset{
		&ctToolset{name: "first", tools: []tool.Tool{ctTool(t, "first_tool")}, onList: first},
		&ctToolset{name: "second", tools: []tool.Tool{ctTool(t, "second_tool")}, onList: second},
	}})

	events := runAgent(t, a)

	if got := finalModelText(events); got != "ok" {
		t.Errorf("final text = %q, want %q", got, "ok")
	}
}
