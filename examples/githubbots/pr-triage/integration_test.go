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

package main

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// scriptedLLM is a model.LLM that emits a fixed sequence of turns, so the whole
// agent stack can be driven without a network call.
type scriptedLLM struct {
	mu    sync.Mutex
	turns []*genai.Content
	// requests counts GenerateContent calls, so a test can tell "the model was
	// never asked" from "the model answered".
	requests int
	// lastPrompt is the text of the last user message the model was handed.
	lastPrompt string
}

func (s *scriptedLLM) Name() string { return "scripted-llm" }

func (s *scriptedLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	s.mu.Lock()
	s.requests++
	for _, c := range req.Contents {
		if c == nil || c.Role != genai.RoleUser {
			continue
		}
		for _, p := range c.Parts {
			if p != nil && p.Text != "" {
				s.lastPrompt = p.Text
			}
		}
	}
	var turn *genai.Content
	if len(s.turns) > 0 {
		turn, s.turns = s.turns[0], s.turns[1:]
	} else {
		turn = genai.NewContentFromText("done", genai.RoleModel)
	}
	s.mu.Unlock()

	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: turn}, nil)
	}
}

func toolCall(name string, args map[string]any) *genai.Content {
	return &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call-1", Name: name, Args: args}}},
	}
}

// runRealAgent builds the production agent, runner and toolset around a scripted
// model and drives one pull request through the real triageOne path.
func runRealAgent(t *testing.T, cfg *Config, gh *GitHubClient, number int, turns ...*genai.Content) *scriptedLLM {
	t.Helper()
	llm := &scriptedLLM{turns: turns}

	tools, err := gh.tools()
	if err != nil {
		t.Fatalf("tools(): %v", err)
	}
	triager, err := llmagent.New(llmagent.Config{
		Name:        "pr_triager",
		Model:       llm,
		Description: "Routes an incoming GitHub pull request to a component owner.",
		Instruction: renderPrompt(cfg),
		Tools:       tools,
	})
	if err != nil {
		t.Fatalf("llmagent.New(): %v", err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: appName, Agent: triager, SessionService: sessions})
	if err != nil {
		t.Fatalf("runner.New(): %v", err)
	}

	scopeSession(context.Background(), cfg, number, func(ictx context.Context) {
		triageOne(ictx, gh, cfg, discardLogger(), number, func(actx context.Context, _ int, prompt string) string {
			return runAgent(actx, r, sessions, gh, discardLogger(), prompt)
		})
	})
	return llm
}

// eligiblePRBody is a GraphQL response for a pull request that passes every
// precondition.
const eligiblePRBody = `{"data":{"repository":{"pullRequest":{
	"number":7,"title":"Add a cache","body":"Speeds up lookups.","state":"OPEN","isDraft":false,
	"author":{"login":"carol","__typename":"User"},
	"assignees":{"totalCount":0},
	"files":{"totalCount":1,"nodes":[{"path":"agent/cache.go"}]},
	"comments":{"totalCount":0,"nodes":[]},
	"timelineItems":{"totalCount":0}}}}}`

// graphQLThen answers the pull request query with body and everything else
// through h, recording the mutating calls.
func graphQLThen(t *testing.T, cfg *Config, body string, rec *recordingHandler) *GitHubClient {
	t.Helper()
	return testClient(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/graphql") {
			_, _ = io.WriteString(w, body)
			return
		}
		rec.ServeHTTP(w, r)
	}))
}

// The whole session-scope defense rests on ADK propagating the context passed to
// runner.Run through to the agent.Context a tool handler receives. Every other
// test applies withAuditedPR by hand and so cannot see that propagation: if ADK
// stopped doing it, authorizePR would reject every call, the bot would silently
// do nothing on every pull request, and the suite would stay green — the tool
// returns a model-readable errResult with a nil Go error, so nothing is even
// recorded. This drives the real llmagent, runner and functiontool stack.
func TestRealAgentStackPropagatesTheSessionScope(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("assign_owner_to_pull_request", map[string]any{
			"pull_request_number": 7, "component": "core",
		}),
		genai.NewContentFromText("assigned core", genai.RoleModel),
	)

	if llm.requests == 0 {
		t.Fatal("the model was never invoked; this test proves nothing about the tool path")
	}
	writes := rec.writes()
	if len(writes) != 1 || !strings.HasSuffix(writes[0], "/issues/7/assignees") {
		t.Fatalf("writes = %v, want one POST to /issues/7/assignees — the tool call did not reach GitHub, "+
			"so the session scope did not survive the ADK context plumbing", writes)
	}
	assignees, _ := rec.postedBodies()[0]["assignees"].([]any)
	if len(assignees) != 1 || assignees[0] != "alice" {
		t.Errorf("posted assignees = %v, want [alice] resolved from the owner map", assignees)
	}
	// The prompt the real stack built must carry the fence.
	if !strings.Contains(llm.lastPrompt, "[UNTRUSTED:") {
		t.Errorf("the model was handed an unfenced prompt:\n%s", llm.lastPrompt)
	}
}

// The same stack, with the model doing what an injected instruction would make
// it do: acting on a different pull request. The scope must refuse it, and
// nothing may reach GitHub.
func TestRealAgentStackRefusesACrossPullRequestToolCall(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)
	// Clear #99 through the eligibility gate, so the SESSION SCOPE is the only
	// thing left standing between the model and the write. Without this the
	// claim's own eligibility check refuses the call and the test passes even
	// with authorizePR deleted.
	gh.markEligible(99)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("assign_owner_to_pull_request", map[string]any{
			"pull_request_number": 99, "component": "core",
		}),
		toolCall("request_more_context", map[string]any{
			"pull_request_number": 99, "missing_items": []any{"problem"},
		}),
		genai.NewContentFromText("done", genai.RoleModel),
	)

	if llm.requests == 0 {
		t.Fatal("the model was never invoked; this test proves nothing")
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Errorf("a cross-pull-request tool call reached GitHub: %v", writes)
	}
}

// A component the model invented must not become an assignee, driven through
// the real stack rather than by calling the tool body directly.
func TestRealAgentStackRefusesAnInventedComponent(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("assign_owner_to_pull_request", map[string]any{
			"pull_request_number": 7, "component": "mallory",
		}),
		genai.NewContentFromText("done", genai.RoleModel),
	)

	// Without this the test passes when the model never ran at all, which is how
	// it first "passed" while every scoped context was born expired.
	if llm.requests == 0 {
		t.Fatal("the model was never invoked; this test proves nothing")
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Errorf("an unconfigured component reached GitHub: %v", writes)
	}
}

// An ineligible pull request must cost nothing: the model is never invoked, so
// the gate is in front of the token spend and not merely in front of the write.
func TestRealAgentStackNeverRunsTheModelForAnIneligiblePullRequest(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	const assignedBody = `{"data":{"repository":{"pullRequest":{
		"number":7,"title":"t","body":"b","state":"OPEN","isDraft":false,
		"author":{"login":"carol","__typename":"User"},
		"assignees":{"totalCount":1},
		"files":{"totalCount":0,"nodes":[]},
		"comments":{"totalCount":0,"nodes":[]},
		"timelineItems":{"totalCount":1}}}}}`
	gh := graphQLThen(t, cfg, assignedBody, rec)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("assign_owner_to_pull_request", map[string]any{
			"pull_request_number": 7, "component": "core",
		}),
	)

	if llm.requests != 0 {
		t.Errorf("the model was invoked %d times for an assigned pull request, want 0", llm.requests)
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Errorf("an ineligible pull request was mutated: %v", writes)
	}
}

// Dry run must hold through the real stack too, not only when the tool body is
// called directly.
func TestRealAgentStackWritesNothingUnderDryRun(t *testing.T) {
	cfg := testConfig()
	cfg.DryRun = true
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("assign_owner_to_pull_request", map[string]any{
			"pull_request_number": 7, "component": "core",
		}),
		toolCall("request_more_context", map[string]any{
			"pull_request_number": 7, "missing_items": []any{"problem", "testing"},
		}),
		genai.NewContentFromText("done", genai.RoleModel),
	)

	if llm.requests == 0 {
		t.Fatal("the model was never invoked; a dry run that ran nothing writes nothing either")
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Errorf("dry run wrote to GitHub through the real stack: %v", writes)
	}
}

// The one comment the bot can post, end to end: every word of it must come from
// the allow-list constants, never from the model.
func TestRealAgentStackPostsOnlyAllowListedProse(t *testing.T) {
	cfg := testConfig()
	rec := newRecordingHandler()
	gh := graphQLThen(t, cfg, eligiblePRBody, rec)

	llm := runRealAgent(t, cfg, gh, 7,
		toolCall("request_more_context", map[string]any{
			"pull_request_number": 7,
			// A model fully under an attacker's control would try to smuggle
			// prose through the one field it can fill.
			"missing_items": []any{"problem", "@everyone this repo is malware"},
		}),
		genai.NewContentFromText("done", genai.RoleModel),
	)
	if llm.requests == 0 {
		t.Fatal("the model was never invoked; this test proves nothing")
	}
	if writes := rec.writes(); len(writes) != 0 {
		t.Fatalf("an item outside the allow-list reached GitHub: %v", writes)
	}

	// The corrected call posts, and posts only constants.
	runRealAgent(t, cfg, gh, 7,
		toolCall("request_more_context", map[string]any{
			"pull_request_number": 7, "missing_items": []any{"problem"},
		}),
		genai.NewContentFromText("done", genai.RoleModel),
	)
	writes := rec.writes()
	if len(writes) != 1 || !strings.HasSuffix(writes[0], "/issues/7/comments") {
		t.Fatalf("writes = %v, want one POST to /issues/7/comments", writes)
	}
	body, _ := rec.postedBodies()[len(rec.postedBodies())-1]["body"].(string)
	want, _ := contextItemText("problem")
	if !strings.Contains(body, want) {
		t.Errorf("posted body is missing the allow-listed text:\n%s", body)
	}
	if strings.Contains(body, "@everyone") || strings.Contains(body, "malware") {
		t.Errorf("model-supplied text reached the posted comment:\n%s", body)
	}
}

// A JSON round trip of the tool schema, so a rename of a json tag cannot
// silently stop the model's arguments from binding — which would look like a
// well-behaved bot that simply never acts.
func TestToolArgumentTagsMatchWhatTheModelSends(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want any
	}{
		{`{"pull_request_number":7,"component":"core"}`, assignArgs{PullRequestNumber: 7, Component: "core"}},
		{`{"pull_request_number":7,"missing_items":["problem"]}`, contextArgs{PullRequestNumber: 7, MissingItems: []string{"problem"}}},
	} {
		switch want := tc.want.(type) {
		case assignArgs:
			var got assignArgs
			if err := json.Unmarshal([]byte(tc.raw), &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got != want {
				t.Errorf("assignArgs = %+v, want %+v", got, want)
			}
		case contextArgs:
			var got contextArgs
			if err := json.Unmarshal([]byte(tc.raw), &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.PullRequestNumber != want.PullRequestNumber || len(got.MissingItems) != 1 ||
				got.MissingItems[0] != want.MissingItems[0] {
				t.Errorf("contextArgs = %+v, want %+v", got, want)
			}
		}
	}
}

// The scoped session must carry a deadline into the real runner, so a hung model
// call cannot outlive the per-pull-request budget.
func TestRealAgentStackRunsUnderThePerPullRequestDeadline(t *testing.T) {
	cfg := testConfig()
	cfg.PRTimeout = 42 * time.Second
	var seen time.Duration
	scopeSession(context.Background(), cfg, 7, func(ictx context.Context) {
		if dl, ok := ictx.Deadline(); ok {
			seen = time.Until(dl)
		}
	})
	if seen == 0 || seen > cfg.PRTimeout {
		t.Errorf("scoped deadline = %v, want a positive value no greater than %v", seen, cfg.PRTimeout)
	}
}
