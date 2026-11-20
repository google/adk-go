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
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// Each fakeModel request is attributed to the declared agent or llm node whose
// instruction contains the marker.
var instructionMarkers = map[string]string{
	"You are the Researcher.":              "research",
	"You are the Editor.":                  "shorten",
	"front desk of a small writing studio": "studio",
	"You are the Writer.":                  "writer",
	"You are the Clarity critic.":          "critic_clarity",
	"You are the Accuracy critic.":         "critic_accuracy",
	"You are the Style critic.":            "critic_style",
	"Classify the user's message":          "classify",
	"The user asked a question.":           "answer_question",
	"The user expressed strong emotion.":   "react_exclamation",
	"The user made a neutral statement.":   "comment_statement",
}

// fakeModel answers every request with reply, called with the name of the
// agent or node that sent it and the text of its last user turn. Calls to
// reply are serialized, so it may keep state; the parallel critics need that.
type fakeModel struct {
	reply func(caller, input string) *genai.Part

	mu    sync.Mutex
	calls []string // callers, in call order
}

func (*fakeModel) Name() string { return "fake" }

func (m *fakeModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		caller, err := callerOf(req)
		if err != nil {
			yield(nil, err)
			return
		}
		// Released before yield: an agent transfer calls the model again from
		// inside it.
		m.mu.Lock()
		m.calls = append(m.calls, caller)
		part := m.reply(caller, lastUserText(req))
		m.mu.Unlock()
		yield(&model.LLMResponse{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}},
			TurnComplete: true,
		}, nil)
	}
}

func (m *fakeModel) callers() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.calls)
}

func callerOf(req *model.LLMRequest) (string, error) {
	if req.Config != nil {
		instruction := contentText(req.Config.SystemInstruction)
		for marker, caller := range instructionMarkers {
			if strings.Contains(instruction, marker) {
				return caller, nil
			}
		}
	}
	return "", errors.New("request from an agent with no instruction marker")
}

func lastUserText(req *model.LLMRequest) string {
	for _, c := range slices.Backward(req.Contents) {
		if c != nil && c.Role == genai.RoleUser {
			if text := contentText(c); text != "" {
				return text
			}
		}
	}
	return ""
}

func loadTestSystem(t *testing.T, llm model.LLM) []agent.Agent {
	t.Helper()
	return loadDoc(t, agentSystem, llm)
}

// loadDoc builds the agents of doc with testRegistry.
func loadDoc(t *testing.T, doc []byte, llm model.LLM) []agent.Agent {
	t.Helper()
	agents, err := loadSystem(doc, testRegistry(t, llm))
	if err != nil {
		t.Fatalf("loadSystem() error = %v", err)
	}
	return agents
}

// testRegistry registers what main does: llm as the model default, and the
// tool word_count.
func testRegistry(t *testing.T, llm model.LLM) registry {
	t.Helper()
	wordCount, err := newWordCountTool()
	if err != nil {
		t.Fatalf("newWordCountTool() error = %v", err)
	}
	return registry{
		models: map[string]model.LLM{"default": llm},
		tools:  map[string]tool.Tool{"word_count": wordCount},
	}
}

// runAgent runs the declared agent name on prompt and returns the text of the
// last event that carries any.
func runAgent(t *testing.T, agents []agent.Agent, name, prompt string) string {
	t.Helper()
	texts := runTurns(t, agents, name, prompt)[0]
	if len(texts) == 0 {
		return ""
	}
	return texts[len(texts)-1]
}

// runTurns runs the declared agent name on each prompt in turn, in one
// session, and returns the text of every event of each run that carries any.
func runTurns(t *testing.T, agents []agent.Agent, name string, prompts ...string) [][]string {
	t.Helper()
	r := newRunner(t, agents, name)
	var runs [][]string
	for _, prompt := range prompts {
		runs = append(runs, eventTexts(runEvents(t, r, genai.NewContentFromText(prompt, genai.RoleUser))))
	}
	return runs
}

// newRunner returns a runner for the declared agent name, with its own
// session service.
func newRunner(t *testing.T, agents []agent.Agent, name string) *runner.Runner {
	t.Helper()
	i := slices.IndexFunc(agents, func(a agent.Agent) bool { return a.Name() == name })
	if i < 0 {
		t.Fatalf("agent %q not found", name)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: agents[i], SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r
}

// runEvents sends content to r, always in the same session, and returns the
// events of the run.
func runEvents(t *testing.T, r *runner.Runner, content *genai.Content) []*session.Event {
	t.Helper()
	events, err := tryRun(t, r, content)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	return events
}

// tryRun is runEvents for a run that may fail: it returns the events up to
// the error.
func tryRun(t *testing.T, r *runner.Runner, content *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var events []*session.Event
	for event, err := range r.Run(t.Context(), "user", "session", content, agent.RunConfig{}) {
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
	return events, nil
}

// eventTexts returns the text of every event that carries any.
func eventTexts(events []*session.Event) []string {
	var texts []string
	for _, event := range events {
		if text := contentText(event.Content); text != "" {
			texts = append(texts, text)
		}
	}
	return texts
}

func TestLoadSystem(t *testing.T) {
	agents := loadTestSystem(t, &fakeModel{})

	var got []string
	for _, a := range agents {
		got = append(got, a.Name())
	}
	want := []string{"studio", "writer_critics", "writer", "llm_router"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("agent names mismatch (-want +got):\n%s", diff)
	}
	if _, err := agent.NewMultiLoader(agents[0], agents[1:]...); err != nil {
		t.Errorf("NewMultiLoader() error = %v", err)
	}
}

// approvals are the critics' approving replies. They vary in case and
// whitespace the way model output does, which the decide node must tolerate.
var approvals = map[string]string{
	"critic_clarity":  "APPROVE",
	"critic_accuracy": "  approve\n",
	"critic_style":    "Approve. Reads well.",
}

// writerCriticsModel drafts "This is draft N." for the Nth writer call and
// lets approves decide each critic's verdict on the latest draft. Every
// writer input is recorded in writerInputs.
func writerCriticsModel(approves func(critic string, turn int) bool, writerInputs *[]string) func(caller, input string) *genai.Part {
	return func(caller, input string) *genai.Part {
		if caller == "writer" {
			*writerInputs = append(*writerInputs, input)
			return genai.NewPartFromText(fmt.Sprintf("This is draft %d.", len(*writerInputs)))
		}
		if approves(caller, len(*writerInputs)) {
			return genai.NewPartFromText(approvals[caller])
		}
		return genai.NewPartFromText(fmt.Sprintf("REVISE: fix %s in draft %d", caller, len(*writerInputs)))
	}
}

// TestWriterCritics checks the CEL decide node: it loops back to the writer
// until all three critics approve or the fifth turn, sends the writer every
// critic's feedback on each revision, and hands the final draft to publish.
func TestWriterCritics(t *testing.T) {
	tests := []struct {
		name string
		// approves reports whether critic approves draft number turn.
		approves  func(critic string, turn int) bool
		wantTurns int
		want      string
	}{
		{
			name:      "approved on first draft",
			approves:  func(string, int) bool { return true },
			wantTurns: 1,
			want:      "Final draft after 1 turn(s), 3/3 approvals:\n\nThis is draft 1.",
		},
		{
			name:      "approved on third draft",
			approves:  func(_ string, turn int) bool { return turn >= 3 },
			wantTurns: 3,
			want:      "Final draft after 3 turn(s), 3/3 approvals:\n\nThis is draft 3.",
		},
		{
			name:      "one critic never approves",
			approves:  func(critic string, _ int) bool { return critic != "critic_style" },
			wantTurns: 5,
			want:      "Final draft after 5 turn(s), 2/3 approvals:\n\nThis is draft 5.",
		},
		{
			name:      "no critic ever approves",
			approves:  func(string, int) bool { return false },
			wantTurns: 5,
			want:      "Final draft after 5 turn(s), 0/3 approvals:\n\nThis is draft 5.",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var writerInputs []string
			agents := loadTestSystem(t, &fakeModel{reply: writerCriticsModel(tc.approves, &writerInputs)})

			got := runAgent(t, agents, "writer_critics", "the history of the bicycle")

			if got != tc.want {
				t.Errorf("output = %q, want %q", got, tc.want)
			}
			if len(writerInputs) != tc.wantTurns {
				t.Fatalf("writer called %d times, want %d", len(writerInputs), tc.wantTurns)
			}
			if writerInputs[0] != "the history of the bicycle" {
				t.Errorf("first writer input = %q, want the user's topic", writerInputs[0])
			}
			for i, input := range writerInputs[1:] {
				draft := i + 1
				want := []string{fmt.Sprintf("DRAFT:\nThis is draft %d.", draft), "CRITIC FEEDBACK:"}
				for _, critic := range []string{"critic_clarity", "critic_accuracy", "critic_style"} {
					if !tc.approves(critic, draft) {
						want = append(want, fmt.Sprintf("REVISE: fix %s in draft %d", critic, draft))
					}
				}
				for _, w := range want {
					if !strings.Contains(input, w) {
						t.Errorf("writer input %d = %q, want it to contain %q", draft+1, input, w)
					}
				}
			}
		})
	}
}

// TestWriterCriticsCountsTurnsPerRequest checks that the turn limit applies to
// each request: a second request in the same session starts again at turn 1
// rather than continuing the first request's count.
func TestWriterCriticsCountsTurnsPerRequest(t *testing.T) {
	var writerInputs []string
	agents := loadTestSystem(t, &fakeModel{reply: writerCriticsModel(func(string, int) bool { return false }, &writerInputs)})

	var got []string
	for _, texts := range runTurns(t, agents, "writer_critics", "the history of the bicycle", "the history of the train") {
		got = append(got, texts[len(texts)-1])
	}

	want := []string{
		"Final draft after 5 turn(s), 0/3 approvals:\n\nThis is draft 5.",
		"Final draft after 5 turn(s), 0/3 approvals:\n\nThis is draft 10.",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("replies mismatch (-want +got):\n%s", diff)
	}
}

// TestLLMRouter checks that the CEL route node normalizes the classifier's
// label, falls back to DEFAULT for an off-script label, and forwards the
// user's message, not the label, to the chosen branch.
func TestLLMRouter(t *testing.T) {
	tests := []struct {
		label      string
		wantBranch string
	}{
		{label: "question", wantBranch: "answer_question"},
		{label: "  Exclamation\n", wantBranch: "react_exclamation"},
		{label: "statement", wantBranch: "comment_statement"},
		{label: "I am not sure", wantBranch: "comment_statement"},
	}
	for _, tc := range tests {
		t.Run(tc.label, func(t *testing.T) {
			m := &fakeModel{reply: func(caller, input string) *genai.Part {
				if caller == "classify" {
					return genai.NewPartFromText(tc.label)
				}
				return genai.NewPartFromText(caller + " saw: " + input)
			}}
			agents := loadTestSystem(t, m)

			got := runAgent(t, agents, "llm_router", "Is it raining?")

			if want := tc.wantBranch + " saw: Is it raining?"; got != want {
				t.Errorf("output = %q, want %q", got, want)
			}
			if diff := cmp.Diff([]string{"classify", tc.wantBranch}, m.callers()); diff != "" {
				t.Errorf("model callers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestCELNodeForwardsInput checks that a cel node without an output passes
// its input on unchanged: with route's output removed, the branch sees the
// classifier's label instead of the user's message.
func TestCELNodeForwardsInput(t *testing.T) {
	doc := editedSystem(t, "        output: user.content\n", "")
	m := &fakeModel{reply: func(caller, input string) *genai.Part {
		if caller == "classify" {
			return genai.NewPartFromText("question")
		}
		return genai.NewPartFromText(caller + " saw: " + input)
	}}
	agents, err := loadSystem(doc, testRegistry(t, m))
	if err != nil {
		t.Fatalf("loadSystem() error = %v", err)
	}

	got := runAgent(t, agents, "llm_router", "Is it raining?")

	if want := "answer_question saw: question"; got != want {
		t.Errorf("output = %q, want %q", got, want)
	}
}

// TestCELNodeShowsOnlyTerminalOutput checks that a cel node shows its output
// to the user only when no edge leaves it, and never shows a null output.
func TestCELNodeShowsOnlyTerminalOutput(t *testing.T) {
	tests := []struct {
		name   string
		doc    []byte
		agent  string
		prompt string
		reply  func(caller, input string) *genai.Part
		want   []string
	}{
		{
			name:   "non-terminal route node is silent",
			doc:    agentSystem,
			agent:  "llm_router",
			prompt: "Is it raining?",
			reply: func(caller, input string) *genai.Part {
				if caller == "classify" {
					return genai.NewPartFromText("question")
				}
				return genai.NewPartFromText("Probably.")
			},
			want: []string{"question", "Probably."},
		},
		{
			name: "terminal node with null output is silent",
			doc: editedSystem(t, `'Final draft after ' + string(state.turn) + ' turn(s), ' +
          string(state.approved) + '/3 approvals:\n\n' + input`, "null"),
			agent:  "writer_critics",
			prompt: "the history of the bicycle",
			reply: func(caller, input string) *genai.Part {
				if caller == "writer" {
					return genai.NewPartFromText("This is draft 1.")
				}
				return genai.NewPartFromText(approvals[caller])
			},
			want: []string{"This is draft 1.", approvals["critic_accuracy"], approvals["critic_clarity"], approvals["critic_style"]},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agents, err := loadSystem(tc.doc, testRegistry(t, &fakeModel{reply: tc.reply}))
			if err != nil {
				t.Fatalf("loadSystem() error = %v", err)
			}

			got := runTurns(t, agents, tc.agent, tc.prompt)[0]

			slices.Sort(got[1:]) // the critics run in parallel
			want := slices.Clone(tc.want)
			slices.Sort(want[1:])
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("event texts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestStudio checks that the root agent can transfer to either workflow
// sub-agent and that the workflow then runs to its reply.
func TestStudio(t *testing.T) {
	tests := []struct {
		name        string
		prompt      string
		target      string
		want        string
		wantCallers []string
	}{
		{
			name:   "transfer to writer_critics",
			prompt: "Write a paragraph about bicycles.",
			target: "writer_critics",
			want:   "Final draft after 1 turn(s), 3/3 approvals:\n\nThis is draft 1.",
			wantCallers: []string{
				"studio", "writer", "critic_accuracy", "critic_clarity", "critic_style",
			},
		},
		{
			name:        "transfer to llm_router",
			prompt:      "What a day!",
			target:      "llm_router",
			want:        "react_exclamation saw: What a day!",
			wantCallers: []string{"studio", "classify", "react_exclamation"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var writerInputs []string
			critics := writerCriticsModel(func(string, int) bool { return true }, &writerInputs)
			m := &fakeModel{reply: func(caller, input string) *genai.Part {
				switch caller {
				case "studio":
					return &genai.Part{FunctionCall: &genai.FunctionCall{
						Name: "transfer_to_agent",
						Args: map[string]any{"agent_name": tc.target},
					}}
				case "classify":
					return genai.NewPartFromText("exclamation")
				case "react_exclamation":
					return genai.NewPartFromText(caller + " saw: " + input)
				default:
					return critics(caller, input)
				}
			}}
			agents := loadTestSystem(t, m)

			got := runAgent(t, agents, "studio", tc.prompt)

			if got != tc.want {
				t.Errorf("output = %q, want %q", got, tc.want)
			}
			// The critics run in parallel, so their order is not fixed.
			callers := m.callers()
			if len(callers) > 2 {
				slices.Sort(callers[2:])
			}
			if diff := cmp.Diff(tc.wantCallers, callers); diff != "" {
				t.Errorf("model callers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestCELNodeRuntimeErrors checks that an expression failing at run time is
// reported by node and field, without the value that made it fail, which
// here is model output.
func TestCELNodeRuntimeErrors(t *testing.T) {
	tests := []struct {
		name     string
		old, new string // replacement applied to agent-system.yaml
		wantErr  string
		wantIs   error // nil when the error has no sentinel
	}{
		{
			name:    "evaluation fails",
			old:     `expr: "input.trim().lowerAscii()"`,
			new:     `expr: "int(input)"`,
			wantErr: "cel node route: vars.label: CEL evaluation failed",
			wantIs:  errEval,
		},
		{
			name:    "dyn condition is not a bool",
			old:     `when: "vars.label.startsWith('question')"`,
			new:     `when: "vars.label"`,
			wantErr: "cel node route: route QUESTION: expression returned string, want bool",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := editedSystem(t, tc.old, tc.new)
			m := &fakeModel{reply: func(string, string) *genai.Part { return genai.NewPartFromText("secret-label") }}
			agents, err := loadSystem(doc, testRegistry(t, m))
			if err != nil {
				t.Fatalf("loadSystem() error = %v", err)
			}
			router := agents[slices.IndexFunc(agents, func(a agent.Agent) bool { return a.Name() == "llm_router" })]
			r, err := runner.New(runner.Config{AppName: "app", Agent: router, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			var runErr error
			for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					runErr = err
				}
			}

			if runErr == nil || !strings.Contains(runErr.Error(), tc.wantErr) {
				t.Errorf("Run() error = %v, want it to contain %q", runErr, tc.wantErr)
			}
			if tc.wantIs != nil && !errors.Is(runErr, tc.wantIs) {
				t.Errorf("Run() error = %v, want errors.Is(err, %v)", runErr, tc.wantIs)
			}
			if runErr != nil && strings.Contains(runErr.Error(), "secret-label") {
				t.Error("Run() error quotes the model output")
			}
		})
	}
}
