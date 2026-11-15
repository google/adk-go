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
	_ "embed"
	"errors"
	"fmt"
	"iter"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

//go:embed briefing.yaml
var briefingDoc []byte

// workflowDoc returns a document whose only agent is a Workflow named flow,
// with the given nodes and edges blocks.
func workflowDoc(nodes, edges string) []byte {
	return []byte("apiVersion: adk/v1alpha1\nkind: AgentSystem\nagents:\n" +
		"  - name: flow\n    agent_class: Workflow\n    nodes:\n" + nodes + "    edges:\n" + edges)
}

// requestedInput returns the request an input node paused the run with.
func requestedInput(t *testing.T, events []*session.Event) *session.RequestInput {
	t.Helper()
	for _, event := range events {
		if event.RequestedInput != nil {
			return event.RequestedInput
		}
	}
	t.Fatal("the run did not pause for input")
	return nil
}

// inputReply is the message that answers req and resumes the workflow.
func inputReply(req *session.RequestInput, reply any) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID:       req.InterruptID,
			Name:     workflow.WorkflowInputFunctionCallName,
			Response: map[string]any{"payload": reply},
		},
	}}}
}

func TestBriefing(t *testing.T) {
	short := func(topic string) string { return "fact about " + topic }
	long := func(topic string) string { return strings.TrimSpace(strings.Repeat(topic+" ", 21)) }
	tests := []struct {
		name   string
		prompt string
		fact   func(topic string) string
		// reply answers the review; the workflow must not ask when it is nil.
		reply       any
		wantCallers []string // sorted
		wantAsk     string
		wantBefore  []string // texts shown before the workflow asks
		want        string
	}{
		{
			name:        "short briefing published on yes",
			prompt:      "bicycles, trains",
			fact:        short,
			reply:       " Yes please",
			wantCallers: []string{"research", "research"},
			wantAsk:     "Publish this briefing? (yes/no)\n\n- fact about bicycles\n- fact about trains",
			want:        "Briefing on 2 topic(s):\n\n- fact about bicycles\n- fact about trains",
		},
		{
			name:        "briefing discarded on no",
			prompt:      "bicycles",
			fact:        short,
			reply:       "no",
			wantCallers: []string{"research"},
			wantAsk:     "Publish this briefing? (yes/no)\n\n- fact about bicycles",
			want:        "Discarded.",
		},
		{
			// The console sends a reply that parses as JSON as that value.
			name:        "briefing discarded on a reply that is not text",
			prompt:      "bicycles",
			fact:        short,
			reply:       true,
			wantCallers: []string{"research"},
			wantAsk:     "Publish this briefing? (yes/no)\n\n- fact about bicycles",
			want:        "Discarded.",
		},
		{
			// Two topics of 21 words each are over the 40 word limit.
			name:        "long briefing condensed in the nested workflow",
			prompt:      "bicycles,trains",
			fact:        long,
			reply:       "yes",
			wantCallers: []string{"research", "research", "shorten"},
			wantAsk:     "Publish this briefing? (yes/no)\n\ncondensed",
			// The model's reply, and not also the cel node that ends the
			// nested workflow.
			wantBefore: []string{"condensed"},
			want:       "Briefing on 2 topic(s):\n\ncondensed",
		},
		{
			// One topic of 21 words is under it.
			name:        "briefing under the limit not condensed",
			prompt:      "bicycles",
			fact:        long,
			reply:       "yes",
			wantCallers: []string{"research"},
			wantAsk:     "Publish this briefing? (yes/no)\n\n- " + long("bicycles"),
			want:        "Briefing on 1 topic(s):\n\n- " + long("bicycles"),
		},
		{
			name:   "no topics",
			prompt: " , ",
			want:   "Give me one or more topics, separated by commas.",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &fakeModel{reply: func(caller, input string) *genai.Part {
				if caller == "shorten" {
					return genai.NewPartFromText("condensed")
				}
				return genai.NewPartFromText(tc.fact(input))
			}}
			r := newRunner(t, loadDoc(t, briefingDoc, m), "briefing")

			events := runEvents(t, r, genai.NewContentFromText(tc.prompt, genai.RoleUser))
			if tc.reply != nil {
				req := requestedInput(t, events)
				if req.Message != tc.wantAsk {
					t.Errorf("review asked %q, want %q", req.Message, tc.wantAsk)
				}
				if diff := cmp.Diff(tc.wantBefore, eventTexts(events)); diff != "" {
					t.Errorf("texts before the review mismatch (-want +got):\n%s", diff)
				}
				events = runEvents(t, r, inputReply(req, tc.reply))
			}

			if diff := cmp.Diff([]string{tc.want}, eventTexts(events)); diff != "" {
				t.Errorf("reply mismatch (-want +got):\n%s", diff)
			}
			callers := m.callers()
			slices.Sort(callers)
			if diff := cmp.Diff(tc.wantCallers, callers, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("model callers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// slowModel answers every request after hold, or as soon as gate requests are
// in flight at once when gate is set. It records the most it had in flight.
type slowModel struct {
	hold time.Duration
	gate int

	mu             sync.Mutex
	inFlight, peak int
	open           chan struct{} // closed once gate requests were in flight
}

func (*slowModel) Name() string { return "slow" }

func (m *slowModel) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := m.wait(ctx); err != nil {
			yield(nil, err)
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("late", genai.RoleModel), TurnComplete: true}, nil)
	}
}

// wait holds one request, and returns the error of ctx if it ended first.
func (m *slowModel) wait(ctx context.Context) error {
	m.mu.Lock()
	if m.open == nil {
		m.open = make(chan struct{})
	}
	m.inFlight++
	m.peak = max(m.peak, m.inFlight)
	if m.inFlight == m.gate {
		close(m.open)
		m.gate = 0
	}
	open := m.open
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.inFlight--
		m.mu.Unlock()
	}()

	select {
	case <-ctx.Done():
	case <-time.After(m.hold):
	case <-open:
		// Stay in flight a little longer, so that requests a cap should have
		// held back would show in peak.
		time.Sleep(20 * time.Millisecond)
	}
	return ctx.Err()
}

// TestParallelWorkerMaxParallelWorkers runs research on five topics. The
// model answers once two requests are in flight, so the count reaches two
// only if the node is a parallel worker, and stays there only if
// max_parallel_workers holds the other topics back.
func TestParallelWorkerMaxParallelWorkers(t *testing.T) {
	m := &slowModel{hold: 5 * time.Second, gate: 2}
	r := newRunner(t, loadDoc(t, briefingDoc, m), "briefing")

	runEvents(t, r, genai.NewContentFromText("a, b, c, d, e", genai.RoleUser))

	if m.peak != 2 {
		t.Errorf("research had %d requests in flight at once, want 2", m.peak)
	}
}

// TestWorkflowNode checks which nodes of a nested workflow reply to the user:
// its last one, and only when the workflow node ends the outer graph itself.
func TestWorkflowNode(t *testing.T) {
	const nested = `
        nodes:
          - { id: shout, type: cel, output: "input.upperAscii()" }
          - { id: mark,  type: cel, output: "input + '!'" }
        edges:
          - { from_node: START, to_node: shout }
          - { from_node: shout, to_node: mark }
`
	tests := []struct {
		name string
		doc  []byte
		want []string
	}{
		{
			name: "ends the graph",
			doc: workflowDoc(`
      - id: each
        type: workflow`+nested, `
      - { from_node: START, to_node: each }
`),
			want: []string{"ONE TWO!"},
		},
		{
			name: "runs once per item as a parallel worker",
			doc: workflowDoc(`
      - { id: split, type: cel, output: "user.content.split(' ')" }
      - id: each
        type: workflow
        parallel_worker: true`+nested+`
      - { id: glue, type: cel, output: "input.join(' ')" }
`, `
      - { from_node: START, to_node: split }
      - { from_node: split, to_node: each }
      - { from_node: each,  to_node: glue }
`),
			want: []string{"ONE! TWO!"},
		},
		{
			name: "a parallel worker that ends the graph shows nothing",
			doc: workflowDoc(`
      - { id: split, type: cel, output: "user.content.split(' ')" }
      - id: each
        type: workflow
        parallel_worker: true`+nested, `
      - { from_node: START, to_node: split }
      - { from_node: split, to_node: each }
`),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := newRunner(t, loadDoc(t, tc.doc, &fakeModel{}), "flow")

			got := eventTexts(runEvents(t, r, genai.NewContentFromText("one two", genai.RoleUser)))

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("reply mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// flakyModel fails its first failures requests.
type flakyModel struct {
	mu       sync.Mutex
	failures int
}

func (*flakyModel) Name() string { return "flaky" }

func (m *flakyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.fail() {
			yield(nil, errors.New("model unavailable"))
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("recovered", genai.RoleModel), TurnComplete: true}, nil)
	}
}

func (m *flakyModel) fail() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failures--
	return m.failures >= 0
}

// retryDocs has, for each kind of node that takes a retry_config and a
// timeout, a document whose node answer is of that kind. RETRY stands for
// either field.
var retryDocs = map[string]string{
	"llm": `
apiVersion: adk/v1alpha1
kind: AgentSystem
agents:
  - name: flow
    agent_class: Workflow
    nodes:
      - id: answer
        type: llm
        model: default
        instruction: Answer the question.
        RETRY
    edges:
      - { from_node: START, to_node: answer }
`,
	"agent": `
apiVersion: adk/v1alpha1
kind: AgentSystem
agents:
  - name: flow
    agent_class: Workflow
    nodes:
      - id: answer
        type: agent
        agent: helper
        RETRY
    edges:
      - { from_node: START, to_node: answer }
  - name: helper
    agent_class: LlmAgent
    model: default
    mode: single_turn
    instruction: Answer the question.
`,
	"tool": `
apiVersion: adk/v1alpha1
kind: AgentSystem
agents:
  - name: flow
    agent_class: Workflow
    nodes:
      - { id: args, type: cel, output: "{'text': user.content}" }
      - id: answer
        type: tool
        tool: flaky
        RETRY
      - { id: show, type: cel, output: "'recovered'" }
    edges:
      - { from_node: START,  to_node: args }
      - { from_node: args,   to_node: answer }
      - { from_node: answer, to_node: show }
`,
	// The retry of a parallel worker applies to each item.
	"parallel llm": `
apiVersion: adk/v1alpha1
kind: AgentSystem
agents:
  - name: flow
    agent_class: Workflow
    nodes:
      - { id: split, type: cel, output: "user.content.split(' ')" }
      - id: answer
        type: llm
        model: default
        instruction: Answer the question.
        parallel_worker: true
        RETRY
      - { id: show, type: cel, output: "input[0]" }
    edges:
      - { from_node: START,  to_node: split }
      - { from_node: split,  to_node: answer }
      - { from_node: answer, to_node: show }
`,
}

// probeRegistry registers m as the model default, and a tool flaky whose
// every call runs call, for the documents of retryDocs.
func probeRegistry(t *testing.T, m model.LLM, call func(context.Context) error) registry {
	t.Helper()
	flaky, err := functiontool.New(functiontool.Config{Name: "flaky", Description: "Fails or stalls."},
		func(ctx agent.Context, _ wordCountArgs) (wordCountResult, error) {
			return wordCountResult{}, call(ctx)
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return registry{models: map[string]model.LLM{"default": m}, tools: map[string]tool.Tool{"flaky": flaky}}
}

func TestNodeRetry(t *testing.T) {
	const twice = "retry_config: { max_attempts: 3, initial_delay: 0.001, jitter: 0 }"
	tests := []struct {
		name     string
		kind     string // key of retryDocs
		retry    string
		failures int
		wantErr  bool
	}{
		{name: "llm node retried until it succeeds", kind: "llm", retry: twice, failures: 2},
		{name: "llm node out of attempts", kind: "llm", retry: twice, failures: 3, wantErr: true},
		{name: "llm node without retry", kind: "llm", failures: 1, wantErr: true},
		{name: "agent node retried", kind: "agent", retry: twice, failures: 2},
		{name: "agent node without retry", kind: "agent", failures: 1, wantErr: true},
		{name: "tool node retried", kind: "tool", retry: twice, failures: 2},
		{name: "tool node without retry", kind: "tool", failures: 1, wantErr: true},
		{name: "parallel worker item retried", kind: "parallel llm", retry: twice, failures: 2},
		{name: "parallel worker without retry", kind: "parallel llm", failures: 1, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// The model fails for the llm and agent kinds, the tool for the
			// tool kind.
			flaky := &flakyModel{failures: tc.failures}
			reg := probeRegistry(t, flaky, func(context.Context) error {
				if flaky.fail() {
					return errors.New("tool unavailable")
				}
				return nil
			})
			doc := strings.Replace(retryDocs[tc.kind], "RETRY", tc.retry, 1)
			agents, err := loadSystem([]byte(doc), reg)
			if err != nil {
				t.Fatalf("loadSystem() error = %v", err)
			}
			r := newRunner(t, agents, "flow")

			events, err := tryRun(t, r, genai.NewContentFromText("hi", genai.RoleUser))

			if (err != nil) != tc.wantErr {
				t.Fatalf("Run() error = %v, wantErr %t", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if diff := cmp.Diff([]string{"recovered"}, eventTexts(events)); diff != "" {
				t.Errorf("reply mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestNodeTimeout gives each kind of node that takes a timeout a model or a
// tool that answers after five seconds.
func TestNodeTimeout(t *testing.T) {
	for kind, doc := range retryDocs {
		t.Run(kind, func(t *testing.T) {
			slow := &slowModel{hold: 5 * time.Second}
			doc := strings.Replace(doc, "RETRY", "timeout: 0.02", 1)
			agents, err := loadSystem([]byte(doc), probeRegistry(t, slow, slow.wait))
			if err != nil {
				t.Fatalf("loadSystem() error = %v", err)
			}
			r := newRunner(t, agents, "flow")

			_, err = tryRun(t, r, genai.NewContentFromText("hi", genai.RoleUser))

			if !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("Run() error = %v, want %v", err, context.DeadlineExceeded)
			}
		})
	}
}

func TestNodeConfig(t *testing.T) {
	ptr := func(f float64) *float64 { return &f }
	def := workflow.DefaultRetryConfig()
	tests := []struct {
		name       string
		node       nodeSpec
		want       workflow.NodeConfig
		wantIssues []string // paths
	}{
		{name: "nothing set", node: nodeSpec{}, want: workflow.NodeConfig{}},
		{name: "timeout in seconds", node: nodeSpec{Timeout: 1.5}, want: workflow.NodeConfig{Timeout: 1500 * time.Millisecond}},
		{
			name: "empty retry_config keeps the defaults",
			node: nodeSpec{RetryConfig: &retrySpec{}},
			want: workflow.NodeConfig{RetryConfig: def},
		},
		{
			name: "every retry_config field",
			node: nodeSpec{RetryConfig: &retrySpec{MaxAttempts: new(7), InitialDelay: ptr(0.5), MaxDelay: ptr(60), BackoffFactor: ptr(3), Jitter: ptr(1)}},
			want: workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{
				MaxAttempts: 7, InitialDelay: 500 * time.Millisecond, MaxDelay: time.Minute, BackoffFactor: 3, Jitter: 1,
			}},
		},
		{
			// As in adk-python, max_attempts: 0 is no retry.
			name: "zero is a value, not the default",
			node: nodeSpec{RetryConfig: &retrySpec{MaxAttempts: new(0), InitialDelay: ptr(0), MaxDelay: ptr(0), BackoffFactor: ptr(0), Jitter: ptr(0)}},
			want: workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{}},
		},
		{
			name:       "negative values",
			node:       nodeSpec{Timeout: -1, RetryConfig: &retrySpec{MaxAttempts: new(-1), InitialDelay: ptr(-1), MaxDelay: ptr(-1), BackoffFactor: ptr(-1), Jitter: ptr(-0.1)}},
			wantIssues: []string{".timeout", ".retry_config.max_attempts", ".retry_config.initial_delay", ".retry_config.max_delay", ".retry_config.backoff_factor", ".retry_config.jitter"},
		},
		{
			name:       "jitter over one",
			node:       nodeSpec{RetryConfig: &retrySpec{Jitter: ptr(1.1)}},
			wantIssues: []string{".retry_config.jitter"},
		},
		{
			// 1e10 seconds is just over what a time.Duration holds.
			name:       "seconds a duration cannot hold",
			node:       nodeSpec{Timeout: math.Inf(1), RetryConfig: &retrySpec{InitialDelay: ptr(1e10), MaxDelay: ptr(math.NaN())}},
			wantIssues: []string{".timeout", ".retry_config.initial_delay", ".retry_config.max_delay"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, issues := nodeConfig(&tc.node)

			var paths []string
			for _, is := range issues {
				paths = append(paths, is.Path)
			}
			if diff := cmp.Diff(tc.wantIssues, paths); diff != "" {
				t.Fatalf("nodeConfig() issue paths mismatch (-want +got):\n%s", diff)
			}
			if tc.wantIssues != nil {
				return
			}
			if diff := cmp.Diff(tc.want, got, cmpopts.IgnoreFields(workflow.RetryConfig{}, "ShouldRetry")); diff != "" {
				t.Errorf("nodeConfig() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestMisapplied sets every field on a node of each type and checks which
// ones the type does not read.
func TestMisapplied(t *testing.T) {
	everything := nodeSpec{
		Agent: "a", Model: "m", Instruction: "i", Tools: []string{"t"},
		Vars: []varSpec{{}}, Routes: []routeSpec{{}}, Output: "o", StateDelta: map[string]string{"k": "v"},
		Tool: "t", Nodes: []nodeSpec{{}}, Edges: []edgeSpec{{}}, Message: "m",
		ParallelWorker: true, RetryConfig: &retrySpec{}, Timeout: 1,
	}
	// In the order of nodeFields.
	fields := []string{
		"agent", "model", "instruction", "tools", "vars", "routes", "output", "state_delta",
		"tool", "nodes", "edges", "message", "parallel_worker", "retry_config", "timeout",
	}
	own := map[string][]string{
		nodeAgent:    {"agent", "parallel_worker", "retry_config", "timeout"},
		nodeLLM:      {"model", "instruction", "tools", "parallel_worker", "retry_config", "timeout"},
		nodeCEL:      {"vars", "routes", "output", "state_delta", "parallel_worker"},
		nodeJoin:     nil,
		nodeTool:     {"tool", "parallel_worker", "retry_config", "timeout"},
		nodeWorkflow: {"nodes", "edges", "parallel_worker"},
		nodeInput:    {"message"},
	}
	if len(own) != len(nodeTypes) {
		t.Fatalf("test covers %d node types, want %d", len(own), len(nodeTypes))
	}
	for typ, reads := range own {
		var want []string
		for _, name := range fields {
			if !slices.Contains(reads, name) {
				want = append(want, name)
			}
		}
		n := everything
		n.Type = typ

		if diff := cmp.Diff(want, misapplied(&n)); diff != "" {
			t.Errorf("misapplied(%s node) mismatch (-want +got):\n%s", typ, diff)
		}
		if got := misapplied(&nodeSpec{Type: typ}); got != nil {
			t.Errorf("misapplied(empty %s node) = %v, want none", typ, got)
		}
	}
}

// TestCELNodePassesDoubles checks that a double one node outputs is a double
// in the next, whether or not it is a whole number.
func TestCELNodePassesDoubles(t *testing.T) {
	doc := workflowDoc(`
      - { id: half,  type: cel, output: "double(user.content) / 2.0" }
      - { id: twice, type: cel, output: "string(input * 2.0)" }
`, `
      - { from_node: START, to_node: half }
      - { from_node: half,  to_node: twice }
`)
	agents := loadDoc(t, doc, &fakeModel{})

	// 7 / 2 is 3.5 and 8 / 2 is 4.0, which must not arrive as the integer 4.
	for _, n := range []string{"7", "8"} {
		if got := runAgent(t, agents, "flow", n); got != n {
			t.Errorf("half then twice of %s = %q, want %q", n, got, n)
		}
	}
}

// TestEdgeRouteList routes two of a node's three outcomes over one edge.
func TestEdgeRouteList(t *testing.T) {
	doc := workflowDoc(`
      - id: classify
        type: cel
        routes:
          - { route: P0, when: "user.content == 'outage'" }
          - { route: P1, when: "user.content == 'slow'" }
      - { id: page,   type: cel, output: "'page'" }
      - { id: ticket, type: cel, output: "'ticket'" }
`, `
      - { from_node: START,    to_node: classify }
      - { from_node: classify, to_node: page,   route: [P0, P1] }
      - { from_node: classify, to_node: ticket, route: DEFAULT }
`)
	agents := loadDoc(t, doc, &fakeModel{})

	for prompt, want := range map[string]string{"outage": "page", "slow": "page", "typo": "ticket"} {
		if got := runAgent(t, agents, "flow", prompt); got != want {
			t.Errorf("%q went to %q, want %q", prompt, got, want)
		}
	}
}

// TestToolNodeNamedByID runs one tool from two nodes. A join keys its output
// by node name, so each node must be named after its id, not after the tool.
func TestToolNodeNamedByID(t *testing.T) {
	doc := workflowDoc(`
      - { id: args,  type: cel,  output: "{'text': user.content}" }
      - { id: one,   type: tool, tool: word_count }
      - { id: two,   type: tool, tool: word_count }
      - { id: both,  type: join }
      - { id: total, type: cel,  output: "string(input.one.words + input.two.words)" }
`, `
      - { from_node: START, to_node: args }
      - { from_node: args,  to_node: one }
      - { from_node: args,  to_node: two }
      - { from_node: one,   to_node: both }
      - { from_node: two,   to_node: both }
      - { from_node: both,  to_node: total }
`)

	got := runAgent(t, loadDoc(t, doc, &fakeModel{}), "flow", "three short words")

	if got != "6" {
		t.Errorf("reply = %q, want %q", got, "6")
	}
}

// toolSpy records the tools each request declares, by the instruction of the
// agent that sent it.
type toolSpy struct {
	mu    sync.Mutex
	tools map[string][]string
}

func (*toolSpy) Name() string { return "spy" }

func (m *toolSpy) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var names []string
		for name := range req.Tools {
			names = append(names, name)
		}
		slices.Sort(names)
		m.mu.Lock()
		m.tools[strings.TrimSpace(contentText(req.Config.SystemInstruction))] = names
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel), TurnComplete: true}, nil)
	}
}

func TestLLMTools(t *testing.T) {
	doc := []byte(`
apiVersion: adk/v1alpha1
kind: AgentSystem
agents:
  - name: flow
    agent_class: Workflow
    nodes:
      - { id: first, type: agent, agent: helper }
      - { id: second, type: llm, model: default, instruction: llm node, tools: [word_count] }
      - { id: third, type: llm, model: default, instruction: llm node without tools }
    edges:
      - { from_node: START,  to_node: first }
      - { from_node: first,  to_node: second }
      - { from_node: second, to_node: third }
  - name: helper
    agent_class: LlmAgent
    model: default
    mode: single_turn
    instruction: agent
    tools: [word_count]
`)
	m := &toolSpy{tools: map[string][]string{}}
	r := newRunner(t, loadDoc(t, doc, m), "flow")

	runEvents(t, r, genai.NewContentFromText("hi", genai.RoleUser))

	want := map[string][]string{
		"agent":                  {"word_count"},
		"llm node":               {"word_count"},
		"llm node without tools": nil,
	}
	if diff := cmp.Diff(want, m.tools, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
}

// TestInputNodeMessageError checks that a message that fails to evaluate
// names the node and not the value it failed on.
func TestInputNodeMessageError(t *testing.T) {
	tests := []struct {
		name    string
		message string
		wantErr string
		wantIs  error // nil when the error has no sentinel
	}{
		{name: "evaluation fails", message: "string(int(user.content))", wantErr: "input node ask: message: CEL evaluation failed", wantIs: errEval},
		{name: "dyn message is not a string", message: "input", wantErr: "input node ask: message: expression returned []interface {}, want string"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := workflowDoc(`
      - { id: split, type: cel, output: "user.content.split(' ')" }
      - { id: ask, type: input, message: "`+tc.message+`" }
`, `
      - { from_node: START, to_node: split }
      - { from_node: split, to_node: ask }
`)
			r := newRunner(t, loadDoc(t, doc, &fakeModel{}), "flow")

			_, err := tryRun(t, r, genai.NewContentFromText("secret-text", genai.RoleUser))

			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Run() error = %v, want it to contain %q", err, tc.wantErr)
			}
			if tc.wantIs != nil && !errors.Is(err, tc.wantIs) {
				t.Errorf("Run() error = %v, want errors.Is(err, %v)", err, tc.wantIs)
			}
			if strings.Contains(err.Error(), "secret-text") {
				t.Error("Run() error echoes the user's message")
			}
		})
	}
}

// TestInputNodeInput asks about the output of a tool. A tool's numbers are
// float64, which CEL would not add to an integer, so the node must see them
// as a cel node does.
func TestInputNodeInput(t *testing.T) {
	doc := workflowDoc(`
      - { id: args,  type: cel,  output: "{'text': user.content}" }
      - { id: count, type: tool, tool: word_count }
      - { id: ask,   type: input, message: "'Double it to ' + string(input.words * 2) + '?'" }
`, `
      - { from_node: START, to_node: args }
      - { from_node: args,  to_node: count }
      - { from_node: count, to_node: ask }
`)
	r := newRunner(t, loadDoc(t, doc, &fakeModel{}), "flow")

	req := requestedInput(t, runEvents(t, r, genai.NewContentFromText("three short words", genai.RoleUser)))

	if want := "Double it to 6?"; req.Message != want {
		t.Errorf("ask asked %q, want %q", req.Message, want)
	}
}

// TestUnknownNodeType checks that a node of no known type is reported for
// that alone, not also for every field it sets.
func TestUnknownNodeType(t *testing.T) {
	doc := strings.Replace(string(briefingDoc), "type: tool, tool: word_count", "type: tol, tool: word_count", 1)

	_, err := loadSystem([]byte(doc), testRegistry(t, &fakeModel{}))

	var verr *validationError
	if !errors.As(err, &verr) {
		t.Fatalf("loadSystem() error = %v, want a *validationError", err)
	}
	want := []issue{{
		Path:    "agents[0].nodes[3].type",
		Message: `got "tol", want one of agent, llm, cel, join, tool, workflow, input`,
	}}
	if diff := cmp.Diff(want, verr.Issues); diff != "" {
		t.Errorf("loadSystem() issues mismatch (-want +got):\n%s", diff)
	}
}

// TestAgentNodeSubAgentCycle checks that agents that transfer to each other
// are reported when built, not followed forever when validated.
func TestAgentNodeSubAgentCycle(t *testing.T) {
	const peer = "  - name: %s\n    agent_class: LlmAgent\n    model: default\n    sub_agents: [%s]\n"
	doc := append(workflowDoc("      - { id: n, type: agent, agent: a }\n", "      - { from_node: START, to_node: n }\n"),
		fmt.Sprintf(peer, "a", "b")+fmt.Sprintf(peer, "b", "a")...)

	_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

	if err == nil || !strings.Contains(err.Error(), `agent "a" refers to itself`) {
		t.Errorf("loadSystem() error = %v, want a reference cycle error", err)
	}
}

// TestScopeIssues checks what a document may not put under a parallel worker
// or in a nested workflow, and what it may.
func TestScopeIssues(t *testing.T) {
	const (
		start = "      - { from_node: START, to_node: n }\n"
		// each is a parallel workflow node around the given nodes, the first
		// of which is named inner.
		each = "      - id: n\n        type: workflow\n        parallel_worker: true\n        nodes:\n%s" +
			"        edges:\n          - { from_node: START, to_node: inner }\n"
		helper = "  - name: helper\n    agent_class: LlmAgent\n    model: default\n    mode: single_turn\n"
		// desk transfers to the agent named as its only sub-agent.
		desk   = "  - name: desk\n    agent_class: LlmAgent\n    model: default\n    sub_agents: [%s]\n"
		pauses = "  - name: pauses\n    agent_class: Workflow\n    nodes:\n" +
			"      - { id: ask, type: input, message: \"'Go on?'\" }\n" +
			"    edges:\n      - { from_node: START, to_node: ask }\n"
	)
	// plain is a workflow agent that does not pause.
	plain := strings.NewReplacer("pauses", "plain", "type: input, message", "type: cel, output").Replace(pauses)
	tests := []struct {
		name   string
		nodes  string
		agents string // more agents of the document
		// wantPath is empty when the document is valid.
		wantPath, wantMsg string
	}{
		{
			name:     "parallel llm node with tools",
			nodes:    "      - { id: n, type: llm, model: default, parallel_worker: true, tools: [word_count] }\n",
			wantPath: "agents[0].nodes[0].tools",
			wantMsg:  "a parallel worker drops tool calls",
		},
		{
			name:     "llm node with tools in a parallel workflow node",
			nodes:    fmt.Sprintf(each, "          - { id: inner, type: llm, model: default, tools: [word_count] }\n"),
			wantPath: "agents[0].nodes[0].nodes[0].tools",
			wantMsg:  "a parallel worker drops tool calls",
		},
		{
			name:     "parallel cel node with routes",
			nodes:    "      - { id: n, type: cel, parallel_worker: true, routes: [{ route: A, when: \"true\" }] }\n",
			wantPath: "agents[0].nodes[0].routes",
			wantMsg:  "a parallel worker drops routes",
		},
		{
			name:  "cel node with routes in a parallel workflow node",
			nodes: fmt.Sprintf(each, "          - { id: inner, type: cel, routes: [{ route: A, when: \"true\" }] }\n"),
		},
		{
			name:     "parallel cel node with state_delta",
			nodes:    "      - { id: n, type: cel, parallel_worker: true, state_delta: { seen: input } }\n",
			wantPath: "agents[0].nodes[0].state_delta",
			wantMsg:  "a parallel worker drops state changes",
		},
		{
			name:     "cel node with state_delta in a parallel workflow node",
			nodes:    fmt.Sprintf(each, "          - { id: inner, type: cel, state_delta: { seen: input } }\n"),
			wantPath: "agents[0].nodes[0].nodes[0].state_delta",
			wantMsg:  "a parallel worker drops state changes",
		},
		{
			name: "cel node with state_delta two workflow nodes under a parallel one",
			nodes: fmt.Sprintf(each, "          - id: inner\n            type: workflow\n            nodes:\n"+
				"              - { id: deep, type: cel, state_delta: { seen: input } }\n"+
				"            edges:\n              - { from_node: START, to_node: deep }\n"),
			wantPath: "agents[0].nodes[0].nodes[0].nodes[0].state_delta",
			wantMsg:  "a parallel worker drops state changes",
		},
		{
			name:     "parallel agent node whose agent has tools",
			nodes:    "      - { id: n, type: agent, agent: helper, parallel_worker: true }\n",
			agents:   helper + "    tools: [word_count]\n",
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `a parallel worker drops the tool calls and output_key of agent "helper"`,
		},
		{
			name:     "agent node whose agent has an output_key in a parallel workflow node",
			nodes:    fmt.Sprintf(each, "          - { id: inner, type: agent, agent: helper }\n"),
			agents:   helper + "    output_key: last\n",
			wantPath: "agents[0].nodes[0].nodes[0].agent",
			wantMsg:  `a parallel worker drops the tool calls and output_key of agent "helper"`,
		},
		{
			name:   "parallel agent node whose agent has neither",
			nodes:  "      - { id: n, type: agent, agent: helper, parallel_worker: true }\n",
			agents: helper,
		},
		{
			name:   "agent node whose agent has both",
			nodes:  "      - { id: n, type: agent, agent: helper }\n",
			agents: helper + "    tools: [word_count]\n    output_key: last\n",
		},
		{
			name:     "parallel agent node whose agent is a workflow",
			nodes:    "      - { id: n, type: agent, agent: plain, parallel_worker: true }\n",
			agents:   plain,
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `a parallel worker cannot run workflow agent "plain"`,
		},
		{
			name:     "agent node whose agent is a workflow in a parallel workflow node",
			nodes:    fmt.Sprintf(each, "          - { id: inner, type: agent, agent: plain }\n"),
			agents:   plain,
			wantPath: "agents[0].nodes[0].nodes[0].agent",
			wantMsg:  `a parallel worker cannot run workflow agent "plain"`,
		},
		{
			name:     "parallel agent node whose agent transfers to a workflow",
			nodes:    "      - { id: n, type: agent, agent: desk, parallel_worker: true }\n",
			agents:   fmt.Sprintf(desk, "plain") + plain,
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `a parallel worker cannot run workflow agent "plain"`,
		},
		{
			name:     "parallel agent node whose agent transfers to an agent with tools",
			nodes:    "      - { id: n, type: agent, agent: desk, parallel_worker: true }\n",
			agents:   fmt.Sprintf(desk, "helper") + helper + "    tools: [word_count]\n",
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `a parallel worker drops the tool calls and output_key of agent "helper"`,
		},
		{
			// desk transfers to mid, and mid to helper.
			name:  "parallel agent node whose agent transfers, two levels down, to an agent with tools",
			nodes: "      - { id: n, type: agent, agent: desk, parallel_worker: true }\n",
			agents: fmt.Sprintf(desk, "mid") + strings.Replace(fmt.Sprintf(desk, "helper"), "name: desk", "name: mid", 1) +
				helper + "    tools: [word_count]\n",
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `a parallel worker drops the tool calls and output_key of agent "helper"`,
		},
		{
			name:   "agent node whose agent transfers to an agent with tools and to a workflow",
			nodes:  "      - { id: n, type: agent, agent: desk }\n",
			agents: fmt.Sprintf(desk, "helper, plain") + helper + "    tools: [word_count]\n" + plain,
		},
		{
			name:     "agent node whose agent transfers to a workflow with an input node",
			nodes:    "      - { id: n, type: agent, agent: desk }\n",
			agents:   fmt.Sprintf(desk, "pauses") + pauses,
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `workflow "pauses" pauses for input`,
		},
		{
			name: "input node in a nested workflow",
			nodes: "      - id: n\n        type: workflow\n        nodes:\n" +
				"          - { id: inner, type: input, message: \"'Go on?'\" }\n" +
				"        edges:\n          - { from_node: START, to_node: inner }\n",
			wantPath: "agents[0].nodes[0].nodes[0].type",
			wantMsg:  "an input node cannot be in a nested workflow",
		},
		{
			name:     "agent node whose agent is a workflow with an input node",
			nodes:    "      - { id: n, type: agent, agent: pauses }\n",
			agents:   pauses,
			wantPath: "agents[0].nodes[0].agent",
			wantMsg:  `workflow "pauses" pauses for input`,
		},
		{
			name:   "agent node whose agent is a workflow without an input node",
			nodes:  "      - { id: n, type: agent, agent: plain }\n",
			agents: plain,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := append(workflowDoc(tc.nodes, start), tc.agents...)

			_, err := loadSystem(doc, testRegistry(t, &fakeModel{}))

			if tc.wantPath == "" {
				if err != nil {
					t.Fatalf("loadSystem() error = %v, want none", err)
				}
				return
			}
			var verr *validationError
			if !errors.As(err, &verr) {
				t.Fatalf("loadSystem() error = %v, want a *validationError", err)
			}
			if len(verr.Issues) != 1 || verr.Issues[0].Path != tc.wantPath || !strings.Contains(verr.Issues[0].Message, tc.wantMsg) {
				t.Errorf("loadSystem() issues = %v, want only one at %s containing %q", verr.Issues, tc.wantPath, tc.wantMsg)
			}
		})
	}
}

// TestLoadSystemReportsNodeIssues is TestLoadSystemReportsIssues for the node
// types of briefing.yaml.
func TestLoadSystemReportsNodeIssues(t *testing.T) {
	tests := []struct {
		name     string
		old, new string // replacement applied to briefing.yaml
		wantPath string
		wantMsg  string // substring of the issue's message
	}{
		{
			name:     "tool node names an unregistered tool",
			old:      "tool: word_count",
			new:      "tool: sentiment",
			wantPath: "agents[0].nodes[3].tool",
			wantMsg:  `tool "sentiment" is not registered`,
		},
		{
			name:     "llm node lists an unregistered tool",
			old:      "parallel_worker: true",
			new:      "parallel_worker: true\n        tools: [word_count, sentiment]",
			wantPath: "agents[0].nodes[1].tools[1]",
			wantMsg:  `tool "sentiment" is not registered`,
		},
		{
			name:     "max_parallel_workers without parallel_worker",
			old:      "parallel_worker: true",
			new:      "parallel_worker: false",
			wantPath: "agents[0].nodes[1].max_parallel_workers",
			wantMsg:  "applies only with parallel_worker",
		},
		{
			name:     "negative max_parallel_workers",
			old:      "max_parallel_workers: 2",
			new:      "max_parallel_workers: -2",
			wantPath: "agents[0].nodes[1].max_parallel_workers",
			wantMsg:  "must not be negative",
		},
		{
			name:     "negative timeout",
			old:      "timeout: 120",
			new:      "timeout: -120",
			wantPath: "agents[0].nodes[1].timeout",
			wantMsg:  "must be between 0 and 9223372036 seconds",
		},
		{
			name:     "negative retry delay",
			old:      "initial_delay: 1 ",
			new:      "initial_delay: -1 ",
			wantPath: "agents[0].nodes[1].retry_config.initial_delay",
			wantMsg:  "must be between 0 and 9223372036 seconds",
		},
		{
			name:     "field of another node type",
			old:      "type: input",
			new:      "type: input\n        timeout: 5",
			wantPath: "agents[0].nodes[7].timeout",
			wantMsg:  "timeout does not apply to input nodes",
		},
		{
			name:     "input message is not a string",
			old:      `+ input"`,
			new:      `+ input == ''"`,
			wantPath: "agents[0].nodes[7].message",
			wantMsg:  "expression has type bool, want string",
		},
		{
			name:     "input message does not compile",
			old:      `+ input"`,
			new:      `+ "`,
			wantPath: "agents[0].nodes[7].message",
			wantMsg:  "Syntax error",
		},
		{
			name:     "input message reads vars",
			old:      `+ input"`,
			new:      `+ vars.briefing"`,
			wantPath: "agents[0].nodes[7].message",
			wantMsg:  "vars.briefing is not declared",
		},
		{
			name:     "DEFAULT in a route list",
			old:      "to_node: review,   route: DEFAULT }",
			new:      "to_node: review,   route: [LONG, DEFAULT] }",
			wantPath: "agents[0].edges[7].route",
			wantMsg:  "DEFAULT cannot be combined with other routes",
		},
		{
			name:     "route list names a route never emitted",
			old:      "to_node: condense, route: LONG }",
			new:      "to_node: condense, route: [LONG, HUGE] }",
			wantPath: "agents[0].edges[6].route",
			wantMsg:  `never emits route "HUGE"`,
		},
		{
			name:     "edge of a nested workflow",
			old:      "to_node: keep }",
			new:      "to_node: kept }",
			wantPath: "agents[0].nodes[6].edges[1].to_node",
			wantMsg:  `node "kept" is not declared`,
		},
		{
			name:     "node of a nested workflow",
			old:      "briefing: input",
			new:      "briefing: inputs",
			wantPath: "agents[0].nodes[6].nodes[1].state_delta.briefing",
			wantMsg:  "undeclared reference",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := strings.Replace(string(briefingDoc), tc.old, tc.new, 1)
			if doc == string(briefingDoc) {
				t.Fatalf("%q not found in briefing.yaml", tc.old)
			}

			_, err := loadSystem([]byte(doc), testRegistry(t, &fakeModel{}))

			var verr *validationError
			if !errors.As(err, &verr) {
				t.Fatalf("loadSystem() error = %v, want a *validationError", err)
			}
			found := slices.ContainsFunc(verr.Issues, func(is issue) bool {
				return is.Path == tc.wantPath && strings.Contains(is.Message, tc.wantMsg)
			})
			if !found {
				t.Errorf("loadSystem() issues = %v, want one at %s containing %q", verr.Issues, tc.wantPath, tc.wantMsg)
			}
		})
	}
}
