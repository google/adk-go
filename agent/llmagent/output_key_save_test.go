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
	"encoding/json"
	"errors"
	"iter"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// streamModel answers its n-th request with the responses of turns[n], in
// order, and fails once the turns run out. Safe for concurrent use.
type streamModel struct {
	mu    sync.Mutex
	turns [][]*model.LLMResponse
}

func (*streamModel) Name() string { return "stream-model" }

func (m *streamModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	var turn []*model.LLMResponse
	if len(m.turns) > 0 {
		turn, m.turns = m.turns[0], m.turns[1:]
	}
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		if turn == nil {
			yield(nil, errors.New("stream-model: script exhausted"))
			return
		}
		for _, resp := range turn {
			if !yield(resp, nil) {
				return
			}
		}
	}
}

// capableStreamModel declares that it takes an output schema and tools in
// one request, so an agent with both sends no set_model_response tool.
type capableStreamModel struct{ *streamModel }

func (capableStreamModel) Capabilities() model.Capabilities {
	return model.Capabilities{OutputSchemaAndTools: true}
}

func partialText(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel), Partial: true}
}

func finalParts(parts ...*genai.Part) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      &genai.Content{Role: genai.RoleModel, Parts: parts},
		FinishReason: genai.FinishReasonStop,
	}
}

func callPart(id, name string, args map[string]any) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args}}
}

// toolT is the tool "t" the scripted models call; it answers {"ok": true}.
func toolT(t *testing.T) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{Name: "t", Description: "Does nothing."},
		func(agent.Context, struct{}) (map[string]any, error) { return map[string]any{"ok": true}, nil })
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return ft
}

func newOutputKeyAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", cfg.Name, err)
	}
	return a
}

// outputKeyRun is what one run of a root agent left behind.
type outputKeyRun struct {
	events  []*session.Event // yielded by the runner, up to the first error and any event that came with it
	state   map[string]any   // session state after the run
	history []*session.Event // the session's events after the run
}

func newOutputKeySession(t *testing.T, root agent.Agent) (*runner.Runner, session.Service) {
	t.Helper()
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return r, svc
}

func readOutputKeySession(t *testing.T, svc session.Service, run *outputKeyRun) {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("session.Get: %v", err)
	}
	run.state = maps.Collect(resp.Session.State().All())
	for ev := range resp.Session.Events().All() {
		run.history = append(run.history, ev)
	}
}

// runSSE runs root once through a runner in SSE mode, and returns what the run
// left behind and the first error the runner yielded.
func runSSE(t *testing.T, root agent.Agent) (outputKeyRun, error) {
	t.Helper()
	r, svc := newOutputKeySession(t, root)
	var run outputKeyRun
	var runErr error
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("go", genai.RoleUser),
		agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if ev != nil {
			run.events = append(run.events, ev)
		}
		if err != nil {
			runErr = err
			break
		}
	}
	readOutputKeySession(t, svc, &run)
	return run, runErr
}

// liveModel is a Gemini model whose live connection goes to the client's
// server. It has no non-live path.
type liveModel struct{ client *genai.Client }

func (*liveModel) Name() string { return "live-model" }

func (*liveModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("live-model: no GenerateContent"))
	}
}

func (m *liveModel) Client() *genai.Client { return m.client }

// startOutputKeyLiveServer starts a fake Gemini Live API server, and returns a
// client for it. After the setup exchange the server sends messages on the
// connection and then reads until the client closes it.
func startOutputKeyLiveServer(t *testing.T, messages ...string) *genai.Client {
	t.Helper()
	var upgrader websocket.Upgrader
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			return
		}
		for _, m := range messages {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	t.Cleanup(ts.Close)
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	return client
}

// runLive runs root through runner.RunLive until the first event that
// completes a model turn, and returns what the run left behind and the first
// error the runner yielded.
func runLive(t *testing.T, root agent.Agent) (outputKeyRun, error) {
	t.Helper()
	r, svc := newOutputKeySession(t, root)
	sess, events, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("runner.RunLive: %v", err)
	}
	var run outputKeyRun
	var runErr error
	for ev, err := range events {
		if ev != nil {
			run.events = append(run.events, ev)
		}
		if err != nil {
			runErr = err
			break
		}
		if ev.TurnComplete {
			break
		}
	}
	if err := sess.Close(); err != nil {
		t.Errorf("LiveSession.Close: %v", err)
	}
	readOutputKeySession(t, svc, &run)
	return run, runErr
}

func liveText(text string, turnComplete bool) string {
	content := `{"modelTurn":{"parts":[{"text":` + jsonString(text) + `}],"role":"model"}`
	if turnComplete {
		content += `,"turnComplete":true`
	}
	return `{"serverContent":` + content + `}}`
}

func jsonString(s string) string {
	b, _ := json.Marshal(s) // A string always marshals.
	return string(b)
}

func hasEventWithText(events []*session.Event, text string) bool {
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.Text == text {
				return true
			}
		}
	}
	return false
}

const fencedJSON = "```json\n{\"a\": 1}\n```"

// checkJSONOutput checks the outcome of one run of agent "a", whose OutputKey
// is "out" and whose OutputJsonSchema is set, that gave reply as its answer.
// A nil wantOut means the run fails with the "not valid JSON" error, does not
// yield or record the reply, and leaves "out" unset.
func checkJSONOutput(t *testing.T, run outputKeyRun, err error, reply string, wantOut any) {
	t.Helper()
	if wantOut != nil {
		if err != nil {
			t.Fatalf("run error = %v, want none", err)
		}
		if diff := cmp.Diff(wantOut, run.state["out"]); diff != "" {
			t.Errorf("state[out] mismatch (-want +got):\n%s", diff)
		}
		return
	}
	const wantErr = `LlmAgent "a" output is not valid JSON`
	if err == nil || !strings.Contains(err.Error(), wantErr) {
		t.Errorf("run error = %v, want one that contains %q", err, wantErr)
	}
	if hasEventWithText(run.events, reply) {
		t.Errorf("the run yielded the event with the reply %q, want it dropped", reply)
	}
	if hasEventWithText(run.history, reply) {
		t.Errorf("the session recorded the event with the reply %q, want it dropped", reply)
	}
	if v, ok := run.state["out"]; ok {
		t.Errorf("state[out] = %#v, want it unset", v)
	}
}

// A chat agent with OutputJsonSchema stores the value its answer parses to,
// after removing a code fence around it, and fails on an answer that is not
// JSON, as adk-python's __maybe_save_output_to_state does through
// validate_schema.
// Fails when the text is stored as it is, when the fence is kept, or when a
// parse error is ignored.
func TestOutputKey_JSONSchemaChatAgent(t *testing.T) {
	tests := []struct {
		name    string
		reply   string
		wantOut any
	}{
		{name: "fenced object", reply: fencedJSON, wantOut: map[string]any{"a": float64(1)}},
		{name: "not JSON", reply: "not json", wantOut: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newOutputKeyAgent(t, llmagent.Config{
				Name:             "a",
				Model:            &streamModel{turns: [][]*model.LLMResponse{{finalParts(genai.NewPartFromText(tc.reply))}}},
				OutputKey:        "out",
				OutputJsonSchema: mapSchema(),
			})

			run, err := runSSE(t, a)

			checkJSONOutput(t, run, err, tc.reply, tc.wantOut)
		})
	}
}

// A before-agent callback's content gets the OutputJsonSchema handling of a
// model answer, as adk-python's LlmAgent._handle_before_agent_callback runs
// __maybe_save_output_to_state on it. All-whitespace content stores nothing
// and is no error. A sequential parent passes on what the agent's Run yields,
// so it shows that the error comes without the event.
// Fails when the callback's text is stored as it is, when its parse error is
// ignored or yielded together with the event, or when whitespace is parsed.
func TestOutputKey_JSONSchemaBeforeAgentCallback(t *testing.T) {
	tests := []struct {
		name             string
		content          string
		sequentialParent bool
		wantOut          any
	}{
		{name: "fenced object", content: fencedJSON, wantOut: map[string]any{"a": float64(1)}},
		{name: "not JSON", content: "not json", wantOut: nil},
		{name: "not JSON under a sequential parent", content: "not json", sequentialParent: true, wantOut: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := newOutputKeyAgent(t, llmagent.Config{
				Name:             "a",
				Model:            &streamModel{},
				OutputKey:        "out",
				OutputJsonSchema: mapSchema(),
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(agent.Context) (*genai.Content, error) {
					return genai.NewContentFromText(tc.content, genai.RoleModel), nil
				}},
			})
			if tc.sequentialParent {
				seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{a}}})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
				a = seq
			}

			run, err := runSSE(t, a)

			checkJSONOutput(t, run, err, tc.content, tc.wantOut)
		})
	}

	t.Run("whitespace", func(t *testing.T) {
		a := newOutputKeyAgent(t, llmagent.Config{
			Name:             "a",
			Model:            &streamModel{},
			OutputKey:        "out",
			OutputJsonSchema: mapSchema(),
			BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(agent.Context) (*genai.Content, error) {
				return genai.NewContentFromText(" \n ", genai.RoleModel), nil
			}},
		})

		run, err := runSSE(t, a)
		if err != nil {
			t.Fatalf("run error = %v, want none", err)
		}
		if !hasEventWithText(run.events, " \n ") {
			t.Error("the run did not yield the callback's event")
		}
		if v, ok := run.state["out"]; ok {
			t.Errorf("state[out] = %#v, want it unset", v)
		}
	})
}

// A live run of an agent with OutputJsonSchema whose JSON answer comes in two
// Live API messages ends without an error, yields both chunks and saves an
// output. The live connection gives each text chunk as its own non-partial
// event, so the first chunk alone is not valid JSON, and RunLive keeps the
// text form. The value stored under OutputKey is not checked: adk-python
// stores the parsed answer, which RunLive cannot give until the live
// connection joins the chunks.
// Fails when RunLive parses the text of each chunk, or saves no output.
func TestOutputKey_JSONSchemaLiveChunkedAnswer(t *testing.T) {
	a := newOutputKeyAgent(t, llmagent.Config{
		Name:             "a",
		Model:            &liveModel{client: startOutputKeyLiveServer(t, liveText(`{"a":`, false), liveText(` 1}`, true))},
		OutputKey:        "out",
		OutputJsonSchema: mapSchema(),
	})

	run, err := runLive(t, a)
	if err != nil {
		t.Fatalf("run error = %v, want none", err)
	}
	for _, chunk := range []string{`{"a":`, ` 1}`} {
		if !hasEventWithText(run.events, chunk) {
			t.Errorf("the run did not yield the event with the chunk %q", chunk)
		}
	}
	if _, ok := run.state["out"]; !ok {
		t.Error("state[out] is unset, want the answer saved")
	}
}

// OutputKey holds all the text the agent streamed in its run, including the
// text of model turns that end in a tool call. This is the sequence of
// adk-python's test_run_async_accumulates_text_around_tool_calls
// (tests/unittests/agents/test_llm_agent_streaming_output.py).
// Fails when only the final response is stored, or when partial events are
// counted too.
func TestOutputKey_AccumulatesTextAroundToolCalls(t *testing.T) {
	m := &streamModel{turns: [][]*model.LLMResponse{
		{
			partialText("Intro one. "),
			partialText("Intro two."),
			finalParts(genai.NewPartFromText("Intro one. Intro two."), callPart("c1", "t", nil)),
		},
		{
			partialText("Progress."),
			finalParts(genai.NewPartFromText("Progress."), callPart("c2", "t", nil)),
		},
		{
			partialText("Conclusion one. "),
			partialText("Conclusion two."),
			finalParts(genai.NewPartFromText("Conclusion one. Conclusion two.")),
		},
	}}
	a := newOutputKeyAgent(t, llmagent.Config{
		Name:      "agent",
		Model:     m,
		OutputKey: "final_output",
		Tools:     []tool.Tool{toolT(t)},
	})

	run, err := runSSE(t, a)
	if err != nil {
		t.Fatalf("run error = %v, want none", err)
	}
	want := "Intro one. Intro two.Progress.Conclusion one. Conclusion two."
	if got := run.state["final_output"]; got != want {
		t.Errorf("state[final_output] = %#v, want %q", got, want)
	}
}

// In live mode the text of every non-partial event of the run accumulates
// too. The Live API connection gives each text chunk as its own non-partial
// event.
// Fails when RunLive does not accumulate, which stores only the last chunk.
func TestOutputKey_AccumulatesTextLive(t *testing.T) {
	a := newOutputKeyAgent(t, llmagent.Config{
		Name:      "a",
		Model:     &liveModel{client: startOutputKeyLiveServer(t, liveText("Hello ", false), liveText("world.", true))},
		OutputKey: "out",
	})

	run, err := runLive(t, a)
	if err != nil {
		t.Fatalf("run error = %v, want none", err)
	}
	if got, want := run.state["out"], "Hello world."; got != want {
		t.Errorf("state[out] = %#v, want %q", got, want)
	}
}

// Text does not accumulate for an agent with either output schema, for
// thought parts, for an agent without OutputKey, for a task agent, or for the
// events of an agent this one transferred to; adk-python's
// __maybe_accumulate_streaming_output leaves each of them alone. An event
// without text adds nothing, not even an empty value.
// Fails when any of those guards is missing.
func TestOutputKey_AccumulationSkips(t *testing.T) {
	tests := []struct {
		name      string
		root      func(t *testing.T) agent.Agent
		wantState map[string]any
	}{
		{
			name: "output schema",
			root: func(t *testing.T) agent.Agent {
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: capableStreamModel{&streamModel{turns: [][]*model.LLMResponse{
						{finalParts(genai.NewPartFromText("Checking."), callPart("c1", "t", nil))},
						{finalParts(genai.NewPartFromText(`{"a": 1}`))},
					}}},
					OutputKey:        "out",
					OutputJsonSchema: mapSchema(),
					Tools:            []tool.Tool{toolT(t)},
				})
			},
			wantState: map[string]any{"out": map[string]any{"a": float64(1)}},
		},
		{
			name: "genai output schema",
			root: func(t *testing.T) agent.Agent {
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: capableStreamModel{&streamModel{turns: [][]*model.LLMResponse{
						{finalParts(genai.NewPartFromText("Checking."), callPart("c1", "t", nil))},
						{finalParts(genai.NewPartFromText(`{"a": 1}`))},
					}}},
					OutputKey:    "out",
					OutputSchema: &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"a": {Type: genai.TypeInteger}}},
					Tools:        []tool.Tool{toolT(t)},
				})
			},
			wantState: map[string]any{"out": `{"a": 1}`},
		},
		{
			name: "thought parts",
			root: func(t *testing.T) agent.Agent {
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: &streamModel{turns: [][]*model.LLMResponse{
						{finalParts(&genai.Part{Text: "Plan.", Thought: true}, callPart("c1", "t", nil))},
						{finalParts(genai.NewPartFromText("Answer."))},
					}},
					OutputKey: "out",
					Tools:     []tool.Tool{toolT(t)},
				})
			},
			wantState: map[string]any{"out": "Answer."},
		},
		{
			name: "no output key",
			root: func(t *testing.T) agent.Agent {
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: &streamModel{turns: [][]*model.LLMResponse{
						{finalParts(genai.NewPartFromText("Intro."), callPart("c1", "t", nil))},
						{finalParts(genai.NewPartFromText("Done."))},
					}},
					Tools: []tool.Tool{toolT(t)},
				})
			},
			wantState: map[string]any{},
		},
		{
			name: "task agent",
			root: func(t *testing.T) agent.Agent {
				task := newOutputKeyAgent(t, llmagent.Config{
					Name:        "a",
					Description: "test agent",
					Model: &streamModel{turns: [][]*model.LLMResponse{
						{finalParts(genai.NewPartFromText("Working."), callPart("c1", "t", nil))},
						{finalParts(genai.NewPartFromText("Done."))},
					}},
					Mode:      llmagent.ModeTask,
					OutputKey: "out",
					Tools:     []tool.Tool{toolT(t)},
				})
				seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{task}}})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
				return seq
			},
			wantState: map[string]any{},
		},
		{
			name: "transfer target's events",
			root: func(t *testing.T) agent.Agent {
				child := newOutputKeyAgent(t, llmagent.Config{
					Name:        "b",
					Description: "test agent",
					Model:       &streamModel{turns: [][]*model.LLMResponse{{finalParts(genai.NewPartFromText("Child answer."))}}},
				})
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: &streamModel{turns: [][]*model.LLMResponse{
						{finalParts(callPart("c1", "transfer_to_agent", map[string]any{"agent_name": "b"}))},
					}},
					OutputKey: "out",
					SubAgents: []agent.Agent{child},
				})
			},
			wantState: map[string]any{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			run, err := runSSE(t, tc.root(t))
			if err != nil {
				t.Fatalf("run error = %v, want none", err)
			}
			if diff := cmp.Diff(tc.wantState, run.state); diff != "" {
				t.Errorf("session state mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// With OutputJsonSchema, an event that gives OutputKey no value is not parsed
// either: the answer of the agent this one transferred to, a task agent's
// text turn, and content without text pass without an error and leave
// OutputKey unset.
// Fails when saveOutputToState parses the text of an event that
// maybeSaveOutputToState did not save.
func TestOutputKey_JSONSchemaSkips(t *testing.T) {
	tests := []struct {
		name string
		root func(t *testing.T) agent.Agent
	}{
		{
			name: "transfer target's answer",
			root: func(t *testing.T) agent.Agent {
				child := newOutputKeyAgent(t, llmagent.Config{
					Name:        "b",
					Description: "test agent",
					Model:       &streamModel{turns: [][]*model.LLMResponse{{finalParts(genai.NewPartFromText("Child answer."))}}},
				})
				return newOutputKeyAgent(t, llmagent.Config{
					Name: "a",
					Model: &streamModel{turns: [][]*model.LLMResponse{
						{finalParts(callPart("c1", "transfer_to_agent", map[string]any{"agent_name": "b"}))},
					}},
					OutputKey:        "out",
					OutputJsonSchema: mapSchema(),
					SubAgents:        []agent.Agent{child},
				})
			},
		},
		{
			name: "task agent's text turn",
			root: func(t *testing.T) agent.Agent {
				task := newOutputKeyAgent(t, llmagent.Config{
					Name:             "a",
					Description:      "test agent",
					Model:            &streamModel{turns: [][]*model.LLMResponse{{finalParts(genai.NewPartFromText("Which one?"))}}},
					Mode:             llmagent.ModeTask,
					OutputKey:        "out",
					OutputJsonSchema: mapSchema(),
				})
				seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{task}}})
				if err != nil {
					t.Fatalf("sequentialagent.New: %v", err)
				}
				return seq
			},
		},
		{
			name: "content without text",
			root: func(t *testing.T) agent.Agent {
				return newOutputKeyAgent(t, llmagent.Config{
					Name:             "a",
					Model:            &streamModel{},
					OutputKey:        "out",
					OutputJsonSchema: mapSchema(),
					BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(agent.Context) (*genai.Content, error) {
						return &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{genai.NewPartFromBytes([]byte{1}, "image/png")}}, nil
					}},
				})
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			run, err := runSSE(t, tc.root(t))
			if err != nil {
				t.Fatalf("run error = %v, want none", err)
			}
			if v, ok := run.state["out"]; ok {
				t.Errorf("state[out] = %#v, want it unset", v)
			}
		})
	}
}

// A model error reaches the caller as it is, with or without an output
// schema.
// Fails when the OutputKey handling replaces the error or fails on the nil
// event that comes with it.
func TestOutputKey_ModelErrorPassesThrough(t *testing.T) {
	tests := []struct {
		name string
		cfg  llmagent.Config
	}{
		{name: "no schema", cfg: llmagent.Config{OutputKey: "out"}},
		{name: "json schema", cfg: llmagent.Config{OutputKey: "out", OutputJsonSchema: mapSchema()}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			cfg.Name = "a"
			cfg.Model = &streamModel{}

			_, err := runSSE(t, newOutputKeyAgent(t, cfg))

			const want = "stream-model: script exhausted"
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("run error = %v, want one that contains %q", err, want)
			}
		})
	}
}
