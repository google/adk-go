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

package acpagent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/acpagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// fakeHarness is an ACP agent served over pipes. A prompt "use <kind>" asks
// permission for a tool of that kind and replies with the chosen option, and
// "ask <kind>" does the same after streaming some text first; "wait"
// reports progress and never finishes, recording the session/cancel it gets;
// "tool <name> <json args>" calls the tool of the session's MCP server and
// replies with its result; any other prompt is echoed back in two chunks,
// prefixed with the session ID.
type fakeHarness struct {
	t   *testing.T
	enc *json.Encoder
	dec *json.Decoder

	canceled chan string

	mu         sync.Mutex
	sessions   int
	prompts    []string
	pending    map[int]chan json.RawMessage
	nextID     int
	mcpServers map[string][]mcpServer // session ID -> MCP servers
}

type mcpServer struct {
	Type    string `json:"type"`
	Name    string `json:"name"`
	URL     string `json:"url"`
	Headers []any  `json:"headers"`
}

type rpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int            `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  any             `json:"result,omitempty"`
}

func startFake(t *testing.T) (*fakeHarness, func(context.Context) (io.ReadWriteCloser, error)) {
	f := &fakeHarness{t: t, pending: map[int]chan json.RawMessage{}, canceled: make(chan string, 1), mcpServers: map[string][]mcpServer{}}
	return f, func(context.Context) (io.ReadWriteCloser, error) {
		clientR, harnessW := io.Pipe()
		harnessR, clientW := io.Pipe()
		f.enc = json.NewEncoder(harnessW)
		f.dec = json.NewDecoder(harnessR)
		go f.serve()
		return pipeRWC{clientR, clientW, harnessW}, nil
	}
}

func (f *fakeHarness) send(m rpcMessage) {
	m.JSONRPC = "2.0"
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.enc.Encode(m); err != nil {
		f.t.Logf("fake harness: write: %v", err)
	}
}

func (f *fakeHarness) serve() {
	for {
		var m struct {
			ID     *int            `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
			Result json.RawMessage `json:"result"`
		}
		if err := f.dec.Decode(&m); err != nil {
			return
		}
		switch m.Method {
		case "":
			f.mu.Lock()
			ch := f.pending[*m.ID]
			f.mu.Unlock()
			ch <- m.Result
		case "initialize":
			f.send(rpcMessage{ID: m.ID, Result: map[string]any{"protocolVersion": 1, "agentCapabilities": map[string]any{}}})
		case "session/new":
			var p struct {
				MCPServers []mcpServer `json:"mcpServers"`
			}
			if err := json.Unmarshal(m.Params, &p); err != nil {
				f.t.Errorf("fake harness: bad session/new: %v", err)
			}
			f.mu.Lock()
			f.sessions++
			id := fmt.Sprintf("sess-%d", f.sessions)
			f.mcpServers[id] = p.MCPServers
			f.mu.Unlock()
			f.send(rpcMessage{ID: m.ID, Result: map[string]any{"sessionId": id}})
		case "session/prompt":
			go f.prompt(*m.ID, m.Params)
		case "session/cancel":
			var n struct {
				SessionID string `json:"sessionId"`
			}
			if err := json.Unmarshal(m.Params, &n); err != nil {
				f.t.Errorf("fake harness: bad cancel: %v", err)
			}
			f.canceled <- n.SessionID
		}
	}
}

func (f *fakeHarness) prompt(id int, params json.RawMessage) {
	var req struct {
		SessionID string `json:"sessionId"`
		Prompt    []struct {
			Text string `json:"text"`
		} `json:"prompt"`
	}
	if err := json.Unmarshal(params, &req); err != nil {
		f.t.Errorf("fake harness: bad prompt: %v", err)
		return
	}
	var texts []string
	for _, c := range req.Prompt {
		texts = append(texts, c.Text)
	}
	text := strings.Join(texts, "|")
	f.mu.Lock()
	f.prompts = append(f.prompts, text)
	f.mu.Unlock()

	if text == "wait" {
		// Report progress, then work until canceled.
		f.chunk(req.SessionID, "working")
		return
	}
	reply := []string{req.SessionID + " echo: ", text}
	if kind, ok := strings.CutPrefix(text, "use "); ok {
		reply = []string{"permission: " + f.askPermission(req.SessionID, kind)}
	}
	if call, ok := strings.CutPrefix(text, "tool "); ok {
		reply = []string{f.callTool(req.SessionID, call)}
	}
	if kind, ok := strings.CutPrefix(text, "ask "); ok {
		f.chunk(req.SessionID, "checking. ")
		reply = []string{"permission: " + f.askPermission(req.SessionID, kind)}
	}
	for _, chunk := range reply {
		f.chunk(req.SessionID, chunk)
	}
	f.send(rpcMessage{ID: &id, Result: map[string]any{"stopReason": "end_turn"}})
}

func (f *fakeHarness) chunk(sessionID, text string) {
	f.send(rpcMessage{Method: "session/update", Params: mustJSON(map[string]any{
		"sessionId": sessionID,
		"update": map[string]any{
			"sessionUpdate": "agent_message_chunk",
			"content":       map[string]any{"type": "text", "text": text},
		},
	})})
}

func (f *fakeHarness) askPermission(sessionID, kind string) string {
	f.mu.Lock()
	f.nextID++
	id := 1000 + f.nextID
	ch := make(chan json.RawMessage, 1)
	f.pending[id] = ch
	f.mu.Unlock()
	f.send(rpcMessage{ID: &id, Method: "session/request_permission", Params: mustJSON(map[string]any{
		"sessionId": sessionID,
		"toolCall":  map[string]any{"toolCallId": "tc1", "title": "run " + kind, "kind": kind},
		"options": []map[string]any{
			{"optionId": "yes", "name": "Allow", "kind": "allow_once"},
			{"optionId": "no", "name": "Reject", "kind": "reject_once"},
		},
	})})
	var resp struct {
		Outcome struct {
			Outcome  string `json:"outcome"`
			OptionID string `json:"optionId"`
		} `json:"outcome"`
	}
	if err := json.Unmarshal(<-ch, &resp); err != nil {
		f.t.Errorf("fake harness: bad permission response: %v", err)
	}
	if resp.Outcome.OptionID != "" {
		return resp.Outcome.OptionID
	}
	return resp.Outcome.Outcome
}

// callTool calls a tool of the session's MCP server: call is the tool name,
// a space and its JSON arguments.
func (f *fakeHarness) callTool(sessionID, call string) string {
	name, args, _ := strings.Cut(call, " ")
	f.mu.Lock()
	servers := f.mcpServers[sessionID]
	f.mu.Unlock()
	if len(servers) != 1 || servers[0].Type != "http" {
		return fmt.Sprintf("unexpected MCP servers: %+v", servers)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "fake-harness"}, nil)
	cs, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: servers[0].URL}, nil)
	if err != nil {
		return "connect error: " + err.Error()
	}
	defer func() { _ = cs.Close() }()
	res, err := cs.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: json.RawMessage(args)})
	if err != nil {
		return "call error: " + err.Error()
	}
	var sb strings.Builder
	if res.IsError {
		sb.WriteString("tool error: ")
	} else {
		sb.WriteString("result: ")
	}
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			sb.WriteString(tc.Text)
		}
	}
	return sb.String()
}

func (f *fakeHarness) recordedPrompts() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.prompts...)
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

// pipeRWC is the client end of the pipes. Close closes both directions, the
// way the stdio of an exiting process would.
type pipeRWC struct {
	r        *io.PipeReader
	w        *io.PipeWriter
	harnessW *io.PipeWriter
}

func (p pipeRWC) Read(b []byte) (int, error)  { return p.r.Read(b) }
func (p pipeRWC) Write(b []byte) (int, error) { return p.w.Write(b) }
func (p pipeRWC) Close() error {
	return errors.Join(p.w.Close(), p.harnessW.Close())
}

func newRunner(t *testing.T, cfg acpagent.Config) *runner.Runner {
	t.Helper()
	r, _ := newRunnerWithSessions(t, cfg)
	return r
}

func newRunnerWithSessions(t *testing.T, cfg acpagent.Config) (*runner.Runner, session.Service) {
	t.Helper()
	cfg.Name = "coder"
	ag, closer, err := acpagent.New(t.Context(), cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := closer.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             ag,
		SessionService:    sessions,
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r, sessions
}

// reply runs one turn and returns the final text of the agent.
func reply(t *testing.T, r *runner.Runner, sessionID, text string, mode agent.StreamingMode) string {
	t.Helper()
	// A turn which waits for the harness, for example a message queued behind
	// a pending question, fails here instead of hanging the test.
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	var final string
	for ev, err := range r.Run(ctx, "user", sessionID, genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{StreamingMode: mode}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.ErrorMessage != "" {
			t.Fatalf("Run() event error = %s", ev.ErrorMessage)
		}
		if ev.Partial || ev.Content == nil {
			continue
		}
		var sb strings.Builder
		for _, p := range ev.Content.Parts {
			sb.WriteString(p.Text)
		}
		final = sb.String()
	}
	return final
}

func TestAgent_Conversation(t *testing.T) {
	for _, mode := range []agent.StreamingMode{agent.StreamingModeSSE, agent.StreamingModeNone} {
		t.Run(string(mode), func(t *testing.T) {
			fake, start := startFake(t)
			r := newRunner(t, acpagent.Config{Start: start})

			got := []string{
				reply(t, r, "s1", "hello", mode),
				reply(t, r, "s1", "again", mode),
				reply(t, r, "s2", "other", mode),
			}
			want := []string{"sess-1 echo: hello", "sess-1 echo: again", "sess-2 echo: other"}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}
			// The harness keeps the history, so only new messages are sent.
			if diff := cmp.Diff([]string{"hello", "again", "other"}, fake.recordedPrompts()); diff != "" {
				t.Errorf("prompts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgent_AbandonedRunCancelsHarness(t *testing.T) {
	fake, start := startFake(t)
	r := newRunner(t, acpagent.Config{Start: start})
	for _, err := range r.Run(t.Context(), "user", "s1", genai.NewContentFromText("wait", genai.RoleUser), agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		break
	}
	select {
	case got := <-fake.canceled:
		if got != "sess-1" {
			t.Errorf("canceled session = %q, want %q", got, "sess-1")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the harness was not asked to cancel the abandoned turn")
	}
}

func TestAgent_Permissions(t *testing.T) {
	tests := []struct {
		kind string
		want string
	}{
		{kind: "read", want: "permission: yes"},
		{kind: "edit", want: "permission: yes"},
		{kind: "delete", want: "permission: no"},
	}
	_, start := startFake(t)
	r := newRunner(t, acpagent.Config{Start: start, AllowTools: []string{"read", "ed*"}, DenyTools: []string{"delete"}})
	for _, tc := range tests {
		t.Run(tc.kind, func(t *testing.T) {
			if got := reply(t, r, "s-"+tc.kind, "use "+tc.kind, agent.StreamingModeSSE); got != tc.want {
				t.Errorf("reply = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestAgent_AnswerInLaterInvocation(t *testing.T) {
	tests := []struct {
		answer string
		want   string
	}{
		{answer: "once", want: "permission: yes"},
		{answer: "no", want: "permission: no"},
	}
	for _, mode := range []agent.StreamingMode{agent.StreamingModeSSE, agent.StreamingModeNone} {
		for _, tc := range tests {
			t.Run(string(mode)+"/"+tc.answer, func(t *testing.T) {
				fake, start := startFake(t)
				r := newRunner(t, acpagent.Config{Start: start, AllowTools: []string{"read"}})

				question := reply(t, r, "s1", "ask execute", mode)
				if !strings.Contains(question, "run execute") {
					t.Fatalf("first reply = %q, want the permission question", question)
				}
				if got := reply(t, r, "s1", tc.answer, mode); got != tc.want {
					t.Errorf("reply to the answer = %q, want %q", got, tc.want)
				}
				// The answer continues the pending prompt rather than starting one.
				if diff := cmp.Diff([]string{"ask execute"}, fake.recordedPrompts()); diff != "" {
					t.Errorf("prompts mismatch (-want +got):\n%s", diff)
				}
				// The task is finished, so the next message is a new prompt.
				if got, want := reply(t, r, "s1", "next", mode), "sess-1 echo: next"; got != want {
					t.Errorf("reply after the answer = %q, want %q", got, want)
				}
			})
		}
	}
}

// noteTool saves its text in the session state under "note", and fails for
// an empty text.
func noteTool(t *testing.T) tool.Tool {
	t.Helper()
	type args struct {
		Text string `json:"text"`
	}
	type result struct {
		Saved string `json:"saved"`
	}
	nt, err := functiontool.New(functiontool.Config{Name: "save_note", Description: "Saves a note."},
		func(ctx agent.Context, a args) (result, error) {
			if a.Text == "" {
				return result{}, errors.New("empty note")
			}
			if err := ctx.State().Set("note", a.Text); err != nil {
				return result{}, err
			}
			return result{Saved: a.Text}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return nt
}

func TestAgent_Tools(t *testing.T) {
	for _, mode := range []agent.StreamingMode{agent.StreamingModeSSE, agent.StreamingModeNone} {
		t.Run(string(mode), func(t *testing.T) {
			_, start := startFake(t)
			r, sessions := newRunnerWithSessions(t, acpagent.Config{Start: start, Tools: []tool.Tool{noteTool(t)}})

			got := []string{
				reply(t, r, "s1", `tool save_note {"text":"first"}`, mode),
				// A later invocation of the session runs the tool too.
				reply(t, r, "s1", `tool save_note {"text":"second"}`, mode),
				reply(t, r, "s1", `tool save_note {"text":""}`, mode),
				// The harness keeps its session, so tools don't start a new one.
				reply(t, r, "s1", "hello", mode),
			}
			want := []string{
				`result: {"saved":"first"}`,
				`result: {"saved":"second"}`,
				"tool error: empty note",
				"sess-1 echo: hello",
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("replies mismatch (-want +got):\n%s", diff)
			}

			resp, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s1"})
			if err != nil {
				t.Fatalf("Get() error = %v", err)
			}
			if got, err := resp.Session.State().Get("note"); err != nil || got != "second" {
				t.Errorf("state note = %v, %v, want %q", got, err, "second")
			}
			var calls, responses []string
			for ev := range resp.Session.Events().All() {
				if ev.Author != "coder" || ev.Content == nil {
					continue
				}
				for _, p := range ev.Content.Parts {
					if p.FunctionCall != nil {
						calls = append(calls, fmt.Sprint(p.FunctionCall.Args["text"]))
					}
					if p.FunctionResponse != nil {
						responses = append(responses, fmt.Sprint(p.FunctionResponse.Response))
					}
				}
			}
			if diff := cmp.Diff([]string{"first", "second", ""}, calls); diff != "" {
				t.Errorf("function calls mismatch (-want +got):\n%s", diff)
			}
			wantResponses := []string{"map[saved:first]", "map[saved:second]", "map[error:empty note]"}
			if diff := cmp.Diff(wantResponses, responses); diff != "" {
				t.Errorf("function responses mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgent_ToolsAbandonedRunKeepsSession(t *testing.T) {
	_, start := startFake(t)
	r := newRunner(t, acpagent.Config{Start: start, Tools: []tool.Tool{noteTool(t)}})
	msg := genai.NewContentFromText(`tool save_note {"text":"x"}`, genai.RoleUser)
	// Stop at the tool's response, so that it is the agent's last event.
	for ev, err := range r.Run(t.Context(), "user", "s1", msg, agent.RunConfig{StreamingMode: agent.StreamingModeSSE}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
			break
		}
	}
	if got, want := reply(t, r, "s1", "hello", agent.StreamingModeSSE), "sess-1 echo: hello"; got != want {
		t.Errorf("reply after the abandoned run = %q, want %q", got, want)
	}
}

func TestAgent_ToolsPerSession(t *testing.T) {
	fake, start := startFake(t)
	r := newRunner(t, acpagent.Config{Start: start, Tools: []tool.Tool{noteTool(t)}})
	reply(t, r, "s1", "hello", agent.StreamingModeSSE)
	reply(t, r, "s2", "hello", agent.StreamingModeSSE)

	fake.mu.Lock()
	s1, s2 := fake.mcpServers["sess-1"], fake.mcpServers["sess-2"]
	fake.mu.Unlock()
	if len(s1) != 1 || len(s2) != 1 {
		t.Fatalf("MCP servers = %v and %v, want one per session", s1, s2)
	}
	if s1[0].URL == s2[0].URL {
		t.Errorf("sessions share the MCP endpoint %s", s1[0].URL)
	}
}

func TestNew_Validation(t *testing.T) {
	start := func(context.Context) (io.ReadWriteCloser, error) { return nil, io.EOF }
	for name, cfg := range map[string]acpagent.Config{
		"neither": {Name: "a"},
		"both":    {Name: "a", Command: []string{"x"}, Start: start},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := acpagent.New(t.Context(), cfg); err == nil {
				t.Error("New() error = nil, want an error")
			}
		})
	}
}
