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

package toolcallintegrity_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/toolcallintegrity"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Stamps produced by adk-python v2.12.0 for the fixture: app "app", user
// "user", session "session", invocation "inv", no branch, author "agent",
// call "fc-1" named "transfer_money".
const (
	// stampA10 is the stamp of args {"amount": 10} under "test-secret-key".
	stampA10 = "v1:cde2f3d2ff68ef038475414221cfd807f5a3e637a3d85aad72d646048bea79cf"
	// stampA20 is the stamp of args {"amount": 20} under "test-secret-key".
	stampA20 = "v1:8a43a352b24bd3b7c5aaabf558057d8db253079b84e5d43437353ce44ded7ef9"
	// stampOldA10 is the stamp of args {"amount": 10} under "old".
	stampOldA10 = "v1:666f831d0386e6b345f9d105fedfb165330e91d994d78e9ef9a5ffad98602762"

	stampKey     = "__adk_internal_fc_hmac"
	restoredKey  = "__adk_internal_restored_event"
	tamperedText = "transfer_money call 'fc-1' does not match its integrity stamp"
	badKeysText  = "toolcallintegrity: each key must be non-empty bytes, and SecretKey must be set"
)

var testKey = []byte("test-secret-key")

// fakeSession is a session whose identity and events the test sets directly.
type fakeSession struct {
	app, user, id string
	events        []*session.Event
}

func newSession(events ...*session.Event) *fakeSession {
	return &fakeSession{app: "app", user: "user", id: "session", events: events}
}

func (s *fakeSession) ID() string                { return s.id }
func (s *fakeSession) AppName() string           { return s.app }
func (s *fakeSession) UserID() string            { return s.user }
func (s *fakeSession) State() session.State      { return nil }
func (s *fakeSession) Events() session.Events    { return fakeEvents(s.events) }
func (s *fakeSession) LastUpdateTime() time.Time { return time.Time{} }

type fakeEvents []*session.Event

func (e fakeEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}
func (e fakeEvents) Len() int                { return len(e) }
func (e fakeEvents) At(i int) *session.Event { return e[i] }

// namedTool is a tool.Tool with a name and nothing else.
type namedTool string

func (n namedTool) Name() string        { return string(n) }
func (n namedTool) Description() string { return "" }
func (n namedTool) IsLongRunning() bool { return false }

func call(id, name string, args map[string]any) *genai.FunctionCall {
	return &genai.FunctionCall{ID: id, Name: name, Args: args}
}

func amount(n any) map[string]any { return map[string]any{"amount": n} }

// callEvent returns the fixture event "ev-1" holding calls.
func callEvent(calls ...*genai.FunctionCall) *session.Event {
	content := &genai.Content{Role: "model"}
	for _, fc := range calls {
		content.Parts = append(content.Parts, &genai.Part{FunctionCall: fc})
	}
	ev := &session.Event{ID: "ev-1", InvocationID: "inv", Author: "agent"}
	ev.Content = content
	return ev
}

func newPlugin(t *testing.T, cfg toolcallintegrity.Config) *plugin.Plugin {
	t.Helper()
	p, err := toolcallintegrity.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p
}

func invocation(t *testing.T, sess session.Session) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: sess})
}

// stamp runs the plugin's on-event callback on ev as an event of sess.
func stamp(t *testing.T, p *plugin.Plugin, sess session.Session, ev *session.Event) {
	t.Helper()
	got, err := p.OnEventCallback()(invocation(t, sess), ev)
	if err != nil || got != nil {
		t.Fatalf("OnEventCallback() = %v, %v, want nil, nil", got, err)
	}
}

// stampsOf returns the stamp map ev carries, or nil.
func stampsOf(ev *session.Event) any {
	return ev.CustomMetadata[stampKey]
}

func beforeRun(t *testing.T, p *plugin.Plugin, sess session.Session) error {
	t.Helper()
	got, err := p.BeforeRunCallback()(invocation(t, sess))
	if got != nil {
		t.Fatalf("BeforeRunCallback() content = %v, want nil", got)
	}
	return err
}

func toolContext(t *testing.T, sess session.Session, callID string) agent.Context {
	t.Helper()
	return agent.NewToolContext(invocation(t, sess), callID, nil, nil)
}

func beforeTool(t *testing.T, p *plugin.Plugin, ctx agent.Context, toolName string, args map[string]any) error {
	t.Helper()
	got, err := p.BeforeToolCallback()(ctx, namedTool(toolName), args)
	if got != nil {
		t.Fatalf("BeforeToolCallback() result = %v, want nil", got)
	}
	return err
}

// checkIntegrityError reports whether err is an integrity error with text want.
func checkIntegrityError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Fatalf("error = nil, want %q", want)
	}
	if err.Error() != want {
		t.Errorf("error = %q, want %q", err.Error(), want)
	}
	if !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("errors.Is(%q, ErrIntegrity) = false, want true", err.Error())
	}
}

func TestNew_RejectsBadKeys(t *testing.T) {
	tests := []struct {
		name    string
		cfg     toolcallintegrity.Config
		wantErr bool
	}{
		{name: "zero config", cfg: toolcallintegrity.Config{}, wantErr: true},
		{name: "empty secret key", cfg: toolcallintegrity.Config{SecretKey: []byte{}}, wantErr: true},
		{name: "empty verification key", cfg: toolcallintegrity.Config{SecretKey: testKey, VerificationKeys: [][]byte{[]byte("old"), {}}}, wantErr: true},
		{name: "nil verification key", cfg: toolcallintegrity.Config{SecretKey: testKey, VerificationKeys: [][]byte{nil}}, wantErr: true},
		{name: "secret key", cfg: toolcallintegrity.Config{SecretKey: testKey}},
		{name: "secret and verification key", cfg: toolcallintegrity.Config{SecretKey: testKey, VerificationKeys: [][]byte{[]byte("old")}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := toolcallintegrity.New(tc.cfg)
			if !tc.wantErr {
				if p == nil || err != nil {
					t.Fatalf("New() = %v, %v, want a plugin and nil", p, err)
				}
				return
			}
			if p != nil {
				t.Errorf("New() plugin = %v, want nil", p)
			}
			if err == nil {
				t.Fatalf("New() error = nil, want %q", badKeysText)
			}
			if err.Error() != badKeysText {
				t.Errorf("New() error = %q, want %q", err.Error(), badKeysText)
			}
			if errors.Is(err, toolcallintegrity.ErrIntegrity) {
				t.Errorf("errors.Is(New() error, ErrIntegrity) = true, want false")
			}
		})
	}
}

func TestNew_Name(t *testing.T) {
	tests := []struct{ name, want string }{
		{name: "", want: "tool_call_integrity"},
		{name: "tci_2", want: "tci_2"},
	}
	for _, tc := range tests {
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, Name: tc.name})
		if got := p.Name(); got != tc.want {
			t.Errorf("New(Config{Name: %q}).Name() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestNew_CopiesKeys(t *testing.T) {
	t.Run("secret key", func(t *testing.T) {
		buf := []byte("test-secret-key")
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: buf})
		for i := range buf {
			buf[i] = 'x'
		}
		ev := callEvent(call("fc-1", "transfer_money", amount(10)))
		stamp(t, p, newSession(), ev)
		if diff := cmp.Diff(map[string]any{"fc-1": stampA10}, stampsOf(ev)); diff != "" {
			t.Errorf("stamps mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("verification key", func(t *testing.T) {
		oldBuf := []byte("old")
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: []byte("new"), VerificationKeys: [][]byte{oldBuf}})
		for i := range oldBuf {
			oldBuf[i] = 'x'
		}
		ev := callEvent(call("fc-1", "transfer_money", amount(10)))
		ev.CustomMetadata = map[string]any{stampKey: map[string]any{"fc-1": stampOldA10}}
		if err := beforeRun(t, p, newSession(ev)); err != nil {
			t.Errorf("BeforeRunCallback() error = %v, want nil", err)
		}
	})
}

func TestOnEvent_StampsMatchAdkPythonLiterals(t *testing.T) {
	negZero := math.Copysign(0, -1)
	tests := []struct {
		name                       string
		key                        string
		verificationKey            string
		invocation, branch, author string
		callName                   string
		args                       map[string]any
		want                       string
	}{
		{
			name: "C1", key: "test-secret-key", invocation: "inv", author: "agent",
			callName: "transfer_money", args: amount(10), want: stampA10,
		},
		{
			name: "C1 signed by SecretKey beside a verification key", key: "test-secret-key", verificationKey: "old",
			invocation: "inv", author: "agent", callName: "transfer_money", args: amount(10), want: stampA10,
		},
		{
			name: "C2 strings, nesting and whole-number floats", key: "test-secret-key",
			invocation: "e-42", branch: "root.sub", author: "bank_agent", callName: "transfer_money",
			args: map[string]any{
				"memo": "café <b>&</b> \u2028\u2029 \x7f \x1f \n\t\"\\ 😀",
				"n":    10.0,
				"l":    []any{1, map[string]any{"m": 2.0}},
				"x":    1.5,
				"none": nil,
				"t":    true,
				"neg0": negZero,
			},
			want: "v1:21160b9d18843d5220b73a69e5a95adf2e2d48afa360a1eef107cbfac82586f3",
		},
		{
			name: "C3 other key", key: "old", invocation: "inv", author: "agent",
			callName: "transfer_money", args: amount(10), want: stampOldA10,
		},
		{
			name: "C4 float forms", key: "k", invocation: "inv", author: "agent", callName: "transfer_money",
			args: map[string]any{
				"a": 1e-05, "b": 0.0001, "c": 1.5e-10, "d": 1e16,
				"e": 123.456, "f": 9007199254740994.0, "g": 1e300, "h": 0.1,
			},
			want: "v1:44db69263caa7cfe719316d79bafc6ca2fe5a82da226072e0eb8f606fc2f5360",
		},
		{
			name: "C5 NaN and infinities", key: "k", invocation: "inv", author: "agent", callName: "transfer_money",
			args: map[string]any{"nan": math.NaN(), "inf": math.Inf(1), "ninf": math.Inf(-1)},
			want: "v1:02d50fa5480ce248a9b1afef00ea57002e6cb031e4530ba119ef84e3b0bb7758",
		},
		{
			name: "C6 no name, nil args", key: "k", invocation: "inv", author: "agent",
			want: "v1:9269089cd99d1f3a6e2eb725d828f96f714d1a7885c0c7e41fd5bfd1d5825d7a",
		},
		{
			name: "C6 no name, empty args", key: "k", invocation: "inv", author: "agent", args: map[string]any{},
			want: "v1:9269089cd99d1f3a6e2eb725d828f96f714d1a7885c0c7e41fd5bfd1d5825d7a",
		},
		{
			name: "C7 key order", key: "k", invocation: "inv", author: "agent", callName: "transfer_money",
			args: map[string]any{"b": 1, "a": 2, "é": 3, "Z": 4, "aa": 5, "😀": 6, "\uffff": 7},
			want: "v1:561515ed31c418515114360fd0206afe7a06720e97699bf4e64ea49dbc180228",
		},
		{
			name: "C8 big integer", key: "k", invocation: "inv", author: "agent", callName: "transfer_money",
			args: map[string]any{"big": json.Number("12345678901234567890123")},
			want: "v1:165272ef98bda79ee519fc5c63a777d060d719c9df2552a15b976923bf4daa44",
		},
		{
			name: "C10 control characters", key: "k", invocation: "inv", author: "agent", callName: "transfer_money",
			args: map[string]any{"s": "\x00\x08\x0c\r\x1b\u00ad\ufeff"},
			want: "v1:46f5223443d2ffc80f1bc2c88e0c765a4765cd28a63648c432f1df347e3ea951",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := toolcallintegrity.Config{SecretKey: []byte(tc.key)}
			if tc.verificationKey != "" {
				cfg.VerificationKeys = [][]byte{[]byte(tc.verificationKey)}
			}
			p := newPlugin(t, cfg)
			ev := callEvent(call("fc-1", tc.callName, tc.args))
			ev.InvocationID, ev.Branch, ev.Author = tc.invocation, tc.branch, tc.author
			stamp(t, p, newSession(), ev)
			if diff := cmp.Diff(map[string]any{"fc-1": tc.want}, stampsOf(ev)); diff != "" {
				t.Errorf("stamps mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOnEvent_EquivalentArgsStampAlike(t *testing.T) {
	const (
		ten      = "v1:b3bf896c94505b5b528401bcbf7a078ab3696208de67eb67365e9a38e174108d"
		hundred  = "v1:45c11bde0351dddb75ccb6219cfcdf73f062eff9577241f1d2252c5da8f456d5"
		oneTenth = "v1:8efc7f7910b26892f4826429da18bf8b6f4a8a718204a1e4a9ebb5f9e22df6b4"
		e21      = "v1:308379ff34d46766d982d3dbc31d38509051dca480085cacb18760c0006d6550"
		zero     = "v1:c5445e96ad241ac7ca50fdce3c8fc20ee1f3e2b743bf70ab1bd7fa392a98a666"
		infinite = "v1:392b1cbf52c9b0ee1d22540d6acf12f474b3bd46147b137f319aba64fa33941c"
		small    = "v1:765a1527c7c0457e82ba1543f90e3b3e443c9aec8897b9cf79dda8b8f4e05174"
		fffd     = "v1:66ceaffe66273b7a8e5ba88a90a5457e75e1e4d7a11dd79a5d165996a0cdb0c2"
		typedMap = "v1:e97aa8181e0e03295269910a62c3e07566001fb6070aee0072e490e7db08ba3f"
	)
	n := func(v any) map[string]any { return map[string]any{"n": v} }
	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{"int", n(int(10)), ten},
		{"int64", n(int64(10)), ten},
		{"uint8", n(uint8(10)), ten},
		{"float32 whole", n(float32(10)), ten},
		{"float64 whole", n(float64(10)), ten},
		{"json.Number 10", n(json.Number("10")), ten},
		{"json.Number 10.0", n(json.Number("10.0")), ten},
		{"json.Number 1e2", n(json.Number("1e2")), hundred},
		{"json.Number 0.1", n(json.Number("0.1")), oneTenth},
		{"float32 0.1", n(float32(0.1)), oneTenth},
		{"json.Number 1e+21", n(json.Number("1e+21")), e21},
		{"float64 1e21", n(float64(1e21)), e21},
		{"json.Number -0", n(json.Number("-0")), zero},
		{"json.Number -0.0", n(json.Number("-0.0")), zero},
		{"json.Number 1e-400", n(json.Number("1e-400")), zero},
		{"json.Number 1e400", n(json.Number("1e400")), infinite},
		{"json.Number -1e400", n(json.Number("-1e400")), infinite},
		{"json.Number 1.5E-10", n(json.Number("1.5E-10")), small},
		{"invalid UTF-8", map[string]any{"s": "\xff"}, fffd},
		{"replacement character", map[string]any{"s": "\ufffd"}, fffd},
		{"typed map", map[string]any{"m": map[string]string{"a": "b"}}, typedMap},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: []byte("k")})
			ev := callEvent(call("fc-1", "transfer_money", tc.args))
			stamp(t, p, newSession(), ev)
			if diff := cmp.Diff(map[string]any{"fc-1": tc.want}, stampsOf(ev)); diff != "" {
				t.Errorf("stamps mismatch (-want +got):\n%s", diff)
			}
		})
	}
	for _, literal := range []string{"+1", "NaN"} {
		t.Run("invalid json.Number "+literal, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: []byte("k")})
			ev := callEvent(call("fc-1", "transfer_money", n(json.Number(literal))))
			_, err := p.OnEventCallback()(invocation(t, newSession()), ev)
			const want = "toolcallintegrity: cannot encode the args of transfer_money call 'fc-1': value of type json.Number is not JSON"
			if err == nil || err.Error() != want {
				t.Fatalf("OnEventCallback() error = %v, want %q", err, want)
			}
			if errors.Is(err, toolcallintegrity.ErrIntegrity) {
				t.Errorf("errors.Is(OnEventCallback() error, ErrIntegrity) = true, want false")
			}
		})
	}
}

func TestOnEvent_StampsOnlyCallsWithID(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	t.Run("calls with and without ID", func(t *testing.T) {
		ev := callEvent(call("fc-1", "transfer_money", amount(10)), call("", "lookup", nil))
		stamp(t, p, newSession(), ev)
		if diff := cmp.Diff(map[string]any{"fc-1": stampA10}, stampsOf(ev)); diff != "" {
			t.Errorf("stamps mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("existing metadata", func(t *testing.T) {
		shared := map[string]any{"keep": "v"}
		ev := callEvent(call("fc-1", "transfer_money", amount(10)))
		ev.CustomMetadata = shared
		stamp(t, p, newSession(), ev)
		want := map[string]any{"keep": "v", stampKey: map[string]any{"fc-1": stampA10}}
		if diff := cmp.Diff(want, ev.CustomMetadata); diff != "" {
			t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]any{"keep": "v"}, shared); diff != "" {
			t.Errorf("the map the event held before was written (-want +got):\n%s", diff)
		}
	})
	t.Run("only calls without ID", func(t *testing.T) {
		ev := callEvent(call("", "lookup", nil), call("", "transfer_money", amount(10)))
		stamp(t, p, newSession(), ev)
		if ev.CustomMetadata != nil {
			t.Errorf("CustomMetadata = %v, want nil", ev.CustomMetadata)
		}
	})
}

func TestOnEvent_RefusesRepeatedCallID(t *testing.T) {
	tests := []struct {
		name        string
		allow       bool
		repeatValue int
	}{
		{name: "same args", repeatValue: 10},
		{name: "other args", repeatValue: 20},
		{name: "other args, unstamped calls allowed", allow: true, repeatValue: 20},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: tc.allow})
			ev := callEvent(
				call("fc-1", "transfer_money", amount(10)),
				call("fc-2", "transfer_money", amount(10)),
				call("fc-1", "transfer_money", amount(tc.repeatValue)),
			)
			ev.ID = "ev-7"
			got, err := p.OnEventCallback()(invocation(t, newSession()), ev)
			if got != nil {
				t.Errorf("OnEventCallback() event = %v, want nil", got)
			}
			checkIntegrityError(t, err, "event 'ev-7' repeats function call ID 'fc-1'")
			if ev.CustomMetadata != nil {
				t.Errorf("CustomMetadata = %v, want nil", ev.CustomMetadata)
			}
		})
	}
}

func TestOnEvent_PartialEventKeepsLastStampForRepeatedID(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	ev := callEvent(call("fc-1", "transfer_money", amount(10)), call("fc-1", "transfer_money", amount(20)))
	ev.Partial = true
	stamp(t, p, newSession(), ev)
	if diff := cmp.Diff(map[string]any{"fc-1": stampA20}, stampsOf(ev)); diff != "" {
		t.Errorf("stamps mismatch (-want +got):\n%s", diff)
	}
}

// stampedEvent returns the fixture event with call fc-1, args
// {"amount": 10}, and stamps as its stamp map.
func stampedEvent(stamps any) *session.Event {
	ev := callEvent(call("fc-1", "transfer_money", amount(10)))
	ev.CustomMetadata = map[string]any{stampKey: stamps}
	return ev
}

// recordingHandler is a slog.Handler that keeps every record it handles.
type recordingHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (h *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler       { return h }
func (h *recordingHandler) WithGroup(string) slog.Handler            { return h }
func (h *recordingHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r)
	return nil
}

func (h *recordingHandler) snapshot() []slog.Record {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.records)
}

func attrsOf(r slog.Record) map[string]any {
	attrs := map[string]any{}
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.Any()
		return true
	})
	return attrs
}

// checkUnstampedWarning reports whether h holds exactly one warning, for the
// fc-1 call of toolName.
func checkUnstampedWarning(t *testing.T, h *recordingHandler, toolName string) {
	t.Helper()
	records := h.snapshot()
	if len(records) != 1 {
		t.Fatalf("got %d log records, want 1", len(records))
	}
	if records[0].Level != slog.LevelWarn {
		t.Errorf("record level = %v, want %v", records[0].Level, slog.LevelWarn)
	}
	want := map[string]any{"tool": toolName, "function_call_id": "fc-1"}
	if diff := cmp.Diff(want, attrsOf(records[0])); diff != "" {
		t.Errorf("record attributes mismatch (-want +got):\n%s", diff)
	}
}

func TestBeforeRun_RejectsTamperedStamps(t *testing.T) {
	digestA10 := strings.TrimPrefix(stampA10, "v1:")
	tests := []struct {
		name  string
		edit  func(*fakeSession, *session.Event)
		want  string
		calls []*genai.FunctionCall
	}{
		{name: "author", edit: func(_ *fakeSession, ev *session.Event) { ev.Author = "other" }, want: tamperedText},
		{name: "invocation", edit: func(_ *fakeSession, ev *session.Event) { ev.InvocationID = "inv2" }, want: tamperedText},
		{name: "user", edit: func(s *fakeSession, _ *session.Event) { s.user = "user2" }, want: tamperedText},
		{name: "session", edit: func(s *fakeSession, _ *session.Event) { s.id = "session2" }, want: tamperedText},
		{
			name: "stamp without prefix",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.CustomMetadata[stampKey] = map[string]any{"fc-1": digestA10}
			},
			want: tamperedText,
		},
		{
			name: "stamp with another prefix",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.CustomMetadata[stampKey] = map[string]any{"fc-1": "v2:" + digestA10}
			},
			want: tamperedText,
		},
		{
			name: "duplicate call ID",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.Content.Parts = append(ev.Content.Parts, &genai.Part{FunctionCall: call("fc-1", "transfer_money", amount(10))})
			},
			want: "event 'ev-1' has duplicate function call ID 'fc-1'",
		},
		{
			name: "leftover stamp",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.CustomMetadata[stampKey] = map[string]any{"fc-1": stampA10, "fc-9": "v1:00"}
			},
			want: "event 'ev-1' has integrity stamps with no matching call: ['fc-9']",
		},
		{
			name: "leftover stamps",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.CustomMetadata[stampKey] = map[string]any{"fc-9": "v1:00", "fc-1": stampA10, "fc-8": "v1:00"}
			},
			want: "event 'ev-1' has integrity stamps with no matching call: ['fc-8', 'fc-9']",
		},
		{
			// The ids sort before quoting: "fc-9" < "fc-9 b", while "'fc-9'" > "'fc-9 b'".
			name: "leftover stamps sorted by bare id",
			edit: func(_ *fakeSession, ev *session.Event) {
				ev.CustomMetadata[stampKey] = map[string]any{"fc-9 b": "v1:00", "fc-1": stampA10, "fc-9": "v1:00"}
			},
			want: "event 'ev-1' has integrity stamps with no matching call: ['fc-9', 'fc-9 b']",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
			ev := stampedEvent(map[string]any{"fc-1": stampA10})
			sess := newSession(ev)
			tc.edit(sess, ev)
			checkIntegrityError(t, beforeRun(t, p, sess), tc.want)
		})
	}
}

func TestBeforeRun_StampMapForms(t *testing.T) {
	const malformed = "event 'ev-1' has a malformed integrity stamp"
	tests := []struct {
		name   string
		stamps any
		want   string // empty: no error
	}{
		{name: "map[string]any", stamps: map[string]any{"fc-1": stampA10}},
		{name: "map[string]string", stamps: map[string]string{"fc-1": stampA10}},
		{name: "nil", stamps: nil, want: "transfer_money call 'fc-1' has no integrity stamp"},
		{name: "string", stamps: "x", want: malformed},
		{name: "number stamp", stamps: map[string]any{"fc-1": 5}, want: malformed},
		{name: "nil stamp value", stamps: map[string]any{"fc-1": nil}, want: malformed},
		{name: "non-ASCII stamp", stamps: map[string]any{"fc-1": "v1:é"}, want: malformed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
			err := beforeRun(t, p, newSession(stampedEvent(tc.stamps)))
			if tc.want == "" {
				if err != nil {
					t.Fatalf("BeforeRunCallback() error = %v, want nil", err)
				}
				return
			}
			checkIntegrityError(t, err, tc.want)
		})
	}
}

func TestBeforeRun_SkipsEventsWithoutCalls(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	for _, stamps := range []any{"x", map[string]any{"fc-1": "v1:00"}} {
		text := &session.Event{ID: "ev-t", Author: "agent"}
		text.Content = genai.NewContentFromText("hello", "model")
		text.CustomMetadata = map[string]any{stampKey: stamps}
		response := &session.Event{ID: "ev-r", Author: "user"}
		response.Content = genai.NewContentFromFunctionResponse("transfer_money", map[string]any{"ok": true}, "user")
		response.CustomMetadata = map[string]any{stampKey: stamps}
		if err := beforeRun(t, p, newSession(text, response)); err != nil {
			t.Errorf("BeforeRunCallback() with stamps %v on events without calls: error = %v, want nil", stamps, err)
		}
	}
	ev := callEvent(call("", "lookup", nil), call("fc-1", "transfer_money", amount(10)))
	ev.CustomMetadata = map[string]any{stampKey: "x"}
	checkIntegrityError(t, beforeRun(t, p, newSession(ev)), "event 'ev-1' has a malformed integrity stamp")
}

func TestBeforeRun_AllowUnstampedWarnsAndContinues(t *testing.T) {
	h := &recordingHandler{}
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: true, Logger: slog.New(h)})
	unstamped := callEvent(call("fc-1", "transfer_money", amount(10)))
	unstamped.ID = "ev-0"
	if err := beforeRun(t, p, newSession(unstamped, stampedEvent(map[string]any{"fc-1": stampA10}))); err != nil {
		t.Fatalf("BeforeRunCallback() error = %v, want nil", err)
	}
	checkUnstampedWarning(t, h, "transfer_money")
}

// Not parallel: it replaces the process-wide default logger.
func TestBeforeRun_NilLoggerWritesNothingToDefaultLogger(t *testing.T) {
	h := &recordingHandler{}
	old := slog.Default()
	slog.SetDefault(slog.New(h))
	t.Cleanup(func() { slog.SetDefault(old) })

	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: true})
	if err := beforeRun(t, p, newSession(callEvent(call("fc-1", "transfer_money", amount(10))))); err != nil {
		t.Fatalf("BeforeRunCallback() error = %v, want nil", err)
	}
	if got := len(h.snapshot()); got != 0 {
		t.Errorf("default logger got %d records, want 0", got)
	}
}

func TestBeforeRun_OldKeyStampFailsUnderNewKeyAlone(t *testing.T) {
	sess := newSession(stampedEvent(map[string]any{"fc-1": stampOldA10}))
	newOnly := newPlugin(t, toolcallintegrity.Config{SecretKey: []byte("new")})
	checkIntegrityError(t, beforeRun(t, newOnly, sess), tamperedText)
	rotating := newPlugin(t, toolcallintegrity.Config{SecretKey: []byte("new"), VerificationKeys: [][]byte{[]byte("old")}})
	if err := beforeRun(t, rotating, sess); err != nil {
		t.Errorf("BeforeRunCallback() with the old key as a verification key: error = %v, want nil", err)
	}
}

func TestBeforeRun_RestoredMarkerTruthiness(t *testing.T) {
	tests := []struct {
		name     string
		marker   any
		restored bool
	}{
		{"true", true, true},
		{"float64 1", float64(1), true},
		{"non-empty string", "yes", true},
		{"non-empty map", map[string]any{"a": 1}, true},
		{"non-empty list", []any{1}, true},
		{"false", false, false},
		{"float64 0", float64(0), false},
		{"empty string", "", false},
		{"nil", nil, false},
		{"empty map", map[string]any{}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
			ev := stampedEvent(map[string]any{"fc-1": "v1:00"})
			ev.CustomMetadata[restoredKey] = tc.marker
			err := beforeRun(t, p, newSession(ev))
			if tc.restored {
				if err != nil {
					t.Errorf("BeforeRunCallback() error = %v, want nil for a restored event", err)
				}
				return
			}
			checkIntegrityError(t, err, tamperedText)
		})
	}
}

func TestPrepareRestoredEvent_Metadata(t *testing.T) {
	if got := toolcallintegrity.PrepareRestoredEvent(nil); got != nil {
		t.Errorf("PrepareRestoredEvent(nil) = %v, want nil", got)
	}
	tests := []struct {
		name     string
		metadata map[string]any
		want     map[string]any
	}{
		{name: "nil metadata", metadata: nil, want: map[string]any{restoredKey: true}},
		{
			name: "mixed keys",
			metadata: map[string]any{
				"user":                 "v",
				stampKey:               map[string]any{"fc-1": stampA10},
				"__adk_internal_x":     1,
				"adk:before_run_reply": true,
				"adk.workflow.n":       "w",
				restoredKey:            false,
			},
			want: map[string]any{
				"user":                 "v",
				"adk:before_run_reply": true,
				"adk.workflow.n":       "w",
				restoredKey:            true,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ev := callEvent(call("fc-1", "transfer_money", amount(10)))
			ev.CustomMetadata = tc.metadata
			got := toolcallintegrity.PrepareRestoredEvent(ev)
			if diff := cmp.Diff(tc.want, got.CustomMetadata); diff != "" {
				t.Errorf("CustomMetadata mismatch (-want +got):\n%s", diff)
			}
			if got.Author != "agent" || got.InvocationID != "inv" {
				t.Errorf("Author, InvocationID = %q, %q, want %q, %q", got.Author, got.InvocationID, "agent", "inv")
			}
			if diff := cmp.Diff(callEvent(call("fc-1", "transfer_money", amount(10))).Content, got.Content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPrepareRestoredEvent_DoesNotAliasOrModifyInput(t *testing.T) {
	build := func() *session.Event {
		return stampedEvent(map[string]any{"fc-1": stampA10})
	}
	in := build()
	in.Content.Parts[0].FunctionCall.Args = map[string]any{"amount": 10, "l": []any{1.0, 2.0}}
	want := build()
	want.Content.Parts[0].FunctionCall.Args = map[string]any{"amount": 10, "l": []any{1.0, 2.0}}

	out := toolcallintegrity.PrepareRestoredEvent(in)
	if diff := cmp.Diff(want, in); diff != "" {
		t.Fatalf("PrepareRestoredEvent changed its input (-want +got):\n%s", diff)
	}
	args := out.Content.Parts[0].FunctionCall.Args
	args["amount"] = 99
	args["l"].([]any)[0] = 99.0
	out.CustomMetadata["new"] = 1
	if diff := cmp.Diff(want, in); diff != "" {
		t.Errorf("writing the copy changed the input (-want +got):\n%s", diff)
	}
}

func TestOnEvent_UnencodableArgs(t *testing.T) {
	const want = "toolcallintegrity: cannot encode the args of transfer_money call 'fc-1': value of type chan int is not JSON"
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	unencodable := map[string]any{"amount": 10, "c": make(chan int)}

	ev := callEvent(call("fc-1", "transfer_money", unencodable))
	_, err := p.OnEventCallback()(invocation(t, newSession()), ev)
	if err == nil || err.Error() != want {
		t.Errorf("OnEventCallback() error = %v, want %q", err, want)
	}
	if errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("errors.Is(OnEventCallback() error, ErrIntegrity) = true, want false")
	}
	if ev.CustomMetadata != nil {
		t.Errorf("CustomMetadata = %v, want nil", ev.CustomMetadata)
	}

	sess := newSession(stampedEvent(map[string]any{"fc-1": stampA10}))
	err = beforeTool(t, p, toolContext(t, sess, "fc-1"), "transfer_money", unencodable)
	if err == nil || err.Error() != want {
		t.Errorf("BeforeToolCallback() error = %v, want %q", err, want)
	}
	if errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("errors.Is(BeforeToolCallback() error, ErrIntegrity) = true, want false")
	}
}

// noIDContext is an agent.Context written outside ADK whose call has no ID.
type noIDContext struct{ agent.Context }

func (noIDContext) FunctionCallID() string { return "" }

func TestBeforeTool_NoCallIDIsRefusedWhateverTheOption(t *testing.T) {
	for _, allow := range []bool{false, true} {
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: allow})
		sess := newSession(stampedEvent(map[string]any{"fc-1": stampA10}))
		ctx := noIDContext{toolContext(t, sess, "fc-1")}
		checkIntegrityError(t, beforeTool(t, p, ctx, "transfer_money", amount(10)),
			"transfer_money call has no ID, so its integrity cannot be checked")
	}
}

func TestBeforeTool_NoStoredCopy(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	checkIntegrityError(t, beforeTool(t, p, toolContext(t, newSession(), "fc-1"), "transfer_money", amount(10)),
		"transfer_money call 'fc-1' has no integrity stamp")

	h := &recordingHandler{}
	allowing := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: true, Logger: slog.New(h)})
	if err := beforeTool(t, allowing, toolContext(t, newSession(), "fc-1"), "transfer_money", amount(10)); err != nil {
		t.Fatalf("BeforeToolCallback() with unstamped calls allowed: error = %v, want nil", err)
	}
	checkUnstampedWarning(t, h, "transfer_money")
}

func TestBeforeTool_RestoredHistory(t *testing.T) {
	const restoredText = "transfer_money call 'fc-1' comes from restored history and cannot be executed"
	restored := func() *session.Event {
		return toolcallintegrity.PrepareRestoredEvent(stampedEvent(map[string]any{"fc-1": stampA10}))
	}
	for _, allow := range []bool{false, true} {
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: allow})
		checkIntegrityError(t, beforeTool(t, p, toolContext(t, newSession(restored()), "fc-1"), "transfer_money", amount(10)), restoredText)
	}

	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	verified := newSession(restored(), stampedEvent(map[string]any{"fc-1": stampA10}))
	if err := beforeTool(t, p, toolContext(t, verified, "fc-1"), "transfer_money", amount(10)); err != nil {
		t.Errorf("BeforeToolCallback() with a restored copy beside a verified one: error = %v, want nil", err)
	}
	unstamped := newSession(restored(), callEvent(call("fc-1", "transfer_money", amount(10))))
	checkIntegrityError(t, beforeTool(t, p, toolContext(t, unstamped, "fc-1"), "transfer_money", amount(10)),
		"transfer_money call 'fc-1' has no integrity stamp")
}

func TestBeforeTool_NameAndArgsMustMatchATrustedCopy(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	single := newSession(stampedEvent(map[string]any{"fc-1": stampA10}))
	checkIntegrityError(t, beforeTool(t, p, toolContext(t, single, "fc-1"), "other_tool", amount(10)),
		"other_tool call 'fc-1' arguments do not match its stored call")
	checkIntegrityError(t, beforeTool(t, p, toolContext(t, single, "fc-1"), "transfer_money", map[string]any{"amount": 10, "x": 1}),
		"transfer_money call 'fc-1' arguments do not match its stored call")

	second := callEvent(call("fc-1", "transfer_money", amount(20)))
	second.ID = "ev-2"
	second.CustomMetadata = map[string]any{stampKey: map[string]any{"fc-1": stampA20}}
	reused := newSession(stampedEvent(map[string]any{"fc-1": stampA10}), second)
	if err := beforeTool(t, p, toolContext(t, reused, "fc-1"), "transfer_money", amount(20)); err != nil {
		t.Errorf("BeforeToolCallback() with the args of the second copy: error = %v, want nil", err)
	}
	err := beforeTool(t, p, toolContext(t, reused, "fc-1"), "transfer_money", amount(30))
	checkIntegrityError(t, err, "transfer_money call 'fc-1' arguments do not match its stored call")
	if err != nil && strings.Contains(err.Error(), "30") {
		t.Errorf("error %q contains the args", err.Error())
	}
}

func TestBeforeTool_RejectsWrongOrMalformedStamp(t *testing.T) {
	// secondCopy returns a second stored copy "ev-2" of call fc-1 with args
	// {"amount": 20} and stamps as its stamp map.
	secondCopy := func(stamps any) *session.Event {
		ev := callEvent(call("fc-1", "transfer_money", amount(20)))
		ev.ID = "ev-2"
		ev.CustomMetadata = map[string]any{stampKey: stamps}
		return ev
	}
	tests := []struct {
		name string
		sess *fakeSession
		args map[string]any
		want string
	}{
		{
			name: "stamp of other args",
			sess: newSession(stampedEvent(map[string]any{"fc-1": stampA20})),
			args: amount(10),
			want: tamperedText,
		},
		{
			name: "stamp map is a string",
			sess: newSession(stampedEvent("x")),
			args: amount(10),
			want: "event 'ev-1' has a malformed integrity stamp",
		},
		{
			// The first copy verifies and the call runs with the args of the
			// second, whose stamp is that of the first copy's args.
			name: "reused ID, second copy with a wrong stamp",
			sess: newSession(stampedEvent(map[string]any{"fc-1": stampA10}), secondCopy(map[string]any{"fc-1": stampA10})),
			args: amount(20),
			want: tamperedText,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
			checkIntegrityError(t, beforeTool(t, p, toolContext(t, tc.sess, "fc-1"), "transfer_money", tc.args), tc.want)
		})
	}
}

func TestBeforeTool_LeavesArgsUnchanged(t *testing.T) {
	stored := func() map[string]any {
		return map[string]any{"amount": 10, "l": []any{1, map[string]any{"m": 2}}}
	}
	asDoubles := func(amount float64) map[string]any {
		return map[string]any{"amount": amount, "l": []any{1.0, map[string]any{"m": 2.0}}}
	}
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	ev := callEvent(call("fc-1", "transfer_money", stored()))
	sess := newSession(ev)
	stamp(t, p, sess, ev)

	args := asDoubles(10)
	if err := beforeTool(t, p, toolContext(t, sess, "fc-1"), "transfer_money", args); err != nil {
		t.Fatalf("BeforeToolCallback() error = %v, want nil", err)
	}
	if diff := cmp.Diff(asDoubles(10), args); diff != "" {
		t.Errorf("args changed (-want +got):\n%s", diff)
	}
	mismatched := asDoubles(11)
	if err := beforeTool(t, p, toolContext(t, sess, "fc-1"), "transfer_money", mismatched); err == nil {
		t.Fatalf("BeforeToolCallback() with other args: error = nil, want one")
	}
	if diff := cmp.Diff(asDoubles(11), mismatched); diff != "" {
		t.Errorf("args changed after a mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(stored(), ev.Content.Parts[0].FunctionCall.Args); diff != "" {
		t.Errorf("stored call's args changed (-want +got):\n%s", diff)
	}
}

// foreignContext is an agent.Context written outside ADK: its Value answers
// no key, and its Session reports sess.
type foreignContext struct {
	agent.Context
	sess session.Session
}

func (foreignContext) Value(any) any              { return nil }
func (c foreignContext) Session() session.Session { return c.sess }

// sessionOverride is an agent.Context written outside ADK that wraps an ADK
// context and overrides only Session, so its Value still answers ADK's keys
// with the wrapped context's session.
type sessionOverride struct {
	agent.Context
	sess session.Session
}

func (c sessionOverride) Session() session.Session { return c.sess }

func TestBeforeTool_SessionSources(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	sess := newSession(stampedEvent(map[string]any{"fc-1": stampA10}))
	if err := beforeTool(t, p, toolContext(t, sess, "fc-1"), "transfer_money", amount(10)); err != nil {
		t.Errorf("BeforeToolCallback() through an ADK tool context: error = %v, want nil", err)
	}
	sessionless := toolContext(t, nil, "fc-1")
	if err := beforeTool(t, p, foreignContext{Context: sessionless, sess: sess}, "transfer_money", amount(10)); err != nil {
		t.Errorf("BeforeToolCallback() through a context's Session: error = %v, want nil", err)
	}
	overEmpty := sessionOverride{Context: toolContext(t, newSession(), "fc-1"), sess: sess}
	if err := beforeTool(t, p, overEmpty, "transfer_money", amount(10)); err != nil {
		t.Errorf("BeforeToolCallback() through a wrapper's Session over an ADK tool context of another session: error = %v, want nil", err)
	}
	checkIntegrityError(t, beforeTool(t, p, foreignContext{Context: sessionless}, "transfer_money", amount(10)),
		"transfer_money call 'fc-1' cannot be checked: the tool context has no session")
}

type transferArgs struct {
	Amount int `json:"amount"`
}

// countingTransfer returns a transfer_money tool and the list of args it ran
// with.
func countingTransfer(t *testing.T) (tool.Tool, *[]transferArgs) {
	t.Helper()
	var mu sync.Mutex
	var calls []transferArgs
	tl, err := functiontool.New(functiontool.Config{Name: "transfer_money", Description: "Transfers money."},
		func(_ agent.Context, args transferArgs) (map[string]any, error) {
			mu.Lock()
			defer mu.Unlock()
			calls = append(calls, args)
			return map[string]any{"ok": true}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return tl, &calls
}

func modelCalls(calls ...*genai.FunctionCall) *genai.Content {
	content := &genai.Content{Role: "model"}
	for _, fc := range calls {
		content.Parts = append(content.Parts, &genai.Part{FunctionCall: fc})
	}
	return content
}

func newLLMAgent(t *testing.T, m *testutil.MockModel, tl tool.Tool) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: m, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

func newRunner(t *testing.T, a agent.Agent, svc session.Service, p *plugin.Plugin) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: svc,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return r
}

func createSession(t *testing.T, svc session.Service) {
	t.Helper()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "session"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
}

func loadSession(t *testing.T, svc session.Service) session.Session {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return resp.Session
}

// run runs r once and returns its events and errors.
func run(t *testing.T, r *runner.Runner, text string) ([]*session.Event, []error) {
	t.Helper()
	var events []*session.Event
	var errs []error
	for ev, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText(text, "user"), agent.RunConfig{}) {
		if err != nil {
			errs = append(errs, err)
			continue
		}
		events = append(events, ev)
	}
	return events, errs
}

func finalText(events []*session.Event) string {
	for _, ev := range slices.Backward(events) {
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
			return ev.Content.Parts[0].Text
		}
	}
	return ""
}

func TestRun_RepeatedCallIDEndsRunAndSessionStaysUsable(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(fmt.Sprintf("AllowUnstampedCalls=%v", allow), func(t *testing.T) {
			m := &testutil.MockModel{Responses: []*genai.Content{
				modelCalls(
					call("fc-1", "transfer_money", amount(1)),
					call("fc-2", "transfer_money", amount(2)),
					call("fc-1", "transfer_money", amount(3)),
				),
				genai.NewContentFromText("ok", "model"),
			}}
			tl, calls := countingTransfer(t)
			svc := session.InMemoryService()
			createSession(t, svc)
			r := newRunner(t, newLLMAgent(t, m, tl), svc,
				newPlugin(t, toolcallintegrity.Config{SecretKey: testKey, AllowUnstampedCalls: allow}))

			_, errs := run(t, r, "pay")
			if len(errs) != 1 || !errors.Is(errs[0], toolcallintegrity.ErrIntegrity) {
				t.Fatalf("first run errors = %v, want one matching ErrIntegrity", errs)
			}
			if got := len(*calls); got != 0 {
				t.Errorf("tool ran %d times after the refused response, want 0", got)
			}
			if got := loadSession(t, svc).Events().Len(); got != 1 {
				t.Errorf("stored events after the first run = %d, want 1 (the user event)", got)
			}

			events, errs := run(t, r, "again")
			if len(errs) != 0 {
				t.Fatalf("second run errors = %v, want none", errs)
			}
			if got := finalText(events); got != "ok" {
				t.Errorf("second run final text = %q, want %q", got, "ok")
			}
			if got := len(*calls); got != 0 {
				t.Errorf("tool ran %d times, want 0", got)
			}
		})
	}
}

func newDatabaseService(t *testing.T) session.Service {
	t.Helper()
	svc, err := database.New(database.Config{Dialector: sqlite.Open(filepath.Join(t.TempDir(), "sessions.db"))})
	if err != nil {
		t.Fatalf("database.New() error = %v", err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		t.Fatalf("AutoMigrate() error = %v", err)
	}
	return svc
}

// emitting returns a custom agent named "agent" that yields events.
func emitting(t *testing.T, events ...*session.Event) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: "agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, ev := range events {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	return a
}

func TestRun_StampsSurviveDatabaseRoundTrip(t *testing.T) {
	t.Run("custom agent", func(t *testing.T) {
		svc := newDatabaseService(t)
		createSession(t, svc)
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
		ev := &session.Event{InvocationID: "inv", Author: "agent"}
		ev.Content = modelCalls(call("fc-1", "transfer_money", map[string]any{
			"amount": 10, "l": []any{1, map[string]any{"m": 2}}, "memo": nil, "ok": true, "raw": []byte("hi\xff"),
		}))
		if _, errs := run(t, newRunner(t, emitting(t, ev), svc, p), "go"); len(errs) != 0 {
			t.Fatalf("first run errors = %v, want none", errs)
		}

		sess := loadSession(t, svc)
		var stamps any
		for stored := range sess.Events().All() {
			if len(stored.FunctionCalls()) > 0 {
				stamps = stampsOf(stored)
			}
		}
		// adk-python stamps the same args
		// v1:fb562cdea87f726b926813e4ebb3d5bb43e0c9d125ff6558ab8eca9fdd7c9375,
		// because pydantic writes bytes as URL-safe base64 (ledger GS-3).
		want := map[string]any{"fc-1": "v1:fc4fea9cfd687c8a7a97a36c093a26b492d8d899f6c1cd54c3f699700e1e8ad1"}
		if diff := cmp.Diff(want, stamps); diff != "" {
			t.Errorf("stored stamps mismatch (-want +got):\n%s", diff)
		}
		if _, errs := run(t, newRunner(t, emitting(t), svc, p), "again"); len(errs) != 0 {
			t.Errorf("run after reload errors = %v, want none", errs)
		}
	})
	t.Run("nil map and slice", func(t *testing.T) {
		svc := newDatabaseService(t)
		createSession(t, svc)
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
		ev := &session.Event{InvocationID: "inv", Author: "agent"}
		ev.Content = modelCalls(call("fc-1", "transfer_money", map[string]any{
			"amount": 10, "m": map[string]any(nil), "l": []any(nil),
		}))
		if _, errs := run(t, newRunner(t, emitting(t, ev), svc, p), "go"); len(errs) != 0 {
			t.Fatalf("first run errors = %v, want none", errs)
		}
		if _, errs := run(t, newRunner(t, emitting(t), svc, p), "again"); len(errs) != 0 {
			t.Errorf("run after reload errors = %v, want none", errs)
		}
	})
	t.Run("LLM agent", func(t *testing.T) {
		svc := newDatabaseService(t)
		createSession(t, svc)
		p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
		m := &testutil.MockModel{Responses: []*genai.Content{
			modelCalls(call("fc-1", "transfer_money", amount(10))),
			genai.NewContentFromText("done", "model"),
			genai.NewContentFromText("ok", "model"),
		}}
		tl, calls := countingTransfer(t)
		a := newLLMAgent(t, m, tl)
		events, errs := run(t, newRunner(t, a, svc, p), "pay")
		if len(errs) != 0 {
			t.Fatalf("first run errors = %v, want none", errs)
		}
		for _, ev := range events {
			for _, fr := range ev.FunctionResponses() {
				if _, refused := fr.Response["error"]; refused {
					t.Errorf("function response = %v, want the tool's result", fr.Response)
				}
			}
		}
		if diff := cmp.Diff([]transferArgs{{Amount: 10}}, *calls); diff != "" {
			t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
		}
		if _, errs := run(t, newRunner(t, a, svc, p), "again"); len(errs) != 0 {
			t.Errorf("run after reload errors = %v, want none", errs)
		}
	})
}

func TestConcurrentUse(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{SecretKey: testKey})
	var wg sync.WaitGroup
	for g := range 32 {
		wg.Go(func() {
			for i := range 50 {
				value, want := 10, stampA10
				if (g+i)%2 == 1 {
					value, want = 20, stampA20
				}
				ev := callEvent(call("fc-1", "transfer_money", amount(value)))
				sess := newSession(ev)
				ic := invocation(t, sess)
				if _, err := p.OnEventCallback()(ic, ev); err != nil {
					t.Errorf("OnEventCallback() error = %v", err)
					return
				}
				if got := stampsOf(ev); !cmp.Equal(map[string]any{"fc-1": want}, got) {
					t.Errorf("stamps = %v, want fc-1: %s", got, want)
					return
				}
				ctx := agent.NewToolContext(ic, "fc-1", nil, nil)
				if _, err := p.BeforeToolCallback()(ctx, namedTool("transfer_money"), amount(value)); err != nil {
					t.Errorf("BeforeToolCallback() error = %v", err)
					return
				}
			}
		})
	}
	wg.Wait()
}
