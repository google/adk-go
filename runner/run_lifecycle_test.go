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

package runner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/workflow"
)

// lifecycleLog records, in one ordered list, the run-level plugin callbacks
// of a run and what the caller's loop body saw: "on_agent_error:<agent>",
// "on_run_error:<message>", "after_run" (or "after_run:<plugin>" for a named
// after-run plugin), "event:<text>" for a yielded event that has text,
// "err:<message>" for a yielded error, and "range_done" once the caller's
// range statement completed.
type lifecycleLog struct {
	mu      sync.Mutex
	entries []string
	runErrs []error
}

func (l *lifecycleLog) add(entry string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries = append(l.entries, entry)
}

func (l *lifecycleLog) got() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string{}, l.entries...)
}

// matching returns the entries that equal one of names or start with one of
// the names that end in ":".
func (l *lifecycleLog) matching(names ...string) []string {
	out := []string{}
	for _, e := range l.got() {
		for _, n := range names {
			if e == n || (strings.HasSuffix(n, ":") && strings.HasPrefix(e, n)) {
				out = append(out, e)
				break
			}
		}
	}
	return out
}

func (l *lifecycleLog) count(name string) int {
	return len(l.matching(name))
}

func (l *lifecycleLog) runErrors() []error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]error{}, l.runErrs...)
}

func (l *lifecycleLog) onRunError(_ agent.InvocationContext, err error) {
	l.mu.Lock()
	l.runErrs = append(l.runErrs, err)
	l.mu.Unlock()
	l.add("on_run_error:" + err.Error())
}

// recorder returns a plugin that logs its on-agent-error, on-run-error and
// after-run calls.
func (l *lifecycleLog) recorder(t *testing.T) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(plugin.Config{
		Name: "recorder",
		OnAgentErrorCallback: func(_ agent.Context, a agent.Agent, _ error) {
			l.add("on_agent_error:" + a.Name())
		},
		OnRunErrorCallback: l.onRunError,
		AfterRunCallback:   func(agent.InvocationContext) { l.add("after_run") },
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	return p
}

// failingAfterRun returns a plugin whose FailableAfterRunCallback logs
// "after_run:failing" and returns err.
func (l *lifecycleLog) failingAfterRun(t *testing.T, err error) *plugin.Plugin {
	t.Helper()
	return mustPlugin(t, plugin.Config{
		Name: "failing",
		FailableAfterRunCallback: func(agent.InvocationContext) error {
			l.add("after_run:failing")
			return err
		},
	})
}

// laterRecorder returns a plugin that logs "after_run:later" and its
// on-run-error calls.
func (l *lifecycleLog) laterRecorder(t *testing.T) *plugin.Plugin {
	t.Helper()
	return mustPlugin(t, plugin.Config{
		Name:               "later",
		AfterRunCallback:   func(agent.InvocationContext) { l.add("after_run:later") },
		OnRunErrorCallback: l.onRunError,
	})
}

// consume ranges seq, logs every item, and stops ranging at the first item
// for which stop returns true. It returns the errors the caller received.
func (l *lifecycleLog) consume(seq iter.Seq2[*session.Event, error], stop func(*session.Event, error) bool) []error {
	var errs []error
	for ev, err := range seq {
		if err != nil {
			errs = append(errs, err)
			l.add("err:" + err.Error())
		} else if text := eventText(ev); text != "" {
			l.add("event:" + text)
		}
		if stop(ev, err) {
			break
		}
	}
	l.add("range_done")
	return errs
}

func stopAtError(_ *session.Event, err error) bool { return err != nil }

func rangeOn(*session.Event, error) bool { return false }

func stopAtEvent(ev *session.Event, err error) bool { return err == nil && eventText(ev) != "" }

func eventText(ev *session.Event) string {
	if ev == nil || ev.Content == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range ev.Content.Parts {
		if p != nil {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func mustPlugin(t *testing.T, cfg plugin.Config) *plugin.Plugin {
	t.Helper()
	p, err := plugin.New(cfg)
	if err != nil {
		t.Fatalf("plugin.New(%q) error = %v", cfg.Name, err)
	}
	return p
}

// newLifecycleRunner returns a runner for cfg under app "app" that creates
// its sessions and runs plugins, with an in-memory session service unless
// cfg names one.
func newLifecycleRunner(t *testing.T, cfg Config, plugins ...*plugin.Plugin) (*Runner, session.Service) {
	t.Helper()
	cfg.AppName = "app"
	cfg.AutoCreateSession = true
	if cfg.SessionService == nil {
		cfg.SessionService = session.InMemoryService()
	}
	cfg.PluginConfig = PluginConfig{Plugins: plugins}
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r, cfg.SessionService
}

func userText(text string) *genai.Content { return genai.NewContentFromText(text, genai.RoleUser) }

func textEvent(ctx agent.InvocationContext, author, text string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Content = genai.NewContentFromText(text, genai.RoleModel)
	return ev
}

// step is one item a scripted agent yields: an event with text, an error, or
// a nil event with no error.
type step struct {
	text     string
	err      error
	nilEvent bool
}

// scriptedAgent returns a custom agent that yields steps in order and stops
// when the caller stops.
func scriptedAgent(t *testing.T, name string, steps ...step) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, s := range steps {
					var ok bool
					switch {
					case s.err != nil:
						ok = yield(nil, s.err)
					case s.nilEvent:
						ok = yield(nil, nil)
					default:
						ok = yield(textEvent(ctx, name, s.text), nil)
					}
					if !ok {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", name, err)
	}
	return a
}

// textNode returns a node root that emits one event per text and completes.
func textNode(texts ...string) workflow.Node {
	return workflow.NewEmittingFunctionNode("fn", func(ctx agent.Context, _ any, emit func(*session.Event) error) (string, error) {
		for _, text := range texts {
			if err := emit(textEvent(ctx, "fn", text)); err != nil {
				return "", err
			}
		}
		return "", nil
	}, workflow.NodeConfig{})
}

func failingNode(err error) workflow.Node {
	return workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) {
		return "", err
	}, workflow.NodeConfig{})
}

// failingModel fails every request with err and counts the requests.
type failingModel struct {
	err   error
	calls atomic.Int32
}

func (m *failingModel) Name() string { return "failing" }

func (m *failingModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, m.err)
	}
}

func newLLMRoot(t *testing.T, m model.LLM) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "root", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

// appendFailingSessionService fails AppendEvent with err for the events fail
// selects.
type appendFailingSessionService struct {
	session.Service
	fail func(*session.Event) bool
	err  error
}

func (s *appendFailingSessionService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if s.fail(ev) {
		return s.err
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}

func storedTexts(t *testing.T, svc session.Service, sessionID string) []string {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: sessionID})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	out := []string{}
	for ev := range resp.Session.Events().All() {
		if text := eventText(ev); text != "" {
			out = append(out, text)
		}
	}
	return out
}

func storedSession(t *testing.T, svc session.Service, sessionID string) session.Session {
	t.Helper()
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u", SessionID: sessionID})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return resp.Session
}

// When the caller stops at a run error, on-run-error is called once, with the
// value the caller received, after the caller's loop body returned and before
// its range statement completes. A node root has no agent layer, and an
// LlmAgent root, which run hands to runNode, has one lifecycle.
func TestRunner_OnRunError_CalledAfterCallerStopsAtError(t *testing.T) {
	errBoom := errors.New("boom")
	tests := []struct {
		name     string
		cfg      func(t *testing.T) Config
		want     []string
		wantSame bool // the notified value is errBoom itself
	}{
		{
			name:     "custom agent root",
			cfg:      func(t *testing.T) Config { return Config{Agent: scriptedAgent(t, "custom", step{err: errBoom})} },
			want:     []string{"on_agent_error:custom", "err:boom", "on_run_error:boom", "range_done"},
			wantSame: true,
		},
		{
			name: "llm agent root",
			cfg:  func(t *testing.T) Config { return Config{Agent: newLLMRoot(t, &failingModel{err: errBoom})} },
			want: []string{"on_agent_error:root", "err:boom", "on_run_error:boom", "range_done"},
		},
		{
			name:     "function node root",
			cfg:      func(t *testing.T) Config { return Config{Node: failingNode(errBoom)} },
			want:     []string{"err:boom", "on_run_error:boom", "range_done"},
			wantSame: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			r, _ := newLifecycleRunner(t, tt.cfg(t), l.recorder(t))

			yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)

			if diff := cmp.Diff(tt.want, l.got()); diff != "" {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}
			notified := l.runErrors()
			if len(yielded) != 1 || len(notified) != 1 {
				t.Fatalf("caller received %v and on-run-error %v, want one each", yielded, notified)
			}
			if notified[0] != yielded[0] {
				t.Errorf("on-run-error received %v, not the value the caller received", notified[0])
			}
			if !errors.Is(notified[0], errBoom) {
				t.Errorf("on-run-error received %v, want an error wrapping errBoom", notified[0])
			}
			if tt.wantSame && notified[0] != errBoom {
				t.Errorf("on-run-error received %v, want errBoom unwrapped", notified[0])
			}
		})
	}
}

// Setup failures, an early close and the caller's cancellation are not run
// errors: on-run-error is never called for them. The table also fixes when
// after-run runs on those paths.
func TestRunner_NotRunErrors_NeverNotifyOnRunError(t *testing.T) {
	mixed := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "hi"},
		{FunctionResponse: &genai.FunctionResponse{ID: "call_1", Name: "f"}},
	}}
	tests := []struct {
		name         string
		run          func(t *testing.T, plugins ...*plugin.Plugin)
		wantAfterRun int
	}{
		{
			name: "completed run",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, _ := newLifecycleRunner(t, Config{Agent: scriptedAgent(t, "custom", step{text: "ok"})}, plugins...)
				(&lifecycleLog{}).consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)
			},
			wantAfterRun: 1,
		},
		{
			name: "session missing",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, err := New(Config{
					AppName:        "app",
					Agent:          scriptedAgent(t, "custom", step{text: "ok"}),
					SessionService: session.InMemoryService(),
					PluginConfig:   PluginConfig{Plugins: plugins},
				})
				if err != nil {
					t.Fatalf("New() error = %v", err)
				}
				if errs := (&lifecycleLog{}).consume(r.Run(t.Context(), "u", "missing", userText("hi"), agent.RunConfig{}), stopAtError); len(errs) != 1 {
					t.Fatalf("errors = %v, want the session lookup error", errs)
				}
			},
		},
		{
			name: "node root refuses a message with text and a function response",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, _ := newLifecycleRunner(t, Config{Node: textNode("ok")}, plugins...)
				if errs := (&lifecycleLog{}).consume(r.Run(t.Context(), "u", "s", mixed, agent.RunConfig{}), stopAtError); len(errs) != 1 {
					t.Fatalf("errors = %v, want the message check error", errs)
				}
			},
		},
		{
			name: "classic root refuses a message with no parts",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, _ := newLifecycleRunner(t, Config{Agent: scriptedAgent(t, "custom", step{text: "ok"})}, plugins...)
				if errs := (&lifecycleLog{}).consume(r.Run(t.Context(), "u", "s", &genai.Content{Role: genai.RoleUser}, agent.RunConfig{}), stopAtError); len(errs) != 1 {
					t.Fatalf("errors = %v, want the message check error", errs)
				}
			},
		},
		{
			name: "early close on a custom agent root",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, _ := newLifecycleRunner(t, Config{Agent: scriptedAgent(t, "custom", step{text: "one"}, step{text: "two"})}, plugins...)
				(&lifecycleLog{}).consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtEvent)
			},
			wantAfterRun: 1,
		},
		{
			name: "early close on a node root",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				r, _ := newLifecycleRunner(t, Config{Node: textNode("one", "two")}, plugins...)
				(&lifecycleLog{}).consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtEvent)
			},
			wantAfterRun: 1,
		},
		{
			name: "caller cancellation",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				a := mustAgent(t, agent.Config{Name: "custom", Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						cancel()
						yield(nil, ic.Err())
					}
				}})
				r, _ := newLifecycleRunner(t, Config{Agent: a}, plugins...)
				(&lifecycleLog{}).consume(r.Run(ctx, "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)
			},
			wantAfterRun: 1,
		},
		{
			name: "caller cancellation wrapped",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				a := mustAgent(t, agent.Config{Name: "custom", Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						cancel()
						yield(nil, fmt.Errorf("model call: %w", ic.Err()))
					}
				}})
				r, _ := newLifecycleRunner(t, Config{Agent: a}, plugins...)
				(&lifecycleLog{}).consume(r.Run(ctx, "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)
			},
			wantAfterRun: 1,
		},
		{
			name: "caller deadline expired",
			run: func(t *testing.T, plugins ...*plugin.Plugin) {
				ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(-time.Second))
				defer cancel()
				a := scriptedAgent(t, "custom", step{err: fmt.Errorf("model call: %w", context.DeadlineExceeded)})
				r, _ := newLifecycleRunner(t, Config{Agent: a}, plugins...)
				(&lifecycleLog{}).consume(r.Run(ctx, "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)
			},
			wantAfterRun: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			sessionLen := -1
			afterRunSession := mustPlugin(t, plugin.Config{
				Name:             "session_len",
				AfterRunCallback: func(ic agent.InvocationContext) { sessionLen = ic.Session().Events().Len() },
			})

			tt.run(t, l.recorder(t), afterRunSession)

			if got := l.count("on_run_error:"); got != 0 {
				t.Errorf("on-run-error calls = %d (%v), want 0", got, l.runErrors())
			}
			if got := l.count("after_run"); got != tt.wantAfterRun {
				t.Errorf("after-run calls = %d, want %d", got, tt.wantAfterRun)
			}
			// After-run runs after the last event: it sees the user message and
			// the agent's event.
			if tt.name == "completed run" && sessionLen != 2 {
				t.Errorf("after-run saw %d session events, want 2", sessionLen)
			}
		})
	}
}

func mustAgent(t *testing.T, cfg agent.Config) agent.Agent {
	t.Helper()
	a, err := agent.New(cfg)
	if err != nil {
		t.Fatalf("agent.New(%q) error = %v", cfg.Name, err)
	}
	return a
}

// A context error that the agent produced while the caller's context is
// still live is a run error.
func TestRunner_ContextErrorWhileCallerContextAlive_IsARunError(t *testing.T) {
	tests := []struct {
		name string
		run  func(ctx context.Context) error
		want []string
	}{
		{
			name: "canceled",
			run: func(ctx context.Context) error {
				own, cancel := context.WithCancel(ctx)
				cancel()
				return own.Err()
			},
			want: []string{"on_run_error:context canceled"},
		},
		{
			name: "deadline exceeded",
			run: func(ctx context.Context) error {
				own, cancel := context.WithTimeout(ctx, time.Nanosecond)
				defer cancel()
				<-own.Done()
				return own.Err()
			},
			want: []string{"on_run_error:context deadline exceeded"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			a := mustAgent(t, agent.Config{Name: "custom", Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, tt.run(ic))
				}
			}})
			r, _ := newLifecycleRunner(t, Config{Agent: a}, l.recorder(t))

			yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)

			if diff := cmp.Diff(tt.want, l.matching("on_run_error:")); diff != "" {
				t.Errorf("on-run-error calls mismatch (-want +got):\n%s", diff)
			}
			if got := l.count("after_run"); got != 0 {
				t.Errorf("after-run calls = %d, want 0", got)
			}
			if notified := l.runErrors(); len(yielded) != 1 || len(notified) != 1 || notified[0] != yielded[0] {
				t.Errorf("caller received %v and on-run-error %v, want the same single error", yielded, notified)
			}
		})
	}
}

// A failing after-run on a completed run reaches a caller that still ranges,
// unwrapped, ends the run, stops the after-run callbacks of later plugins and
// is reported to every plugin once.
func TestRunner_AfterRunFailure_YieldedUnwrappedAndNotifiedOnce(t *testing.T) {
	errAfter := errors.New("after-run failed")
	roots := []struct {
		name string
		cfg  func(t *testing.T) Config
	}{
		{name: "custom agent root", cfg: func(t *testing.T) Config { return Config{Agent: scriptedAgent(t, "custom", step{text: "ok"})} }},
		{name: "node root", cfg: func(t *testing.T) Config { return Config{Node: textNode("ok")} }},
	}
	callers := []struct {
		name string
		stop func(*session.Event, error) bool
	}{
		{name: "caller ranges on", stop: rangeOn},
		{name: "caller stops at the error", stop: stopAtError},
	}
	for _, root := range roots {
		for _, caller := range callers {
			t.Run(root.name+"/"+caller.name, func(t *testing.T) {
				l := &lifecycleLog{}
				first := mustPlugin(t, plugin.Config{
					Name:             "first",
					AfterRunCallback: func(agent.InvocationContext) { l.add("after_run:first") },
				})
				r, _ := newLifecycleRunner(t, root.cfg(t), first, l.failingAfterRun(t, errAfter), l.laterRecorder(t))

				yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), caller.stop)

				want := []string{"event:ok", "after_run:first", "after_run:failing", "err:after-run failed", "on_run_error:after-run failed", "range_done"}
				if diff := cmp.Diff(want, l.got()); diff != "" {
					t.Errorf("log mismatch (-want +got):\n%s", diff)
				}
				if len(yielded) != 1 || yielded[0] != errAfter {
					t.Errorf("caller received %v, want [errAfter] unwrapped", yielded)
				}
				if notified := l.runErrors(); len(notified) != 1 || notified[0] != errAfter {
					t.Errorf("on-run-error received %v, want [errAfter]", notified)
				}
			})
		}
	}
}

// After the caller stopped at an event, a failing after-run goes to
// on-run-error and to the invocation span; nothing is yielded.
func TestRunner_AfterRunFailureAfterEarlyClose_NotifiedNotYielded(t *testing.T) {
	errAfter := errors.New("after-run failed")
	tests := []struct {
		name string
		cfg  func(t *testing.T) Config
		opts []RunOption
	}{
		{name: "custom agent root", cfg: func(t *testing.T) Config {
			return Config{Agent: scriptedAgent(t, "custom", step{text: "one"}, step{text: "two"})}
		}},
		{name: "node root stopped on the user message", cfg: func(t *testing.T) Config { return Config{Node: textNode("one")} }, opts: []RunOption{WithYieldUserMessage()}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			exp := tracetest.NewInMemoryExporter()
			telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
			l := &lifecycleLog{}
			r, _ := newLifecycleRunner(t, tt.cfg(t), l.failingAfterRun(t, errAfter), l.laterRecorder(t))

			items := 0
			for range r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}, tt.opts...) { //nolint:errcheck // the test counts the items before its break; their errors are not under test
				items++
				break
			}

			if items != 1 {
				t.Errorf("loop body ran %d times, want 1", items)
			}
			if notified := l.runErrors(); len(notified) != 1 || notified[0] != errAfter {
				t.Errorf("on-run-error received %v, want [errAfter]", notified)
			}
			if got := l.count("after_run:later"); got != 0 {
				t.Errorf("later plugin's after-run calls = %d, want 0", got)
			}
			var recorded []string
			for _, sp := range exp.GetSpans() {
				for _, ev := range sp.Events {
					if ev.Name != "exception" {
						continue
					}
					for _, a := range ev.Attributes {
						if a.Key == "exception.message" {
							recorded = append(recorded, a.Value.AsString())
						}
					}
				}
			}
			if diff := cmp.Diff([]string{"after-run failed"}, recorded); diff != "" {
				t.Errorf("span exception events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// An error the caller ranges past does not end the run. When the run then
// ends without a further run error, after-run runs and on-run-error is not
// called; when it ends because of a later run error, on-run-error is called
// once, for that error.
func TestRunner_CallerRangesPastError_RunGoesOn(t *testing.T) {
	first, second := errors.New("first"), errors.New("second")
	tests := []struct {
		name           string
		stop           func(*session.Event, error) bool
		wantRunErrors  []string
		wantAfterRuns  int
		wantCallerSeen []string
	}{
		{
			name:           "caller ranges on",
			stop:           rangeOn,
			wantRunErrors:  []string{},
			wantAfterRuns:  1,
			wantCallerSeen: []string{"err:first", "event:x", "err:second"},
		},
		{
			name:           "caller stops at the second error",
			stop:           func(_ *session.Event, err error) bool { return err == second },
			wantRunErrors:  []string{"on_run_error:second"},
			wantCallerSeen: []string{"err:first", "event:x", "err:second"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			a := scriptedAgent(t, "custom", step{err: first}, step{text: "x"}, step{err: second})
			r, _ := newLifecycleRunner(t, Config{Agent: a}, l.recorder(t))

			l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), tt.stop)

			if diff := cmp.Diff(tt.wantCallerSeen, l.matching("err:", "event:")); diff != "" {
				t.Errorf("caller items mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantRunErrors, l.matching("on_run_error:")); diff != "" {
				t.Errorf("on-run-error calls mismatch (-want +got):\n%s", diff)
			}
			if got := l.count("after_run"); got != tt.wantAfterRuns {
				t.Errorf("after-run calls = %d, want %d", got, tt.wantAfterRuns)
			}
		})
	}
}

// An on-event plugin error ends the run, so a caller that ranges on still gets
// no further event, on-run-error is called and after-run is skipped.
func TestRunner_OnEventError_EndsRunNotifiesAndSkipsAfterRun(t *testing.T) {
	tests := []struct {
		name string
		cfg  func(t *testing.T) Config
	}{
		{name: "custom agent root", cfg: func(t *testing.T) Config {
			return Config{Agent: scriptedAgent(t, "custom", step{text: "one"}, step{text: "two"}, step{text: "three"})}
		}},
		{name: "node root", cfg: func(t *testing.T) Config { return Config{Node: textNode("one", "two", "three")} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			rejecter := mustPlugin(t, plugin.Config{
				Name: "rejecter",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if eventText(ev) == "two" {
						return nil, errors.New("on event failed")
					}
					return nil, nil
				},
			})
			r, svc := newLifecycleRunner(t, tt.cfg(t), rejecter, l.recorder(t))

			l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), rangeOn)

			want := []string{"event:one", "err:on event failed", "on_run_error:on event failed", "range_done"}
			if diff := cmp.Diff(want, l.got()); diff != "" {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]string{"hi", "one"}, storedTexts(t, svc, "s")); diff != "" {
				t.Errorf("stored events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A before-run failure ends the run and notifies on-run-error; after-run
// still runs once on that path, as TestRunner_BeforeRunShortCircuit requires.
func TestRunner_BeforeRunError_NotifiesOnRunErrorAndKeepsAfterRun(t *testing.T) {
	errBefore := errors.New("before failed")
	var bodyRuns atomic.Int32
	tests := []struct {
		name string
		cfg  func(t *testing.T) Config
	}{
		{name: "custom agent root", cfg: func(t *testing.T) Config {
			return Config{Agent: mustAgent(t, agent.Config{Name: "custom", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				bodyRuns.Add(1)
				return func(func(*session.Event, error) bool) {}
			}})}
		}},
		{name: "llm agent root", cfg: func(t *testing.T) Config {
			return Config{Agent: newLLMRoot(t, &countingFailModel{calls: &bodyRuns})}
		}},
		{name: "node root", cfg: func(t *testing.T) Config {
			return Config{Node: workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) {
				bodyRuns.Add(1)
				return "", nil
			}, workflow.NodeConfig{})}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bodyRuns.Store(0)
			l := &lifecycleLog{}
			before := mustPlugin(t, plugin.Config{
				Name:              "before",
				BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) { return nil, errBefore },
			})
			r, _ := newLifecycleRunner(t, tt.cfg(t), before, l.recorder(t))

			yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), rangeOn)

			want := []string{"err:before failed", "on_run_error:before failed", "after_run", "range_done"}
			if diff := cmp.Diff(want, l.got()); diff != "" {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}
			if len(yielded) != 1 || yielded[0] != errBefore {
				t.Errorf("caller received %v, want [errBefore]", yielded)
			}
			if got := bodyRuns.Load(); got != 0 {
				t.Errorf("root body ran %d times, want 0", got)
			}
		})
	}
}

// countingFailModel counts its requests and fails each of them.
type countingFailModel struct{ calls *atomic.Int32 }

func (m *countingFailModel) Name() string { return "counting" }

func (m *countingFailModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls.Add(1)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, errors.New("model reached"))
	}
}

// The on-user-message failure on both root kinds and the user-event append
// failure on the node path are run errors that end the run, notified once
// whether the caller ranges on or stops at the error; the same append failure
// on the classic path is a setup failure.
func TestRunner_StageErrors_AreRunErrors(t *testing.T) {
	userAppendFails := func() session.Service {
		return &appendFailingSessionService{
			Service: session.InMemoryService(),
			fail:    func(ev *session.Event) bool { return ev.Author == "user" },
			err:     errors.New("append failed"),
		}
	}
	rejectUserMessage := errors.New("user message rejected")
	tests := []struct {
		name          string
		cfg           func(t *testing.T) Config
		plugin        *plugin.Plugin
		msg           *genai.Content
		opts          []RunOption
		stopAtError   bool
		wantErr       string
		wantRunErrors []string
	}{
		{
			name: "on-user-message failure on a classic root",
			cfg:  func(t *testing.T) Config { return Config{Agent: scriptedAgent(t, "custom", step{text: "ok"})} },
			plugin: mustPlugin(t, plugin.Config{
				Name:                  "reject",
				OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) { return nil, rejectUserMessage },
			}),
			msg:           userText("hi"),
			wantErr:       "error running on run user message callback : user message rejected",
			wantRunErrors: []string{"on_run_error:error running on run user message callback : user message rejected"},
		},
		{
			name:          "user-event append failure on a node root",
			cfg:           func(t *testing.T) Config { return Config{Node: textNode("ok"), SessionService: userAppendFails()} },
			msg:           userText("hi"),
			wantErr:       "failed to append event to sessionService: append failed",
			wantRunErrors: []string{"on_run_error:failed to append event to sessionService: append failed"},
		},
		{
			name:          "state-delta append failure on a node root",
			cfg:           func(t *testing.T) Config { return Config{Node: textNode("ok"), SessionService: userAppendFails()} },
			opts:          []RunOption{WithStateDelta(map[string]any{"k": 1})},
			wantErr:       "failed to append event to sessionService: append failed",
			wantRunErrors: []string{"on_run_error:failed to append event to sessionService: append failed"},
		},
		{
			name:          "user-event append failure on a node root, caller stops at the error",
			cfg:           func(t *testing.T) Config { return Config{Node: textNode("ok"), SessionService: userAppendFails()} },
			msg:           userText("hi"),
			stopAtError:   true,
			wantErr:       "failed to append event to sessionService: append failed",
			wantRunErrors: []string{"on_run_error:failed to append event to sessionService: append failed"},
		},
		{
			name: "on-user-message failure on an llm agent root, caller stops at the error",
			cfg: func(t *testing.T) Config {
				return Config{Agent: newLLMRoot(t, &failingModel{err: errors.New("unreached")})}
			},
			plugin: mustPlugin(t, plugin.Config{
				Name:                  "reject",
				OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) { return nil, rejectUserMessage },
			}),
			msg:           userText("hi"),
			stopAtError:   true,
			wantErr:       "error running on run user message callback : user message rejected",
			wantRunErrors: []string{"on_run_error:error running on run user message callback : user message rejected"},
		},
		{
			name: "user-event append failure on a classic root",
			cfg: func(t *testing.T) Config {
				return Config{Agent: scriptedAgent(t, "custom", step{text: "ok"}), SessionService: userAppendFails()}
			},
			msg:           userText("hi"),
			wantErr:       "failed to append event to sessionService: append failed",
			wantRunErrors: []string{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			plugins := []*plugin.Plugin{l.recorder(t)}
			if tt.plugin != nil {
				plugins = append([]*plugin.Plugin{tt.plugin}, plugins...)
			}
			r, _ := newLifecycleRunner(t, tt.cfg(t), plugins...)
			stop := rangeOn
			if tt.stopAtError {
				stop = stopAtError
			}

			yielded := l.consume(r.Run(t.Context(), "u", "s", tt.msg, agent.RunConfig{}, tt.opts...), stop)

			if len(yielded) != 1 || yielded[0].Error() != tt.wantErr {
				t.Fatalf("caller received %v, want one error %q", yielded, tt.wantErr)
			}
			if tt.plugin != nil && !errors.Is(yielded[0], rejectUserMessage) {
				t.Errorf("errors.Is(%v, cause) = false, want true", yielded[0])
			}
			if diff := cmp.Diff(tt.wantRunErrors, l.matching("on_run_error:")); diff != "" {
				t.Errorf("on-run-error calls mismatch (-want +got):\n%s", diff)
			}
			if got := l.count("after_run"); got != 0 {
				t.Errorf("after-run calls = %d, want 0", got)
			}
		})
	}
}

// A post-invocation compaction failure is a run error on both root paths, and
// it happens after after-run.
func TestRunner_PostInvocationCompactionFailure_NotifiesOnRunErrorAfterAfterRun(t *testing.T) {
	failing := &compaction.Config{CompactionInterval: 1, Summarizer: failingSummarizer{}}
	tests := []struct {
		name      string
		root      func(t *testing.T) agent.Agent
		stop      func(*session.Event, error) bool
		earlyStop bool
	}{
		{name: "llm agent root, normal end", root: func(t *testing.T) agent.Agent { return newLLMRoot(t, &scriptedModel{replyFmt: "answer %d"}) }, stop: rangeOn},
		{name: "custom agent root, normal end", root: func(t *testing.T) agent.Agent { return scriptedAgent(t, "custom", step{text: "ok"}) }, stop: rangeOn},
		{name: "llm agent root, caller stops at the first event", root: func(t *testing.T) agent.Agent {
			return newLLMRoot(t, &scriptedModel{replyFmt: "answer %d"})
		}, stop: stopAtEvent, earlyStop: true},
		{name: "custom agent root, caller stops at the first event", root: func(t *testing.T) agent.Agent {
			return scriptedAgent(t, "custom", step{text: "ok"})
		}, stop: stopAtEvent, earlyStop: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			r, _ := newLifecycleRunner(t, Config{Agent: tt.root(t), Compaction: failing}, l.recorder(t))

			yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), tt.stop)

			got := l.matching("after_run", "on_run_error:")
			if len(got) != 2 || got[0] != "after_run" || !strings.HasPrefix(got[1], "on_run_error:") || !strings.Contains(got[1], "summarizer exploded") {
				t.Fatalf("lifecycle = %v, want after_run then on_run_error with the compaction error", got)
			}
			if notified := l.runErrors(); !errors.Is(notified[0], compaction.ErrCompaction) {
				t.Errorf("on-run-error received %v, want an ErrCompaction", notified[0])
			}
			items := l.matching("err:", "event:")
			if tt.earlyStop {
				if len(yielded) != 0 || len(items) != 1 {
					t.Errorf("caller items = %v, want the first event only", items)
				}
				return
			}
			if len(yielded) != 1 || items[len(items)-1] != "err:"+yielded[0].Error() || len(items) < 2 {
				t.Errorf("caller items = %v, want the turn's events, then the compaction error last", items)
			}
		})
	}
}

// A panic on the goroutine that ranges the run runs after-run and notifies
// nothing. A panic inside a node body is not one: the workflow scheduler turns
// it into an error, which is a run error.
func TestRunner_PanicOnRunnerGoroutine_RunsAfterRunWithoutNotification(t *testing.T) {
	t.Run("custom agent root panics", func(t *testing.T) {
		l := &lifecycleLog{}
		a := mustAgent(t, agent.Config{Name: "custom", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) { panic("kaboom") }
		}})
		r, _ := newLifecycleRunner(t, Config{Agent: a}, l.recorder(t))

		got := recoverRun(r)

		if got != "kaboom" {
			t.Errorf("recovered %v, want kaboom", got)
		}
		if diff := cmp.Diff([]string{"after_run"}, l.got()); diff != "" {
			t.Errorf("log mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("on-event callback panics on a node root", func(t *testing.T) {
		l := &lifecycleLog{}
		panicker := mustPlugin(t, plugin.Config{
			Name: "panicker",
			OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				if eventText(ev) == "ok" {
					panic("plugin kaboom")
				}
				return nil, nil
			},
		})
		r, _ := newLifecycleRunner(t, Config{Node: textNode("ok")}, panicker, l.recorder(t))

		got := recoverRun(r)

		if got != "plugin kaboom" {
			t.Errorf("recovered %v, want plugin kaboom", got)
		}
		if diff := cmp.Diff([]string{"after_run"}, l.got()); diff != "" {
			t.Errorf("log mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("node body panics", func(t *testing.T) {
		l := &lifecycleLog{}
		n := workflow.NewFunctionNode("fn", func(agent.Context, any) (string, error) { panic("kaboom") }, workflow.NodeConfig{})
		r, _ := newLifecycleRunner(t, Config{Node: n}, l.recorder(t))

		yielded := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), stopAtError)

		if len(yielded) != 1 || !strings.Contains(yielded[0].Error(), `node "`) || !strings.Contains(yielded[0].Error(), "panicked: kaboom") {
			t.Fatalf("caller received %v, want one node panic error", yielded)
		}
		if got := l.count("on_run_error:"); got != 1 {
			t.Errorf("on-run-error calls = %d, want 1", got)
		}
		if got := l.count("after_run"); got != 0 {
			t.Errorf("after-run calls = %d, want 0", got)
		}
	})
}

// recoverRun drains one run of r and returns the value of a panic that
// escaped it.
func recoverRun(r *Runner) (recovered any) {
	defer func() { recovered = recover() }()
	for range r.Run(context.Background(), "u", "s", userText("hi"), agent.RunConfig{}) { //nolint:errcheck // only the panic the run raises is under test
	}
	return nil
}

// newLiveLifecycleRunner returns a runner whose root is a live agent named
// "live" that runLive starts.
func newLiveLifecycleRunner(t *testing.T, svc session.Service, runLive func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error), plugins ...*plugin.Plugin) *Runner {
	t.Helper()
	r, _ := newLifecycleRunner(t, Config{
		Agent:          &mockLiveAgent{Agent: mustAgent(t, agent.Config{Name: "live"}), runLiveFn: runLive},
		SessionService: svc,
	}, plugins...)
	return r
}

// liveStream returns a RunLive function whose stream yields steps in order.
func liveStream(steps ...step) func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
		return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
			for _, s := range steps {
				var ok bool
				switch {
				case s.err != nil:
					ok = yield(nil, s.err)
				case s.nilEvent:
					ok = yield(nil, nil)
				default:
					ok = yield(textEvent(ctx, "live", s.text), nil)
				}
				if !ok {
					return
				}
			}
		}, nil
	}
}

// On the live path a run error that ends the run notifies once and skips
// after-run; an error the caller ranges past does not; a failing after-run is
// reported.
func TestRunner_RunLive_RunErrorNotifiesOnceAndSkipsAfterRun(t *testing.T) {
	errStart := errors.New("start failed")
	errBefore := errors.New("before failed")
	errStream := errors.New("stream failed")
	errAfter := errors.New("after-run failed")
	beforeRun := func(content *genai.Content, err error) *plugin.Plugin {
		return mustPlugin(t, plugin.Config{
			Name:              "before",
			BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) { return content, err },
		})
	}
	tests := []struct {
		name          string
		svc           session.Service
		runLive       func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
		plugins       func(l *lifecycleLog) []*plugin.Plugin
		stop          func(*session.Event, error) bool
		wantRunLive   error // the error RunLive returns
		wantLog       []string
		wantAfterRuns int
	}{
		{
			name: "live agent fails to start",
			runLive: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
				return nil, nil, errStart
			},
			plugins:     func(l *lifecycleLog) []*plugin.Plugin { return []*plugin.Plugin{l.recorder(t)} },
			wantRunLive: errStart,
			wantLog:     []string{"on_run_error:start failed"},
		},
		{
			name:    "before-run fails",
			runLive: liveStream(step{text: "e1"}),
			plugins: func(l *lifecycleLog) []*plugin.Plugin {
				return []*plugin.Plugin{beforeRun(nil, errBefore), l.recorder(t)}
			},
			wantRunLive: errBefore,
			wantLog:     []string{"on_run_error:before failed"},
		},
		{
			name:          "caller ranges past a stream error",
			runLive:       liveStream(step{err: errStream}, step{text: "e1"}),
			plugins:       func(l *lifecycleLog) []*plugin.Plugin { return []*plugin.Plugin{l.recorder(t)} },
			stop:          rangeOn,
			wantLog:       []string{"err:stream failed", "event:e1", "after_run", "range_done"},
			wantAfterRuns: 1,
		},
		{
			name:    "caller stops at a stream error",
			runLive: liveStream(step{err: errStream}, step{text: "e1"}),
			plugins: func(l *lifecycleLog) []*plugin.Plugin { return []*plugin.Plugin{l.recorder(t)} },
			stop:    stopAtError,
			wantLog: []string{"err:stream failed", "on_run_error:stream failed", "range_done"},
		},
		{
			name:    "after-run fails on a clean stream",
			runLive: liveStream(step{text: "e1"}),
			plugins: func(l *lifecycleLog) []*plugin.Plugin {
				return []*plugin.Plugin{l.failingAfterRun(t, errAfter), l.laterRecorder(t)}
			},
			stop:    rangeOn,
			wantLog: []string{"event:e1", "after_run:failing", "err:after-run failed", "on_run_error:after-run failed", "range_done"},
		},
		{
			name:    "after-run fails after a before-run reply",
			runLive: liveStream(step{text: "e1"}),
			plugins: func(l *lifecycleLog) []*plugin.Plugin {
				return []*plugin.Plugin{beforeRun(genai.NewContentFromText("reply", genai.RoleModel), nil), l.failingAfterRun(t, errAfter), l.laterRecorder(t)}
			},
			stop:    rangeOn,
			wantLog: []string{"event:reply", "after_run:failing", "err:after-run failed", "on_run_error:after-run failed", "range_done"},
		},
		{
			name: "caller ranges past an append failure",
			svc: &appendFailingSessionService{
				Service: session.InMemoryService(),
				fail:    func(ev *session.Event) bool { return eventText(ev) == "e1" },
				err:     errors.New("store down"),
			},
			runLive:       liveStream(step{text: "e1"}, step{text: "e2"}),
			plugins:       func(l *lifecycleLog) []*plugin.Plugin { return []*plugin.Plugin{l.recorder(t)} },
			stop:          rangeOn,
			wantLog:       []string{"err:failed to add event to session: store down", "event:e2", "after_run", "range_done"},
			wantAfterRuns: 1,
		},
		{
			name:    "live agent yields a nil event",
			runLive: liveStream(step{nilEvent: true}),
			plugins: func(l *lifecycleLog) []*plugin.Plugin { return []*plugin.Plugin{l.recorder(t)} },
			stop:    rangeOn,
			wantLog: []string{`err:adk: agent "live" yielded a nil event`, `on_run_error:adk: agent "live" yielded a nil event`, "range_done"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			r := newLiveLifecycleRunner(t, tt.svc, tt.runLive, tt.plugins(l)...)

			_, events, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
			if err != tt.wantRunLive {
				t.Fatalf("RunLive() error = %v, want %v", err, tt.wantRunLive)
			}
			if err == nil {
				l.consume(events, tt.stop)
			}

			if diff := cmp.Diff(tt.wantLog, l.got()); diff != "" {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}
			for _, notified := range l.runErrors() {
				if notified != tt.wantRunLive && notified != errStream && notified != errAfter && !strings.Contains(notified.Error(), "nil event") {
					t.Errorf("on-run-error received %v, not the error that ended the run", notified)
				}
			}
		})
	}
}

// An on-event error stops the live stream: no further event is yielded or
// stored.
func TestRunner_RunLive_OnEventErrorEndsStream(t *testing.T) {
	l := &lifecycleLog{}
	rejecter := mustPlugin(t, plugin.Config{
		Name: "rejecter",
		OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if eventText(ev) == "e2" {
				return nil, errors.New("on event failed")
			}
			return nil, nil
		},
	})
	svc := session.InMemoryService()
	r := newLiveLifecycleRunner(t, svc, liveStream(step{text: "e1"}, step{text: "e2"}, step{text: "e3"}), rejecter, l.recorder(t))

	_, events, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	l.consume(events, rangeOn)

	want := []string{"event:e1", "err:on event failed", "on_run_error:on event failed", "range_done"}
	if diff := cmp.Diff(want, l.got()); diff != "" {
		t.Errorf("log mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"e1"}, storedTexts(t, svc, "s")); diff != "" {
		t.Errorf("stored events mismatch (-want +got):\n%s", diff)
	}
}

// Concurrent runs of one runner each notify once, with their own error.
func TestRunner_ConcurrentRuns_EachInvocationNotifiedOnce(t *testing.T) {
	var mu sync.Mutex
	bySession := map[string][]string{}
	var afterRuns atomic.Int32
	p := mustPlugin(t, plugin.Config{
		Name: "recorder",
		OnRunErrorCallback: func(ctx agent.InvocationContext, err error) {
			mu.Lock()
			defer mu.Unlock()
			bySession[ctx.Session().ID()] = append(bySession[ctx.Session().ID()], err.Error())
		},
		AfterRunCallback: func(agent.InvocationContext) { afterRuns.Add(1) },
	})
	a := mustAgent(t, agent.Config{Name: "custom", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			yield(nil, fmt.Errorf("boom-%s", ctx.Session().ID()))
		}
	}})
	r, _ := newLifecycleRunner(t, Config{Agent: a}, p)

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for _, err := range r.Run(t.Context(), "u", fmt.Sprintf("s%d", i), userText("hi"), agent.RunConfig{}) {
				if err != nil {
					break
				}
			}
		})
	}
	wg.Wait()

	want := map[string][]string{}
	for i := range 8 {
		want[fmt.Sprintf("s%d", i)] = []string{fmt.Sprintf("boom-s%d", i)}
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff(want, bySession); diff != "" {
		t.Errorf("on-run-error calls by session mismatch (-want +got):\n%s", diff)
	}
	if got := afterRuns.Load(); got != 0 {
		t.Errorf("after-run calls = %d, want 0", got)
	}
}

// At each place where the runner itself stops on a run error, a caller that
// ranges on still gets on-run-error once and no after-run.
func TestRunner_RunnerStopsOnRunError_NotifiesRangingCaller(t *testing.T) {
	storeDown := func() session.Service {
		return &appendFailingSessionService{
			Service: session.InMemoryService(),
			fail:    func(ev *session.Event) bool { return ev.Author != "user" },
			err:     errors.New("store down"),
		}
	}
	tests := []struct {
		name    string
		cfg     func(t *testing.T) Config
		wantLog []string
		wantIs  error
	}{
		{
			name: "model call limit",
			cfg: func(t *testing.T) Config {
				return Config{Agent: scriptedAgent(t, "custom", step{err: fmt.Errorf("budget: %w", agent.ErrLLMCallsLimitExceeded)}, step{text: "after"})}
			},
			wantLog: []string{
				"on_agent_error:custom",
				"err:budget: " + agent.ErrLLMCallsLimitExceeded.Error(),
				"on_run_error:budget: " + agent.ErrLLMCallsLimitExceeded.Error(),
				"range_done",
			},
			wantIs: agent.ErrLLMCallsLimitExceeded,
		},
		{
			name:    "nil event",
			cfg:     func(t *testing.T) Config { return Config{Agent: scriptedAgent(t, "custom", step{nilEvent: true})} },
			wantLog: []string{`err:adk: agent "custom" yielded a nil event`, `on_run_error:adk: agent "custom" yielded a nil event`, "range_done"},
		},
		{
			name: "event append failure on a classic root",
			cfg: func(t *testing.T) Config {
				return Config{Agent: scriptedAgent(t, "custom", step{text: "ok"}, step{text: "more"}), SessionService: storeDown()}
			},
			wantLog: []string{"err:failed to add event to session: store down", "on_run_error:failed to add event to session: store down", "range_done"},
		},
		{
			name:    "event append failure on a node root",
			cfg:     func(t *testing.T) Config { return Config{Node: textNode("ok", "more"), SessionService: storeDown()} },
			wantLog: []string{"err:failed to add event to session: store down", "on_run_error:failed to add event to session: store down", "range_done"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &lifecycleLog{}
			r, _ := newLifecycleRunner(t, tt.cfg(t), l.recorder(t))

			l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}), rangeOn)

			if diff := cmp.Diff(tt.wantLog, l.got()); diff != "" {
				t.Errorf("log mismatch (-want +got):\n%s", diff)
			}
			if tt.wantIs != nil {
				if notified := l.runErrors(); len(notified) != 1 || !errors.Is(notified[0], tt.wantIs) {
					t.Errorf("on-run-error received %v, want an error wrapping %v", notified, tt.wantIs)
				}
			}
		})
	}
}

// At the normal end of a run, after-run runs before post-invocation
// compaction on both root paths, and a failing after-run leaves no
// compaction.
func TestRunner_AfterRunRunsBeforePostInvocationCompaction(t *testing.T) {
	errAfter := errors.New("after-run failed")
	tests := []struct {
		name           string
		root           func(t *testing.T) agent.Agent
		failAfterRun   bool
		wantSeen       int // compaction events after-run sees
		wantStored     int // compaction events stored after the run
		wantLastErrNil bool
	}{
		{name: "llm agent root", root: func(t *testing.T) agent.Agent { return newLLMRoot(t, &scriptedModel{replyFmt: "answer %d"}) }, wantStored: 1, wantLastErrNil: true},
		{name: "custom agent root", root: func(t *testing.T) agent.Agent { return scriptedAgent(t, "custom", step{text: "ok"}) }, wantStored: 1, wantLastErrNil: true},
		{name: "llm agent root with a failing after-run", root: func(t *testing.T) agent.Agent {
			return newLLMRoot(t, &scriptedModel{replyFmt: "answer %d"})
		}, failAfterRun: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seen := -1
			svc := session.InMemoryService()
			cfg := plugin.Config{Name: "after_run_probe"}
			// Compaction appends to a session it reloads, so the stored session,
			// not ic.Session(), shows whether it already ran.
			probe := func(agent.InvocationContext) { seen = len(compactionEventsIn(storedSession(t, svc, "s"))) }
			if tt.failAfterRun {
				cfg.FailableAfterRunCallback = func(ic agent.InvocationContext) error { probe(ic); return errAfter }
			} else {
				cfg.AfterRunCallback = probe
			}
			r, _ := newLifecycleRunner(t, Config{
				Agent:          tt.root(t),
				SessionService: svc,
				Compaction:     &compaction.Config{CompactionInterval: 1, Summarizer: &recordingSummarizer{summary: "summary"}},
			}, mustPlugin(t, cfg))

			var lastErr error
			for _, err := range r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}) {
				lastErr = err
			}

			if seen != tt.wantSeen {
				t.Errorf("after-run saw %d compaction events, want %d", seen, tt.wantSeen)
			}
			if got := len(compactionEventsIn(storedSession(t, svc, "s"))); got != tt.wantStored {
				t.Errorf("stored compaction events = %d, want %d", got, tt.wantStored)
			}
			if tt.wantLastErrNil && lastErr != nil {
				t.Errorf("last item error = %v, want nil", lastErr)
			}
			if !tt.wantLastErrNil && lastErr != errAfter {
				t.Errorf("last item error = %v, want errAfter", lastErr)
			}
		})
	}
}

// On a node root, the user-event append of a retried stored turn that carries
// a state delta is part of the run: its failure is a run error, notified once.
func TestRunner_StageErrors_RetriedTurnStateDeltaAppendFailureOnNodeRoot(t *testing.T) {
	var failUserAppends atomic.Bool
	svc := &appendFailingSessionService{
		Service: session.InMemoryService(),
		fail:    func(ev *session.Event) bool { return failUserAppends.Load() && ev.Author == "user" },
		err:     errors.New("append failed"),
	}
	l := &lifecycleLog{}
	r, _ := newLifecycleRunner(t, Config{Node: textNode("ok"), SessionService: svc}, l.recorder(t))
	if errs := l.consume(r.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{}, WithInvocationID("inv_retry")), rangeOn); len(errs) != 0 {
		t.Fatalf("first run errors = %v, want none", errs)
	}
	failUserAppends.Store(true)
	retry := &lifecycleLog{}
	r2, _ := newLifecycleRunner(t, Config{Node: textNode("ok"), SessionService: svc}, retry.recorder(t))

	yielded := retry.consume(r2.Run(t.Context(), "u", "s", userText("hi"), agent.RunConfig{},
		WithInvocationID("inv_retry"), WithStateDelta(map[string]any{"k": 1})), rangeOn)

	if len(yielded) != 1 || yielded[0].Error() != "failed to append event to sessionService: append failed" {
		t.Fatalf("retry received %v, want the user-event append error", yielded)
	}
	want := []string{"on_run_error:failed to append event to sessionService: append failed"}
	if diff := cmp.Diff(want, retry.matching("on_run_error:")); diff != "" {
		t.Errorf("on-run-error calls mismatch (-want +got):\n%s", diff)
	}
	if got := retry.count("after_run"); got != 0 {
		t.Errorf("after-run calls = %d, want 0", got)
	}
}

// On a node root, a turn whose run state cannot be reconstructed (an answer
// that fails its request's response schema) ends the run with that error,
// even for a caller that ranges past it: on-run-error once, no after-run.
func TestRunner_NodeRootRunStateReconstructionFailure_NotifiesRangingCaller(t *testing.T) {
	svc := session.InMemoryService()
	ask := workflow.NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
			InterruptID: "approval", Message: "approve", ResponseSchema: &jsonschema.Schema{Type: "string"},
		})
	}, workflow.NodeConfig{})
	paused := &lifecycleLog{}
	r, _ := newLifecycleRunner(t, Config{Node: ask, SessionService: svc}, paused.recorder(t))
	if errs := paused.consume(r.Run(t.Context(), "u", "s", userText("go"), agent.RunConfig{}), rangeOn); len(errs) != 0 {
		t.Fatalf("first run errors = %v, want none", errs)
	}
	l := &lifecycleLog{}
	r2, _ := newLifecycleRunner(t, Config{Node: ask, SessionService: svc}, l.recorder(t))
	answer := genai.NewContentFromFunctionResponse(workflow.WorkflowInputFunctionCallName, map[string]any{"result": 42}, genai.RoleUser)
	answer.Parts[0].FunctionResponse.ID = "approval"

	yielded := l.consume(r2.Run(t.Context(), "u", "s", answer, agent.RunConfig{}), rangeOn)

	if len(yielded) != 1 || !errors.Is(yielded[0], workflow.ErrInvalidResumeResponse) ||
		!strings.HasPrefix(yielded[0].Error(), "failed to reconstruct workflow run state: ") {
		t.Fatalf("second run errors = %v, want one run state reconstruction error", yielded)
	}
	if notified := l.runErrors(); len(notified) != 1 || notified[0] != yielded[0] {
		t.Errorf("on-run-error received %v, want once the yielded %v", notified, yielded[0])
	}
	if got := l.count("after_run"); got != 0 {
		t.Errorf("after-run calls = %d, want 0", got)
	}
}
