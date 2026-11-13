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
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/toolcallintegrity"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

var testKey = []byte("test-secret-key")

// fakeContext is the invocation and tool context the callbacks are called
// with directly.
type fakeContext struct {
	agent.StrictContextMock
	sess   session.Session
	callID string
	ended  bool
}

func (c *fakeContext) Session() session.Session { return c.sess }
func (c *fakeContext) InvocationID() string     { return "inv" }
func (c *fakeContext) FunctionCallID() string   { return c.callID }
func (c *fakeContext) EndInvocation()           { c.ended = true }

type fakeTool struct{ name string }

func (f fakeTool) Name() string        { return f.name }
func (f fakeTool) Description() string { return "" }
func (f fakeTool) IsLongRunning() bool { return false }

var _ tool.Tool = fakeTool{}

func newPlugin(t *testing.T, cfg toolcallintegrity.Config) *plugin.Plugin {
	t.Helper()
	if cfg.SecretKeys == nil {
		cfg.SecretKeys = [][]byte{testKey}
	}
	p, err := toolcallintegrity.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p
}

func newSession(t *testing.T) (session.Service, session.Session) {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatal(err)
	}
	return svc, resp.Session
}

func callEvent(calls ...*genai.FunctionCall) *session.Event {
	ev := session.NewEvent(context.TODO(), "inv")
	ev.Author = "agent"
	c := &genai.Content{Role: genai.RoleModel}
	for _, fc := range calls {
		c.Parts = append(c.Parts, &genai.Part{FunctionCall: fc})
	}
	ev.Content = c
	return ev
}

// appendStamped stamps ev with p's OnEventCallback, then appends it.
func appendStamped(t *testing.T, p *plugin.Plugin, svc session.Service, sess session.Session, ev *session.Event) {
	t.Helper()
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
	if _, err := p.OnEventCallback()(ctx, ev); err != nil {
		t.Fatalf("OnEventCallback() error = %v", err)
	}
	if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
		t.Fatal(err)
	}
}

func beforeRun(t *testing.T, p *plugin.Plugin, sess session.Session) error {
	t.Helper()
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
	_, err := p.BeforeRunCallback()(ctx)
	return err
}

func beforeTool(t *testing.T, p *plugin.Plugin, sess session.Session, name, callID string, args map[string]any) error {
	t.Helper()
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess, callID: callID}
	res, err := p.BeforeToolCallback()(ctx, fakeTool{name}, args)
	if res != nil {
		t.Errorf("BeforeToolCallback() result = %v, want nil", res)
	}
	return err
}

func wantIntegrityError(t *testing.T, err error, substr string) {
	t.Helper()
	if _, ok := errors.AsType[*toolcallintegrity.Error](err); !ok {
		t.Fatalf("error = %v, want a *toolcallintegrity.Error", err)
	}
	if !strings.Contains(err.Error(), substr) {
		t.Errorf("error = %q, want it to contain %q", err, substr)
	}
}

func TestNewRejectsBadKeys(t *testing.T) {
	for name, keys := range map[string][][]byte{
		"none":      nil,
		"empty":     {{}},
		"one empty": {[]byte("k"), {}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := toolcallintegrity.New(toolcallintegrity.Config{SecretKeys: keys}); err == nil || !strings.Contains(err.Error(), "non-empty bytes") {
				t.Errorf("New() error = %v, want one about non-empty bytes", err)
			}
		})
	}
}

func TestOnEventStampsCalls(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{})
	svc, sess := newSession(t)
	ev := callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: map[string]any{"amount": 10}})
	ev.CustomMetadata = map[string]any{"mine": "kept"}
	appendStamped(t, p, svc, sess, ev)

	stamps, ok := ev.CustomMetadata[toolcallintegrity.StampsMetadataKey].(map[string]any)
	if !ok || !strings.HasPrefix(stamps["fc-1"].(string), "v1:") {
		t.Errorf("stamps = %v, want a v1 stamp for fc-1", ev.CustomMetadata[toolcallintegrity.StampsMetadataKey])
	}
	if ev.CustomMetadata["mine"] != "kept" {
		t.Errorf("custom metadata lost the caller's key: %v", ev.CustomMetadata)
	}

	text := session.NewEvent(context.TODO(), "inv")
	text.Content = genai.NewContentFromText("hi", genai.RoleModel)
	appendStamped(t, p, svc, sess, text)
	if text.CustomMetadata != nil {
		t.Errorf("an event without calls got metadata %v", text.CustomMetadata)
	}
}

func TestOnEventRefusesRepeatedCallID(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{})
	_, sess := newSession(t)
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
	fc := func() *genai.FunctionCall { return &genai.FunctionCall{ID: "fc-1", Name: "t"} }

	_, err := p.OnEventCallback()(ctx, callEvent(fc(), fc()))
	wantIntegrityError(t, err, "repeats function call ID")
	if !strings.Contains(err.Error(), "'tool_call_integrity' during 'on_event_callback'") {
		t.Errorf("error = %q, want it to name the plugin and callback", err)
	}

	partial := callEvent(fc(), fc())
	partial.Partial = true
	if _, err := p.OnEventCallback()(ctx, partial); err != nil {
		t.Errorf("OnEventCallback(partial) error = %v, want nil", err)
	}
}

func TestBeforeRun(t *testing.T) {
	call := func() *genai.FunctionCall {
		return &genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: map[string]any{"amount": 10}}
	}
	tests := []struct {
		name    string
		cfg     toolcallintegrity.Config
		edit    func(ev *session.Event)
		stamp   bool
		wantErr string
	}{
		{name: "untouched", stamp: true},
		{name: "modified args", stamp: true, edit: func(ev *session.Event) {
			ev.Content.Parts[0].FunctionCall.Args["amount"] = 10000
		}, wantErr: "does not match"},
		{name: "renamed", stamp: true, edit: func(ev *session.Event) {
			ev.Content.Parts[0].FunctionCall.Name = "other"
		}, wantErr: "does not match"},
		{name: "moved branch", stamp: true, edit: func(ev *session.Event) { ev.Branch = "other" }, wantErr: "does not match"},
		{name: "re-keyed", stamp: true, edit: func(ev *session.Event) {
			ev.Content.Parts[0].FunctionCall.ID = "fc-2"
		}, wantErr: "no integrity stamp"},
		{name: "re-keyed allowed", cfg: toolcallintegrity.Config{AllowUnstampedCalls: true}, stamp: true, edit: func(ev *session.Event) {
			ev.Content.Parts[0].FunctionCall.ID = "fc-2"
		}, wantErr: "no matching call"},
		{name: "whole-number doubles", stamp: true, edit: func(ev *session.Event) {
			ev.Content.Parts[0].FunctionCall.Args["amount"] = 10.0
		}},
		{name: "malformed", stamp: true, edit: func(ev *session.Event) {
			ev.CustomMetadata[toolcallintegrity.StampsMetadataKey] = "bad"
		}, wantErr: "malformed"},
		{name: "never stamped", wantErr: "no integrity stamp"},
		{name: "never stamped allowed", cfg: toolcallintegrity.Config{AllowUnstampedCalls: true}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, tc.cfg)
			svc, sess := newSession(t)
			ev := callEvent(call())
			if tc.stamp {
				ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
				if _, err := p.OnEventCallback()(ctx, ev); err != nil {
					t.Fatal(err)
				}
			}
			if tc.edit != nil {
				tc.edit(ev)
			}
			if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
				t.Fatal(err)
			}
			err := beforeRun(t, p, sess)
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("BeforeRunCallback() error = %v, want nil", err)
				}
				return
			}
			wantIntegrityError(t, err, tc.wantErr)
		})
	}
}

func TestBeforeRunRejectsStampFromOtherSession(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{})
	svc, other := newSession(t)
	ev := callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer"})
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: other}
	if _, err := p.OnEventCallback()(ctx, ev); err != nil {
		t.Fatal(err)
	}
	// Same ids, but a different app.
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "other", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatal(err)
	}
	if err := svc.AppendEvent(t.Context(), resp.Session, ev); err != nil {
		t.Fatal(err)
	}
	wantIntegrityError(t, beforeRun(t, p, resp.Session), "does not match")
}

func TestKeyRotation(t *testing.T) {
	old := newPlugin(t, toolcallintegrity.Config{SecretKeys: [][]byte{[]byte("old")}})
	rotating := newPlugin(t, toolcallintegrity.Config{SecretKeys: [][]byte{[]byte("new"), []byte("old")}})
	newOnly := newPlugin(t, toolcallintegrity.Config{SecretKeys: [][]byte{[]byte("new")}})

	svc, sess := newSession(t)
	appendStamped(t, old, svc, sess, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "t"}))
	if err := beforeRun(t, rotating, sess); err != nil {
		t.Errorf("rotating BeforeRunCallback() error = %v, want nil", err)
	}
	wantIntegrityError(t, beforeRun(t, newOnly, sess), "does not match")

	svc2, sess2 := newSession(t)
	appendStamped(t, rotating, svc2, sess2, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "t"}))
	if err := beforeRun(t, newOnly, sess2); err != nil {
		t.Errorf("new key did not sign: BeforeRunCallback() error = %v", err)
	}
}

func TestBeforeTool(t *testing.T) {
	args := map[string]any{"amount": 10}
	t.Run("verified call runs", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{})
		svc, sess := newSession(t)
		appendStamped(t, p, svc, sess, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args}))
		if err := beforeTool(t, p, sess, "transfer", "fc-1", map[string]any{"amount": 10.0}); err != nil {
			t.Errorf("BeforeToolCallback() error = %v, want nil", err)
		}
	})
	t.Run("other args are refused", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{})
		svc, sess := newSession(t)
		appendStamped(t, p, svc, sess, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args}))
		wantIntegrityError(t, beforeTool(t, p, sess, "transfer", "fc-1", map[string]any{"amount": 5}), "arguments")
	})
	t.Run("no ID is refused", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{AllowUnstampedCalls: true})
		_, sess := newSession(t)
		wantIntegrityError(t, beforeTool(t, p, sess, "transfer", "", args), "no ID")
	})
	t.Run("unstamped call is refused by default", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{})
		svc, sess := newSession(t)
		if err := svc.AppendEvent(t.Context(), sess, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args})); err != nil {
			t.Fatal(err)
		}
		wantIntegrityError(t, beforeTool(t, p, sess, "transfer", "fc-1", args), "no integrity stamp")
	})
	t.Run("modified copy is refused", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{AllowUnstampedCalls: true})
		svc, sess := newSession(t)
		appendStamped(t, p, svc, sess, callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args}))
		tampered := callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: map[string]any{"amount": 10}})
		ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
		if _, err := p.OnEventCallback()(ctx, tampered); err != nil {
			t.Fatal(err)
		}
		tampered.Content.Parts[0].FunctionCall.Args["amount"] = 10000
		if err := svc.AppendEvent(t.Context(), sess, tampered); err != nil {
			t.Fatal(err)
		}
		wantIntegrityError(t, beforeTool(t, p, sess, "transfer", "fc-1", args), "does not match")
	})
	t.Run("restored call is refused", func(t *testing.T) {
		p := newPlugin(t, toolcallintegrity.Config{AllowUnstampedCalls: true})
		svc, sess := newSession(t)
		restored, err := toolcallintegrity.PrepareRestoredEvent(callEvent(&genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args}))
		if err != nil {
			t.Fatal(err)
		}
		if err := svc.AppendEvent(t.Context(), sess, restored); err != nil {
			t.Fatal(err)
		}
		if err := beforeRun(t, p, sess); err != nil {
			t.Errorf("BeforeRunCallback() error = %v, want restored history accepted", err)
		}
		wantIntegrityError(t, beforeTool(t, p, sess, "transfer", "fc-1", args), "restored history")
	})
}

func TestPrepareRestoredEvent(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{})
	_, sess := newSession(t)
	ev := callEvent(&genai.FunctionCall{ID: "fc-1", Name: "t"})
	ev.CustomMetadata = map[string]any{"mine": "kept"}
	ctx := &fakeContext{StrictContextMock: agent.NewStrictContextMock(t.Context()), sess: sess}
	if _, err := p.OnEventCallback()(ctx, ev); err != nil {
		t.Fatal(err)
	}
	cp, err := toolcallintegrity.PrepareRestoredEvent(ev)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cp.CustomMetadata[toolcallintegrity.StampsMetadataKey]; ok {
		t.Errorf("copy kept the stamps: %v", cp.CustomMetadata)
	}
	if cp.CustomMetadata[toolcallintegrity.RestoredEventKey] != true || cp.CustomMetadata["mine"] != "kept" {
		t.Errorf("copy metadata = %v, want the caller's key and the restored marker", cp.CustomMetadata)
	}
	if _, ok := ev.CustomMetadata[toolcallintegrity.StampsMetadataKey]; !ok {
		t.Errorf("PrepareRestoredEvent modified its argument")
	}
}

// transferAgent returns an agent whose model calls transfer with args, then
// answers "done" (and "ok" to a later run), and the number of times the tool
// ran.
func transferAgent(t *testing.T, args map[string]any) (agent.Agent, *int) {
	t.Helper()
	runs := new(int)
	transfer, err := functiontool.New(functiontool.Config{Name: "transfer", Description: "Transfers money."},
		func(_ agent.Context, in struct {
			Amount int `json:"amount"`
		},
		) (map[string]any, error) {
			*runs++
			return map[string]any{"status": "done"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	model := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "transfer", Args: args}}}},
		genai.NewContentFromText("done", genai.RoleModel),
		genai.NewContentFromText("ok", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "agent", Model: model, Tools: []tool.Tool{transfer}})
	if err != nil {
		t.Fatal(err)
	}
	return a, runs
}

func TestRunnerStampsAndRunsFreshCall(t *testing.T) {
	a, runs := transferAgent(t, map[string]any{"amount": 10})
	p := newPlugin(t, toolcallintegrity.Config{})
	r := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{p}})
	if _, err := testutil.CollectEvents(r.Run(t, "s", "pay")); err != nil {
		t.Fatalf("first run error = %v", err)
	}
	if *runs != 1 {
		t.Errorf("tool ran %d times, want 1", *runs)
	}
	// The second run checks the stored, stamped history.
	if _, err := testutil.CollectEvents(r.Run(t, "s", "again")); err != nil {
		t.Errorf("second run error = %v", err)
	}
}

func TestRunnerRefusedCallFailsRun(t *testing.T) {
	a, runs := transferAgent(t, map[string]any{"amount": 10})
	rewrite, err := plugin.New(plugin.Config{Name: "rewrite", BeforeToolCallback: func(_ agent.Context, _ tool.Tool, args map[string]any) (map[string]any, error) {
		args["amount"] = 5
		return nil, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	p := newPlugin(t, toolcallintegrity.Config{})
	// The rewrite runs first, so the integrity plugin sees the changed args.
	r := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{rewrite, p}})
	_, err = testutil.CollectEvents(r.Run(t, "s", "pay"))
	wantIntegrityError(t, err, "'tool_call_integrity' during 'before_tool_callback'")
	if *runs != 0 {
		t.Errorf("tool ran %d times, want 0", *runs)
	}
}
