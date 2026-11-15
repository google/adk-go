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
	"errors"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/toolcallintegrity"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

var (
	keyA = []byte("key-a")
	keyB = []byte("key-b")
)

type fakeCtx struct {
	agent.StrictContextMock
	sess         session.Session
	callID       string
	invocationID string
}

func (c *fakeCtx) Session() session.Session { return c.sess }
func (c *fakeCtx) FunctionCallID() string   { return c.callID }
func (c *fakeCtx) InvocationID() string {
	if c.invocationID == "" {
		return "inv-current"
	}
	return c.invocationID
}

type fakeTool struct{ name string }

func (t fakeTool) Name() string        { return t.name }
func (t fakeTool) Description() string { return "" }
func (t fakeTool) IsLongRunning() bool { return false }

var _ tool.Tool = fakeTool{}

func newPlugin(t *testing.T, cfg toolcallintegrity.Config) *plugin.Plugin {
	t.Helper()
	p, err := toolcallintegrity.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p
}

func callEvent(id, invocationID string, calls ...*genai.FunctionCall) *session.Event {
	ev := &session.Event{ID: id, InvocationID: invocationID, Author: "assistant", Branch: "root"}
	var parts []*genai.Part
	for _, fc := range calls {
		parts = append(parts, &genai.Part{FunctionCall: fc})
	}
	ev.Content = &genai.Content{Role: "model", Parts: parts}
	return ev
}

func weatherCall(id, city string) *genai.FunctionCall {
	return &genai.FunctionCall{ID: id, Name: "get_weather", Args: map[string]any{"city": city, "days": 3}}
}

// newSession returns an empty in-memory session and its service.
func newSession(t *testing.T) (session.Service, session.Session) {
	t.Helper()
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	return svc, resp.Session
}

// stampAndStore runs ev through OnEventCallback, applies mutate to the stamped
// event, appends it to the session and returns the session as stored.
func stampAndStore(t *testing.T, p *plugin.Plugin, svc session.Service, sess session.Session, ev *session.Event, mutate func(*session.Event)) session.Session {
	t.Helper()
	got, err := p.OnEventCallback()(&fakeCtx{sess: sess}, ev)
	if err != nil {
		t.Fatalf("OnEventCallback() error = %v", err)
	}
	if got != nil {
		t.Fatalf("OnEventCallback() returned a replacement event, want nil")
	}
	if mutate != nil {
		mutate(ev)
	}
	if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	resp, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "user", SessionID: sess.ID()})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	return resp.Session
}

func TestNew_RejectsMissingKeys(t *testing.T) {
	for name, keys := range map[string][][]byte{
		"nil":       nil,
		"empty key": {keyA, {}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := toolcallintegrity.New(toolcallintegrity.Config{Keys: keys}); err == nil {
				t.Errorf("New() error = nil, want an error")
			}
		})
	}
}

func TestNew_CopiesKeys(t *testing.T) {
	key := []byte("mutable")
	p := newPlugin(t, toolcallintegrity.Config{Keys: [][]byte{key}})
	svc, sess := newSession(t)
	stored := stampAndStore(t, p, svc, sess, callEvent("e1", "inv1", weatherCall("c1", "Paris")), nil)
	key[0] = 'X'

	if _, err := p.BeforeRunCallback()(&fakeCtx{sess: stored}); err != nil {
		t.Errorf("BeforeRunCallback() error = %v after the caller changed its key slice, want nil", err)
	}
}

func TestOnEvent(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{Keys: [][]byte{keyA}})
	_, sess := newSession(t)
	ctx := &fakeCtx{sess: sess}

	t.Run("stamps each call and keeps other metadata", func(t *testing.T) {
		ev := callEvent("e1", "inv1", weatherCall("c1", "Paris"), weatherCall("c2", "Rome"), &genai.FunctionCall{Name: "no_id"})
		original := map[string]any{"other": "kept"}
		ev.CustomMetadata = original

		if _, err := p.OnEventCallback()(ctx, ev); err != nil {
			t.Fatalf("OnEventCallback() error = %v", err)
		}
		stamps, ok := ev.CustomMetadata[toolcallintegrity.MetadataKey].(map[string]string)
		if !ok || len(stamps) != 2 || stamps["c1"] == "" || stamps["c2"] == "" || stamps["c1"] == stamps["c2"] {
			t.Errorf("stamps = %v, want two distinct stamps for c1 and c2", ev.CustomMetadata[toolcallintegrity.MetadataKey])
		}
		if ev.CustomMetadata["other"] != "kept" {
			t.Errorf("CustomMetadata lost an existing key: %v", ev.CustomMetadata)
		}
		if _, ok := original[toolcallintegrity.MetadataKey]; ok {
			t.Errorf("OnEventCallback() wrote into the caller's metadata map")
		}
	})

	t.Run("leaves an event without calls alone", func(t *testing.T) {
		ev := &session.Event{ID: "e0", InvocationID: "inv1"}
		ev.Content = genai.NewContentFromText("hi", genai.RoleModel)
		if _, err := p.OnEventCallback()(ctx, ev); err != nil {
			t.Fatalf("OnEventCallback() error = %v", err)
		}
		if ev.CustomMetadata != nil {
			t.Errorf("CustomMetadata = %v, want nil", ev.CustomMetadata)
		}
	})

	t.Run("rejects a repeated call ID", func(t *testing.T) {
		ev := callEvent("e1", "inv1", weatherCall("c1", "Paris"), weatherCall("c1", "Rome"))
		if _, err := p.OnEventCallback()(ctx, ev); !errors.Is(err, toolcallintegrity.ErrIntegrity) {
			t.Errorf("OnEventCallback() error = %v, want ErrIntegrity", err)
		}
	})

	t.Run("allows a repeated call ID on a partial event", func(t *testing.T) {
		ev := callEvent("e1", "inv1", weatherCall("c1", "Paris"), weatherCall("c1", "Rome"))
		ev.Partial = true
		if _, err := p.OnEventCallback()(ctx, ev); err != nil {
			t.Errorf("OnEventCallback() error = %v, want nil", err)
		}
	})
}

func TestBeforeRun(t *testing.T) {
	tests := []struct {
		name    string
		cfg     toolcallintegrity.Config
		signCfg *toolcallintegrity.Config
		mutate  func(*session.Event)
		wantErr bool
	}{
		{name: "untouched calls verify"},
		{
			name:    "changed argument",
			mutate:  func(ev *session.Event) { ev.Content.Parts[0].FunctionCall.Args["city"] = "Berlin" },
			wantErr: true,
		},
		{
			name:    "changed tool name",
			mutate:  func(ev *session.Event) { ev.Content.Parts[0].FunctionCall.Name = "delete_all" },
			wantErr: true,
		},
		{
			name:    "changed author",
			mutate:  func(ev *session.Event) { ev.Author = "other_agent" },
			wantErr: true,
		},
		{
			name:    "changed call ID",
			mutate:  func(ev *session.Event) { ev.Content.Parts[0].FunctionCall.ID = "c9" },
			wantErr: true,
		},
		{
			name: "removed call leaves an unmatched stamp",
			mutate: func(ev *session.Event) {
				ev.Content.Parts = ev.Content.Parts[:1]
			},
			wantErr: true,
		},
		{
			name: "added call without a stamp",
			mutate: func(ev *session.Event) {
				ev.Content.Parts = append(ev.Content.Parts, &genai.Part{FunctionCall: weatherCall("c3", "Oslo")})
			},
			wantErr: true,
		},
		{
			name: "added call without a stamp, unstamped calls allowed",
			cfg:  toolcallintegrity.Config{AllowUnstampedCalls: true},
			mutate: func(ev *session.Event) {
				ev.Content.Parts = append(ev.Content.Parts, &genai.Part{FunctionCall: weatherCall("c3", "Oslo")})
			},
		},
		{
			name: "wrong stamp, unstamped calls allowed",
			cfg:  toolcallintegrity.Config{AllowUnstampedCalls: true},
			mutate: func(ev *session.Event) {
				ev.CustomMetadata[toolcallintegrity.MetadataKey] = map[string]string{"c1": "v1:00", "c2": "v1:00"}
			},
			wantErr: true,
		},
		{
			name: "stamp without version prefix",
			mutate: func(ev *session.Event) {
				stamps := ev.CustomMetadata[toolcallintegrity.MetadataKey].(map[string]string)
				ev.CustomMetadata[toolcallintegrity.MetadataKey] = map[string]string{"c1": stamps["c1"][3:], "c2": stamps["c2"]}
			},
			wantErr: true,
		},
		{
			name:    "malformed stamp map",
			mutate:  func(ev *session.Event) { ev.CustomMetadata[toolcallintegrity.MetadataKey] = "v1:00" },
			wantErr: true,
		},
		{
			name: "non-string stamp",
			mutate: func(ev *session.Event) {
				ev.CustomMetadata[toolcallintegrity.MetadataKey] = map[string]any{"c1": 1, "c2": 2}
			},
			wantErr: true,
		},
		{
			name:    "stamps removed",
			mutate:  func(ev *session.Event) { delete(ev.CustomMetadata, toolcallintegrity.MetadataKey) },
			wantErr: true,
		},
		{
			name: "identical call repeated in its event",
			mutate: func(ev *session.Event) {
				ev.Content.Parts = append(ev.Content.Parts, &genai.Part{FunctionCall: weatherCall("c1", "Paris")})
			},
			wantErr: true,
		},
		{
			name:    "duplicate call ID",
			mutate:  func(ev *session.Event) { ev.Content.Parts[1].FunctionCall.ID = "c1" },
			wantErr: true,
		},
		{
			name:    "rotated key still verifies",
			signCfg: &toolcallintegrity.Config{Keys: [][]byte{keyB}},
			cfg:     toolcallintegrity.Config{Keys: [][]byte{keyA, keyB}},
		},
		{
			name:    "retired key fails",
			signCfg: &toolcallintegrity.Config{Keys: [][]byte{keyB}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.cfg.Keys == nil {
				tt.cfg.Keys = [][]byte{keyA}
			}
			p := newPlugin(t, tt.cfg)
			signer := p
			if tt.signCfg != nil {
				signer = newPlugin(t, *tt.signCfg)
			}
			svc, sess := newSession(t)
			stored := stampAndStore(t, signer, svc, sess, callEvent("e1", "inv1", weatherCall("c1", "Paris"), weatherCall("c2", "Rome")), tt.mutate)

			_, err := p.BeforeRunCallback()(&fakeCtx{sess: stored})
			if tt.wantErr != (err != nil) {
				t.Fatalf("BeforeRunCallback() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, toolcallintegrity.ErrIntegrity) {
				t.Errorf("BeforeRunCallback() error = %v, want it to wrap ErrIntegrity", err)
			}
		})
	}
}

func TestBeforeRun_CallMovedToAnotherSession(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{Keys: [][]byte{keyA}})
	svc, sess := newSession(t)
	ev := callEvent("e1", "inv1", weatherCall("c1", "Paris"))
	stampAndStore(t, p, svc, sess, ev, nil)

	other, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s2"})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	copied := callEvent("e1", "inv1", weatherCall("c1", "Paris"))
	copied.CustomMetadata = ev.CustomMetadata
	if err := svc.AppendEvent(t.Context(), other.Session, copied); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}
	if _, err := p.BeforeRunCallback()(&fakeCtx{sess: other.Session}); !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("BeforeRunCallback() error = %v, want ErrIntegrity", err)
	}
}

func TestBeforeTool(t *testing.T) {
	tests := []struct {
		name     string
		cfg      toolcallintegrity.Config
		mutate   func(*session.Event)
		callID   string
		toolName string
		args     map[string]any
		wantErr  bool
	}{
		{name: "verified call runs"},
		{name: "argument as a float after a JSON round trip", args: map[string]any{"city": "Paris", "days": 3.0}},
		{name: "no call ID", callID: "-", wantErr: true},
		{name: "no call ID, unstamped calls allowed", callID: "-", cfg: toolcallintegrity.Config{AllowUnstampedCalls: true}, wantErr: true},
		{name: "call not in session", callID: "c9", wantErr: true},
		{name: "call not in session, unstamped calls allowed", callID: "c9", cfg: toolcallintegrity.Config{AllowUnstampedCalls: true}},
		{name: "different arguments", args: map[string]any{"city": "Berlin", "days": 3}, wantErr: true},
		{name: "different tool", toolName: "delete_all", wantErr: true},
		{
			name:   "unstamped stored call, unstamped calls allowed",
			cfg:    toolcallintegrity.Config{AllowUnstampedCalls: true},
			mutate: func(ev *session.Event) { delete(ev.CustomMetadata, toolcallintegrity.MetadataKey) },
		},
		{
			name:    "unstamped stored call with other arguments, unstamped calls allowed",
			cfg:     toolcallintegrity.Config{AllowUnstampedCalls: true},
			mutate:  func(ev *session.Event) { delete(ev.CustomMetadata, toolcallintegrity.MetadataKey) },
			args:    map[string]any{"city": "Berlin", "days": 3},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.cfg.Keys == nil {
				tt.cfg.Keys = [][]byte{keyA}
			}
			switch tt.callID {
			case "":
				tt.callID = "c1"
			case "-":
				tt.callID = ""
			}
			if tt.toolName == "" {
				tt.toolName = "get_weather"
			}
			if tt.args == nil {
				tt.args = map[string]any{"city": "Paris", "days": 3}
			}
			p := newPlugin(t, tt.cfg)
			svc, sess := newSession(t)
			stored := stampAndStore(t, p, svc, sess, callEvent("e1", "inv1", weatherCall("c1", "Paris")), tt.mutate)
			if _, err := p.BeforeRunCallback()(&fakeCtx{sess: stored}); err != nil {
				t.Fatalf("BeforeRunCallback() error = %v", err)
			}

			result, err := p.BeforeToolCallback()(&fakeCtx{callID: tt.callID}, fakeTool{name: tt.toolName}, tt.args)
			if result != nil {
				t.Errorf("BeforeToolCallback() result = %v, want nil", result)
			}
			if tt.wantErr != (err != nil) {
				t.Fatalf("BeforeToolCallback() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, toolcallintegrity.ErrIntegrity) {
				t.Errorf("BeforeToolCallback() error = %v, want it to wrap ErrIntegrity", err)
			}
		})
	}
}

func TestBeforeTool_CallStampedDuringTheRun(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{Keys: [][]byte{keyA}})
	_, sess := newSession(t)
	ctx := &fakeCtx{sess: sess}
	if _, err := p.BeforeRunCallback()(ctx); err != nil {
		t.Fatalf("BeforeRunCallback() error = %v", err)
	}
	partial := callEvent("e0", "inv-current", weatherCall("c2", "Rome"))
	partial.Partial = true
	for _, ev := range []*session.Event{partial, callEvent("e1", "inv-current", weatherCall("c1", "Paris"))} {
		if _, err := p.OnEventCallback()(ctx, ev); err != nil {
			t.Fatalf("OnEventCallback() error = %v", err)
		}
	}
	beforeTool := func(callID string, args map[string]any) error {
		_, err := p.BeforeToolCallback()(&fakeCtx{callID: callID}, fakeTool{name: "get_weather"}, args)
		return err
	}

	if err := beforeTool("c1", map[string]any{"city": "Paris", "days": 3}); err != nil {
		t.Errorf("BeforeToolCallback() for a call stamped in this run: error = %v, want nil", err)
	}
	if err := beforeTool("c2", map[string]any{"city": "Rome", "days": 3}); !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("BeforeToolCallback() for a call seen only on a partial event: error = %v, want ErrIntegrity", err)
	}
	other := &fakeCtx{callID: "c1", invocationID: "inv-other"}
	if _, err := p.BeforeToolCallback()(other, fakeTool{name: "get_weather"}, map[string]any{"city": "Paris", "days": 3}); !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("BeforeToolCallback() from another invocation: error = %v, want ErrIntegrity", err)
	}

	p.AfterRunCallback()(ctx)
	if err := beforeTool("c1", map[string]any{"city": "Paris", "days": 3}); !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("BeforeToolCallback() after the run ended: error = %v, want ErrIntegrity", err)
	}
}

func TestOnEvent_OutsideARunRecordsNothing(t *testing.T) {
	p := newPlugin(t, toolcallintegrity.Config{Keys: [][]byte{keyA}})
	_, sess := newSession(t)
	if _, err := p.OnEventCallback()(&fakeCtx{sess: sess}, callEvent("e1", "inv-current", weatherCall("c1", "Paris"))); err != nil {
		t.Fatalf("OnEventCallback() error = %v", err)
	}
	_, err := p.BeforeToolCallback()(&fakeCtx{callID: "c1"}, fakeTool{name: "get_weather"}, map[string]any{"city": "Paris", "days": 3})
	if !errors.Is(err, toolcallintegrity.ErrIntegrity) {
		t.Errorf("BeforeToolCallback() error = %v, want ErrIntegrity", err)
	}
}
