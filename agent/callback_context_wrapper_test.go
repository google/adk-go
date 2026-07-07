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

package agent

import (
	"bytes"
	"context"
	"iter"
	"log"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// captureLog redirects the output of the default [log.Logger] for the
// duration of fn and returns everything that was written to it.
func captureLog(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	origOut := log.Writer()
	origFlags := log.Flags()
	origPrefix := log.Prefix()
	log.SetOutput(&buf)
	// Reset flags/prefix so the captured output is deterministic and easy to
	// assert against.
	log.SetFlags(0)
	log.SetPrefix("")
	t.Cleanup(func() {
		log.SetOutput(origOut)
		log.SetFlags(origFlags)
		log.SetPrefix(origPrefix)
	})
	fn()
	return buf.String()
}

// fakeSession is a minimal session.Session implementation sufficient for the
// callback context wrapper tests.
type fakeSession struct {
	session.Session
	id      string
	appName string
	userID  string
	state   session.State
}

func (s *fakeSession) ID() string           { return s.id }
func (s *fakeSession) AppName() string      { return s.appName }
func (s *fakeSession) UserID() string       { return s.userID }
func (s *fakeSession) State() session.State { return s.state }

// fakeState is a no-op session.State implementation.
type fakeState struct {
	session.State
}

func (s *fakeState) Get(string) (any, error) { return nil, nil }
func (s *fakeState) Set(string, any) error   { return nil }
func (s *fakeState) All() iter.Seq2[string, any] {
	return func(func(string, any) bool) {}
}

// fakeArtifacts is a no-op Artifacts implementation.
type fakeArtifacts struct{}

func (fakeArtifacts) Save(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
	return nil, nil
}
func (fakeArtifacts) List(context.Context) (*artifact.ListResponse, error) { return nil, nil }
func (fakeArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return nil, nil
}

func (fakeArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, nil
}

// fakeMemory is a no-op Memory implementation.
type fakeMemory struct{}

func (fakeMemory) AddSessionToMemory(context.Context, session.Session) error { return nil }
func (fakeMemory) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return &memory.SearchResponse{}, nil
}

// newTestCallbackContext builds a callback context (a *callbackContextWrapper)
// backed by a fully-populated invocationContext so that the supported methods
// have meaningful values to delegate to.
func newTestCallbackContext(t *testing.T) Context {
	t.Helper()

	a, err := New(Config{
		Name: "test-agent",
		Run: func(InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ic := &invocationContext{
		Context:      t.Context(),
		agent:        a,
		artifacts:    fakeArtifacts{},
		memory:       fakeMemory{},
		session:      &fakeSession{id: "sess-1", appName: "app", userID: "user", state: &fakeState{}},
		invocationID: "inv-1",
		branch:       "branch-1",
		userContent:  genai.NewContentFromText("hi", genai.RoleUser),
	}
	return NewCallbackContext(ic, nil)
}

// TestCallbackContextWrapper_LogsForToolContextMethods verifies that calling
// tool-context-only methods on a callback context (obtained via
// NewCallbackContext, which wraps the underlying commonContext in a
// callbackContextWrapper) emits a log entry indicating the method is not
// supported, and returns the documented "no-op" value.
func TestCallbackContextWrapper_LogsForToolContextMethods(t *testing.T) {
	t.Run("Actions logs and returns nil", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got *session.EventActions
		output := captureLog(t, func() { got = cc.Actions() })
		if got != nil {
			t.Errorf("Actions() = %v, want nil", got)
		}
		if !strings.Contains(output, "Actions()") || !strings.Contains(output, "not supported") {
			t.Errorf("Actions() log = %q, want log mentioning %q and %q", output, "Actions()", "not supported")
		}
	})

	t.Run("FunctionCallID logs and returns empty string", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got string
		output := captureLog(t, func() { got = cc.FunctionCallID() })
		if got != "" {
			t.Errorf("FunctionCallID() = %q, want \"\"", got)
		}
		if !strings.Contains(output, "FunctionCallID()") || !strings.Contains(output, "not supported") {
			t.Errorf("FunctionCallID() log = %q, want log mentioning %q and %q", output, "FunctionCallID()", "not supported")
		}
	})

	t.Run("RequestConfirmation logs and returns error", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var gotErr error
		output := captureLog(t, func() { gotErr = cc.RequestConfirmation("hint", "payload") })
		if gotErr == nil {
			t.Errorf("RequestConfirmation() error = nil, want non-nil")
		}
		if !strings.Contains(output, "RequestConfirmation()") || !strings.Contains(output, "not supported") {
			t.Errorf("RequestConfirmation() log = %q, want log mentioning %q and %q", output, "RequestConfirmation()", "not supported")
		}
	})

	t.Run("SearchMemory logs and returns error", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var (
			gotResp *memory.SearchResponse
			gotErr  error
		)
		output := captureLog(t, func() { gotResp, gotErr = cc.SearchMemory(t.Context(), "query") })
		if gotResp != nil {
			t.Errorf("SearchMemory() resp = %v, want nil", gotResp)
		}
		if gotErr == nil {
			t.Errorf("SearchMemory() error = nil, want non-nil")
		}
		if !strings.Contains(output, "SearchMemory()") || !strings.Contains(output, "not supported") {
			t.Errorf("SearchMemory() log = %q, want log mentioning %q and %q", output, "SearchMemory()", "not supported")
		}
	})

	t.Run("ToolConfirmation logs and returns nil", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got *toolconfirmation.ToolConfirmation
		output := captureLog(t, func() { got = cc.ToolConfirmation() })
		if got != nil {
			t.Errorf("ToolConfirmation() = %v, want nil", got)
		}
		if !strings.Contains(output, "ToolConfirmation()") || !strings.Contains(output, "not supported") {
			t.Errorf("ToolConfirmation() log = %q, want log mentioning %q and %q", output, "ToolConfirmation()", "not supported")
		}
	})
}

// TestCallbackContextWrapper_NoLogForSupportedMethods verifies that methods
// which are valid on a callback context (i.e. the ones the wrapper simply
// delegates to the underlying context) do not emit any log entry and return
// the values produced by the underlying invocation context.
func TestCallbackContextWrapper_NoLogForSupportedMethods(t *testing.T) {
	cases := []struct {
		name string
		call func(cc Context) any
		want any
	}{
		{"AgentName", func(cc Context) any { return cc.AgentName() }, "test-agent"},
		{"AppName", func(cc Context) any { return cc.AppName() }, "app"},
		{"Branch", func(cc Context) any { return cc.Branch() }, "branch-1"},
		{"InvocationID", func(cc Context) any { return cc.InvocationID() }, "inv-1"},
		{"SessionID", func(cc Context) any { return cc.SessionID() }, "sess-1"},
		{"UserID", func(cc Context) any { return cc.UserID() }, "user"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cc := newTestCallbackContext(t)
			var got any
			output := captureLog(t, func() { got = tc.call(cc) })
			if got != tc.want {
				t.Errorf("%s() = %v, want %v", tc.name, got, tc.want)
			}
			if output != "" {
				t.Errorf("%s() unexpectedly emitted log output: %q", tc.name, output)
			}
		})
	}

	t.Run("Artifacts", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got Artifacts
		output := captureLog(t, func() { got = cc.Artifacts() })
		if got == nil {
			t.Errorf("Artifacts() = nil, want non-nil")
		}
		if output != "" {
			t.Errorf("Artifacts() unexpectedly emitted log output: %q", output)
		}
	})

	t.Run("UserContent", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got *genai.Content
		output := captureLog(t, func() { got = cc.UserContent() })
		if got == nil {
			t.Errorf("UserContent() = nil, want non-nil")
		}
		if output != "" {
			t.Errorf("UserContent() unexpectedly emitted log output: %q", output)
		}
	})

	t.Run("ReadonlyState", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got session.ReadonlyState
		output := captureLog(t, func() { got = cc.ReadonlyState() })
		if got == nil {
			t.Errorf("ReadonlyState() = nil, want non-nil")
		}
		if output != "" {
			t.Errorf("ReadonlyState() unexpectedly emitted log output: %q", output)
		}
	})

	t.Run("State", func(t *testing.T) {
		cc := newTestCallbackContext(t)
		var got session.State
		output := captureLog(t, func() { got = cc.State() })
		if got == nil {
			t.Errorf("State() = nil, want non-nil")
		}
		if output != "" {
			t.Errorf("State() unexpectedly emitted log output: %q", output)
		}
	})
}
