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
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// newTestInvocationContext builds a minimal *invocationContext for wrapper tests.
func newTestInvocationContext(t *testing.T) InvocationContext {
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

	return &invocationContext{
		Context:      t.Context(),
		agent:        a,
		artifacts:    fakeArtifacts{},
		memory:       fakeMemory{},
		session:      &fakeSession{id: "sess-1", appName: "app", userID: "user", state: &fakeState{}},
		invocationID: "inv-1",
		branch:       "branch-1",
		userContent:  genai.NewContentFromText("hi", genai.RoleUser),
	}
}

func TestWithAgentTimeoutOnToolContext(t *testing.T) {
	toolCtx := NewToolContext(newTestInvocationContext(t), "call-1", nil, nil)

	var (
		ctx    Context
		cancel func()
	)
	output := captureLog(t, func() {
		ctx, cancel = toolCtx.WithAgentTimeout(5 * time.Second)
	})
	if ctx == nil || cancel == nil {
		t.Fatalf("WithAgentTimeout() = (%T, nilCancel=%v), want a usable Context and a non-nil CancelFunc", ctx, cancel == nil)
	}
	defer cancel()

	if output != "" {
		t.Errorf("WithAgentTimeout() unexpectedly emitted log output: %q", output)
	}

	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("WithAgentTimeout() context has no deadline")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > 5*time.Second {
		t.Errorf("deadline remaining = %v, want in (0, 5s]", remaining)
	}
}

func TestWithAgentTimeoutOnCallbackContext(t *testing.T) {
	cbCtx := NewCallbackContext(newTestInvocationContext(t), &session.EventActions{})

	var (
		ctx    Context
		cancel func()
	)
	output := captureLog(t, func() {
		ctx, cancel = cbCtx.WithAgentTimeout(5 * time.Second)
	})
	if ctx == nil || cancel == nil {
		t.Fatalf("WithAgentTimeout() = (%T, nilCancel=%v), want a usable Context and a non-nil CancelFunc", ctx, cancel == nil)
	}
	defer cancel()

	if output != "" {
		t.Errorf("WithAgentTimeout() unexpectedly emitted log output: %q", output)
	}

	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("WithAgentTimeout() context has no deadline")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > 5*time.Second {
		t.Errorf("deadline remaining = %v, want in (0, 5s]", remaining)
	}
}
