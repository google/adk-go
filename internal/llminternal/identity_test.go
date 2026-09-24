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

package llminternal

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

// TestCancelledToolContextKeepsIdentity pins that the context a streaming tool
// runs under is one of ADK's own, not a foreign decorator.
//
// It lives outside package agent and holds no session of its own — its embedded
// tool context returns nil by design — so without the module marker the identity
// procedure would read that nil session and report no user. A streaming tool
// that re-derives its context would then lose the acting user, and a per-user
// credential could not be minted for it at all.
func TestCancelledToolContextKeepsIdentity(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() = %v", err)
	}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
	toolCtx := agent.NewToolContext(ic, "fc", nil, nil)
	cancelCtx, cancel := toolCtx.WithAgentCancel()
	defer cancel()
	cancelToolCtx := &cancelledToolContext{Context: toolCtx, cancelCtx: cancelCtx}

	var buf bytes.Buffer
	out := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(out) })

	want := agent.Identity{UserID: "u", AppName: "app", SessionID: resp.Session.ID()}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"the cancel-scoped context itself", cancelToolCtx},
		{"promoted", agent.Promote(cancelToolCtx)},
		{"a tool context re-derived from it", agent.NewToolContext(cancelToolCtx, "fc2", nil, nil)},
		{"a callback context re-derived from it", agent.NewCallbackContext(cancelToolCtx, nil)},
		{"a readonly context over it", icontext.NewReadonlyContext(cancelToolCtx)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := agent.IdentityFromContext(tc.ctx)
			if !ok {
				t.Fatalf("IdentityFromContext() ok = false, want the invocation's identity")
			}
			if id != want {
				t.Errorf("IdentityFromContext() = %+v, want %+v", id, want)
			}
		})
	}
	// Reading the session to find the identity would also log on every lookup,
	// and auth.Transport resolves a credential per outbound request.
	if got := buf.String(); strings.Contains(got, "Session()") {
		t.Errorf("resolving the identity logged %q; it must not read a tool context's session", got)
	}
}
