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
	"iter"
	"log"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
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

func TestCancelledToolContextForwardsPrivateKeys(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() = %v", err)
	}
	leafAg, err := agent.New(agent.Config{Name: "leaf"})
	if err != nil {
		t.Fatalf("agent.New(leaf): %v", err)
	}
	wantPath := "wf/node@1"
	pm := map[string]agent.Agent{"leaf": leafAg}
	ic := icontext.NewInvocationContext(adkcontext.WithParentMap(t.Context(), pm), icontext.InvocationContextParams{Session: resp.Session})

	var cancelToolCtx *cancelledToolContext
	var toolCtx agent.Context
	var wantTok any
	bot, err := agent.New(agent.Config{
		Name: "bot",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {
				wantTok = adkcontext.ReadRunToken(ctx)
				carried := agent.Promote(ctx).WithDelta(&agent.CommonContextDelta{Path: &wantPath}).WithAgentContext(ctx)
				toolCtx = agent.NewToolContext(carried, "fc-1", nil, nil)
				cancelCtx, cancel := toolCtx.WithAgentCancel()
				defer cancel()
				cancelToolCtx = &cancelledToolContext{Context: toolCtx, cancelCtx: cancelCtx}

				if got := adkcontext.ReadPath(cancelToolCtx); got != wantPath {
					t.Errorf("adkcontext.ReadPath(cancelToolCtx) = %q, want %q", got, wantPath)
				}
				if got, ok := adkcontext.ReadParentMap(cancelToolCtx).(map[string]agent.Agent); !ok || len(got) != 1 {
					t.Errorf("adkcontext.ReadParentMap(cancelToolCtx) = %#v, want 1-entry map", adkcontext.ReadParentMap(cancelToolCtx))
				}
				if got := adkcontext.ReadRunToken(cancelToolCtx); got != wantTok || wantTok == nil {
					t.Errorf("adkcontext.ReadRunToken(cancelToolCtx) = %#v, want %#v", got, wantTok)
				}
				wantCtxTok := ctx.(interface{ ADKContextToken() any }).ADKContextToken()
				if lin := adkcontext.WalkLineage(cancelToolCtx); !slices.Contains(lin, wantCtxTok) || !slices.Contains(lin, wantTok) {
					t.Errorf("adkcontext.WalkLineage(cancelToolCtx) = %#v, want ctx token %#v and runToken %#v", lin, wantCtxTok, wantTok)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(bot): %v", err)
	}
	for _, runErr := range bot.Run(ic) {
		if runErr != nil {
			t.Fatalf("bot.Run error: %v", runErr)
		}
	}
	if cancelToolCtx == nil {
		t.Fatal("cancelToolCtx was not constructed")
	}
}
