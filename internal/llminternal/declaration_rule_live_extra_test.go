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
	"iter"
	"testing"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// registeringOther is a declared tool whose ProcessRequest registers a
// different declared tool under the same name.
type registeringOther struct {
	packingTool
	registers *packingTool
}

func (r *registeringOther) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, r.registers)
}

// runLiveUntilResponses serves the single server message toolCall to a live
// session of f and returns once the session has yielded want function
// responses and its goroutines have exited.
func runLiveUntilResponses(t *testing.T, f *Flow, toolCall string, want int) {
	t.Helper()
	// The sender goroutine logs each tool response it sends. Waiting for it
	// to exit keeps that log line out of a later test that captures the log.
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	client, _ := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			t.Errorf("writing toolCall failed: %v", err)
			return
		}
		blockUntilClientCloses(conn)
	})
	f.Model = &fakeLiveModel{client: client}
	f.RequestProcessors = []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()
	next, stop := iter.Pull2(seq)
	defer stop()

	for got := 0; got < want; {
		ev, err, ok := next()
		if !ok {
			t.Fatalf("iterator ended after %d of %d function responses", got, want)
		}
		if err != nil {
			t.Fatalf("RunLive yielded an error: %v", err)
		}
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				got++
			}
		}
	}
}

// A declared tool of f.Tools runs for its own name in a live session, even
// when its ProcessRequest registers another declared tool under that name.
func TestRunLive_FlowToolRunsWhenProcessRequestRegistersAnotherUnderItsName(t *testing.T) {
	registered := &packingTool{name: "shared", decl: &genai.FunctionDeclaration{Name: "shared"}}
	own := &registeringOther{
		packingTool: packingTool{name: "shared", decl: &genai.FunctionDeclaration{Name: "shared"}},
		registers:   registered,
	}
	f := &Flow{Tools: []tool.Tool{own}}

	runLiveUntilResponses(t, f, `{"toolCall":{"functionCalls":[{"id":"s_1","name":"shared","args":{}}]}}`, 1)

	if got := own.runs.Load(); got != 1 {
		t.Errorf("f.Tools tool Run called %d times, want 1", got)
	}
	if got := registered.runs.Load(); got != 0 {
		t.Errorf("registered tool Run called %d times, want 0", got)
	}
}

// A tool that never registers itself stays callable in a live session, also
// when it follows a tool that registers itself without a declaration.
func TestRunLive_NonRegisteringToolAfterUndeclaredToolStaysCallable(t *testing.T) {
	ran := make(chan struct{}, 1)
	plain := &mockFunctionTool{name: "plain", runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
		ran <- struct{}{}
		return map[string]any{"ok": true}, nil
	}}
	f := &Flow{Tools: []tool.Tool{&packingTool{name: "no_decl"}, plain}}

	runLiveUntilResponses(t, f, `{"toolCall":{"functionCalls":[{"id":"p_1","name":"plain","args":{}}]}}`, 1)

	if len(ran) != 1 {
		t.Errorf("plain Run was called %d times, want 1", len(ran))
	}
}
