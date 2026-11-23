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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

func TestRunLive_UndeclaredToolIsNotCallable(t *testing.T) {
	// The sender goroutine logs each tool response it sends. Waiting for it
	// to exit keeps that log line out of a later test that captures the log.
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	const toolCall = `{"toolCall":{"functionCalls":[{"id":"nd_1","name":"no_decl","args":{}},{"id":"d_1","name":"decl_tool","args":{}}]}}`
	client, _ := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if err := conn.WriteMessage(websocket.TextMessage, []byte(toolCall)); err != nil {
			t.Errorf("writing toolCall failed: %v", err)
			return
		}
		blockUntilClientCloses(conn)
	})

	noDecl := &packingTool{name: "no_decl"}
	declTool := &packingTool{name: "decl_tool", decl: &genai.FunctionDeclaration{Name: "decl_tool"}}
	f := &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
		Tools:             []tool.Tool{noDecl, declTool},
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()
	next, stop := iter.Pull2(seq)
	defer stop()

	responses := map[string]map[string]any{}
	for len(responses) < 2 {
		ev, err, ok := next()
		if !ok {
			t.Fatalf("iterator ended after function responses for %d of 2 calls", len(responses))
		}
		if err != nil {
			t.Fatalf("RunLive yielded an error: %v", err)
		}
		if ev == nil || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.FunctionResponse != nil {
				responses[p.FunctionResponse.Name] = p.FunctionResponse.Response
			}
		}
	}

	// The live callable map holds decl_tool only, so the error lists it alone.
	errText, _ := responses["no_decl"]["error"].(string)
	if want := "Invoking `no_decl()` failed as no tool with that name is available. The tools you can call are: decl_tool."; !strings.HasPrefix(errText, want) {
		t.Errorf("no_decl response error = %q, want prefix %q", errText, want)
	}
	if diff := cmp.Diff(map[string]any{"ran": "decl_tool"}, responses["decl_tool"]); diff != "" {
		t.Errorf("decl_tool response mismatch (-want +got):\n%s", diff)
	}
	if got := noDecl.runs.Load(); got != 0 {
		t.Errorf("no_decl Run called %d times, want 0", got)
	}
	if got := declTool.runs.Load(); got != 1 {
		t.Errorf("decl_tool Run called %d times, want 1", got)
	}
}
