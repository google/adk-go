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

package anthropicmodel_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// TestAgentRunner_ToolResultIsError runs real tools through the runner and
// checks the is_error Claude receives for each result: a functiontool that
// succeeds with an "error" field in its data, a functiontool that fails, and an
// MCP tool whose CallToolResult reports IsError.
func TestAgentRunner_ToolResultIsError(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []map[string]any
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(data, &body)
		mu.Lock()
		requests = append(requests, body)
		first := len(requests) == 1
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if first {
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"tool_use","content":[`+
				`{"type":"tool_use","id":"toolu_measure","name":"measure","input":{}},`+
				`{"type":"tool_use","id":"toolu_fail","name":"fail","input":{}},`+
				`{"type":"tool_use","id":"toolu_lookup","name":"lookup","input":{}}],"usage":{"input_tokens":5,"output_tokens":5}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"done"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
	}))
	defer srv.Close()

	claude, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	type noArgs struct{}
	// A successful measurement whose data has an "error" field.
	measure, err := functiontool.New(functiontool.Config{Name: "measure", Description: "measures"},
		func(agent.Context, noArgs) (map[string]any, error) {
			return map[string]any{"estimate": 42, "error": 0.125}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(measure) error = %v", err)
	}
	fail, err := functiontool.New(functiontool.Config{Name: "fail", Description: "fails"},
		func(agent.Context, noArgs) (map[string]any, error) {
			return nil, errors.New("upstream timeout")
		})
	if err != nil {
		t.Fatalf("functiontool.New(fail) error = %v", err)
	}

	mcpServer := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "1"}, nil)
	mcpServer.AddTool(&mcp.Tool{Name: "lookup", Description: "looks up", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "backend down"}}}, nil
		})
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	if _, err := mcpServer.Connect(t.Context(), serverTransport, nil); err != nil {
		t.Fatalf("mcp server Connect() error = %v", err)
	}
	mcpTools, err := mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	if err != nil {
		t.Fatalf("mcptoolset.New() error = %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:     "root",
		Model:    claude,
		Tools:    []tool.Tool{measure, fail},
		Toolsets: []tool.Toolset{mcpTools},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("model requests = %d, want 2", len(requests))
	}
	got := map[string]any{}
	msgs, _ := requests[1]["messages"].([]any)
	for _, m := range msgs {
		blocks, _ := m.(map[string]any)["content"].([]any)
		for _, b := range blocks {
			block, _ := b.(map[string]any)
			if block["type"] == "tool_result" {
				got[block["tool_use_id"].(string)] = block["is_error"]
			}
		}
	}
	want := map[string]any{"toolu_measure": false, "toolu_fail": true, "toolu_lookup": true}
	for id, w := range want {
		if got[id] != w {
			t.Errorf("tool_result %s is_error = %v, want %v", id, got[id], w)
		}
	}
}
