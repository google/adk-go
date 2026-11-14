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

package services

import (
	"context"
	"io"
	"iter"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// slowStdioServerEnv makes the test binary act as the slow MCP server.
const slowStdioServerEnv = "ADK_TEST_SLOW_MCP_SERVER"

// TestSlowStdioServerHelper is re-executed as a stdio MCP server that takes
// longer to answer initialize than the graph waits, like a server launched
// through npx -y or uvx that downloads its package first.
func TestSlowStdioServerHelper(t *testing.T) {
	if os.Getenv(slowStdioServerEnv) != "1" {
		t.Skip("helper process for TestAgentRunAfterGraphWithSlowStdioServer")
	}
	time.Sleep(time.Second)
	srv := mcp.NewServer(&mcp.Implementation{Name: "slow", Version: "v0"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "lookup"}, func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
		return &mcp.CallToolResult{}, struct{}{}, nil
	})
	_ = srv.Run(context.Background(), &mcp.StdioTransport{})
	os.Exit(0)
}

// okModel answers every request with a short text reply.
type okModel struct{}

func (okModel) Name() string { return "ok" }

func (okModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("ok", genai.RoleModel)}, nil)
	}
}

// TestAgentRunAfterGraphWithSlowStdioServer pins that drawing the graph while
// a stdio MCP server is still starting leaves the toolset usable. If the graph
// canceled the listing, the MCP SDK would kill the server's process, and
// mcptoolset, which cannot start that process again, would fail every later
// run of the agent with "exec: Stdout already set".
func TestAgentRunAfterGraphWithSlowStdioServer(t *testing.T) {
	shortenToolsetWait(t, 100*time.Millisecond)
	cmd := exec.Command(os.Args[0], "-test.run=^TestSlowStdioServerHelper$")
	cmd.Env = append(os.Environ(), slowStdioServerEnv+"=1")
	ts, err := mcptoolset.New(mcptoolset.Config{Transport: &mcp.CommandTransport{Command: cmd}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ts.(io.Closer).Close() })
	root, err := llmagent.New(llmagent.Config{Name: "root", Model: okModel{}, Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatal(err)
	}

	// The dev UI requests the graph when the app is opened, before any run.
	dot := buildGraphWithin(t, context.Background(), root, 10*time.Second)
	if parseGraph(t, dot).IsNode("lookup") {
		t.Fatalf("drew lookup before the server answered; the test needs a server slower than the graph's budget\n%s", dot)
	}

	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range r.Run(context.Background(), "user", "s1", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("agent run after the graph: %v", err)
		}
	}

	// The listing the graph started has finished, so the next graph has the
	// server's tool.
	if dot := buildGraphWithin(t, context.Background(), root, 10*time.Second); lookupEdge(t, parseGraph(t, dot), "root", "lookup") == nil {
		t.Errorf("missing edge root -> lookup once the server answered\n%s", dot)
	}
}
