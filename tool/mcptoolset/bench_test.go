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

package mcptoolset_test

import (
	"context"
	"fmt"
	"io"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// benchMCPServer runs an in-memory MCP server with n tools whose tools/list
// handler counts its calls and sleeps listDelay before answering, so a
// network round trip is simulated and the saving shows in ns/op too.
func benchMCPServer(b *testing.B, n int, listDelay time.Duration) (mcp.Transport, *atomic.Int64) {
	b.Helper()
	var listCalls atomic.Int64
	server := mcp.NewServer(&mcp.Implementation{Name: "bench_server", Version: "v1.0.0"}, nil)
	for i := range n {
		mcp.AddTool(server, &mcp.Tool{Name: fmt.Sprintf("tool%d", i), Description: "returns weather in the given city"}, weatherFunc)
	}
	server.AddReceivingMiddleware(func(next mcp.MethodHandler) mcp.MethodHandler {
		return func(ctx context.Context, method string, req mcp.Request) (mcp.Result, error) {
			if method == "tools/list" {
				listCalls.Add(1)
				if listDelay > 0 {
					time.Sleep(listDelay)
				}
			}
			return next(ctx, method, req)
		}
	})
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(b.Context(), serverTransport, nil)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = serverSession.Close() })
	return clientTransport, &listCalls
}

// BenchmarkMCPToolLoopTurn drives the tool loop of runner's
// BenchmarkRunToolLoopTurn (three MCP tool calls, then a text answer, so four
// model calls per turn) against one MCP toolset over an in-memory transport,
// and reports the tools/list RPCs the server saw per turn. The toolset lives
// across turns, as it does in a server process; every turn is a new session.
func BenchmarkMCPToolLoopTurn(b *testing.B) {
	const tools, calls = 20, 3
	for _, delay := range []time.Duration{0, 20 * time.Millisecond} {
		b.Run(fmt.Sprintf("list-delay=%s", delay), func(b *testing.B) {
			transport, listCalls := benchMCPServer(b, tools, delay)
			ts, err := mcptoolset.New(mcptoolset.Config{Transport: transport})
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() { _ = ts.(io.Closer).Close() })
			m := &benchutil.ToolLoopModel{Calls: calls, ToolName: "tool0", Args: map[string]any{"city": "london"}}
			a, err := llmagent.New(llmagent.Config{Name: "agent", Model: m, Instruction: "You are terse.", Toolsets: []tool.Toolset{ts}})
			if err != nil {
				b.Fatal(err)
			}
			r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				b.Fatal(err)
			}
			msg := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			turns := 0
			for b.Loop() {
				turns++
				for _, err := range r.Run(b.Context(), "user", fmt.Sprintf("s%d", turns), msg, agent.RunConfig{}) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(listCalls.Load())/float64(turns), "list-tools-rpcs/turn")
			b.ReportMetric(float64(m.ModelCalls.Load())/float64(turns), "model-calls/turn")
		})
	}
}
