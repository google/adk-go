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
	"errors"
	"io"
	"slices"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

func newPrefixTestToolset(t *testing.T, cfg mcptoolset.Config, names ...string) (tool.Toolset, <-chan string) {
	t.Helper()
	calls := make(chan string, 16)
	server := mcp.NewServer(&mcp.Implementation{Name: "search_server", Version: "1"}, nil)
	for _, name := range names {
		server.AddTool(&mcp.Tool{
			Name:        name,
			Description: "Search a collection",
			InputSchema: map[string]any{"type": "object"},
		}, func(_ context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			calls <- req.Params.Name
			return &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: cfg.ToolNamePrefix},
			}}, nil
		})
	}
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatalf("server.Connect() failed: %v", err)
	}
	t.Cleanup(func() { _ = serverSession.Close() })
	cfg.Transport = clientTransport
	ts, err := mcptoolset.New(cfg)
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}
	t.Cleanup(func() {
		if err := ts.(io.Closer).Close(); err != nil {
			t.Errorf("toolset.Close() failed: %v", err)
		}
	})
	return ts, calls
}

func prefixTestTools(t *testing.T, ts tool.Toolset) []tool.Tool {
	t.Helper()
	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))
	tools, err := ts.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools() failed: %v", err)
	}
	return tools
}

func TestMCPToolNamePrefixRunner(t *testing.T) {
	docs, docsCalls := newPrefixTestToolset(t, mcptoolset.Config{ToolNamePrefix: "docs"}, "search")
	issues, issuesCalls := newPrefixTestToolset(t, mcptoolset.Config{ToolNamePrefix: "issues"}, "search")
	llm := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "docs_call", Name: "docs_search", Args: map[string]any{}}},
			{FunctionCall: &genai.FunctionCall{ID: "issues_call", Name: "issues_search", Args: map[string]any{}}},
		}},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	callbackNames := make(chan string, 2)
	a, err := llmagent.New(llmagent.Config{
		Name:     "search_agent",
		Model:    llm,
		Toolsets: []tool.Toolset{docs, issues},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{
			func(_ agent.Context, called tool.Tool, _ map[string]any) (map[string]any, error) {
				callbackNames <- called.Name()
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	wantResponses := map[string]string{"docs_search": "docs", "issues_search": "issues"}
	responses := 0
	for event, err := range testutil.NewTestAgentRunner(t, a).Run(t, "session", "Search both collections") {
		if err != nil {
			t.Fatalf("Runner.Run() failed: %v", err)
		}
		if event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if response := part.FunctionResponse; response != nil {
				source, ok := wantResponses[response.Name]
				if !ok || response.Response["output"] != source || response.ID != source+"_call" {
					t.Error("function response does not match its tool, server, and call ID")
				}
				delete(wantResponses, response.Name)
				responses++
			}
		}
	}
	if responses != 2 || len(wantResponses) != 0 {
		t.Errorf("received %d function responses, want one per server", responses)
	}
	for _, calls := range []<-chan string{docsCalls, issuesCalls} {
		if len(calls) != 1 {
			t.Fatalf("MCP call count = %d, want 1", len(calls))
		}
		if <-calls != "search" {
			t.Error("MCP server received a renamed tool")
		}
	}
	if len(callbackNames) != 2 {
		t.Fatalf("tool callback count = %d, want 2", len(callbackNames))
	}
	names := []string{<-callbackNames, <-callbackNames}
	slices.Sort(names)
	if !slices.Equal(names, []string{"docs_search", "issues_search"}) {
		t.Error("tool callbacks did not receive exposed names")
	}
}

func TestMCPToolNamePrefixRegistration(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prefix string
		want   string
	}{
		{name: "empty", want: "search"},
		{name: "prefix", prefix: "docs", want: "docs_search"},
		{name: "literal prefix", prefix: "docs_", want: "docs__search"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts, _ := newPrefixTestToolset(t, mcptoolset.Config{ToolNamePrefix: tc.prefix}, "search")
			first := prefixTestTools(t, ts)
			second := prefixTestTools(t, ts)
			for _, tools := range [][]tool.Tool{first, second} {
				if len(tools) != 1 {
					t.Fatalf("Tools() returned %d tools, want 1", len(tools))
				}
				got := tools[0].(toolinternal.FunctionTool)
				if got.Name() != tc.want || got.Declaration().Name != tc.want {
					t.Error("tool and declaration names do not match the configured prefix")
				}
			}
			req := &model.LLMRequest{}
			if err := first[0].(toolinternal.RequestProcessor).ProcessRequest(nil, req); err != nil {
				t.Fatalf("ProcessRequest() failed: %v", err)
			}
			if len(req.Tools) != 1 || req.Tools[tc.want] != first[0] {
				t.Error("request registry did not preserve the exposed name and tool")
			}
			if req.Config == nil || len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
				t.Fatal("request must contain exactly one function declaration")
			}
			if req.Config.Tools[0].FunctionDeclarations[0].Name != tc.want {
				t.Error("request declaration does not use the exposed name")
			}
		})
	}
}

func TestMCPToolNamePrefixFilter(t *testing.T) {
	for _, tc := range []struct {
		name     string
		internal string
		external string
		want     int
	}{
		{name: "original inside, exposed outside", internal: "search", external: "docs_search", want: 1},
		{name: "exposed inside is excluded", internal: "docs_search", external: "docs_search"},
		{name: "original outside is excluded", internal: "search", external: "search"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts, _ := newPrefixTestToolset(t, mcptoolset.Config{
				ToolNamePrefix: "docs",
				ToolFilter:     tool.AllowedToolsPredicate([]string{tc.internal}),
			}, "search", "list")
			filtered := tool.FilterToolset(ts, tool.AllowedToolsPredicate([]string{tc.external}))
			if got := len(prefixTestTools(t, filtered)); got != tc.want {
				t.Errorf("filtered tool count = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestMCPToolNamePrefixConfirmation(t *testing.T) {
	for _, mode := range []string{"native", "wrapper"} {
		t.Run(mode, func(t *testing.T) {
			providerCalls := 0
			provider := func(name string, _ any) bool {
				providerCalls++
				if name != "docs_search" {
					t.Error("confirmation provider did not receive the exposed name")
				}
				return true
			}
			cfg := mcptoolset.Config{ToolNamePrefix: "docs"}
			if mode == "native" {
				cfg.RequireConfirmationProvider = provider
			}
			ts, calls := newPrefixTestToolset(t, cfg, "search")
			if mode == "wrapper" {
				ts = tool.WithConfirmation(ts, false, provider)
			}
			for _, tc := range []struct {
				name         string
				confirmation *toolconfirmation.ToolConfirmation
				wantErr      error
				wantCalls    int
			}{
				{name: "pending", wantErr: tool.ErrConfirmationRequired},
				{name: "approved", confirmation: &toolconfirmation.ToolConfirmation{Confirmed: true}, wantCalls: 1},
				{name: "rejected", confirmation: &toolconfirmation.ToolConfirmation{}, wantErr: tool.ErrConfirmationRejected},
			} {
				t.Run(tc.name, func(t *testing.T) {
					tools := prefixTestTools(t, ts)
					if len(tools) != 1 {
						t.Fatalf("Tools() returned %d tools, want 1", len(tools))
					}
					req := &model.LLMRequest{}
					if err := tools[0].(toolinternal.RequestProcessor).ProcessRequest(nil, req); err != nil {
						t.Fatalf("ProcessRequest() failed: %v", err)
					}
					dispatched, ok := req.Tools["docs_search"].(toolinternal.FunctionTool)
					if !ok || dispatched != tools[0] {
						t.Fatal("request registry did not preserve the tool's confirmation wrapper")
					}
					ctx := agent.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), "call", nil, tc.confirmation)
					result, err := dispatched.Run(ctx, map[string]any{})
					if !errors.Is(err, tc.wantErr) {
						t.Errorf("Run() error = %v, want %v", err, tc.wantErr)
					}
					if tc.confirmation == nil {
						confirmation, ok := ctx.Actions().RequestedToolConfirmations["call"]
						if !ok || !strings.Contains(confirmation.Hint, "docs_search()") || !ctx.Actions().SkipSummarization {
							t.Error("pending confirmation does not identify the exposed tool or pause the flow")
						}
					}
					if len(calls) != tc.wantCalls {
						t.Fatalf("MCP call count = %d, want %d", len(calls), tc.wantCalls)
					}
					if tc.wantCalls == 1 {
						if <-calls != "search" || result["output"] != "docs" {
							t.Error("approved call did not execute the original server tool")
						}
					}
				})
			}
			if providerCalls != 1 {
				t.Errorf("confirmation provider call count = %d, want 1", providerCalls)
			}
		})
	}
}
