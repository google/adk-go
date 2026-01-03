// Copyright 2025 Google LLC
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
	"iter"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
)

type Input struct {
	City string `json:"city" jsonschema:"city name"`
}

type Output struct {
	WeatherSummary string `json:"weather_summary" jsonschema:"weather summary in the given city"`
}

func weatherFunc(ctx context.Context, req *mcp.CallToolRequest, input Input) (*mcp.CallToolResult, Output, error) {
	return nil, Output{
		WeatherSummary: fmt.Sprintf("Today in %q is sunny", input.City),
	}, nil
}

const modelName = "gemini-2.5-flash"

//go:generate go test -httprecord=.*

func TestMCPToolSet(t *testing.T) {
	const (
		toolName        = "get_weather"
		toolDescription = "returns weather in the given city"
	)

	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	// Run in-memory MCP server.
	server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: toolName, Description: toolDescription}, weatherFunc)
	_, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: clientTransport,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       newGeminiModel(t, modelName),
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city.",
		Toolsets: []tool.Toolset{
			ts,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	prompt := "what is the weather in london?"
	runner := newTestAgentRunner(t, agent)

	var gotEvents []*session.Event
	for event, err := range runner.Run(t, "session1", prompt) {
		if err != nil {
			t.Fatal(err)
		}
		gotEvents = append(gotEvents, event)
	}

	wantEvents := []*session.Event{
		{
			Author: "weather_time_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "get_weather",
								Args: map[string]any{"city": "london"},
							},
						},
					},
					Role: genai.RoleModel,
				},
			},
		},
		{
			Author: "weather_time_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name: "get_weather",
								Response: map[string]any{
									"output": map[string]any{"weather_summary": string(`Today in "london" is sunny`)},
								},
							},
						},
					},
					Role: genai.RoleUser,
				},
			},
		},
		{
			Author: "weather_time_agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							Text: `Today in "london" is sunny`,
						},
					},
					Role: genai.RoleModel,
				},
			},
		},
	}

	if diff := cmp.Diff(wantEvents, gotEvents,
		cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
		cmpopts.IgnoreFields(model.LLMResponse{}, "UsageMetadata", "AvgLogprobs", "FinishReason"),
		cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
		cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
		cmpopts.IgnoreFields(genai.Part{}, "ThoughtSignature")); diff != "" {
		t.Errorf("event[i] mismatch (-want +got):\n%s", diff)
	}
}

func newGeminiTestClientConfig(t *testing.T, rrfile string) (http.RoundTripper, bool) {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	recording, _ := httprr.Recording(rrfile)
	return rr, recording
}

func newGeminiModel(t *testing.T, modelName string) model.LLM {
	apiKey := "fakeKey"
	trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
	recording := false
	transport, recording := newGeminiTestClientConfig(t, trace)
	if recording { // if we are recording httprr trace, don't use the fakeKey.
		apiKey = ""
	}

	model, err := gemini.NewModel(t.Context(), modelName, &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: transport},
		APIKey:     apiKey,
	})
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	return model
}

func newTestAgentRunner(t *testing.T, agent agent.Agent) *testAgentRunner {
	appName := "test_app"
	sessionService := session.InMemoryService()

	runner, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          agent,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}

	return &testAgentRunner{
		agent:          agent,
		sessionService: sessionService,
		appName:        appName,
		runner:         runner,
	}
}

type testAgentRunner struct {
	agent          agent.Agent
	sessionService session.Service
	lastSession    session.Session
	appName        string
	// TODO: move runner definition to the adk package and it's a part of public api, but the logic to the internal runner
	runner *runner.Runner
}

func (r *testAgentRunner) session(t *testing.T, appName, userID, sessionID string) (session.Session, error) {
	ctx := t.Context()
	if last := r.lastSession; last != nil && last.ID() == sessionID {
		resp, err := r.sessionService.Get(ctx, &session.GetRequest{
			AppName:   "test_app",
			UserID:    "test_user",
			SessionID: sessionID,
		})
		r.lastSession = resp.Session
		return resp.Session, err
	}
	resp, err := r.sessionService.Create(ctx, &session.CreateRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: sessionID,
	})
	r.lastSession = resp.Session
	return resp.Session, err
}

func (r *testAgentRunner) Run(t *testing.T, sessionID, newMessage string) iter.Seq2[*session.Event, error] {
	t.Helper()
	ctx := t.Context()

	userID := "test_user"

	session, err := r.session(t, r.appName, userID, sessionID)
	if err != nil {
		t.Fatalf("failed to get/create session: %v", err)
	}

	var content *genai.Content
	if newMessage != "" {
		content = genai.NewContentFromText(newMessage, genai.RoleUser)
	}

	return r.runner.Run(ctx, userID, session.ID(), content, agent.RunConfig{})
}

func TestToolFilter(t *testing.T) {
	const toolDescription = "returns weather in the given city"

	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: toolDescription}, weatherFunc)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather1", Description: toolDescription}, weatherFunc)
	_, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport:  clientTransport,
		ToolFilter: tool.StringPredicate([]string{"get_weather"}),
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	tools, err := ts.Tools(icontext.NewReadonlyContext(
		icontext.NewInvocationContext(
			t.Context(),
			icontext.InvocationContextParams{},
		),
	))
	if err != nil {
		t.Fatalf("Failed to get tools: %v", err)
	}

	gotToolNames := make([]string, len(tools))
	for i, tool := range tools {
		gotToolNames[i] = tool.Name()
	}
	wantToolNames := []string{"get_weather"}

	if diff := cmp.Diff(wantToolNames, gotToolNames); diff != "" {
		t.Errorf("tools mismatch (-want +got):\n%s", diff)
	}
}

// TestSessionValidationOnReuse verifies that the MCP toolset validates
// cached sessions before reuse using Ping health check.
func TestSessionValidationOnReuse(t *testing.T) {
	const toolDescription = "returns weather in the given city"

	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	// Create server instance.
	server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: toolDescription}, weatherFunc)
	_, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: clientTransport,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	readonlyCtx := icontext.NewReadonlyContext(
		icontext.NewInvocationContext(
			t.Context(),
			icontext.InvocationContextParams{},
		),
	)

	// First call should establish a session and return tools.
	tools, err := ts.Tools(readonlyCtx)
	if err != nil {
		t.Fatalf("First Tools() call failed: %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "get_weather" {
		t.Fatalf("Expected 1 tool named 'get_weather', got %d tools", len(tools))
	}

	// Second call should reuse the cached session (validated via Ping).
	tools2, err := ts.Tools(readonlyCtx)
	if err != nil {
		t.Fatalf("Second Tools() call failed: %v", err)
	}
	if len(tools2) != 1 || tools2[0].Name() != "get_weather" {
		t.Fatalf("Expected 1 tool named 'get_weather', got %d tools", len(tools2))
	}
}

// reconnectableTransport wraps another transport and allows swapping it out
// to simulate reconnection scenarios.
type reconnectableTransport struct {
	mu        sync.Mutex
	transport mcp.Transport
}

func (r *reconnectableTransport) setTransport(t mcp.Transport) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.transport = t
}

func (r *reconnectableTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.transport.Connect(ctx)
}

// TestStaleSessionIsRecreated verifies that when a cached session becomes stale
// (e.g., server closes the connection), the MCP toolset detects this via Ping
// and automatically creates a new session.
func TestStaleSessionIsRecreated(t *testing.T) {
	const toolDescription = "returns weather in the given city"

	// Create first transport pair.
	clientTransport1, serverTransport1 := mcp.NewInMemoryTransports()

	// Create server instance.
	server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: toolDescription}, weatherFunc)

	// Capture the server-side session to be able to close it later.
	serverSession, err := server.Connect(t.Context(), serverTransport1, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Use a reconnectable transport that we can swap.
	reconnectable := &reconnectableTransport{transport: clientTransport1}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: reconnectable,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	readonlyCtx := icontext.NewReadonlyContext(
		icontext.NewInvocationContext(
			t.Context(),
			icontext.InvocationContextParams{},
		),
	)

	// First call should establish a session and return tools.
	tools, err := ts.Tools(readonlyCtx)
	if err != nil {
		t.Fatalf("First Tools() call failed: %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "get_weather" {
		t.Fatalf("Expected 1 tool named 'get_weather', got %d tools", len(tools))
	}

	// Simulate connection drop by closing the server-side session.
	if err := serverSession.Close(); err != nil {
		t.Fatalf("Failed to close server session: %v", err)
	}

	// Create a new transport pair for reconnection.
	clientTransport2, serverTransport2 := mcp.NewInMemoryTransports()
	reconnectable.setTransport(clientTransport2)

	// Create a new server session for the new transport.
	_, err = server.Connect(t.Context(), serverTransport2, nil)
	if err != nil {
		t.Fatalf("Failed to create new server session: %v", err)
	}

	// Second call should detect the stale session via Ping, reconnect, and succeed.
	tools2, err := ts.Tools(readonlyCtx)
	if err != nil {
		t.Fatalf("Second Tools() call failed after session drop: %v", err)
	}
	if len(tools2) != 1 || tools2[0].Name() != "get_weather" {
		t.Fatalf("Expected 1 tool named 'get_weather' after reconnect, got %d tools", len(tools2))
	}
}
