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
	"errors"
	"fmt"
	"io"
	"log"
	"maps"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/httprr"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
	"google.golang.org/adk/v2/tool/toolconfirmation"
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

//go:generate go test -v -httprecord=.*

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
	runner := testutil.NewTestAgentRunner(t, agent)

	var gotEvents []*session.Event
	for event, err := range runner.Run(t, "session1", prompt) {
		if err != nil {
			t.Fatal(err)
		}
		gotEvents = append(gotEvents, event)
	}

	wantEvents := []*session.Event{
		{
			Author:   "weather_time_agent",
			NodeInfo: &session.NodeInfo{Path: "weather_time_agent"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: toolName,
								Args: map[string]any{"city": "london"},
							},
						},
					},
					Role: genai.RoleModel,
				},
				ModelVersion: "gemini-2.5-flash",
			},
		},
		{
			Author:   "weather_time_agent",
			NodeInfo: &session.NodeInfo{Path: "weather_time_agent"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name: toolName,
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
			Author:   "weather_time_agent",
			NodeInfo: &session.NodeInfo{Path: "weather_time_agent"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							Text: `The weather in London is sunny.`,
						},
					},
					Role: genai.RoleModel,
				},
				ModelVersion: "gemini-2.5-flash",
			},
		},
	}

	if diff := cmp.Diff(wantEvents, gotEvents,
		cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta", "ArtifactDelta"),
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

	// Ensure the transport is closed to flush data and release locks
	if c, ok := rr.(io.Closer); ok {
		t.Cleanup(func() {
			if err := c.Close(); err != nil {
				t.Errorf("failed to close transport: %v", err)
			}
		})
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

func TestMCPToolSetClose(t *testing.T) {
	for _, connected := range []bool{false, true} {
		t.Run(fmt.Sprintf("connected=%t", connected), func(t *testing.T) {
			server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "1"}, nil)
			mcp.AddTool(server, &mcp.Tool{Name: "get_weather"}, weatherFunc)
			transport := &spyTransport{Transport: &reconnectableTransport{server: server}}
			ts, err := mcptoolset.New(mcptoolset.Config{Transport: transport})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				for session := range server.Sessions() {
					_ = session.Close()
				}
			})
			closer, ok := ts.(io.Closer)
			if !ok {
				t.Fatal("MCP toolset does not implement io.Closer")
			}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			ctx := icontext.NewReadonlyContext(invCtx)
			var tools []tool.Tool
			var sessionClosed <-chan struct{}
			if connected {
				tools, err = ts.Tools(ctx)
				if err != nil {
					t.Fatal(err)
				}
				for session := range server.Sessions() {
					done := make(chan struct{})
					sessionClosed = done
					go func() {
						_ = session.Wait()
						close(done)
					}()
				}
			}
			var wg sync.WaitGroup
			for range 2 {
				wg.Go(func() {
					if err := closer.Close(); err != nil {
						t.Errorf("Close() failed: %v", err)
					}
				})
			}
			wg.Wait()
			if connected {
				select {
				case <-sessionClosed:
				case <-time.After(5 * time.Second):
					t.Fatal("Close() left the MCP session open")
				}
				fnTool := tools[0].(toolinternal.FunctionTool)
				if _, err := fnTool.Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{"city": "Paris"}); !errors.Is(err, mcp.ErrConnectionClosed) {
					t.Errorf("Run() after Close() error = %v, want ErrConnectionClosed", err)
				}
			}
			if _, err := ts.Tools(ctx); !errors.Is(err, mcp.ErrConnectionClosed) {
				t.Errorf("Tools() after Close() error = %v, want ErrConnectionClosed", err)
			}
			wantConnections := 0
			if connected {
				wantConnections = 1
			}
			if transport.connectCount != wantConnections {
				t.Errorf("Connect() called %d times, want %d", transport.connectCount, wantConnections)
			}
		})
	}
}

func TestListToolsReconnection(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: "returns weather in the given city"}, weatherFunc)

	rt := &reconnectableTransport{server: server}
	spyTransport := &spyTransport{Transport: rt}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: spyTransport,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))

	// First call to Tools should create a session.
	_, err = ts.Tools(ctx)
	if err != nil {
		t.Fatalf("First Tools call failed: %v", err)
	}

	// Kill the transport by closing the connection.
	if err := spyTransport.lastConn.Close(); err != nil {
		t.Fatalf("Failed to close connection: %v", err)
	}

	// Second call should detect the closed connection and reconnect.
	_, err = ts.Tools(ctx)
	if err != nil {
		t.Fatalf("Second Tools call failed: %v", err)
	}

	// Verify that we reconnected (should have 2 connections).
	if spyTransport.connectCount != 2 {
		t.Errorf("Expected 2 Connect calls (reconnect after close), got %d", spyTransport.connectCount)
	}
}

func TestCallToolReconnection(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "get_weather", Description: "returns weather in the given city"}, weatherFunc)

	rt := &reconnectableTransport{server: server}
	spyTransport := &spyTransport{Transport: rt}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: spyTransport,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	ctx := icontext.NewReadonlyContext(invCtx)
	toolCtx := agent.NewToolContext(invCtx, "", nil, nil)

	// Get tools first to establish a session.
	tools, err := ts.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools call failed: %v", err)
	}

	// Kill the transport by closing the connection.
	if err := spyTransport.lastConn.Close(); err != nil {
		t.Fatalf("Failed to close connection: %v", err)
	}

	// Call the tool - should reconnect and succeed.
	fnTool := tools[0].(toolinternal.FunctionTool)
	result, err := fnTool.Run(toolCtx, map[string]any{"city": "Paris"})
	if err != nil {
		t.Fatalf("Tool call after reconnect failed: %v", err)
	}
	if result == nil {
		t.Fatal("Expected non-nil result after reconnect")
	}

	// Verify that we reconnected (should have 2 connections).
	if spyTransport.connectCount != 2 {
		t.Errorf("Expected 2 Connect calls (reconnect after close), got %d", spyTransport.connectCount)
	}
}

type spyTransport struct {
	mcp.Transport
	connectCount int
	lastConn     mcp.Connection
}

func (t *spyTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	t.connectCount++
	conn, err := t.Transport.Connect(ctx)
	t.lastConn = conn
	return conn, err
}

type reconnectableTransport struct {
	server *mcp.Server
}

func (rt *reconnectableTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	ct, st := mcp.NewInMemoryTransports()
	_, err := rt.server.Connect(ctx, st, nil)
	if err != nil {
		return nil, err
	}
	return ct.Connect(ctx)
}

func TestMCPToolSetConfirmation(t *testing.T) {
	const (
		toolName        = "get_weather"
		toolDescription = "returns weather in the given city"
	)

	requireConfirmationProvider := func(name string, args any) bool {
		if name != toolName {
			return false
		}

		if input, ok := args.(Input); ok {
			return input.City == "Lisbon"
		}

		if m, ok := args.(map[string]any); ok {
			if cityVal, found := m["city"]; found {
				if cityStr, isStr := cityVal.(string); isStr {
					return cityStr == "Lisbon"
				}
			}
		}

		return true
	}

	testCases := []struct {
		name                    string
		toolSetConfig           mcptoolset.Config
		city                    string
		confirmFunctionResponse *genai.FunctionResponse // User's confirmation response
		want                    []*genai.Content
	}{
		{
			name:          "No Confirmation Required",
			toolSetConfig: mcptoolset.Config{},
			city:          "Lisbon",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"output": map[string]any{"weather_summary": string(`Today in "Lisbon" is sunny`)},
				}, "user"),
				genai.NewContentFromText(`Today in "Lisbon" is sunny`, "model"),
			},
		},
		{
			name: "Confirmation Required",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmation: true,
			},
			city: "Lisbon",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
			},
		},
		{
			name: "Confirmation Required and is confirmed",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmation: true,
			},
			city:                    "Lisbon",
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": true}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"output": map[string]any{"weather_summary": string(`Today in "Lisbon" is sunny`)},
				}, "user"),
				genai.NewContentFromText(`Today in "Lisbon" is sunny`, "model"),
			},
		},
		{
			name: "Confirmation Required and is rejected",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmation: true,
			},
			city:                    "Lisbon",
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": false}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" call is rejected"),
				}, "user"),
				genai.NewContentFromText("I am sorry, I cannot get the weather in Lisbon.", "model"),
			},
		},
		{
			name: "Conditional Confirmation Not Required",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: requireConfirmationProvider,
			},
			city: "Porto",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Porto"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"output": map[string]any{"weather_summary": string(`Today in "Porto" is sunny`)},
				}, "user"),
				genai.NewContentFromText(`Today in "Porto" is sunny`, "model"),
			},
		},
		{
			name: "Conditional Confirmation Not Required For This Tool",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: func(name string, args any) bool {
					return name != toolName
				},
			},
			city: "Lisbon",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"output": map[string]any{"weather_summary": string(`Today in "Lisbon" is sunny`)},
				}, "user"),
				genai.NewContentFromText(`Today in "Lisbon" is sunny`, "model"),
			},
		},
		{
			name: "Conditional Confirmation Required For This Tool",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: func(name string, args any) bool {
					return name == toolName
				},
			},
			city: "Lisbon",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
			},
		},
		{
			name: "Conditional Confirmation Required",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: requireConfirmationProvider,
			},
			city: "Lisbon",
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
			},
		},
		{
			name: "Conditional Confirmation Required and is confirmed",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: requireConfirmationProvider,
			},
			city:                    "Lisbon",
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": true}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"output": map[string]any{"weather_summary": string(`Today in "Lisbon" is sunny`)},
				}, "user"),
				genai.NewContentFromText(`Today in "Lisbon" is sunny`, "model"),
			},
		},
		{
			name: "Conditional Confirmation Required and is rejected",
			toolSetConfig: mcptoolset.Config{
				RequireConfirmationProvider: requireConfirmationProvider,
			},
			city:                    "Lisbon",
			confirmFunctionResponse: &genai.FunctionResponse{Name: toolconfirmation.FunctionCallName, Response: map[string]any{"confirmed": false}},
			want: []*genai.Content{
				genai.NewContentFromFunctionCall(toolName, map[string]any{"city": "Lisbon"}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionCall(toolconfirmation.FunctionCallName, map[string]any{
					"originalFunctionCall": &genai.FunctionCall{
						Args: map[string]any{"city": "Lisbon"},
						Name: toolName,
					},
					"toolConfirmation": toolconfirmation.ToolConfirmation{
						Hint: "Please approve or reject the tool call get_weather() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
					},
				}, "model"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" call is rejected"),
				}, "user"),
				genai.NewContentFromText("I am sorry, I cannot get the weather in Lisbon.", "model"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clientTransport, serverTransport := mcp.NewInMemoryTransports()

			// Run in-memory MCP server.
			server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
			mcp.AddTool(server, &mcp.Tool{Name: toolName, Description: toolDescription}, weatherFunc)
			_, err := server.Connect(t.Context(), serverTransport, nil)
			if err != nil {
				t.Fatal(err)
			}

			tc.toolSetConfig.Transport = clientTransport
			ts, err := mcptoolset.New(tc.toolSetConfig)
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

			prompt := fmt.Sprintf("what is the weather in %s?", tc.city)
			runner := testutil.NewTestAgentRunner(t, agent)

			ev := runner.Run(t, "session1", prompt)

			comptsList := []cmp.Option{
				cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
				cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
				cmpopts.IgnoreFields(model.LLMResponse{}, "UsageMetadata", "AvgLogprobs", "FinishReason"),
				cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
				cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
				cmpopts.IgnoreFields(genai.Part{}, "ThoughtSignature"),
				cmp.Transformer("StringifyMapErrors", func(m map[string]any) map[string]any {
					out := make(map[string]any, len(m))
					for k, v := range m {
						// Check if the value inside the map is an error
						if err, ok := v.(error); ok {
							out[k] = err.Error() // Convert to string
						} else {
							out[k] = v // Keep as is
						}
					}
					return out
				}),
			}

			eventCount := 0
			var confirmFunctionCall *genai.FunctionCall
			for got, err := range ev {
				if err != nil && err.Error() == "no data" {
					break
				}
				if err != nil {
					// Check if an error was expected
					t.Fatalf("runner returned unexpected error: %v", err)
					// If error was expected, we can stop here or check for a specific error type.
					return
				}

				if eventCount >= len(tc.want) {
					t.Fatalf("stream generated more values than the expected %d. Got: %+v", len(tc.want), got.Content)
				}

				if diff := cmp.Diff(tc.want[eventCount], got.Content, comptsList...); diff != "" {
					t.Errorf("LoopAgent Run() mismatch (-want +got):\n%s", diff)
				}
				for _, p := range got.Content.Parts {
					if p.FunctionCall != nil && p.FunctionCall.Name == toolconfirmation.FunctionCallName {
						confirmFunctionCall = p.FunctionCall
					}
				}
				eventCount++
			}

			if confirmFunctionCall != nil && tc.confirmFunctionResponse != nil {
				tc.confirmFunctionResponse.ID = confirmFunctionCall.ID
				ev := runner.RunContent(t, "session1", &genai.Content{
					Parts: []*genai.Part{{FunctionResponse: tc.confirmFunctionResponse}},
				})
				for got, err := range ev {
					if err != nil && err.Error() == "no data" {
						break
					}
					if err != nil {
						// Check if an error was expected
						t.Fatalf("runner returned unexpected error: %v", err)
						// If error was expected, we can stop here or check for a specific error type.
						return
					}

					if eventCount >= len(tc.want) {
						t.Fatalf("stream generated more values than the expected %d. Got: %+v", len(tc.want), got.Content)
					}

					if diff := cmp.Diff(tc.want[eventCount], got.Content, comptsList...); diff != "" {
						t.Errorf("LoopAgent Run() mismatch (-want +got):\n%s", diff)
					}
					for _, p := range got.Content.Parts {
						if p.FunctionCall != nil && p.FunctionCall.Name == toolconfirmation.FunctionCallName {
							confirmFunctionCall = p.FunctionCall
						}
					}
					eventCount++
				}
			}

			// Final check on the number of events
			if eventCount != len(tc.want) {
				t.Errorf("unexpected stream length, want %d got %d", len(tc.want), eventCount)
			}
		})
	}
}

// Mock types for TArgs and TResults
type TestArgs struct {
	Name string
}

type TestResult struct {
	Value int
}

func TestNewToolSet_RequireConfirmationProvider_Validation(t *testing.T) {
	tests := []struct {
		name     string
		provider tool.ConfirmationProvider // The provider to test
	}{
		// --- Happy Paths ---
		{
			name:     "Valid: Nil provider is allowed",
			provider: nil,
		},
		{
			name:     "Valid: Correct function signature",
			provider: func(name string, args any) bool { return true },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Construct config with the provider under test
			clientTransport, serverTransport := mcp.NewInMemoryTransports()

			// Run in-memory MCP server.
			server := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "v1.0.0"}, nil)
			mcp.AddTool(server, &mcp.Tool{Name: "test", Description: "test"}, weatherFunc)
			_, err := server.Connect(t.Context(), serverTransport, nil)
			if err != nil {
				t.Fatal(err)
			}

			toolSetConfig := mcptoolset.Config{
				Transport:                   clientTransport,
				RequireConfirmationProvider: tt.provider,
			}
			toolset, err := mcptoolset.New(toolSetConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if toolset == nil {
				t.Error("expected valid toolset, got nil")
			}
		})
	}
}

func TestMCPTool_EmptyTextResponse(t *testing.T) {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "empty_tool", Description: "returns empty response"}, func(ctx context.Context, req *mcp.CallToolRequest, args any) (*mcp.CallToolResult, any, error) {
		return &mcp.CallToolResult{
			Content: []mcp.Content{&mcp.TextContent{Text: ""}},
		}, nil, nil
	})
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

	tools, err := ts.Tools(icontext.NewReadonlyContext(
		icontext.NewInvocationContext(
			t.Context(),
			icontext.InvocationContextParams{},
		),
	))
	if err != nil {
		t.Fatalf("Failed to get tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	toolCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	tc := agent.NewToolContext(toolCtx, "", nil, nil)

	fnTool, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("Expected tool to implement toolinternal.FunctionTool")
	}

	res, err := fnTool.Run(tc, map[string]any{})
	if err != nil {
		t.Fatalf("Expected Run to succeed on empty text response, got: %v", err)
	}
	if res["output"] != "" {
		t.Fatalf("Expected output to be empty string, got: %v", res["output"])
	}
}

func TestMCPTool_NonTextContentRoundTrip(t *testing.T) {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	resourceSize := int64(42)

	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "content_tool", Description: "returns non-text content"}, func(ctx context.Context, req *mcp.CallToolRequest, args any) (*mcp.CallToolResult, any, error) {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{
					URI:      "repo://owner/project/main/file.go",
					MIMEType: "text/plain",
					Blob:     []byte("package example\n"),
				}},
				&mcp.ResourceLink{
					URI:      "https://example.com/report.pdf",
					Name:     "report.pdf",
					MIMEType: "application/pdf",
					Size:     &resourceSize,
				},
				&mcp.ImageContent{MIMEType: "image/png", Data: []byte{1, 2, 3, 4}},
				&mcp.AudioContent{MIMEType: "audio/wav", Data: []byte{1, 2, 3}},
			},
		}, nil, nil
	})
	_, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{Transport: clientTransport})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	tools, err := ts.Tools(icontext.NewReadonlyContext(
		icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}),
	))
	if err != nil {
		t.Fatalf("Failed to get tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	fnTool, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("Expected tool to implement toolinternal.FunctionTool")
	}
	toolCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	got, err := fnTool.Run(agent.NewToolContext(toolCtx, "", nil, nil), map[string]any{})
	if err != nil {
		t.Fatalf("Run() failed: %v", err)
	}

	want := map[string]any{"output": "[MCP embedded resource: " +
		"uri=\"repo://owner/project/main/file.go\", mimeType=\"text/plain\"]\n" +
		"package example\n" +
		"[MCP resource link: uri=\"https://example.com/report.pdf\", mimeType=\"application/pdf\", " +
		"name=\"report.pdf\", size=42 bytes]\n" +
		"[MCP image: mimeType=\"image/png\", size=4 bytes]\n" +
		"[MCP audio: mimeType=\"audio/wav\", size=3 bytes]"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
}

// metaCapture records the `_meta` an MCP server received on tool calls.
type metaCapture struct {
	called bool
	meta   map[string]any
	metas  []map[string]any
}

// mcpClientMetaKeys are the `_meta` entries the MCP client puts on every
// outgoing request. They are not what a MetadataProvider contributed, so the
// tests below filter them out before comparing.
var mcpClientMetaKeys = []string{
	mcp.MetaKeyProtocolVersion,
	mcp.MetaKeyClientInfo,
	mcp.MetaKeyClientCapabilities,
}

// providerMeta returns the received `_meta` without the client's own entries.
func providerMeta(meta map[string]any) map[string]any {
	got := maps.Clone(meta)
	for _, k := range mcpClientMetaKeys {
		delete(got, k)
	}
	return got
}

// startMetaEchoServer runs an in-memory MCP server with a single tool that
// records the `_meta` of each call, and returns a toolset wired to it.
func startMetaEchoServer(t *testing.T, provider mcptoolset.MetadataProvider) (tool.Toolset, *metaCapture) {
	t.Helper()
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	got := &metaCapture{}
	server := mcp.NewServer(&mcp.Implementation{Name: "meta_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server,
		&mcp.Tool{Name: "get_weather", Description: "returns weather in the given city"},
		func(ctx context.Context, req *mcp.CallToolRequest, input Input) (*mcp.CallToolResult, Output, error) {
			got.called = true
			got.meta = req.Params.Meta
			got.metas = append(got.metas, req.Params.Meta)
			return nil, Output{WeatherSummary: "sunny"}, nil
		})
	if _, err := server.Connect(t.Context(), serverTransport, nil); err != nil {
		t.Fatalf("server.Connect() err = %v", err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport:        clientTransport,
		MetadataProvider: provider,
	})
	if err != nil {
		t.Fatalf("mcptoolset.New() err = %v", err)
	}
	return ts, got
}

// runSingleTool invokes the toolset's only tool with a fixed argument.
func runSingleTool(t *testing.T, ctx context.Context, ts tool.Toolset) (agent.InvocationContext, error) {
	t.Helper()
	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{})
	tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
	if err != nil {
		t.Fatalf("Tools() err = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Tools() returned %d tools, want 1", len(tools))
	}
	fnTool, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("tool is %T, want toolinternal.FunctionTool", tools[0])
	}
	_, err = fnTool.Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{"city": "Paris"})
	return invCtx, err
}

func TestMetadataProvider(t *testing.T) {
	tests := []struct {
		name     string
		provider mcptoolset.MetadataProvider
		wantMeta map[string]any
	}{
		{
			name: "provider metadata reaches the server",
			provider: func(ctx agent.Context) (map[string]any, error) {
				return map[string]any{"trace_id": "abc-123", "tenant": "acme"}, nil
			},
			wantMeta: map[string]any{"trace_id": "abc-123", "tenant": "acme"},
		},
		{
			name:     "no provider configured sends no metadata",
			provider: nil,
			wantMeta: nil,
		},
		{
			name: "provider returning nil sends no metadata",
			provider: func(ctx agent.Context) (map[string]any, error) {
				return nil, nil
			},
			wantMeta: nil,
		},
		{
			name: "provider returning an empty map sends no metadata",
			provider: func(ctx agent.Context) (map[string]any, error) {
				return map[string]any{}, nil
			},
			wantMeta: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ts, got := startMetaEchoServer(t, tc.provider)
			if _, err := runSingleTool(t, t.Context(), ts); err != nil {
				t.Fatalf("Run() err = %v, want nil", err)
			}
			if !got.called {
				t.Fatal("the MCP tool was never invoked")
			}
			if diff := cmp.Diff(tc.wantMeta, providerMeta(got.meta), cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("server-side _meta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestMetadataProviderError checks that a provider failure aborts the call
// rather than silently sending the tool request without its metadata, which
// would defeat the point for auth- or tenant-scoped metadata.
func TestMetadataProviderError(t *testing.T) {
	wantErr := errors.New("no trace id in context")
	ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
		return nil, wantErr
	})

	_, err := runSingleTool(t, t.Context(), ts)
	if err == nil {
		t.Fatal("Run() err = nil, want the provider error")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("Run() err = %v, want it to wrap %v", err, wantErr)
	}
	if !strings.Contains(err.Error(), "get_weather") {
		t.Errorf("Run() err = %q, want it to name the tool", err)
	}
	if got.called {
		t.Error("the MCP tool ran despite the metadata provider failing")
	}
}

func TestMetadataProviderRejectsReservedKeys(t *testing.T) {
	tests := []struct {
		key string
		val any
	}{
		{key: "progressToken", val: "spoofed"},
		{key: mcp.MetaKeyProtocolVersion, val: "2025-06-18"},
		{key: mcp.MetaKeyClientInfo, val: map[string]any{"name": "spoofed", "version": "0.0.1"}},
		{key: mcp.MetaKeyClientCapabilities, val: map[string]any{}},
		{key: "io.modelcontextprotocol/custom", val: "spoofed"},
	}

	for _, tc := range tests {
		t.Run(tc.key, func(t *testing.T) {
			ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
				return map[string]any{
					"trace_id": "abc-123",
					tc.key:     tc.val,
				}, nil
			})

			_, err := runSingleTool(t, t.Context(), ts)
			if err == nil {
				t.Fatalf("Run() err = nil, want error for reserved key %q", tc.key)
			}
			if !strings.Contains(err.Error(), tc.key) {
				t.Errorf("Run() err = %q, want it to name reserved key %q", err, tc.key)
			}
			if !strings.Contains(err.Error(), "get_weather") {
				t.Errorf("Run() err = %q, want it to name the tool", err)
			}
			if got.called {
				t.Error("the MCP tool ran despite the metadata provider returning a reserved key")
			}
		})
	}
}

func TestMetadataProviderAllowsNonReservedKeys(t *testing.T) {
	tests := []struct {
		key string
		val any
	}{
		{key: "com.example/tenant", val: "acme"},
		{key: "io.modelcontextprotocolx", val: "custom"},
		{key: "ProgressToken", val: "tok-1"},
	}

	for _, tc := range tests {
		t.Run(tc.key, func(t *testing.T) {
			wantMeta := map[string]any{tc.key: tc.val}
			ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
				return wantMeta, nil
			})

			if _, err := runSingleTool(t, t.Context(), ts); err != nil {
				t.Fatalf("Run() err = %v, want nil", err)
			}
			if !got.called {
				t.Fatal("the MCP tool was never invoked")
			}
			if diff := cmp.Diff(wantMeta, providerMeta(got.meta)); diff != "" {
				t.Errorf("server-side _meta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestMetadataProviderKeepsClientMeta checks that the server receives both the
// provider's metadata and the `_meta` entries the MCP client sets for itself,
// including the adk-mcp-client implementation info.
func TestMetadataProviderKeepsClientMeta(t *testing.T) {
	ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
		return map[string]any{"trace_id": "abc-123"}, nil
	})
	if _, err := runSingleTool(t, t.Context(), ts); err != nil {
		t.Fatalf("Run() err = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"trace_id": "abc-123"}, providerMeta(got.meta)); diff != "" {
		t.Errorf("provider _meta mismatch (-want +got):\n%s", diff)
	}
	for _, key := range mcpClientMetaKeys {
		if _, ok := got.meta[key]; !ok {
			t.Errorf("received _meta is missing the client's own %q, got keys %v", key, slices.Sorted(maps.Keys(got.meta)))
		}
	}
	wantClientInfo := map[string]any{
		"name":    "adk-mcp-client",
		"version": version.Version,
	}
	if diff := cmp.Diff(wantClientInfo, got.meta[mcp.MetaKeyClientInfo]); diff != "" {
		t.Errorf("received _meta[%q] mismatch (-want +got):\n%s", mcp.MetaKeyClientInfo, diff)
	}
}

// TestMetadataProviderMapIsCopied checks the map a provider returns is not
// written to, so a provider may hand back one it reuses between invocations.
func TestMetadataProviderMapIsCopied(t *testing.T) {
	reused := map[string]any{"trace_id": "abc-123"}
	ts, _ := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
		return reused, nil
	})
	if _, err := runSingleTool(t, t.Context(), ts); err != nil {
		t.Fatalf("Run() err = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"trace_id": "abc-123"}, reused); diff != "" {
		t.Errorf("the provider's own map was modified (-want +got):\n%s", diff)
	}
}

func TestMetadataProviderPerCall(t *testing.T) {
	calls := 0
	ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
		calls++
		return map[string]any{"trace_id": fmt.Sprintf("call-%d", calls)}, nil
	})

	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
	if err != nil {
		t.Fatalf("Tools() err = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Tools() returned %d tools, want 1", len(tools))
	}
	fnTool, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("tool is %T, want toolinternal.FunctionTool", tools[0])
	}

	for i := 1; i <= 2; i++ {
		if _, err := fnTool.Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{"city": "Paris"}); err != nil {
			t.Fatalf("Run() call %d err = %v", i, err)
		}
	}

	if calls != 2 {
		t.Errorf("provider called %d times, want 2", calls)
	}
	if len(got.metas) != 2 {
		t.Fatalf("server received %d calls, want 2", len(got.metas))
	}
	gotMetas := []map[string]any{
		providerMeta(got.metas[0]),
		providerMeta(got.metas[1]),
	}
	wantMetas := []map[string]any{
		{"trace_id": "call-1"},
		{"trace_id": "call-2"},
	}
	if diff := cmp.Diff(wantMetas, gotMetas); diff != "" {
		t.Errorf("server-side _meta across calls mismatch (-want +got):\n%s", diff)
	}
}

// TestMetadataProviderReceivesToolContext checks the provider is handed the
// live invocation's context, including values set on the request context.
func TestMetadataProviderReceivesToolContext(t *testing.T) {
	type traceKey struct{}
	var gotInvocationID string
	ts, got := startMetaEchoServer(t, func(ctx agent.Context) (map[string]any, error) {
		gotInvocationID = ctx.InvocationID()
		traceID, _ := ctx.Value(traceKey{}).(string)
		return map[string]any{"trace_id": traceID}, nil
	})
	reqCtx := context.WithValue(t.Context(), traceKey{}, "abc-123")
	invCtx, err := runSingleTool(t, reqCtx, ts)
	if err != nil {
		t.Fatalf("Run() err = %v", err)
	}
	if gotInvocationID != invCtx.InvocationID() {
		t.Errorf("provider got invocation ID %q, want %q", gotInvocationID, invCtx.InvocationID())
	}
	wantMeta := map[string]any{"trace_id": "abc-123"}
	if diff := cmp.Diff(wantMeta, providerMeta(got.meta)); diff != "" {
		t.Errorf("server-side _meta mismatch (-want +got):\n%s", diff)
	}
}

func TestMetadataProviderConfirmation(t *testing.T) {
	tests := []struct {
		name               string
		confirmation       *toolconfirmation.ToolConfirmation
		providerErr        error
		wantErr            error
		wantProviderCalled bool
		wantServerCalled   bool
		wantMeta           map[string]any
	}{
		{
			name:               "no confirmation",
			confirmation:       nil,
			providerErr:        errors.New("provider ran before confirmation"),
			wantErr:            tool.ErrConfirmationRequired,
			wantProviderCalled: false,
			wantServerCalled:   false,
		},
		{
			name:               "rejected confirmation",
			confirmation:       &toolconfirmation.ToolConfirmation{Confirmed: false},
			providerErr:        errors.New("provider ran on rejected confirmation"),
			wantErr:            tool.ErrConfirmationRejected,
			wantProviderCalled: false,
			wantServerCalled:   false,
		},
		{
			name:               "approved confirmation",
			confirmation:       &toolconfirmation.ToolConfirmation{Confirmed: true},
			wantErr:            nil,
			wantProviderCalled: true,
			wantServerCalled:   true,
			wantMeta:           map[string]any{"trace_id": "abc-123"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clientTransport, serverTransport := mcp.NewInMemoryTransports()
			got := &metaCapture{}
			server := mcp.NewServer(&mcp.Implementation{Name: "meta_server", Version: "v1.0.0"}, nil)
			mcp.AddTool(server,
				&mcp.Tool{Name: "get_weather", Description: "returns weather in the given city"},
				func(ctx context.Context, req *mcp.CallToolRequest, input Input) (*mcp.CallToolResult, Output, error) {
					got.called = true
					got.meta = req.Params.Meta
					return nil, Output{WeatherSummary: "sunny"}, nil
				})
			if _, err := server.Connect(t.Context(), serverTransport, nil); err != nil {
				t.Fatalf("server.Connect() err = %v", err)
			}

			providerCalled := false
			ts, err := mcptoolset.New(mcptoolset.Config{
				Transport:           clientTransport,
				RequireConfirmation: true,
				MetadataProvider: func(ctx agent.Context) (map[string]any, error) {
					providerCalled = true
					if tc.providerErr != nil {
						return nil, tc.providerErr
					}
					return map[string]any{"trace_id": "abc-123"}, nil
				},
			})
			if err != nil {
				t.Fatalf("mcptoolset.New() err = %v", err)
			}

			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
			if err != nil {
				t.Fatalf("Tools() err = %v", err)
			}
			if len(tools) != 1 {
				t.Fatalf("Tools() returned %d tools, want 1", len(tools))
			}
			fnTool, ok := tools[0].(toolinternal.FunctionTool)
			if !ok {
				t.Fatalf("tool is %T, want toolinternal.FunctionTool", tools[0])
			}

			_, err = fnTool.Run(agent.NewToolContext(invCtx, "", nil, tc.confirmation), map[string]any{"city": "Paris"})
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("Run() err = %v, want %v", err, tc.wantErr)
			}
			if providerCalled != tc.wantProviderCalled {
				t.Errorf("providerCalled = %v, want %v", providerCalled, tc.wantProviderCalled)
			}
			if got.called != tc.wantServerCalled {
				t.Errorf("server called = %v, want %v", got.called, tc.wantServerCalled)
			}
			if tc.wantServerCalled {
				if diff := cmp.Diff(tc.wantMeta, providerMeta(got.meta)); diff != "" {
					t.Errorf("server-side _meta mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
