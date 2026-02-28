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
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
	"google.golang.org/adk/tool/toolconfirmation"
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
			Author: "weather_time_agent",
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
			},
		},
		{
			Author: "weather_time_agent",
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
	toolCtx := toolinternal.NewToolContext(invCtx, "", nil, nil)

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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
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
					"error": errors.New("error tool \"get_weather\" requires confirmation, please approve or reject"),
				}, "user"),
				genai.NewContentFromFunctionResponse(toolName, map[string]any{
					"error": errors.New("error tool \"get_weather\" call is rejected"),
				}, "user"),
				genai.NewContentFromText("I am sorry, I cannot get the weather in Lisbon for you. The tool is not working at the moment.", "model"),
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
		provider mcptoolset.ConfirmationProvider // The provider to test
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

func TestMetadataProvider(t *testing.T) {
	var receivedMeta map[string]any
	var toolCalled bool
	echoToolFunc := func(ctx context.Context, req *mcp.CallToolRequest, input struct{}) (*mcp.CallToolResult, struct{ Message string }, error) {
		toolCalled = true
		receivedMeta = req.Params.Meta
		return nil, struct{ Message string }{Message: "ok"}, nil
	}

	testMetadata := map[string]any{
		"request_id":  "test-123",
		"user_id":     "user-456",
		"nested_data": map[string]any{"key": "value"},
	}
	metadataProvider := func(ctx tool.Context) map[string]any {
		return testMetadata
	}

	result := runMetadataTest(t, metadataProvider, echoToolFunc)
	if result == nil {
		t.Fatal("Expected non-nil result")
	}

	if !toolCalled {
		t.Fatal("Tool was not called")
	}

	if diff := cmp.Diff(testMetadata, receivedMeta); diff != "" {
		t.Errorf("metadata mismatch (-want +got):\n%s", diff)
	}
}

func TestMetadataProviderNil(t *testing.T) {
	var toolCalled bool
	echoToolFunc := func(ctx context.Context, req *mcp.CallToolRequest, input struct{}) (*mcp.CallToolResult, struct{ Message string }, error) {
		toolCalled = true
		if req.Params.Meta != nil {
			t.Errorf("Expected nil metadata, got %v", req.Params.Meta)
		}
		return nil, struct{ Message string }{Message: "ok"}, nil
	}
	_ = runMetadataTest(t, nil, echoToolFunc)
	if !toolCalled {
		t.Fatal("Tool was not called")
	}
}

func TestMetadataProviderReturnsNil(t *testing.T) {
	var receivedMeta map[string]any
	var toolCalled bool

	echoToolFunc := func(ctx context.Context, req *mcp.CallToolRequest, input struct{}) (*mcp.CallToolResult, struct{ Message string }, error) {
		toolCalled = true
		receivedMeta = req.Params.Meta
		return nil, struct{ Message string }{Message: "ok"}, nil
	}

	metadataProvider := func(ctx tool.Context) map[string]any {
		return nil
	}

	_ = runMetadataTest(t, metadataProvider, echoToolFunc)

	if !toolCalled {
		t.Fatal("Tool was not called")
	}

	if receivedMeta != nil {
		t.Errorf("Expected nil metadata when provider returns nil, got %v", receivedMeta)
	}
}

func runMetadataTest[In, Out any](t *testing.T, provider mcptoolset.MetadataProvider, toolFunc mcp.ToolHandlerFor[In, Out]) map[string]any {
	t.Helper()

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "echo_tool", Description: "echoes input"}, toolFunc)
	_, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}

	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport:        clientTransport,
		MetadataProvider: provider,
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	readonlyCtx := icontext.NewReadonlyContext(invCtx)
	tools, err := ts.Tools(readonlyCtx)
	if err != nil {
		t.Fatalf("Failed to get tools: %v", err)
	}

	if len(tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools))
	}

	fnTool, ok := tools[0].(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("Tool does not implement FunctionTool interface")
	}

	toolCtx := toolinternal.NewToolContext(invCtx, "", nil, nil)
	result, err := fnTool.Run(toolCtx, map[string]any{})
	if err != nil {
		t.Fatalf("Failed to run tool: %v", err)
	}
	return result
}
