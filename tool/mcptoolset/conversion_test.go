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

package mcptoolset

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type stubToolWithoutDecl struct {
	name        string
	description string
}

func (s *stubToolWithoutDecl) Name() string        { return s.name }
func (s *stubToolWithoutDecl) Description() string { return s.description }
func (s *stubToolWithoutDecl) IsLongRunning() bool { return false }

type stubToolWithDecl struct {
	name        string
	description string
	decl        *genai.FunctionDeclaration
}

func (s *stubToolWithDecl) Name() string                            { return s.name }
func (s *stubToolWithDecl) Description() string                     { return s.description }
func (s *stubToolWithDecl) IsLongRunning() bool                     { return false }
func (s *stubToolWithDecl) Declaration() *genai.FunctionDeclaration { return s.decl }

type weatherInput struct {
	City  string `json:"city" jsonschema:"The city name"`
	Units string `json:"units,omitempty" jsonschema:"Temperature units"`
}

type weatherOutput struct {
	Summary string `json:"summary"`
}

func newWeatherFunctionTool(t *testing.T) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Gets weather information",
	}, func(_ agent.Context, in weatherInput) (weatherOutput, error) {
		return weatherOutput{Summary: in.City}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return ft
}

func newPointerWeatherFunctionTool(t *testing.T) tool.Tool {
	t.Helper()
	ft, err := functiontool.New(functiontool.Config{
		Name:        "get_weather_ptr",
		Description: "Gets weather information",
	}, func(_ agent.Context, in *weatherInput) (weatherOutput, error) {
		return weatherOutput{Summary: in.City}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return ft
}

func TestToMCPTool(t *testing.T) {
	fnTool := newWeatherFunctionTool(t)
	fnDecl := fnTool.(interface {
		Declaration() *genai.FunctionDeclaration
	}).Declaration()

	nestedJSONSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"username": map[string]any{"type": "string"},
			"profile": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"email": map[string]any{"type": "string"},
					"age":   map[string]any{"type": "integer"},
					"tags": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
				"required": []any{"email"},
			},
		},
		"required": []any{"username", "profile"},
	}

	tests := []struct {
		name    string
		tool    tool.Tool
		want    *mcp.Tool
		wantErr bool
	}{
		{
			name:    "nil tool returns error",
			tool:    nil,
			wantErr: true,
		},
		{
			name: "tool without Declaration method defaults to object schema",
			tool: &stubToolWithoutDecl{
				name:        "ping",
				description: "Pings the service",
			},
			want: &mcp.Tool{
				Name:        "ping",
				Description: "Pings the service",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "tool with nil declaration defaults to object schema",
			tool: &stubToolWithDecl{
				name:        "noop",
				description: "Does nothing",
				decl:        nil,
			},
			want: &mcp.Tool{
				Name:        "noop",
				Description: "Does nothing",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "tool with declaration but no parameters defaults to object schema",
			tool: &stubToolWithDecl{
				name:        "clock",
				description: "Gets the current time",
				decl: &genai.FunctionDeclaration{
					Name:        "clock",
					Description: "Gets the current time",
				},
			},
			want: &mcp.Tool{
				Name:        "clock",
				Description: "Gets the current time",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "tool with Parameters schema converts to JSON schema",
			tool: &stubToolWithDecl{
				name:        "get_weather",
				description: "Gets weather information",
				decl: &genai.FunctionDeclaration{
					Name:        "get_weather",
					Description: "Gets weather information",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"location": {
								Type:        genai.TypeString,
								Description: "The location to get weather for",
							},
							"units": {
								Type:        genai.TypeString,
								Description: "Temperature units",
							},
						},
						Required:         []string{"location"},
						PropertyOrdering: []string{"location", "units"},
					},
				},
			},
			want: &mcp.Tool{
				Name:        "get_weather",
				Description: "Gets weather information",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The location to get weather for",
						},
						"units": map[string]any{
							"type":        "string",
							"description": "Temperature units",
						},
					},
					"required": []any{"location"},
				},
			},
		},
		{
			name: "tool with ParametersJsonSchema uses it directly",
			tool: &stubToolWithDecl{
				name:        "search",
				description: "Searches a database",
				decl: &genai.FunctionDeclaration{
					Name:        "search",
					Description: "Searches a database",
					ParametersJsonSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"query": map[string]any{
								"type":        "string",
								"description": "The search query",
							},
							"limit": map[string]any{
								"type":        "integer",
								"description": "Maximum number of results",
							},
						},
						"required": []any{"query"},
					},
				},
			},
			want: &mcp.Tool{
				Name:        "search",
				Description: "Searches a database",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "The search query",
						},
						"limit": map[string]any{
							"type":        "integer",
							"description": "Maximum number of results",
						},
					},
					"required": []any{"query"},
				},
			},
		},
		{
			name: "tool prefers ParametersJsonSchema over Parameters",
			tool: &stubToolWithDecl{
				name:        "prefer_json",
				description: "Prefers JSON schema",
				decl: &genai.FunctionDeclaration{
					Name:        "prefer_json",
					Description: "Prefers JSON schema",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"schema_param": {Type: genai.TypeString},
						},
					},
					ParametersJsonSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"json_param": map[string]any{"type": "string"},
						},
					},
				},
			},
			want: &mcp.Tool{
				Name:        "prefer_json",
				Description: "Prefers JSON schema",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"json_param": map[string]any{"type": "string"},
					},
				},
			},
		},
		{
			name: "tool with complex nested schema",
			tool: &stubToolWithDecl{
				name:        "create_user",
				description: "Creates a new user",
				decl: &genai.FunctionDeclaration{
					Name:                 "create_user",
					Description:          "Creates a new user",
					ParametersJsonSchema: nestedJSONSchema,
				},
			},
			want: &mcp.Tool{
				Name:        "create_user",
				Description: "Creates a new user",
				InputSchema: nestedJSONSchema,
			},
		},
		{
			name: "tool built with functiontool.New",
			tool: fnTool,
			want: &mcp.Tool{
				Name:        "get_weather",
				Description: "Gets weather information",
				InputSchema: fnDecl.ParametersJsonSchema,
			},
		},
		{
			name: "tool built with functiontool.New and a pointer argument gets object type",
			tool: newPointerWeatherFunctionTool(t),
			want: &mcp.Tool{
				Name:        "get_weather_ptr",
				Description: "Gets weather information",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
						"units": map[string]any{
							"type":        "string",
							"description": "Temperature units",
						},
					},
					"required":             []any{"city"},
					"additionalProperties": false,
				},
			},
		},
		{
			name: "tool with typed-nil ParametersJsonSchema defaults to object schema",
			tool: &stubToolWithDecl{
				name:        "typed_nil",
				description: "Typed nil schema",
				decl: &genai.FunctionDeclaration{
					Name:                 "typed_nil",
					ParametersJsonSchema: (*jsonschema.Schema)(nil),
				},
			},
			want: &mcp.Tool{
				Name:        "typed_nil",
				Description: "Typed nil schema",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "tool with empty ParametersJsonSchema map defaults to object schema",
			tool: &stubToolWithDecl{
				name:        "empty_map",
				description: "Empty schema",
				decl: &genai.FunctionDeclaration{
					Name:                 "empty_map",
					ParametersJsonSchema: map[string]any{},
				},
			},
			want: &mcp.Tool{
				Name:        "empty_map",
				Description: "Empty schema",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		{
			name: "tool with ParametersJsonSchema listing object among types gets object type",
			tool: &stubToolWithDecl{
				name:        "nullable_object",
				description: "Nullable object",
				decl: &genai.FunctionDeclaration{
					Name: "nullable_object",
					ParametersJsonSchema: &jsonschema.Schema{
						Types: []string{"null", "object"},
						Properties: map[string]*jsonschema.Schema{
							"q": {Type: "string"},
						},
					},
				},
			},
			want: &mcp.Tool{
				Name:        "nullable_object",
				Description: "Nullable object",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"q": map[string]any{"type": "string"},
					},
				},
			},
		},
		{
			name: "tool with Parameters lacking top-level type gets object type",
			tool: &stubToolWithDecl{
				name:        "typeless",
				description: "Typeless parameters",
				decl: &genai.FunctionDeclaration{
					Name: "typeless",
					Parameters: &genai.Schema{
						Properties: map[string]*genai.Schema{
							"q": {Type: genai.TypeString},
						},
					},
				},
			},
			want: &mcp.Tool{
				Name:        "typeless",
				Description: "Typeless parameters",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"q": map[string]any{"type": "string"},
					},
				},
			},
		},
		{
			name: "tool with non-object ParametersJsonSchema type returns error",
			tool: &stubToolWithDecl{
				name: "string_schema",
				decl: &genai.FunctionDeclaration{
					Name:                 "string_schema",
					ParametersJsonSchema: map[string]any{"type": "string"},
				},
			},
			wantErr: true,
		},
		{
			name: "tool with ParametersJsonSchema types excluding object returns error",
			tool: &stubToolWithDecl{
				name: "string_or_null",
				decl: &genai.FunctionDeclaration{
					Name:                 "string_or_null",
					ParametersJsonSchema: &jsonschema.Schema{Types: []string{"null", "string"}},
				},
			},
			wantErr: true,
		},
		{
			name: "tool with ParametersJsonSchema that is not a JSON object returns error",
			tool: &stubToolWithDecl{
				name: "not_object",
				decl: &genai.FunctionDeclaration{
					Name:                 "not_object",
					ParametersJsonSchema: "not a schema",
				},
			},
			wantErr: true,
		},
		{
			name: "tool with non-object Parameters type returns error",
			tool: &stubToolWithDecl{
				name: "string_params",
				decl: &genai.FunctionDeclaration{
					Name:       "string_params",
					Parameters: &genai.Schema{Type: genai.TypeString},
				},
			},
			wantErr: true,
		},
		{
			name: "tool with unmarshalable Parameters returns error",
			tool: &stubToolWithDecl{
				name:        "broken",
				description: "Broken schema",
				decl: &genai.FunctionDeclaration{
					Name: "broken",
					Parameters: &genai.Schema{
						Type:    genai.TypeString,
						Example: make(chan int),
					},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ToMCPTool(tc.tool)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ToMCPTool() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ToMCPTool() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestToMCPToolDoesNotModifyCallerSchema(t *testing.T) {
	schema := map[string]any{
		"type": []any{"null", "object"},
		"properties": map[string]any{
			"q": map[string]any{"type": "string"},
		},
	}
	want := map[string]any{
		"type": []any{"null", "object"},
		"properties": map[string]any{
			"q": map[string]any{"type": "string"},
		},
	}
	adkTool := &stubToolWithDecl{
		name: "caller_schema",
		decl: &genai.FunctionDeclaration{Name: "caller_schema", ParametersJsonSchema: schema},
	}

	got, err := ToMCPTool(adkTool)
	if err != nil {
		t.Fatalf("ToMCPTool() error = %v", err)
	}
	if typ := got.InputSchema.(map[string]any)["type"]; typ != "object" {
		t.Errorf("ToMCPTool() InputSchema type = %v, want object", typ)
	}
	if diff := cmp.Diff(want, schema); diff != "" {
		t.Errorf("ToMCPTool() modified the caller's schema (-want +got):\n%s", diff)
	}
}

func TestToMCPToolServerRegistration(t *testing.T) {
	fnTool := newWeatherFunctionTool(t)
	paramTool := &stubToolWithDecl{
		name:        "lookup",
		description: "Looks up a record",
		decl: &genai.FunctionDeclaration{
			Name:        "lookup",
			Description: "Looks up a record",
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"id": {
						Type: genai.TypeObject,
						AnyOf: []*genai.Schema{
							{Type: genai.TypeString},
							{Type: genai.TypeInteger},
						},
					},
				},
				Required:         []string{"id"},
				PropertyOrdering: []string{"id"},
			},
		},
	}
	noDeclTool := &stubToolWithoutDecl{
		name:        "ping",
		description: "Pings the server",
	}

	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	noopHandler := func(_ context.Context, _ *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return &mcp.CallToolResult{}, nil
	}

	ptrTool := newPointerWeatherFunctionTool(t)
	typedNilTool := &stubToolWithDecl{
		name:        "typed_nil",
		description: "Typed nil schema",
		decl: &genai.FunctionDeclaration{
			Name:                 "typed_nil",
			ParametersJsonSchema: (*jsonschema.Schema)(nil),
		},
	}
	typelessTool := &stubToolWithDecl{
		name:        "typeless",
		description: "Typeless parameters",
		decl: &genai.FunctionDeclaration{
			Name: "typeless",
			Parameters: &genai.Schema{
				Properties: map[string]*genai.Schema{"q": {Type: genai.TypeString}},
			},
		},
	}

	// server.AddTool panics on a schema it rejects, failing the test.
	for _, adkTool := range []tool.Tool{fnTool, paramTool, noDeclTool, ptrTool, typedNilTool, typelessTool} {
		mcpTool, err := ToMCPTool(adkTool)
		if err != nil {
			t.Fatalf("ToMCPTool(%q) error = %v", adkTool.Name(), err)
		}
		server.AddTool(mcpTool, noopHandler)
	}

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(t.Context(), serverTransport, nil)
	if err != nil {
		t.Fatalf("server.Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = serverSession.Close() })

	client := mcp.NewClient(&mcp.Implementation{Name: "test_client", Version: "v1.0.0"}, nil)
	clientSession, err := client.Connect(t.Context(), clientTransport, nil)
	if err != nil {
		t.Fatalf("client.Connect() error = %v", err)
	}
	t.Cleanup(func() { _ = clientSession.Close() })

	listed, err := clientSession.ListTools(t.Context(), nil)
	if err != nil {
		t.Fatalf("clientSession.ListTools() error = %v", err)
	}

	gotByName := make(map[string]*mcp.Tool, len(listed.Tools))
	for _, tool := range listed.Tools {
		gotByName[tool.Name] = tool
	}

	if len(gotByName) != 6 {
		t.Fatalf("ListTools() returned %d tools, want 6", len(gotByName))
	}

	wantWeatherSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type":        "string",
				"description": "The city name",
			},
			"units": map[string]any{
				"type":        "string",
				"description": "Temperature units",
			},
		},
		"required":             []any{"city"},
		"additionalProperties": false,
	}
	if diff := cmp.Diff(wantWeatherSchema, gotByName["get_weather"].InputSchema); diff != "" {
		t.Errorf("get_weather InputSchema mismatch (-want +got):\n%s", diff)
	}

	wantLookupSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		"required": []any{"id"},
	}
	if diff := cmp.Diff(wantLookupSchema, gotByName["lookup"].InputSchema); diff != "" {
		t.Errorf("lookup InputSchema mismatch (-want +got):\n%s", diff)
	}

	wantPingSchema := map[string]any{"type": "object"}
	if diff := cmp.Diff(wantPingSchema, gotByName["ping"].InputSchema); diff != "" {
		t.Errorf("ping InputSchema mismatch (-want +got):\n%s", diff)
	}

	wantPtrSchema := map[string]any{
		"type":                 "object",
		"properties":           wantWeatherSchema["properties"],
		"required":             []any{"city"},
		"additionalProperties": false,
	}
	if diff := cmp.Diff(wantPtrSchema, gotByName["get_weather_ptr"].InputSchema); diff != "" {
		t.Errorf("get_weather_ptr InputSchema mismatch (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(map[string]any{"type": "object"}, gotByName["typed_nil"].InputSchema); diff != "" {
		t.Errorf("typed_nil InputSchema mismatch (-want +got):\n%s", diff)
	}

	wantTypelessSchema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"q": map[string]any{"type": "string"}},
	}
	if diff := cmp.Diff(wantTypelessSchema, gotByName["typeless"].InputSchema); diff != "" {
		t.Errorf("typeless InputSchema mismatch (-want +got):\n%s", diff)
	}
}
