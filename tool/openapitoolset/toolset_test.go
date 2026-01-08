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

package openapitoolset

import (
	"testing"

	"google.golang.org/adk/auth"
)

func TestNew_FromSpecDict(t *testing.T) {
	specDict := map[string]any{
		"openapi": "3.0.0",
		"info":    map[string]any{"title": "Test", "version": "1.0"},
		"servers": []any{map[string]any{"url": "https://api.example.com"}},
		"paths": map[string]any{
			"/users": map[string]any{
				"get": map[string]any{
					"operationId": "listUsers",
					"summary":     "List users",
				},
			},
		},
	}

	toolset, err := New(Config{SpecDict: specDict})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if toolset == nil {
		t.Fatal("New() returned nil")
	}
	if toolset.Name() != "openapi_toolset" {
		t.Errorf("Name() = %q, want %q", toolset.Name(), "openapi_toolset")
	}

	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("Tools() returned %d tools, want 1", len(tools))
	}
}

func TestNew_FromSpecStr_JSON(t *testing.T) {
	specJSON := `{
		"openapi": "3.0.0",
		"info": {"title": "Test", "version": "1.0"},
		"servers": [{"url": "https://api.example.com"}],
		"paths": {
			"/items": {
				"get": {
					"operationId": "listItems",
					"summary": "List items"
				}
			}
		}
	}`

	toolset, err := New(Config{SpecStr: specJSON, SpecStrType: "json"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if toolset == nil {
		t.Fatal("New() returned nil")
	}

	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("Tools() returned %d tools, want 1", len(tools))
	}
}

func TestNew_FromSpecStr_YAML(t *testing.T) {
	specYAML := `
openapi: "3.0.0"
info:
  title: Test
  version: "1.0"
servers:
  - url: https://api.example.com
paths:
  /products:
    get:
      operationId: listProducts
      summary: List products
`

	toolset, err := New(Config{SpecStr: specYAML, SpecStrType: "yaml"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if toolset == nil {
		t.Fatal("New() returned nil")
	}

	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("Tools() returned %d tools, want 1", len(tools))
	}
}

func TestNew_NoSpec(t *testing.T) {
	_, err := New(Config{})
	if err == nil {
		t.Error("New() should error when no spec provided")
	}
}

func TestNew_InvalidJSON(t *testing.T) {
	_, err := New(Config{SpecStr: "not valid json", SpecStrType: "json"})
	if err == nil {
		t.Error("New() should error with invalid JSON")
	}
}

func TestNew_InvalidYAML(t *testing.T) {
	_, err := New(Config{SpecStr: "not: valid: yaml: [", SpecStrType: "yaml"})
	if err == nil {
		t.Error("New() should error with invalid YAML")
	}
}

func TestNew_WithToolNamePrefix(t *testing.T) {
	specDict := map[string]any{
		"openapi": "3.0.0",
		"info":    map[string]any{"title": "Test", "version": "1.0"},
		"servers": []any{map[string]any{"url": "https://api.example.com"}},
		"paths": map[string]any{
			"/users": map[string]any{
				"get": map[string]any{"operationId": "getUsers"},
			},
		},
	}

	toolset, err := New(Config{
		SpecDict:       specDict,
		ToolNamePrefix: "github_",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("Tools() returned %d tools, want 1", len(tools))
	}
	if tools[0].Name() != "github_getUsers" {
		t.Errorf("Tool name = %q, want %q", tools[0].Name(), "github_getUsers")
	}
}

func TestNew_WithAuthScheme(t *testing.T) {
	specDict := map[string]any{
		"openapi": "3.0.0",
		"info":    map[string]any{"title": "Test", "version": "1.0"},
		"servers": []any{map[string]any{"url": "https://api.example.com"}},
		"paths": map[string]any{
			"/users": map[string]any{
				"get": map[string]any{"operationId": "getUsers"},
			},
		},
	}

	authScheme := &auth.APIKeyScheme{
		In:   auth.APIKeyInHeader,
		Name: "X-API-Key",
	}

	toolset, err := New(Config{
		SpecDict:   specDict,
		AuthScheme: authScheme,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Verify the toolset was created (auth scheme is set internally on tools)
	tools, err := toolset.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 {
		t.Errorf("Tools() returned %d tools, want 1", len(tools))
	}
}

func TestOpenAPIToolset_GetTool(t *testing.T) {
	specDict := map[string]any{
		"openapi": "3.0.0",
		"info":    map[string]any{"title": "Test", "version": "1.0"},
		"servers": []any{map[string]any{"url": "https://api.example.com"}},
		"paths": map[string]any{
			"/users": map[string]any{
				"get":  map[string]any{"operationId": "getUsers"},
				"post": map[string]any{"operationId": "createUser"},
			},
		},
	}

	toolset, err := New(Config{SpecDict: specDict})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Cast to our internal type to access GetTool
	ts, ok := toolset.(*openAPIToolset)
	if !ok {
		t.Fatal("toolset is not *openAPIToolset")
	}

	tool := ts.GetTool("getUsers")
	if tool == nil {
		t.Error("GetTool('getUsers') returned nil")
	}

	tool = ts.GetTool("nonExistent")
	if tool != nil {
		t.Error("GetTool('nonExistent') should return nil")
	}
}
