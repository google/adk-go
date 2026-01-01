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
)

func TestParseOpenAPISpec_Basic(t *testing.T) {
	spec := map[string]any{
		"openapi": "3.0.0",
		"info": map[string]any{
			"title":   "Test API",
			"version": "1.0",
		},
		"servers": []any{
			map[string]any{"url": "https://api.example.com"},
		},
		"paths": map[string]any{
			"/users": map[string]any{
				"get": map[string]any{
					"operationId": "listUsers",
					"summary":     "List all users",
				},
			},
		},
	}

	tools, err := parseOpenAPISpec(spec)
	if err != nil {
		t.Fatalf("parseOpenAPISpec() error = %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("parseOpenAPISpec() returned %d tools, want 1", len(tools))
	}

	tool := tools[0]
	if tool.name != "listUsers" {
		t.Errorf("tool.name = %q, want %q", tool.name, "listUsers")
	}
	if tool.method != "GET" {
		t.Errorf("tool.method = %q, want %q", tool.method, "GET")
	}
	if tool.baseURL != "https://api.example.com" {
		t.Errorf("tool.baseURL = %q, want %q", tool.baseURL, "https://api.example.com")
	}
}

func TestParseOpenAPISpec_NoPaths(t *testing.T) {
	spec := map[string]any{
		"openapi": "3.0.0",
	}

	_, err := parseOpenAPISpec(spec)
	if err == nil {
		t.Error("parseOpenAPISpec() should error when no paths")
	}
}

func TestGenerateOperationName(t *testing.T) {
	tests := []struct {
		method string
		path   string
		want   string
	}{
		{"get", "/users", "get_users"},
		{"post", "/users/{id}", "post_users_id"},
		{"get", "/repos/{owner}/{repo}/issues", "get_repos_owner_repo_issues"},
		{"delete", "/items/{item-id}", "delete_items_item_id"},
	}

	for _, tt := range tests {
		t.Run(tt.method+"_"+tt.path, func(t *testing.T) {
			got := generateOperationName(tt.method, tt.path)
			if got != tt.want {
				t.Errorf("generateOperationName(%q, %q) = %q, want %q", tt.method, tt.path, got, tt.want)
			}
		})
	}
}

func TestParseParameters(t *testing.T) {
	op := map[string]any{
		"parameters": []any{
			map[string]any{
				"name":        "id",
				"in":          "path",
				"required":    true,
				"description": "User ID",
				"schema":      map[string]any{"type": "string"},
			},
			map[string]any{
				"name":        "limit",
				"in":          "query",
				"required":    false,
				"description": "Max results",
				"schema":      map[string]any{"type": "integer"},
			},
		},
	}
	pathItem := map[string]any{}

	params := parseParameters(op, pathItem)

	if len(params) != 2 {
		t.Fatalf("parseParameters() returned %d params, want 2", len(params))
	}

	if params[0].Name != "id" || params[0].In != "path" || !params[0].Required {
		t.Errorf("params[0] = %+v, want path param 'id'", params[0])
	}
	if params[1].Name != "limit" || params[1].In != "query" || params[1].Required {
		t.Errorf("params[1] = %+v, want query param 'limit'", params[1])
	}
}

func TestParseRequestBody(t *testing.T) {
	reqBody := map[string]any{
		"description": "User data",
		"required":    true,
		"content": map[string]any{
			"application/json": map[string]any{
				"schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
					},
				},
			},
		},
	}

	rb := parseRequestBody(reqBody)

	if rb.Description != "User data" {
		t.Errorf("Description = %q, want %q", rb.Description, "User data")
	}
	if !rb.Required {
		t.Error("Required should be true")
	}
	if len(rb.Content) != 1 {
		t.Errorf("Content has %d entries, want 1", len(rb.Content))
	}
	if _, ok := rb.Content["application/json"]; !ok {
		t.Error("Content should have 'application/json' key")
	}
}

func TestParseOperation_WithDescription(t *testing.T) {
	op := map[string]any{
		"operationId": "getUser",
		"description": "Get a user by ID",
		"parameters": []any{
			map[string]any{
				"name":     "id",
				"in":       "path",
				"required": true,
			},
		},
	}
	pathItem := map[string]any{}

	parsed, err := parseOperation("/users/{id}", "get", op, "https://api.example.com", pathItem)
	if err != nil {
		t.Fatalf("parseOperation() error = %v", err)
	}

	if parsed.Name != "getUser" {
		t.Errorf("Name = %q, want %q", parsed.Name, "getUser")
	}
	if parsed.Description != "Get a user by ID" {
		t.Errorf("Description = %q, want %q", parsed.Description, "Get a user by ID")
	}
}

func TestParseOperation_UseSummaryWhenNoDescription(t *testing.T) {
	op := map[string]any{
		"operationId": "getUser",
		"summary":     "Get user",
	}
	pathItem := map[string]any{}

	parsed, err := parseOperation("/users/{id}", "get", op, "https://api.example.com", pathItem)
	if err != nil {
		t.Fatalf("parseOperation() error = %v", err)
	}

	if parsed.Description != "Get user" {
		t.Errorf("Description = %q, want %q (should use summary)", parsed.Description, "Get user")
	}
}
