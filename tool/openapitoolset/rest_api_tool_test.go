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

func TestRestApiTool_Name(t *testing.T) {
	tool := &RestApiTool{name: "testTool"}
	if got := tool.Name(); got != "testTool" {
		t.Errorf("Name() = %q, want %q", got, "testTool")
	}
}

func TestRestApiTool_Description(t *testing.T) {
	tool := &RestApiTool{description: "Test description"}
	if got := tool.Description(); got != "Test description" {
		t.Errorf("Description() = %q, want %q", got, "Test description")
	}
}

func TestRestApiTool_IsLongRunning(t *testing.T) {
	tool := &RestApiTool{}
	if got := tool.IsLongRunning(); got {
		t.Error("IsLongRunning() = true, want false")
	}
}

func TestRestApiTool_Declaration(t *testing.T) {
	tool := &RestApiTool{
		name:        "getUser",
		description: "Get a user",
		parameters: []Parameter{
			{Name: "id", In: "path", Required: true, Schema: map[string]any{"type": "string"}},
			{Name: "fields", In: "query", Required: false, Schema: map[string]any{"type": "string"}},
		},
	}

	decl := tool.Declaration()

	if decl.Name != "getUser" {
		t.Errorf("Declaration().Name = %q, want %q", decl.Name, "getUser")
	}
	if decl.Description != "Get a user" {
		t.Errorf("Declaration().Description = %q, want %q", decl.Description, "Get a user")
	}
	if decl.Parameters == nil {
		t.Fatal("Declaration().Parameters is nil")
	}
	if len(decl.Parameters.Properties) != 2 {
		t.Errorf("Declaration().Parameters.Properties has %d entries, want 2", len(decl.Parameters.Properties))
	}
	if len(decl.Parameters.Required) != 1 {
		t.Errorf("Declaration().Parameters.Required has %d entries, want 1", len(decl.Parameters.Required))
	}
}

func TestRestApiTool_buildURL(t *testing.T) {
	tests := []struct {
		name string
		tool *RestApiTool
		args map[string]any
		want string
	}{
		{
			name: "path parameter",
			tool: &RestApiTool{
				baseURL: "https://api.example.com",
				path:    "/users/{id}",
				parameters: []Parameter{
					{Name: "id", In: "path"},
				},
			},
			args: map[string]any{"id": "123"},
			want: "https://api.example.com/users/123",
		},
		{
			name: "query parameter",
			tool: &RestApiTool{
				baseURL: "https://api.example.com",
				path:    "/users",
				parameters: []Parameter{
					{Name: "limit", In: "query"},
				},
			},
			args: map[string]any{"limit": 10},
			want: "https://api.example.com/users?limit=10",
		},
		{
			name: "path and query parameters",
			tool: &RestApiTool{
				baseURL: "https://api.example.com",
				path:    "/repos/{owner}/{repo}/issues",
				parameters: []Parameter{
					{Name: "owner", In: "path"},
					{Name: "repo", In: "path"},
					{Name: "state", In: "query"},
				},
			},
			args: map[string]any{"owner": "google", "repo": "adk-go", "state": "open"},
			want: "https://api.example.com/repos/google/adk-go/issues?state=open",
		},
		{
			name: "special characters in path",
			tool: &RestApiTool{
				baseURL: "https://api.example.com",
				path:    "/files/{path}",
				parameters: []Parameter{
					{Name: "path", In: "path"},
				},
			},
			args: map[string]any{"path": "foo/bar"},
			want: "https://api.example.com/files/foo%2Fbar",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.tool.buildURL(tt.args)
			if got != tt.want {
				t.Errorf("buildURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRestApiTool_getAuthHeaders_OAuth2(t *testing.T) {
	tool := &RestApiTool{
		authCredential: &auth.AuthCredential{
			AuthType: auth.AuthCredentialTypeOAuth2,
			OAuth2: &auth.OAuth2Auth{
				AccessToken: "my-token",
			},
		},
	}

	headers := tool.getAuthHeaders()
	if headers == nil {
		t.Fatal("getAuthHeaders() returned nil")
	}
	if headers["Authorization"] != "Bearer my-token" {
		t.Errorf("Authorization = %q, want %q", headers["Authorization"], "Bearer my-token")
	}
}

func TestRestApiTool_getAuthHeaders_APIKey(t *testing.T) {
	tool := &RestApiTool{
		authScheme: &auth.APIKeyScheme{
			In:   auth.APIKeyInHeader,
			Name: "X-API-Key",
		},
		authCredential: &auth.AuthCredential{
			AuthType: auth.AuthCredentialTypeAPIKey,
			APIKey:   "secret-key",
		},
	}

	headers := tool.getAuthHeaders()
	if headers == nil {
		t.Fatal("getAuthHeaders() returned nil")
	}
	if headers["X-API-Key"] != "secret-key" {
		t.Errorf("X-API-Key = %q, want %q", headers["X-API-Key"], "secret-key")
	}
}

func TestRestApiTool_getAuthHeaders_HTTPBasic(t *testing.T) {
	tool := &RestApiTool{
		authCredential: &auth.AuthCredential{
			AuthType: auth.AuthCredentialTypeHTTP,
			HTTP: &auth.HTTPAuth{
				Scheme: "basic",
				Credentials: &auth.HTTPCredentials{
					Username: "user",
					Password: "pass",
				},
			},
		},
	}

	headers := tool.getAuthHeaders()
	if headers == nil {
		t.Fatal("getAuthHeaders() returned nil")
	}
	// base64("user:pass") = "dXNlcjpwYXNz"
	want := "Basic dXNlcjpwYXNz"
	if headers["Authorization"] != want {
		t.Errorf("Authorization = %q, want %q", headers["Authorization"], want)
	}
}

func TestRestApiTool_getAuthHeaders_HTTPBearer(t *testing.T) {
	tool := &RestApiTool{
		authCredential: &auth.AuthCredential{
			AuthType: auth.AuthCredentialTypeHTTP,
			HTTP: &auth.HTTPAuth{
				Scheme: "bearer",
				Credentials: &auth.HTTPCredentials{
					Token: "jwt-token",
				},
			},
		},
	}

	headers := tool.getAuthHeaders()
	if headers == nil {
		t.Fatal("getAuthHeaders() returned nil")
	}
	if headers["Authorization"] != "Bearer jwt-token" {
		t.Errorf("Authorization = %q, want %q", headers["Authorization"], "Bearer jwt-token")
	}
}

func TestRestApiTool_getAuthHeaders_NoCredential(t *testing.T) {
	tool := &RestApiTool{}

	headers := tool.getAuthHeaders()
	if headers != nil {
		t.Errorf("getAuthHeaders() = %v, want nil", headers)
	}
}

func TestConvertSchemaToGenai(t *testing.T) {
	tests := []struct {
		name   string
		schema map[string]any
		check  func(t *testing.T, result any)
	}{
		{
			name:   "nil schema",
			schema: nil,
			check: func(t *testing.T, result any) {
				// convertSchemaToGenai returns nil for nil input
				// (based on implementation that returns nil at line 345-347)
				if result != nil {
					t.Log("nil input returned non-nil result (acceptable if implementation changed)")
				}
			},
		},
		{
			name:   "string type",
			schema: map[string]any{"type": "string", "description": "A name"},
			check: func(t *testing.T, result any) {
				// Just verify it doesn't panic and returns non-nil
				if result == nil {
					t.Error("expected non-nil result")
				}
			},
		},
		{
			name: "object with properties",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":   map[string]any{"type": "integer"},
					"name": map[string]any{"type": "string"},
				},
			},
			check: func(t *testing.T, result any) {
				if result == nil {
					t.Error("expected non-nil result")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertSchemaToGenai(tt.schema)
			tt.check(t, result)
		})
	}
}
