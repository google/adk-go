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

package openapitoolset

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/tool"
)

func TestNewResolvesReferencesAndOverridesPathParameters(t *testing.T) {
	spec := readFixture(t)
	ts, err := New(context.Background(), spec, Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if got, want := len(tools), 2; got != want {
		t.Fatalf("len(Tools()) = %d, want %d", got, want)
	}

	getPet := findOperationTool(t, tools, "get_pet")
	getDecl := getPet.Declaration()
	if got, want := getDecl.Description, "Get a pet by ID"; got != want {
		t.Errorf("get_pet description = %q, want %q", got, want)
	}
	if got, want := getDecl.Parameters.Properties["id"].Pattern, "^[a-z0-9/-]+$"; got != want {
		t.Errorf("resolved id pattern = %q, want %q", got, want)
	}
	limit := getDecl.Parameters.Properties["limit"]
	if got, want := limit.Description, "Operation-level limit"; got != want {
		t.Errorf("limit description = %q, want operation override %q", got, want)
	}
	if limit.Maximum == nil || *limit.Maximum != 10 {
		t.Errorf("limit maximum = %v, want 10", limit.Maximum)
	}
	if !contains(getDecl.Parameters.Required, "limit") {
		t.Errorf("get_pet required = %v, want limit", getDecl.Parameters.Required)
	}
	if getDecl.Parameters.Properties["x_trace"] == nil {
		t.Errorf("get_pet parameters = %v, want sanitized x_trace header argument", getDecl.Parameters.Properties)
	}

	updatePet := findOperationTool(t, tools, "update_pet")
	body := updatePet.Declaration().Parameters
	if got := body.Properties["name"]; got == nil || got.Type != genai.TypeString {
		t.Errorf("resolved body name schema = %#v, want string", got)
	}
	if got := body.Properties["age"]; got == nil || got.Type != genai.TypeInteger {
		t.Errorf("resolved body age schema = %#v, want integer", got)
	}
	if !contains(body.Required, "name") {
		t.Errorf("update_pet required = %v, want name", body.Required)
	}
}

func TestLoadSources(t *testing.T) {
	spec := readFixture(t)
	path := filepath.Join(t.TempDir(), "petstore.yaml")
	if err := os.WriteFile(path, spec, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml")
		_, _ = w.Write(spec)
	}))
	t.Cleanup(server.Close)

	tests := []struct {
		name string
		load func() (*Toolset, error)
	}{
		{name: "bytes", load: func() (*Toolset, error) {
			return New(context.Background(), spec, Config{})
		}},
		{name: "file", load: func() (*Toolset, error) {
			return NewFromFile(context.Background(), path, Config{})
		}},
		{name: "URL", load: func() (*Toolset, error) {
			return NewFromURL(context.Background(), server.URL, Config{HTTPClient: server.Client()})
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, err := tt.load()
			if err != nil {
				t.Fatalf("load() error = %v", err)
			}
			tools, err := ts.Tools(nil)
			if err != nil {
				t.Fatalf("Tools() error = %v", err)
			}
			if got, want := len(tools), 2; got != want {
				t.Errorf("len(Tools()) = %d, want %d", got, want)
			}
		})
	}
}

func TestToolFilter(t *testing.T) {
	ts, err := New(context.Background(), readFixture(t), Config{
		ToolFilter: tool.AllowedToolsPredicate([]string{"update_pet"}),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if got, want := len(tools), 1; got != want {
		t.Fatalf("len(Tools()) = %d, want %d", got, want)
	}
	if got, want := tools[0].Name(), "update_pet"; got != want {
		t.Errorf("tool name = %q, want %q", got, want)
	}
}

func TestNewFromURLRejectsNonHTTPURL(t *testing.T) {
	_, err := NewFromURL(context.Background(), "file:///tmp/openapi.yaml", Config{})
	if err == nil || !strings.Contains(err.Error(), "scheme") {
		t.Fatalf("NewFromURL() error = %v, want unsupported scheme", err)
	}
}

func TestNewFromURLDoesNotSendOperationCredential(t *testing.T) {
	spec := readFixture(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("document request Authorization = %q, want empty", got)
		}
		_, _ = w.Write(spec)
	}))
	t.Cleanup(server.Close)

	_, err := NewFromURL(context.Background(), server.URL, Config{
		HTTPClient: server.Client(),
		Auth:       auth.StaticToken("operation-token"),
	})
	if err != nil {
		t.Fatalf("NewFromURL() error = %v", err)
	}
}

func readFixture(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/petstore.yaml")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	return b
}

func findOperationTool(t *testing.T, tools []tool.Tool, name string) *operationTool {
	t.Helper()
	for _, candidate := range tools {
		if candidate.Name() == name {
			op, ok := candidate.(*operationTool)
			if !ok {
				t.Fatalf("tool %q has type %T, want *operationTool", name, candidate)
			}
			return op
		}
	}
	t.Fatalf("tool %q not found", name)
	return nil
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
