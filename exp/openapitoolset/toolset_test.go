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
	"os"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

func loadFixture(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/test.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func toolNames(t *testing.T, ts *Toolset) []string {
	t.Helper()
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(tools))
	for i, tl := range tools {
		names[i] = tl.Name()
	}
	return names
}

func TestNewFromYAML(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	if got := len(toolNames(t, ts)); got != 5 {
		t.Fatalf("got %d tools, want 5", got)
	}
}

func TestNewFromSpecMap(t *testing.T) {
	v, err := decodeDocument(loadFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	specMap := plainSchema(v).(map[string]any)
	ts, err := New(Config{SpecMap: specMap})
	if err != nil {
		t.Fatal(err)
	}
	if got := len(toolNames(t, ts)); got != 5 {
		t.Fatalf("got %d tools, want 5", got)
	}
}

func TestNewRejectsBothSpecAndMap(t *testing.T) {
	_, err := New(Config{Spec: []byte("{}"), SpecMap: map[string]any{"openapi": "3.0.0"}})
	if err == nil {
		t.Fatal("New succeeded, want error when both Spec and SpecMap are set")
	}
}

func TestNewRejectsNoSpec(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("New succeeded, want error when neither Spec nor SpecMap is set")
	}
}

func TestToolExisting(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	tl, ok := ts.Tool("calendar_calendars_insert")
	if !ok {
		t.Fatal("calendar_calendars_insert not found")
	}
	if tl.Name() != "calendar_calendars_insert" {
		t.Errorf("name = %q", tl.Name())
	}
	if tl.Description() != "Creates a secondary calendar." {
		t.Errorf("description = %q", tl.Description())
	}
	if tl.endpoint.Method != "post" {
		t.Errorf("method = %q, want post", tl.endpoint.Method)
	}
	if tl.endpoint.BaseURL != "https://www.googleapis.com/calendar/v3" {
		t.Errorf("baseURL = %q", tl.endpoint.BaseURL)
	}
	if tl.endpoint.Path != "/calendars" {
		t.Errorf("path = %q, want /calendars", tl.endpoint.Path)
	}
	if tl.authScheme == nil || tl.authScheme.Type != "oauth2" {
		t.Errorf("authScheme = %+v, want oauth2", tl.authScheme)
	}

	get, ok := ts.Tool("calendar_calendars_get")
	if !ok {
		t.Fatal("calendar_calendars_get not found")
	}
	if get.endpoint.Path != "/calendars/{calendarId}" {
		t.Errorf("get path = %q", get.endpoint.Path)
	}
	for _, name := range []string{"calendar_calendars_update", "calendar_calendars_delete", "calendar_calendars_patch"} {
		if _, ok := ts.Tool(name); !ok {
			t.Errorf("%q not found", name)
		}
	}
}

func TestToolNonExisting(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := ts.Tool("non_existent_tool"); ok {
		t.Fatal("non_existent_tool should not be found")
	}
}

func TestToolNamePrefix(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t), ToolNamePrefix: "my_api"})
	if err != nil {
		t.Fatal(err)
	}
	names := toolNames(t, ts)
	if len(names) != 5 {
		t.Fatalf("got %d tools, want 5", len(names))
	}
	found := false
	for _, n := range names {
		if len(n) < len("my_api_") || n[:len("my_api_")] != "my_api_" {
			t.Errorf("name %q is not prefixed", n)
		}
		if n == "my_api_calendar_calendars_insert" {
			found = true
		}
	}
	if !found {
		t.Errorf("my_api_calendar_calendars_insert not in %v", names)
	}
	// Lookup uses the unprefixed name.
	if _, ok := ts.Tool("calendar_calendars_insert"); !ok {
		t.Error("lookup should use unprefixed names")
	}
	if _, ok := ts.Tool("my_api_calendar_calendars_insert"); ok {
		t.Error("lookup should not match prefixed names")
	}
}

func TestToolFilter(t *testing.T) {
	filter := tool.AllowedToolsPredicate([]string{"calendar_calendars_get", "calendar_calendars_delete"})
	ts, err := New(Config{Spec: loadFixture(t), ToolFilter: filter})
	if err != nil {
		t.Fatal(err)
	}
	names := toolNames(t, ts)
	if len(names) != 2 {
		t.Fatalf("got %d tools, want 2: %v", len(names), names)
	}
	// The filter does not affect direct lookup.
	if _, ok := ts.Tool("calendar_calendars_insert"); !ok {
		t.Error("Tool lookup should ignore the filter")
	}
}

func TestToolFilterWithPrefix(t *testing.T) {
	// The filter sees unprefixed names even when a prefix is set.
	filter := tool.AllowedToolsPredicate([]string{"calendar_calendars_get"})
	ts, err := New(Config{Spec: loadFixture(t), ToolNamePrefix: "my_api", ToolFilter: filter})
	if err != nil {
		t.Fatal(err)
	}
	names := toolNames(t, ts)
	if len(names) != 1 || names[0] != "my_api_calendar_calendars_get" {
		t.Fatalf("names = %v, want [my_api_calendar_calendars_get]", names)
	}
}

func TestPreservePropertyNamesToolset(t *testing.T) {
	ts, err := New(Config{Spec: loadFixture(t), PreservePropertyNames: true})
	if err != nil {
		t.Fatal(err)
	}
	get, ok := ts.Tool("calendar_calendars_get")
	if !ok {
		t.Fatal("calendar_calendars_get not found")
	}
	props := get.parameters["properties"].(map[string]any)
	if _, ok := props["calendarId"]; !ok {
		t.Errorf("calendarId not preserved; got %v", props)
	}

	tsDefault, err := New(Config{Spec: loadFixture(t)})
	if err != nil {
		t.Fatal(err)
	}
	getDefault, _ := tsDefault.Tool("calendar_calendars_get")
	propsDefault := getDefault.parameters["properties"].(map[string]any)
	if _, ok := propsDefault["calendar_id"]; !ok {
		t.Errorf("calendar_id missing in default mode; got %v", propsDefault)
	}
	if _, ok := propsDefault["calendarId"]; ok {
		t.Error("calendarId present in default mode; want snake_case only")
	}
}

func TestOptionalSecurity(t *testing.T) {
	tests := []struct {
		name       string
		security   string
		wantScheme bool
	}{
		{"optional operation security", `"security": [{}, {"apiKey": []}],`, false},
		{"required operation security", `"security": [{"apiKey": []}],`, true},
		{"optional listed last", `"security": [{"apiKey": []}, {}],`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0"},
  "servers": [{"url": "https://api.example.com"}],
  "components": {"securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}},
  "paths": {"/items": {"get": {"operationId": "listItems", ` + tc.security + ` "responses": {"200": {"description": "OK"}}}}}
}`)
			ts, err := New(Config{Spec: spec})
			if err != nil {
				t.Fatal(err)
			}
			tl, ok := ts.Tool("list_items")
			if !ok {
				t.Fatal("list_items not found")
			}
			if tc.wantScheme && tl.authScheme == nil {
				t.Error("authScheme is nil, want apiKey")
			}
			if !tc.wantScheme && tl.authScheme != nil {
				t.Errorf("authScheme = %+v, want nil", tl.authScheme)
			}
		})
	}
}

func TestOperationSecurityOverridesGlobal(t *testing.T) {
	spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0"},
  "servers": [{"url": "https://api.example.com"}],
  "security": [{"apiKey": []}],
  "components": {"securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}},
  "paths": {
    "/items": {"get": {"operationId": "listItems", "security": [{}], "responses": {"200": {"description": "OK"}}}},
    "/other": {"get": {"operationId": "listOther", "responses": {"200": {"description": "OK"}}}}
  }
}`)
	ts, err := New(Config{Spec: spec})
	if err != nil {
		t.Fatal(err)
	}
	items, _ := ts.Tool("list_items")
	if items.authScheme != nil {
		t.Error("list_items should opt out of global security")
	}
	other, _ := ts.Tool("list_other")
	if other.authScheme == nil {
		t.Error("list_other should inherit global security")
	}
}

func TestDisambiguateTruncatedNames(t *testing.T) {
	prefix := "list_all_company_organization_department_users_by_filter_crit"
	if len(prefix) <= maxToolNameLength {
		t.Fatalf("test prefix must exceed %d chars", maxToolNameLength)
	}
	spec := []byte(`{
  "openapi": "3.0.0",
  "info": {"title": "Test API", "version": "1.0"},
  "servers": [{"url": "https://api.example.com"}],
  "paths": {
    "/users/v1": {"get": {"operationId": "` + prefix + `v1", "responses": {"200": {"description": "OK"}}}},
    "/users/v2": {"get": {"operationId": "` + prefix + `v2", "responses": {"200": {"description": "OK"}}}}
  }
}`)
	ts, err := New(Config{Spec: spec})
	if err != nil {
		t.Fatal(err)
	}
	names := toolNames(t, ts)
	seen := map[string]bool{}
	for _, n := range names {
		if seen[n] {
			t.Fatalf("names collided: %v", names)
		}
		seen[n] = true
		if len(n) > maxToolNameLength {
			t.Errorf("name %q exceeds %d chars", n, maxToolNameLength)
		}
	}
	// Both endpoints stay reachable.
	t1, _ := ts.Tool(names[0])
	t2, _ := ts.Tool(names[1])
	if t1.endpoint.Path != "/users/v1" {
		t.Errorf("tool 1 path = %q, want /users/v1", t1.endpoint.Path)
	}
	if t2.endpoint.Path != "/users/v2" {
		t.Errorf("tool 2 path = %q, want /users/v2", t2.endpoint.Path)
	}
}

func TestDisambiguateSkipsReservedSuffix(t *testing.T) {
	tests := []struct {
		operationIDs []string
		want         []string
	}{
		{[]string{"getItem", "get_item", "get_item_2"}, []string{"get_item", "get_item_3", "get_item_2"}},
		{[]string{"get_item_2", "getItem", "get_item"}, []string{"get_item_2", "get_item", "get_item_3"}},
	}
	for _, tc := range tests {
		paths := ""
		for i, id := range tc.operationIDs {
			if i > 0 {
				paths += ","
			}
			paths += `"/items/` + string(rune('0'+i)) + `": {"get": {"operationId": "` + id + `", "responses": {"200": {"description": "OK"}}}}`
		}
		spec := []byte(`{"openapi": "3.0.0", "info": {"title": "x", "version": "1"}, "servers": [{"url": "https://api.example.com"}], "paths": {` + paths + `}}`)
		ts, err := New(Config{Spec: spec})
		if err != nil {
			t.Fatal(err)
		}
		names := toolNames(t, ts)
		if len(names) != len(tc.want) {
			t.Fatalf("got %d names, want %d: %v", len(names), len(tc.want), names)
		}
		for i := range names {
			if names[i] != tc.want[i] {
				t.Errorf("names = %v, want %v", names, tc.want)
				break
			}
		}
	}
}

// TestToolsUsesReadonlyContext checks a filter receives the context it is given.
func TestToolsUsesReadonlyContext(t *testing.T) {
	var gotCtx bool
	filter := func(ctx agent.ReadonlyContext, _ tool.Tool) bool {
		gotCtx = true
		return true
	}
	ts, err := New(Config{Spec: loadFixture(t), ToolFilter: filter})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ts.Tools(nil); err != nil {
		t.Fatal(err)
	}
	if !gotCtx {
		t.Error("filter was not called")
	}
}
