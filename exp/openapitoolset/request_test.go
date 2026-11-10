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
	"encoding/json"
	"errors"
	"maps"
	"net/url"
	"strings"
	"testing"
)

func boolPtr(b bool) *bool { return &b }

// testTool builds a tool directly from parts, for request-building tests that
// need precise control over the parameters.
func testTool(endpoint Endpoint, params []*apiParameter, body *requestBodyInfo) *RESTAPITool {
	return &RESTAPITool{name: "test_tool", endpoint: endpoint, params: params, body: body}
}

var sampleEndpoint = Endpoint{BaseURL: "https://example.com", Path: "/test", Method: "get"}

func TestPrepareRequestQueryAndBody(t *testing.T) {
	tool := testTool(
		sampleEndpoint,
		[]*apiParameter{
			{originalName: "param1", pyName: "param1", location: "body"},
			{originalName: "param2", pyName: "param2", location: "body"},
			{originalName: "testQueryParam", pyName: "test_query_param", location: "query"},
		},
		&requestBodyInfo{mediaType: "application/json", object: true},
	)
	req, err := tool.prepareRequest(map[string]any{
		"param1":           "value1",
		"param2":           float64(123),
		"test_query_param": "query_value",
	})
	if err != nil {
		t.Fatal(err)
	}
	if req.method != "get" {
		t.Errorf("method = %q, want get", req.method)
	}
	if req.url != "https://example.com/test" {
		t.Errorf("url = %q", req.url)
	}
	if got := req.query.Get("testQueryParam"); got != "query_value" {
		t.Errorf("query testQueryParam = %q", got)
	}
	var body map[string]any
	if err := json.Unmarshal(req.body, &body); err != nil {
		t.Fatalf("body: %v", err)
	}
	if body["param1"] != "value1" || body["param2"] != float64(123) {
		t.Errorf("json body = %v, want {param1:value1, param2:123}", body)
	}
	if req.contentType != "application/json" {
		t.Errorf("content type = %q", req.contentType)
	}
}

func TestPrepareRequestFalsyQueryParams(t *testing.T) {
	tool := testTool(sampleEndpoint, []*apiParameter{
		{originalName: "flag", pyName: "flag", location: "query"},
		{originalName: "offset", pyName: "offset", location: "query"},
		{originalName: "cursor", pyName: "cursor", location: "query"},
		{originalName: "empty_param", pyName: "empty_param", location: "query"},
	}, nil)
	req, err := tool.prepareRequest(map[string]any{
		"flag":        false,
		"offset":      float64(0),
		"cursor":      nil,
		"empty_param": "",
	})
	if err != nil {
		t.Fatal(err)
	}
	// False, 0 and "" are kept; nil is omitted.
	want := url.Values{"flag": {"false"}, "offset": {"0"}, "empty_param": {""}}
	if !maps.EqualFunc(req.query, want, slicesEqual) {
		t.Errorf("query = %v, want %v", req.query, want)
	}
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPrepareRequestQueryStyleAndExplode(t *testing.T) {
	objectProps := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{}, "priority": map[string]any{}, "active": map[string]any{},
		},
	}
	tests := []struct {
		name       string
		schemaType string
		style      string
		explode    *bool
		value      any
		want       url.Values
	}{
		{
			"object default explode", "object", "", nil,
			map[string]any{"status": "open", "priority": "P1"},
			url.Values{"status": {"open"}, "priority": {"P1"}},
		},
		{
			"object drops undeclared", "object", "", nil,
			map[string]any{"status": "open", "priority": "P1", "undeclared": "dropped"},
			url.Values{"status": {"open"}, "priority": {"P1"}},
		},
		{
			"object form not exploded", "object", "form", boolPtr(false),
			map[string]any{"status": "open", "priority": nil, "active": true},
			url.Values{"match": {"active,true,status,open"}},
		}, // keys sorted (Go cannot keep spec order)
		{
			"object form empty", "object", "form", boolPtr(false),
			map[string]any{"priority": nil},
			url.Values{},
		},
		{
			"deep object", "object", "deepObject", boolPtr(true),
			map[string]any{"status": "open", "priority": "P1"},
			url.Values{"match[status]": {"open"}, "match[priority]": {"P1"}},
		},
		{
			"array default explode", "array", "", nil,
			[]any{"a", nil, "b"},
			url.Values{"match": {"a", "b"}},
		},
		{
			"array form not exploded", "array", "form", boolPtr(false),
			[]any{"a", nil, "b"},
			url.Values{"match": {"a,b"}},
		},
		{"array all nil", "array", "form", boolPtr(false), []any{nil}, url.Values{}},
		{
			"array space delimited", "array", "spaceDelimited", boolPtr(false),
			[]any{"a", "b"},
			url.Values{"match": {"a b"}},
		},
		{
			"array pipe delimited", "array", "pipeDelimited", boolPtr(false),
			[]any{"a", "b"},
			url.Values{"match": {"a|b"}},
		},
		{"string", "string", "form", boolPtr(false), "plain", url.Values{"match": {"plain"}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			schema := map[string]any{"type": tc.schemaType}
			if tc.schemaType == "object" {
				schema = objectProps
			}
			tool := testTool(sampleEndpoint, []*apiParameter{
				{originalName: "match", pyName: "match", location: "query", schema: schema, style: tc.style, explode: tc.explode},
			}, nil)
			req, err := tool.prepareRequest(map[string]any{"match": tc.value})
			if err != nil {
				t.Fatal(err)
			}
			if !maps.EqualFunc(req.query, tc.want, slicesEqual) {
				t.Errorf("query = %v, want %v", req.query, tc.want)
			}
		})
	}
}

func TestPrepareRequestQueryObjectWithoutProperties(t *testing.T) {
	for _, schema := range []map[string]any{
		{"type": "object"},
		{"type": "object", "additionalProperties": map[string]any{"type": "string"}},
	} {
		tool := testTool(sampleEndpoint, []*apiParameter{
			{originalName: "filter", pyName: "filter", location: "query", schema: schema},
		}, nil)
		req, err := tool.prepareRequest(map[string]any{"filter": map[string]any{"status": "open", "priority": "P1"}})
		if err != nil {
			t.Fatal(err)
		}
		want := url.Values{"status": {"open"}, "priority": {"P1"}}
		if !maps.EqualFunc(req.query, want, slicesEqual) {
			t.Errorf("schema %v: query = %v, want %v", schema, req.query, want)
		}
	}
}

func TestPrepareRequestExplodedObjectPreservesOtherQueryParams(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/items?api-version=2024-01-01", Method: "get"},
		[]*apiParameter{
			{originalName: "api_key", pyName: "api_key", location: "query"},
			{originalName: "limit", pyName: "limit", location: "query"},
			{originalName: "page", pyName: "page", location: "query"},
			{originalName: "filter", pyName: "filter", location: "query", schema: map[string]any{"type": "object"}},
		}, nil)
	req, err := tool.prepareRequest(map[string]any{
		"api_key": "secret",
		"limit":   float64(10),
		"filter": map[string]any{
			"status": "open", "api_key": "spoofed", "limit": float64(999), "page": float64(2), "api-version": "spoofed",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := url.Values{
		"api_key":     {"secret"},
		"limit":       {"10"},
		"status":      {"open"},
		"api-version": {"2024-01-01"},
	}
	if !maps.EqualFunc(req.query, want, slicesEqual) {
		t.Errorf("query = %v, want %v", req.query, want)
	}
}

func TestPrepareRequestArrayBody(t *testing.T) {
	tool := testTool(sampleEndpoint,
		[]*apiParameter{{originalName: "array", pyName: "array", location: "body"}},
		&requestBodyInfo{mediaType: "application/json", object: false})
	req, err := tool.prepareRequest(map[string]any{"array": []any{"item1", "item2"}})
	if err != nil {
		t.Fatal(err)
	}
	var body []any
	if err := json.Unmarshal(req.body, &body); err != nil {
		t.Fatalf("body: %v", err)
	}
	if len(body) != 2 || body[0] != "item1" || body[1] != "item2" {
		t.Errorf("json body = %v, want [item1 item2]", body)
	}
}

func TestPrepareRequestArrayBodyWithConflict(t *testing.T) {
	// A query param and an array body both named "array": the body is renamed
	// array_0 by the parser and still sent as the JSON payload.
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "test_op",
		Endpoint: sampleEndpoint,
		Operation: map[string]any{
			"operationId": "test_op",
			"parameters":  []any{map[string]any{"name": "array", "in": "query", "schema": map[string]any{"type": "string"}}},
			"requestBody": map[string]any{"content": map[string]any{"application/json": map[string]any{
				"schema": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req, err := tool.prepareRequest(map[string]any{"array": "query_val", "array_0": []any{"item1", "item2"}})
	if err != nil {
		t.Fatal(err)
	}
	var body []any
	if err := json.Unmarshal(req.body, &body); err != nil {
		t.Fatalf("body: %v", err)
	}
	if len(body) != 2 {
		t.Errorf("json body = %v, want 2 items", body)
	}
	if got := req.query.Get("array"); got != "query_val" {
		t.Errorf("query array = %q, want query_val", got)
	}
}

func TestPrepareRequestStringBody(t *testing.T) {
	tool := testTool(sampleEndpoint,
		[]*apiParameter{{originalName: "", pyName: "input_string", location: "body"}},
		&requestBodyInfo{mediaType: "text/plain", object: false})
	req, err := tool.prepareRequest(map[string]any{"input_string": "test_value"})
	if err != nil {
		t.Fatal(err)
	}
	if string(req.body) != "test_value" {
		t.Errorf("body = %q, want test_value", req.body)
	}
	if req.contentType != "text/plain" {
		t.Errorf("content type = %q", req.contentType)
	}
}

func TestPrepareRequestOctetStream(t *testing.T) {
	tool := testTool(sampleEndpoint,
		[]*apiParameter{{originalName: "", pyName: "data", location: "body"}},
		&requestBodyInfo{mediaType: "application/octet-stream", object: false})
	req, err := tool.prepareRequest(map[string]any{"data": "binary_data"})
	if err != nil {
		t.Fatal(err)
	}
	if string(req.body) != "binary_data" {
		t.Errorf("body = %q", req.body)
	}
	if req.contentType != "application/octet-stream" {
		t.Errorf("content type = %q", req.contentType)
	}
}

func TestPrepareRequestFormData(t *testing.T) {
	tool := testTool(sampleEndpoint,
		[]*apiParameter{{originalName: "key1", pyName: "key1", location: "body"}},
		&requestBodyInfo{mediaType: "application/x-www-form-urlencoded", object: true})
	req, err := tool.prepareRequest(map[string]any{"key1": "value1"})
	if err != nil {
		t.Fatal(err)
	}
	form, err := url.ParseQuery(string(req.body))
	if err != nil {
		t.Fatal(err)
	}
	if form.Get("key1") != "value1" {
		t.Errorf("form = %v, want key1=value1", form)
	}
	if req.contentType != "application/x-www-form-urlencoded" {
		t.Errorf("content type = %q", req.contentType)
	}
}

func TestPrepareRequestMultipart(t *testing.T) {
	tool := testTool(sampleEndpoint,
		[]*apiParameter{{originalName: "file1", pyName: "file1", location: "body"}},
		&requestBodyInfo{mediaType: "multipart/form-data", object: true})
	req, err := tool.prepareRequest(map[string]any{"file1": "file_content"})
	if err != nil {
		t.Fatal(err)
	}
	// The boundary-bearing Content-Type is set by the multipart writer, not the
	// bare media type, so net/http does not have to generate one.
	if req.contentType == "multipart/form-data" || req.contentType == "" {
		t.Errorf("content type = %q, want one carrying a boundary", req.contentType)
	}
	if !req.multipart {
		t.Error("multipart flag not set")
	}
}

func TestPrepareRequestPathParam(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/test/{user_id}", Method: "get"},
		[]*apiParameter{{originalName: "user_id", pyName: "user_id", location: "path"}}, nil)
	req, err := tool.prepareRequest(map[string]any{"user_id": "123"})
	if err != nil {
		t.Fatal(err)
	}
	if req.url != "https://example.com/test/123" {
		t.Errorf("url = %q", req.url)
	}
}

func TestPrepareRequestPathParamPercentEncoded(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/users/{user_id}/messages", Method: "get"},
		[]*apiParameter{{originalName: "user_id", pyName: "user_id", location: "path"}}, nil)
	tests := []struct {
		value, want string
	}{
		{"foo/bar", "https://example.com/users/foo%2Fbar/messages"},
		{"me?impersonate=other#", "https://example.com/users/me%3Fimpersonate%3Dother%23/messages"},
		{"file..txt", "https://example.com/users/file..txt/messages"},
		{".gitignore", "https://example.com/users/.gitignore/messages"},
		{"ok.txt", "https://example.com/users/ok.txt/messages"},
	}
	for _, tc := range tests {
		req, err := tool.prepareRequest(map[string]any{"user_id": tc.value})
		if err != nil {
			t.Fatalf("%q: %v", tc.value, err)
		}
		if req.url != tc.want {
			t.Errorf("%q: url = %q, want %q", tc.value, req.url, tc.want)
		}
		if len(req.query) != 0 {
			t.Errorf("%q: query = %v, want empty", tc.value, req.query)
		}
	}
}

func TestPrepareRequestPathParamRejectsDotSegments(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/users/{user_id}/messages", Method: "get"},
		[]*apiParameter{{originalName: "user_id", pyName: "user_id", location: "path"}}, nil)
	for _, value := range []string{".", "..", "../../admin/v1/tenants", "foo/../bar", `..\admin`, "./x"} {
		_, err := tool.prepareRequest(map[string]any{"user_id": value})
		var invalid *inputValidationError
		if !errors.As(err, &invalid) {
			t.Errorf("%q: error = %v, want inputValidationError", value, err)
		}
	}
}

func TestPrepareRequestMissingRequiredPathParam(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/users/{user_id}/messages", Method: "get"},
		[]*apiParameter{{originalName: "user_id", pyName: "user_id", location: "path", required: true}}, nil)
	_, err := tool.prepareRequest(map[string]any{})
	var invalid *inputValidationError
	if !errors.As(err, &invalid) {
		t.Fatalf("error = %v, want inputValidationError", err)
	}
	if got := invalid.Error(); !strings.Contains(got, "user_id") {
		t.Errorf("error = %q, want it to name user_id", got)
	}
}

func TestPrepareRequestRequiredParamDefault(t *testing.T) {
	tool := testTool(
		Endpoint{BaseURL: "https://example.com", Path: "/users/{userId}/messages", Method: "get"},
		[]*apiParameter{{originalName: "userId", pyName: "user_id", location: "path", required: true, schema: map[string]any{"type": "string", "default": "me"}}}, nil)
	req, err := tool.prepareRequest(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if req.url != "https://example.com/users/me/messages" {
		t.Errorf("url = %q, want the default applied", req.url)
	}
}

func TestPrepareRequestHeaderAndCookie(t *testing.T) {
	tool := testTool(sampleEndpoint, []*apiParameter{
		{originalName: "X-Custom-Header", pyName: "x_custom_header", location: "header"},
		{originalName: "session_id", pyName: "session_id", location: "cookie"},
	}, nil)
	req, err := tool.prepareRequest(map[string]any{"x_custom_header": "header_value", "session_id": "cookie_value"})
	if err != nil {
		t.Fatal(err)
	}
	if req.headers["X-Custom-Header"] != "header_value" {
		t.Errorf("header = %q", req.headers["X-Custom-Header"])
	}
	if req.cookies["session_id"] != "cookie_value" {
		t.Errorf("cookie = %q", req.cookies["session_id"])
	}
}

func TestPrepareRequestSerializesHeaderAndCookieValues(t *testing.T) {
	tests := []struct {
		value        any
		wantHeader   string
		wantPathPart string
	}{
		{float64(50), "50", "50"},
		{1.5, "1.5", "1.5"},
		{true, "true", "true"},
		{false, "false", "false"},
		{[]any{"a", "b", float64(3)}, "a,b,3", "a%2Cb%2C3"},
		{[]any{true, false}, "true,false", "true%2Cfalse"},
		{[]any{float64(1), nil, true}, "1,true", "1%2Ctrue"},
		{"already-a-string", "already-a-string", "already-a-string"},
	}
	for _, tc := range tests {
		tool := testTool(
			Endpoint{BaseURL: "https://example.com", Path: "/items/{item_filter}", Method: "get"},
			[]*apiParameter{
				{originalName: "item_filter", pyName: "item_filter", location: "path"},
				{originalName: "X-Page-Size", pyName: "x_page_size", location: "header"},
				{originalName: "page_size", pyName: "page_size", location: "cookie"},
			}, nil)
		req, err := tool.prepareRequest(map[string]any{"item_filter": tc.value, "x_page_size": tc.value, "page_size": tc.value})
		if err != nil {
			t.Fatalf("%v: %v", tc.value, err)
		}
		if req.url != "https://example.com/items/"+tc.wantPathPart {
			t.Errorf("%v: url = %q, want path part %q", tc.value, req.url, tc.wantPathPart)
		}
		if req.headers["X-Page-Size"] != tc.wantHeader {
			t.Errorf("%v: header = %q, want %q", tc.value, req.headers["X-Page-Size"], tc.wantHeader)
		}
		if req.cookies["page_size"] != tc.wantHeader {
			t.Errorf("%v: cookie = %q, want %q", tc.value, req.cookies["page_size"], tc.wantHeader)
		}
	}
}

func TestPrepareRequestOmitsNoneHeaderAndCookie(t *testing.T) {
	tool := testTool(sampleEndpoint, []*apiParameter{
		{originalName: "X-Page-Size", pyName: "x_page_size", location: "header"},
		{originalName: "page_size", pyName: "page_size", location: "cookie"},
	}, nil)
	req, err := tool.prepareRequest(map[string]any{"x_page_size": nil, "page_size": nil})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := req.headers["X-Page-Size"]; ok {
		t.Error("nil header should be omitted")
	}
	if _, ok := req.cookies["page_size"]; ok {
		t.Error("nil cookie should be omitted")
	}
}

func TestPrepareRequestUnknownParameter(t *testing.T) {
	tool := testTool(sampleEndpoint, []*apiParameter{
		{originalName: "known_param", pyName: "known_param", location: "query"},
	}, nil)
	req, err := tool.prepareRequest(map[string]any{"known_param": "value", "unknown_param": "unknown"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := req.query["unknown_param"]; ok {
		t.Error("unknown parameter should be ignored")
	}
}

func TestPrepareRequestBaseURLHandling(t *testing.T) {
	noBase := testTool(Endpoint{BaseURL: "", Path: "/no_base", Method: "get"}, nil, nil)
	req, err := noBase.prepareRequest(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if req.url != "/no_base" {
		t.Errorf("url = %q, want /no_base", req.url)
	}

	trailing := testTool(Endpoint{BaseURL: "https://example.com/", Path: "/trailing", Method: "get"}, nil, nil)
	req, err = trailing.prepareRequest(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if req.url != "https://example.com/trailing" {
		t.Errorf("url = %q, want the trailing slash removed", req.url)
	}
}

func TestPrepareRequestMultipleMimeTypes(t *testing.T) {
	// The first media type of the request body is used.
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:     "test_op",
		Endpoint: sampleEndpoint,
		Operation: map[string]any{
			"operationId": "test_op",
			"requestBody": map[string]any{"content": map[string]any{
				"application/json": map[string]any{"schema": map[string]any{"type": "string"}},
				"text/plain":       map[string]any{"schema": map[string]any{"type": "string"}},
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if tool.body.mediaType != "application/json" {
		t.Errorf("media type = %q, want application/json (first)", tool.body.mediaType)
	}
}

func TestPrepareRequestEmbeddedQueryParams(t *testing.T) {
	tool := testTool(Endpoint{
		BaseURL: "https://integrations.googleapis.com",
		Path:    "/v2/projects/p/integrations/Flow:execute?triggerId=api_trigger/Flow#POST_files",
		Method:  "post",
	}, nil, nil)
	req, err := tool.prepareRequest(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if got := req.query.Get("triggerId"); got != "api_trigger/Flow" {
		t.Errorf("triggerId = %q", got)
	}
	if req.url != "https://integrations.googleapis.com/v2/projects/p/integrations/Flow:execute" {
		t.Errorf("url = %q, want query and fragment stripped", req.url)
	}
}

func TestPrepareRequestEmbeddedAndExplicitQueryParams(t *testing.T) {
	tool := testTool(Endpoint{BaseURL: "https://example.com", Path: "/api?embedded_key=embedded_val", Method: "get"},
		[]*apiParameter{{originalName: "explicit_key", pyName: "explicit_key", location: "query"}}, nil)
	req, err := tool.prepareRequest(map[string]any{"explicit_key": "explicit_val"})
	if err != nil {
		t.Fatal(err)
	}
	if req.query.Get("embedded_key") != "embedded_val" || req.query.Get("explicit_key") != "explicit_val" {
		t.Errorf("query = %v, want both keys", req.query)
	}
}

func TestPrepareRequestExplicitQueryParamPrecedence(t *testing.T) {
	tool := testTool(Endpoint{BaseURL: "https://example.com", Path: "/api?key=embedded", Method: "get"},
		[]*apiParameter{{originalName: "key", pyName: "key", location: "query"}}, nil)
	req, err := tool.prepareRequest(map[string]any{"key": "explicit"})
	if err != nil {
		t.Fatal(err)
	}
	if got := req.query.Get("key"); got != "explicit" {
		t.Errorf("key = %q, want explicit (explicit wins over embedded)", got)
	}
}

func TestPrepareRequestStripsFragmentOnly(t *testing.T) {
	tool := testTool(Endpoint{BaseURL: "https://example.com", Path: "/api#fragment", Method: "get"}, nil, nil)
	req, err := tool.prepareRequest(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if req.url != "https://example.com/api" {
		t.Errorf("url = %q, want fragment stripped", req.url)
	}
}
