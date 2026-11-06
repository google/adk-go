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

package bigquerytoolset

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// fakeBigQuery is a BigQuery REST API stand-in. It answers each request with
// the response registered for its path and records what it was sent.
type fakeBigQuery struct {
	t         *testing.T
	server    *httptest.Server
	responses map[string]fakeResponse

	mu       sync.Mutex
	requests []*http.Request
}

type fakeResponse struct {
	status int
	body   string
}

func newFakeBigQuery(t *testing.T, responses map[string]fakeResponse) *fakeBigQuery {
	t.Helper()
	f := &fakeBigQuery{t: t, responses: responses}
	f.server = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeBigQuery) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.requests = append(f.requests, r.Clone(context.Background()))
	f.mu.Unlock()

	key := r.Method + " " + r.URL.Path
	if pageToken := r.URL.Query().Get("pageToken"); pageToken != "" {
		key += "?pageToken=" + pageToken
	}
	resp, ok := f.responses[key]
	if !ok {
		f.t.Errorf("unexpected request %s", key)
		http.Error(w, `{"error":{"code":500,"message":"unexpected request"}}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if resp.status != 0 {
		w.WriteHeader(resp.status)
	}
	_, _ = w.Write([]byte(resp.body))
}

func (f *fakeBigQuery) lastRequest() *http.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) == 0 {
		f.t.Fatal("no request reached the fake BigQuery server")
	}
	return f.requests[len(f.requests)-1]
}

// clientOptions points the toolset at the fake server without credentials.
func (f *fakeBigQuery) clientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint(f.server.URL + "/bigquery/v2/"),
		option.WithoutAuthentication(),
	}
}

func newTestToolset(t *testing.T, cfg Config) tool.Toolset {
	t.Helper()
	ts, err := New(t.Context(), cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return ts
}

// runnable is what the LLM flow calls on a function tool.
type runnable interface {
	tool.Tool
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (map[string]any, error)
}

func toolByName(t *testing.T, ts tool.Toolset, name string) runnable {
	t.Helper()
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	for _, tl := range tools {
		if tl.Name() != name {
			continue
		}
		r, ok := tl.(runnable)
		if !ok {
			t.Fatalf("tool %q (%T) is not runnable", name, tl)
		}
		if _, ok := tl.(toolinternal.FunctionTool); !ok {
			t.Fatalf("tool %q (%T) is not a toolinternal.FunctionTool", name, tl)
		}
		return r
	}
	t.Fatalf("toolset has no tool %q", name)
	return nil
}

func newToolContext(t *testing.T, ctx context.Context, state map[string]any) agent.Context {
	t.Helper()
	service := session.InMemoryService()
	created, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", State: state})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Session: created.Session})
	return agent.NewToolContext(invCtx, "call-1", &session.EventActions{}, nil)
}

func run(t *testing.T, tl runnable, args map[string]any) map[string]any {
	t.Helper()
	got, err := tl.Run(newToolContext(t, t.Context(), nil), args)
	if err != nil {
		t.Fatalf("%s Run() error = %v", tl.Name(), err)
	}
	return got
}

func TestNew_RejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{
			name:    "application name with a space",
			cfg:     Config{ApplicationName: "my app"},
			wantErr: "ApplicationName",
		},
		{
			name:    "application name with a newline",
			cfg:     Config{ApplicationName: "my-app\n"},
			wantErr: "ApplicationName",
		},
		{
			name:    "external access token key with client credentials",
			cfg:     Config{ExternalAccessTokenKey: "token", ClientOptions: []option.ClientOption{option.WithAPIKey("key")}},
			wantErr: "ExternalAccessTokenKey",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(t.Context(), tc.cfg)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("New() error = %v, want an error mentioning %q", err, tc.wantErr)
			}
		})
	}
}

func TestToolset_ToolNames(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	var got []string
	for _, tl := range tools {
		got = append(got, tl.Name())
	}
	want := []string{"get_dataset_info", "list_dataset_ids"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("tool names mismatch (-want +got):\n%s", diff)
	}
	if ts.Name() != "bigquery_toolset" {
		t.Errorf("Name() = %q, want %q", ts.Name(), "bigquery_toolset")
	}
}

func TestToolset_FilterToolset(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := tool.FilterToolset(newTestToolset(t, Config{ClientOptions: f.clientOptions()}), tool.StringPredicate([]string{"list_dataset_ids"}))

	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "list_dataset_ids" {
		t.Errorf("filtered tools = %v, want only list_dataset_ids", tools)
	}
}

// TestToolset_Declarations checks the parameters the model sees, which must
// match adk-python's so prompts written for it work unchanged.
func TestToolset_Declarations(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	want := map[string][]string{
		"list_dataset_ids": {"project_id"},
		"get_dataset_info": {"project_id", "dataset_id"},
	}
	for name, wantRequired := range want {
		t.Run(name, func(t *testing.T) {
			decl := toolByName(t, ts, name).Declaration()
			schema := declarationSchema(t, decl)
			want := slices.Sorted(slices.Values(wantRequired))
			if diff := cmp.Diff(want, slices.Sorted(slices.Values(schema.Required))); diff != "" {
				t.Errorf("required mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(want, slices.Sorted(maps.Keys(schema.Properties))); diff != "" {
				t.Errorf("properties mismatch (-want +got):\n%s", diff)
			}
			for prop, p := range schema.Properties {
				if p.Type != "string" {
					t.Errorf("property %q type = %q, want string", prop, p.Type)
				}
			}
			if !strings.HasPrefix(decl.Description, descriptionFirstLine[name]) {
				t.Errorf("description = %q, want it to start with %q", firstLine(decl.Description), descriptionFirstLine[name])
			}
			if strings.HasSuffix(decl.Description, "\n") {
				t.Errorf("description ends with a newline")
			}
			if strings.Contains(decl.Description, "credentials (Credentials)") {
				t.Errorf("description mentions the credentials argument, which Go tools do not have")
			}
		})
	}
}

var descriptionFirstLine = map[string]string{
	"list_dataset_ids": "List BigQuery dataset ids in a Google Cloud project.",
	"get_dataset_info": "Get metadata information about a BigQuery dataset.",
}

func TestToolset_DefaultIDsMakeArgsOptional(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), DefaultProjectID: "my-project", DefaultDatasetID: "my_dataset"})

	schema := declarationSchema(t, toolByName(t, ts, "get_dataset_info").Declaration())
	if len(schema.Required) != 0 {
		t.Errorf("required = %v, want none", schema.Required)
	}
	wantDesc := map[string]string{
		"project_id": "Defaults to 'my-project' when not provided.",
		"dataset_id": "Defaults to 'my_dataset' when not provided.",
	}
	for prop, want := range wantDesc {
		if got := schema.Properties[prop].Description; got != want {
			t.Errorf("property %q description = %q, want %q", prop, got, want)
		}
	}
}

func TestToolset_DefaultIDsFillMissingArgs(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/my-project/datasets/my_dataset": {body: `{"id":"my-project:my_dataset"}`},
		"GET /bigquery/v2/projects/other/datasets/my_dataset":      {body: `{"id":"other:my_dataset"}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), DefaultProjectID: "my-project", DefaultDatasetID: "my_dataset"})
	getDataset := toolByName(t, ts, "get_dataset_info")

	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{name: "omitted", args: map[string]any{}, want: "my-project:my_dataset"},
		{name: "empty", args: map[string]any{"project_id": "", "dataset_id": ""}, want: "my-project:my_dataset"},
		{name: "model names another project", args: map[string]any{"project_id": "other"}, want: "other:my_dataset"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := run(t, getDataset, tc.args)
			if got["id"] != tc.want {
				t.Errorf("Run() = %v, want id %q", got, tc.want)
			}
		})
	}
}

func TestToolset_MissingRequiredArgIsRejected(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	_, err := toolByName(t, ts, "get_dataset_info").Run(newToolContext(t, t.Context(), nil), map[string]any{"project_id": "p"})
	if err == nil {
		t.Fatal("Run() without dataset_id succeeded, want a validation error")
	}
}

func TestListDatasetIDs(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets":                {body: `{"datasets":[{"datasetReference":{"projectId":"p","datasetId":"a"}}],"nextPageToken":"next"}`},
		"GET /bigquery/v2/projects/p/datasets?pageToken=next": {body: `{"datasets":[{"datasetReference":{"projectId":"p","datasetId":"b"}}]}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := run(t, toolByName(t, ts, "list_dataset_ids"), map[string]any{"project_id": "p"})

	want := map[string]any{"result": []any{"a", "b"}}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
}

func TestListDatasetIDs_EmptyProject(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets": {body: `{}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := run(t, toolByName(t, ts, "list_dataset_ids"), map[string]any{"project_id": "p"})

	// An empty list, not null, so the model sees there are no datasets.
	want := map[string]any{"result": []any{}}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
}

// TestGetInfoTools_ReturnRESTResource checks that the info tools return the
// resource as the BigQuery REST API represents it, like adk-python's
// to_api_repr and _properties, including fields this package does not model.
func TestGetInfoTools_ReturnRESTResource(t *testing.T) {
	datasetBody := `{"kind":"bigquery#dataset","id":"p:d","datasetReference":{"projectId":"p","datasetId":"d"},"creationTime":"1640891845643","location":"US","maxTimeTravelHours":"168"}`
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets/d": {body: datasetBody},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	tests := []struct {
		tool string
		args map[string]any
		want string
	}{
		{tool: "get_dataset_info", args: map[string]any{"project_id": "p", "dataset_id": "d"}, want: datasetBody},
	}
	for _, tc := range tests {
		t.Run(tc.tool, func(t *testing.T) {
			got := run(t, toolByName(t, ts, tc.tool), tc.args)
			if diff := cmp.Diff(decodeJSON(t, tc.want), roundTrip(t, got)); diff != "" {
				t.Errorf("Run() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestTools_APIErrorIsReturnedToModel checks that a failed BigQuery call is
// reported to the model the way adk-python reports it, so the model can
// correct the call.
func TestTools_APIErrorIsReturnedToModel(t *testing.T) {
	notFound := fakeResponse{status: http.StatusNotFound, body: `{"error":{"code":404,"message":"Not found: Dataset p:missing","errors":[{"reason":"notFound","message":"Not found: Dataset p:missing"}]}}`}
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets/missing": notFound,
		"GET /bigquery/v2/projects/missing/datasets":   notFound,
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	tests := []struct {
		tool string
		args map[string]any
	}{
		{tool: "list_dataset_ids", args: map[string]any{"project_id": "missing"}},
		{tool: "get_dataset_info", args: map[string]any{"project_id": "p", "dataset_id": "missing"}},
	}
	for _, tc := range tests {
		t.Run(tc.tool, func(t *testing.T) {
			got := run(t, toolByName(t, ts, tc.tool), tc.args)
			if got["status"] != "ERROR" {
				t.Errorf("status = %v, want ERROR", got["status"])
			}
			details, _ := got["error_details"].(string)
			if !strings.Contains(details, "Not found") {
				t.Errorf("error_details = %q, want the BigQuery error message", details)
			}
		})
	}
}

// TestTools_CanceledContextIsAnError checks that a canceled invocation is not
// reported to the model as a BigQuery failure.
func TestTools_CanceledContextIsAnError(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets": {body: `{}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	ctx, cancel := context.WithCancel(t.Context())
	toolCtx := newToolContext(t, ctx, nil)
	cancel()

	_, err := toolByName(t, ts, "list_dataset_ids").Run(toolCtx, map[string]any{"project_id": "p"})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Run() error = %v, want context.Canceled", err)
	}
}

func TestTools_UserAgent(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets/d": {body: `{}`},
	})

	tests := []struct {
		name    string
		appName string
		want    []string
	}{
		{name: "default", want: []string{"adk-bigquery-tool", "google-adk/", "get_dataset_info"}},
		{name: "application name", appName: "my-agent", want: []string{"adk-bigquery-tool", "google-adk/", "my-agent", "get_dataset_info"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), ApplicationName: tc.appName})
			run(t, toolByName(t, ts, "get_dataset_info"), map[string]any{"project_id": "p", "dataset_id": "d"})

			ua := f.lastRequest().UserAgent()
			last := -1
			for _, part := range tc.want {
				i := strings.Index(ua, part)
				if i <= last {
					t.Fatalf("User-Agent = %q, want %q in this order", ua, tc.want)
				}
				last = i
			}
		})
	}
}

func TestTools_ExternalAccessToken(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets": {body: `{}`},
	})
	ts := newTestToolset(t, Config{
		ClientOptions:          []option.ClientOption{option.WithEndpoint(f.server.URL + "/bigquery/v2/")},
		ExternalAccessTokenKey: "bq_token",
	})
	listDatasets := toolByName(t, ts, "list_dataset_ids")

	for _, token := range []string{"token-a", "token-b"} {
		got, err := listDatasets.Run(newToolContext(t, t.Context(), map[string]any{"bq_token": token}), map[string]any{"project_id": "p"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if got["status"] == "ERROR" {
			t.Fatalf("Run() = %v, want success", got)
		}
		if auth := f.lastRequest().Header.Get("Authorization"); auth != "Bearer "+token {
			t.Errorf("Authorization = %q, want %q", auth, "Bearer "+token)
		}
	}
}

func TestTools_ExternalAccessTokenMissing(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{
		ClientOptions:          []option.ClientOption{option.WithEndpoint(f.server.URL + "/bigquery/v2/")},
		ExternalAccessTokenKey: "bq_token",
	})

	for _, state := range []map[string]any{nil, {"bq_token": ""}, {"bq_token": 42}} {
		got, err := toolByName(t, ts, "list_dataset_ids").Run(newToolContext(t, t.Context(), state), map[string]any{"project_id": "p"})
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		details, _ := got["error_details"].(string)
		if got["status"] != "ERROR" || !strings.Contains(details, "bq_token") {
			t.Errorf("Run() with state %v = %v, want an ERROR naming the state key", state, got)
		}
	}
}

func declarationSchema(t *testing.T, decl *genai.FunctionDeclaration) *genai.Schema {
	t.Helper()
	if decl.Parameters != nil {
		return decl.Parameters
	}
	raw, err := json.Marshal(decl.ParametersJsonSchema)
	if err != nil {
		t.Fatalf("marshal ParametersJsonSchema: %v", err)
	}
	var s struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			Type        string `json:"type"`
			Description string `json:"description"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("unmarshal ParametersJsonSchema: %v", err)
	}
	out := &genai.Schema{Required: s.Required, Properties: map[string]*genai.Schema{}}
	for name, p := range s.Properties {
		out.Properties[name] = &genai.Schema{Type: genai.Type(p.Type), Description: p.Description}
	}
	return out
}

func roundTrip(t *testing.T, v any) any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	return decodeJSON(t, string(raw))
}

func decodeJSON(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("unmarshal %s: %v", s, err)
	}
	return v
}

func firstLine(s string) string {
	line, _, _ := strings.Cut(s, "\n")
	return line
}

// TestTools_InvalidIDsAreRejected checks that an id that would change the
// request path never reaches BigQuery.
func TestTools_InvalidIDsAreRejected(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	tests := []struct {
		tool string
		args map[string]any
	}{
		{tool: "list_dataset_ids", args: map[string]any{"project_id": ""}},
		{tool: "list_dataset_ids", args: map[string]any{"project_id": ".."}},
		{tool: "get_dataset_info", args: map[string]any{"project_id": "p", "dataset_id": "."}},
		{tool: "get_dataset_info", args: map[string]any{"project_id": "..", "dataset_id": "d"}},
		{tool: "get_dataset_info", args: map[string]any{"project_id": "p", "dataset_id": ""}},
	}
	for _, tc := range tests {
		t.Run(tc.tool, func(t *testing.T) {
			got := run(t, toolByName(t, ts, tc.tool), tc.args)
			if got["status"] != "ERROR" {
				t.Errorf("Run(%v) = %v, want an ERROR result", tc.args, got)
			}
			if details, _ := got["error_details"].(string); strings.Contains(details, `".."`) || strings.Contains(details, `"."`) {
				t.Errorf("error_details = %q, want it to name the argument, not echo its value", details)
			}
		})
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.requests) != 0 {
		t.Errorf("%d requests reached BigQuery, want none", len(f.requests))
	}
}

// TestTools_IDsAreEscaped checks that an id with a slash stays one path
// segment.
func TestTools_IDsAreEscaped(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets/a/b": {body: `{}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	run(t, toolByName(t, ts, "get_dataset_info"), map[string]any{"project_id": "p", "dataset_id": "a/b"})

	if got := f.lastRequest().URL.EscapedPath(); got != "/bigquery/v2/projects/p/datasets/a%2Fb" {
		t.Errorf("request path = %q, want the dataset id escaped as one segment", got)
	}
}

func TestListDatasetIDs_RepeatedPageTokenStops(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets":                {body: `{"datasets":[],"nextPageToken":"same"}`},
		"GET /bigquery/v2/projects/p/datasets?pageToken=same": {body: `{"datasets":[],"nextPageToken":"same"}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	// Without the guard the tool would page forever; the deadline turns that
	// into a failure.
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	got, err := toolByName(t, ts, "list_dataset_ids").Run(newToolContext(t, ctx, nil), map[string]any{"project_id": "p"})
	if err != nil {
		t.Fatalf("Run() error = %v, want an ERROR result", err)
	}
	if got["status"] != "ERROR" {
		t.Errorf("Run() = %v, want an ERROR result", got)
	}
}

func TestNew_ClientError(t *testing.T) {
	_, err := New(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithAuthCredentialsFile(option.ServiceAccount, "/nonexistent/key.json")}})
	if err == nil || !strings.Contains(err.Error(), "creating the BigQuery HTTP client") {
		t.Fatalf("New() error = %v, want the client creation error", err)
	}
}

func TestNew_InvalidEndpoint(t *testing.T) {
	_, err := New(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithEndpoint("http://[::1/"), option.WithoutAuthentication()}})
	if err == nil || !strings.Contains(err.Error(), "invalid BigQuery endpoint") {
		t.Fatalf("New() error = %v, want an invalid endpoint error", err)
	}
}

func TestNewTool_SchemaError(t *testing.T) {
	_, err := newTool(&bigQuery{}, "bad", "", func(agent.Context, struct{ C chan int }) (any, error) { return nil, nil })
	if err == nil || !strings.Contains(err.Error(), "bad") {
		t.Fatalf("newTool() error = %v, want an error naming the tool", err)
	}
}

func TestTools_EndpointWithoutTrailingSlash(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets": {body: `{}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: []option.ClientOption{option.WithEndpoint(f.server.URL + "/bigquery/v2"), option.WithoutAuthentication()}})

	got := run(t, toolByName(t, ts, "list_dataset_ids"), map[string]any{"project_id": "p"})
	if got["status"] == "ERROR" {
		t.Errorf("Run() = %v, want success", got)
	}
}

func TestTools_TransportErrors(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets/d": {body: `not json`},
	})

	tests := []struct {
		name string
		opts []option.ClientOption
		want string
	}{
		{name: "invalid response", opts: f.clientOptions(), want: "decoding the BigQuery response"},
		{name: "unreachable endpoint", opts: []option.ClientOption{option.WithEndpoint("http://127.0.0.1:1/"), option.WithoutAuthentication()}, want: "connect"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ts := newTestToolset(t, Config{ClientOptions: tc.opts})
			got := run(t, toolByName(t, ts, "get_dataset_info"), map[string]any{"project_id": "p", "dataset_id": "d"})
			details, _ := got["error_details"].(string)
			if got["status"] != "ERROR" || !strings.Contains(details, tc.want) {
				t.Errorf("Run() = %v, want an ERROR containing %q", got, tc.want)
			}
			// The request URL holds the tool's arguments.
			if strings.Contains(details, "/projects/") {
				t.Errorf("error_details = %q, want no request URL", details)
			}
		})
	}
}

func TestListDatasetIDs_PageTokenCycleStops(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{
		"GET /bigquery/v2/projects/p/datasets":             {body: `{"nextPageToken":"a"}`},
		"GET /bigquery/v2/projects/p/datasets?pageToken=a": {body: `{"nextPageToken":"b"}`},
		"GET /bigquery/v2/projects/p/datasets?pageToken=b": {body: `{"nextPageToken":"a"}`},
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	got, err := toolByName(t, ts, "list_dataset_ids").Run(newToolContext(t, ctx, nil), map[string]any{"project_id": "p"})
	if err != nil {
		t.Fatalf("Run() error = %v, want an ERROR result", err)
	}
	if got["status"] != "ERROR" {
		t.Errorf("Run() = %v, want an ERROR result", got)
	}
}

// failingStateSession is a session whose state cannot be read.
type failingStateSession struct {
	session.Session
}

func (s failingStateSession) State() session.State { return failingState{s.Session.State()} }

type failingState struct {
	session.State
}

func (failingState) Get(string) (any, error) { return nil, errors.New("state backend unavailable") }

func TestTools_ExternalAccessTokenStateError(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{
		ClientOptions:          []option.ClientOption{option.WithEndpoint(f.server.URL + "/bigquery/v2/")},
		ExternalAccessTokenKey: "bq_token",
	})
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: failingStateSession{created.Session}})
	toolCtx := agent.NewToolContext(invCtx, "call-1", &session.EventActions{}, nil)

	got, err := toolByName(t, ts, "list_dataset_ids").Run(toolCtx, map[string]any{"project_id": "p"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	details, _ := got["error_details"].(string)
	if !strings.Contains(details, "state backend unavailable") {
		t.Errorf("error_details = %q, want the state error, not a missing token", details)
	}
}

// TestTools_ExternalAccessTokenKeepsClientTimeout checks that the per-call
// client of the external token path keeps the settings of the client it
// wraps.
func TestTools_ExternalAccessTokenKeepsClientTimeout(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	t.Cleanup(slow.Close)
	ts := newTestToolset(t, Config{
		ClientOptions: []option.ClientOption{
			option.WithEndpoint(slow.URL + "/bigquery/v2/"),
			option.WithHTTPClient(&http.Client{Timeout: 50 * time.Millisecond}),
		},
		ExternalAccessTokenKey: "bq_token",
	})

	start := time.Now()
	got, err := toolByName(t, ts, "list_dataset_ids").Run(newToolContext(t, t.Context(), map[string]any{"bq_token": "token"}), map[string]any{"project_id": "p"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got["status"] != "ERROR" {
		t.Errorf("Run() = %v, want an ERROR result from the timeout", got)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("Run() took %v, want the client's 50ms timeout to apply", elapsed)
	}
}
