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

package spannertoolset

import (
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const (
	testDB      = "/v1/projects/p/instances/i/databases/d"
	testSession = "projects/p/instances/i/databases/d/sessions/s1"
)

// fakeSpanner is a Spanner REST API stand-in. It answers the database lookup
// with its dialect, creates and deletes one session, and answers each query
// with the first registered response whose key the SQL contains.
type fakeSpanner struct {
	t       *testing.T
	server  *httptest.Server
	dialect string
	// queries maps a part of a query's SQL to the executeStreamingSql
	// response body for it.
	queries map[string]string
	// override, when set, answers a request before the defaults do.
	override func(w http.ResponseWriter, r *http.Request, body map[string]any) bool

	mu       sync.Mutex
	requests []fakeRequest
}

type fakeRequest struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

func newFakeSpanner(t *testing.T, queries map[string]string) *fakeSpanner {
	t.Helper()
	f := &fakeSpanner{t: t, dialect: "GOOGLE_STANDARD_SQL", queries: queries}
	f.server = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeSpanner) serve(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		f.t.Errorf("reading the request body: %v", err)
	}
	var body map[string]any
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &body); err != nil {
			f.t.Errorf("decoding the request body: %v", err)
		}
	}
	f.mu.Lock()
	f.requests = append(f.requests, fakeRequest{method: r.Method, path: r.URL.EscapedPath(), header: r.Header.Clone(), body: body})
	f.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	if f.override != nil && f.override(w, r, body) {
		return
	}
	switch {
	case r.Method == http.MethodGet && r.URL.Path == testDB:
		writeJSON(w, map[string]any{"name": "projects/p/instances/i/databases/d", "databaseDialect": f.dialect})
	case r.Method == http.MethodPost && r.URL.Path == testDB+"/sessions":
		writeJSON(w, map[string]any{"name": testSession})
	case r.Method == http.MethodDelete && r.URL.Path == "/v1/"+testSession:
		writeJSON(w, map[string]any{})
	case r.Method == http.MethodPost && r.URL.Path == "/v1/"+testSession+":executeStreamingSql":
		sql, _ := body["sql"].(string)
		for _, key := range sortedKeys(f.queries) {
			if strings.Contains(sql, key) {
				_, _ = io.WriteString(w, f.queries[key])
				return
			}
		}
		f.t.Errorf("unexpected query %q", sql)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"unexpected query"}}`)
	default:
		f.t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"code":500,"message":"unexpected request"}}`)
	}
}

// sortedKeys returns the keys of m, longest first, so a specific key wins over
// one it contains.
func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b string) int { return len(b) - len(a) })
	return keys
}

func writeJSON(w http.ResponseWriter, v any) {
	_ = json.NewEncoder(w).Encode(v)
}

func (f *fakeSpanner) recorded() []fakeRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

// queryBodies returns the body of each executeStreamingSql request, in order.
func (f *fakeSpanner) queryBodies() []map[string]any {
	var out []map[string]any
	for _, r := range f.recorded() {
		if strings.HasSuffix(r.path, ":executeStreamingSql") {
			out = append(out, r.body)
		}
	}
	return out
}

// requestLine returns "METHOD path" for each request, in order.
func (f *fakeSpanner) requestLines() []string {
	var out []string
	for _, r := range f.recorded() {
		out = append(out, r.method+" "+r.path)
	}
	return out
}

func (f *fakeSpanner) clientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint(f.server.URL + "/v1/"),
		option.WithoutAuthentication(),
	}
}

// resultStream is an executeStreamingSql response with the given columns,
// all of one type code, and values, in one partial result set.
func resultStream(columns []string, code string, values ...any) string {
	fields := make([]any, len(columns))
	for i, c := range columns {
		fields[i] = map[string]any{"name": c, "type": map[string]any{"code": code}}
	}
	b, _ := json.Marshal([]any{map[string]any{
		"metadata": map[string]any{"rowType": map[string]any{"fields": fields}},
		"values":   orEmpty(values),
	}})
	return string(b)
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

// roundTrip passes v through JSON, as the function tool does with a result.
func roundTrip(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	return out
}

var dbArgs = map[string]any{"project_id": "p", "instance_id": "i", "database_id": "d"}

func withArgs(extra map[string]any) map[string]any {
	return merge(dbArgs, extra)
}

// merge returns a new map with the entries of a and then b.
func merge(a, b map[string]any) map[string]any {
	out := maps.Clone(a)
	maps.Copy(out, b)
	return out
}

func wantError(t *testing.T, got map[string]any, contains string) {
	t.Helper()
	details, _ := got["error_details"].(string)
	if got["status"] != "ERROR" || !strings.Contains(details, contains) {
		t.Fatalf("result = %v, want an error result containing %q", got, contains)
	}
}

func TestNew_RejectsInvalidConfig(t *testing.T) {
	validStore := VectorStoreConfig{
		ProjectID: "p", InstanceID: "i", DatabaseID: "d", TableName: "t",
		ContentColumn: "c", EmbeddingColumn: "e", VectorLength: 3, EmbeddingModel: "m",
	}
	withStore := func(edit func(*VectorStoreConfig)) Config {
		vs := validStore
		edit(&vs)
		return Config{VectorStore: &vs}
	}
	tests := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{"negative max rows", Config{MaxQueryResultRows: -1}, "MaxQueryResultRows"},
		{"unknown result mode", Config{QueryResultMode: "rows"}, "QueryResultMode"},
		{"vector store without project", withStore(func(v *VectorStoreConfig) { v.ProjectID = "" }), "ProjectID"},
		{"vector store without instance", withStore(func(v *VectorStoreConfig) { v.InstanceID = "" }), "InstanceID"},
		{"vector store without database", withStore(func(v *VectorStoreConfig) { v.DatabaseID = "" }), "DatabaseID"},
		{"vector store without table", withStore(func(v *VectorStoreConfig) { v.TableName = "" }), "TableName"},
		{"vector store without content column", withStore(func(v *VectorStoreConfig) { v.ContentColumn = "" }), "ContentColumn"},
		{"vector store without embedding column", withStore(func(v *VectorStoreConfig) { v.EmbeddingColumn = "" }), "EmbeddingColumn"},
		{"vector store without model", withStore(func(v *VectorStoreConfig) { v.EmbeddingModel = "" }), "EmbeddingModel"},
		{"zero vector length", withStore(func(v *VectorStoreConfig) { v.VectorLength = 0 }), "VectorLength"},
		{"negative vector length", withStore(func(v *VectorStoreConfig) { v.VectorLength = -1 }), "VectorLength"},
		{"negative top k", withStore(func(v *VectorStoreConfig) { v.TopK = -1 }), "TopK"},
		{"negative leaves", withStore(func(v *VectorStoreConfig) { v.NumLeavesToSearch = -1 }), "NumLeavesToSearch"},
		{"unknown distance type", withStore(func(v *VectorStoreConfig) { v.DistanceType = "MANHATTAN" }), "distance type"},
		{"unknown algorithm", withStore(func(v *VectorStoreConfig) { v.NearestNeighborsAlgorithm = "BRUTE" }), "NearestNeighborsAlgorithm"},
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

func TestNew_ClientError(t *testing.T) {
	_, err := New(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithAuthCredentialsFile(option.ServiceAccount, "/nonexistent/key.json")}})
	if err == nil || !strings.Contains(err.Error(), "creating the Spanner HTTP client") {
		t.Fatalf("New() error = %v, want a client creation error", err)
	}
}

func TestNew_InvalidEndpoint(t *testing.T) {
	_, err := New(t.Context(), Config{ClientOptions: []option.ClientOption{option.WithEndpoint("http://[::1/"), option.WithoutAuthentication()}})
	if err == nil || !strings.Contains(err.Error(), "invalid Spanner endpoint") {
		t.Fatalf("New() error = %v, want an endpoint error", err)
	}
}

func toolNames(t *testing.T, ts tool.Toolset) []string {
	t.Helper()
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}
	var names []string
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	return names
}

func TestToolset_ToolNames(t *testing.T) {
	f := newFakeSpanner(t, nil)
	base := []string{
		"spanner_list_table_names",
		"spanner_list_table_indexes",
		"spanner_list_table_index_columns",
		"spanner_list_named_schemas",
		"spanner_get_table_schema",
		"spanner_execute_sql",
		"spanner_similarity_search",
	}

	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	if diff := cmp.Diff(base, toolNames(t, ts)); diff != "" {
		t.Errorf("tool names mismatch (-want +got):\n%s", diff)
	}
	if ts.Name() != "spanner_toolset" {
		t.Errorf("Name() = %q, want %q", ts.Name(), "spanner_toolset")
	}

	withStore := newTestToolset(t, Config{ClientOptions: f.clientOptions(), VectorStore: &VectorStoreConfig{
		ProjectID: "p", InstanceID: "i", DatabaseID: "d", TableName: "t",
		ContentColumn: "c", EmbeddingColumn: "e", VectorLength: 3, EmbeddingModel: "m",
	}})
	want := append(slices.Clone(base), "spanner_vector_store_similarity_search")
	if diff := cmp.Diff(want, toolNames(t, withStore)); diff != "" {
		t.Errorf("tool names with a vector store mismatch (-want +got):\n%s", diff)
	}
}

func TestToolset_ToolsReturnsACopy(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	first, _ := ts.Tools(nil)
	first[0] = nil
	second, _ := ts.Tools(nil)
	if second[0] == nil {
		t.Error("Tools() returned the toolset's own slice")
	}
}

func TestToolset_FilterToolset(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := tool.FilterToolset(newTestToolset(t, Config{ClientOptions: f.clientOptions()}), tool.AllowedToolsPredicate([]string{"spanner_list_table_names"}))
	if diff := cmp.Diff([]string{"spanner_list_table_names"}, toolNames(t, ts)); diff != "" {
		t.Errorf("filtered tool names mismatch (-want +got):\n%s", diff)
	}
}

// TestToolset_Declarations checks the parameters the model sees, which must
// match adk-python's so prompts written for it work unchanged.
func TestToolset_Declarations(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), VectorStore: &VectorStoreConfig{
		ProjectID: "p", InstanceID: "i", DatabaseID: "d", TableName: "t",
		ContentColumn: "c", EmbeddingColumn: "e", VectorLength: 3, EmbeddingModel: "m",
	}})
	db := []string{"database_id", "instance_id", "project_id"}
	tests := []struct {
		tool         string
		wantRequired []string
		wantOptional []string
	}{
		{"spanner_list_table_names", db, []string{"named_schema"}},
		{"spanner_list_table_indexes", append(slices.Clone(db), "table_id"), nil},
		{"spanner_list_table_index_columns", append(slices.Clone(db), "table_id"), nil},
		{"spanner_list_named_schemas", db, nil},
		{"spanner_get_table_schema", append(slices.Clone(db), "table_name"), []string{"named_schema"}},
		{"spanner_execute_sql", append(slices.Clone(db), "query"), nil},
		{"spanner_similarity_search", append(slices.Clone(db), "columns", "embedding_column_to_search", "embedding_options", "query", "table_name"), []string{"additional_filter", "search_options"}},
		{"spanner_vector_store_similarity_search", []string{"query"}, nil},
	}
	for _, tc := range tests {
		t.Run(tc.tool, func(t *testing.T) {
			decl := toolByName(t, ts, tc.tool).Declaration()
			schema, ok := decl.ParametersJsonSchema.(map[string]any)
			if !ok {
				b, _ := json.Marshal(decl.ParametersJsonSchema)
				_ = json.Unmarshal(b, &schema)
			}
			required := toStrings(schema["required"])
			slices.Sort(required)
			want := slices.Clone(tc.wantRequired)
			slices.Sort(want)
			if diff := cmp.Diff(want, required); diff != "" {
				t.Errorf("required parameters mismatch (-want +got):\n%s", diff)
			}
			props, _ := schema["properties"].(map[string]any)
			var got []string
			for name := range props {
				got = append(got, name)
			}
			slices.Sort(got)
			all := append(slices.Clone(tc.wantRequired), tc.wantOptional...)
			slices.Sort(all)
			if diff := cmp.Diff(all, got); diff != "" {
				t.Errorf("parameters mismatch (-want +got):\n%s", diff)
			}
			if strings.Contains(decl.Description, "credentials") || strings.Contains(decl.Description, "tool_context") {
				t.Errorf("description mentions an argument the Go tool does not have:\n%s", decl.Description)
			}
		})
	}
}

// declaredArgs returns the entries of args that tl declares as parameters.
func declaredArgs(t *testing.T, tl runnable, args map[string]any) map[string]any {
	t.Helper()
	var schema map[string]any
	b, _ := json.Marshal(tl.Declaration().ParametersJsonSchema)
	_ = json.Unmarshal(b, &schema)
	props, _ := schema["properties"].(map[string]any)
	out := map[string]any{}
	for k, v := range args {
		if _, ok := props[k]; ok {
			out[k] = v
		}
	}
	return out
}

func toStrings(v any) []string {
	var out []string
	switch v := v.(type) {
	case []string:
		return v
	case []any:
		for _, s := range v {
			str, _ := s.(string)
			out = append(out, str)
		}
	}
	return out
}

func TestToolset_ExecuteSQLDescriptionFollowsResultMode(t *testing.T) {
	f := newFakeSpanner(t, nil)
	def := toolByName(t, newTestToolset(t, Config{ClientOptions: f.clientOptions()}), "spanner_execute_sql").Description()
	dict := toolByName(t, newTestToolset(t, Config{ClientOptions: f.clientOptions(), QueryResultMode: QueryResultModeDictList}), "spanner_execute_sql").Description()
	if def != strings.TrimRight(executeSQLDescription, "\n") {
		t.Errorf("default mode description is not execute_sql.txt")
	}
	if dict != strings.TrimRight(executeSQLDictListDescription, "\n") {
		t.Errorf("dict list mode description is not execute_sql_dict_list.txt")
	}
}

func TestTools_APIErrorIsReturnedToModel(t *testing.T) {
	f := newFakeSpanner(t, nil)
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"code":404,"message":"Database not found: d","status":"NOT_FOUND"}}`)
		return true
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	all := withArgs(map[string]any{
		"table_name": "t", "table_id": "t", "query": "SELECT 1",
		"embedding_column_to_search": "e", "columns": []any{"c"},
		"embedding_options": map[string]any{"vertex_ai_embedding_model_name": "m"},
	})
	for _, name := range toolNames(t, ts) {
		tl := toolByName(t, ts, name)
		wantError(t, run(t, tl, declaredArgs(t, tl, all)), "Database not found")
	}
}

func TestTools_CanceledContextIsAnError(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := toolByName(t, ts, "spanner_list_named_schemas").Run(newToolContext(t, ctx, nil), dbArgs)
	if err == nil {
		t.Fatal("Run() with a canceled context error = nil, want an error")
	}
}

func TestTools_InvalidIDsAreRejected(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{"empty project", map[string]any{"project_id": "", "instance_id": "i", "database_id": "d"}, "project_id must not be empty"},
		{"empty instance", map[string]any{"project_id": "p", "instance_id": "", "database_id": "d"}, "instance_id must not be empty"},
		{"dot database", map[string]any{"project_id": "p", "instance_id": "i", "database_id": "."}, "database_id is not a valid Spanner id"},
		{"dot dot instance", map[string]any{"project_id": "p", "instance_id": "..", "database_id": "d"}, "instance_id is not a valid Spanner id"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantError(t, run(t, toolByName(t, ts, "spanner_list_named_schemas"), tc.args), tc.want)
		})
	}
	if got := f.recorded(); len(got) != 0 {
		t.Errorf("invalid ids reached Spanner: %v", f.requestLines())
	}
}

func TestTools_IDsAreEscaped(t *testing.T) {
	f := newFakeSpanner(t, nil)
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"code":404,"message":"not found"}}`)
		return true
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	run(t, toolByName(t, ts, "spanner_list_named_schemas"), map[string]any{"project_id": "p/x", "instance_id": "i?y", "database_id": "d#z"})
	want := []string{"GET /v1/projects/p%2Fx/instances/i%3Fy/databases/d%23z"}
	if diff := cmp.Diff(want, f.requestLines()); diff != "" {
		t.Errorf("requests mismatch (-want +got):\n%s", diff)
	}
}

func TestTools_UserAgent(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"SCHEMATA": resultStream([]string{"SCHEMA_NAME"}, "STRING")})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs)
	for _, r := range f.recorded() {
		if ua := r.header.Get("User-Agent"); !strings.HasPrefix(ua, "adk-spanner-tool google-adk/") {
			t.Errorf("%s %s User-Agent = %q, want the adk-spanner-tool prefix", r.method, r.path, ua)
		}
	}
}

func TestTools_ExternalAccessToken(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"SCHEMATA": resultStream([]string{"SCHEMA_NAME"}, "STRING")})
	ts := newTestToolset(t, Config{ClientOptions: []option.ClientOption{option.WithEndpoint(f.server.URL + "/v1/")}, ExternalAccessTokenKey: "token"})
	got, err := toolByName(t, ts, "spanner_list_named_schemas").Run(newToolContext(t, t.Context(), map[string]any{"token": "abc"}), dbArgs)
	if err != nil || got["status"] != "SUCCESS" {
		t.Fatalf("Run() = %v, %v, want success", got, err)
	}
	for _, r := range f.recorded() {
		if auth := r.header.Get("Authorization"); auth != "Bearer abc" {
			t.Errorf("%s %s Authorization = %q, want the token from the state", r.method, r.path, auth)
		}
	}
}

func TestTools_ExternalAccessTokenMissing(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: []option.ClientOption{option.WithEndpoint(f.server.URL + "/v1/")}, ExternalAccessTokenKey: "token"})
	tests := []struct {
		name  string
		state map[string]any
	}{
		{"no key", nil},
		{"empty token", map[string]any{"token": ""}},
		{"not a string", map[string]any{"token": 42}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := toolByName(t, ts, "spanner_list_named_schemas").Run(newToolContext(t, t.Context(), tc.state), dbArgs)
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			wantError(t, got, `no access token found in the session state under the key "token"`)
		})
	}
	if len(f.recorded()) != 0 {
		t.Errorf("a call without a token reached Spanner: %v", f.requestLines())
	}
}

func TestSession_DeletedAfterQueryError(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{
		"SCHEMATA": `[{"error":{"code":400,"message":"Syntax error"}}]`,
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs), "Syntax error")
	want := []string{
		"GET " + testDB,
		"POST " + testDB + "/sessions",
		"POST /v1/" + testSession + ":executeStreamingSql",
		"DELETE /v1/" + testSession,
	}
	if diff := cmp.Diff(want, f.requestLines()); diff != "" {
		t.Errorf("requests mismatch (-want +got):\n%s", diff)
	}
}

func TestSession_DeletedWhenContextEnds(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	f := newFakeSpanner(t, nil)
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		if strings.HasSuffix(r.URL.Path, ":executeStreamingSql") {
			cancel()
			<-r.Context().Done()
			return true
		}
		return false
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	if _, err := toolByName(t, ts, "spanner_list_named_schemas").Run(newToolContext(t, ctx, nil), dbArgs); err == nil {
		t.Fatal("Run() error = nil, want the context's error")
	}
	lines := f.requestLines()
	if len(lines) == 0 || lines[len(lines)-1] != "DELETE /v1/"+testSession {
		t.Errorf("requests = %v, want the session deleted last", lines)
	}
}

func TestSession_DeleteErrorIsReturned(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"SCHEMATA": resultStream([]string{"SCHEMA_NAME"}, "STRING", "s1")})
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		if r.Method != http.MethodDelete {
			return false
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"code":503,"message":"unavailable"}}`)
		return true
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs), "deleting the Spanner session")
}

func TestSession_WithoutNameIsAnError(t *testing.T) {
	f := newFakeSpanner(t, nil)
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/sessions") {
			writeJSON(w, map[string]any{})
			return true
		}
		return false
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs), "the Spanner session has no name")
}

func TestSession_UndecodableResponseIsAnError(t *testing.T) {
	f := newFakeSpanner(t, nil)
	f.override = func(w http.ResponseWriter, r *http.Request, _ map[string]any) bool {
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, `{"databaseDialect":`)
			return true
		}
		return false
	}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs), "decoding the Spanner response")
}
