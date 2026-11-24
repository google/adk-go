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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestValidateIdentifier(t *testing.T) {
	valid := []string{"users", "_t1", "sales.orders", "`my table`", `"Quoted"`, "a.`b c`.d", " users "}
	invalid := []string{"", "1users", "users;", "users --", "a b", "users)", "`a`b`", "`a\\b`", "a..b", "users OR 1=1"}
	for _, s := range valid {
		if err := validateIdentifier(s, "x"); err != nil {
			t.Errorf("validateIdentifier(%q) error = %v, want nil", s, err)
		}
	}
	for _, s := range invalid {
		if err := validateIdentifier(s, "x"); err == nil {
			t.Errorf("validateIdentifier(%q) = nil, want an error", s)
		}
	}
}

func TestValidateFilter(t *testing.T) {
	valid := []string{
		"price < 100",
		"price <= -1.5",
		"name = 'mop'",
		"name LIKE '%mop%'",
		"deleted IS NULL",
		"deleted is not null",
		"category IN ('a', 'b')",
		"category NOT IN (1, 2)",
		"price BETWEEN 1 AND 10",
		"in_stock",
		"1=1",
		"a = 1 AND b = 2 or c = 3",
		"(a = 1 OR b = 2) AND c = TRUE",
		"((a = 1 OR b = 2) AND c = 3) OR d = 4",
		"t.`col` != FALSE",
	}
	invalid := []string{
		"",
		"price < 100; DROP TABLE users",
		"price < (SELECT MAX(price) FROM t)",
		"name = 'a\\' OR 1=1 --'",
		"name = \"x\"",
		"a = 1 -- comment",
		"(((a = 1)))",
		"a = b",
		"a = 1 AND",
		"a IN ()",
		"EXISTS (SELECT 1)",
		"a = 1 UNION SELECT 1",
	}
	for _, s := range valid {
		if err := validateFilter(s); err != nil {
			t.Errorf("validateFilter(%q) error = %v, want nil", s, err)
		}
	}
	for _, s := range invalid {
		if err := validateFilter(s); err == nil {
			t.Errorf("validateFilter(%q) = nil, want an error", s)
		}
	}
}

// fakeEmbedder replaces the genai client with one that talks to a fake Gemini
// API server answering every embedding request with values.
type fakeEmbedder struct {
	mu       sync.Mutex
	requests []map[string]any
	paths    []string
}

func newFakeEmbedder(t *testing.T, status int, response string) *fakeEmbedder {
	t.Helper()
	e := &fakeEmbedder{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		e.mu.Lock()
		e.requests = append(e.requests, body)
		e.paths = append(e.paths, r.URL.Path)
		e.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	original := newGenAIClient
	newGenAIClient = func(ctx context.Context) (*genai.Client, error) {
		return genai.NewClient(ctx, &genai.ClientConfig{
			APIKey:      "key",
			Backend:     genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{BaseURL: server.URL},
		})
	}
	t.Cleanup(func() { newGenAIClient = original })
	return e
}

// searchStream is the response to a search query: two rows of a content
// column and a distance.
func searchStream(t *testing.T) string {
	return fieldStream(t, fields("content", "STRING", "distance", "FLOAT64"), "mop", 0.25, "broom", 0.5)
}

var wantSearchRows = map[string]any{"status": "SUCCESS", "rows": []any{[]any{"mop", 0.25}, []any{"broom", 0.5}}}

func searchArgs(extra map[string]any) map[string]any {
	return withArgs(merge(map[string]any{
		"table_name":                 "products",
		"query":                      "clean my house",
		"embedding_column_to_search": "embedding",
		"columns":                    []any{"content"},
		"embedding_options":          map[string]any{"vertex_ai_embedding_model_name": "text-embedding-005"},
	}, extra))
}

func TestSimilaritySearch_VertexAI(t *testing.T) {
	e := newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[0.5,0.25]}]}`)
	f := newFakeSpanner(t, map[string]string{"COSINE_DISTANCE": searchStream(t)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(nil)))

	if diff := cmp.Diff(wantSearchRows, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	if len(e.paths) != 1 || !strings.HasSuffix(e.paths[0], "models/text-embedding-005:batchEmbedContents") {
		t.Errorf("embedding requests = %v, want one for text-embedding-005", e.paths)
	}
	if strings.Contains(mustJSON(t, e.requests[0]), "outputDimensionality") {
		t.Errorf("embedding request %v sets outputDimensionality, want none", e.requests[0])
	}
	body := f.queryBodies()[0]
	if diff := cmp.Diff(map[string]any{"embedding": []any{0.5, 0.25}}, body["params"]); diff != "" {
		t.Errorf("params mismatch (-want +got):\n%s", diff)
	}
	if body["paramTypes"] != nil {
		t.Errorf("paramTypes = %v, want the embedding untyped as adk-python sends it", body["paramTypes"])
	}
	sql, _ := body["sql"].(string)
	for _, want := range []string{"SELECT content, COSINE_DISTANCE(", "embedding,\n      @embedding) AS distance", "FROM products", "WHERE 1=1", "ORDER BY distance", "LIMIT 4"} {
		if !strings.Contains(sql, want) {
			t.Errorf("sql = %q, want it to contain %q", sql, want)
		}
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return string(b)
}

func TestSimilaritySearch_VertexAIOutputDimensionality(t *testing.T) {
	e := newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[1]}]}`)
	f := newFakeSpanner(t, map[string]string{"COSINE_DISTANCE": searchStream(t)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(map[string]any{
		"embedding_options": map[string]any{"vertex_ai_embedding_model_name": "m", "output_dimensionality": 256},
	}))
	if !strings.Contains(mustJSON(t, e.requests[0]), `"outputDimensionality":256`) {
		t.Errorf("embedding request = %v, want outputDimensionality 256", e.requests[0])
	}
}

func TestSimilaritySearch_VertexAIErrors(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		response string
		want     string
	}{
		{"model error", http.StatusBadRequest, `{"error":{"code":400,"message":"model not found","status":"INVALID_ARGUMENT"}}`, "failed to embed content"},
		{"no embeddings", http.StatusOK, `{"embeddings":[]}`, "the model returned no embedding"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			newFakeEmbedder(t, tc.status, tc.response)
			f := newFakeSpanner(t, nil)
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			wantError(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(nil)), tc.want)
		})
	}
}

func TestSimilaritySearch_GenAIClientError(t *testing.T) {
	original := newGenAIClient
	newGenAIClient = func(context.Context) (*genai.Client, error) { return nil, errors.New("no API key") }
	t.Cleanup(func() { newGenAIClient = original })
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(nil)), "failed to embed content: no API key")
}

func TestSimilaritySearch_GoogleSQLModel(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{
		"ML.PREDICT":      fieldStream(t, []field{{Name: "values", Type: arrayOf(typeOf("FLOAT64"))}}, []any{0.5, 0.25}),
		"COSINE_DISTANCE": searchStream(t),
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(map[string]any{
		"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "EmbeddingsModel"},
	})))

	if diff := cmp.Diff(wantSearchRows, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	bodies := f.queryBodies()
	if len(bodies) != 2 {
		t.Fatalf("got %d queries, want the embedding query and the search", len(bodies))
	}
	if sql, _ := bodies[0]["sql"].(string); !strings.Contains(sql, "MODEL EmbeddingsModel") || !strings.Contains(sql, "CAST(@query AS STRING)") {
		t.Errorf("embedding sql = %q, want ML.PREDICT over the model with @query", sql)
	}
	if diff := cmp.Diff(map[string]any{"query": "clean my house"}, bodies[0]["params"]); diff != "" {
		t.Errorf("embedding params mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"embedding": []any{0.5, 0.25}}, bodies[1]["params"]); diff != "" {
		t.Errorf("search params mismatch (-want +got):\n%s", diff)
	}
}

func TestSimilaritySearch_GoogleSQLModelBadResult(t *testing.T) {
	vector := []field{{Name: "values", Type: arrayOf(typeOf("FLOAT64"))}}
	tests := []struct {
		name   string
		stream string
		want   string
	}{
		{"no rows", fieldStream(t, vector), "did not return one embedding"},
		{"two rows", fieldStream(t, vector, []any{1.0}, []any{2.0}), "did not return one embedding"},
		{"NaN", fieldStream(t, vector, []any{"NaN"}), "not a number"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeSpanner(t, map[string]string{"ML.PREDICT": tc.stream})
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			wantError(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(map[string]any{
				"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "m"},
			})), tc.want)
		})
	}
}

func TestSimilaritySearch_PostgreSQLEndpoint(t *testing.T) {
	const endpoint = "projects/p/locations/us-central1/publishers/google/models/text-embedding-005"
	f := newFakeSpanner(t, map[string]string{
		"ML_PREDICT_ROW":          fieldStream(t, []field{{Name: "v", Type: arrayOf(typeOf("FLOAT32"))}}, []any{0.5}),
		"spanner.cosine_distance": searchStream(t),
	})
	f.dialect = "POSTGRESQL"
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(map[string]any{
		"embedding_options": map[string]any{"spanner_postgresql_vertex_ai_embedding_model_endpoint": endpoint, "output_dimensionality": 8},
	})))

	if diff := cmp.Diff(wantSearchRows, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	bodies := f.queryBodies()
	embedSQL, _ := bodies[0]["sql"].(string)
	for _, want := range []string{"'" + endpoint + "'", "$1::TEXT", "'outputDimensionality',\n            8"} {
		if !strings.Contains(embedSQL, want) {
			t.Errorf("embedding sql = %q, want it to contain %q", embedSQL, want)
		}
	}
	if diff := cmp.Diff(map[string]any{"p1": "clean my house"}, bodies[0]["params"]); diff != "" {
		t.Errorf("embedding params mismatch (-want +got):\n%s", diff)
	}
	searchSQL, _ := bodies[1]["sql"].(string)
	if !strings.Contains(searchSQL, "spanner.cosine_distance(\n      embedding,\n      $1)") {
		t.Errorf("search sql = %q, want the PostgreSQL distance function over $1", searchSQL)
	}
	if diff := cmp.Diff(map[string]any{"p1": []any{0.5}}, bodies[1]["params"]); diff != "" {
		t.Errorf("search params mismatch (-want +got):\n%s", diff)
	}
}

func TestSimilaritySearch_PostgreSQLWithoutDimensionality(t *testing.T) {
	if sql := postgreSQLEmbeddingQuery("projects/p/locations/l/publishers/google/models/m", nil); strings.Contains(sql, "parameters") {
		t.Errorf("sql = %q, want no parameters object without an output dimensionality", sql)
	}
}

func TestSimilaritySearch_SearchOptions(t *testing.T) {
	tests := []struct {
		name    string
		options map[string]any
		filter  string
		want    []string
		notWant []string
	}{
		{
			name:    "top k and distance",
			options: map[string]any{"top_k": 2, "distance_type": "EUCLIDEAN"},
			want:    []string{"EUCLIDEAN_DISTANCE(", "LIMIT 2"},
		},
		{
			name:    "dot product",
			options: map[string]any{"distance_type": "DOT_PRODUCT"},
			want:    []string{"DOT_PRODUCT("},
		},
		{
			name:    "top k zero has no limit",
			options: map[string]any{"top_k": 0},
			notWant: []string{"LIMIT"},
		},
		{
			name:   "filter",
			filter: "price < 100",
			want:   []string{"WHERE price < 100"},
		},
		{
			name:    "approximate",
			options: map[string]any{"nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS", "num_leaves_to_search": 50, "top_k": 3},
			filter:  "price < 100",
			want:    []string{"APPROX_COSINE_DISTANCE(", `options => JSON '{"num_leaves_to_search": 50}'`, "WHERE embedding IS NOT NULL AND price < 100", "LIMIT 3"},
		},
		{
			name:    "approximate default leaves",
			options: map[string]any{"nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS"},
			want:    []string{`{"num_leaves_to_search": 1000}`, "WHERE embedding IS NOT NULL\n", "LIMIT 4"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[1]}]}`)
			f := newFakeSpanner(t, map[string]string{"SELECT content": searchStream(t)})
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			extra := map[string]any{}
			if tc.options != nil {
				extra["search_options"] = tc.options
			}
			if tc.filter != "" {
				extra["additional_filter"] = tc.filter
			}
			got := run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(extra))
			if got["status"] != "SUCCESS" {
				t.Fatalf("result = %v, want success", got)
			}
			sql, _ := f.queryBodies()[0]["sql"].(string)
			for _, want := range tc.want {
				if !strings.Contains(sql, want) {
					t.Errorf("sql = %q, want it to contain %q", sql, want)
				}
			}
			for _, notWant := range tc.notWant {
				if strings.Contains(sql, notWant) {
					t.Errorf("sql = %q, want it not to contain %q", sql, notWant)
				}
			}
		})
	}
}

func TestSimilaritySearch_InvalidArguments(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]any
		want  string
	}{
		{"table name", map[string]any{"table_name": "products; DROP TABLE x"}, "invalid SQL identifier for table_name"},
		{"embedding column", map[string]any{"embedding_column_to_search": "e)"}, "invalid SQL identifier for embedding_column_to_search"},
		{"column", map[string]any{"columns": []any{"content", "1+1"}}, "invalid SQL identifier for columns"},
		{"filter", map[string]any{"additional_filter": "1=1; DELETE FROM x"}, "additional_filter contains unsafe or unsupported patterns"},
		{"googlesql model", map[string]any{"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "m) --"}}, "invalid SQL identifier for spanner_googlesql_embedding_model_name"},
		{"postgresql endpoint", map[string]any{"embedding_options": map[string]any{"spanner_postgresql_vertex_ai_embedding_model_endpoint": "x'); DROP"}}, "invalid Vertex AI endpoint format"},
		{"no model", map[string]any{"embedding_options": map[string]any{}}, "exactly one embedding model option must be specified"},
		{"two models", map[string]any{"embedding_options": map[string]any{"vertex_ai_embedding_model_name": "a", "spanner_googlesql_embedding_model_name": "b"}}, "exactly one embedding model option"},
		{"dimensionality with googlesql model", map[string]any{"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "m", "output_dimensionality": 8}}, "output_dimensionality] is not supported"},
		{"unknown algorithm", map[string]any{"search_options": map[string]any{"nearest_neighbors_algorithm": "BRUTE"}}, "unsupported search_options['nearest_neighbors_algorithm']: BRUTE"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeSpanner(t, nil)
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			wantError(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(tc.extra)), tc.want)
			if len(f.recorded()) != 0 {
				t.Errorf("invalid arguments reached Spanner: %v", f.requestLines())
			}
		})
	}
}

func TestSimilaritySearch_InvalidAfterDialect(t *testing.T) {
	tests := []struct {
		name    string
		dialect string
		extra   map[string]any
		want    string
	}{
		{"postgresql endpoint on GoogleSQL", "GOOGLE_STANDARD_SQL", map[string]any{"embedding_options": map[string]any{"spanner_postgresql_vertex_ai_embedding_model_endpoint": "projects/p/locations/l/publishers/google/models/m"}}, "must be specified for GoogleSQL dialect"},
		{"googlesql model on PostgreSQL", "POSTGRESQL", map[string]any{"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "m"}}, "must be specified for PostgreSQL dialect"},
		{"approximate on PostgreSQL", "POSTGRESQL", map[string]any{"search_options": map[string]any{"nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS"}}, "APPROXIMATE_NEAREST_NEIGHBORS is not supported for PostgreSQL dialect"},
		{"unknown distance", "GOOGLE_STANDARD_SQL", map[string]any{"search_options": map[string]any{"distance_type": "MANHATTAN"}}, `unsupported distance type "MANHATTAN"`},
		{"dimensionality out of range", "GOOGLE_STANDARD_SQL", map[string]any{"embedding_options": map[string]any{"vertex_ai_embedding_model_name": "m", "output_dimensionality": 0}}, "must be a positive 32-bit integer"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[1]}]}`)
			f := newFakeSpanner(t, nil)
			f.dialect = tc.dialect
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			wantError(t, run(t, toolByName(t, ts, "spanner_similarity_search"), searchArgs(tc.extra)), tc.want)
		})
	}
}

func TestSimilaritySearch_ArgumentOfWrongTypeIsRejected(t *testing.T) {
	f := newFakeSpanner(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	_, err := toolByName(t, ts, "spanner_similarity_search").Run(newToolContext(t, t.Context(), nil), searchArgs(map[string]any{
		"search_options": map[string]any{"top_k": "two"},
	}))
	if err == nil {
		t.Fatal("Run() with a string top_k error = nil, want a validation error")
	}
}

func vectorStore() *VectorStoreConfig {
	return &VectorStoreConfig{
		ProjectID: "p", InstanceID: "i", DatabaseID: "d", TableName: "docs",
		ContentColumn: "content", EmbeddingColumn: "embedding", VectorLength: 768,
		EmbeddingModel: "text-embedding-005",
	}
}

func TestVectorStoreSimilaritySearch(t *testing.T) {
	e := newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[0.5,0.25]}]}`)
	f := newFakeSpanner(t, map[string]string{"FROM docs": searchStream(t)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), VectorStore: vectorStore()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_vector_store_similarity_search"), map[string]any{"query": "tuning"}))

	if diff := cmp.Diff(wantSearchRows, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	if !strings.Contains(mustJSON(t, e.requests[0]), `"outputDimensionality":768`) {
		t.Errorf("embedding request = %v, want outputDimensionality set to the vector length", e.requests[0])
	}
	sql, _ := f.queryBodies()[0]["sql"].(string)
	for _, want := range []string{"SELECT content, COSINE_DISTANCE(", "FROM docs", "WHERE 1=1", "LIMIT 4"} {
		if !strings.Contains(sql, want) {
			t.Errorf("sql = %q, want it to contain %q", sql, want)
		}
	}
}

func TestVectorStoreSimilaritySearch_Config(t *testing.T) {
	newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[1]}]}`)
	f := newFakeSpanner(t, map[string]string{"FROM docs": searchStream(t)})
	vs := vectorStore()
	vs.SelectedColumns = []string{"title", "content"}
	vs.TopK = 7
	vs.DistanceType = DistanceDotProduct
	vs.NearestNeighborsAlgorithm = ApproximateNearestNeighbors
	vs.NumLeavesToSearch = 20
	vs.AdditionalFilter = "lang = 'en'"
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), VectorStore: vs})
	// The toolset keeps its own copy of the config.
	vs.SelectedColumns[0] = "changed"
	vs.TableName = "changed"

	run(t, toolByName(t, ts, "spanner_vector_store_similarity_search"), map[string]any{"query": "q"})

	sql, _ := f.queryBodies()[0]["sql"].(string)
	for _, want := range []string{"SELECT title, content, APPROX_DOT_PRODUCT(", `{"num_leaves_to_search": 20}`, "FROM docs", "WHERE embedding IS NOT NULL AND lang = 'en'", "LIMIT 7"} {
		if !strings.Contains(sql, want) {
			t.Errorf("sql = %q, want it to contain %q", sql, want)
		}
	}
}

func TestVectorStoreSimilaritySearch_Error(t *testing.T) {
	newFakeEmbedder(t, http.StatusOK, `{"embeddings":[{"values":[1]}]}`)
	f := newFakeSpanner(t, map[string]string{"FROM docs": `[{"error":{"code":5,"message":"Table not found: docs"}}]`})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), VectorStore: vectorStore()})
	wantError(t, run(t, toolByName(t, ts, "spanner_vector_store_similarity_search"), map[string]any{"query": "q"}), "Table not found")
}
