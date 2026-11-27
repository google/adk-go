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
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func executeSQLStream(t *testing.T, rows int) string {
	t.Helper()
	var values []any
	for i := range rows {
		values = append(values, fmt.Sprint(i), fmt.Sprintf("name%d", i))
	}
	return fieldStream(t, fields("id", "INT64", "name", "STRING"), values...)
}

func TestExecuteSQL(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"FROM users": executeSQLStream(t, 2)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "SELECT id, name FROM users"})))

	want := map[string]any{"status": "SUCCESS", "rows": []any{[]any{0.0, "name0"}, []any{1.0, "name1"}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	body := f.queryBodies()[0]
	if body["sql"] != "SELECT id, name FROM users" {
		t.Errorf("sql = %v, want the model's query unchanged", body["sql"])
	}
	if diff := cmp.Diff(roundTrip(t, singleUseReadOnly), body["transaction"]); diff != "" {
		t.Errorf("transaction mismatch (-want +got):\n%s", diff)
	}
	if body["params"] != nil {
		t.Errorf("params = %v, want none", body["params"])
	}
}

func TestExecuteSQL_DictList(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"FROM users": executeSQLStream(t, 1)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), QueryResultMode: QueryResultModeDictList})
	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "SELECT id, name FROM users"})))
	want := map[string]any{"status": "SUCCESS", "rows": []any{map[string]any{"id": 0.0, "name": "name0"}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

func TestExecuteSQL_DictListUnnamedAndDuplicateColumns(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"COUNT": fieldStream(t, fields("", "INT64", "x", "STRING", "x", "STRING"), "3", "first", "second")})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), QueryResultMode: QueryResultModeDictList})
	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "SELECT COUNT(*), 'first' AS x, 'second' AS x"})))
	want := map[string]any{"status": "SUCCESS", "rows": []any{map[string]any{"": 3.0, "x": "second"}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

func TestExecuteSQL_RowLimit(t *testing.T) {
	tests := []struct {
		name          string
		maxRows       int
		rows          int
		wantRows      int
		wantTruncated bool
	}{
		{"default limit, fewer rows", 0, 3, 3, false},
		{"default limit, exactly 50 rows", 0, 50, 50, false},
		{"default limit, more rows", 0, 51, 50, true},
		{"configured limit, more rows", 2, 5, 2, true},
		{"configured limit, exactly as many rows", 2, 2, 2, false},
		{"no rows", 2, 0, 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeSpanner(t, map[string]string{"FROM users": executeSQLStream(t, tc.rows)})
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), MaxQueryResultRows: tc.maxRows})
			got := run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "SELECT * FROM users"}))
			if rows, _ := got["rows"].([]any); len(rows) != tc.wantRows {
				t.Errorf("got %d rows, want %d", len(rows), tc.wantRows)
			}
			if _, truncated := got["result_is_likely_truncated"]; truncated != tc.wantTruncated {
				t.Errorf("result_is_likely_truncated set = %v, want %v", truncated, tc.wantTruncated)
			}
		})
	}
}

func TestExecuteSQL_DatabaseRole(t *testing.T) {
	queries := map[string]string{
		"FROM users":  executeSQLStream(t, 1),
		"SCHEMATA":    resultStream([]string{"SCHEMA_NAME"}, "STRING"),
		"ML.PREDICT":  fieldStream(t, []field{{Name: "values", Type: arrayOf(typeOf("FLOAT64"))}}, []any{0.5}),
		"COSINE_DIST": resultStream([]string{"c"}, "STRING"),
	}
	f := newFakeSpanner(t, queries)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), DatabaseRole: "analyst"})

	run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "SELECT * FROM users"}))
	run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs)
	run(t, toolByName(t, ts, "spanner_similarity_search"), withArgs(map[string]any{
		"table_name": "t", "query": "q", "embedding_column_to_search": "e", "columns": []any{"c"},
		"embedding_options": map[string]any{"spanner_googlesql_embedding_model_name": "m"},
	}))

	var sessionBodies []any
	for _, r := range f.recorded() {
		if r.method == "POST" && r.path == testDB+"/sessions" {
			sessionBodies = append(sessionBodies, r.body)
		}
	}
	// As in adk-python, only execute_sql queries as the database role.
	want := []any{
		map[string]any{"session": map[string]any{"creatorRole": "analyst"}},
		map[string]any{},
		map[string]any{},
	}
	if diff := cmp.Diff(want, sessionBodies); diff != "" {
		t.Errorf("session requests mismatch (-want +got):\n%s", diff)
	}
}

func TestExecuteSQL_QueryErrorIsReturnedToModel(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"UPDATE": `[{"error":{"code":9,"message":"DML statements can only be performed in a read-write or partitioned-dml transaction"}}]`})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	wantError(t, run(t, toolByName(t, ts, "spanner_execute_sql"), withArgs(map[string]any{"query": "UPDATE users SET name = 'x' WHERE true"})), "read-write")
}
