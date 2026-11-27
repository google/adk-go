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
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

var stringParam = map[string]any{"code": "STRING"}

func TestListTableNames(t *testing.T) {
	tests := []struct {
		name       string
		dialect    string
		args       map[string]any
		wantSQL    string
		wantParams map[string]any
	}{
		{
			name:       "GoogleSQL default schema",
			dialect:    "GOOGLE_STANDARD_SQL",
			args:       dbArgs,
			wantSQL:    "WHERE TABLE_SCHEMA = @schema AND SPANNER_STATE = 'COMMITTED'",
			wantParams: map[string]any{"schema": ""},
		},
		{
			name:       "GoogleSQL named schema",
			dialect:    "GOOGLE_STANDARD_SQL",
			args:       withArgs(map[string]any{"named_schema": "sales"}),
			wantSQL:    "WHERE TABLE_SCHEMA = @schema AND SPANNER_STATE = 'COMMITTED'",
			wantParams: map[string]any{"schema": "sales"},
		},
		{
			name:       "PostgreSQL default schema",
			dialect:    "POSTGRESQL",
			args:       dbArgs,
			wantSQL:    "WHERE TABLE_SCHEMA = $1",
			wantParams: map[string]any{"p1": "public"},
		},
		{
			name:       "PostgreSQL named schema",
			dialect:    "POSTGRESQL",
			args:       withArgs(map[string]any{"named_schema": "sales"}),
			wantSQL:    "WHERE TABLE_SCHEMA = $1",
			wantParams: map[string]any{"p1": "sales"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.TABLES": resultStream([]string{"TABLE_NAME"}, "STRING", "users", "orders")})
			f.dialect = tc.dialect
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

			got := run(t, toolByName(t, ts, "spanner_list_table_names"), tc.args)

			want := map[string]any{"status": "SUCCESS", "results": []any{"users", "orders"}}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("result mismatch (-want +got):\n%s", diff)
			}
			body := f.queryBodies()[0]
			if sql, _ := body["sql"].(string); !strings.Contains(sql, tc.wantSQL) {
				t.Errorf("sql = %q, want it to contain %q", sql, tc.wantSQL)
			}
			if diff := cmp.Diff(tc.wantParams, body["params"]); diff != "" {
				t.Errorf("params mismatch (-want +got):\n%s", diff)
			}
			for name := range tc.wantParams {
				if diff := cmp.Diff(stringParam, body["paramTypes"].(map[string]any)[name]); diff != "" {
					t.Errorf("param type of %s mismatch (-want +got):\n%s", name, diff)
				}
			}
			if diff := cmp.Diff(roundTrip(t, singleUseReadOnly), body["transaction"]); diff != "" {
				t.Errorf("transaction mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestListTableNames_Empty(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.TABLES": resultStream([]string{"TABLE_NAME"}, "STRING")})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_list_table_names"), dbArgs))
	want := map[string]any{"status": "SUCCESS", "results": []any{}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

// fieldStream is an executeStreamingSql response with the given columns and
// values, and a transaction id.
func fieldStream(t *testing.T, fields []field, values ...any) string {
	t.Helper()
	b, err := json.Marshal([]any{map[string]any{
		"metadata": map[string]any{"rowType": map[string]any{"fields": fields}, "transaction": map[string]any{"id": "dHgx"}},
		"values":   orEmpty(values),
	}})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	return string(b)
}

func fields(spec ...string) []field {
	var out []field
	for i := 0; i < len(spec); i += 2 {
		out = append(out, field{Name: spec[i], Type: typeOf(spec[i+1])})
	}
	return out
}

func TestGetTableSchema(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{
		"INFORMATION_SCHEMA.COLUMNS": fieldStream(t, fields(
			"COLUMN_NAME", "STRING", "TABLE_SCHEMA", "STRING", "SPANNER_TYPE", "STRING", "ORDINAL_POSITION", "INT64",
			"COLUMN_DEFAULT", "STRING", "IS_NULLABLE", "STRING", "IS_GENERATED", "STRING", "GENERATION_EXPRESSION", "STRING", "IS_STORED", "STRING"),
			"id", "", "INT64", "1", nil, "NO", "NEVER", nil, nil,
			"name", "", "STRING(MAX)", "2", nil, "YES", "NEVER", nil, nil,
		),
		"INFORMATION_SCHEMA.KEY_COLUMN_USAGE": fieldStream(t, fields(
			"COLUMN_NAME", "STRING", "CONSTRAINT_NAME", "STRING", "ORDINAL_POSITION", "INT64", "POSITION_IN_UNIQUE_CONSTRAINT", "INT64"),
			"id", "PK_users", "1", nil,
			"gone", "PK_other", "1", nil,
		),
		"INFORMATION_SCHEMA.TABLES": fieldStream(t, fields(
			"TABLE_SCHEMA", "STRING", "TABLE_NAME", "STRING", "TABLE_TYPE", "STRING", "PARENT_TABLE_NAME", "STRING",
			"ON_DELETE_ACTION", "STRING", "SPANNER_STATE", "STRING", "INTERLEAVE_TYPE", "STRING", "ROW_DELETION_POLICY_EXPRESSION", "STRING"),
			"", "users", "BASE TABLE", nil, nil, "COMMITTED", nil, nil,
		),
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_get_table_schema"), withArgs(map[string]any{"table_name": "users"})))

	want := map[string]any{
		"status": "SUCCESS",
		"results": map[string]any{
			"schema": map[string]any{
				"id": map[string]any{
					"SPANNER_TYPE": "INT64", "TABLE_SCHEMA": "", "ORDINAL_POSITION": 1.0, "COLUMN_DEFAULT": nil,
					"IS_NULLABLE": "NO", "IS_GENERATED": "NEVER", "GENERATION_EXPRESSION": nil, "IS_STORED": nil,
					"KEY_COLUMN_USAGE": []any{map[string]any{"CONSTRAINT_NAME": "PK_users", "ORDINAL_POSITION": 1.0, "POSITION_IN_UNIQUE_CONSTRAINT": nil}},
				},
				"name": map[string]any{
					"SPANNER_TYPE": "STRING(MAX)", "TABLE_SCHEMA": "", "ORDINAL_POSITION": 2.0, "COLUMN_DEFAULT": nil,
					"IS_NULLABLE": "YES", "IS_GENERATED": "NEVER", "GENERATION_EXPRESSION": nil, "IS_STORED": nil,
				},
			},
			"metadata": []any{map[string]any{
				"TABLE_SCHEMA": "", "TABLE_NAME": "users", "TABLE_TYPE": "BASE TABLE", "PARENT_TABLE_NAME": nil,
				"ON_DELETE_ACTION": nil, "SPANNER_STATE": "COMMITTED", "INTERLEAVE_TYPE": nil, "ROW_DELETION_POLICY_EXPRESSION": nil,
			}},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}

	bodies := f.queryBodies()
	if len(bodies) != 3 {
		t.Fatalf("got %d queries, want 3", len(bodies))
	}
	wantTxns := []any{roundTrip(t, beginReadOnly), map[string]any{"id": "dHgx"}, map[string]any{"id": "dHgx"}}
	for i, b := range bodies {
		if diff := cmp.Diff(wantTxns[i], b["transaction"]); diff != "" {
			t.Errorf("query %d transaction mismatch (-want +got):\n%s", i, diff)
		}
		if diff := cmp.Diff(map[string]any{"table_name": "users", "named_schema": ""}, b["params"]); diff != "" {
			t.Errorf("query %d params mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestGetTableSchema_NamedSchema(t *testing.T) {
	empty := fieldStream(t, fields("COLUMN_NAME", "STRING"))
	f := newFakeSpanner(t, map[string]string{
		"INFORMATION_SCHEMA.COLUMNS":          empty,
		"INFORMATION_SCHEMA.KEY_COLUMN_USAGE": empty,
		"INFORMATION_SCHEMA.TABLES":           empty,
	})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_get_table_schema"), withArgs(map[string]any{"table_name": "users", "named_schema": "sales"})))
	want := map[string]any{"status": "SUCCESS", "results": map[string]any{"schema": map[string]any{}, "metadata": []any{}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	for _, b := range f.queryBodies() {
		if got := b["params"].(map[string]any)["named_schema"]; got != "sales" {
			t.Errorf("named_schema param = %v, want %q", got, "sales")
		}
	}
}

func TestGetTableSchema_QueryErrors(t *testing.T) {
	ok := fieldStream(t, fields("COLUMN_NAME", "STRING"))
	failed := `[{"error":{"code":7,"message":"permission denied"}}]`
	for _, failing := range []string{"INFORMATION_SCHEMA.COLUMNS", "INFORMATION_SCHEMA.KEY_COLUMN_USAGE", "INFORMATION_SCHEMA.TABLES"} {
		t.Run(failing, func(t *testing.T) {
			queries := map[string]string{
				"INFORMATION_SCHEMA.COLUMNS":          ok,
				"INFORMATION_SCHEMA.KEY_COLUMN_USAGE": ok,
				"INFORMATION_SCHEMA.TABLES":           ok,
			}
			queries[failing] = failed
			f := newFakeSpanner(t, queries)
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			wantError(t, run(t, toolByName(t, ts, "spanner_get_table_schema"), withArgs(map[string]any{"table_name": "users"})), "permission denied")
		})
	}
}

func TestListTableIndexes(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.INDEXES": fieldStream(t, fields(
		"INDEX_NAME", "STRING", "TABLE_SCHEMA", "STRING", "INDEX_TYPE", "STRING", "PARENT_TABLE_NAME", "STRING",
		"IS_UNIQUE", "BOOL", "IS_NULL_FILTERED", "BOOL", "INDEX_STATE", "STRING"),
		"PRIMARY_KEY", "", "PRIMARY_KEY", "", true, false, nil,
	)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_list_table_indexes"), withArgs(map[string]any{"table_id": "users"})))

	want := map[string]any{"status": "SUCCESS", "results": []any{map[string]any{
		"INDEX_NAME": "PRIMARY_KEY", "TABLE_SCHEMA": "", "INDEX_TYPE": "PRIMARY_KEY", "PARENT_TABLE_NAME": "",
		"IS_UNIQUE": true, "IS_NULL_FILTERED": false, "INDEX_STATE": nil,
	}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	body := f.queryBodies()[0]
	if diff := cmp.Diff(map[string]any{"table_id": "users"}, body["params"]); diff != "" {
		t.Errorf("params mismatch (-want +got):\n%s", diff)
	}
}

func TestListTableIndexColumns(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.INDEX_COLUMNS": fieldStream(t, fields(
		"INDEX_NAME", "STRING", "TABLE_SCHEMA", "STRING", "COLUMN_NAME", "STRING", "ORDINAL_POSITION", "INT64",
		"IS_NULLABLE", "STRING", "SPANNER_TYPE", "STRING"),
		"PRIMARY_KEY", "", "id", "1", "NO", "INT64",
	)})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_list_table_index_columns"), withArgs(map[string]any{"table_id": "users"})))

	want := map[string]any{"status": "SUCCESS", "results": []any{map[string]any{
		"INDEX_NAME": "PRIMARY_KEY", "TABLE_SCHEMA": "", "COLUMN_NAME": "id", "ORDINAL_POSITION": 1.0,
		"IS_NULLABLE": "NO", "SPANNER_TYPE": "INT64",
	}}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

func TestListTableIndexes_Empty(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.INDEXES": fieldStream(t, fields("INDEX_NAME", "STRING"))})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	got := roundTrip(t, run(t, toolByName(t, ts, "spanner_list_table_indexes"), withArgs(map[string]any{"table_id": "users"})))
	if diff := cmp.Diff(map[string]any{"status": "SUCCESS", "results": []any{}}, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

func TestListNamedSchemas(t *testing.T) {
	f := newFakeSpanner(t, map[string]string{"INFORMATION_SCHEMA.SCHEMATA": resultStream([]string{"SCHEMA_NAME"}, "STRING", "sales", "hr")})
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
	got := run(t, toolByName(t, ts, "spanner_list_named_schemas"), dbArgs)
	if diff := cmp.Diff(map[string]any{"status": "SUCCESS", "results": []any{"sales", "hr"}}, got); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
	if body := f.queryBodies()[0]; body["params"] != nil {
		t.Errorf("params = %v, want none", body["params"])
	}
}

func TestGoogleSQLOnlyTools_PostgreSQL(t *testing.T) {
	tests := []struct {
		tool string
		args map[string]any
		want string
	}{
		{"spanner_get_table_schema", withArgs(map[string]any{"table_name": "t"}), "PostgreSQL dialect is not supported"},
		{"spanner_list_table_indexes", withArgs(map[string]any{"table_id": "t"}), "PostgreSQL dialect is not supported."},
		{"spanner_list_table_index_columns", withArgs(map[string]any{"table_id": "t"}), "PostgreSQL dialect is not supported."},
		{"spanner_list_named_schemas", dbArgs, "PostgreSQL dialect is not supported."},
		{"spanner_execute_sql", withArgs(map[string]any{"query": "SELECT 1"}), "PostgreSQL dialect is not supported."},
	}
	for _, tc := range tests {
		t.Run(tc.tool, func(t *testing.T) {
			f := newFakeSpanner(t, nil)
			f.dialect = "POSTGRESQL"
			ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})
			got := run(t, toolByName(t, ts, tc.tool), tc.args)
			if diff := cmp.Diff(map[string]any{"status": "ERROR", "error_details": tc.want}, got); diff != "" {
				t.Errorf("result mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]string{"GET " + testDB}, f.requestLines()); diff != "" {
				t.Errorf("requests mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRowObject_ShortRow(t *testing.T) {
	got := rowObject([]string{"a", "b"}, []any{1})
	if diff := cmp.Diff(map[string]any{"a": 1, "b": nil}, got); diff != "" {
		t.Errorf("rowObject() mismatch (-want +got):\n%s", diff)
	}
}
