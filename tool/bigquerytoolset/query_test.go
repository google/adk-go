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
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

const (
	selectDryRun = `{"kind":"bigquery#job","configuration":{"dryRun":true},"statistics":{"totalBytesProcessed":"10","query":{"statementType":"SELECT"}}}`
	insertDryRun = `{"statistics":{"query":{"statementType":"INSERT"}}}`

	dryRunPath  = "POST /bigquery/v2/projects/p/jobs"
	queryPath   = "POST /bigquery/v2/projects/p/queries"
	resultsPath = "GET /bigquery/v2/projects/p/queries/job1"
)

// oneColumn is a finished query result with an INT64 column n and the given
// rows.
func oneColumn(rows ...string) string {
	cells := make([]string, len(rows))
	for i, r := range rows {
		cells[i] = `{"f":[{"v":"` + r + `"}]}`
	}
	return `{"jobComplete":true,"jobReference":{"projectId":"p","jobId":"job1","location":"US"},"schema":{"fields":[{"name":"n","type":"INTEGER"}]},"rows":[` + strings.Join(cells, ",") + `]`
}

func page(rows, pageToken string) string {
	if pageToken != "" {
		return rows + `,"pageToken":"` + pageToken + `"}`
	}
	return rows + `}`
}

func runSQL(t *testing.T, cfg Config, responses map[string]fakeResponse, args map[string]any) (*fakeBigQuery, map[string]any) {
	t.Helper()
	f := newFakeBigQuery(t, responses)
	cfg.ClientOptions = f.clientOptions()
	ts := newTestToolset(t, cfg)
	return f, run(t, toolByName(t, ts, "execute_sql"), args)
}

func requestKeys(f *fakeBigQuery) []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var keys []string
	for _, r := range f.requests {
		keys = append(keys, r.Method+" "+r.URL.Path)
	}
	return keys
}

func TestExecuteSQL_Declaration(t *testing.T) {
	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions()})

	decl := toolByName(t, ts, "execute_sql").Declaration()
	schema := declarationSchema(t, decl)

	if diff := cmp.Diff([]string{"project_id", "query"}, slices.Sorted(slices.Values(schema.Required))); diff != "" {
		t.Errorf("required mismatch (-want +got):\n%s", diff)
	}
	wantTypes := map[string]string{"project_id": "string", "query": "string", "dry_run": "boolean"}
	gotTypes := map[string]string{}
	for name, p := range schema.Properties {
		gotTypes[name] = string(p.Type)
	}
	if diff := cmp.Diff(wantTypes, gotTypes); diff != "" {
		t.Errorf("property types mismatch (-want +got):\n%s", diff)
	}
	if want := "Run a BigQuery or BigQuery ML SQL query in the project and return the result."; firstLine(decl.Description) != want {
		t.Errorf("description first line = %q, want %q", firstLine(decl.Description), want)
	}
	for _, arg := range []string{"credentials (Credentials)", "settings (BigQueryToolConfig)", "tool_context (ToolContext)", `"""`} {
		if strings.Contains(decl.Description, arg) {
			t.Errorf("description contains %q, which the Go tool does not have", arg)
		}
	}
	if strings.HasSuffix(decl.Description, "\n") {
		t.Errorf("description ends with a newline")
	}
}

// TestExecuteSQL_Select checks the two requests a read-only query makes, a
// dry run that finds the statement type and the query itself, and the result.
func TestExecuteSQL_Select(t *testing.T) {
	f, got := runSQL(t, Config{}, map[string]fakeResponse{
		dryRunPath: {body: selectDryRun},
		queryPath:  {body: page(oneColumn("1", "2"), "")},
	}, map[string]any{"project_id": "p", "query": "SELECT n FROM t"})

	want := map[string]any{"status": "SUCCESS", "rows": []any{map[string]any{"n": float64(1)}, map[string]any{"n": float64(2)}}}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
	wantBodies := []any{
		decodeJSON(t, `{
			"jobReference": {"projectId": "p"},
			"configuration": {"dryRun": true, "labels": {"adk-bigquery-tool": "execute_sql"}, "query": {"query": "SELECT n FROM t", "useLegacySql": false}}
		}`),
		decodeJSON(t, `{
			"query": "SELECT n FROM t", "useLegacySql": false, "formatOptions": {"useInt64Timestamp": true},
			"maxResults": 50, "labels": {"adk-bigquery-tool": "execute_sql"}
		}`),
	}
	if diff := cmp.Diff(wantBodies, f.requestBodies()); diff != "" {
		t.Errorf("request bodies mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{dryRunPath, queryPath}, requestKeys(f)); diff != "" {
		t.Errorf("requests mismatch (-want +got):\n%s", diff)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, r := range f.requests {
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("%s Content-Type = %q, want application/json", r.URL.Path, got)
		}
	}
}

func TestExecuteSQL_EmptyResult(t *testing.T) {
	_, got := runSQL(t, Config{}, map[string]fakeResponse{
		dryRunPath: {body: selectDryRun},
		queryPath:  {body: `{"jobComplete":true,"jobReference":{"jobId":"job1"},"schema":{"fields":[{"name":"n","type":"INTEGER"}]},"totalRows":"0"}`},
	}, map[string]any{"project_id": "p", "query": "SELECT n FROM t WHERE false"})

	want := map[string]any{"status": "SUCCESS", "rows": []any{}}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
}

// TestExecuteSQL_NonSelectIsRefused checks the read-only guard: a statement
// other than SELECT is refused after the dry run, with adk-python's message,
// and never runs.
func TestExecuteSQL_NonSelectIsRefused(t *testing.T) {
	for _, dryRun := range []bool{false, true} {
		f, got := runSQL(t, Config{}, map[string]fakeResponse{
			dryRunPath: {body: insertDryRun},
		}, map[string]any{"project_id": "p", "query": "INSERT INTO t VALUES (1)", "dry_run": dryRun})

		want := map[string]any{"status": "ERROR", "error_details": "Read-only mode only supports SELECT statements."}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("dry_run=%v: Run() mismatch (-want +got):\n%s", dryRun, diff)
		}
		if diff := cmp.Diff([]string{dryRunPath}, requestKeys(f)); diff != "" {
			t.Errorf("dry_run=%v: requests mismatch (-want +got):\n%s", dryRun, diff)
		}
	}
}

// TestExecuteSQL_MissingStatementTypeIsRefused checks that a dry run that
// does not say the query is a SELECT counts as a write.
func TestExecuteSQL_MissingStatementTypeIsRefused(t *testing.T) {
	_, got := runSQL(t, Config{}, map[string]fakeResponse{
		dryRunPath: {body: `{}`},
	}, map[string]any{"project_id": "p", "query": "SELECT 1"})

	if got["status"] != "ERROR" {
		t.Errorf("Run() = %v, want an ERROR result", got)
	}
}

// TestExecuteSQL_DryRun checks that a dry run returns the dry run job as the
// REST API sent it, like adk-python's to_api_repr, and runs nothing.
func TestExecuteSQL_DryRun(t *testing.T) {
	f, got := runSQL(t, Config{}, map[string]fakeResponse{
		dryRunPath: {body: selectDryRun},
	}, map[string]any{"project_id": "p", "query": "SELECT 1", "dry_run": true})

	want := map[string]any{"status": "SUCCESS", "dry_run_info": decodeJSON(t, selectDryRun)}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{dryRunPath}, requestKeys(f)); diff != "" {
		t.Errorf("requests mismatch (-want +got):\n%s", diff)
	}
}

func TestExecuteSQL_DefaultProjectID(t *testing.T) {
	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{name: "omitted", args: map[string]any{"query": "SELECT 1", "dry_run": true}, want: "POST /bigquery/v2/projects/my-project/jobs"},
		{name: "null", args: map[string]any{"project_id": nil, "query": "SELECT 1", "dry_run": true}, want: "POST /bigquery/v2/projects/my-project/jobs"},
		{name: "overridden", args: map[string]any{"project_id": "other", "query": "SELECT 1", "dry_run": true}, want: "POST /bigquery/v2/projects/other/jobs"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f, got := runSQL(t, Config{DefaultProjectID: "my-project"}, map[string]fakeResponse{
				"POST /bigquery/v2/projects/my-project/jobs": {body: selectDryRun},
				"POST /bigquery/v2/projects/other/jobs":      {body: selectDryRun},
			}, tc.args)

			if got["status"] != "SUCCESS" {
				t.Fatalf("Run() = %v, want SUCCESS", got)
			}
			if diff := cmp.Diff([]string{tc.want}, requestKeys(f)); diff != "" {
				t.Errorf("requests mismatch (-want +got):\n%s", diff)
			}
		})
	}

	f := newFakeBigQuery(t, nil)
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), DefaultProjectID: "my-project"})
	schema := declarationSchema(t, toolByName(t, ts, "execute_sql").Declaration())
	if diff := cmp.Diff([]string{"query"}, schema.Required); diff != "" {
		t.Errorf("required mismatch (-want +got):\n%s", diff)
	}
}

// TestExecuteSQL_ComputeProjectID checks the guardrail: a query in any other
// project fails with adk-python's message before reaching BigQuery.
func TestExecuteSQL_ComputeProjectID(t *testing.T) {
	f, got := runSQL(t, Config{ComputeProjectID: "compute"}, nil, map[string]any{"project_id": "p", "query": "SELECT 1"})

	want := map[string]any{"status": "ERROR", "error_details": "Cannot execute query in the project p, as the tool is restricted to execute queries only in the project compute."}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
	if keys := requestKeys(f); len(keys) != 0 {
		t.Errorf("requests = %v, want none", keys)
	}

	_, got = runSQL(t, Config{ComputeProjectID: "compute"}, map[string]fakeResponse{
		"POST /bigquery/v2/projects/compute/jobs": {body: selectDryRun},
	}, map[string]any{"project_id": "compute", "query": "SELECT 1", "dry_run": true})
	if got["status"] != "SUCCESS" {
		t.Errorf("Run() in the compute project = %v, want SUCCESS", got)
	}
}

// TestExecuteSQL_ConfigInRequests checks that each Config setting reaches
// the requests it belongs in, and only those.
func TestExecuteSQL_ConfigInRequests(t *testing.T) {
	labels := map[string]string{"team": "data"}
	cfg := Config{
		ApplicationName:    "my-agent",
		Location:           "EU",
		MaxQueryResultRows: 7,
		MaximumBytesBilled: 20 << 20,
		JobLabels:          labels,
	}
	f, _ := runSQL(t, cfg, map[string]fakeResponse{
		dryRunPath: {body: selectDryRun},
		queryPath:  {body: page(oneColumn("1"), "")},
	}, map[string]any{"project_id": "p", "query": "SELECT 1"})

	wantLabels := `{"team": "data", "adk-bigquery-tool": "execute_sql", "adk-bigquery-application-name": "my-agent"}`
	wantBodies := []any{
		decodeJSON(t, `{
			"jobReference": {"projectId": "p", "location": "EU"},
			"configuration": {"dryRun": true, "labels": `+wantLabels+`, "query": {"query": "SELECT 1", "useLegacySql": false}}
		}`),
		decodeJSON(t, `{
			"query": "SELECT 1", "useLegacySql": false, "formatOptions": {"useInt64Timestamp": true},
			"maxResults": 7, "labels": `+wantLabels+`, "maximumBytesBilled": "20971520", "location": "EU"
		}`),
	}
	if diff := cmp.Diff(wantBodies, f.requestBodies()); diff != "" {
		t.Errorf("request bodies mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]string{"team": "data"}, labels); diff != "" {
		t.Errorf("Config.JobLabels changed (-want +got):\n%s", diff)
	}
}

// TestExecuteSQL_JobLabelsAreCopied checks that changing Config.JobLabels
// after New does not change the labels of later jobs.
func TestExecuteSQL_JobLabelsAreCopied(t *testing.T) {
	f := newFakeBigQuery(t, map[string]fakeResponse{dryRunPath: {body: selectDryRun}})
	labels := map[string]string{"team": "data"}
	ts := newTestToolset(t, Config{ClientOptions: f.clientOptions(), JobLabels: labels})
	labels["team"] = "changed"

	run(t, toolByName(t, ts, "execute_sql"), map[string]any{"project_id": "p", "query": "SELECT 1", "dry_run": true})

	body, _ := f.requestBodies()[0].(map[string]any)
	got := body["configuration"].(map[string]any)["labels"]
	want := map[string]any{"team": "data", "adk-bigquery-tool": "execute_sql"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("labels mismatch (-want +got):\n%s", diff)
	}
}

// TestExecuteSQL_RowLimit checks MaxQueryResultRows across pages, and that
// result_is_likely_truncated is set only when rows were left out.
func TestExecuteSQL_RowLimit(t *testing.T) {
	tests := []struct {
		name          string
		responses     map[string]fakeResponse
		wantRows      []string
		wantTruncated bool
		wantResults   []string // maxResults of each results request
	}{
		{
			name:      "fewer rows than the limit",
			responses: map[string]fakeResponse{queryPath: {body: page(oneColumn("1"), "")}},
			wantRows:  []string{"1"},
		},
		{
			name:      "exactly the limit",
			responses: map[string]fakeResponse{queryPath: {body: page(oneColumn("1", "2"), "")}},
			wantRows:  []string{"1", "2"},
		},
		{
			name:          "limit reached with more pages",
			responses:     map[string]fakeResponse{queryPath: {body: page(oneColumn("1", "2"), "next")}},
			wantRows:      []string{"1", "2"},
			wantTruncated: true,
		},
		{
			name:          "page with more rows than the limit",
			responses:     map[string]fakeResponse{queryPath: {body: page(oneColumn("1", "2", "3"), "")}},
			wantRows:      []string{"1", "2"},
			wantTruncated: true,
		},
		{
			name: "rows from a second page",
			responses: map[string]fakeResponse{
				queryPath:                     {body: page(oneColumn("1"), "p2")},
				resultsPath + "?pageToken=p2": {body: `{"jobComplete":true,"schema":{"fields":[{"name":"n","type":"INTEGER"}]},"rows":[{"f":[{"v":"2"}]}]}`},
			},
			wantRows:    []string{"1", "2"},
			wantResults: []string{"1"},
		},
		{
			// Later pages need not repeat the job reference, so the next
			// request reuses the first one's.
			name: "pages without a job reference",
			responses: map[string]fakeResponse{
				queryPath:                     {body: page(oneColumn(), "p2")},
				resultsPath + "?pageToken=p2": {body: `{"jobComplete":true,"schema":{"fields":[{"name":"n","type":"INTEGER"}]},"rows":[{"f":[{"v":"1"}]}],"pageToken":"p3"}`},
				resultsPath + "?pageToken=p3": {body: `{"jobComplete":true,"schema":{"fields":[{"name":"n","type":"INTEGER"}]},"rows":[{"f":[{"v":"2"}]}]}`},
			},
			wantRows:    []string{"1", "2"},
			wantResults: []string{"2", "1"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.responses[dryRunPath] = fakeResponse{body: selectDryRun}
			f, got := runSQL(t, Config{MaxQueryResultRows: 2}, tc.responses, map[string]any{"project_id": "p", "query": "SELECT n FROM t"})

			rows := []any{}
			for _, r := range tc.wantRows {
				rows = append(rows, map[string]any{"n": decodeJSON(t, r)})
			}
			want := map[string]any{"status": "SUCCESS", "rows": rows}
			if tc.wantTruncated {
				want["result_is_likely_truncated"] = true
			}
			if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
				t.Errorf("Run() mismatch (-want +got):\n%s", diff)
			}
			var gotResults []string
			f.mu.Lock()
			for _, r := range f.requests[2:] {
				gotResults = append(gotResults, r.URL.Query().Get("maxResults"))
				if loc := r.URL.Query().Get("location"); loc != "US" {
					t.Errorf("results request location = %q, want the job's location US", loc)
				}
				if r.URL.Query().Get("formatOptions.useInt64Timestamp") != "true" {
					t.Errorf("results request %q does not ask for integer timestamps", r.URL.RawQuery)
				}
			}
			f.mu.Unlock()
			if diff := cmp.Diff(tc.wantResults, gotResults); diff != "" {
				t.Errorf("results requests maxResults mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestExecuteSQL_WaitsForQuery checks that a query that has not finished
// when jobs.query returns is waited for with jobs.getQueryResults.
func TestExecuteSQL_WaitsForQuery(t *testing.T) {
	f, got := runSQL(t, Config{}, map[string]fakeResponse{
		dryRunPath:  {body: selectDryRun},
		queryPath:   {body: `{"jobComplete":false,"jobReference":{"projectId":"p","jobId":"job1","location":"US"}}`},
		resultsPath: {body: page(oneColumn("7"), "")},
	}, map[string]any{"project_id": "p", "query": "SELECT 7 AS n"})

	want := map[string]any{"status": "SUCCESS", "rows": []any{map[string]any{"n": float64(7)}}}
	if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
		t.Errorf("Run() mismatch (-want +got):\n%s", diff)
	}
	if q := f.lastRequest().URL.Query(); q.Has("pageToken") || q.Get("maxResults") != "50" {
		t.Errorf("results request query = %q, want maxResults=50 and no page token", f.lastRequest().URL.RawQuery)
	}
}

func TestExecuteSQL_ResultErrors(t *testing.T) {
	tests := []struct {
		name      string
		responses map[string]fakeResponse
		want      string
	}{
		{
			name: "repeated page token",
			responses: map[string]fakeResponse{
				queryPath:                       {body: page(oneColumn(), "same")},
				resultsPath + "?pageToken=same": {body: page(oneColumn(), "same")},
			},
			want: "page token it had already returned",
		},
		{
			name: "results request fails",
			responses: map[string]fakeResponse{
				queryPath:   {body: `{"jobComplete":false,"jobReference":{"jobId":"job1"}}`},
				resultsPath: {status: http.StatusNotFound, body: `{"error":{"code":404,"message":"Not found: Job p:job1"}}`},
			},
			want: "Not found: Job p:job1",
		},
		{
			name:      "unfinished query without a job",
			responses: map[string]fakeResponse{queryPath: {body: `{"jobComplete":false}`}},
			want:      "did not return the query's job",
		},
		{
			name:      "row with too few cells",
			responses: map[string]fakeResponse{queryPath: {body: `{"jobComplete":true,"schema":{"fields":[{"name":"a","type":"STRING"},{"name":"b","type":"STRING"}]},"rows":[{"f":[{"v":"x"}]}]}`}},
			want:      "row of 1 cells for 2 columns",
		},
		{
			name:      "malformed integer",
			responses: map[string]fakeResponse{queryPath: {body: page(oneColumn("x"), "")}},
			want:      "column n",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.responses[dryRunPath] = fakeResponse{body: selectDryRun}
			_, got := runSQL(t, Config{}, tc.responses, map[string]any{"project_id": "p", "query": "SELECT 1"})

			if details, _ := got["error_details"].(string); got["status"] != "ERROR" || !strings.Contains(details, tc.want) {
				t.Errorf("Run() = %v, want an ERROR result mentioning %q", got, tc.want)
			}
		})
	}
}

// TestRowValues checks each column type against what adk-python's
// execute_sql returns for it.
func TestRowValues(t *testing.T) {
	tests := []struct {
		name  string
		field string
		cell  string
		want  string // JSON
	}{
		{name: "null", field: `{"type":"INTEGER"}`, cell: `null`, want: `null`},
		{name: "string", field: `{"type":"STRING"}`, cell: `"a\"b"`, want: `"a\"b"`},
		{name: "bool true", field: `{"type":"BOOLEAN"}`, cell: `"true"`, want: `true`},
		{name: "bool false", field: `{"type":"BOOL"}`, cell: `"false"`, want: `false`},
		{name: "bool t", field: `{"type":"BOOL"}`, cell: `"T"`, want: `true`},
		{name: "bool 1", field: `{"type":"BOOL"}`, cell: `"1"`, want: `true`},
		{name: "int64", field: `{"type":"INT64"}`, cell: `"-9223372036854775808"`, want: `-9223372036854775808`},
		{name: "float", field: `{"type":"FLOAT"}`, cell: `"1.5"`, want: `1.5`},
		{name: "float NaN", field: `{"type":"FLOAT64"}`, cell: `"NaN"`, want: `"NaN"`},
		{name: "float infinity", field: `{"type":"FLOAT64"}`, cell: `"-Infinity"`, want: `"-Infinity"`},
		{name: "numeric", field: `{"type":"NUMERIC"}`, cell: `"123.450"`, want: `"123.450"`},
		{name: "bignumeric", field: `{"type":"BIGNUMERIC"}`, cell: `"1e-38"`, want: `"1e-38"`},
		{name: "bytes", field: `{"type":"BYTES"}`, cell: `"AAE="`, want: `"AAE="`},
		{name: "date", field: `{"type":"DATE"}`, cell: `"2024-01-02"`, want: `"2024-01-02"`},
		{name: "timestamp", field: `{"type":"TIMESTAMP"}`, cell: `"1704164645000000"`, want: `"2024-01-02 03:04:05+00:00"`},
		{name: "timestamp with micros", field: `{"type":"TIMESTAMP"}`, cell: `"1704164645000120"`, want: `"2024-01-02 03:04:05.000120+00:00"`},
		{name: "timestamp before 1970", field: `{"type":"TIMESTAMP"}`, cell: `"-1"`, want: `"1969-12-31 23:59:59.999999+00:00"`},
		{name: "datetime", field: `{"type":"DATETIME"}`, cell: `"2024-01-02T03:04:05"`, want: `"2024-01-02 03:04:05"`},
		{name: "datetime with short fraction", field: `{"type":"DATETIME"}`, cell: `"2024-01-02T03:04:05.5"`, want: `"2024-01-02 03:04:05.500000"`},
		{name: "datetime with zero fraction", field: `{"type":"DATETIME"}`, cell: `"2024-01-02T03:04:05.000000"`, want: `"2024-01-02 03:04:05"`},
		{name: "time", field: `{"type":"TIME"}`, cell: `"03:04:05"`, want: `"03:04:05"`},
		{name: "time with micros", field: `{"type":"TIME"}`, cell: `"03:04:05.000001"`, want: `"03:04:05.000001"`},
		{name: "json", field: `{"type":"JSON"}`, cell: `"{\"a\":[1,true,null],\"big\":12345678901234567890}"`, want: `{"a":[1,true,null],"big":12345678901234567890}`},
		{name: "interval", field: `{"type":"INTERVAL"}`, cell: `"1-2 3 4:5:6"`, want: `"1-2 3 4:5:6"`},
		{name: "range", field: `{"type":"RANGE"}`, cell: `"[2024-01-01, UNBOUNDED)"`, want: `"[2024-01-01, UNBOUNDED)"`},
		{name: "geography", field: `{"type":"GEOGRAPHY"}`, cell: `"POINT(1 2)"`, want: `"POINT(1 2)"`},
		{name: "repeated", field: `{"type":"INTEGER","mode":"REPEATED"}`, cell: `[{"v":"1"},{"v":"2"}]`, want: `[1,2]`},
		{name: "empty repeated", field: `{"type":"INTEGER","mode":"REPEATED"}`, cell: `[]`, want: `[]`},
		{
			name:  "record",
			field: `{"type":"RECORD","fields":[{"name":"ts","type":"TIMESTAMP"},{"name":"tags","type":"STRING","mode":"REPEATED"}]}`,
			cell:  `{"f":[{"v":"0"},{"v":[{"v":"x"}]}]}`,
			want:  `{"ts":"1970-01-01 00:00:00+00:00","tags":["x"]}`,
		},
		{
			name:  "repeated struct",
			field: `{"type":"STRUCT","mode":"REPEATED","fields":[{"name":"n","type":"INT64"}]}`,
			cell:  `[{"v":{"f":[{"v":"1"}]}}]`,
			want:  `[{"n":1}]`,
		},
		{name: "null record", field: `{"type":"RECORD","fields":[{"name":"n","type":"INT64"}]}`, cell: `null`, want: `null`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var f field
			if err := json.Unmarshal([]byte(tc.field), &f); err != nil {
				t.Fatal(err)
			}
			f.Name = "c"
			row := decodeJSON(t, `{"f":[{"v":`+tc.cell+`}]}`).(map[string]any)

			got, err := rowValues([]field{f}, row)
			if err != nil {
				t.Fatalf("rowValues() error = %v", err)
			}
			want := map[string]any{"c": decodeJSON(t, tc.want)}
			if diff := cmp.Diff(want, roundTrip(t, got)); diff != "" {
				t.Errorf("rowValues() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRowValues_JSONKeepsLargeIntegers checks that an integer in a JSON
// cell keeps every digit, as Python's json.loads keeps it.
func TestRowValues_JSONKeepsLargeIntegers(t *testing.T) {
	row := decodeJSON(t, `{"f":[{"v":"{\"big\":12345678901234567891}"}]}`).(map[string]any)

	got, err := rowValues([]field{{Name: "c", Type: "JSON"}}, row)
	if err != nil {
		t.Fatalf("rowValues() error = %v", err)
	}
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if want := `{"c":{"big":12345678901234567891}}`; string(raw) != want {
		t.Errorf("rowValues() = %s, want %s", raw, want)
	}
}

func TestRowValues_Errors(t *testing.T) {
	tests := []struct {
		name  string
		field string
		cell  string
	}{
		{name: "float", field: `{"type":"FLOAT"}`, cell: `"x"`},
		{name: "timestamp", field: `{"type":"TIMESTAMP"}`, cell: `"2024-01-02"`},
		{name: "datetime", field: `{"type":"DATETIME"}`, cell: `"x"`},
		{name: "time", field: `{"type":"TIME"}`, cell: `"x"`},
		{name: "json", field: `{"type":"JSON"}`, cell: `"{"`},
		{name: "non-string cell", field: `{"type":"STRING"}`, cell: `1`},
		{name: "repeated item", field: `{"type":"INTEGER","mode":"REPEATED"}`, cell: `[{"v":"x"}]`},
		{name: "record field", field: `{"type":"RECORD","fields":[{"name":"n","type":"INT64"}]}`, cell: `{"f":[{"v":"x"}]}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var f field
			if err := json.Unmarshal([]byte(tc.field), &f); err != nil {
				t.Fatal(err)
			}
			row := decodeJSON(t, `{"f":[{"v":`+tc.cell+`}]}`).(map[string]any)

			if got, err := rowValues([]field{f}, row); err == nil {
				t.Errorf("rowValues() = %v, want an error", got)
			}
		})
	}
}

func TestNew_RejectsInvalidQueryConfig(t *testing.T) {
	tooMany := map[string]string{}
	for i := range 21 {
		tooMany[string(rune('a'+i))] = "v"
	}
	tests := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{name: "default project outside the compute project", cfg: Config{DefaultProjectID: "a", ComputeProjectID: "b"}, wantErr: "ComputeProjectID"},
		{name: "negative row limit", cfg: Config{MaxQueryResultRows: -1}, wantErr: "MaxQueryResultRows"},
		{name: "bytes billed below the minimum", cfg: Config{MaximumBytesBilled: 10<<20 - 1}, wantErr: "MaximumBytesBilled"},
		{name: "negative bytes billed", cfg: Config{MaximumBytesBilled: -1}, wantErr: "MaximumBytesBilled"},
		{name: "too many labels", cfg: Config{JobLabels: tooMany}, wantErr: "JobLabels"},
		{name: "empty label key", cfg: Config{JobLabels: map[string]string{"": "v"}}, wantErr: "JobLabels"},
		{name: "reserved label key", cfg: Config{JobLabels: map[string]string{"adk-bigquery-tool": "v"}}, wantErr: "reserved"},
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

func TestNew_AcceptsQueryConfigLimits(t *testing.T) {
	labels := map[string]string{}
	for i := range 20 {
		labels[string(rune('a'+i))] = "v"
	}
	f := newFakeBigQuery(t, nil)
	newTestToolset(t, Config{
		ClientOptions:      f.clientOptions(),
		DefaultProjectID:   "p",
		ComputeProjectID:   "p",
		MaximumBytesBilled: 10 << 20,
		JobLabels:          labels,
	})
}
