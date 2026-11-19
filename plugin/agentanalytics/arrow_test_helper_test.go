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

package agentanalytics

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

// arrowEOS is the end-of-stream marker of the Arrow IPC stream format.
var arrowEOS = []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}

// decodeArrowRows decodes the rows of an AppendRowsRequest the way BigQuery
// reads them: the writer schema followed by the serialized record batch.
// Null values are left out of the returned rows, and so are empty
// content_parts lists.
func decodeArrowRows(t *testing.T, req *storagepb.AppendRowsRequest) []map[string]any {
	t.Helper()
	data := req.GetArrowRows()
	if data == nil {
		t.Fatal("AppendRowsRequest carries no Arrow rows")
	}
	var stream bytes.Buffer
	stream.Write(bytes.TrimSuffix(data.GetWriterSchema().GetSerializedSchema(), arrowEOS))
	stream.Write(data.GetRows().GetSerializedRecordBatch())
	stream.Write(arrowEOS)

	r, err := ipc.NewReader(&stream)
	if err != nil {
		t.Fatalf("ipc.NewReader() error: %v", err)
	}
	defer r.Release()
	var rows []map[string]any
	for r.Next() {
		rec := r.RecordBatch()
		for i := range int(rec.NumRows()) {
			row := map[string]any{}
			for j := range int(rec.NumCols()) {
				if v, ok := arrowValue(t, rec.Column(j), i); ok {
					row[rec.Schema().Field(j).Name] = v
				}
			}
			rows = append(rows, row)
		}
	}
	if err := r.Err(); err != nil {
		t.Fatalf("reading the Arrow stream: %v", err)
	}
	if got, want := int64(len(rows)), data.GetRows().GetRowCount(); got != want {
		t.Errorf("decoded %d rows, want RowCount = %d", got, want)
	}
	return rows
}

// arrowWriterSchemaColumns returns the top-level field names of the Arrow
// writer schema carried by req.
func arrowWriterSchemaColumns(t *testing.T, req *storagepb.AppendRowsRequest) []string {
	t.Helper()
	data := req.GetArrowRows()
	if data == nil {
		t.Fatal("AppendRowsRequest carries no Arrow rows")
	}
	var stream bytes.Buffer
	stream.Write(bytes.TrimSuffix(data.GetWriterSchema().GetSerializedSchema(), arrowEOS))
	stream.Write(arrowEOS)
	r, err := ipc.NewReader(&stream)
	if err != nil {
		t.Fatalf("ipc.NewReader() error: %v", err)
	}
	defer r.Release()
	var names []string
	for _, f := range r.Schema().Fields() {
		names = append(names, f.Name)
	}
	return names
}

// arrowValue returns element i of arr as a Go value; ok is false for nulls
// and empty lists.
func arrowValue(t *testing.T, arr arrow.Array, i int) (v any, ok bool) {
	t.Helper()
	if arr.IsNull(i) {
		return nil, false
	}
	switch a := arr.(type) {
	case *array.String:
		return a.Value(i), true
	case *array.Boolean:
		return a.Value(i), true
	case *array.Int64:
		return a.Value(i), true
	case *array.Timestamp:
		return a.Value(i).ToTime(arrow.Microsecond), true
	case *array.List:
		start, end := a.ValueOffsets(i)
		if start == end {
			return nil, false
		}
		elems := make([]map[string]any, 0, end-start)
		for k := start; k < end; k++ {
			elem, _ := arrowValue(t, a.ListValues(), int(k))
			m, _ := elem.(map[string]any)
			elems = append(elems, m)
		}
		return elems, true
	case *array.Struct:
		st := a.DataType().(*arrow.StructType)
		m := map[string]any{}
		for f := range a.NumField() {
			if fv, ok := arrowValue(t, a.Field(f), i); ok {
				m[st.Field(f).Name] = fv
			}
		}
		return m, true
	default:
		t.Fatalf("unexpected Arrow type %s", arr.DataType())
		return nil, false
	}
}

// collectRows decodes the requests received on ch until done reports that
// the rows needed by the test have arrived. Requests arrive in the order the
// rows were queued.
func collectRows(t *testing.T, ch <-chan *storagepb.AppendRowsRequest, done func([]map[string]any) bool) []map[string]any {
	t.Helper()
	var rows []map[string]any
	timeout := time.After(5 * time.Second)
	for !done(rows) {
		select {
		case req := <-ch:
			rows = append(rows, decodeArrowRows(t, req)...)
		case <-timeout:
			t.Fatalf("timed out waiting for rows; got event types %v", eventTypes(rows))
		}
	}
	return rows
}

// untilEventType returns a collectRows condition that holds once a row of
// eventType has arrived.
func untilEventType(eventType string) func([]map[string]any) bool {
	return func(rows []map[string]any) bool {
		return slices.ContainsFunc(rows, func(row map[string]any) bool {
			return row["event_type"] == eventType
		})
	}
}

// expectNoRequest fails the test if a request arrives on ch within d.
func expectNoRequest(t *testing.T, ch <-chan *storagepb.AppendRowsRequest, d time.Duration) {
	t.Helper()
	select {
	case req := <-ch:
		t.Fatalf("unexpected AppendRowsRequest with event types %v", eventTypes(decodeArrowRows(t, req)))
	case <-time.After(d):
	}
}

func eventTypes(rows []map[string]any) []string {
	types := make([]string, 0, len(rows))
	for _, row := range rows {
		s, _ := row["event_type"].(string)
		types = append(types, s)
	}
	return types
}

// rowsOfType returns the rows whose event_type is eventType.
func rowsOfType(rows []map[string]any, eventType string) []map[string]any {
	var out []map[string]any
	for _, row := range rows {
		if row["event_type"] == eventType {
			out = append(out, row)
		}
	}
	return out
}

// onlyRowOfType returns the single row of eventType.
func onlyRowOfType(t *testing.T, rows []map[string]any, eventType string) map[string]any {
	t.Helper()
	matches := rowsOfType(rows, eventType)
	if len(matches) != 1 {
		t.Fatalf("got %d %s rows, want 1; event types: %v", len(matches), eventType, eventTypes(rows))
	}
	return matches[0]
}

// jsonColumnValue decodes the JSON column of row, keeping numbers as
// json.Number. ok is false if the column is absent.
func jsonColumnValue(t *testing.T, row map[string]any, column string) (v any, ok bool) {
	t.Helper()
	raw, ok := row[column]
	if !ok {
		return nil, false
	}
	s, isString := raw.(string)
	if !isString {
		t.Fatalf("%s column is %T, want a JSON string", column, raw)
	}
	if err := unmarshalJSONPreserveNumbers([]byte(s), &v); err != nil {
		t.Fatalf("%s column is not valid JSON: %v\n%s", column, err, s)
	}
	return v, true
}

// jsonObjectColumn decodes a JSON column that must hold an object.
func jsonObjectColumn(t *testing.T, row map[string]any, column string) map[string]any {
	t.Helper()
	v, ok := jsonColumnValue(t, row, column)
	if !ok {
		t.Fatalf("%v row has no %s column", row["event_type"], column)
	}
	m, isObject := v.(map[string]any)
	if !isObject {
		t.Fatalf("%v row: %s column is %T, want a JSON object", row["event_type"], column, v)
	}
	return m
}

// lookup returns the value at path in a tree of decoded JSON objects.
func lookup(v any, path ...string) (any, bool) {
	for _, key := range path {
		m, ok := v.(map[string]any)
		if !ok {
			return nil, false
		}
		if v, ok = m[key]; !ok {
			return nil, false
		}
	}
	return v, true
}

// checkJSON reports a difference between got, a decoded JSON value, and
// want, a JSON document. Numbers are compared by value.
func checkJSON(t *testing.T, name string, got any, want string) {
	t.Helper()
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("%s: json.Marshal(%v) error: %v", name, got, err)
	}
	var gotV, wantV any
	if err := json.Unmarshal(gotJSON, &gotV); err != nil {
		t.Fatalf("%s: re-decoding %s: %v", name, gotJSON, err)
	}
	if err := json.Unmarshal([]byte(want), &wantV); err != nil {
		t.Fatalf("%s: invalid want JSON %s: %v", name, want, err)
	}
	if diff := cmp.Diff(wantV, gotV); diff != "" {
		t.Errorf("%s mismatch (-want +got):\n%s", name, diff)
	}
}

// checkJSONAt compares the value at path in obj with want, a JSON document.
func checkJSONAt(t *testing.T, obj map[string]any, want string, path ...string) {
	t.Helper()
	name := strings.Join(path, ".")
	got, ok := lookup(obj, path...)
	if !ok {
		t.Errorf("%s is missing; got %v", name, obj)
		return
	}
	checkJSON(t, name, got, want)
}

// recordingLogger is a Logger that keeps what is logged.
type recordingLogger struct {
	mu    sync.Mutex
	lines []string
}

func (l *recordingLogger) Printf(format string, v ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, fmt.Sprintf(format, v...))
}

func (l *recordingLogger) Println(v ...any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.lines = append(l.lines, fmt.Sprintln(v...))
}

func (l *recordingLogger) contains(substr string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.ContainsFunc(l.lines, func(line string) bool {
		return strings.Contains(line, substr)
	})
}

func (l *recordingLogger) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return strings.Join(l.lines, "\n")
}

// fakeTool is a tool.Tool for callback tests.
type fakeTool struct {
	name               string
	description        string
	longRunning        bool
	panicOnDescription bool
}

func (f *fakeTool) Name() string { return f.name }

func (f *fakeTool) Description() string {
	if f.panicOnDescription {
		panic("tool description unavailable")
	}
	return f.description
}

func (f *fakeTool) IsLongRunning() bool { return f.longRunning }

// declaredTool is a fakeTool with a function declaration.
type declaredTool struct {
	fakeTool
	decl *genai.FunctionDeclaration
}

func (d *declaredTool) Declaration() *genai.FunctionDeclaration { return d.decl }
