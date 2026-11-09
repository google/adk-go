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
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// field is a column of a query result schema.
type field struct {
	Name   string  `json:"name"`
	Type   string  `json:"type"`
	Mode   string  `json:"mode"`
	Fields []field `json:"fields"`
}

// rowValues turns a result row, which the REST API sends as {"f": [{"v":
// cell}, ...]}, into a map from column name to value.
//
// Values are what adk-python's execute_sql returns: the Python client turns
// each cell into a Python value, and execute_sql keeps it if it is JSON and
// otherwise turns it into its string form. So BOOL, INT64 and FLOAT64 cells
// become JSON booleans and numbers, JSON cells their parsed value, NUMERIC
// cells decimal strings, and DATE, TIME, DATETIME and TIMESTAMP cells strings
// such as "2024-01-02", "03:04:05", "2024-01-02 03:04:05" and
// "2024-01-02 03:04:05.123456+00:00". Where adk-python returns the string form
// of a Python object instead, the cell stays the string the REST API sent:
// BYTES (base64), INTERVAL and RANGE cells, and FLOAT64 NaN and infinities,
// which JSON cannot carry. adk-python also turns a whole STRUCT or ARRAY into
// one Python string when any value inside it is not JSON; here each value
// inside converts on its own.
func rowValues(fields []field, row map[string]any) (map[string]any, error) {
	cells, _ := row["f"].([]any)
	if len(cells) != len(fields) {
		return nil, fmt.Errorf("BigQuery returned a row of %d cells for %d columns", len(cells), len(fields))
	}
	out := make(map[string]any, len(fields))
	for i, f := range fields {
		cell, _ := cells[i].(map[string]any)
		v, err := cellValue(f, cell["v"])
		if err != nil {
			return nil, fmt.Errorf("column %s: %w", f.Name, err)
		}
		out[f.Name] = v
	}
	return out, nil
}

func cellValue(f field, v any) (any, error) {
	if f.Mode != "REPEATED" {
		return scalarValue(f, v)
	}
	items, _ := v.([]any)
	out := make([]any, 0, len(items))
	for _, item := range items {
		m, _ := item.(map[string]any)
		val, err := scalarValue(f, m["v"])
		if err != nil {
			return nil, err
		}
		out = append(out, val)
	}
	return out, nil
}

func scalarValue(f field, v any) (any, error) {
	if v == nil {
		return nil, nil
	}
	switch strings.ToUpper(f.Type) {
	case "RECORD", "STRUCT":
		m, _ := v.(map[string]any)
		return rowValues(f.Fields, m)
	}
	s, ok := v.(string)
	if !ok {
		return nil, fmt.Errorf("BigQuery returned a %T for a %s value", v, f.Type)
	}
	switch strings.ToUpper(f.Type) {
	case "BOOL", "BOOLEAN":
		switch strings.ToLower(s) {
		case "t", "true", "1":
			return true, nil
		}
		return false, nil
	case "INTEGER", "INT64":
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return n, nil
	case "FLOAT", "FLOAT64":
		x, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return s, nil
		}
		return x, nil
	case "TIMESTAMP":
		micros, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return pythonDateTime(time.UnixMicro(micros).UTC()) + "+00:00", nil
	case "DATETIME":
		t, err := time.Parse("2006-01-02T15:04:05.999999", s)
		if err != nil {
			return nil, err
		}
		return pythonDateTime(t), nil
	case "TIME":
		t, err := time.Parse("15:04:05.999999", s)
		if err != nil {
			return nil, err
		}
		return pythonTime(t), nil
	case "JSON":
		var val any
		d := json.NewDecoder(strings.NewReader(s))
		d.UseNumber()
		if err := d.Decode(&val); err != nil {
			return nil, err
		}
		return val, nil
	}
	return s, nil
}

// pythonDateTime formats t the way Python's str(datetime) does, without the
// UTC offset.
func pythonDateTime(t time.Time) string {
	return t.Format("2006-01-02 ") + pythonTime(t)
}

// pythonTime formats the time of day of t the way Python's str(time) does:
// microseconds are shown, as six digits, only when there are any.
func pythonTime(t time.Time) string {
	s := t.Format("15:04:05")
	if micros := t.Nanosecond() / 1000; micros != 0 {
		s += fmt.Sprintf(".%06d", micros)
	}
	return s
}
