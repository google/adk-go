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
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"google.golang.org/api/googleapi"

	"google.golang.org/adk/v2/agent"
)

const (
	dialectPostgreSQL = "POSTGRESQL"

	// sessionCleanupTimeout bounds deleting a session after a call, which
	// runs even when the call's context has ended, as in adk-python.
	sessionCleanupTimeout = 30 * time.Second
)

// postgreSQLNotSupported is adk-python's error details for a tool that
// supports only the GoogleSQL dialect.
const postgreSQLNotSupported = "PostgreSQL dialect is not supported."

// statement is a SQL query with its parameters. A parameter without a type
// is typed by Spanner from the query, as with adk-python's untyped params.
type statement struct {
	SQL        string         `json:"sql"`
	Params     map[string]any `json:"params,omitempty"`
	ParamTypes map[string]any `json:"paramTypes,omitempty"`
}

// stringParams returns params with each of the given string values, all
// typed STRING, as adk-python passes them.
func stringParams(values map[string]string) (map[string]any, map[string]any) {
	params := make(map[string]any, len(values))
	types := make(map[string]any, len(values))
	for name, v := range values {
		params[name] = v
		types[name] = map[string]any{"code": "STRING"}
	}
	return params, types
}

// singleUseReadOnly is a strong read-only transaction for one query, what
// adk-python's database.snapshot() runs a query in.
var singleUseReadOnly = map[string]any{"singleUse": map[string]any{"readOnly": map[string]any{"strong": true}}}

// beginReadOnly starts a strong read-only transaction that later queries in
// the same session reuse by id, like adk-python's multi-use snapshot.
var beginReadOnly = map[string]any{"begin": map[string]any{"readOnly": map[string]any{"strong": true}}}

func useTransaction(id string) map[string]any { return map[string]any{"id": id} }

// spannerType is a Spanner value type as the REST API describes it.
type spannerType struct {
	Code             string       `json:"code"`
	ArrayElementType *spannerType `json:"arrayElementType"`
	StructType       *struct {
		Fields []field `json:"fields"`
	} `json:"structType"`
}

// field is a column of a result set or a field of a STRUCT.
type field struct {
	Name string      `json:"name"`
	Type spannerType `json:"type"`
}

// resultSet is the result of a query: its columns, its rows with each value
// converted by convertValue, and whether rows beyond the limit were left out.
type resultSet struct {
	fields        []field
	rows          [][]any
	truncated     bool
	transactionID string
}

// partialResultSet is one element of an executeStreamingSql response. Values
// hold the rows flattened, and a row can span elements. With ChunkedValue
// set, the last value is incomplete and continues in the next element.
type partialResultSet struct {
	Metadata *struct {
		RowType struct {
			Fields []field `json:"fields"`
		} `json:"rowType"`
		Transaction *struct {
			ID string `json:"id"`
		} `json:"transaction"`
	} `json:"metadata"`
	Values       []any `json:"values"`
	ChunkedValue bool  `json:"chunkedValue"`

	// Error is set on an element that ends a stream which failed after it
	// started.
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// dbSession is a Spanner session that a tool call queries in.
type dbSession struct {
	client     *client
	httpClient *http.Client
	name       string
}

// withSession creates a session in the database at path db, calls fn with it
// and deletes it. A role, when set, is the database role the session queries
// as. The session is deleted even when ctx has ended, and an error deleting it
// is returned, joined with any error from fn.
func (s *spanner) withSession(ctx agent.Context, db, role string, fn func(*dbSession) error) (err error) {
	httpClient, err := s.client.httpClientFor(ctx)
	if err != nil {
		return err
	}
	body := map[string]any{}
	if role != "" {
		body["session"] = map[string]any{"creatorRole": role}
	}
	var created struct {
		Name string `json:"name"`
	}
	if err := s.client.call(ctx, httpClient, http.MethodPost, db+"/sessions", body, &created); err != nil {
		return err
	}
	if created.Name == "" {
		return errors.New("the Spanner session has no name")
	}
	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), sessionCleanupTimeout)
		defer cancel()
		if delErr := s.client.call(cleanupCtx, httpClient, http.MethodDelete, created.Name, nil, nil); delErr != nil {
			err = errors.Join(err, fmt.Errorf("deleting the Spanner session: %w", delErr))
		}
	}()
	return fn(&dbSession{client: s.client, httpClient: httpClient, name: created.Name})
}

// dialect returns the dialect of the database at path db, which adk-python
// reads the same way, through the database admin API.
func (s *spanner) dialect(ctx agent.Context, db string) (string, error) {
	httpClient, err := s.client.httpClientFor(ctx)
	if err != nil {
		return "", err
	}
	var database struct {
		DatabaseDialect string `json:"databaseDialect"`
	}
	if err := s.client.call(ctx, httpClient, http.MethodGet, db, nil, &database); err != nil {
		return "", err
	}
	return database.DatabaseDialect, nil
}

// query runs stmt in the session in the transaction txn and returns at most
// limit rows, or every row when limit is negative.
func (ss *dbSession) query(ctx context.Context, stmt statement, txn map[string]any, limit int) (*resultSet, error) {
	body := struct {
		statement
		Transaction map[string]any `json:"transaction"`
	}{stmt, txn}
	resp, err := ss.client.send(ctx, ss.httpClient, http.MethodPost, ss.name+":executeStreamingSql", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	rs, err := readResultSet(json.NewDecoder(resp.Body), limit)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, err
	}
	return rs, nil
}

// readResultSet reads the stream of partial result sets an
// executeStreamingSql response holds. It stops reading after limit+1 rows,
// so a query is not read to its end only to find out that it has more rows
// than are returned.
func readResultSet(dec *json.Decoder, limit int) (*resultSet, error) {
	if tok, err := dec.Token(); err != nil || tok != json.Delim('[') {
		return nil, errors.New("malformed Spanner query response")
	}
	rs := &resultSet{}
	var (
		values  []any
		pending any
		chunked bool
	)
	for dec.More() {
		var part partialResultSet
		if err := dec.Decode(&part); err != nil {
			return nil, fmt.Errorf("decoding the Spanner query response: %w", err)
		}
		if part.Error != nil {
			return nil, &googleapi.Error{Code: part.Error.Code, Message: part.Error.Message}
		}
		if part.Metadata != nil {
			rs.fields = part.Metadata.RowType.Fields
			if part.Metadata.Transaction != nil {
				rs.transactionID = part.Metadata.Transaction.ID
			}
		}
		next := part.Values
		if chunked && len(next) > 0 {
			merged, err := mergeChunks(pending, next[0])
			if err != nil {
				return nil, err
			}
			next = append([]any{merged}, next[1:]...)
			chunked = false
		}
		if part.ChunkedValue && len(next) > 0 {
			pending, next = next[len(next)-1], next[:len(next)-1]
			chunked = true
		}
		values = append(values, next...)

		n := len(rs.fields)
		for n > 0 && len(values) >= n {
			if limit >= 0 && len(rs.rows) == limit {
				rs.truncated = true
				return rs, nil
			}
			row, err := convertRow(rs.fields, values[:n])
			if err != nil {
				return nil, err
			}
			rs.rows = append(rs.rows, row)
			values = values[n:]
		}
	}
	if chunked || len(values) > 0 {
		return nil, errors.New("the Spanner query response ended in the middle of a row")
	}
	return rs, nil
}

// mergeChunks joins the two parts of a value Spanner sent in two chunks:
// strings are concatenated, and lists are concatenated with the last element
// of a and the first of b merged when both are strings or both are lists.
func mergeChunks(a, b any) (any, error) {
	switch a := a.(type) {
	case string:
		if b, ok := b.(string); ok {
			return a + b, nil
		}
	case []any:
		b, ok := b.([]any)
		if !ok {
			break
		}
		if len(a) == 0 || len(b) == 0 {
			return append(a, b...), nil
		}
		last, first := a[len(a)-1], b[0]
		if mergeable(last, first) {
			merged, err := mergeChunks(last, first)
			if err != nil {
				return nil, err
			}
			out := append(a[:len(a)-1:len(a)-1], merged)
			return append(out, b[1:]...), nil
		}
		return append(a, b...), nil
	}
	return nil, fmt.Errorf("cannot merge chunked Spanner values of types %T and %T", a, b)
}

func mergeable(a, b any) bool {
	switch a.(type) {
	case string:
		_, ok := b.(string)
		return ok
	case []any:
		_, ok := b.([]any)
		return ok
	}
	return false
}

func convertRow(fields []field, values []any) ([]any, error) {
	row := make([]any, len(values))
	for i, v := range values {
		converted, err := convertValue(fields[i].Type, v)
		if err != nil {
			return nil, fmt.Errorf("column %d: %w", i, err)
		}
		row[i] = converted
	}
	return row, nil
}

// convertValue turns a value as the REST API sends it into what the tools
// return. BOOL, INT64, ENUM and FLOAT64 and FLOAT32 values become JSON
// booleans and numbers, JSON values their parsed value, and ARRAY and STRUCT
// values lists of converted values. Every other type stays the string the
// REST API sends: STRING, NUMERIC, DATE, TIMESTAMP, BYTES (base64) and so on,
// and NaN and the infinities, which JSON cannot carry. INT64 values and
// integers in JSON values beyond 2^53 in magnitude become decimal strings,
// since the tool result reaches the model as float64 numbers.
func convertValue(t spannerType, v any) (any, error) {
	if v == nil {
		return nil, nil
	}
	switch t.Code {
	case "BOOL":
		if _, ok := v.(bool); !ok {
			return nil, invalidValue(t)
		}
		return v, nil
	case "INT64", "ENUM":
		s, ok := v.(string)
		if !ok {
			return nil, invalidValue(t)
		}
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, invalidValue(t)
		}
		return exactInt(n), nil
	case "FLOAT64", "FLOAT32":
		switch v.(type) {
		case float64, string:
			return v, nil
		}
		return nil, invalidValue(t)
	case "JSON":
		s, ok := v.(string)
		if !ok {
			return nil, invalidValue(t)
		}
		var val any
		d := json.NewDecoder(strings.NewReader(s))
		d.UseNumber()
		if err := d.Decode(&val); err != nil {
			return nil, invalidValue(t)
		}
		return exactJSON(val), nil
	case "ARRAY":
		items, ok := v.([]any)
		if !ok || t.ArrayElementType == nil {
			return nil, invalidValue(t)
		}
		out := make([]any, len(items))
		for i, item := range items {
			converted, err := convertValue(*t.ArrayElementType, item)
			if err != nil {
				return nil, err
			}
			out[i] = converted
		}
		return out, nil
	case "STRUCT":
		items, ok := v.([]any)
		if !ok || t.StructType == nil || len(items) != len(t.StructType.Fields) {
			return nil, invalidValue(t)
		}
		return convertRow(t.StructType.Fields, items)
	}
	if _, ok := v.(string); !ok {
		return nil, invalidValue(t)
	}
	return v, nil
}

// maxExactInt is 2^53. The function tool sends its result through JSON
// decoded into float64, which cannot hold every integer beyond it.
const maxExactInt = 1 << 53

// exactInt returns n, or its decimal string when float64 cannot hold it.
func exactInt(n int64) any {
	if n > maxExactInt || n < -maxExactInt {
		return strconv.FormatInt(n, 10)
	}
	return n
}

// exactJSON replaces the integers in v, a value decoded with UseNumber, that
// float64 cannot hold with their decimal strings.
func exactJSON(v any) any {
	switch v := v.(type) {
	case json.Number:
		s := v.String()
		if strings.ContainsAny(s, ".eE") {
			return v
		}
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return s
		}
		return exactInt(n)
	case map[string]any:
		for k, x := range v {
			v[k] = exactJSON(x)
		}
	case []any:
		for i, x := range v {
			v[i] = exactJSON(x)
		}
	}
	return v
}

// invalidValue reports a value that does not match its column type. It does
// not quote the value, which is customer data.
func invalidValue(t spannerType) error {
	return fmt.Errorf("invalid Spanner %s value", t.Code)
}
