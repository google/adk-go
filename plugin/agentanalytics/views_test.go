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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestConfig_CreateViews(t *testing.T) {
	tests := []struct {
		name       string
		config     Config
		wantCreate bool
		wantPrefix string
	}{
		{
			name:       "DefaultConfig enables CreateViews with v prefix",
			config:     DefaultConfig(),
			wantCreate: true,
			wantPrefix: "v",
		},
		{
			name:       "Config literal enables CreateViews by default with v prefix",
			config:     applyDefaults(Config{Enabled: true}),
			wantCreate: true,
			wantPrefix: "v",
		},
		{
			name:       "Config literal with DisableCreateViews disables view creation",
			config:     applyDefaults(Config{Enabled: true, DisableCreateViews: true, ViewPrefix: "custom_v"}),
			wantCreate: false,
			wantPrefix: "custom_v",
		},
		{
			name: "DefaultConfig with CreateViews=false disables view creation",
			config: func() Config {
				c := DefaultConfig()
				c.CreateViews = false
				c.ViewPrefix = "custom_v"
				return c
			}(),
			wantCreate: false,
			wantPrefix: "custom_v",
		},
		{
			name: "DisableCreateViews overrides CreateViews",
			config: func() Config {
				c := DefaultConfig()
				c.DisableCreateViews = true
				return c
			}(),
			wantCreate: false,
			wantPrefix: "v",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.config.createViews(); got != tt.wantCreate {
				t.Errorf("createViews() = %v, want %v", got, tt.wantCreate)
			}
			if got := tt.config.ViewPrefix; got != tt.wantPrefix {
				t.Errorf("ViewPrefix = %q, want %q", got, tt.wantPrefix)
			}
		})
	}
}

func TestViewDefinitions_OrderAndSQLStructure(t *testing.T) {
	const (
		projectID = "my-domain.com:analytics-proj-1"
		datasetID = "agent_analytics"
		tableName = "agent_events"
	)

	defs, err := ViewDefinitions(projectID, datasetID, tableName, "")
	if err != nil {
		t.Fatalf("ViewDefinitions() error = %v", err)
	}

	wantEventTypes := []string{
		EventUserMessageReceived,
		EventLLMRequest,
		EventLLMResponse,
		EventLLMError,
		EventToolStarting,
		EventToolCompleted,
		EventToolError,
		EventAgentStarting,
		EventAgentCompleted,
		EventAgentError,
		EventInvocationStarting,
		EventInvocationCompleted,
		EventInvocationError,
		EventStateDelta,
		EventHITLCredentialRequest,
		EventHITLConfirmationRequest,
		EventHITLInputRequest,
		EventA2AInteraction,
		EventAgentResponse,
		EventAgentTransfer,
		EventEventCompaction,
		EventAgentStateCheckpoint,
		EventToolPaused,
		EventNodeOutput,
		EventNodeError,
	}

	if len(defs) != len(wantEventTypes) {
		t.Fatalf("len(ViewDefinitions) = %d, want %d", len(defs), len(wantEventTypes))
	}

	// Verify deterministic ordering across repeated calls.
	defs2, err := ViewDefinitions(projectID, datasetID, tableName, "")
	if err != nil {
		t.Fatalf("second ViewDefinitions() error = %v", err)
	}
	if diff := cmp.Diff(defs, defs2); diff != "" {
		t.Fatalf("ViewDefinitions() is not deterministic (-first +second):\n%s", diff)
	}

	for i, def := range defs {
		wantType := wantEventTypes[i]
		wantName := "v_" + strings.ToLower(wantType)
		if def.EventType != wantType {
			t.Errorf("defs[%d].EventType = %q, want %q", i, def.EventType, wantType)
		}
		if def.Name != wantName {
			t.Errorf("defs[%d].Name = %q, want %q", i, def.Name, wantName)
		}

		wantHeader := fmt.Sprintf("CREATE OR REPLACE VIEW `%s.%s.%s` AS\nSELECT\n", projectID, datasetID, wantName)
		if !strings.HasPrefix(def.Query, wantHeader) {
			t.Errorf("defs[%d] (%s) query does not start with %q:\n%s", i, def.Name, wantHeader, def.Query)
		}

		wantFooter := fmt.Sprintf("\nFROM\n  `%s.%s.%s`\nWHERE\n  event_type = '%s'\n", projectID, datasetID, tableName, wantType)
		if !strings.HasSuffix(def.Query, wantFooter) {
			t.Errorf("defs[%d] (%s) query does not end with %q:\n%s", i, def.Name, wantFooter, def.Query)
		}

		for _, commonCol := range viewCommonColumns {
			if !containsIdentifierToken(def.Query, commonCol) {
				t.Errorf("defs[%d] (%s) query missing common column %q:\n%s", i, def.Name, commonCol, def.Query)
			}
		}
	}

	// Verify v_user_message_received matches adk-python (common columns only) so
	// Go and Python writers sharing a dataset do not overwrite each other's view schema.
	userMsgView := defs[0]
	if _, gotAliases := extractExtraProjections(t, userMsgView.Query); len(gotAliases) != 0 {
		t.Errorf("v_user_message_received extra projections = %v, want empty (matching adk-python)", gotAliases)
	}
}

func TestViewDefinitions_CustomViewPrefixAndMultiTableDisjoint(t *testing.T) {
	defsA, err := ViewDefinitions("proj", "shared_ds", "events_a", "v_agent_a")
	if err != nil {
		t.Fatalf("ViewDefinitions(events_a) error = %v", err)
	}
	defsB, err := ViewDefinitions("proj", "shared_ds", "events_b", "v_agent_b")
	if err != nil {
		t.Fatalf("ViewDefinitions(events_b) error = %v", err)
	}

	seenNames := make(map[string]bool, len(defsA))
	for _, d := range defsA {
		if !strings.HasPrefix(d.Name, "v_agent_a_") {
			t.Errorf("defsA view name = %q, want prefix v_agent_a_", d.Name)
		}
		if !strings.Contains(d.Query, "`proj.shared_ds.events_a`") {
			t.Errorf("defsA view %s query does not target events_a:\n%s", d.Name, d.Query)
		}
		seenNames[d.Name] = true
	}
	for _, d := range defsB {
		if !strings.HasPrefix(d.Name, "v_agent_b_") {
			t.Errorf("defsB view name = %q, want prefix v_agent_b_", d.Name)
		}
		if !strings.Contains(d.Query, "`proj.shared_ds.events_b`") {
			t.Errorf("defsB view %s query does not target events_b:\n%s", d.Name, d.Query)
		}
		if seenNames[d.Name] {
			t.Errorf("view name %q collided between events_a and events_b", d.Name)
		}
	}
}

func TestViewDefinitions_IdentifierValidation(t *testing.T) {
	if _, err := ViewDefinitions("proj", "ds", "agent events", "v"); err != nil {
		t.Errorf("ViewDefinitions() with space in tableName error = %v, want nil", err)
	}

	tests := []struct {
		name       string
		projectID  string
		datasetID  string
		tableName  string
		viewPrefix string
		wantErr    string
	}{
		{
			name:      "empty projectID",
			projectID: "",
			datasetID: "ds",
			tableName: "tbl",
			wantErr:   "invalid BigQuery project ID",
		},
		{
			name:      "backtick in projectID",
			projectID: "proj` --",
			datasetID: "ds",
			tableName: "tbl",
			wantErr:   "invalid BigQuery project ID",
		},
		{
			name:      "empty datasetID",
			projectID: "proj",
			datasetID: "",
			tableName: "tbl",
			wantErr:   "invalid BigQuery dataset ID",
		},
		{
			name:      "dot in datasetID",
			projectID: "proj",
			datasetID: "ds.other",
			tableName: "tbl",
			wantErr:   "invalid BigQuery dataset ID",
		},
		{
			name:      "empty tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "semicolon in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "tbl;DROP",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:       "space in viewPrefix",
			projectID:  "proj",
			datasetID:  "ds",
			tableName:  "tbl",
			viewPrefix: "v prefix",
			wantErr:    "invalid BigQuery view prefix",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ViewDefinitions(tt.projectID, tt.datasetID, tt.tableName, tt.viewPrefix)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("ViewDefinitions() error = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestViewDefinitions_PayloadColumnDenylistProjection(t *testing.T) {
	byEventType := func(defs []ViewDefinition) map[string]ViewDefinition {
		m := make(map[string]ViewDefinition, len(defs))
		for _, d := range defs {
			m[d.EventType] = d
		}
		return m
	}

	t.Run("content denied drops content expressions", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ProjectID = "proj"
		cfg.DatasetID = "ds"
		cfg.TableName = "tbl"
		cfg.PayloadColumnDenylist = []string{columnContent}

		defs, err := ViewDefinitionsForConfig(cfg)
		if err != nil {
			t.Fatalf("ViewDefinitionsForConfig() error = %v", err)
		}
		methodDefs, err := cfg.ViewDefinitions()
		if err != nil {
			t.Fatalf("cfg.ViewDefinitions() error = %v", err)
		}
		if diff := cmp.Diff(defs, methodDefs); diff != "" {
			t.Errorf("ViewDefinitionsForConfig vs cfg.ViewDefinitions mismatch (-want +got):\n%s", diff)
		}
		m := byEventType(defs)

		if strings.Contains(m[EventLLMRequest].Query, "request_content") {
			t.Errorf("v_llm_request retained request_content when content is denied:\n%s", m[EventLLMRequest].Query)
		}
		if !strings.Contains(m[EventLLMRequest].Query, "AS model") {
			t.Errorf("v_llm_request dropped attributes-backed model column when only content is denied:\n%s", m[EventLLMRequest].Query)
		}
		if strings.Contains(m[EventLLMResponse].Query, "context_cache_hit_rate") || strings.Contains(m[EventLLMResponse].Query, "usage_prompt_tokens") {
			t.Errorf("v_llm_response retained content-dependent columns when content is denied:\n%s", m[EventLLMResponse].Query)
		}
		if !strings.Contains(m[EventLLMResponse].Query, "AS usage_cached_tokens") || !strings.Contains(m[EventLLMResponse].Query, "AS total_ms") {
			t.Errorf("v_llm_response dropped attributes/latency_ms columns when only content is denied:\n%s", m[EventLLMResponse].Query)
		}
	})

	t.Run("attributes denied drops attributes expressions", func(t *testing.T) {
		defs, err := viewDefinitionsWithDenied("proj", "ds", "tbl", "v", newSet(columnAttributes))
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
		}
		m := byEventType(defs)

		if strings.Contains(m[EventLLMRequest].Query, "AS model") || strings.Contains(m[EventLLMRequest].Query, "AS llm_config") {
			t.Errorf("v_llm_request retained attributes columns when attributes is denied:\n%s", m[EventLLMRequest].Query)
		}
		if !strings.Contains(m[EventLLMRequest].Query, "content AS request_content") {
			t.Errorf("v_llm_request dropped content column when only attributes is denied:\n%s", m[EventLLMRequest].Query)
		}
		if strings.Contains(m[EventStateDelta].Query, "state_delta\nFROM") {
			t.Errorf("v_state_delta retained state_delta when attributes is denied:\n%s", m[EventStateDelta].Query)
		}
		if strings.Contains(m[EventNodeOutput].Query, "AS node_path") {
			t.Errorf("v_node_output retained node_path when attributes is denied:\n%s", m[EventNodeOutput].Query)
		}
		if !strings.Contains(m[EventNodeOutput].Query, "content AS output") {
			t.Errorf("v_node_output dropped output when only attributes is denied:\n%s", m[EventNodeOutput].Query)
		}
	})

	t.Run("latency_ms denied drops latency expressions", func(t *testing.T) {
		defs, err := viewDefinitionsWithDenied("proj", "ds", "tbl", "v", newSet(columnLatencyMS))
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
		}
		m := byEventType(defs)

		if strings.Contains(m[EventLLMResponse].Query, "AS total_ms") || strings.Contains(m[EventLLMResponse].Query, "AS ttft_ms") {
			t.Errorf("v_llm_response retained latency_ms columns when latency_ms is denied:\n%s", m[EventLLMResponse].Query)
		}
		if !strings.Contains(m[EventLLMResponse].Query, "AS usage_prompt_tokens") {
			t.Errorf("v_llm_response dropped content columns when only latency_ms is denied:\n%s", m[EventLLMResponse].Query)
		}
		if strings.Contains(m[EventLLMError].Query, "AS total_ms") {
			t.Errorf("v_llm_error retained total_ms when latency_ms is denied:\n%s", m[EventLLMError].Query)
		}
	})

	t.Run("containsIdentifierToken respects token boundaries, aliases, and quoted JSON paths", func(t *testing.T) {
		if containsIdentifierToken("my_content", "content") {
			t.Error("containsIdentifierToken matched prefix substring my_content")
		}
		if containsIdentifierToken("content_parts", "content") {
			t.Error("containsIdentifierToken matched suffix substring content_parts")
		}
		if containsIdentifierToken("something", "") {
			t.Error("containsIdentifierToken matched empty identifier")
		}
		if referencesDeniedColumn("JSON_VALUE(attributes, '$.model') AS content", newSet(columnContent)) {
			t.Error("referencesDeniedColumn matched alias after AS")
		}
		if referencesDeniedColumn("JSON_VALUE(attributes, '$.content')", newSet(columnContent)) {
			t.Error("referencesDeniedColumn matched single-quoted JSON path literal '$.content'")
		}
		if referencesDeniedColumn("JSON_QUERY(attributes, '$.nested.content') AS x", newSet(columnContent)) {
			t.Error("referencesDeniedColumn matched single-quoted JSON path literal '$.nested.content'")
		}
	})
}

type sqlJSONExpr struct {
	fn     string // JSON_VALUE or JSON_QUERY
	column string // content, attributes, or latency_ms
	path   string // $.foo.bar or $
}

var sqlJSONExprRe = regexp.MustCompile(`(JSON_(?:VALUE|QUERY))\(\s*(content|attributes|latency_ms)\s*,\s*'(\$[^']*)'\s*\)`)

type sqlCastExpr struct {
	column   string
	path     string
	castType string // INT64, FLOAT64, TIMESTAMP, or BOOL
}

var sqlCastJSONValueRe = regexp.MustCompile(`(?:SAFE_)?CAST\(\s*JSON_VALUE\(\s*(content|attributes|latency_ms)\s*,\s*'(\$[^']*)'\s*\)\s+AS\s+(INT64|FLOAT64|BOOL)\s*\)`)

func extractSQLCastExprs(query string) []sqlCastExpr {
	matches := sqlCastJSONValueRe.FindAllStringSubmatch(query, -1)
	out := make([]sqlCastExpr, 0, len(matches))
	for _, m := range matches {
		out = append(out, sqlCastExpr{
			column:   m[1],
			path:     m[2],
			castType: m[3],
		})
	}
	return out
}

func verifyCastValue(t *testing.T, viewName, eventType string, cast sqlCastExpr, rawVal any) {
	t.Helper()
	s := fmt.Sprint(rawVal)
	switch cast.castType {
	case "INT64":
		if _, err := strconv.ParseInt(s, 10, 64); err != nil {
			t.Errorf("view %s (%s): CAST(JSON_VALUE(%s, '%s') AS INT64) value %q (%T) does not parse as INT64: %v", viewName, eventType, cast.column, cast.path, s, rawVal, err)
		}
	case "FLOAT64":
		if _, err := strconv.ParseFloat(s, 64); err != nil {
			t.Errorf("view %s (%s): CAST(JSON_VALUE(%s, '%s') AS FLOAT64) value %q (%T) does not parse as FLOAT64: %v", viewName, eventType, cast.column, cast.path, s, rawVal, err)
		}
	case "BOOL":
		if _, err := strconv.ParseBool(s); err != nil {
			t.Errorf("view %s (%s): CAST(JSON_VALUE(%s, '%s') AS BOOL) value %q (%T) does not parse as BOOL: %v", viewName, eventType, cast.column, cast.path, s, rawVal, err)
		}
	}
}

func extractSQLJSONExprs(query string) []sqlJSONExpr {
	matches := sqlJSONExprRe.FindAllStringSubmatch(query, -1)
	out := make([]sqlJSONExpr, 0, len(matches))
	for _, m := range matches {
		out = append(out, sqlJSONExpr{
			fn:     m[1],
			column: m[2],
			path:   m[3],
		})
	}
	return out
}

// resolveJSONPath resolves a BigQuery JSON path (such as "$.usage.prompt" or
// `$.a2a_metadata."a2a:task_id"`) against a decoded Go JSON value.
func resolveJSONPath(doc any, path string) (any, bool) {
	if doc == nil {
		return nil, false
	}
	if path == "$" {
		return doc, true
	}
	if !strings.HasPrefix(path, "$.") {
		return nil, false
	}
	segments := splitJSONPathSegments(path[2:])
	cur := doc
	for _, seg := range segments {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		cur, ok = m[seg]
		if !ok || cur == nil {
			return nil, false
		}
	}
	return cur, true
}

func splitJSONPathSegments(s string) []string {
	var segs []string
	var cur strings.Builder
	inQuotes := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		switch {
		case ch == '"':
			inQuotes = !inQuotes
		case ch == '.' && !inQuotes:
			segs = append(segs, cur.String())
			cur.Reset()
		default:
			cur.WriteByte(ch)
		}
	}
	segs = append(segs, cur.String())
	return segs
}

// extractExtraProjections parses the SELECT clause of a CREATE OR REPLACE VIEW
// query, verifies the leading common columns, and returns the extra projection
// expressions and their output column aliases in order.
func extractExtraProjections(t *testing.T, query string) (exprs, aliases []string) {
	t.Helper()
	const selectMarker = "SELECT\n  "
	const fromMarker = "\nFROM\n"
	start := strings.Index(query, selectMarker)
	end := strings.Index(query, fromMarker)
	if start < 0 || end < 0 || end <= start+len(selectMarker) {
		t.Fatalf("failed to parse SELECT clause from query:\n%s", query)
	}
	selectBody := query[start+len(selectMarker) : end]
	cols := strings.Split(selectBody, ",\n  ")
	if len(cols) < len(viewCommonColumns) {
		t.Fatalf("SELECT has %d columns, want at least %d:\n%s", len(cols), len(viewCommonColumns), query)
	}
	if diff := cmp.Diff(viewCommonColumns, cols[:len(viewCommonColumns)]); diff != "" {
		t.Fatalf("common columns prefix mismatch (-want +got):\n%s", diff)
	}
	extra := cols[len(viewCommonColumns):]
	if len(extra) == 0 {
		return nil, nil
	}
	aliases = make([]string, 0, len(extra))
	for _, col := range extra {
		alias := strings.TrimSpace(col)
		if idx := strings.LastIndex(col, " AS "); idx >= 0 {
			alias = strings.TrimSpace(col[idx+4:])
		}
		aliases = append(aliases, alias)
	}
	return extra, aliases
}

var subMicroTimestampRe = regexp.MustCompile(`(\.\d{6})\d+`)

// parseBigQueryTimestampString emulates BigQuery's SAFE_CAST(string AS TIMESTAMP),
// which accepts RFC3339 timestamps with at most 6 fractional second digits
// (microsecond precision) and returns an error on 7..9 nanosecond digits.
func parseBigQueryTimestampString(s string) (time.Time, error) {
	if subMicroTimestampRe.MatchString(s) {
		return time.Time{}, fmt.Errorf("BigQuery TIMESTAMP cast rejects >6 fractional digits in %q", s)
	}
	return time.Parse(time.RFC3339Nano, s)
}

// evalCompactionTimestampSQL simulates BigQuery's evaluation of a compaction
// seconds or window_start/window_end projection expression against a decoded
// content JSON map, enforcing BigQuery's conversion rules where CAST(... AS FLOAT64)
// rejects RFC3339 strings and SAFE_CAST(... AS TIMESTAMP) rejects >6 fractional
// digits unless normalized by REGEXP_REPLACE(..., r'(\.\d{6})\d+', r'\1').
func evalCompactionTimestampSQL(expr string, content map[string]any) (float64, time.Time, bool, error) {
	sqlPart := expr
	if idx := strings.LastIndex(expr, " AS "); idx >= 0 {
		sqlPart = strings.TrimSpace(expr[:idx])
	}
	isWindow := false
	inner := sqlPart
	if strings.HasPrefix(sqlPart, "TIMESTAMP_MICROS(CAST(") && strings.HasSuffix(sqlPart, " * 1000000 AS INT64))") {
		isWindow = true
		inner = strings.TrimSuffix(strings.TrimPrefix(sqlPart, "TIMESTAMP_MICROS(CAST("), " * 1000000 AS INT64))")
	}

	exprs := extractSQLJSONExprs(inner)
	if len(exprs) == 0 {
		return 0, time.Time{}, false, fmt.Errorf("no JSON_VALUE expression found in %q", expr)
	}
	path := exprs[0].path
	rawVal, ok := resolveJSONPath(content, path)
	if !ok {
		return 0, time.Time{}, false, fmt.Errorf("path %q missing in content %v", path, content)
	}
	// JSON_VALUE extracts a scalar as its SQL STRING representation.
	scalarStr := fmt.Sprint(rawVal)

	wantCoalesce := fmt.Sprintf(
		`COALESCE(SAFE_CAST(JSON_VALUE(content, '%s') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '%s'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6)`,
		path, path,
	)
	var secs float64
	if inner == wantCoalesce {
		if f, err := strconv.ParseFloat(scalarStr, 64); err == nil {
			secs = f
		} else {
			normalized := subMicroTimestampRe.ReplaceAllString(scalarStr, "$1")
			ts, err := parseBigQueryTimestampString(normalized)
			if err != nil {
				return 0, time.Time{}, isWindow, fmt.Errorf("COALESCE failed to parse %q as FLOAT64 or BigQuery TIMESTAMP: %w", scalarStr, err)
			}
			secs = float64(ts.UnixMicro()) / 1e6
		}
	} else {
		// Without the REGEXP_REPLACE + COALESCE expression, BigQuery either fails
		// CAST(... AS FLOAT64) on strings or rejects >6 fractional digits in TIMESTAMP.
		if f, err := strconv.ParseFloat(scalarStr, 64); err == nil {
			secs = f
		} else if ts, err := parseBigQueryTimestampString(scalarStr); err == nil && strings.Contains(inner, "AS TIMESTAMP") {
			secs = float64(ts.UnixMicro()) / 1e6
		} else {
			return 0, time.Time{}, isWindow, fmt.Errorf("expression %q failed to evaluate %q under BigQuery conversion rules", expr, scalarStr)
		}
	}
	if isWindow {
		micros := int64(secs * 1e6)
		return secs, time.UnixMicro(micros).UTC(), true, nil
	}
	return secs, time.Time{}, false, nil
}

func TestViews_JSONPathsMatchEmittedPayloads(t *testing.T) {
	defs, err := ViewDefinitions("test-project", "test-dataset", "test-table", "v")
	if err != nil {
		t.Fatalf("ViewDefinitions() error = %v", err)
	}
	defsByType := make(map[string]ViewDefinition, len(defs))
	for _, d := range defs {
		defsByType[d.EventType] = d
	}

	// Part 1: Emit events through live Go plugin callbacks (including all
	// Go-produced session.Event signal types via OnEventCallback), decode the
	// Arrow IPC RecordBatch rows, and verify that the view JSON paths resolve to
	// valid scalars/objects and parse under BigQuery CAST rules.
	t.Run("live plugin callbacks emit JSON keys matching view definitions", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		mCtx.functionCallID = "call-42"

		lookupTool := &declaredTool{
			fakeTool: fakeTool{name: "lookup", description: "Looks up facts"},
			decl: &genai.FunctionDeclaration{
				Name:                 "lookup",
				Description:          "Looks up facts",
				ParametersJsonSchema: map[string]any{"type": "object"},
			},
		}

		llmReq := &model.LLMRequest{
			Model:    "gemini-2.5-pro",
			Contents: []*genai.Content{textContent("user", "what is 6*7?")},
			Config: &genai.GenerateContentConfig{
				SystemInstruction: textContent("", "answer concisely"),
				Temperature:       genai.Ptr[float32](0.2),
			},
			Tools: map[string]any{"lookup": lookupTool},
		}
		llmRes := &model.LLMResponse{
			Content: textContent("model", "42"),
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        100,
				CandidatesTokenCount:    10,
				TotalTokenCount:         115,
				CachedContentTokenCount: 20,
				ThoughtsTokenCount:      5,
				ToolUsePromptTokenCount: 15,
			},
			ModelVersion: "gemini-2.5-pro-001",
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}

		compactionStart := time.Date(2026, 10, 10, 1, 0, 0, 500_000_000, time.UTC)
		compactionEnd := time.Date(2026, 10, 10, 1, 5, 0, 250_000_000, time.UTC)
		type compactionRecord struct {
			StartTimestamp   time.Time
			EndTimestamp     time.Time
			CompactedContent *genai.Content
		}
		compactionRecordVal := &compactionRecord{
			StartTimestamp:   compactionStart,
			EndTimestamp:     compactionEnd,
			CompactedContent: textContent("model", "compacted summary"),
		}
		compactionEvent := &session.Event{
			ID:     "ev-native-compaction",
			Author: "root-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": compactionRecordVal,
					"compaction":     compactionRecordVal,
				},
			},
		}

		// Map-shaped CustomMetadata["adk_compaction"] with 9-digit nanosecond RFC3339
		// strings (as produced by time.Time.MarshalJSON when sub-microsecond
		// nanoseconds are non-zero) passes through formatCompactionValue unchanged.
		nanoCompactionMap := map[string]any{
			"start_timestamp":   "2026-10-10T01:00:00.500000123Z",
			"end_timestamp":     "2026-10-10T01:05:00.250000987Z",
			"compacted_content": "compacted summary",
		}
		nanoStringCompactionEvent := &session.Event{
			ID:     "ev-nano-string-compaction",
			Author: "root-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": nanoCompactionMap,
					"compaction":     nanoCompactionMap,
				},
			},
		}

		checkpointAndTransferEvent := &session.Event{
			ID:     "ev-state-transfer-ckpt",
			Author: "root-agent",
			Branch: "root.agent",
			Actions: session.EventActions{
				StateDelta:      map[string]any{"counter": 1},
				TransferToAgent: "specialist-agent",
			},
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_agent_state": map[string]any{"step": 2},
					"end_of_agent":    true,
				},
			},
		}

		hitlAndPauseEvent := &session.Event{
			ID:                 "ev-hitl-pause",
			Author:             "root-agent",
			LongRunningToolIDs: []string{"call-cred", "call-confirm", "call-input", "call-pause"},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-cred",
							Name: "adk_request_credential",
							Args: map[string]any{"scope": "read"},
						}},
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-confirm",
							Name: "adk_request_confirmation",
							Args: map[string]any{"action": "delete"},
						}},
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-input",
							Name: "adk_request_input",
							Args: map[string]any{"prompt": "name?"},
						}},
						{FunctionCall: &genai.FunctionCall{
							ID:   "call-pause",
							Name: "deploy_cluster",
							Args: map[string]any{"region": "us-central1"},
						}},
					},
				},
			},
		}

		a2aNodeOutAndFinalEvent := &session.Event{
			ID:       "ev-a2a-node-final",
			Author:   "root-agent",
			Branch:   "root.agent",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@1"},
			Output:   map[string]any{"result": "step done"},
			LLMResponse: model.LLMResponse{
				Content: textContent("model", "42"),
				CustomMetadata: map[string]any{
					"a2a:task_id":    "task-1",
					"a2a:context_id": "ctx-1",
					"a2a:request":    map[string]any{"method": "message/send"},
					"a2a:response":   map[string]any{"status": "completed"},
				},
			},
		}

		nodeErrorEvent := &session.Event{
			ID:       "ev-node-err",
			Author:   "root-agent",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@2"},
			LLMResponse: model.LLMResponse{
				ErrorCode:    "STEP_TIMEOUT",
				ErrorMessage: "step timed out",
			},
		}

		var errs callbackErrors
		errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "what is 6*7?")))
		errs.add(p.BeforeRunCallback()(mCtx))
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeModelCallback()(mCtx, llmReq))
		errs.add(p.AfterModelCallback()(mCtx, llmRes, nil))
		errs.add(p.BeforeModelCallback()(mCtx, llmReq))
		errs.add(p.OnModelErrorCallback()(mCtx, llmReq, errors.New("rate limited")))
		errs.add(p.BeforeToolCallback()(mCtx, lookupTool, map[string]any{"q": "6*7"}))
		errs.add(p.AfterToolCallback()(mCtx, lookupTool, map[string]any{"q": "6*7"}, map[string]any{"ok": true}, nil))
		errs.add(p.BeforeToolCallback()(mCtx, lookupTool, map[string]any{"q": "6*7"}))
		errs.add(p.OnToolErrorCallback()(mCtx, lookupTool, map[string]any{"q": "6*7"}, errors.New("timeout")))
		errs.add(p.OnEventCallback()(mCtx, compactionEvent))
		errs.add(p.OnEventCallback()(mCtx, nanoStringCompactionEvent))
		errs.add(p.OnEventCallback()(mCtx, checkpointAndTransferEvent))
		errs.add(p.OnEventCallback()(mCtx, hitlAndPauseEvent))
		errs.add(p.OnEventCallback()(mCtx, a2aNodeOutAndFinalEvent))
		errs.add(p.OnEventCallback()(mCtx, nodeErrorEvent))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

		// Verify callback-emitted JSON paths (content, attributes, and latency_ms)
		// for each event type populated by live Go callbacks.
		callbackChecks := []struct {
			eventType string
			wantPaths map[string][]string // column -> JSON paths expected on callback-emitted row
		}{
			{
				eventType: EventLLMRequest,
				wantPaths: map[string][]string{
					columnAttributes: {"$.model", "$.llm_config", "$.tools"},
				},
			},
			{
				eventType: EventLLMResponse,
				wantPaths: map[string][]string{
					columnContent: {"$.response", "$.usage.prompt", "$.usage.completion", "$.usage.total"},
					columnAttributes: {
						"$.usage_metadata.cached_content_token_count",
						"$.usage_metadata.thoughts_token_count",
						"$.usage_metadata.tool_use_prompt_token_count",
						"$.model_version",
						"$.usage_metadata",
						"$.cache_type",
						"$.finish_reason",
					},
					columnLatencyMS: {"$.total_ms"},
				},
			},
			{
				eventType: EventLLMError,
				wantPaths: map[string][]string{
					columnLatencyMS: {"$.total_ms"},
				},
			},
			{
				eventType: EventToolStarting,
				wantPaths: map[string][]string{
					columnContent: {"$.tool", "$.args", "$.tool_origin"},
				},
			},
			{
				eventType: EventToolCompleted,
				wantPaths: map[string][]string{
					columnContent:    {"$.tool", "$.result", "$.tool_origin"},
					columnAttributes: {"$.adk.function_call_id"},
					columnLatencyMS:  {"$.total_ms"},
				},
			},
			{
				eventType: EventToolError,
				wantPaths: map[string][]string{
					columnContent:   {"$.tool", "$.args", "$.tool_origin"},
					columnLatencyMS: {"$.total_ms"},
				},
			},
			{
				eventType: EventAgentCompleted,
				wantPaths: map[string][]string{
					columnLatencyMS: {"$.total_ms"},
				},
			},
			{
				eventType: EventStateDelta,
				wantPaths: map[string][]string{
					columnAttributes: {"$.state_delta"},
				},
			},
			{
				eventType: EventHITLCredentialRequest,
				wantPaths: map[string][]string{
					columnContent: {"$.tool", "$.args"},
				},
			},
			{
				eventType: EventHITLConfirmationRequest,
				wantPaths: map[string][]string{
					columnContent: {"$.tool", "$.args"},
				},
			},
			{
				eventType: EventHITLInputRequest,
				wantPaths: map[string][]string{
					columnContent: {"$.tool", "$.args"},
				},
			},
			{
				eventType: EventA2AInteraction,
				wantPaths: map[string][]string{
					columnAttributes: {
						`$.a2a_metadata."a2a:task_id"`,
						`$.a2a_metadata."a2a:context_id"`,
						`$.a2a_metadata."a2a:request"`,
						`$.a2a_metadata."a2a:response"`,
					},
				},
			},
			{
				eventType: EventAgentResponse,
				wantPaths: map[string][]string{
					columnContent:    {"$.response"},
					columnAttributes: {"$.source_event_id", "$.source_event_author", "$.source_event_branch"},
				},
			},
			{
				eventType: EventAgentTransfer,
				wantPaths: map[string][]string{
					columnContent:    {"$.from_agent", "$.to_agent"},
					columnAttributes: {"$.adk.source_event_id"},
				},
			},
			{
				eventType: EventEventCompaction,
				wantPaths: map[string][]string{
					columnContent: {"$.start_timestamp", "$.end_timestamp", "$.compacted_content"},
				},
			},
			{
				eventType: EventAgentStateCheckpoint,
				wantPaths: map[string][]string{
					columnContent:    {"$.agent_state", "$.end_of_agent"},
					columnAttributes: {"$.adk.source_event_id"},
				},
			},
			{
				eventType: EventToolPaused,
				wantPaths: map[string][]string{
					columnContent:    {"$.tool", "$.args"},
					columnAttributes: {"$.adk.pause_kind", "$.adk.function_call_id"},
				},
			},
			{
				eventType: EventNodeOutput,
				wantPaths: map[string][]string{
					columnAttributes: {"$.adk.node.path", "$.adk.node.run_id", "$.adk.node.parent_run_id"},
				},
			},
			{
				eventType: EventNodeError,
				wantPaths: map[string][]string{
					columnContent:    {"$.error_code"},
					columnAttributes: {"$.adk.node.path", "$.adk.node.run_id", "$.adk.node.parent_run_id"},
				},
			},
		}

		for _, tc := range callbackChecks {
			matchingRows := rowsOfType(rows, tc.eventType)
			if len(matchingRows) == 0 {
				t.Fatalf("no %s rows found; event types: %v", tc.eventType, eventTypes(rows))
			}
			row := matchingRows[0]
			viewDef := defsByType[tc.eventType]
			viewExprs := extractSQLJSONExprs(viewDef.Query)
			castExprs := extractSQLCastExprs(viewDef.Query)
			for col, paths := range tc.wantPaths {
				decoded, ok := jsonColumnValue(t, row, col)
				if !ok {
					t.Fatalf("%s row missing %s column", tc.eventType, col)
				}
				for _, path := range paths {
					var matchedExpr *sqlJSONExpr
					for i := range viewExprs {
						if viewExprs[i].column == col && viewExprs[i].path == path {
							matchedExpr = &viewExprs[i]
							break
						}
					}
					if matchedExpr == nil {
						t.Fatalf("%s view query does not contain (%s, %s)", tc.eventType, col, path)
					}
					val, ok := resolveJSONPath(decoded, path)
					if !ok {
						t.Errorf("%s callback-emitted %s did not contain path %q; payload = %v", tc.eventType, col, path, decoded)
						continue
					}
					if matchedExpr.fn == "JSON_VALUE" && path != "$" {
						switch val.(type) {
						case map[string]any, []any:
							t.Errorf("%s view %s: JSON_VALUE(%s, '%s') targeted non-scalar %T (%v)", tc.eventType, viewDef.Name, col, path, val, val)
						}
					}
					for _, ce := range castExprs {
						if ce.column == col && ce.path == path {
							verifyCastValue(t, viewDef.Name, tc.eventType, ce, val)
						}
					}
				}
			}
		}

		// Verify that v_event_compaction's 4 timestamp expressions (start_seconds,
		// end_seconds, window_start, window_end) evaluate cleanly under BigQuery
		// conversion rules for:
		// 1) the native Go callback payload (float64 epoch seconds matching adk-python),
		// 2) microsecond RFC3339 timestamp strings, and
		// 3) 9-digit nanosecond RFC3339 timestamp strings emitted via OnEventCallback
		//    and decoded from the Arrow IPC RecordBatch.
		wantStartSec := float64(compactionStart.UnixMicro()) / 1e6
		wantEndSec := float64(compactionEnd.UnixMicro()) / 1e6

		compactionRows := rowsOfType(rows, EventEventCompaction)
		if len(compactionRows) != 2 {
			t.Fatalf("got %d EVENT_COMPACTION rows, want 2", len(compactionRows))
		}
		compactionContent := jsonObjectColumn(t, compactionRows[0], columnContent)
		startNum, ok := compactionContent["start_timestamp"].(json.Number)
		if !ok {
			t.Fatalf("native EVENT_COMPACTION start_timestamp = %v (%T), want json.Number", compactionContent["start_timestamp"], compactionContent["start_timestamp"])
		}
		if got, err := startNum.Float64(); err != nil || got != wantStartSec {
			t.Fatalf("native EVENT_COMPACTION start_timestamp = %v (err=%v), want %v", got, err, wantStartSec)
		}
		endNum, ok := compactionContent["end_timestamp"].(json.Number)
		if !ok {
			t.Fatalf("native EVENT_COMPACTION end_timestamp = %v (%T), want json.Number", compactionContent["end_timestamp"], compactionContent["end_timestamp"])
		}
		if got, err := endNum.Float64(); err != nil || got != wantEndSec {
			t.Fatalf("native EVENT_COMPACTION end_timestamp = %v (err=%v), want %v", got, err, wantEndSec)
		}

		nanoCompactionContent := jsonObjectColumn(t, compactionRows[1], columnContent)
		if got := nanoCompactionContent["start_timestamp"]; got != "2026-10-10T01:00:00.500000123Z" {
			t.Fatalf("nano EVENT_COMPACTION start_timestamp = %v, want 9-digit RFC3339Nano string", got)
		}

		compactionExprs, compactionAliases := extractExtraProjections(t, defsByType[EventEventCompaction].Query)
		exprByAlias := make(map[string]string, len(compactionAliases))
		for i, alias := range compactionAliases {
			exprByAlias[alias] = compactionExprs[i]
		}
		for _, alias := range []string{"start_seconds", "end_seconds", "window_start", "window_end"} {
			if !strings.Contains(exprByAlias[alias], `REGEXP_REPLACE(`) || !strings.Contains(exprByAlias[alias], `r'(\.\d{6})\d+'`) {
				t.Errorf("v_event_compaction %s expression missing sub-microsecond REGEXP_REPLACE normalization: %s", alias, exprByAlias[alias])
			}
		}

		rfc3339Content := map[string]any{
			"start_timestamp":   compactionStart.UTC().Format(time.RFC3339Nano),
			"end_timestamp":     compactionEnd.UTC().Format(time.RFC3339Nano),
			"compacted_content": "compacted summary",
		}

		for _, representation := range []struct {
			name    string
			content map[string]any
		}{
			{name: "native Go callback float64 epoch seconds", content: compactionContent},
			{name: "RFC3339 microsecond timestamp strings", content: rfc3339Content},
			{name: "Arrow-decoded 9-digit nanosecond RFC3339 timestamp strings", content: nanoCompactionContent},
		} {
			gotStartSec, _, isWin, err := evalCompactionTimestampSQL(exprByAlias["start_seconds"], representation.content)
			if err != nil || isWin || gotStartSec != wantStartSec {
				t.Errorf("%s: start_seconds eval = (%v, isWin=%v, err=%v), want (%v, false, nil)", representation.name, gotStartSec, isWin, err, wantStartSec)
			}
			gotEndSec, _, isWin, err := evalCompactionTimestampSQL(exprByAlias["end_seconds"], representation.content)
			if err != nil || isWin || gotEndSec != wantEndSec {
				t.Errorf("%s: end_seconds eval = (%v, isWin=%v, err=%v), want (%v, false, nil)", representation.name, gotEndSec, isWin, err, wantEndSec)
			}
			_, gotWinStart, isWin, err := evalCompactionTimestampSQL(exprByAlias["window_start"], representation.content)
			if err != nil || !isWin || !gotWinStart.Equal(compactionStart) {
				t.Errorf("%s: window_start eval = (%v, isWin=%v, err=%v), want (%v, true, nil)", representation.name, gotWinStart, isWin, err, compactionStart)
			}
			_, gotWinEnd, isWin, err := evalCompactionTimestampSQL(exprByAlias["window_end"], representation.content)
			if err != nil || !isWin || !gotWinEnd.Equal(compactionEnd) {
				t.Errorf("%s: window_end eval = (%v, isWin=%v, err=%v), want (%v, true, nil)", representation.name, gotWinEnd, isWin, err, compactionEnd)
			}
		}
	})

	// Part 2: Verify projected column aliases, Arrow IPC round-trip, JSON_VALUE
	// scalar requirements, and CAST(... AS INT64|FLOAT64|BOOL) parsing across all
	// 25 views, including the Python-only event types (EventAgentError,
	// EventInvocationError) and Python-only optional fields (static_instruction on
	// EventAgentStarting, cache_metadata and time_to_first_token_ms on
	// EventLLMResponse, and error_traceback on EventAgentError / EventInvocationError).
	t.Run("all 25 view SQL projections and CAST expressions resolve across Arrow-encoded rows", func(t *testing.T) {
		pythonOnlyEventTypes := map[string]bool{
			EventAgentError:      true,
			EventInvocationError: true,
		}
		_ = pythonOnlyEventTypes

		wantAliases := map[string][]string{
			EventUserMessageReceived: nil,
			EventLLMRequest:          {"model", "request_content", "llm_config", "tools"},
			EventLLMResponse: {
				"response",
				"usage_prompt_tokens",
				"usage_completion_tokens",
				"usage_total_tokens",
				"usage_cached_tokens",
				"usage_thinking_tokens",
				"usage_tool_use_tokens",
				"context_cache_hit_rate",
				"total_ms",
				"ttft_ms",
				"model_version",
				"usage_metadata",
				"cache_metadata",
				"cache_type",
				"finish_reason",
			},
			EventLLMError:      {"total_ms"},
			EventToolStarting:  {"tool_name", "tool_args", "tool_origin"},
			EventToolCompleted: {"tool_name", "tool_result", "tool_origin", "total_ms", "pause_kind", "function_call_id"},
			EventToolError:     {"tool_name", "tool_args", "tool_origin", "total_ms"},
			EventAgentStarting: {
				"agent_instruction",
				"instruction_source",
				"static_instruction",
				"agent_description",
				"model",
				"sub_agents",
			},
			EventAgentCompleted:          {"total_ms"},
			EventAgentError:              {"total_ms", "error_traceback"},
			EventInvocationStarting:      nil,
			EventInvocationCompleted:     nil,
			EventInvocationError:         {"error_traceback"},
			EventStateDelta:              {"state_delta"},
			EventHITLCredentialRequest:   {"tool_name", "tool_args"},
			EventHITLConfirmationRequest: {"tool_name", "tool_args"},
			EventHITLInputRequest:        {"tool_name", "tool_args"},
			EventA2AInteraction:          {"response_content", "a2a_task_id", "a2a_context_id", "a2a_request", "a2a_response"},
			EventAgentResponse:           {"response_text", "source_event_id", "source_event_author", "source_event_branch"},
			EventAgentTransfer:           {"from_agent", "to_agent", "source_event_id"},
			EventEventCompaction:         {"start_seconds", "end_seconds", "window_start", "window_end", "compacted_content"},
			EventAgentStateCheckpoint:    {"agent_state", "agent_state_type", "end_of_agent", "source_event_id"},
			EventToolPaused:              {"tool_name", "tool_args", "pause_kind", "function_call_id"},
			EventNodeOutput:              {"node_path", "node_run_id", "node_parent_run_id", "output"},
			EventNodeError:               {"node_path", "node_run_id", "node_parent_run_id", "error_code"},
		}
		if len(wantAliases) != len(defs) {
			t.Fatalf("len(wantAliases) = %d, want %d", len(wantAliases), len(defs))
		}

		ctx := context.Background()
		cfg := applyDefaults(DefaultConfig())
		cfg.RetryConfig.MaxRetries = 0

		bp, err := NewBatchProcessor(ctx, nil, "test_stream", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		mStream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}}
		bp.streamWriter.stream = mStream

		ap := &analyticsPlugin{
			config:         cfg,
			processor:      bp,
			customMetadata: newCustomMetadataFilter(cfg.CustomMetadataAllowlist),
		}
		mCtx := &mockInvocationContext{
			ctx:            ctx,
			agentName:      "root-agent",
			sessionID:      "sess-1",
			invocationID:   "inv-1",
			userID:         "user-1",
			appName:        "app-1",
			branch:         "root",
			functionCallID: "call-1",
		}

		makeRow := func(eventType string, d *eventData, latency map[string]any) map[string]any {
			row := ap.buildRow(mCtx, identityOf(mCtx), eventType, d)
			if latency != nil {
				b, err := json.Marshal(latency)
				if err != nil {
					t.Fatalf("json.Marshal(latency): %v", err)
				}
				row[columnLatencyMS] = string(b)
			}
			return row
		}

		nodeSourceEvent := &session.Event{
			ID:       "src-ev-1",
			Author:   "root-agent",
			Branch:   "root",
			NodeInfo: &session.NodeInfo{Path: "wf@0/step@1"},
		}

		sampleRows := []map[string]any{
			makeRow(EventUserMessageReceived, &eventData{
				content: textContent("user", "hello"),
				author:  userAuthor,
			}, nil),
			makeRow(EventLLMRequest, &eventData{
				content: &model.LLMRequest{
					Model:    "gemini-2.5-pro",
					Contents: []*genai.Content{textContent("user", "hello")},
				},
				attributes: map[string]any{
					"model":      "gemini-2.5-pro",
					"llm_config": map[string]any{"temperature": 0.5},
					"tools":      []map[string]any{{"name": "lookup"}},
				},
			}, nil),
			makeRow(EventLLMResponse, &eventData{
				content: map[string]any{
					"response": "text: 'hi'",
					"usage":    map[string]any{"prompt": 10, "completion": 5, "total": 15},
				},
				attributes: map[string]any{
					"model_version":  "gemini-2.5-pro-001",
					"finish_reason":  "STOP",
					"cache_type":     "EXPLICIT",
					"cache_metadata": map[string]any{"cache_name": "caches/123"},
					"usage_metadata": map[string]any{
						"prompt_token_count":          10,
						"candidates_token_count":      5,
						"total_token_count":           15,
						"cached_content_token_count":  4,
						"thoughts_token_count":        2,
						"tool_use_prompt_token_count": 3,
					},
				},
			}, map[string]any{"total_ms": 120, "time_to_first_token_ms": 35}),
			makeRow(EventLLMError, &eventData{
				failed: true,
				errMsg: "quota exceeded",
			}, map[string]any{"total_ms": 45}),
			makeRow(EventToolStarting, &eventData{
				content:        toolContent("lookup", "args", map[string]any{"q": "go"}, ToolOriginLocal),
				functionCallID: "call-1",
			}, nil),
			func() map[string]any {
				row := makeRow(EventToolCompleted, &eventData{
					content:        toolContent("lookup", "result", map[string]any{"ok": true}, ToolOriginLocal),
					functionCallID: "call-1",
				}, map[string]any{"total_ms": 25})
				// Inject adk.pause_kind alongside adk.function_call_id as emitted on paused/resumed tool completions.
				var attrs map[string]any
				_ = json.Unmarshal([]byte(row[columnAttributes].(string)), &attrs)
				attrs["adk"].(map[string]any)["pause_kind"] = "long_running"
				b, _ := json.Marshal(attrs)
				row[columnAttributes] = string(b)
				return row
			}(),
			makeRow(EventToolError, &eventData{
				content:        toolContent("lookup", "args", map[string]any{"q": "go"}, ToolOriginLocal),
				failed:         true,
				errMsg:         "tool failed",
				functionCallID: "call-1",
			}, map[string]any{"total_ms": 30}),
			makeRow(EventAgentStarting, &eventData{
				content: map[string]any{"text_summary": "You are a helpful agent"},
				attributes: map[string]any{
					"instruction_source": "static",
					"static_instruction": "You are a helpful agent",
					"agent_description":  "Root agent",
					"model":              "gemini-2.5-pro",
					"sub_agents":         []string{"helper"},
				},
			}, nil),
			makeRow(EventAgentCompleted, &eventData{}, map[string]any{"total_ms": 200}),
			makeRow(EventAgentError, &eventData{
				content: map[string]any{"error_traceback": "Traceback ..."},
				failed:  true,
				errMsg:  "agent crashed",
			}, map[string]any{"total_ms": 210}),
			makeRow(EventInvocationStarting, &eventData{}, nil),
			makeRow(EventInvocationCompleted, &eventData{}, nil),
			makeRow(EventInvocationError, &eventData{
				content: map[string]any{"error_traceback": "Traceback ..."},
				failed:  true,
				errMsg:  "invocation crashed",
			}, nil),
			makeRow(EventStateDelta, &eventData{
				attributes: map[string]any{
					"state_delta": map[string]any{"counter": 1},
				},
			}, nil),
			makeRow(EventHITLCredentialRequest, &eventData{
				content: map[string]any{"tool": "adk_request_credential", "args": map[string]any{"scope": "read"}},
			}, nil),
			makeRow(EventHITLConfirmationRequest, &eventData{
				content: map[string]any{"tool": "adk_request_confirmation", "args": map[string]any{"action": "delete"}},
			}, nil),
			makeRow(EventHITLInputRequest, &eventData{
				content: map[string]any{"tool": "adk_request_input", "args": map[string]any{"prompt": "name?"}},
			}, nil),
			makeRow(EventA2AInteraction, &eventData{
				content: map[string]any{"text_summary": "remote answer"},
				attributes: map[string]any{
					"a2a_metadata": map[string]any{
						"a2a:task_id":    "task-1",
						"a2a:context_id": "ctx-1",
						"a2a:request":    map[string]any{"method": "message/send"},
						"a2a:response":   map[string]any{"status": "completed"},
					},
				},
			}, nil),
			makeRow(EventAgentResponse, &eventData{
				content: map[string]any{"response": "final answer"},
				attributes: map[string]any{
					"source_event_id":     "src-ev-1",
					"source_event_author": "root-agent",
					"source_event_branch": "root",
				},
			}, nil),
			makeRow(EventAgentTransfer, &eventData{
				content: map[string]any{
					"from_agent": "root-agent",
					"to_agent":   "sub-agent",
				},
				sourceEvent: nodeSourceEvent,
			}, nil),
			makeRow(EventEventCompaction, &eventData{
				content: map[string]any{
					"start_timestamp":   1700000000.5,
					"end_timestamp":     1700000010.5,
					"compacted_content": map[string]any{"text_summary": "summary"},
				},
			}, nil),
			makeRow(EventAgentStateCheckpoint, &eventData{
				content: map[string]any{
					"agent_state":  map[string]any{"step": 2},
					"end_of_agent": true,
				},
				sourceEvent: nodeSourceEvent,
			}, nil),
			func() map[string]any {
				row := makeRow(EventToolPaused, &eventData{
					content:        map[string]any{"tool": "long_op", "args": map[string]any{"id": 1}},
					functionCallID: "call-1",
				}, nil)
				var attrs map[string]any
				_ = json.Unmarshal([]byte(row[columnAttributes].(string)), &attrs)
				attrs["adk"].(map[string]any)["pause_kind"] = "long_running"
				b, _ := json.Marshal(attrs)
				row[columnAttributes] = string(b)
				return row
			}(),
			makeRow(EventNodeOutput, &eventData{
				content:     map[string]any{"result": "step done"},
				sourceEvent: nodeSourceEvent,
			}, nil),
			makeRow(EventNodeError, &eventData{
				content:     map[string]any{"error_code": "STEP_TIMEOUT"},
				sourceEvent: nodeSourceEvent,
				failed:      true,
				errMsg:      "step timed out",
			}, nil),
		}

		if err := bp.writeBatch(ctx, sampleRows); err != nil {
			t.Fatalf("writeBatch() error = %v", err)
		}
		sent := mStream.requests()
		if len(sent) != 1 {
			t.Fatalf("Send count = %d, want 1", len(sent))
		}
		decodedRows := decodeArrowRows(t, sent[0])

		totalExprsChecked := 0
		for _, def := range defs {
			wantExtraAliases, known := wantAliases[def.EventType]
			if !known {
				t.Fatalf("view %s (%s) missing from wantAliases", def.Name, def.EventType)
			}
			_, gotAliases := extractExtraProjections(t, def.Query)
			if diff := cmp.Diff(wantExtraAliases, gotAliases); diff != "" {
				t.Errorf("view %s (%s) projected aliases mismatch (-want +got):\n%s", def.Name, def.EventType, diff)
			}

			row := onlyRowOfType(t, decodedRows, def.EventType)
			exprs := extractSQLJSONExprs(def.Query)
			castExprs := extractSQLCastExprs(def.Query)
			for _, expr := range exprs {
				totalExprsChecked++
				colVal, ok := jsonColumnValue(t, row, expr.column)
				if !ok {
					t.Errorf("view %s (%s): decoded row missing column %q for path %q", def.Name, def.EventType, expr.column, expr.path)
					continue
				}
				val, ok := resolveJSONPath(colVal, expr.path)
				if !ok {
					t.Errorf("view %s (%s): path %q in column %q resolved to NULL; column JSON = %v", def.Name, def.EventType, expr.path, expr.column, colVal)
					continue
				}
				if expr.fn == "JSON_VALUE" {
					// JSON_VALUE in BigQuery extracts scalars (string, number, bool), not objects/arrays.
					switch val.(type) {
					case map[string]any, []any:
						if expr.path != "$" {
							t.Errorf("view %s (%s): JSON_VALUE(%s, '%s') targeted non-scalar %T (%v); use JSON_QUERY instead", def.Name, def.EventType, expr.column, expr.path, val, val)
						}
					}
				}
				for _, ce := range castExprs {
					if ce.column == expr.column && ce.path == expr.path {
						verifyCastValue(t, def.Name, def.EventType, ce, val)
					}
				}
			}
		}
		if totalExprsChecked == 0 {
			t.Fatal("extractSQLJSONExprs matched 0 expressions across all 25 views")
		}
	})
}

type fakeViewRunner struct {
	mu      sync.Mutex
	names   []string
	queries []string
	errFor  map[string]error
	onRun   func(ctx context.Context, viewName string) error
}

func (r *fakeViewRunner) RunViewDDL(ctx context.Context, viewName, query string) error {
	r.mu.Lock()
	r.names = append(r.names, viewName)
	r.queries = append(r.queries, query)
	err := r.errFor[viewName]
	onRun := r.onRun
	r.mu.Unlock()
	if onRun != nil {
		if runErr := onRun(ctx, viewName); runErr != nil {
			return runErr
		}
	}
	return err
}

func (r *fakeViewRunner) executedNames() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.names))
	copy(out, r.names)
	return out
}

func TestViews_AsyncStartupAndCloseLifecycle(t *testing.T) {
	newWriteClient := func(t *testing.T) *bqstorage.BigQueryWriteClient {
		t.Helper()
		ctx := context.Background()
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		wc, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}
		return wc
	}

	t.Run("CreateViews=true runs asynchronously without blocking constructor and creates all 25 views", func(t *testing.T) {
		ctx := context.Background()
		firstJobStarted := make(chan struct{})
		releaseJobs := make(chan struct{})
		allJobsDone := make(chan struct{})
		var jobCount atomic.Int32
		var mu sync.Mutex
		var jobQueries []string

		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						b, _ := io.ReadAll(r.Body)
						var payload struct {
							Configuration struct {
								Query struct {
									Query string `json:"query"`
								} `json:"query"`
							} `json:"configuration"`
						}
						_ = json.Unmarshal(b, &payload)
						mu.Lock()
						jobQueries = append(jobQueries, payload.Configuration.Query.Query)
						mu.Unlock()

						n := jobCount.Add(1)
						if n == 1 {
							close(firstJobStarted)
						}
						<-releaseJobs
						if int(n) == len(eventViewSpecs) {
							close(allJobsDone)
						}
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"job-1"},"status":{"state":"DONE"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}

		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test-dataset"
		cfg.TableName = "test-table"
		cfg.CreateViews = true

		// Use a short startup context that we cancel immediately after construction
		// to verify background view creation uses context.WithoutCancel(ctx).
		startupCtx, cancelStartup := context.WithCancel(ctx)
		p, err := NewBigQueryAgentAnalyticsPluginWithClients(startupCtx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		cancelStartup()

		// Constructor returned while releaseJobs is still closed!
		select {
		case <-firstJobStarted:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for async view creation goroutine to start")
		}

		close(releaseJobs)
		select {
		case <-allJobsDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for all %d view DDL jobs; got %d", len(eventViewSpecs), jobCount.Load())
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		if got := int(jobCount.Load()); got != len(eventViewSpecs) {
			t.Fatalf("view DDL job count = %d, want %d", got, len(eventViewSpecs))
		}
		mu.Lock()
		defer mu.Unlock()
		if !strings.Contains(jobQueries[0], "v_user_message_received") {
			t.Errorf("first view query = %q, want v_user_message_received", jobQueries[0])
		}
		if !strings.Contains(jobQueries[len(jobQueries)-1], "v_node_error") {
			t.Errorf("last view query = %q, want v_node_error", jobQueries[len(jobQueries)-1])
		}
	})

	t.Run("Close cancels in-flight async view creation cleanly", func(t *testing.T) {
		ctx := context.Background()
		inFlight := make(chan struct{})
		canceled := make(chan struct{})

		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						select {
						case <-inFlight:
						default:
							close(inFlight)
						}
						<-r.Context().Done()
						select {
						case <-canceled:
						default:
							close(canceled)
						}
						return nil, r.Context().Err()
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}

		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test-dataset"
		cfg.TableName = "test-table"
		cfg.CreateViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}

		select {
		case <-inFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for view creation request to enter RoundTrip")
		}

		closeDone := make(chan error, 1)
		go func() {
			closeDone <- p.Close()
		}()

		select {
		case err := <-closeDone:
			if err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("Close() hung waiting for async view creation")
		}

		select {
		case <-canceled:
		default:
			t.Error("expected in-flight view DDL request context to be canceled by Close()")
		}
		if logger.contains("failed to create view") || logger.contains("view creation finished") || logger.contains("call SyncViews to retry") {
			t.Errorf("Close() cancellation should not log failure or retry summary, got:\n%s", logger.String())
		}
	})

	t.Run("CreateViews=false and DisableCreateViews=true execute zero view queries", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			mutate func(*Config)
		}{
			{name: "CreateViews=false", mutate: func(c *Config) { c.CreateViews = false }},
			{name: "DisableCreateViews=true", mutate: func(c *Config) { c.DisableCreateViews = true }},
		} {
			t.Run(tc.name, func(t *testing.T) {
				ctx := context.Background()
				var jobsCalled atomic.Int32
				httpClient := &http.Client{
					Transport: &mockTransport{
						roundTrip: func(r *http.Request) (*http.Response, error) {
							if strings.Contains(r.URL.Path, "/jobs") {
								jobsCalled.Add(1)
							}
							return &http.Response{
								StatusCode: http.StatusOK,
								Body:       io.NopCloser(strings.NewReader("{}")),
							}, nil
						},
					},
				}
				bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
				if err != nil {
					t.Fatalf("bq.NewClient: %v", err)
				}
				t.Cleanup(func() { _ = bqClient.Close() })

				cfg := DefaultConfig()
				cfg.ProjectID = "test-project"
				cfg.DatasetID = "test-dataset"
				cfg.TableName = "test-table"
				tc.mutate(&cfg)

				p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
				if err != nil {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("Close() error = %v", err)
				}
				if got := jobsCalled.Load(); got != 0 {
					t.Errorf("/jobs requests = %d, want 0", got)
				}
			})
		}
	})

	t.Run("HTTP 403 logs warning without failing startup and HTTP 409 is tolerated", func(t *testing.T) {
		ctx := context.Background()
		var jobIdx atomic.Int32
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						switch jobIdx.Add(1) {
						case 1:
							return &http.Response{
								StatusCode: http.StatusConflict,
								Body:       io.NopCloser(strings.NewReader(`{"error":{"code":409,"message":"Already Exists"}}`)),
							}, nil
						case 2:
							return &http.Response{
								StatusCode: http.StatusForbidden,
								Body:       io.NopCloser(strings.NewReader(`{"error":{"code":403,"message":"Access Denied"}}`)),
							}, nil
						default:
							return &http.Response{
								StatusCode: http.StatusOK,
								Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"job-ok"},"status":{"state":"DONE"}}`)),
							}, nil
						}
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}

		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test-dataset"
		cfg.TableName = "test-table"
		cfg.CreateViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		// Wait for all 25 view jobs to finish, then close.
		deadline := time.Now().Add(5 * time.Second)
		for int(jobIdx.Load()) < len(eventViewSpecs) && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		if got := int(jobIdx.Load()); got != len(eventViewSpecs) {
			t.Errorf("executed %d view DDL jobs, want %d (must continue after 409 and 403)", got, len(eventViewSpecs))
		}
		if logger.contains("v_user_message_received") {
			t.Errorf("409 Conflict on v_user_message_received should be benign, but logged:\n%s", logger.String())
		}
		if !logger.contains("failed to create view v_llm_request") {
			t.Errorf("expected warning for 403 on v_llm_request, got logs:\n%s", logger.String())
		}
		if !logger.contains("view creation finished: 24 created, 1 failed, 0 skipped of 25; call SyncViews to retry") {
			t.Errorf("expected summary log line for partial failure, got logs:\n%s", logger.String())
		}
	})
}

func TestBigQueryAgentAnalyticsPlugin_SyncViews(t *testing.T) {
	ctx := context.Background()

	var nilPlugin *BigQueryAgentAnalyticsPlugin
	if err := nilPlugin.SyncViews(ctx); err == nil {
		t.Error("nilPlugin.SyncViews() error = nil, want error")
	}

	disabledP, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, Config{Enabled: false}, nil, nil)
	if err != nil {
		t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients(disabled) error = %v", err)
	}
	disabledBQAAP, ok := AsBigQueryAgentAnalyticsPlugin(disabledP)
	if !ok {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(disabledP) = false, want true")
	}
	if err := disabledBQAAP.SyncViews(ctx); err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Errorf("disabledBQAAP.SyncViews() error = %v, want disabled error", err)
	}

	// Plugin with nil bqClient -> nil runner.
	noBQP, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
	noBQBQAAP, ok := AsBigQueryAgentAnalyticsPlugin(noBQP)
	if !ok {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(noBQP) = false, want true")
	}
	savedRunner := noBQBQAAP.runner
	noBQBQAAP.runner = nil
	if err := noBQBQAAP.SyncViews(ctx); err == nil || !strings.Contains(err.Error(), "not configured") {
		t.Errorf("SyncViews() with nil runner error = %v, want not configured error", err)
	}
	noBQBQAAP.runner = savedRunner

	t.Run("executes all 25 views synchronously and tolerates 409 Conflict", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.CreateViews = false
			c.ViewPrefix = "v_sync"
		})
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{
			errFor: map[string]error{
				"v_sync_user_message_received": &googleapi.Error{Code: http.StatusConflict, Message: "conflict"},
				"v_sync_llm_request":           status.Error(codes.AlreadyExists, "already exists"),
			},
		}
		bqaap.runner = fakeRunner

		if err := bqaap.SyncViews(ctx); err != nil {
			t.Fatalf("SyncViews() error = %v, want nil", err)
		}
		if got := len(fakeRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() executed %d views, want %d", got, len(eventViewSpecs))
		}
	})

	t.Run("applies per-statement viewDDLTimeout and preserves shorter caller deadline", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		// 1. Under context.Background(), every RunViewDDL call must receive a deadline <= viewDDLTimeout.
		var sawDeadlineCount int
		bqaap.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, viewName string) error {
				deadline, hasDeadline := runCtx.Deadline()
				if !hasDeadline {
					t.Errorf("RunViewDDL(%s): missing per-statement deadline under background context", viewName)
					return nil
				}
				rem := time.Until(deadline)
				if rem <= 0 || rem > viewDDLTimeout {
					t.Errorf("RunViewDDL(%s): time.Until(deadline) = %v, want in (0, %v]", viewName, rem, viewDDLTimeout)
				}
				sawDeadlineCount++
				return nil
			},
		}
		if err := bqaap.SyncViews(context.Background()); err != nil {
			t.Fatalf("SyncViews(Background) error = %v", err)
		}
		if sawDeadlineCount != len(eventViewSpecs) {
			t.Errorf("inspected %d deadlines, want %d", sawDeadlineCount, len(eventViewSpecs))
		}

		// 2. Under a shorter caller deadline (50ms), the tighter caller deadline must win over viewDDLTimeout.
		const shortTimeout = 50 * time.Millisecond
		shortCtx, cancel := context.WithTimeout(context.Background(), shortTimeout)
		defer cancel()
		bqaap.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, viewName string) error {
				deadline, hasDeadline := runCtx.Deadline()
				if !hasDeadline {
					t.Errorf("RunViewDDL(%s): missing deadline under short caller context", viewName)
					return nil
				}
				if rem := time.Until(deadline); rem > shortTimeout {
					t.Errorf("RunViewDDL(%s): time.Until(deadline) = %v, want <= %v (caller deadline must win)", viewName, rem, shortTimeout)
				}
				return nil
			},
		}
		if err := bqaap.SyncViews(shortCtx); err != nil {
			t.Fatalf("SyncViews(shortCtx) error = %v", err)
		}
	})

	t.Run("aggregates errors across views and stops on context cancellation", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		fakeRunner := &fakeViewRunner{
			errFor: map[string]error{
				"v_llm_request": errors.New("ddl syntax error"),
			},
		}
		bqaap.runner = fakeRunner

		err := bqaap.SyncViews(ctx)
		if err == nil || !strings.Contains(err.Error(), "v_llm_request") {
			t.Errorf("SyncViews() error = %v, want error naming v_llm_request", err)
		}
		if got := len(fakeRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() executed %d views on single error, want %d", got, len(eventViewSpecs))
		}

		canceledCtx, cancel := context.WithCancel(ctx)
		cancel()
		fakeRunner2 := &fakeViewRunner{}
		bqaap.runner = fakeRunner2
		if err := bqaap.SyncViews(canceledCtx); !errors.Is(err, context.Canceled) {
			t.Errorf("SyncViews(canceledCtx) error = %v, want context.Canceled", err)
		}
		if got := len(fakeRunner2.executedNames()); got != 0 {
			t.Errorf("SyncViews(canceledCtx) executed %d views, want 0", got)
		}
	})

	t.Run("bqViewRunner waits on running jobs and surfaces job status errors", func(t *testing.T) {
		var pollCalls atomic.Int32
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(bytes.NewBufferString(`{"jobReference":{"projectId":"test-project","jobId":"job-running"},"status":{"state":"RUNNING"}}`)),
						}, nil
					}
					if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/jobs/job-running") {
						pollCalls.Add(1)
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(bytes.NewBufferString(`{"jobReference":{"projectId":"test-project","jobId":"job-running"},"status":{"state":"DONE","errorResult":{"reason":"invalidQuery","message":"bad view query"}}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}
		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		runner := bqViewRunner{client: bqClient}
		err = runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `p.d.v_test` AS SELECT 1")
		if err == nil || !strings.Contains(err.Error(), "bad view query") {
			t.Errorf("RunViewDDL() error = %v, want error containing 'bad view query'", err)
		}
		if pollCalls.Load() == 0 {
			t.Error("expected job.Wait to poll /jobs/job-running when initial state is RUNNING")
		}
	})

	t.Run("bqViewRunner skips CREATE OR REPLACE VIEW when existing view query matches", func(t *testing.T) {
		var jobsPosted atomic.Int32
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test-dataset/tables/v_test") {
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"type":"VIEW","view":{"query":"SELECT 1"}}`)),
						}, nil
					}
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						jobsPosted.Add(1)
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"job-1"},"status":{"state":"DONE"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}
		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		runner := bqViewRunner{client: bqClient, datasetID: "test-dataset"}
		if err := runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test-dataset.v_test` AS\nSELECT 1"); err != nil {
			t.Fatalf("RunViewDDL(unchanged) error = %v", err)
		}
		if got := jobsPosted.Load(); got != 0 {
			t.Errorf("jobsPosted for unchanged view = %d, want 0", got)
		}

		if err := runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test-dataset.v_test` AS\nSELECT 2"); err != nil {
			t.Fatalf("RunViewDDL(changed) error = %v", err)
		}
		if got := jobsPosted.Load(); got != 1 {
			t.Errorf("jobsPosted for changed view = %d, want 1", got)
		}
	})

	t.Run("Close preempts in-flight SyncViews call", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		syncInFlight := make(chan struct{})
		var once sync.Once
		bqaap.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, _ string) error {
				once.Do(func() { close(syncInFlight) })
				<-runCtx.Done()
				return runCtx.Err()
			},
		}

		syncDone := make(chan error, 1)
		go func() {
			syncDone <- bqaap.SyncViews(context.Background())
		}()

		select {
		case <-syncInFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for SyncViews to enter RunViewDDL")
		}

		closeDone := make(chan error, 1)
		go func() {
			closeDone <- p.Close()
		}()

		select {
		case err := <-closeDone:
			if err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("Close() blocked on in-flight SyncViews")
		}

		select {
		case err := <-syncDone:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("SyncViews() preempted by Close() error = %v, want context.Canceled", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("SyncViews() did not return after Close()")
		}
	})

	t.Run("SyncViews after Close returns closed error without touching runner", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{}
		bqaap.runner = fakeRunner

		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		err := bqaap.SyncViews(ctx)
		if err == nil || !strings.Contains(err.Error(), "closed") {
			t.Errorf("SyncViews() after Close() error = %v, want error containing 'closed'", err)
		}
		if got := len(fakeRunner.executedNames()); got != 0 {
			t.Errorf("SyncViews() after Close() executed %d views, want 0", got)
		}
	})

	t.Run("SyncViews while async startup is in-flight preempts background run and executes views once", func(t *testing.T) {
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}

		bgInFlight := make(chan struct{})
		bgCanceled := make(chan struct{})
		var callIndex atomic.Int32
		var syncCompleted atomic.Int32

		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						if callIndex.Add(1) == 1 {
							close(bgInFlight)
							<-r.Context().Done()
							close(bgCanceled)
							return nil, r.Context().Err()
						}
						syncCompleted.Add(1)
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"job-sync"},"status":{"state":"DONE"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}

		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test-dataset"
		cfg.TableName = "test-table"
		cfg.CreateViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		t.Cleanup(func() { _ = p.Close() })

		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		select {
		case <-bgInFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for background startup view creation to enter RoundTrip")
		}

		if err := bqaap.SyncViews(ctx); err != nil {
			t.Fatalf("SyncViews() error = %v, want nil", err)
		}

		select {
		case <-bgCanceled:
		default:
			t.Error("expected SyncViews() to cancel in-flight background view creation")
		}
		if got := int(syncCompleted.Load()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() completed %d view DDLs after preemption, want %d", got, len(eventViewSpecs))
		}
		if logger.contains("failed to create view") || logger.contains("view creation finished") || logger.contains("call SyncViews to retry") {
			t.Errorf("preempted background view creation should not log context.Canceled warning or retry summary, got:\n%s", logger.String())
		}
	})

	t.Run("concurrent SyncViews honors caller context deadline while waiting for active refresh", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		firstInFlight := make(chan struct{})
		releaseFirst := make(chan struct{})
		var blockedOnce sync.Once
		bqaap.runner = &fakeViewRunner{
			onRun: func(ctx context.Context, _ string) error {
				blockedOnce.Do(func() {
					close(firstInFlight)
					<-releaseFirst
				})
				return ctx.Err()
			},
		}

		firstDone := make(chan error, 1)
		go func() {
			firstDone <- bqaap.SyncViews(ctx)
		}()

		select {
		case <-firstInFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for first SyncViews to enter RunViewDDL")
		}

		shortCtx, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
		defer cancel()
		start := time.Now()
		err := bqaap.SyncViews(shortCtx)
		elapsed := time.Since(start)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("second SyncViews(shortCtx) error = %v, want context.DeadlineExceeded", err)
		}
		if elapsed > 500*time.Millisecond {
			t.Errorf("second SyncViews(shortCtx) took %v, expected prompt return on deadline", elapsed)
		}

		close(releaseFirst)
		if firstErr := <-firstDone; firstErr != nil {
			t.Fatalf("first SyncViews() error = %v, want nil", firstErr)
		}

		afterRunner := &fakeViewRunner{}
		bqaap.runner = afterRunner
		if err := bqaap.SyncViews(ctx); err != nil {
			t.Fatalf("SyncViews() after releasing first refresh error = %v, want nil", err)
		}
		if got := len(afterRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("subsequent SyncViews() executed %d views, want %d", got, len(eventViewSpecs))
		}
	})

	t.Run("SyncViews honors caller context deadline while waiting for in-flight startup refresh", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.CreateViews = false })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		bgEntered := make(chan struct{})
		releaseBg := make(chan struct{})
		var once sync.Once
		bqaap.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, _ string) error {
				once.Do(func() {
					close(bgEntered)
					<-releaseBg
				})
				return runCtx.Err()
			},
		}

		defs, err := ViewDefinitions("test-project", "test-dataset", "test-table", "v")
		if err != nil {
			t.Fatalf("ViewDefinitions: %v", err)
		}
		bqaap.startAsyncViewCreation(ctx, defs)
		select {
		case <-bgEntered:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for startup view creation to enter RunViewDDL")
		}

		shortCtx, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
		defer cancel()
		start := time.Now()
		err = bqaap.SyncViews(shortCtx)
		elapsed := time.Since(start)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("SyncViews(shortCtx) while startup refresh blocked error = %v, want context.DeadlineExceeded", err)
		}
		if elapsed > 500*time.Millisecond {
			t.Errorf("SyncViews(shortCtx) took %v, expected prompt return on deadline", elapsed)
		}

		close(releaseBg)
		select {
		case <-bqaap.asyncViewsDone:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for startup view creation to complete")
		}
		afterRunner := &fakeViewRunner{}
		bqaap.runner = afterRunner
		if err := bqaap.SyncViews(ctx); err != nil {
			t.Fatalf("SyncViews() after startup refresh unblocked error = %v, want nil", err)
		}
		if got := len(afterRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() executed %d views, want %d", got, len(eventViewSpecs))
		}
	})
}
