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
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os/exec"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestConfig_CreateViews(t *testing.T) {
	var decoded Config
	if err := json.Unmarshal([]byte(`{"Enabled": true, "DisableCreateViews": true}`), &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	tests := []struct {
		name       string
		config     Config
		wantCreate bool
		wantPrefix string
	}{
		{
			name:       "DefaultConfig creates views with v prefix",
			config:     DefaultConfig(),
			wantCreate: true,
			wantPrefix: "v",
		},
		{
			name:       "Config literal creates views by default with v prefix",
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
			name: "DefaultConfig with DisableCreateViews disables view creation",
			config: func() Config {
				c := DefaultConfig()
				c.DisableCreateViews = true
				return c
			}(),
			wantCreate: false,
			wantPrefix: "v",
		},
		{
			name:       "Config decoded from JSON with DisableCreateViews disables view creation",
			config:     applyDefaults(decoded),
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

	defs, err := viewDefinitionsWithDenied(projectID, datasetID, tableName, "", nil)
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
	defs2, err := viewDefinitionsWithDenied(projectID, datasetID, tableName, "", nil)
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

// TestViewDefinitions_ExactProjectionParityWithADKPython compares the full
// generated CREATE OR REPLACE VIEW query of every view character-for-character
// against adk-python's _VIEW_COMMON_COLUMNS and _EVENT_VIEW_DEFS (for the 24
// identical views) and the documented safe expressions for v_event_compaction.
func TestViewDefinitions_ExactProjectionParityWithADKPython(t *testing.T) {
	const (
		projectID = "my-proj"
		datasetID = "agent_analytics"
		tableName = "agent_events"
	)
	defs, err := (Config{ProjectID: projectID, DatasetID: datasetID, TableName: tableName}).ViewDefinitions()
	if err != nil {
		t.Fatalf("ViewDefinitions() error = %v", err)
	}

	wantCommon := []string{
		"timestamp",
		"event_id",
		"event_type",
		"agent",
		"session_id",
		"invocation_id",
		"user_id",
		"trace_id",
		"span_id",
		"parent_span_id",
		"status",
		"error_message",
		"is_truncated",
	}

	wantExtras := map[string][]string{
		"USER_MESSAGE_RECEIVED": nil,
		"LLM_REQUEST": {
			"JSON_VALUE(attributes, '$.model') AS model",
			"content AS request_content",
			"JSON_QUERY(attributes, '$.llm_config') AS llm_config",
			"JSON_QUERY(attributes, '$.tools') AS tools",
		},
		"LLM_RESPONSE": {
			"JSON_QUERY(content, '$.response') AS response",
			"CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64) AS usage_prompt_tokens",
			"CAST(JSON_VALUE(content, '$.usage.completion') AS INT64) AS usage_completion_tokens",
			"CAST(JSON_VALUE(content, '$.usage.total') AS INT64) AS usage_total_tokens",
			"CAST(JSON_VALUE(attributes, '$.usage_metadata.cached_content_token_count') AS INT64) AS usage_cached_tokens",
			"CAST(JSON_VALUE(attributes, '$.usage_metadata.thoughts_token_count') AS INT64) AS usage_thinking_tokens",
			"CAST(JSON_VALUE(attributes, '$.usage_metadata.tool_use_prompt_token_count') AS INT64) AS usage_tool_use_tokens",
			"SAFE_DIVIDE(CAST(JSON_VALUE(attributes, '$.usage_metadata.cached_content_token_count') AS INT64),CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64)) AS context_cache_hit_rate",
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
			"CAST(JSON_VALUE(latency_ms, '$.time_to_first_token_ms') AS INT64) AS ttft_ms",
			"JSON_VALUE(attributes, '$.model_version') AS model_version",
			"JSON_QUERY(attributes, '$.usage_metadata') AS usage_metadata",
			"JSON_QUERY(attributes, '$.cache_metadata') AS cache_metadata",
			"JSON_VALUE(attributes, '$.cache_type') AS cache_type",
			"JSON_VALUE(attributes, '$.finish_reason') AS finish_reason",
		},
		"LLM_ERROR": {
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
		"TOOL_STARTING": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
		},
		"TOOL_COMPLETED": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.result') AS tool_result",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
			"JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
			"JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
		},
		"TOOL_ERROR": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
		"AGENT_STARTING": {
			`COALESCE(JSON_VALUE(content, '$.text_summary'), IF(REGEXP_CONTAINS(JSON_VALUE(content, '$'), r'^(?:<(?:function|bound method) |<.+ at 0x[0-9a-fA-F]+>|functools\.partial\()'), NULL, JSON_VALUE(content, '$'))) AS agent_instruction`,
			"JSON_VALUE(attributes, '$.instruction_source') AS instruction_source",
			"JSON_VALUE(attributes, '$.static_instruction') AS static_instruction",
			"JSON_VALUE(attributes, '$.agent_description') AS agent_description",
			"JSON_VALUE(attributes, '$.model') AS model",
			"JSON_QUERY(attributes, '$.sub_agents') AS sub_agents",
		},
		"AGENT_COMPLETED": {
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
		"AGENT_ERROR": {
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
			"JSON_VALUE(content, '$.error_traceback') AS error_traceback",
		},
		"INVOCATION_STARTING":  nil,
		"INVOCATION_COMPLETED": nil,
		"INVOCATION_ERROR": {
			"JSON_VALUE(content, '$.error_traceback') AS error_traceback",
		},
		"STATE_DELTA": {
			"JSON_QUERY(attributes, '$.state_delta') AS state_delta",
		},
		"HITL_CREDENTIAL_REQUEST": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
		"HITL_CONFIRMATION_REQUEST": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
		"HITL_INPUT_REQUEST": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
		"A2A_INTERACTION": {
			"content AS response_content",
			`JSON_VALUE(attributes, '$.a2a_metadata."a2a:task_id"') AS a2a_task_id`,
			`JSON_VALUE(attributes, '$.a2a_metadata."a2a:context_id"') AS a2a_context_id`,
			`JSON_QUERY(attributes, '$.a2a_metadata."a2a:request"') AS a2a_request`,
			`JSON_QUERY(attributes, '$.a2a_metadata."a2a:response"') AS a2a_response`,
		},
		"AGENT_RESPONSE": {
			"JSON_VALUE(content, '$.response') AS response_text",
			"JSON_VALUE(attributes, '$.source_event_id') AS source_event_id",
			"JSON_VALUE(attributes, '$.source_event_author') AS source_event_author",
			"JSON_VALUE(attributes, '$.source_event_branch') AS source_event_branch",
		},
		"AGENT_TRANSFER": {
			"JSON_VALUE(content, '$.from_agent') AS from_agent",
			"JSON_VALUE(content, '$.to_agent') AS to_agent",
			"JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
		},
		"EVENT_COMPACTION": {
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS start_seconds`,
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS end_seconds`,
			`SAFE.TIMESTAMP_MICROS(SAFE_CAST(SAFE_MULTIPLY(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6), 1000000) AS INT64)) AS window_start`,
			`SAFE.TIMESTAMP_MICROS(SAFE_CAST(SAFE_MULTIPLY(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6), 1000000) AS INT64)) AS window_end`,
			"JSON_QUERY(content, '$.compacted_content') AS compacted_content",
		},
		"AGENT_STATE_CHECKPOINT": {
			"JSON_QUERY(content, '$.agent_state') AS agent_state",
			"JSON_TYPE(JSON_QUERY(content, '$.agent_state')) AS agent_state_type",
			"SAFE_CAST(JSON_VALUE(content, '$.end_of_agent') AS BOOL) AS end_of_agent",
			"JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
		},
		"TOOL_PAUSED": {
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
			"JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
		},
		"NODE_OUTPUT": {
			"JSON_VALUE(attributes, '$.adk.node.path') AS node_path",
			"JSON_VALUE(attributes, '$.adk.node.run_id') AS node_run_id",
			"JSON_VALUE(attributes, '$.adk.node.parent_run_id') AS node_parent_run_id",
			"content AS output",
		},
		"NODE_ERROR": {
			"JSON_VALUE(attributes, '$.adk.node.path') AS node_path",
			"JSON_VALUE(attributes, '$.adk.node.run_id') AS node_run_id",
			"JSON_VALUE(attributes, '$.adk.node.parent_run_id') AS node_parent_run_id",
			"JSON_VALUE(content, '$.error_code') AS error_code",
		},
	}

	if len(defs) != len(wantExtras) {
		t.Fatalf("len(ViewDefinitions) = %d, want %d", len(defs), len(wantExtras))
	}
	for _, def := range defs {
		extras, ok := wantExtras[def.EventType]
		if !ok {
			t.Fatalf("unexpected view EventType %q", def.EventType)
		}
		cols := append(append([]string(nil), wantCommon...), extras...)
		wantViewName := "v_" + strings.ToLower(def.EventType)
		wantQuery := fmt.Sprintf(
			"CREATE OR REPLACE VIEW `%s.%s.%s` AS\nSELECT\n  %s\nFROM\n  `%s.%s.%s`\nWHERE\n  event_type = '%s'\n",
			projectID, datasetID, wantViewName,
			strings.Join(cols, ",\n  "),
			projectID, datasetID, tableName,
			def.EventType,
		)
		if diff := cmp.Diff(wantQuery, def.Query); diff != "" {
			t.Errorf("%s (%s) query mismatch (-want +got):\n%s", def.Name, def.EventType, diff)
		}
	}
}

func TestViewDefinitions_CustomViewPrefixAndMultiTableDisjoint(t *testing.T) {
	defsA, err := viewDefinitionsWithDenied("proj", "shared_ds", "events_a", "v_agent_a", nil)
	if err != nil {
		t.Fatalf("ViewDefinitions(events_a) error = %v", err)
	}
	defsB, err := viewDefinitionsWithDenied("proj", "shared_ds", "events_b", "v_agent_b", nil)
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
	if _, err := viewDefinitionsWithDenied("proj", "ds", "agent events", "v", nil); err != nil {
		t.Errorf("ViewDefinitions() with space in tableName error = %v, want nil", err)
	}
	// The longest names BigQuery accepts are valid.
	if _, err := viewDefinitionsWithDenied("proj", strings.Repeat("d", maxBigQueryIdentifierLen), strings.Repeat("t", maxBigQueryIdentifierLen), "v", nil); err != nil {
		t.Errorf("viewDefinitionsWithDenied() with %d-byte dataset ID and table name error = %v, want nil", maxBigQueryIdentifierLen, err)
	}
	// The public entry point applies the same checks after the defaults.
	if _, err := (Config{ProjectID: "proj", DatasetID: "ds", TableName: "v_llm_request"}).ViewDefinitions(); err == nil || !strings.Contains(err.Error(), "collides with the events table") {
		t.Errorf("Config.ViewDefinitions() with TableName v_llm_request error = %v, want collision error", err)
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
			name:      "hyphen in datasetID",
			projectID: "proj",
			datasetID: "test-dataset",
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
			name:      "backtick in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "table`01",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "backslash in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: `table\01`,
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "newline in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "table\n01",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "line separator U+2028 in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "table\u202801",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "whitespace-only Unicode Zs in tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "\u00a0\u2003",
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
		{
			name:      "whitespace-only tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "   ",
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "tableName longer than 1024 bytes",
			projectID: "proj",
			datasetID: "ds",
			tableName: strings.Repeat("t", maxBigQueryIdentifierLen+1),
			wantErr:   "invalid BigQuery table name",
		},
		{
			name:      "datasetID longer than 1024 bytes",
			projectID: "proj",
			datasetID: strings.Repeat("d", maxBigQueryIdentifierLen+1),
			tableName: "tbl",
			wantErr:   "invalid BigQuery dataset ID",
		},
		{
			name:       "view names longer than 1024 bytes",
			projectID:  "proj",
			datasetID:  "ds",
			tableName:  "tbl",
			viewPrefix: strings.Repeat("v", maxBigQueryIdentifierLen),
			wantErr:    "longer than 1024 bytes",
		},
		{
			name:      "view name equal to tableName",
			projectID: "proj",
			datasetID: "ds",
			tableName: "v_llm_request",
			wantErr:   `BigQuery view name "v_llm_request" collides with the events table "v_llm_request"`,
		},
		{
			name:      "view name equal to tableName but for case",
			projectID: "proj",
			datasetID: "ds",
			tableName: "V_Tool_Starting",
			wantErr:   `BigQuery view name "v_tool_starting" collides with the events table "V_Tool_Starting"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := viewDefinitionsWithDenied(tt.projectID, tt.datasetID, tt.tableName, tt.viewPrefix, nil)
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

		defs, err := cfg.ViewDefinitions()
		if err != nil {
			t.Fatalf("cfg.ViewDefinitions() error = %v", err)
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

// BigQuery TIMESTAMP values range from 0001-01-01 00:00:00 to
// 9999-12-31 23:59:59.999999 UTC.
const (
	minBigQueryTimestampMicros = -62135596800000000
	maxBigQueryTimestampMicros = 253402300799999999
)

// evalCompactionTimestampSQL simulates BigQuery's evaluation of a compaction
// seconds or window_start/window_end projection expression against a decoded
// content JSON map, enforcing BigQuery's conversion rules where CAST(... AS FLOAT64)
// rejects RFC3339 strings and SAFE_CAST(... AS TIMESTAMP) rejects >6 fractional
// digits unless normalized by REGEXP_REPLACE(..., r'(\.\d{6})\d+', r'\1').
// For window expressions, finite seconds whose microseconds exceed the FLOAT64
// range fail the query under a bare multiplication, which SAFE_ functions do
// not protect since they apply only to their own evaluation, and yield NULL
// under SAFE_MULTIPLY. A value out of the INT64 or TIMESTAMP range fails the
// query under CAST and TIMESTAMP_MICROS, and yields NULL under SAFE_CAST and
// SAFE.TIMESTAMP_MICROS. NULL is returned as the zero time.
func evalCompactionTimestampSQL(expr string, content map[string]any) (float64, time.Time, bool, error) {
	sqlPart := expr
	if idx := strings.LastIndex(expr, " AS "); idx >= 0 {
		sqlPart = strings.TrimSpace(expr[:idx])
	}
	const castSuffix = " AS INT64))"
	isWindow, safeWindow, safeMultiply := false, false, false
	inner := sqlPart
	for _, w := range []struct {
		prefix string
		safe   bool
	}{
		{prefix: "SAFE.TIMESTAMP_MICROS(SAFE_CAST(", safe: true},
		{prefix: "TIMESTAMP_MICROS(CAST(", safe: false},
	} {
		micros, ok := strings.CutPrefix(sqlPart, w.prefix)
		if !ok || !strings.HasSuffix(micros, castSuffix) {
			continue
		}
		isWindow, safeWindow = true, w.safe
		micros = strings.TrimSuffix(micros, castSuffix)
		switch {
		case strings.HasPrefix(micros, "SAFE_MULTIPLY(") && strings.HasSuffix(micros, ", 1000000)"):
			inner, safeMultiply = strings.TrimSuffix(strings.TrimPrefix(micros, "SAFE_MULTIPLY("), ", 1000000)"), true
		case strings.HasSuffix(micros, " * 1000000"):
			inner = strings.TrimSuffix(micros, " * 1000000")
		default:
			return 0, time.Time{}, true, fmt.Errorf("expression %q: unrecognized conversion of seconds to microseconds", expr)
		}
		break
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
		micros := secs * 1e6
		if math.IsInf(micros, 0) && !math.IsInf(secs, 0) {
			if safeMultiply {
				return secs, time.Time{}, true, nil
			}
			return 0, time.Time{}, true, fmt.Errorf("expression %q: %v * 1000000 overflows FLOAT64", expr, secs)
		}
		if math.IsNaN(micros) || micros < float64(math.MinInt64) || micros >= float64(math.MaxInt64) {
			if safeWindow {
				return secs, time.Time{}, true, nil
			}
			return 0, time.Time{}, true, fmt.Errorf("expression %q: CAST(%v AS INT64) overflows INT64", expr, micros)
		}
		m := int64(micros)
		if m < minBigQueryTimestampMicros || m > maxBigQueryTimestampMicros {
			if safeWindow {
				return secs, time.Time{}, true, nil
			}
			return 0, time.Time{}, true, fmt.Errorf("expression %q: TIMESTAMP_MICROS(%d) is out of the TIMESTAMP range", expr, m)
		}
		return secs, time.UnixMicro(m).UTC(), true, nil
	}
	return secs, time.Time{}, false, nil
}

func TestViews_JSONPathsMatchEmittedPayloads(t *testing.T) {
	defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
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
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.CustomMetadataAllowlist = []string{"adk_compaction", "adk_agent_state", "end_of_agent"}
		})
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
		// nanoseconds are non-zero) is converted to float64 epoch seconds so
		// adk-python's v_event_compaction view can also read the row.
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
		unparseableCompactionMap := map[string]any{
			"start_timestamp":   "not-a-timestamp",
			"end_timestamp":     "0001-01-01T00:00:00Z",
			"compacted_content": "compacted summary",
		}
		unparseableStringCompactionEvent := &session.Event{
			ID:     "ev-unparseable-string-compaction",
			Author: "root-agent",
			LLMResponse: model.LLMResponse{
				CustomMetadata: map[string]any{
					"adk_compaction": unparseableCompactionMap,
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
		errs.add(p.OnEventCallback()(mCtx, unparseableStringCompactionEvent))
		errs.add(p.OnEventCallback()(mCtx, checkpointAndTransferEvent))
		errs.add(p.OnEventCallback()(mCtx, hitlAndPauseEvent))
		errs.add(p.OnEventCallback()(mCtx, a2aNodeOutAndFinalEvent))
		errs.add(p.OnEventCallback()(mCtx, nodeErrorEvent))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, func(rows []map[string]any) bool {
			return untilEventType(EventInvocationCompleted)(rows) || untilEventType(EventInvocationError)(rows)
		})

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
		// 2) 9-digit nanosecond RFC3339 timestamp strings emitted via OnEventCallback
		//    and converted by Go to float64 epoch seconds, and
		// 3) legacy microsecond and 9-digit nanosecond RFC3339 timestamp strings.
		wantStartSec := float64(compactionStart.UnixMicro()) / 1e6
		wantEndSec := float64(compactionEnd.UnixMicro()) / 1e6

		compactionRows := rowsOfType(rows, EventEventCompaction)
		if len(compactionRows) != 3 {
			t.Fatalf("got %d EVENT_COMPACTION rows, want 3", len(compactionRows))
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

		parsedNanoStart, err := time.Parse(time.RFC3339Nano, "2026-10-10T01:00:00.500000123Z")
		if err != nil {
			t.Fatalf("time.Parse(nano start): %v", err)
		}
		parsedNanoEnd, err := time.Parse(time.RFC3339Nano, "2026-10-10T01:05:00.250000987Z")
		if err != nil {
			t.Fatalf("time.Parse(nano end): %v", err)
		}
		wantNanoStartSec := epochSeconds(parsedNanoStart).(float64)
		wantNanoEndSec := epochSeconds(parsedNanoEnd).(float64)

		nanoCompactionContent := jsonObjectColumn(t, compactionRows[1], columnContent)
		nanoStartNum, ok := nanoCompactionContent["start_timestamp"].(json.Number)
		if !ok {
			t.Fatalf("nano EVENT_COMPACTION start_timestamp = %v (%T), want json.Number", nanoCompactionContent["start_timestamp"], nanoCompactionContent["start_timestamp"])
		}
		if got, err := nanoStartNum.Float64(); err != nil || got != wantNanoStartSec {
			t.Fatalf("nano EVENT_COMPACTION start_timestamp = %v (err=%v), want %v", got, err, wantNanoStartSec)
		}
		nanoEndNum, ok := nanoCompactionContent["end_timestamp"].(json.Number)
		if !ok {
			t.Fatalf("nano EVENT_COMPACTION end_timestamp = %v (%T), want json.Number", nanoCompactionContent["end_timestamp"], nanoCompactionContent["end_timestamp"])
		}
		if got, err := nanoEndNum.Float64(); err != nil || got != wantNanoEndSec {
			t.Fatalf("nano EVENT_COMPACTION end_timestamp = %v (err=%v), want %v", got, err, wantNanoEndSec)
		}

		unparseableCompactionContent := jsonObjectColumn(t, compactionRows[2], columnContent)
		if got := unparseableCompactionContent["start_timestamp"]; got != nil {
			t.Errorf("unparseable EVENT_COMPACTION start_timestamp = %v (%T), want nil", got, got)
		}
		if got := unparseableCompactionContent["end_timestamp"]; got != nil {
			t.Errorf("zero RFC3339 EVENT_COMPACTION end_timestamp = %v (%T), want nil", got, got)
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
		rawNanoRFC3339Content := map[string]any{
			"start_timestamp":   "2026-10-10T01:00:00.500000123Z",
			"end_timestamp":     "2026-10-10T01:05:00.250000987Z",
			"compacted_content": "compacted summary",
		}

		wantNanoWinStart := time.UnixMicro(int64(math.Round(wantNanoStartSec * 1e6))).UTC()
		wantNanoWinEnd := time.UnixMicro(int64(math.Round(wantNanoEndSec * 1e6))).UTC()

		for _, representation := range []struct {
			name         string
			content      map[string]any
			wantStartSec float64
			wantEndSec   float64
			wantWinStart time.Time
			wantWinEnd   time.Time
		}{
			{
				name:         "native Go callback float64 epoch seconds",
				content:      compactionContent,
				wantStartSec: wantStartSec,
				wantEndSec:   wantEndSec,
				wantWinStart: compactionStart,
				wantWinEnd:   compactionEnd,
			},
			{
				name:         "Arrow-decoded RFC3339Nano string converted to float64 epoch seconds",
				content:      nanoCompactionContent,
				wantStartSec: wantNanoStartSec,
				wantEndSec:   wantNanoEndSec,
				wantWinStart: wantNanoWinStart,
				wantWinEnd:   wantNanoWinEnd,
			},
			{
				name:         "legacy RFC3339 microsecond timestamp strings",
				content:      rfc3339Content,
				wantStartSec: wantStartSec,
				wantEndSec:   wantEndSec,
				wantWinStart: compactionStart,
				wantWinEnd:   compactionEnd,
			},
			{
				name:         "legacy 9-digit nanosecond RFC3339 timestamp strings",
				content:      rawNanoRFC3339Content,
				wantStartSec: wantStartSec,
				wantEndSec:   wantEndSec,
				wantWinStart: compactionStart,
				wantWinEnd:   compactionEnd,
			},
		} {
			gotStartSec, _, isWin, err := evalCompactionTimestampSQL(exprByAlias["start_seconds"], representation.content)
			if err != nil || isWin || gotStartSec != representation.wantStartSec {
				t.Errorf("%s: start_seconds eval = (%v, isWin=%v, err=%v), want (%v, false, nil)", representation.name, gotStartSec, isWin, err, representation.wantStartSec)
			}
			gotEndSec, _, isWin, err := evalCompactionTimestampSQL(exprByAlias["end_seconds"], representation.content)
			if err != nil || isWin || gotEndSec != representation.wantEndSec {
				t.Errorf("%s: end_seconds eval = (%v, isWin=%v, err=%v), want (%v, false, nil)", representation.name, gotEndSec, isWin, err, representation.wantEndSec)
			}
			_, gotWinStart, isWin, err := evalCompactionTimestampSQL(exprByAlias["window_start"], representation.content)
			if err != nil || !isWin || !gotWinStart.Equal(representation.wantWinStart) {
				t.Errorf("%s: window_start eval = (%v, isWin=%v, err=%v), want (%v, true, nil)", representation.name, gotWinStart, isWin, err, representation.wantWinStart)
			}
			_, gotWinEnd, isWin, err := evalCompactionTimestampSQL(exprByAlias["window_end"], representation.content)
			if err != nil || !isWin || !gotWinEnd.Equal(representation.wantWinEnd) {
				t.Errorf("%s: window_end eval = (%v, isWin=%v, err=%v), want (%v, true, nil)", representation.name, gotWinEnd, isWin, err, representation.wantWinEnd)
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
				content:        toolContent("lookup", "args", map[string]any{"q": "go"}, toolOriginLocal),
				functionCallID: "call-1",
			}, nil),
			func() map[string]any {
				row := makeRow(EventToolCompleted, &eventData{
					content:        toolContent("lookup", "result", map[string]any{"ok": true}, toolOriginLocal),
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
				content:        toolContent("lookup", "args", map[string]any{"q": "go"}, toolOriginLocal),
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

// collectingRunner runs the garbage collector before it hands each view to
// inner, so that a BigQuery client that only weak pointers refer to is gone
// by the time inner runs the view.
type collectingRunner struct {
	inner viewRunner
}

func (r collectingRunner) RunViewDDL(ctx context.Context, viewName, query string) error {
	runtime.GC()
	return r.inner.RunViewDDL(ctx, viewName, query)
}

// newJobCountingBQClient returns a BigQuery client whose view DDL jobs
// succeed and are counted in jobs. It holds no connection, so it need not be
// closed, and nothing else refers to it.
func newJobCountingBQClient(t *testing.T, jobs *atomic.Int32) *bq.Client {
	t.Helper()
	transport := &mockTransport{roundTrip: func(r *http.Request) (*http.Response, error) {
		if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
			jobs.Add(1)
		}
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
	}}
	client, err := bq.NewClient(context.Background(), "test-project", option.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatalf("bq.NewClient: %v", err)
	}
	return client
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

	t.Run("view creation runs asynchronously by default without blocking constructor and creates all 25 views", func(t *testing.T) {
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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"

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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
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

	t.Run("DisableCreateViews executes zero view queries", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			config Config
		}{
			{
				name: "DefaultConfig with DisableCreateViews",
				config: func() Config {
					c := DefaultConfig()
					c.DisableCreateViews = true
					return c
				}(),
			},
			{name: "Config literal with DisableCreateViews", config: Config{Enabled: true, DisableCreateViews: true}},
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

				cfg := tc.config
				cfg.ProjectID = "test-project"
				cfg.DatasetID = "test_dataset"
				cfg.TableName = "test-table"

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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		// Wait for the pass to log its summary, the last thing it does, then
		// close. Closing once the last job is sent instead could cancel the
		// pass before the summary, which a canceled pass does not log.
		deadline := time.Now().Add(5 * time.Second)
		for !logger.contains("view creation finished") && time.Now().Before(deadline) {
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
		if !logger.contains("view creation finished: 24 created, 1 failed of 25; call SyncViews to retry") {
			t.Errorf("expected summary log line for partial failure, got logs:\n%s", logger.String())
		}
	})

	t.Run("DefaultConfig does not write background view failures or startup skip to log.Default", func(t *testing.T) {
		ctx := context.Background()
		var stdLogBuf bytes.Buffer
		origOut := log.Default().Writer()
		log.Default().SetOutput(&stdLogBuf)
		t.Cleanup(func() { log.Default().SetOutput(origOut) })

		var jobIdx atomic.Int32
		allFailed := make(chan struct{})
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						if int(jobIdx.Add(1)) == len(eventViewSpecs) {
							close(allFailed)
						}
						return &http.Response{
							StatusCode: http.StatusForbidden,
							Body:       io.NopCloser(strings.NewReader(`{"error":{"code":403,"message":"Access Denied"}}`)),
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

		// DefaultConfig() leaves Logger == log.Default(); neither per-view
		// failures, the retry summary, nor startup view-skip diagnostics may
		// write to log.Default().
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-allFailed:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for all %d view DDL jobs; got %d", len(eventViewSpecs), jobIdx.Load())
		}
		// Give syncAnalyticsViews a moment after the 25th RunViewDDL returns to
		// reach the post-loop summary check before Close() cancels lifeCtx.
		time.Sleep(20 * time.Millisecond)
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		skipCfg := DefaultConfig()
		skipCfg.ProjectID = "test-project"
		skipCfg.DatasetID = "test_dataset"
		skipCfg.TableName = "v_llm_request"
		pSkip, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, skipCfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients(skipCfg) error = %v", err)
		}
		if err := pSkip.Close(); err != nil {
			t.Fatalf("pSkip.Close() error = %v", err)
		}

		if got := stdLogBuf.String(); got != "" {
			t.Errorf("DefaultConfig() wrote diagnostics to log.Default():\n%s", got)
		}
	})

	t.Run("startup view creation keeps its BigQuery client alive until it ends", func(t *testing.T) {
		ctx := context.Background()
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
		}
		// Holding the view gate keeps the pass from starting before done is
		// read.
		if err := bqaap.acquireViewGate(ctx); err != nil {
			t.Fatalf("acquireViewGate() error = %v", err)
		}
		var jobs atomic.Int32
		logger := &recordingLogger{}
		func() {
			// Only the pass refers to this client, and its runner collects
			// garbage before every view.
			client := newJobCountingBQClient(t, &jobs)
			bqaap.runner = collectingRunner{inner: bqViewRunner{client: weak.Make(client), projectID: "test-project", datasetID: "test_dataset"}}
			bqaap.startAsyncViewCreation(ctx, defs, logger.Printf, client)
		}()
		bqaap.viewsMu.Lock()
		done := bqaap.asyncViewsDone
		bqaap.viewsMu.Unlock()
		bqaap.releaseViewGate()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("timed out waiting for view creation to end")
		}
		if got := int(jobs.Load()); got != len(eventViewSpecs) {
			t.Errorf("view creation ran %d view jobs, want %d; log:\n%s", got, len(eventViewSpecs), logger)
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
	noBQP, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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
			c.DisableCreateViews = true
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
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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

		runner := bqViewRunner{client: weak.Make(bqClient)}
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
					if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/v_test") {
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

		runner := bqViewRunner{client: weak.Make(bqClient), datasetID: "test_dataset"}
		if err := runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test_dataset.v_test` AS\nSELECT 1"); err != nil {
			t.Fatalf("RunViewDDL(unchanged) error = %v", err)
		}
		if got := jobsPosted.Load(); got != 0 {
			t.Errorf("jobsPosted for unchanged view = %d, want 0", got)
		}

		if err := runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test_dataset.v_test` AS\nSELECT 2"); err != nil {
			t.Fatalf("RunViewDDL(changed) error = %v", err)
		}
		if got := jobsPosted.Load(); got != 1 {
			t.Errorf("jobsPosted for changed view = %d, want 1", got)
		}
	})

	t.Run("bqViewRunner without a BigQuery client reports it", func(t *testing.T) {
		err := bqViewRunner{datasetID: "test_dataset"}.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test_dataset.v_test` AS\nSELECT 1")
		if err == nil || !strings.Contains(err.Error(), "BigQuery client is not configured") {
			t.Errorf("RunViewDDL() error = %v, want the missing client reported", err)
		}
	})

	t.Run("SyncViews keeps the plugin's BigQuery client alive while it runs", func(t *testing.T) {
		// Only the plugin refers to its client, as with
		// NewBigQueryAgentAnalyticsPluginWithConfig, and only the wrapper that
		// SyncViews is called on refers to the plugin. The runner collects
		// garbage before every view.
		var jobs atomic.Int32
		var key weak.Pointer[baseplugin.Plugin]
		var bqaap *BigQueryAgentAnalyticsPlugin
		func() {
			_, writeClient := newTestWriteClient(t)
			cfg := DefaultConfig()
			cfg.Enabled = true
			cfg.ProjectID = "test-project"
			cfg.DatasetID = "test_dataset"
			cfg.TableName = "test-table"
			cfg.DisableCreateViews = true
			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, newJobCountingBQClient(t, &jobs), writeClient)
			if err != nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
			}
			key = weak.Make(p)
			bqaap, _ = AsBigQueryAgentAnalyticsPlugin(p)
			bqaap.runner = collectingRunner{inner: bqaap.runner}
		}()
		err := bqaap.SyncViews(ctx)
		if p := key.Value(); p != nil {
			_ = p.Close()
		}
		if err != nil {
			t.Fatalf("SyncViews() error = %v", err)
		}
		if got := int(jobs.Load()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() ran %d view jobs, want %d", got, len(eventViewSpecs))
		}
	})

	t.Run("Close preempts in-flight SyncViews call", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}

		syncInFlight := make(chan struct{})
		runnerCanceled := make(chan struct{})
		releaseRunner := make(chan struct{})
		var once sync.Once
		bqaap.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, _ string) error {
				once.Do(func() {
					close(syncInFlight)
					<-runCtx.Done()
					close(runnerCanceled)
					<-releaseRunner
				})
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

		// Start a second SyncViews call that must wait on viewGate behind the
		// in-flight SyncViews call.
		contenderDone := make(chan error, 1)
		contenderStarted := make(chan struct{})
		go func() {
			close(contenderStarted)
			contenderDone <- bqaap.SyncViews(context.Background())
		}()
		<-contenderStarted
		time.Sleep(15 * time.Millisecond)

		closeDone := make(chan error, 1)
		go func() {
			closeDone <- p.Close()
		}()

		select {
		case <-runnerCanceled:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for Close() to cancel in-flight SyncViews context")
		}

		// The gate-waiting contender returns errPluginClosed promptly without
		// waiting for the gate holder's RunViewDDL step to finish.
		select {
		case err := <-contenderDone:
			if !errors.Is(err, errPluginClosed) {
				t.Errorf("gate-waiting SyncViews() error = %v, want %v", err, errPluginClosed)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("gate-waiting SyncViews() did not return after Close() canceled lifeCtx")
		}

		// Close() must NOT return while the gate-holding SyncViews step is still
		// running (verifying the viewGate drain in stopAsyncViewCreation).
		select {
		case err := <-closeDone:
			t.Fatalf("Close() returned (%v) before in-flight gate-holding SyncViews finished", err)
		case <-time.After(20 * time.Millisecond):
		}

		close(releaseRunner)

		select {
		case err := <-closeDone:
			if err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("Close() blocked after in-flight SyncViews released gate")
		}

		select {
		case err := <-syncDone:
			if !errors.Is(err, errPluginClosed) {
				t.Errorf("SyncViews() preempted by Close() error = %v, want %v", err, errPluginClosed)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("SyncViews() did not return after Close()")
		}
	})

	t.Run("Close preempts SyncViews waiting on startup pass and post-gate close check returns errPluginClosed", func(t *testing.T) {
		// Subcase A: SyncViews blocked inside preemptAsyncViewCreation when Close() runs.
		pA, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
		bqaapA, ok := AsBigQueryAgentAnalyticsPlugin(pA)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(pA) = false")
		}
		bgEntered := make(chan struct{})
		releaseBg := make(chan struct{})
		var once sync.Once
		bqaapA.runner = &fakeViewRunner{
			onRun: func(runCtx context.Context, _ string) error {
				once.Do(func() {
					close(bgEntered)
					<-runCtx.Done()
					<-releaseBg
				})
				return runCtx.Err()
			},
		}
		defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied: %v", err)
		}
		bqaapA.startAsyncViewCreation(ctx, defs, nil, nil)
		<-bgEntered

		syncErrCh := make(chan error, 1)
		go func() {
			syncErrCh <- bqaapA.SyncViews(context.Background())
		}()
		time.Sleep(15 * time.Millisecond)

		closeCh := make(chan error, 1)
		go func() { closeCh <- pA.Close() }()

		select {
		case syncErr := <-syncErrCh:
			if !errors.Is(syncErr, errPluginClosed) {
				t.Errorf("SyncViews() waiting on startup pass error = %v, want %v", syncErr, errPluginClosed)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("SyncViews() waiting on startup pass did not return when Close() canceled lifeCtx")
		}
		close(releaseBg)
		if closeErr := <-closeCh; closeErr != nil {
			t.Fatalf("pA.Close() error = %v", closeErr)
		}

		// Subcase B: SyncViews acquires viewGate after s.closed becomes true
		// before context.AfterFunc cancels ctx; post-gate check returns errPluginClosed.
		pB, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
		bqaapB, ok := AsBigQueryAgentAnalyticsPlugin(pB)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(pB) = false")
		}
		postGateRunner := &fakeViewRunner{}
		bqaapB.runner = postGateRunner
		if err := bqaapB.acquireViewGate(ctx); err != nil {
			t.Fatalf("acquireViewGate: %v", err)
		}
		postGateErrCh := make(chan error, 1)
		go func() {
			postGateErrCh <- bqaapB.SyncViews(context.Background())
		}()
		time.Sleep(15 * time.Millisecond)
		bqaapB.viewsMu.Lock()
		bqaapB.closed = true
		bqaapB.viewsMu.Unlock()
		bqaapB.releaseViewGate()

		select {
		case postErr := <-postGateErrCh:
			if !errors.Is(postErr, errPluginClosed) {
				t.Errorf("post-gate SyncViews() error = %v, want %v", postErr, errPluginClosed)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("post-gate SyncViews() did not return")
		}
		if got := len(postGateRunner.executedNames()); got != 0 {
			t.Errorf("post-gate closed SyncViews() executed %d views, want 0", got)
		}
	})

	t.Run("SyncViews after Close returns closed error without touching runner", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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
		if !errors.Is(err, errPluginClosed) {
			t.Errorf("SyncViews() after Close() error = %v, want %v", err, errPluginClosed)
		}
		if got := len(fakeRunner.executedNames()); got != 0 {
			t.Errorf("SyncViews() after Close() executed %d views, want 0", got)
		}
	})

	t.Run("SyncViews returns why the config has no views, and startup skips them", func(t *testing.T) {
		logger := &recordingLogger{}
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.TableName = "v_llm_request"
			c.Logger = logger
		})
		if !logger.contains("agentanalytics: skipping view creation") {
			t.Errorf("startup log does not report skipped view creation; got:\n%s", logger)
		}
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{}
		bqaap.runner = fakeRunner
		err := bqaap.SyncViews(ctx)
		if err == nil || !strings.Contains(err.Error(), "collides with the events table") {
			t.Errorf("SyncViews() error = %v, want the view name collision", err)
		}
		if got := len(fakeRunner.executedNames()); got != 0 {
			t.Errorf("SyncViews() executed %d views, want 0", got)
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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
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
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
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

		defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
		if err != nil {
			t.Fatalf("ViewDefinitions: %v", err)
		}
		bqaap.startAsyncViewCreation(ctx, defs, nil, nil)
		bqaap.viewsMu.Lock()
		bgDone := bqaap.asyncViewsDone
		bqaap.viewsMu.Unlock()
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
		case <-bgDone:
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

// TestPluginState_ReleasesViewPassAndRunner checks that the registry value of
// a plugin forgets the startup view pass once the pass ends, because its
// cancel function keeps the constructor context, and forgets the view runner
// on Close, because its BigQuery client can refer to the caller's objects.
func TestPluginState_ReleasesViewPassAndRunner(t *testing.T) {
	p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
	bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
	if !ok {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
	}
	release := make(chan struct{})
	bqaap.runner = &fakeViewRunner{onRun: func(context.Context, string) error {
		<-release
		return nil
	}}
	defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
	if err != nil {
		t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
	}

	bqaap.startAsyncViewCreation(context.Background(), defs, nil, nil)
	bqaap.viewsMu.Lock()
	done := bqaap.asyncViewsDone
	bqaap.viewsMu.Unlock()
	close(release)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for the view pass to end")
	}
	bqaap.viewsMu.Lock()
	cancelKept, doneKept := bqaap.viewsCancel != nil, bqaap.asyncViewsDone != nil
	bqaap.viewsMu.Unlock()
	if cancelKept || doneKept {
		t.Errorf("after the view pass ended: viewsCancel kept = %v, asyncViewsDone kept = %v; want neither kept", cancelKept, doneKept)
	}

	if err := p.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	bqaap.viewsMu.Lock()
	runnerKept := bqaap.runner != nil
	bqaap.viewsMu.Unlock()
	if runnerKept {
		t.Error("after Close: view runner kept, want it dropped")
	}
}

// TestViews_EventCompactionWindowOutOfRangeIsNull checks that a compaction
// timestamp that BigQuery cannot turn into a TIMESTAMP, such as epoch
// milliseconds, a number beyond INT64 microseconds or one whose microseconds
// exceed the FLOAT64 range, makes window_start and window_end NULL instead of
// failing every query of v_event_compaction, including when the plugin wrote
// the value.
func TestViews_EventCompactionWindowOutOfRangeIsNull(t *testing.T) {
	defs, err := (Config{ProjectID: "test-project", DatasetID: "test_dataset", TableName: "test-table"}).ViewDefinitions()
	if err != nil {
		t.Fatalf("Config.ViewDefinitions() error = %v", err)
	}
	var query string
	for _, d := range defs {
		if d.EventType == EventEventCompaction {
			query = d.Query
		}
	}
	exprs, aliases := extractExtraProjections(t, query)
	exprByAlias := make(map[string]string, len(aliases))
	for i, alias := range aliases {
		exprByAlias[alias] = exprs[i]
	}

	for _, tc := range []struct {
		name  string
		value json.Number
	}{
		{name: "epoch milliseconds beyond year 9999", value: "1729634400000"},
		{name: "seconds beyond INT64 microseconds", value: "1e300"},
		{name: "seconds before year 1", value: "-1e12"},
		{name: "seconds whose microseconds exceed FLOAT64", value: "1e308"},
		{name: "negative seconds whose microseconds exceed FLOAT64", value: "-1e308"},
	} {
		content := map[string]any{"start_timestamp": tc.value, "end_timestamp": tc.value}
		for _, alias := range []string{"window_start", "window_end"} {
			_, got, isWin, err := evalCompactionTimestampSQL(exprByAlias[alias], content)
			if err != nil || !isWin || !got.IsZero() {
				t.Errorf("%s: %s = (%v, isWindow=%v, err=%v), want NULL without failing the query", tc.name, alias, got, isWin, err)
			}
		}
	}

	// A bare multiplication would overflow FLOAT64 before the safe cast runs,
	// and fail the query.
	for _, value := range []json.Number{"1e308", "-1e308"} {
		content := map[string]any{"start_timestamp": value, "end_timestamp": value}
		for _, alias := range []string{"window_start", "window_end"} {
			expr := exprByAlias[alias]
			bare := strings.Replace(strings.Replace(expr, "SAFE_CAST(SAFE_MULTIPLY(", "SAFE_CAST(", 1), ", 1000000) AS INT64))", " * 1000000 AS INT64))", 1)
			if bare == expr {
				t.Fatalf("%s does not multiply by 1000000 with SAFE_MULTIPLY: %s", alias, expr)
			}
			if _, _, _, err := evalCompactionTimestampSQL(bare, content); err == nil {
				t.Errorf("%s with a bare multiplication of %s: error = nil, want a FLOAT64 overflow", alias, value)
			}
		}
	}

	// A value in range still yields its timestamp.
	content := map[string]any{"start_timestamp": json.Number("1760000000.5"), "end_timestamp": json.Number("1760000001")}
	if _, got, _, err := evalCompactionTimestampSQL(exprByAlias["window_start"], content); err != nil || !got.Equal(time.Unix(1760000000, 500000000)) {
		t.Errorf("window_start = (%v, err=%v), want %v", got, err, time.Unix(1760000000, 500000000).UTC())
	}
	if _, got, _, err := evalCompactionTimestampSQL(exprByAlias["window_end"], content); err != nil || !got.Equal(time.Unix(1760000001, 0)) {
		t.Errorf("window_end = (%v, err=%v), want %v", got, err, time.Unix(1760000001, 0).UTC())
	}

	// The plugin writes such numbers unchanged, so the view meets them.
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.CustomMetadataAllowlist = []string{"adk_compaction"}
		c.DisableCreateViews = true
	})
	var errs callbackErrors
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		ID:     "ev-extreme-compaction",
		Author: "test-agent",
		LLMResponse: model.LLMResponse{
			CustomMetadata: map[string]any{
				"adk_compaction": map[string]any{
					"start_timestamp":   1e308,
					"end_timestamp":     -1e308,
					"compacted_content": "compacted summary",
				},
			},
		},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)
	rows := rowsOfType(collectRows(t, requests, untilEventType(EventEventCompaction)), EventEventCompaction)
	written := jsonObjectColumn(t, rows[0], columnContent)
	for key, want := range map[string]float64{"start_timestamp": 1e308, "end_timestamp": -1e308} {
		n, ok := written[key].(json.Number)
		if !ok {
			t.Fatalf("written %s = %v (%T), want the number %v", key, written[key], written[key], want)
		}
		if got, err := n.Float64(); err != nil || got != want {
			t.Fatalf("written %s = %v (err=%v), want %v", key, got, err, want)
		}
	}
	for _, alias := range []string{"window_start", "window_end"} {
		_, got, isWin, err := evalCompactionTimestampSQL(exprByAlias[alias], written)
		if err != nil || !isWin || !got.IsZero() {
			t.Errorf("written row: %s = (%v, isWindow=%v, err=%v), want NULL without failing the query", alias, got, isWin, err)
		}
	}
}

// TestSyncAnalyticsViews_ContextEndsPass checks that once the context of a
// view creation pass is done, the view whose DDL was cut short is reported
// once as canceled, whatever error the runner returned, and the remaining
// views are skipped without being counted or logged as failures, while a
// per-view timeout under a live context is a failure that the pass goes past.
func TestSyncAnalyticsViews_ContextEndsPass(t *testing.T) {
	ctx := context.Background()

	t.Run("SyncViews caller deadline is reported once as canceled", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) { c.DisableCreateViews = true })
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		runner := &fakeViewRunner{onRun: func(runCtx context.Context, _ string) error {
			<-runCtx.Done()
			return runCtx.Err()
		}}
		bqaap.runner = runner

		deadlineCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
		defer cancel()
		err := bqaap.SyncViews(deadlineCtx)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("SyncViews() error = %v, want context.DeadlineExceeded", err)
		}
		if want := "view creation canceled during v_user_message_received: context deadline exceeded"; err.Error() != want {
			t.Errorf("SyncViews() error = %q, want only %q", err, want)
		}
		if got := len(runner.executedNames()); got != 1 {
			t.Errorf("SyncViews() ran %d views, want 1", got)
		}
	})

	t.Run("cancellation reported by a transport error that does not wrap it is not a failure", func(t *testing.T) {
		defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
		}
		passCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		runner := &fakeViewRunner{onRun: func(context.Context, string) error {
			cancel() // As Close or a preempting SyncViews does.
			return status.Error(codes.Canceled, "transport: stream canceled")
		}}
		logger := &recordingLogger{}
		err = syncAnalyticsViews(passCtx, runner, defs, logger.Printf)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("syncAnalyticsViews() error = %v, want context.Canceled", err)
		}
		if !strings.Contains(err.Error(), "view creation canceled during v_user_message_received") || strings.Contains(err.Error(), "failed to create view") {
			t.Errorf("syncAnalyticsViews() error = %q, want one cancellation and no failure", err)
		}
		if got := logger.String(); got != "" {
			t.Errorf("canceled pass logged:\n%s\nwant nothing", got)
		}
		if got := len(runner.executedNames()); got != 1 {
			t.Errorf("syncAnalyticsViews() ran %d views, want 1", got)
		}
	})

	t.Run("per-view timeout under a live context is a failure and the pass continues", func(t *testing.T) {
		defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
		if err != nil {
			t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
		}
		runner := &fakeViewRunner{errFor: map[string]error{"v_user_message_received": context.DeadlineExceeded}}
		logger := &recordingLogger{}
		err = syncAnalyticsViews(ctx, runner, defs, logger.Printf)
		if err == nil || !strings.Contains(err.Error(), "failed to create view v_user_message_received") {
			t.Errorf("syncAnalyticsViews() error = %v, want failure of v_user_message_received", err)
		}
		if got := len(runner.executedNames()); got != len(defs) {
			t.Errorf("syncAnalyticsViews() ran %d views, want %d", got, len(defs))
		}
		if want := "view creation finished: 24 created, 1 failed of 25; call SyncViews to retry"; !logger.contains(want) {
			t.Errorf("log does not contain %q; got:\n%s", want, logger)
		}
	})
}

// TestSyncAnalyticsViews_NilLogf checks that the startup pass of a plugin
// without a Logger, which gets a nil logf, still runs every view and reports
// its failures in the returned error.
func TestSyncAnalyticsViews_NilLogf(t *testing.T) {
	defs, err := viewDefinitionsWithDenied("test-project", "test_dataset", "test-table", "v", nil)
	if err != nil {
		t.Fatalf("viewDefinitionsWithDenied() error = %v", err)
	}
	runner := &fakeViewRunner{errFor: map[string]error{"v_llm_request": errors.New("access denied")}}
	err = syncAnalyticsViews(context.Background(), runner, defs, nil)
	if err == nil || !strings.Contains(err.Error(), "failed to create view v_llm_request") {
		t.Errorf("syncAnalyticsViews() error = %v, want failure of v_llm_request", err)
	}
	if got := len(runner.executedNames()); got != len(defs) {
		t.Errorf("syncAnalyticsViews() ran %d views, want %d", got, len(defs))
	}
}

// TestBQViewRunner_DoesNotReplaceNonView checks that a table of the view's
// name that is not a view is reported without running a DDL job.
func TestBQViewRunner_DoesNotReplaceNonView(t *testing.T) {
	ctx := context.Background()
	for _, tableType := range []string{"TABLE", "MATERIALIZED_VIEW"} {
		t.Run(tableType, func(t *testing.T) {
			var jobsPosted atomic.Int32
			httpClient := &http.Client{
				Transport: &mockTransport{
					roundTrip: func(r *http.Request) (*http.Response, error) {
						if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/v_test") {
							return &http.Response{
								StatusCode: http.StatusOK,
								Body:       io.NopCloser(strings.NewReader(`{"type":"` + tableType + `"}`)),
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

			runner := bqViewRunner{client: weak.Make(bqClient), datasetID: "test_dataset"}
			err = runner.RunViewDDL(ctx, "v_test", "CREATE OR REPLACE VIEW `test-project.test_dataset.v_test` AS\nSELECT 1")
			if want := "cannot replace existing BigQuery " + tableType + ` "v_test" with a view`; err == nil || err.Error() != want {
				t.Errorf("RunViewDDL() error = %v, want %q", err, want)
			}
			if got := jobsPosted.Load(); got != 0 {
				t.Errorf("DDL jobs posted = %d, want 0", got)
			}
		})
	}
}

// TestNewBigQueryAgentAnalyticsPlugin_ResolvesProjectForTableAndViews checks
// that an empty or detected ProjectID takes the project of bqClient, and that
// the table check, the write stream, the view runner and the views all use the
// configured project even when bqClient was made for another one.
func TestNewBigQueryAgentAnalyticsPlugin_ResolvesProjectForTableAndViews(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name          string
		configProject string
		wantProject   string
	}{
		{name: "empty ProjectID takes the client's project", configProject: "", wantProject: "client-project"},
		{name: "DetectProjectID takes the client's project", configProject: bq.DetectProjectID, wantProject: "client-project"},
		{name: "explicit ProjectID is used for the table check", configProject: "config-project", wantProject: "config-project"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var tableGets []string
			httpClient := &http.Client{
				Transport: &mockTransport{
					roundTrip: func(r *http.Request) (*http.Response, error) {
						if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/tables/test-table") {
							mu.Lock()
							tableGets = append(tableGets, r.URL.Path)
							mu.Unlock()
						}
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader("{}")),
						}, nil
					},
				},
			}
			bqClient, err := bq.NewClient(ctx, "client-project", option.WithHTTPClient(httpClient))
			if err != nil {
				t.Fatalf("bq.NewClient: %v", err)
			}
			t.Cleanup(func() { _ = bqClient.Close() })

			requests := make(chan *storagepb.AppendRowsRequest, 10)
			lis, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("listen: %v", err)
			}
			gSrv := grpc.NewServer()
			storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: requests})
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

			cfg := DefaultConfig()
			cfg.ProjectID = tc.configProject
			cfg.DatasetID = "test_dataset"
			cfg.TableName = "test-table"
			cfg.DisableCreateViews = true
			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
			if err != nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
			}
			t.Cleanup(func() { _ = p.Close() })
			bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
			if !ok {
				t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
			}
			if r, _ := bqaap.runner.(bqViewRunner); r.projectID != tc.wantProject {
				t.Errorf("view runner project = %q, want %q", r.projectID, tc.wantProject)
			}

			mu.Lock()
			gets := append([]string(nil), tableGets...)
			mu.Unlock()
			if wantPath := "/projects/" + tc.wantProject + "/datasets/test_dataset/tables/test-table"; len(gets) == 0 || !strings.HasSuffix(gets[0], wantPath) {
				t.Errorf("table metadata GETs = %v, want one ending in %q", gets, wantPath)
			}

			mCtx := &mockInvocationContext{
				ctx:          ctx,
				agentName:    "test-agent",
				sessionID:    "sess-1",
				invocationID: "inv-1",
				userID:       "user-1",
			}
			if _, err := p.BeforeRunCallback()(mCtx); err != nil {
				t.Fatalf("BeforeRunCallback() error = %v", err)
			}
			p.AfterRunCallback()(mCtx)
			wantStream := "projects/" + tc.wantProject + "/datasets/test_dataset/tables/test-table/streams/_default"
			select {
			case req := <-requests:
				if got := req.GetWriteStream(); got != wantStream {
					t.Errorf("AppendRowsRequest.WriteStream = %q, want %q", got, wantStream)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("timed out waiting for AppendRowsRequest")
			}

			runner := &fakeViewRunner{}
			bqaap.runner = runner
			if err := bqaap.SyncViews(ctx); err != nil {
				t.Fatalf("SyncViews() error = %v", err)
			}
			runner.mu.Lock()
			firstQuery := runner.queries[0]
			runner.mu.Unlock()
			if want := "`" + tc.wantProject + ".test_dataset.v_user_message_received`"; !strings.Contains(firstQuery, want) {
				t.Errorf("first view DDL = %q, want it to name %s", firstQuery, want)
			}
		})
	}
}

var finalSelectAliasRe = regexp.MustCompile(`(?m)\bAS\s+([a-z0-9_]+)\s*(?:,|$)`)

func extractFinalSelectAliases(t *testing.T, query string) []string {
	t.Helper()
	const selectMarker = "\nSELECT\n  "
	const fromMarker = "\nFROM\n"
	start := strings.LastIndex(query, selectMarker)
	end := strings.LastIndex(query, fromMarker)
	if start < 0 || end < 0 || end <= start+len(selectMarker) {
		t.Fatalf("failed to locate final SELECT ... FROM block in query:\n%s", query)
	}
	body := query[start+len(selectMarker) : end]
	matches := finalSelectAliasRe.FindAllStringSubmatch(body, -1)
	aliases := make([]string, 0, len(matches))
	for _, m := range matches {
		aliases = append(aliases, m[1])
	}
	return aliases
}

func extractRawEventsCTE(t *testing.T, query string) string {
	t.Helper()
	start := strings.Index(query, "raw_events AS (")
	end := strings.Index(query, "ranked_events AS (")
	if start < 0 || end < 0 || end <= start {
		t.Fatalf("failed to locate raw_events CTE in query:\n%s", query)
	}
	return query[start:end]
}

func TestEvalViewDefinitions_ColumnsAndSQLStructure(t *testing.T) {
	wantSessionColumns := []string{
		"session_id",
		"user_id",
		"root_agent",
		"session_start_timestamp",
		"session_end_timestamp",
		"turn_count",
		"tool_call_count",
		"tool_error_count",
		"llm_error_count",
		"agent_error_count",
		"has_error",
		"has_truncated_content",
		"has_tool_loop",
		"max_consecutive_identical_tool_calls",
		"distinct_tools_used",
		"total_prompt_tokens",
		"total_completion_tokens",
		"total_tokens",
		"total_latency_ms",
		"eval_ready",
		"trace",
	}
	wantTurnColumns := []string{
		"session_id",
		"invocation_id",
		"turn_index",
		"user_id",
		"root_agent",
		"turn_start_timestamp",
		"turn_end_timestamp",
		"tool_call_count",
		"tool_error_count",
		"llm_error_count",
		"agent_error_count",
		"has_error",
		"has_truncated_content",
		"has_tool_loop",
		"max_consecutive_identical_tool_calls",
		"distinct_tools_used",
		"total_prompt_tokens",
		"total_completion_tokens",
		"total_tokens",
		"turn_latency_ms",
		"eval_ready",
		"trace",
	}

	defs, err := (Config{ProjectID: "p", DatasetID: "d", TableName: "t"}).EvalViewDefinitions()
	if err != nil {
		t.Fatalf("EvalViewDefinitions() error = %v", err)
	}
	if len(defs) != 2 {
		t.Fatalf("len(EvalViewDefinitions) = %d, want 2", len(defs))
	}

	// Verify deterministic ordering across repeated calls.
	defs2, err := (Config{ProjectID: "p", DatasetID: "d", TableName: "t"}).EvalViewDefinitions()
	if err != nil {
		t.Fatalf("second EvalViewDefinitions() error = %v", err)
	}
	if diff := cmp.Diff(defs, defs2); diff != "" {
		t.Fatalf("EvalViewDefinitions() is not deterministic (-first +second):\n%s", diff)
	}

	sessionDef := defs[0]
	turnDef := defs[1]
	if sessionDef.Name != "v_eval_session" || sessionDef.EventType != "" {
		t.Errorf("defs[0] = {Name:%q, EventType:%q}, want {Name:\"v_eval_session\", EventType:\"\"};", sessionDef.Name, sessionDef.EventType)
	}
	if turnDef.Name != "v_eval_turn" || turnDef.EventType != "" {
		t.Errorf("defs[1] = {Name:%q, EventType:%q}, want {Name:\"v_eval_turn\", EventType:\"\"};", turnDef.Name, turnDef.EventType)
	}

	if got := extractFinalSelectAliases(t, sessionDef.Query); !cmp.Equal(got, wantSessionColumns) {
		t.Errorf("v_eval_session columns mismatch (-want +got):\n%s", cmp.Diff(wantSessionColumns, got))
	}
	if got := extractFinalSelectAliases(t, turnDef.Query); !cmp.Equal(got, wantTurnColumns) {
		t.Errorf("v_eval_turn columns mismatch (-want +got):\n%s", cmp.Diff(wantTurnColumns, got))
	}

	if !strings.HasPrefix(sessionDef.Query, "CREATE OR REPLACE VIEW `p.d.v_eval_session` AS\nWITH\n") {
		t.Errorf("v_eval_session unexpected prefix:\n%s", sessionDef.Query[:120])
	}
	if !strings.HasPrefix(turnDef.Query, "CREATE OR REPLACE VIEW `p.d.v_eval_turn` AS\nWITH\n") {
		t.Errorf("v_eval_turn unexpected prefix:\n%s", turnDef.Query[:120])
	}

	for _, def := range defs {
		if strings.Contains(def.Query, ";") {
			t.Errorf("%s query contains a semicolon", def.Name)
		}
		if got := strings.Count(def.Query, "`"); got != 4 {
			t.Errorf("%s backtick count = %d, want 4", def.Name, got)
		}
		if !strings.Contains(def.Query, "FROM\n      `p.d.t`") {
			t.Errorf("%s missing table reference `p.d.t`", def.Name)
		}
		for _, cte := range []string{
			"raw_events AS (",
			"ranked_events AS (",
			"scoped_events AS (",
			"deduped_events AS (",
			"typed_events AS (",
			"response_events AS (",
			"annotated_events AS (",
			"trace_events AS (",
			"turn_rollup AS (",
			"turns AS (",
			"agent_rows AS (",
			"agent_latest AS (",
			"agent_tool_declarations AS (",
			"agent_configs AS (",
			"agent_maps AS (",
		} {
			if !strings.Contains(def.Query, cte) {
				t.Errorf("%s missing CTE %q", def.Name, cte)
			}
		}
		if strings.Contains(def.Query, "USING (") {
			t.Errorf("%s must use NULL-safe IS NOT DISTINCT FROM join predicates instead of USING (...)", def.Name)
		}
	}

	assertQuerySHA256(t, "v_eval_session", sessionDef.Query, 24501, "a54b5f63ac7a1482109421d24a877a71ff59121423d3ca919b002b78cbab2bca")
	assertQuerySHA256(t, "v_eval_turn", turnDef.Query, 23065, "7a5663fbf316198b894497a0ceb5c817885aa8d06a2effe94290bb147b3c5cce")

	if !strings.Contains(sessionDef.Query, "session_rollup AS (") {
		t.Error("v_eval_session missing session_rollup CTE")
	}
	wantSessionJoin := "ON m.session_id = s.session_id\n  AND m.user_id IS NOT DISTINCT FROM s.user_id\n  AND m.app_name IS NOT DISTINCT FROM s.app_name"
	if !strings.Contains(sessionDef.Query, wantSessionJoin) {
		t.Error("v_eval_session missing composite NULL-safe agent_maps join")
	}
	wantTurnJoin := "ON m.session_id = t.session_id\n  AND m.user_id IS NOT DISTINCT FROM t.user_id\n  AND m.app_name IS NOT DISTINCT FROM t.app_name\n  AND m.invocation_id = t.invocation_id"
	if !strings.Contains(turnDef.Query, wantTurnJoin) {
		t.Error("v_eval_turn missing composite NULL-safe turn-scoped agent_maps join")
	}
	if !strings.Contains(turnDef.Query, "'turn_index', 0,\n        'turn_id', t.invocation_id") {
		t.Error("v_eval_turn trace.turns[0] must normalize turn_index to 0 for single-turn evaluation")
	}
}

func assertQuerySHA256(t *testing.T, viewName, query string, wantLen int, wantSHA256 string) {
	t.Helper()
	if len(query) != wantLen {
		t.Errorf("len(%s.Query) = %d, want %d", viewName, len(query), wantLen)
	}
	sum := sha256.Sum256([]byte(query))
	gotSHA256 := fmt.Sprintf("%x", sum)
	if gotSHA256 != wantSHA256 {
		t.Errorf("%s.Query sha256 = %s, want %s", viewName, gotSHA256, wantSHA256)
	}
}

func TestEvalViewDefinitions_CustomViewPrefixAndValidation(t *testing.T) {
	defs, err := (Config{
		ProjectID:  "my-proj",
		DatasetID:  "shared_ds",
		TableName:  "events_a",
		ViewPrefix: "v_agent_a",
	}).EvalViewDefinitions()
	if err != nil {
		t.Fatalf("EvalViewDefinitions() error = %v", err)
	}
	if len(defs) != 2 || defs[0].Name != "v_agent_a_eval_session" || defs[1].Name != "v_agent_a_eval_turn" {
		t.Fatalf("EvalViewDefinitions() names = [%q, %q], want [v_agent_a_eval_session, v_agent_a_eval_turn]", defs[0].Name, defs[1].Name)
	}
	if !strings.HasPrefix(defs[0].Query, "CREATE OR REPLACE VIEW `my-proj.shared_ds.v_agent_a_eval_session` AS\n") {
		t.Errorf("defs[0] query header mismatch:\n%s", defs[0].Query[:120])
	}
	if !strings.HasPrefix(defs[1].Query, "CREATE OR REPLACE VIEW `my-proj.shared_ds.v_agent_a_eval_turn` AS\n") {
		t.Errorf("defs[1] query header mismatch:\n%s", defs[1].Query[:120])
	}

	for _, badTable := range []string{"events", ".events", "proj.ds."} {
		if _, err := buildEvalSessionViewSQL(badTable, "v", nil); err == nil || !strings.Contains(err.Error(), "dataset-qualified") {
			t.Errorf("buildEvalSessionViewSQL(%q) error = %v, want dataset-qualified error", badTable, err)
		}
		if _, err := buildEvalTurnViewSQL(badTable, "v", nil); err == nil || !strings.Contains(err.Error(), "dataset-qualified") {
			t.Errorf("buildEvalTurnViewSQL(%q) error = %v, want dataset-qualified error", badTable, err)
		}
	}

	for _, tc := range []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{
			name:    "invalid project ID",
			cfg:     Config{ProjectID: "proj` --", DatasetID: "ds", TableName: "tbl"},
			wantErr: "invalid BigQuery project ID",
		},
		{
			name:    "invalid dataset ID",
			cfg:     Config{ProjectID: "proj", DatasetID: "bad-ds", TableName: "tbl"},
			wantErr: "invalid BigQuery dataset ID",
		},
		{
			name:    "invalid table name",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "tbl;DROP"},
			wantErr: "invalid BigQuery table name",
		},
		{
			name:    "backtick in table name",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "table`01"},
			wantErr: "invalid BigQuery table name",
		},
		{
			name:    "invalid view prefix",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "tbl", ViewPrefix: "bad prefix"},
			wantErr: "invalid BigQuery view prefix",
		},
		{
			name:    "overlong eval view name",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "tbl", ViewPrefix: strings.Repeat("v", maxBigQueryIdentifierLen)},
			wantErr: "longer than 1024 bytes",
		},
		{
			name:    "table name collides with v_eval_session",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "v_eval_session"},
			wantErr: `BigQuery view name "v_eval_session" collides with the events table "v_eval_session"`,
		},
		{
			name:    "table name collides case-insensitively with v_eval_turn",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "V_Eval_Turn"},
			wantErr: `BigQuery view name "v_eval_turn" collides with the events table "V_Eval_Turn"`,
		},
		{
			name:    "unknown column in PayloadColumnDenylist",
			cfg:     Config{ProjectID: "proj", DatasetID: "ds", TableName: "tbl", PayloadColumnDenylist: []string{"session_id"}},
			wantErr: "unsupported PayloadColumnDenylist entry",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.cfg.EvalViewDefinitions(); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("EvalViewDefinitions() error = %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestEvalViewDefinitions_PayloadColumnDenylistProjection(t *testing.T) {
	baseCfg := Config{ProjectID: "p", DatasetID: "d", TableName: "t"}

	hashCases := []struct {
		name              string
		viewPrefix        string
		denylist          []string
		wantSessionLen    int
		wantSessionSHA256 string
		wantTurnLen       int
		wantTurnSHA256    string
	}{
		{
			name:              "default",
			wantSessionLen:    24501,
			wantSessionSHA256: "a54b5f63ac7a1482109421d24a877a71ff59121423d3ca919b002b78cbab2bca",
			wantTurnLen:       23065,
			wantTurnSHA256:    "7a5663fbf316198b894497a0ceb5c817885aa8d06a2effe94290bb147b3c5cce",
		},
		{
			name:              "deny content_parts",
			denylist:          []string{columnContentParts},
			wantSessionLen:    24614,
			wantSessionSHA256: "d95d3f6af72e1951ed98a914d9478be1ee0fab6614f0a8d224f8c2be87846383",
			wantTurnLen:       23178,
			wantTurnSHA256:    "d94d4535ee26a035d0e9e10287033c67b5a58364d56170445d277506cc9488f4",
		},
		{
			name:              "deny attributes",
			denylist:          []string{columnAttributes},
			wantSessionLen:    24523,
			wantSessionSHA256: "c1432d62a90543248fc1d91e728cf2e41866f9932e873fe67dbd2348439a0b88",
			wantTurnLen:       23087,
			wantTurnSHA256:    "439c446ad5bd8033bfa8a5a1e5282ce8e7d4545ca7ce48760f903c4244b45bab",
		},
		{
			name:              "deny latency_ms",
			denylist:          []string{columnLatencyMS},
			wantSessionLen:    24523,
			wantSessionSHA256: "e5f8b90c1ae64871235d7afcfb0fdf3b0539779ff3b8779e33628051e15b3d8a",
			wantTurnLen:       23087,
			wantTurnSHA256:    "d15050bb6203aedc43e2e8d8be75dd34642e33d36647e352f6e7d7b9963ef15a",
		},
		{
			name:              "deny all three with custom ViewPrefix",
			viewPrefix:        "custom_p",
			denylist:          []string{columnContentParts, columnAttributes, columnLatencyMS},
			wantSessionLen:    24665,
			wantSessionSHA256: "32791bd7c6258325fe34f1fa74e51b83ead80757e5eb27370dcab7847fa9d91f",
			wantTurnLen:       23229,
			wantTurnSHA256:    "c1d9fcceab14e2b241e5eee12a5446768160e32228fdc8a4e0f7e2842c48eed5",
		},
	}

	for _, tc := range hashCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg
			cfg.ViewPrefix = tc.viewPrefix
			cfg.PayloadColumnDenylist = tc.denylist
			defs, err := cfg.EvalViewDefinitions()
			if err != nil {
				t.Fatalf("EvalViewDefinitions() error = %v", err)
			}
			if len(defs) != 2 {
				t.Fatalf("len(EvalViewDefinitions()) = %d, want 2", len(defs))
			}
			assertQuerySHA256(t, defs[0].Name, defs[0].Query, tc.wantSessionLen, tc.wantSessionSHA256)
			assertQuerySHA256(t, defs[1].Name, defs[1].Query, tc.wantTurnLen, tc.wantTurnSHA256)
		})
	}

	t.Run("default projects all payload columns", func(t *testing.T) {
		defs, err := baseCfg.EvalViewDefinitions()
		if err != nil {
			t.Fatalf("EvalViewDefinitions() error = %v", err)
		}
		for _, def := range defs {
			rawCTE := extractRawEventsCTE(t, def.Query)
			for _, col := range []string{"\n      content,\n", "\n      content_parts,\n", "\n      attributes,\n", "\n      latency_ms,\n"} {
				if !strings.Contains(rawCTE, col) {
					t.Errorf("%s raw_events CTE missing %q:\n%s", def.Name, strings.TrimSpace(col), rawCTE)
				}
			}
			if strings.Contains(rawCTE, "CAST(NULL AS JSON)") || strings.Contains(rawCTE, "ARRAY<STRUCT<") {
				t.Errorf("%s raw_events CTE unexpectedly contains NULL projection:\n%s", def.Name, rawCTE)
			}
		}
	})

	t.Run("content_parts, attributes, and latency_ms denied project typed NULLs in raw_events", func(t *testing.T) {
		cfg := baseCfg
		cfg.PayloadColumnDenylist = []string{columnContentParts, columnAttributes, columnLatencyMS}
		defs, err := cfg.EvalViewDefinitions()
		if err != nil {
			t.Fatalf("EvalViewDefinitions() error = %v", err)
		}
		for _, def := range defs {
			rawCTE := extractRawEventsCTE(t, def.Query)
			for _, want := range []string{
				"ARRAY<STRUCT<part_index INT64, mime_type STRING, text STRING, part_attributes STRING, storage_mode STRING>>[] AS content_parts",
				"CAST(NULL AS JSON) AS attributes",
				"CAST(NULL AS JSON) AS latency_ms",
			} {
				if !strings.Contains(rawCTE, want) {
					t.Errorf("%s raw_events CTE missing %q:\n%s", def.Name, want, rawCTE)
				}
			}
		}
	})

	t.Run("content denied is rejected", func(t *testing.T) {
		cfg := baseCfg
		cfg.PayloadColumnDenylist = []string{columnContent}
		if _, err := cfg.EvalViewDefinitions(); err == nil || !strings.Contains(err.Error(), `cannot build eval view SQL when "content" is in PayloadColumnDenylist`) {
			t.Errorf("EvalViewDefinitions() with denied content error = %v, want content denylist error", err)
		}
	})
}

func TestViewAndEvalViewDefinitions_TableNameUnicodeAndControlChars(t *testing.T) {
	positiveCases := []struct {
		name              string
		tableName         string
		wantEvalSHA       bool
		wantSessionLen    int
		wantSessionSHA256 string
		wantTurnLen       int
		wantTurnSHA256    string
	}{
		{
			name:              "ASCII space",
			tableName:         "table 01",
			wantEvalSHA:       true,
			wantSessionLen:    24508,
			wantSessionSHA256: "967ac1a7bf5147ace32550601567e62bbf38fc143b5f7c02973a3c38f6d4429c",
			wantTurnLen:       23072,
			wantTurnSHA256:    "9a10b5456a96cd340a1f1608fa9ad8f007510c415a94ceb9311621ce02b8d29d",
		},
		{
			name:              "Unicode Zs NO-BREAK SPACE",
			tableName:         "table\u00a001",
			wantEvalSHA:       true,
			wantSessionLen:    24509,
			wantSessionSHA256: "6f8801b51d0f66c15214f3d8d3c505b4b671e47eaefe6e8dcd62e47894fdf53b",
			wantTurnLen:       23073,
			wantTurnSHA256:    "711f9f18c3920cbc77c4a8643b00311dae6aa0fdb5f12212008dd68c3822331f",
		},
		{
			name:      "Unicode Zs EM SPACE",
			tableName: "table\u200301",
		},
		{
			name:      "Unicode Zs IDEOGRAPHIC SPACE",
			tableName: "table\u300001",
		},
		{
			name:      "Unicode combining acute mark",
			tableName: "tablea\u0301",
		},
	}

	for _, tc := range positiveCases {
		t.Run("accept_"+tc.name, func(t *testing.T) {
			cfg := Config{ProjectID: "p", DatasetID: "d", TableName: tc.tableName}
			wantTableRef := "`p.d." + tc.tableName + "`"

			baseDefs, err := cfg.ViewDefinitions()
			if err != nil {
				t.Fatalf("ViewDefinitions() error = %v", err)
			}
			if len(baseDefs) == 0 {
				t.Fatal("ViewDefinitions() returned 0 views")
			}
			for _, def := range baseDefs {
				if !strings.Contains(def.Query, wantTableRef) {
					t.Errorf("base view %s query missing %q", def.Name, wantTableRef)
				}
			}

			evalDefs, err := cfg.EvalViewDefinitions()
			if err != nil {
				t.Fatalf("EvalViewDefinitions() error = %v", err)
			}
			if len(evalDefs) != 2 {
				t.Fatalf("len(EvalViewDefinitions()) = %d, want 2", len(evalDefs))
			}
			for _, def := range evalDefs {
				if !strings.Contains(def.Query, wantTableRef) {
					t.Errorf("eval view %s query missing %q", def.Name, wantTableRef)
				}
			}
			if tc.wantEvalSHA {
				assertQuerySHA256(t, evalDefs[0].Name, evalDefs[0].Query, tc.wantSessionLen, tc.wantSessionSHA256)
				assertQuerySHA256(t, evalDefs[1].Name, evalDefs[1].Query, tc.wantTurnLen, tc.wantTurnSHA256)
			}
		})
	}

	negativeCases := []struct {
		name      string
		tableName string
	}{
		{name: "backtick", tableName: "table`01"},
		{name: "backslash", tableName: `table\01`},
		{name: "newline", tableName: "table\n01"},
		{name: "carriage return", tableName: "table\r01"},
		{name: "NUL byte", tableName: "table\x0001"},
		{name: "Unicode Zl U+2028", tableName: "table\u202801"},
		{name: "Unicode Zp U+2029", tableName: "table\u202901"},
		{name: "whitespace-only Zs", tableName: "\u00a0\u2003"},
		{name: "semicolon", tableName: "tbl;DROP"},
		{name: "dot", tableName: "proj.ds.tbl"},
		{name: "exceeds max length", tableName: strings.Repeat("a", 1025)},
		{name: "invalid UTF-8", tableName: "\xff\xfe"},
	}

	for _, tc := range negativeCases {
		t.Run("reject_"+tc.name, func(t *testing.T) {
			cfg := Config{ProjectID: "p", DatasetID: "d", TableName: tc.tableName}
			if _, err := cfg.ViewDefinitions(); err == nil || !strings.Contains(err.Error(), "invalid BigQuery table name") {
				t.Errorf("ViewDefinitions() error = %v, want invalid BigQuery table name", err)
			}
			if _, err := cfg.EvalViewDefinitions(); err == nil || !strings.Contains(err.Error(), "invalid BigQuery table name") {
				t.Errorf("EvalViewDefinitions() error = %v, want invalid BigQuery table name", err)
			}
		})
	}
}

func TestCreateEvalViews_ConfigValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("rejects content in PayloadColumnDenylist when CreateEvalViews is true", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.CreateEvalViews = true
		cfg.PayloadColumnDenylist = []string{columnContent}
		_, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, nil)
		if err == nil || !strings.Contains(err.Error(), `CreateEvalViews cannot be used when "content" is in PayloadColumnDenylist`) {
			t.Errorf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want content denylist rejection", err)
		}
	})

	t.Run("allows content_parts, attributes, and latency_ms in PayloadColumnDenylist when CreateEvalViews is true", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.PayloadColumnDenylist = []string{columnContentParts, columnAttributes, columnLatencyMS}
		})
		_ = p
	})

	t.Run("rejects required events in EventDenylist when CreateEvalViews is true", func(t *testing.T) {
		wantLegacyDenied := []string{
			EventAgentResponse,
			EventAgentTransfer,
			EventHITLConfirmationRequest,
			EventHITLConfirmationCompleted,
			EventHITLCredentialRequest,
			EventHITLCredentialCompleted,
			EventHITLInputRequest,
			EventHITLInputCompleted,
			EventToolCompleted,
			EventToolPaused,
		}
		for _, reqEvent := range append(append([]string(nil), evalViewRequiredEvents...), legacyEventType) {
			cfg := DefaultConfig()
			cfg.ProjectID = "test-project"
			cfg.DatasetID = "test_dataset"
			cfg.TableName = "test-table"
			cfg.CreateEvalViews = true
			cfg.EventDenylist = []string{reqEvent}
			_, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, nil)
			if err == nil || !strings.Contains(err.Error(), "CreateEvalViews cannot be used when required events are in EventDenylist") {
				t.Errorf("EventDenylist=[%q] error = %v, want required trace events error", reqEvent, err)
				continue
			}
			if reqEvent == legacyEventType {
				_, rawList, found := strings.Cut(err.Error(), ": [")
				if !found {
					t.Errorf("EventDenylist=[%q] error = %v, want bracketed event list", reqEvent, err)
					continue
				}
				gotDenied := strings.Fields(strings.TrimSuffix(rawList, "]"))
				if diff := cmp.Diff(wantLegacyDenied, gotDenied); diff != "" {
					t.Errorf("EventDenylist=[%q] denied events mismatch (-want +got):\n%s", reqEvent, diff)
				}
			}
		}
	})

	t.Run("allows non-required events in EventDenylist when CreateEvalViews is true", func(t *testing.T) {
		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.EventDenylist = []string{EventStateDelta, EventNodeOutput}
		})
		_ = p
	})

	t.Run("rejects partial EventAllowlist and checks EVENT-umbrella parity with FinalResponseToolNames", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.CreateEvalViews = true
		cfg.EventAllowlist = []string{EventUserMessageReceived, EventAgentResponse}
		_, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, nil)
		if err == nil || !strings.Contains(err.Error(), "CreateEvalViews cannot be used unless EventAllowlist includes all required events; missing:") || !strings.Contains(err.Error(), EventToolStarting) {
			t.Errorf("partial EventAllowlist error = %v, want missing required events error", err)
		}

		pFull, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.EventAllowlist = append([]string(nil), evalViewRequiredEvents...)
		})
		_ = pFull

		umbrellaAllowlist := []string{
			legacyEventType,
			EventAgentStarting,
			EventInvocationCompleted,
			EventLLMRequest,
			EventToolCompleted,
			EventToolStarting,
			EventUserMessageReceived,
		}
		// Without FinalResponseToolNames, "EVENT" covers AGENT_RESPONSE (only emitted via logSessionEvent).
		pUmbrella, umbrellaReqs, umbrellaCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.EventAllowlist = append([]string(nil), umbrellaAllowlist...)
		})
		bqaapUmbrella, ok := AsBigQueryAgentAnalyticsPlugin(pUmbrella)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(pUmbrella) = false")
		}
		umbrellaRunner := &fakeViewRunner{}
		bqaapUmbrella.runner = umbrellaRunner
		if err := bqaapUmbrella.SyncEvalViews(ctx); err != nil {
			t.Fatalf("pUmbrella.SyncEvalViews() error = %v, want nil", err)
		}
		if got := len(umbrellaRunner.executedNames()); got != 2 {
			t.Errorf("pUmbrella.SyncEvalViews() executed %d views, want 2", got)
		}
		var umbrellaErrs callbackErrors
		umbrellaErrs.add(pUmbrella.OnEventCallback()(umbrellaCtx, &session.Event{
			ID:     "ev-umbrella-model",
			Author: "agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "umbrella answer"}}},
			},
		}))
		umbrellaErrs.check(t)
		pUmbrella.AfterRunCallback()(umbrellaCtx)
		umbrellaRows := collectRows(t, umbrellaReqs, untilEventType(EventInvocationCompleted))
		if diff := cmp.Diff([]string{EventAgentResponse, EventInvocationCompleted}, eventTypes(umbrellaRows)); diff != "" {
			t.Errorf("umbrella without FinalResponseToolNames emitted types mismatch (-want +got):\n%s", diff)
		}

		// With FinalResponseToolNames, final-response-tool AGENT_RESPONSE uses logEvent (no "EVENT" alias),
		// so both CreateEvalViews=true and SyncEvalViews must reject umbrella allowlist missing AGENT_RESPONSE.
		cfgUmbrellaFinal := DefaultConfig()
		cfgUmbrellaFinal.ProjectID = "test-project"
		cfgUmbrellaFinal.DatasetID = "test_dataset"
		cfgUmbrellaFinal.TableName = "test-table"
		cfgUmbrellaFinal.CreateEvalViews = true
		cfgUmbrellaFinal.EventAllowlist = append([]string(nil), umbrellaAllowlist...)
		cfgUmbrellaFinal.FinalResponseToolNames = []string{"final_answer"}
		_, err = NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfgUmbrellaFinal, nil, nil)
		if err == nil || !strings.Contains(err.Error(), "missing: [AGENT_RESPONSE]") {
			t.Errorf("umbrella EventAllowlist + FinalResponseToolNames error = %v, want missing AGENT_RESPONSE", err)
		}

		pUmbrellaFinalSync, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = false
			c.EventAllowlist = append([]string(nil), umbrellaAllowlist...)
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		bqaapUmbrellaFinalSync, _ := AsBigQueryAgentAnalyticsPlugin(pUmbrellaFinalSync)
		rejectedRunner := &fakeViewRunner{}
		bqaapUmbrellaFinalSync.runner = rejectedRunner
		if err := bqaapUmbrellaFinalSync.SyncEvalViews(ctx); err == nil || !strings.Contains(err.Error(), "missing: [AGENT_RESPONSE]") {
			t.Errorf("SyncEvalViews() with umbrella EventAllowlist + FinalResponseToolNames error = %v, want missing AGENT_RESPONSE", err)
		}
		if got := len(rejectedRunner.executedNames()); got != 0 {
			t.Errorf("failed SyncEvalViews() executed %d DDL queries, want 0", got)
		}

		// Control: explicit EventAgentResponse with FinalResponseToolNames succeeds and writes both
		// final-response-tool and model AGENT_RESPONSE rows.
		explicitAllowlist := append(append([]string(nil), umbrellaAllowlist...), EventAgentResponse)
		pExplicitFinal, explicitReqs, explicitCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.EventAllowlist = explicitAllowlist
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		explicitCtx.functionCallID = "call-final-1"
		bqaapExplicitFinal, _ := AsBigQueryAgentAnalyticsPlugin(pExplicitFinal)
		explicitRunner := &fakeViewRunner{}
		bqaapExplicitFinal.runner = explicitRunner
		if err := bqaapExplicitFinal.SyncEvalViews(ctx); err != nil {
			t.Fatalf("explicit AGENT_RESPONSE SyncEvalViews() error = %v, want nil", err)
		}
		if got := len(explicitRunner.executedNames()); got != 2 {
			t.Errorf("explicit AGENT_RESPONSE SyncEvalViews() executed %d views, want 2", got)
		}
		var explicitErrs callbackErrors
		explicitErrs.add(pExplicitFinal.AfterToolCallback()(explicitCtx, &fakeTool{name: "final_answer"}, map[string]any{"answer": "42"}, map[string]any{}, nil))
		explicitErrs.add(pExplicitFinal.OnEventCallback()(explicitCtx, &session.Event{
			ID:     "ev-explicit-model",
			Author: "agent",
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "explicit model answer"}}},
			},
		}))
		explicitErrs.check(t)
		pExplicitFinal.AfterRunCallback()(explicitCtx)
		explicitRows := collectRows(t, explicitReqs, untilEventType(EventInvocationCompleted))
		wantExplicitTypes := []string{EventToolCompleted, EventAgentResponse, EventAgentResponse, EventInvocationCompleted}
		if diff := cmp.Diff(wantExplicitTypes, eventTypes(explicitRows)); diff != "" {
			t.Errorf("explicit AGENT_RESPONSE + FinalResponseToolNames emitted types mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestCreateEvalViews_StartupAndSyncLifecycle(t *testing.T) {
	ctx := context.Background()

	newWriteClient := func(t *testing.T) *bqstorage.BigQueryWriteClient {
		t.Helper()
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

	t.Run("CreateEvalViews with DisableCreateViews false creates 27 views on startup and in SyncViews", func(t *testing.T) {
		const wantTotalViews = 27
		allDone := make(chan struct{})
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
						if int(jobCount.Add(1)) == wantTotalViews {
							close(allDone)
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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.CreateEvalViews = true

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		t.Cleanup(func() { _ = p.Close() })

		select {
		case <-allDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %d startup view DDL jobs; got %d", wantTotalViews, jobCount.Load())
		}

		mu.Lock()
		gotQueries := append([]string(nil), jobQueries...)
		mu.Unlock()
		if len(gotQueries) != wantTotalViews {
			t.Fatalf("startup DDL count = %d, want %d", len(gotQueries), wantTotalViews)
		}
		if !strings.Contains(gotQueries[len(gotQueries)-2], "v_eval_session") {
			t.Errorf("second-to-last startup view query missing v_eval_session:\n%s", gotQueries[len(gotQueries)-2][:120])
		}
		if !strings.Contains(gotQueries[len(gotQueries)-1], "v_eval_turn") {
			t.Errorf("last startup view query missing v_eval_turn:\n%s", gotQueries[len(gotQueries)-1][:120])
		}

		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{}
		bqaap.runner = fakeRunner
		if err := bqaap.SyncViews(ctx); err != nil {
			t.Fatalf("SyncViews() error = %v", err)
		}
		names := fakeRunner.executedNames()
		if len(names) != wantTotalViews {
			t.Fatalf("SyncViews() with CreateEvalViews=true executed %d views, want %d", len(names), wantTotalViews)
		}
		if names[len(names)-2] != "v_eval_session" || names[len(names)-1] != "v_eval_turn" {
			t.Errorf("SyncViews() trailing views = %v, want [v_eval_session, v_eval_turn]", names[len(names)-2:])
		}
	})

	t.Run("standalone CreateEvalViews with DisableCreateViews true creates only the 2 eval views on startup", func(t *testing.T) {
		allDone := make(chan struct{})
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
						if int(jobCount.Add(1)) == 2 {
							close(allDone)
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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.DisableCreateViews = true
		cfg.CreateEvalViews = true
		cfg.ViewPrefix = "v_only_eval"

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-allDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for 2 eval view DDL jobs; got %d", jobCount.Load())
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		if len(jobQueries) != 2 {
			t.Fatalf("executed %d DDL jobs, want 2", len(jobQueries))
		}
		if !strings.Contains(jobQueries[0], "v_only_eval_eval_session") || !strings.Contains(jobQueries[1], "v_only_eval_eval_turn") {
			t.Errorf("unexpected standalone eval view queries:\n[0]=%s\n[1]=%s", jobQueries[0][:120], jobQueries[1][:120])
		}
	})

	t.Run("SyncEvalViews executes the 2 eval views even when CreateEvalViews is false and validates config", func(t *testing.T) {
		var nilPlugin *BigQueryAgentAnalyticsPlugin
		if err := nilPlugin.SyncEvalViews(ctx); err == nil {
			t.Error("nilPlugin.SyncEvalViews() error = nil, want error")
		}

		p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = false
			c.ViewPrefix = "v_manual"
		})
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{
			errFor: map[string]error{
				"v_manual_eval_session": &googleapi.Error{Code: http.StatusConflict, Message: "conflict"},
			},
		}
		bqaap.runner = fakeRunner
		if err := bqaap.SyncEvalViews(ctx); err != nil {
			t.Fatalf("SyncEvalViews() error = %v, want nil", err)
		}
		if diff := cmp.Diff([]string{"v_manual_eval_session", "v_manual_eval_turn"}, fakeRunner.executedNames()); diff != "" {
			t.Errorf("SyncEvalViews() executed names mismatch (-want +got):\n%s", diff)
		}

		// When content is in PayloadColumnDenylist (with CreateEvalViews=false at construction),
		// SyncEvalViews rejects the call with a clear error.
		pDeniedContent, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.PayloadColumnDenylist = []string{columnContent}
		})
		bqaapDenied, _ := AsBigQueryAgentAnalyticsPlugin(pDeniedContent)
		deniedRunner := &fakeViewRunner{}
		bqaapDenied.runner = deniedRunner
		if err := bqaapDenied.SyncEvalViews(ctx); err == nil || !strings.Contains(err.Error(), `cannot create eval views when "content" is in PayloadColumnDenylist`) {
			t.Errorf("SyncEvalViews() with content denied error = %v, want content denylist error", err)
		}
		if got := len(deniedRunner.executedNames()); got != 0 {
			t.Errorf("bqaapDenied.SyncEvalViews() executed %d DDL queries, want 0", got)
		}

		// When a required trace event is excluded by EventDenylist (with CreateEvalViews=false),
		// SyncEvalViews rejects the call.
		pDeniedEvent, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.EventDenylist = []string{EventToolStarting}
		})
		bqaapDeniedEvent, _ := AsBigQueryAgentAnalyticsPlugin(pDeniedEvent)
		deniedEventRunner := &fakeViewRunner{}
		bqaapDeniedEvent.runner = deniedEventRunner
		if err := bqaapDeniedEvent.SyncEvalViews(ctx); err == nil || !strings.Contains(err.Error(), "cannot create eval views when required events are in EventDenylist") {
			t.Errorf("SyncEvalViews() with TOOL_STARTING denied error = %v, want required trace events error", err)
		}
		if got := len(deniedEventRunner.executedNames()); got != 0 {
			t.Errorf("bqaapDeniedEvent.SyncEvalViews() executed %d DDL queries, want 0", got)
		}

		// When TableName collides with v_eval_session and CreateEvalViews=true, startup skips view creation and logs why.
		logger := &recordingLogger{}
		pCollision, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.TableName = "v_eval_session"
			c.Logger = logger
		})
		if !logger.contains("agentanalytics: skipping view creation") {
			t.Errorf("expected startup log for skipped eval view creation on table collision, got:\n%s", logger)
		}
		bqaapCollision, _ := AsBigQueryAgentAnalyticsPlugin(pCollision)
		collisionRunner := &fakeViewRunner{}
		bqaapCollision.runner = collisionRunner
		if err := bqaapCollision.SyncEvalViews(ctx); err == nil || !strings.Contains(err.Error(), "collides with the events table") {
			t.Errorf("SyncEvalViews() on colliding TableName error = %v, want collision error", err)
		}
		if got := len(collisionRunner.executedNames()); got != 0 {
			t.Errorf("bqaapCollision.SyncEvalViews() executed %d DDL queries, want 0", got)
		}
	})

	t.Run("standalone CreateEvalViews logs SyncEvalViews retry hint when DDL fails", func(t *testing.T) {
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						return &http.Response{
							StatusCode: http.StatusForbidden,
							Body:       io.NopCloser(strings.NewReader(`{"error":{"code":403,"message":"Access Denied"}}`)),
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
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		cfg.DisableCreateViews = true
		cfg.CreateEvalViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		deadline := time.Now().Add(5 * time.Second)
		for !logger.contains("view creation finished") && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if want := "view creation finished: 0 created, 2 failed of 2; call SyncEvalViews to retry"; !logger.contains(want) {
			t.Errorf("expected retry hint %q in logs, got:\n%s", want, logger.String())
		}
		if logger.contains("call SyncViews to retry") {
			t.Errorf("eval-only startup pass should not tell caller to call SyncViews, got:\n%s", logger.String())
		}
	})

	t.Run("startup still runs per-event views when eval views are skipped on TableName collision", func(t *testing.T) {
		allDone := make(chan struct{})
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
						if int(jobCount.Add(1)) == len(eventViewSpecs) {
							close(allDone)
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

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "v_eval_session"
		cfg.DisableCreateViews = false
		cfg.CreateEvalViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-allDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %d per-event startup views; got %d", len(eventViewSpecs), jobCount.Load())
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if !logger.contains("agentanalytics: skipping view creation") || !logger.contains("v_eval_session") {
			t.Errorf("expected startup log for skipped eval view creation on TableName collision, got:\n%s", logger.String())
		}
		mu.Lock()
		gotQueries := append([]string(nil), jobQueries...)
		mu.Unlock()
		if len(gotQueries) != len(eventViewSpecs) {
			t.Fatalf("startup DDL count = %d, want %d", len(gotQueries), len(eventViewSpecs))
		}
		for _, q := range gotQueries {
			if strings.Contains(q, "CREATE OR REPLACE VIEW `test-project.test_dataset.v_eval_turn`") {
				t.Errorf("unexpected eval view DDL executed despite eval collision skip:\n%s", q[:min(120, len(q))])
			}
		}
	})

	t.Run("SyncViews isolates per-event and eval failures so valid view groups still sync and return joined errors", func(t *testing.T) {
		// 1. Valid per-event views + colliding eval views (TableName="v_eval_session"):
		// transient startup failure logs "call SyncViews to retry", and calling SyncViews
		// executes all 25 per-event views while still reporting the eval collision error.
		var failedFirstStartupJob atomic.Bool
		allStartupDone := make(chan struct{})
		var startupCount atomic.Int32
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						idx := int(startupCount.Add(1))
						if idx == 1 && !failedFirstStartupJob.Swap(true) {
							return &http.Response{
								StatusCode: http.StatusForbidden,
								Body:       io.NopCloser(strings.NewReader(`{"error":{"code":403,"message":"Access Denied"}}`)),
							}, nil
						}
						if idx == len(eventViewSpecs) {
							close(allStartupDone)
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

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "v_eval_session"
		cfg.DisableCreateViews = false
		cfg.CreateEvalViews = true
		cfg.Logger = logger

		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-allStartupDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %d startup views; got %d", len(eventViewSpecs), startupCount.Load())
		}
		// Give logger a moment to record async completion after final HTTP response.
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) && !logger.contains("call SyncViews to retry") {
			time.Sleep(5 * time.Millisecond)
		}
		if !logger.contains("call SyncViews to retry") {
			t.Fatalf("startup log missing SyncViews retry hint after transient failure; got:\n%s", logger.String())
		}

		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		fakeRunner := &fakeViewRunner{}
		bqaap.runner = fakeRunner
		syncErr := bqaap.SyncViews(ctx)
		if syncErr == nil || !strings.Contains(syncErr.Error(), "collides with the events table") {
			t.Errorf("SyncViews() error = %v, want eval TableName collision error", syncErr)
		}
		if got := len(fakeRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() executed %d per-event views, want %d", got, len(eventViewSpecs))
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		// 2. Valid per-event views + evalConfigErr + partial DDL failure: all 25 per-event
		// views run and both the DDL error and eval config error are joined.
		pEvalDenied, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
		})
		bqaapEvalDenied, _ := AsBigQueryAgentAnalyticsPlugin(pEvalDenied)
		bqaapEvalDenied.evalConfigErr = errors.New("eval config error")
		evalDeniedRunner := &fakeViewRunner{
			errFor: map[string]error{"v_tool_error": errors.New("ddl failed")},
		}
		bqaapEvalDenied.runner = evalDeniedRunner
		errEvalDenied := bqaapEvalDenied.SyncViews(ctx)
		if errEvalDenied == nil || !strings.Contains(errEvalDenied.Error(), "ddl failed") || !strings.Contains(errEvalDenied.Error(), "eval config error") {
			t.Errorf("SyncViews() error = %v, want joined ddl failed and eval config error", errEvalDenied)
		}
		if got := len(evalDeniedRunner.executedNames()); got != len(eventViewSpecs) {
			t.Errorf("SyncViews() executed %d views, want %d per-event views", got, len(eventViewSpecs))
		}

		// 3. Colliding per-event views (TableName="v_llm_request") + valid eval views:
		// the 2 eval views still sync while returning the per-event collision error.
		pPerEventCollision, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.TableName = "v_llm_request"
		})
		bqaapPerEventCol, _ := AsBigQueryAgentAnalyticsPlugin(pPerEventCollision)
		perEventColRunner := &fakeViewRunner{}
		bqaapPerEventCol.runner = perEventColRunner
		errPerEventCol := bqaapPerEventCol.SyncViews(ctx)
		if errPerEventCol == nil || !strings.Contains(errPerEventCol.Error(), "collides with the events table") {
			t.Errorf("SyncViews() error = %v, want per-event TableName collision error", errPerEventCol)
		}
		if diff := cmp.Diff([]string{"v_eval_session", "v_eval_turn"}, perEventColRunner.executedNames()); diff != "" {
			t.Errorf("SyncViews() executed names mismatch (-want +got):\n%s", diff)
		}

		// 4. Both groups invalid: SyncViews returns joined error with 0 executed views.
		pBothInvalid, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.TableName = "v_llm_request"
		})
		bqaapBothInvalid, _ := AsBigQueryAgentAnalyticsPlugin(pBothInvalid)
		bqaapBothInvalid.evalConfigErr = errors.New("eval config error")
		bothInvalidRunner := &fakeViewRunner{}
		bqaapBothInvalid.runner = bothInvalidRunner
		errBoth := bqaapBothInvalid.SyncViews(ctx)
		if errBoth == nil || !strings.Contains(errBoth.Error(), "collides with the events table") || !strings.Contains(errBoth.Error(), "eval config error") {
			t.Errorf("SyncViews() error = %v, want joined per-event collision and eval config errors", errBoth)
		}
		if got := len(bothInvalidRunner.executedNames()); got != 0 {
			t.Errorf("SyncViews() executed %d views, want 0 when both groups are invalid", got)
		}

		// 5. Mixed valid per-event + colliding eval views closed mid-sync still returns errPluginClosed verbatim.
		pClosedMidSync, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.DisableCreateViews = true
			c.CreateEvalViews = true
			c.TableName = "v_eval_session"
		})
		bqaapClosedMidSync, _ := AsBigQueryAgentAnalyticsPlugin(pClosedMidSync)
		inDDL := make(chan struct{})
		closeDone := make(chan error, 1)
		bqaapClosedMidSync.runner = &fakeViewRunner{
			onRun: func(callCtx context.Context, viewName string) error {
				select {
				case <-inDDL:
				default:
					close(inDDL)
					go func() {
						closeDone <- pClosedMidSync.Close()
					}()
				}
				<-callCtx.Done()
				return callCtx.Err()
			},
		}
		errClosed := bqaapClosedMidSync.SyncViews(ctx)
		select {
		case <-inDDL:
			if closeErr := <-closeDone; closeErr != nil {
				t.Fatalf("Close() error = %v", closeErr)
			}
		default:
			t.Errorf("SyncViews() never entered DDL runner (err=%v)", errClosed)
		}
		if !errors.Is(errClosed, errPluginClosed) || errClosed.Error() != errPluginClosed.Error() {
			t.Errorf("SyncViews() on mid-sync Close = %v, want exact errPluginClosed (%v)", errClosed, errPluginClosed)
		}
	})

	t.Run("SyncEvalViews validation failure leaves blocked startup pass untouched and valid SyncEvalViews preempts it", func(t *testing.T) {
		// 1. Failing validation in SyncEvalViews must NOT cancel an in-flight startup pass.
		bgInFlight := make(chan struct{})
		unblockBg := make(chan struct{})
		allBgDone := make(chan struct{})
		var bgCanceled atomic.Bool
		var bgJobCount atomic.Int32

		httpClientNoCancel := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						n := int(bgJobCount.Add(1))
						if n == 1 {
							close(bgInFlight)
							select {
							case <-r.Context().Done():
								bgCanceled.Store(true)
								return nil, r.Context().Err()
							case <-unblockBg:
							}
						}
						if n == len(eventViewSpecs) {
							close(allBgDone)
						}
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"job-bg"},"status":{"state":"DONE"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}
		bqClient1, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClientNoCancel))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient1.Close() })

		cfg1 := DefaultConfig()
		cfg1.ProjectID = "test-project"
		cfg1.DatasetID = "test_dataset"
		cfg1.TableName = "test-table"
		cfg1.EventDenylist = []string{EventToolStarting}

		p1, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg1, bqClient1, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		bqaap1, ok := AsBigQueryAgentAnalyticsPlugin(p1)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p1) = false")
		}
		select {
		case <-bgInFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for background startup view creation")
		}
		if err := bqaap1.SyncEvalViews(ctx); err == nil || !strings.Contains(err.Error(), "cannot create eval views when required events are in EventDenylist") {
			t.Errorf("invalid SyncEvalViews() error = %v, want EventDenylist error", err)
		}
		if bgCanceled.Load() {
			t.Error("invalid SyncEvalViews() should not cancel in-flight startup view creation")
		}
		close(unblockBg)
		select {
		case <-allBgDone:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for %d startup views to finish; got %d", len(eventViewSpecs), bgJobCount.Load())
		}
		if err := p1.Close(); err != nil {
			t.Fatalf("p1.Close() error = %v", err)
		}

		// 2. Valid SyncEvalViews preempts blocked startup pass and creates only the 2 eval views.
		preemptInFlight := make(chan struct{})
		preemptCanceled := make(chan struct{})
		var preemptIdx atomic.Int32
		var syncMu sync.Mutex
		var syncQueries []string

		httpClientPreempt := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						if preemptIdx.Add(1) == 1 {
							close(preemptInFlight)
							<-r.Context().Done()
							close(preemptCanceled)
							return nil, r.Context().Err()
						}
						b, _ := io.ReadAll(r.Body)
						var payload struct {
							Configuration struct {
								Query struct {
									Query string `json:"query"`
								} `json:"query"`
							} `json:"configuration"`
						}
						_ = json.Unmarshal(b, &payload)
						syncMu.Lock()
						syncQueries = append(syncQueries, payload.Configuration.Query.Query)
						syncMu.Unlock()
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
		bqClient2, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClientPreempt))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient2.Close() })

		logger := &recordingLogger{}
		cfg2 := DefaultConfig()
		cfg2.ProjectID = "test-project"
		cfg2.DatasetID = "test_dataset"
		cfg2.TableName = "test-table"
		cfg2.Logger = logger

		p2, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg2, bqClient2, newWriteClient(t))
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		t.Cleanup(func() { _ = p2.Close() })

		bqaap2, ok := AsBigQueryAgentAnalyticsPlugin(p2)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p2) = false")
		}
		select {
		case <-preemptInFlight:
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for background startup view creation to enter RoundTrip")
		}
		if err := bqaap2.SyncEvalViews(ctx); err != nil {
			t.Fatalf("SyncEvalViews() error = %v, want nil", err)
		}
		select {
		case <-preemptCanceled:
		default:
			t.Error("expected SyncEvalViews() to cancel in-flight background view creation")
		}
		syncMu.Lock()
		gotSyncQueries := append([]string(nil), syncQueries...)
		syncMu.Unlock()
		if len(gotSyncQueries) != 2 {
			t.Fatalf("SyncEvalViews() executed %d DDL jobs, want 2", len(gotSyncQueries))
		}
		if !strings.Contains(gotSyncQueries[0], "v_eval_session") || !strings.Contains(gotSyncQueries[1], "v_eval_turn") {
			t.Errorf("SyncEvalViews() executed unexpected views:\n[0]=%s\n[1]=%s", gotSyncQueries[0][:min(120, len(gotSyncQueries[0]))], gotSyncQueries[1][:min(120, len(gotSyncQueries[1]))])
		}
	})
}

type evalAgentInvocationContext struct {
	*mockInvocationContext
	ag agent.Agent
}

func (c *evalAgentInvocationContext) Agent() agent.Agent { return c.ag }

func TestEvalViews_JSONPathCompatibilityWithWrittenRows(t *testing.T) {
	evalDefs, err := (Config{ProjectID: "test-project", DatasetID: "test_dataset", TableName: "agent_events"}).EvalViewDefinitions()
	if err != nil {
		t.Fatalf("EvalViewDefinitions() error = %v", err)
	}
	jsonPathLiteralRe := regexp.MustCompile(`'(\$(?:\.[^']+)?)'`)
	extractedSet := make(map[string]bool)
	for _, def := range evalDefs {
		for _, m := range jsonPathLiteralRe.FindAllStringSubmatch(def.Query, -1) {
			extractedSet[m[1]] = true
		}
	}
	gotSQLPaths := make([]string, 0, len(extractedSet))
	for p := range extractedSet {
		gotSQLPaths = append(gotSQLPaths, p)
	}
	slices.Sort(gotSQLPaths)
	wantSQLPaths := []string{
		"$",
		"$.adk.app_name",
		"$.agent_description",
		"$.args",
		"$.cache_type",
		"$.description",
		"$.finish_reason",
		"$.from_agent",
		"$.function_response",
		"$.name",
		"$.parameters",
		"$.response",
		"$.result",
		"$.root_agent_name",
		"$.static_instruction",
		"$.sub_agents",
		"$.text_summary",
		"$.to_agent",
		"$.tool",
		"$.tools",
		"$.total_ms",
		"$.usage.completion",
		"$.usage.prompt",
		"$.usage.total",
	}
	if diff := cmp.Diff(wantSQLPaths, gotSQLPaths); diff != "" {
		t.Fatalf("extracted eval SQL JSON paths mismatch (-want +got):\n%s", diff)
	}

	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.DisableCreateViews = true
	})
	mCtx.appName = "weather-app"
	mCtx.functionCallID = "fc-weather-1"
	rootName := mCtx.AgentName()
	if rootName == "" {
		rootName = "test-agent"
		mCtx.agentName = rootName
	}

	subAg, err := llmagent.New(llmagent.Config{
		Name:        "summary-agent",
		Description: "Summarizes weather",
	})
	if err != nil {
		t.Fatalf("llmagent.New(summary-agent) error = %v", err)
	}
	rootAg, err := llmagent.New(llmagent.Config{
		Name:        rootName,
		Description: "Weather coordinator",
		Model:       &scriptedLLM{},
		Instruction: "Check weather before answering.",
		SubAgents:   []agent.Agent{subAg},
	})
	if err != nil {
		t.Fatalf("llmagent.New(%s) error = %v", rootName, err)
	}
	invCtx := &evalAgentInvocationContext{
		mockInvocationContext: mCtx,
		ag:                    rootAg,
	}

	weatherTool := &declaredTool{
		fakeTool: fakeTool{name: "get_weather", description: "Fetches current weather for a city"},
		decl: &genai.FunctionDeclaration{
			Name:                 "get_weather",
			Description:          "Fetches current weather for a city",
			ParametersJsonSchema: map[string]any{"type": "object"},
		},
	}

	userMsg := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "check weather in Paris"}},
	}
	toolRespMsg := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "fc-weather-1",
				Name:     "get_weather",
				Response: map[string]any{"temp_c": 20},
			},
		}},
	}
	llmReq := &model.LLMRequest{
		Model: "gemini-2.5-pro",
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a helpful weather assistant."}},
			},
		},
		Tools:    map[string]any{"get_weather": weatherTool},
		Contents: []*genai.Content{userMsg, toolRespMsg},
	}
	llmResp := &model.LLMResponse{
		ModelVersion: "gemini-2.5-pro-001",
		FinishReason: genai.FinishReasonStop,
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "It is 20C and sunny in Paris."}},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:        12,
			CandidatesTokenCount:    8,
			TotalTokenCount:         20,
			CachedContentTokenCount: 4,
		},
	}

	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(invCtx, userMsg))
	errs.add(p.BeforeRunCallback()(invCtx))
	errs.add(p.BeforeAgentCallback()(invCtx))
	errs.add(p.BeforeModelCallback()(invCtx, llmReq))
	errs.add(p.AfterModelCallback()(invCtx, llmResp, nil))
	errs.add(p.BeforeToolCallback()(invCtx, weatherTool, map[string]any{"city": "Paris"}))
	errs.add(p.AfterToolCallback()(invCtx, weatherTool, map[string]any{"city": "Paris"}, map[string]any{"temp_c": 20}, nil))
	errs.add(p.OnUserMessageCallback()(invCtx, toolRespMsg))

	errs.add(p.OnEventCallback()(invCtx, &session.Event{
		ID:     "ev-transfer-1",
		Author: rootName,
		Actions: session.EventActions{
			TransferToAgent: "summary-agent",
		},
	}))
	errs.add(p.OnEventCallback()(invCtx, &session.Event{
		ID:          "ev-final-1",
		Author:      rootName,
		LLMResponse: *llmResp,
	}))
	errs.add(p.AfterAgentCallback()(invCtx))
	errs.check(t)
	p.AfterRunCallback()(invCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	byType := make(map[string]map[string]any, len(rows))
	var userRows []map[string]any
	for _, r := range rows {
		et, _ := r["event_type"].(string)
		if _, exists := byType[et]; !exists {
			byType[et] = r
		}
		if et == EventUserMessageReceived {
			userRows = append(userRows, r)
		}
	}

	verifiedGoPaths := make(map[string]bool)
	checkDocPath := func(label string, doc any, path string) {
		t.Helper()
		if _, ok := resolveJSONPath(doc, path); !ok {
			t.Errorf("%s path %q did not resolve against emitted payload: %#v", label, path, doc)
			return
		}
		verifiedGoPaths[path] = true
	}
	checkPath := func(eventType, col, path string) {
		t.Helper()
		row, ok := byType[eventType]
		if !ok {
			t.Fatalf("missing emitted row for event_type %s", eventType)
		}
		doc := jsonObjectColumn(t, row, col)
		checkDocPath(eventType+"."+col, doc, path)
	}

	// Verify all JSON paths extracted in raw_events, parsed_events, and agent_rows match Go plugin payloads.
	checkPath(EventUserMessageReceived, columnContent, "$.text_summary")
	checkPath(EventAgentStarting, columnAttributes, "$.adk.app_name")
	checkPath(EventAgentStarting, columnAttributes, "$.root_agent_name")
	checkPath(EventAgentStarting, columnAttributes, "$.agent_description")
	checkPath(EventAgentStarting, columnAttributes, "$.sub_agents")
	checkPath(EventLLMRequest, columnAttributes, "$.tools")
	checkPath(EventLLMResponse, columnContent, "$.usage.prompt")
	checkPath(EventLLMResponse, columnContent, "$.usage.completion")
	checkPath(EventLLMResponse, columnContent, "$.usage.total")
	checkPath(EventLLMResponse, columnAttributes, "$.finish_reason")
	checkPath(EventLLMResponse, columnAttributes, "$.cache_type")
	checkPath(EventToolStarting, columnContent, "$.tool")
	checkPath(EventToolStarting, columnContent, "$.args")
	checkPath(EventToolCompleted, columnContent, "$.tool")
	checkPath(EventToolCompleted, columnContent, "$.result")
	checkPath(EventAgentTransfer, columnContent, "$.from_agent")
	checkPath(EventAgentTransfer, columnContent, "$.to_agent")
	checkPath(EventAgentResponse, columnContent, "$.response")
	checkPath(EventInvocationCompleted, columnLatencyMS, "$.total_ms")

	llmReqRow := byType[EventLLMRequest]
	reqAttrs := jsonObjectColumn(t, llmReqRow, columnAttributes)
	toolsSlice, ok := reqAttrs["tools"].([]any)
	if !ok || len(toolsSlice) == 0 {
		t.Fatalf("LLM_REQUEST.attributes.tools = %#v, want non-empty array", reqAttrs["tools"])
	}
	toolDecl, ok := toolsSlice[0].(map[string]any)
	if !ok {
		t.Fatalf("LLM_REQUEST.attributes.tools[0] = %#v, want object", toolsSlice[0])
	}
	checkDocPath("LLM_REQUEST.attributes.tools[0]", toolDecl, "$.name")
	checkDocPath("LLM_REQUEST.attributes.tools[0]", toolDecl, "$.description")
	checkDocPath("LLM_REQUEST.attributes.tools[0]", toolDecl, "$.parameters")

	checkFuncRespPartAttr := func(label string, targetRows ...map[string]any) bool {
		t.Helper()
		for _, r := range targetRows {
			for _, part := range contentPartsOf(t, r) {
				rawPartAttrs, _ := part["part_attributes"].(string)
				if rawPartAttrs == "" {
					continue
				}
				var partAttrs map[string]any
				if err := json.Unmarshal([]byte(rawPartAttrs), &partAttrs); err != nil {
					t.Fatalf("%s part_attributes JSON unmarshal error: %v (%q)", label, err, rawPartAttrs)
				}
				if _, ok := partAttrs["function_response"]; ok {
					checkDocPath(label+".content_parts[].part_attributes", partAttrs, "$.function_response")
					return true
				}
			}
		}
		return false
	}
	if !checkFuncRespPartAttr(EventUserMessageReceived, userRows...) {
		t.Fatalf("USER_MESSAGE_RECEIVED content_parts missing $.function_response in part_attributes across %d rows", len(userRows))
	}
	if !checkFuncRespPartAttr(EventLLMRequest, llmReqRow) {
		t.Fatalf("LLM_REQUEST content_parts missing $.function_response in part_attributes: %#v", llmReqRow[columnContentParts])
	}

	fmtP, fmtReqs, fmtCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.DisableCreateViews = true
		c.Logger = &recordingLogger{}
		c.ContentFormatter = func(any, string) (any, error) {
			return nil, errors.New("boom")
		}
	})
	var fmtErrs callbackErrors
	fmtErrs.add(fmtP.OnUserMessageCallback()(fmtCtx, &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "hi"}},
	}))
	fmtErrs.check(t)
	fmtP.AfterRunCallback()(fmtCtx)
	fmtRows := collectRows(t, fmtReqs, untilEventType(EventInvocationCompleted))
	fmtRow := onlyRowOfType(t, fmtRows, EventUserMessageReceived)
	fmtContent, ok := jsonColumnValue(t, fmtRow, columnContent)
	if !ok {
		t.Fatalf("FORMATTER_FAILED USER_MESSAGE_RECEIVED row missing %s column", columnContent)
	}
	checkDocPath("FORMATTER_FAILED.content", fmtContent, "$")
	if got, ok := fmtContent.(string); !ok || got != "[FORMATTER_FAILED]" {
		t.Errorf("FORMATTER_FAILED content = %#v, want %q", fmtContent, "[FORMATTER_FAILED]")
	}

	pythonOnlySQLPaths := map[string]bool{
		// Emitted by Python plugin variants; eval views coalesce $.static_instruction with instruction fields.
		"$.static_instruction": true,
	}
	for _, sqlPath := range gotSQLPaths {
		if !verifiedGoPaths[sqlPath] && !pythonOnlySQLPaths[sqlPath] {
			t.Errorf("eval SQL JSON path %q extracted from rendered views was not verified against Go-emitted rows", sqlPath)
		}
	}
}

const evalViewSQLiteRunnerScript = `
import json
import re
import sqlite3
import sys

if sqlite3.sqlite_version_info < (3, 25, 0):
    sys.stdout.write(json.dumps({"skip": f"sqlite3 {sqlite3.sqlite_version} < 3.25.0"}))
    sys.exit(0)

JSON_TAG = "\x1fJSON:"
BOOL_COLS = {"has_error", "has_truncated_content", "has_tool_loop", "eval_ready"}


def tag_json(obj):
    return JSON_TAG + json.dumps(obj, separators=(",", ":"), sort_keys=True)


def parse_tagged_or_json(val):
    if val is None:
        return None, False
    if isinstance(val, str):
        if val.startswith(JSON_TAG):
            return json.loads(val[len(JSON_TAG):]), True
        try:
            return json.loads(val), False
        except Exception:
            return val, False
    return val, False


def resolve_json_path(doc, path):
    if doc is None or path is None:
        return None, False
    if path == "$":
        return doc, True
    if not path.startswith("$."):
        return None, False
    cur = doc
    for seg in path[2:].split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None, False
        cur = cur[seg]
    return cur, True


def udf_json_value(raw_doc, path):
    if raw_doc is None or path is None:
        return None
    doc, _ = parse_tagged_or_json(raw_doc)
    val, ok = resolve_json_path(doc, path)
    if not ok or val is None or isinstance(val, (dict, list)):
        return None
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def udf_json_query(raw_doc, path):
    if raw_doc is None or path is None:
        return None
    doc, _ = parse_tagged_or_json(raw_doc)
    val, ok = resolve_json_path(doc, path)
    if not ok:
        return None
    return tag_json(val)


def udf_json_query_array(raw_doc):
    if raw_doc is None:
        return None
    doc, _ = parse_tagged_or_json(raw_doc)
    if not isinstance(doc, list):
        return None
    return json.dumps([tag_json(el) for el in doc])


def udf_json_type(raw_doc):
    if raw_doc is None:
        return None
    doc, is_tagged = parse_tagged_or_json(raw_doc)
    if not is_tagged and isinstance(raw_doc, str):
        try:
            doc = json.loads(raw_doc)
        except Exception:
            return None
    if doc is None:
        return "null"
    if isinstance(doc, bool):
        return "boolean"
    if isinstance(doc, (int, float)):
        return "number"
    if isinstance(doc, str):
        return "string"
    if isinstance(doc, list):
        return "array"
    if isinstance(doc, dict):
        return "object"
    return None


def to_python_for_json(val):
    if val is None:
        return None
    if isinstance(val, str) and val.startswith(JSON_TAG):
        return json.loads(val[len(JSON_TAG):])
    return val


def udf_to_json_string(val):
    if val is None:
        return None
    if isinstance(val, str) and val.startswith(JSON_TAG):
        return val[len(JSON_TAG):]
    return json.dumps(val, separators=(",", ":"), sort_keys=True)


def udf_to_json(val):
    if val is None:
        return tag_json(None)
    if isinstance(val, str) and val.startswith(JSON_TAG):
        return val
    return tag_json(val)


def udf_json_object(*args):
    if len(args) == 0:
        return tag_json({})
    if (
        len(args) == 2
        and isinstance(args[0], str)
        and args[0].startswith(JSON_TAG)
        and isinstance(args[1], str)
        and args[1].startswith(JSON_TAG)
    ):
        keys = json.loads(args[0][len(JSON_TAG):])
        vals = json.loads(args[1][len(JSON_TAG):])
        if isinstance(keys, list) and isinstance(vals, list):
            return tag_json(dict(zip(keys, vals)))
    out = {}
    for i in range(0, len(args), 2):
        k = args[i]
        v = args[i + 1]
        if k is None:
            continue
        out[str(k)] = to_python_for_json(v)
    return tag_json(out)


def udf_json_array(*args):
    return tag_json([to_python_for_json(a) for a in args])


def udf_array_length(val):
    if val is None:
        return None
    if isinstance(val, str):
        s = val[len(JSON_TAG):] if val.startswith(JSON_TAG) else val
        arr = json.loads(s)
        if isinstance(arr, list):
            return len(arr)
    return 0


def udf_regexp_contains(val, pattern):
    if val is None or pattern is None:
        return None
    return 1 if re.search(pattern, str(val)) is not None else 0


def udf_regexp_replace(val, pattern, repl):
    if val is None or pattern is None or repl is None:
        return None
    return re.sub(pattern, str(repl), str(val))


def udf_safe_cast_int64(val):
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def udf_format_timestamp(fmt, ts):
    if ts is None:
        return None
    return str(ts)


def udf_chr(code):
    if code is None:
        return None
    return chr(int(code))


class LogicalOrAgg:
    def __init__(self):
        self.seen = False
        self.val = False

    def step(self, x):
        if x is not None:
            self.seen = True
            if bool(x):
                self.val = True

    def finalize(self):
        return (1 if self.val else 0) if self.seen else None


class LogicalAndAgg:
    def __init__(self):
        self.seen = False
        self.val = True

    def step(self, x):
        if x is not None:
            self.seen = True
            if not bool(x):
                self.val = False

    def finalize(self):
        return (1 if self.val else 0) if self.seen else None


class ArrayConcatAgg:
    def __init__(self):
        self.items = []
        self.seen = False

    def step(self, raw_arr):
        if raw_arr is None:
            return
        self.seen = True
        s = raw_arr[len(JSON_TAG):] if isinstance(raw_arr, str) and raw_arr.startswith(JSON_TAG) else raw_arr
        arr = json.loads(s)
        if isinstance(arr, list):
            self.items.extend(arr)

    def finalize(self):
        return tag_json(self.items) if self.seen else None


def cmp_key(keys_and_dirs):
    class K:
        def __init__(self, kd):
            self.kd = kd

        def _cmp(self, other):
            for (d1, v1), (_, v2) in zip(self.kd, other.kd):
                if v1 == v2:
                    continue
                if v1 is None:
                    res = -1
                elif v2 is None:
                    res = 1
                else:
                    res = -1 if v1 < v2 else 1
                return res * d1
            return 0

        def __lt__(self, other):
            return self._cmp(other) < 0

        def __eq__(self, other):
            return self._cmp(other) == 0
    return K(keys_and_dirs)


class BQArrayAgg:
    def __init__(self):
        self.rows = []
        self.distinct = False
        self.ignore_nulls = False
        self.first_only = False

    def step(self, distinct, ignore_nulls, first_only, val, *dir_and_keys):
        self.distinct = bool(distinct)
        self.ignore_nulls = bool(ignore_nulls)
        self.first_only = bool(first_only)
        if self.ignore_nulls and val is None:
            return
        kd = []
        for i in range(0, len(dir_and_keys), 2):
            kd.append((int(dir_and_keys[i]), dir_and_keys[i + 1]))
        self.rows.append((kd, val))

    def finalize(self):
        if not self.rows:
            return None
        sorted_rows = sorted(self.rows, key=lambda r: cmp_key(r[0]))
        vals = []
        seen = set()
        for _, v in sorted_rows:
            py_v = to_python_for_json(v)
            if self.distinct:
                canon = json.dumps(py_v, sort_keys=True)
                if canon in seen:
                    continue
                seen.add(canon)
            vals.append((v, py_v))
        if not vals:
            return None
        if self.first_only:
            return vals[0][0]
        return tag_json([pv for _, pv in vals])


def normalize_sql_strings(sql):
    out = []
    i = 0
    n = len(sql)
    while i < n:
        if sql[i] in ("r", "R") and i + 1 < n and sql[i + 1] in ("'", '"'):
            quote = sql[i + 1]
            i += 2
            buf = []
            while i < n and sql[i] != quote:
                buf.append(sql[i])
                i += 1
            i += 1
            out.append("'" + "".join(buf).replace("'", "''") + "'")
        elif sql[i] in ("'", '"'):
            quote = sql[i]
            i += 1
            buf = []
            while i < n and sql[i] != quote:
                if sql[i] == "\\" and i + 1 < n:
                    nxt = sql[i + 1]
                    if nxt == "n":
                        buf.append("\n")
                    elif nxt == "r":
                        buf.append("\r")
                    elif nxt == "t":
                        buf.append("\t")
                    else:
                        buf.append(nxt)
                    i += 2
                else:
                    buf.append(sql[i])
                    i += 1
            i += 1
            out.append("'" + "".join(buf).replace("'", "''") + "'")
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


def find_matching_paren(s, open_idx):
    depth = 0
    in_str = False
    i = open_idx
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_str and i + 1 < len(s) and s[i + 1] == "'":
                i += 2
                continue
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError(f"unmatched paren at {open_idx}")


def split_top_level_commas(s):
    parts = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_str and i + 1 < len(s) and s[i + 1] == "'":
                i += 2
                continue
            in_str = not in_str
        elif not in_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(s[start:i].strip())
                start = i + 1
        i += 1
    parts.append(s[start:].strip())
    return parts


def strip_sql_strings(sql):
    out = []
    i = 0
    n = len(sql)
    while i < n:
        if sql[i] == "'":
            i += 1
            while i < n:
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                if sql[i] == "'":
                    i += 1
                    break
                i += 1
            out.append("''")
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


ALLOWED_SQLITE_PAREN_IDENTS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS", "OVER",
    "PARTITION", "WHEN", "THEN", "ELSE", "AS", "ON", "BY",
    "CAST", "COALESCE", "NULLIF", "MIN", "MAX", "SUM", "COUNT", "AVG",
    "LAG", "LEAD", "ROW_NUMBER", "SUBSTR", "CHAR", "GROUP_CONCAT",
    "JSON_EACH", "JSON_EXTRACT",
    "JSON_VALUE", "JSON_QUERY", "JSON_QUERY_ARRAY", "JSON_TYPE",
    "TO_JSON_STRING", "TO_JSON", "JSON_OBJECT", "JSON_ARRAY",
    "ARRAY_LENGTH", "REGEXP_CONTAINS", "REGEXP_REPLACE", "SAFE_CAST_INT64",
    "FORMAT_TIMESTAMP", "CHR", "LOGICAL_OR", "LOGICAL_AND",
    "ARRAY_CONCAT_AGG", "BQ_ARRAY_AGG",
}


def replace_func_calls(sql, func_name, handler):
    pat = re.compile(r"\b" + re.escape(func_name) + r"\s*\(", re.IGNORECASE)
    while True:
        prev = sql
        pos = 0
        while True:
            m = None
            in_str = False
            i = 0
            while i < len(sql):
                if sql[i] == "'":
                    if in_str and i + 1 < len(sql) and sql[i + 1] == "'":
                        i += 2
                        continue
                    in_str = not in_str
                    i += 1
                    continue
                if not in_str and i >= pos:
                    cand = pat.match(sql, i)
                    if cand:
                        m = cand
                        break
                i += 1
            if not m:
                break
            open_idx = m.end() - 1
            close_idx = find_matching_paren(sql, open_idx)
            inner = replace_func_calls(sql[open_idx + 1:close_idx], func_name, handler)
            after = sql[close_idx + 1:]
            repl, consumed_after = handler(inner, after)
            sql = sql[:m.start()] + repl + after[consumed_after:]
            pos = m.start() + len(repl)
        if sql == prev:
            break
    return sql


def transpile_bigquery_to_sqlite(bq_sql):
    sql = re.sub(r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW\s+\x60[^\x60]+\x60\s+AS\s*", "", bq_sql)
    sql = re.sub(r"\x60[^\x60]+\x60", "events_table", sql)
    sql = normalize_sql_strings(sql)

    sql = re.sub(r"ARRAY<STRUCT<[^>]+>>\[\]", "'[]'", sql)
    sql = re.sub(r"ARRAY<STRING>\[\]", f"'{JSON_TAG}[]'", sql)
    sql = re.sub(r"CAST\(NULL\s+AS\s+JSON\)", "NULL", sql)
    sql = replace_func_calls(
        sql,
        "SAFE_CAST",
        lambda inner, _: (
            "SAFE_CAST_INT64(" + re.sub(r"\s+AS\s+INT64\s*$", "", inner, flags=re.IGNORECASE) + ")",
            0,
        ),
    )

    sql = re.sub(
        r"FROM\s+agent_latest,\s*UNNEST\(JSON_QUERY_ARRAY\(tool_declarations\)\)\s+AS\s+declaration\s+WITH\s+OFFSET\s+AS\s+declaration_offset",
        "FROM (SELECT agent_latest.*, json_each.value AS declaration, json_each.key AS declaration_offset FROM agent_latest, json_each(COALESCE(JSON_QUERY_ARRAY(tool_declarations), '[]')))",
        sql,
    )

    sql = re.sub(
        r"ARRAY\(\s*SELECT\s+DISTINCT\s+tool\s+FROM\s+UNNEST\(s\.tools_used_by_turn\)\s+AS\s+tool\s+ORDER\s+BY\s+tool\s*\)",
        "(SELECT COALESCE(BQ_ARRAY_AGG(1, 1, 0, value, 1, value), '\x1fJSON:[]') FROM json_each(CASE WHEN s.tools_used_by_turn IS NULL THEN '[]' WHEN substr(s.tools_used_by_turn, 1, 6) = char(31) || 'JSON:' THEN substr(s.tools_used_by_turn, 7) ELSE s.tools_used_by_turn END))",
        sql,
    )

    agg_pat = re.compile(
        r"\(\s*SELECT\s+STRING_AGG\(\s*p\.text\s*,\s*('(?:''|[^'])*')\s+ORDER\s+BY\s+p\.part_index\s*\)\s+FROM\s+UNNEST\(content_parts\)\s+AS\s+p\s+WHERE\s+",
        re.DOTALL,
    )
    while True:
        m = agg_pat.search(sql)
        if not m:
            break
        open_idx = m.start()
        close_idx = find_matching_paren(sql, open_idx)
        sep = m.group(1)
        where_body = sql[m.end():close_idx]
        where_body = re.sub(r"\bp\.(part_index|mime_type|text|part_attributes|storage_mode)\b", r"json_extract(p.value, '$.\1')", where_body)
        repl = (
            f"(SELECT group_concat(_txt, {sep}) FROM ("
            f"SELECT json_extract(p.value, '$.text') AS _txt "
            f"FROM json_each(COALESCE(content_parts, '[]')) AS p "
            f"WHERE {where_body} "
            f"ORDER BY CAST(json_extract(p.value, '$.part_index') AS INTEGER) LIMIT -1"
            f"))"
        )
        sql = sql[:open_idx] + repl + sql[close_idx + 1:]

    ex_pat = re.compile(
        r"EXISTS\(\s*SELECT\s+1\s+FROM\s+UNNEST\(content_parts\)\s+AS\s+p\s+WHERE\s+",
        re.DOTALL,
    )
    while True:
        m = ex_pat.search(sql)
        if not m:
            break
        open_idx = sql.find("(", m.start())
        close_idx = find_matching_paren(sql, open_idx)
        where_body = sql[m.end():close_idx]
        where_body = re.sub(r"\bp\.(part_index|mime_type|text|part_attributes|storage_mode)\b", r"json_extract(p.value, '$.\1')", where_body)
        repl = f"EXISTS(SELECT 1 FROM json_each(COALESCE(content_parts, '[]')) AS p WHERE {where_body})"
        sql = sql[:m.start()] + repl + sql[close_idx + 1:]

    def rewrite_if(inner, _):
        parts = split_top_level_commas(inner)
        if len(parts) != 3:
            raise ValueError(f"expected 3 arguments to IF(), got {len(parts)}: {inner}")
        return f"CASE WHEN ({parts[0]}) THEN ({parts[1]}) ELSE ({parts[2]}) END", 0

    sql = replace_func_calls(sql, "COUNTIF", lambda inner, _: (f"SUM(CASE WHEN ({inner}) THEN 1 ELSE 0 END)", 0))
    sql = replace_func_calls(sql, "IF", rewrite_if)
    sql = re.sub(r"\bIS\s+NOT\s+DISTINCT\s+FROM\b", "IS", sql, flags=re.IGNORECASE)

    def rewrite_array_agg(inner, after):
        s = inner.strip()
        distinct = 0
        if re.match(r"^DISTINCT\b", s, re.IGNORECASE):
            distinct = 1
            s = re.sub(r"^DISTINCT\s+", "", s, count=1, flags=re.IGNORECASE)
        limit_1 = 0
        if re.search(r"\s+LIMIT\s+1\s*$", s, re.IGNORECASE):
            limit_1 = 1
            s = re.sub(r"\s+LIMIT\s+1\s*$", "", s, count=1, flags=re.IGNORECASE)
        order_exprs = []
        depth = 0
        in_str = False
        order_pos = -1
        i = 0
        while i < len(s):
            if s[i] == "'":
                if in_str and i + 1 < len(s) and s[i + 1] == "'":
                    i += 2
                    continue
                in_str = not in_str
            elif not in_str:
                if s[i] == "(":
                    depth += 1
                elif s[i] == ")":
                    depth -= 1
                elif depth == 0 and re.match(r"\bORDER\s+BY\b", s[i:], re.IGNORECASE):
                    order_pos = i
                    break
            i += 1
        if order_pos >= 0:
            order_str = re.sub(r"^ORDER\s+BY\s+", "", s[order_pos:], count=1, flags=re.IGNORECASE)
            s = s[:order_pos].strip()
            for item in split_top_level_commas(order_str):
                m_dir = re.search(r"\s+(ASC|DESC)\s*$", item, re.IGNORECASE)
                direction = 1
                if m_dir:
                    if m_dir.group(1).upper() == "DESC":
                        direction = -1
                    item = item[:m_dir.start()].strip()
                order_exprs.append((direction, item))

        ignore_nulls = 0
        if re.search(r"\s+IGNORE\s+NULLS\s*$", s, re.IGNORECASE):
            ignore_nulls = 1
            s = re.sub(r"\s+IGNORE\s+NULLS\s*$", "", s, count=1, flags=re.IGNORECASE)

        m_off = re.match(r"^\s*\[SAFE_OFFSET\(0\)\]", after)
        consumed = m_off.end() if m_off else 0
        first_only = 1 if (m_off or limit_1) else 0

        args = [str(distinct), str(ignore_nulls), str(first_only), s]
        for d, expr in order_exprs:
            args.append(str(d))
            args.append(expr)
        return f"BQ_ARRAY_AGG({', '.join(args)})", consumed

    sql = replace_func_calls(sql, "ARRAY_AGG", rewrite_array_agg)

    qual_pat = re.compile(
        r"SELECT\s+(?P<cols>(?:(?!\bSELECT\b).)+?)\s+FROM\s+(?P<src>[a-zA-Z0-9_]+)\s+QUALIFY\s+(?P<qual>(?:(?!\n\s*\)\s*,).)+)",
        re.DOTALL,
    )
    sql = qual_pat.sub(
        lambda m: (
            f"SELECT {m.group('cols')} FROM ("
            f"SELECT *, CASE WHEN ({m.group('qual').strip()}) THEN 1 ELSE 0 END AS _qualify_keep "
            f"FROM {m.group('src')}"
            f") WHERE _qualify_keep = 1"
        ),
        sql,
    )

    stripped_sql = strip_sql_strings(sql)
    assert re.search(
        r"\b(IF|COUNTIF|ARRAY_AGG|QUALIFY|SAFE_CAST|SAFE_OFFSET)\s*\(",
        stripped_sql,
        re.IGNORECASE,
    ) is None, f"untranspiled GoogleSQL call in SQLite SQL: {stripped_sql}"
    assert re.search(
        r"\b(IF|IIF|COUNTIF|ARRAY_AGG|QUALIFY|SAFE_CAST|SAFE_OFFSET|STRING_AGG|UNNEST)\b|\bIS\s+NOT\s+DISTINCT\s+FROM\b",
        stripped_sql,
        re.IGNORECASE,
    ) is None, f"untranspiled GoogleSQL or SQLite>=3.32 construct in SQLite SQL: {stripped_sql}"
    cte_names = {
        m.group(1).upper()
        for m in re.finditer(r"(?:\bWITH\b|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(", stripped_sql, re.IGNORECASE)
    }
    for m_id in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", stripped_sql):
        ident = m_id.group(1).upper()
        assert ident in ALLOWED_SQLITE_PAREN_IDENTS or ident in cte_names, (
            f"unallowed or version-dependent SQLite function {ident}() in transpiled SQL"
        )
    return sql


payload = json.load(sys.stdin)
conn = sqlite3.connect(":memory:")
conn.create_function("JSON_VALUE", 2, udf_json_value)
conn.create_function("JSON_QUERY", 2, udf_json_query)
conn.create_function("JSON_QUERY_ARRAY", 1, udf_json_query_array)
conn.create_function("JSON_TYPE", 1, udf_json_type)
conn.create_function("TO_JSON_STRING", 1, udf_to_json_string)
conn.create_function("TO_JSON", 1, udf_to_json)
conn.create_function("JSON_OBJECT", -1, udf_json_object)
conn.create_function("JSON_ARRAY", -1, udf_json_array)
conn.create_function("ARRAY_LENGTH", 1, udf_array_length)
conn.create_function("REGEXP_CONTAINS", 2, udf_regexp_contains)
conn.create_function("REGEXP_REPLACE", 3, udf_regexp_replace)
conn.create_function("SAFE_CAST_INT64", 1, udf_safe_cast_int64)
conn.create_function("FORMAT_TIMESTAMP", 2, udf_format_timestamp)
conn.create_function("CHR", 1, udf_chr)
conn.create_aggregate("LOGICAL_OR", 1, LogicalOrAgg)
conn.create_aggregate("LOGICAL_AND", 1, LogicalAndAgg)
conn.create_aggregate("ARRAY_CONCAT_AGG", 1, ArrayConcatAgg)
conn.create_aggregate("BQ_ARRAY_AGG", -1, BQArrayAgg)

conn.execute(
    """
    CREATE TABLE events_table (
        timestamp TEXT,
        event_id TEXT,
        event_type TEXT,
        agent TEXT,
        session_id TEXT,
        invocation_id TEXT,
        user_id TEXT,
        content TEXT,
        content_parts TEXT,
        attributes TEXT,
        latency_ms TEXT,
        status TEXT,
        error_message TEXT,
        is_truncated INTEGER
    )
    """
)

for r in payload["rows"]:
    conn.execute(
        """
        INSERT INTO events_table (
            timestamp, event_id, event_type, agent, session_id, invocation_id,
            user_id, content, content_parts, attributes, latency_ms,
            status, error_message, is_truncated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            r.get("timestamp"),
            r.get("event_id"),
            r.get("event_type"),
            r.get("agent"),
            r.get("session_id"),
            r.get("invocation_id"),
            r.get("user_id"),
            (JSON_TAG + r["content"]) if r.get("content") is not None else None,
            json.dumps(r["content_parts"]) if r.get("content_parts") is not None else None,
            (JSON_TAG + r["attributes"]) if r.get("attributes") is not None else None,
            (JSON_TAG + r["latency_ms"]) if r.get("latency_ms") is not None else None,
            r.get("status"),
            r.get("error_message"),
            (1 if r["is_truncated"] else 0) if r.get("is_truncated") is not None else None,
        ),
    )

sqlite_sql = transpile_bigquery_to_sqlite(payload["query"])
cur = conn.execute(sqlite_sql)
cols = [d[0] for d in cur.description]
out_rows = []
for row in cur.fetchall():
    d = {}
    for col, v in zip(cols, row):
        if isinstance(v, str) and v.startswith(JSON_TAG):
            d[col] = json.loads(v[len(JSON_TAG):])
        elif col in BOOL_COLS and v is not None:
            d[col] = bool(v)
        else:
            d[col] = v
    out_rows.append(d)
sys.stdout.write(json.dumps({"rows": out_rows}))
`

type evalViewExecResultRow struct {
	SessionID                        string         `json:"session_id"`
	InvocationID                     string         `json:"invocation_id"`
	TurnIndex                        int64          `json:"turn_index"`
	UserID                           *string        `json:"user_id"`
	RootAgent                        string         `json:"root_agent"`
	TurnCount                        int64          `json:"turn_count"`
	ToolCallCount                    int64          `json:"tool_call_count"`
	ToolErrorCount                   int64          `json:"tool_error_count"`
	LLMErrorCount                    int64          `json:"llm_error_count"`
	AgentErrorCount                  int64          `json:"agent_error_count"`
	HasError                         bool           `json:"has_error"`
	HasTruncatedContent              bool           `json:"has_truncated_content"`
	HasToolLoop                      bool           `json:"has_tool_loop"`
	MaxConsecutiveIdenticalToolCalls int64          `json:"max_consecutive_identical_tool_calls"`
	DistinctToolsUsed                []string       `json:"distinct_tools_used"`
	TotalPromptTokens                int64          `json:"total_prompt_tokens"`
	TotalCompletionTokens            int64          `json:"total_completion_tokens"`
	TotalTokens                      int64          `json:"total_tokens"`
	TotalLatencyMS                   int64          `json:"total_latency_ms"`
	TurnLatencyMS                    int64          `json:"turn_latency_ms"`
	EvalReady                        bool           `json:"eval_ready"`
	Trace                            map[string]any `json:"trace"`
}

func serializeArrowDecodedRowsForSQLite(rows []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(rows))
	for _, r := range rows {
		cp := make(map[string]any, len(r))
		for k, v := range r {
			if ts, ok := v.(time.Time); ok {
				cp[k] = ts.UTC().Format("2006-01-02T15:04:05.000000Z")
			} else {
				cp[k] = v
			}
		}
		out = append(out, cp)
	}
	return out
}

func cloneEventRow(r map[string]any) map[string]any {
	cp := make(map[string]any, len(r))
	for k, v := range r {
		if parts, ok := v.([]map[string]any); ok {
			partsCopy := make([]map[string]any, len(parts))
			for i, p := range parts {
				pm := make(map[string]any, len(p))
				for pk, pv := range p {
					pm[pk] = pv
				}
				partsCopy[i] = pm
			}
			cp[k] = partsCopy
		} else {
			cp[k] = v
		}
	}
	return cp
}

func executeEvalViewSQL(t *testing.T, query string, rows []map[string]any) []evalViewExecResultRow {
	t.Helper()
	pythonBin, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not found in PATH")
	}
	payloadBytes, err := json.Marshal(map[string]any{
		"query": query,
		"rows":  serializeArrowDecodedRowsForSQLite(rows),
	})
	if err != nil {
		t.Fatalf("json.Marshal(eval payload) error = %v", err)
	}
	cmd := exec.Command(pythonBin, "-c", evalViewSQLiteRunnerScript)
	cmd.Stdin = bytes.NewReader(payloadBytes)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("sqlite3 eval view execution failed: %v\nstderr:\n%s", err, stderr.String())
	}
	var resp struct {
		Skip string                  `json:"skip"`
		Rows []evalViewExecResultRow `json:"rows"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatalf("json.Unmarshal(sqlite3 output) error = %v\nraw:\n%s", err, stdout.String())
	}
	if resp.Skip != "" {
		t.Skip(resp.Skip)
	}
	return resp.Rows
}

type syntheticTurnSpec struct {
	appName          string
	userID           string
	sessionID        string
	invocationID     string
	agentName        string
	agentDesc        string
	instruction      string
	toolDecl         *declaredTool
	toolCallArgs     map[string]any
	toolCallResult   map[string]any
	pauseToolID      string
	userMsg          *genai.Content
	responseParts    []*genai.Part
	promptTokens     int32
	completionTokens int32
	totalTokens      int32
	pluginModifier   func(*Config)
}

func emitCallbackTurnRows(t *testing.T, spec syntheticTurnSpec) []map[string]any {
	t.Helper()
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.DisableCreateViews = true
		if spec.pluginModifier != nil {
			spec.pluginModifier(c)
		}
	})
	mCtx.appName = spec.appName
	mCtx.userID = spec.userID
	mCtx.sessionID = spec.sessionID
	mCtx.invocationID = spec.invocationID
	mCtx.agentName = spec.agentName
	mCtx.functionCallID = "fc-" + spec.invocationID

	rootAg, err := llmagent.New(llmagent.Config{
		Name:        spec.agentName,
		Description: spec.agentDesc,
		Model:       &scriptedLLM{},
		Instruction: spec.instruction,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%s) error = %v", spec.agentName, err)
	}
	invCtx := &evalAgentInvocationContext{
		mockInvocationContext: mCtx,
		ag:                    rootAg,
	}

	var errs callbackErrors
	if spec.userMsg != nil {
		errs.add(p.OnUserMessageCallback()(invCtx, spec.userMsg))
	}
	errs.add(p.BeforeRunCallback()(invCtx))
	errs.add(p.BeforeAgentCallback()(invCtx))

	reqTools := map[string]any{}
	if spec.toolDecl != nil {
		reqTools[spec.toolDecl.Name()] = spec.toolDecl
	}
	llmReq := &model.LLMRequest{
		Model: "test-model-v1",
		Tools: reqTools,
	}
	if spec.userMsg != nil {
		llmReq.Contents = []*genai.Content{spec.userMsg}
	}
	errs.add(p.BeforeModelCallback()(invCtx, llmReq))

	if spec.pauseToolID != "" && spec.toolDecl != nil {
		pausePart := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   spec.pauseToolID,
				Name: spec.toolDecl.Name(),
				Args: spec.toolCallArgs,
			},
		}
		pauseResp := &model.LLMResponse{
			FinishReason: genai.FinishReasonStop,
			Content:      &genai.Content{Role: "model", Parts: []*genai.Part{pausePart}},
		}
		errs.add(p.AfterModelCallback()(invCtx, pauseResp, nil))
		errs.add(p.BeforeToolCallback()(invCtx, spec.toolDecl, spec.toolCallArgs))
		errs.add(p.OnEventCallback()(invCtx, &session.Event{
			ID:                 "ev-pause-" + spec.invocationID,
			Author:             spec.agentName,
			LLMResponse:        *pauseResp,
			LongRunningToolIDs: []string{spec.pauseToolID},
		}))
	} else {
		if spec.toolDecl != nil && spec.toolCallArgs != nil {
			errs.add(p.BeforeToolCallback()(invCtx, spec.toolDecl, spec.toolCallArgs))
			errs.add(p.AfterToolCallback()(invCtx, spec.toolDecl, spec.toolCallArgs, spec.toolCallResult, nil))
		}
		var respContent *genai.Content
		if len(spec.responseParts) > 0 {
			respContent = &genai.Content{
				Role:  "model",
				Parts: spec.responseParts,
			}
		}
		llmResp := &model.LLMResponse{
			FinishReason: genai.FinishReasonStop,
			Content:      respContent,
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     spec.promptTokens,
				CandidatesTokenCount: spec.completionTokens,
				TotalTokenCount:      spec.totalTokens,
			},
		}
		errs.add(p.AfterModelCallback()(invCtx, llmResp, nil))
		if respContent != nil {
			errs.add(p.OnEventCallback()(invCtx, &session.Event{
				ID:          "ev-final-" + spec.invocationID,
				Author:      spec.agentName,
				LLMResponse: *llmResp,
			}))
		}
	}

	errs.add(p.AfterAgentCallback()(invCtx))
	errs.check(t)
	p.AfterRunCallback()(invCtx)
	return collectRows(t, requests, untilEventType(EventInvocationCompleted))
}

func extractModelResponseTextsFromTrace(t *testing.T, trace map[string]any) []string {
	t.Helper()
	turnsAny, ok := trace["turns"].([]any)
	if !ok {
		t.Fatalf("trace.turns = %#v, want []any", trace["turns"])
	}
	var texts []string
	for _, turnAny := range turnsAny {
		tm, _ := turnAny.(map[string]any)
		eventsAny, _ := tm["events"].([]any)
		for _, evAny := range eventsAny {
			em, _ := evAny.(map[string]any)
			cm, _ := em["content"].(map[string]any)
			if cm["role"] != "model" {
				continue
			}
			partsAny, _ := cm["parts"].([]any)
			for _, pAny := range partsAny {
				pm, _ := pAny.(map[string]any)
				if txt, hasTxt := pm["text"].(string); hasTxt {
					texts = append(texts, txt)
				}
			}
		}
	}
	return texts
}

func TestEvalViews_SemanticExecution(t *testing.T) {
	defaultDefs, err := (Config{ProjectID: "p", DatasetID: "d", TableName: "t"}).EvalViewDefinitions()
	if err != nil {
		t.Fatalf("EvalViewDefinitions() error = %v", err)
	}
	sessionSQL := defaultDefs[0].Query
	turnSQL := defaultDefs[1].Query

	t.Run("cross_user_and_cross_app_session_isolation_and_pause_windows", func(t *testing.T) {
		alphaTool := &declaredTool{
			fakeTool: fakeTool{name: "alpha_gate", description: "Alpha approval gate", longRunning: true},
			decl: &genai.FunctionDeclaration{
				Name:                 "alpha_gate",
				Description:          "Alpha approval gate",
				ParametersJsonSchema: map[string]any{"type": "object"},
			},
		}
		bobTool := &declaredTool{
			fakeTool: fakeTool{name: "bob_lookup", description: "Bob lookup"},
			decl: &genai.FunctionDeclaration{
				Name:        "bob_lookup",
				Description: "Bob lookup",
			},
		}
		betaTool := &declaredTool{
			fakeTool: fakeTool{name: "beta_search", description: "Beta search"},
			decl: &genai.FunctionDeclaration{
				Name:                 "beta_search",
				Description:          "Beta search",
				ParametersJsonSchema: map[string]any{"type": "object"},
			},
		}

		// Session 1: (app-alpha, alice, shared-sess) Turn 0 pauses for HITL/long-running tool.
		aliceTurn0 := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:      "app-alpha",
			userID:       "alice",
			sessionID:    "shared-sess",
			invocationID: "inv-10",
			agentName:    "alpha-agent",
			agentDesc:    "Alpha agent",
			instruction:  "Use alpha_gate.",
			toolDecl:     alphaTool,
			toolCallArgs: map[string]any{"ticket": "T-1"},
			pauseToolID:  "call-pause-1",
			userMsg: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "deploy service"}},
			},
		})
		time.Sleep(2 * time.Millisecond)

		// Session 2: (app-alpha, bob, shared-sess) arrives chronologically BETWEEN Alice's paused turn
		// and Alice's resumed turn, sharing both app_name and session_id.
		bobTurn0 := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:          "app-alpha",
			userID:           "bob",
			sessionID:        "shared-sess",
			invocationID:     "inv-20",
			agentName:        "bob-agent",
			agentDesc:        "Bob agent",
			instruction:      "Answer Bob.",
			toolDecl:         bobTool,
			toolCallArgs:     map[string]any{"q": "status"},
			toolCallResult:   map[string]any{"ok": true},
			userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "status check"}}},
			responseParts:    []*genai.Part{{Text: "Bob status OK"}},
			promptTokens:     10,
			completionTokens: 5,
			totalTokens:      15,
		})
		time.Sleep(2 * time.Millisecond)

		// Session 1 resumed: (app-alpha, alice, shared-sess) Turn 1 supplies FunctionResponse and completes.
		aliceTurn1 := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:      "app-alpha",
			userID:       "alice",
			sessionID:    "shared-sess",
			invocationID: "inv-30",
			agentName:    "alpha-agent",
			agentDesc:    "Alpha agent",
			instruction:  "Use alpha_gate.",
			toolDecl:     alphaTool,
			userMsg: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "call-pause-1",
						Name:     "alpha_gate",
						Response: map[string]any{"approved": true},
					},
				}},
			},
			responseParts:    []*genai.Part{{Text: "Deployment completed"}},
			promptTokens:     20,
			completionTokens: 8,
			totalTokens:      28,
		})
		time.Sleep(2 * time.Millisecond)

		// Session 3: (app-beta, alice, shared-sess) shares BOTH user_id="alice" and session_id="shared-sess"
		// under a second ADK application, and includes one row with adk.app_name omitted to verify
		// invocation-scoped app_name backfill.
		betaTurn0 := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:          "app-beta",
			userID:           "alice",
			sessionID:        "shared-sess",
			invocationID:     "inv-40",
			agentName:        "beta-agent",
			agentDesc:        "Beta agent",
			instruction:      "Search beta.",
			toolDecl:         betaTool,
			toolCallArgs:     map[string]any{"k": "v"},
			toolCallResult:   map[string]any{"found": 1},
			userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "find beta"}}},
			responseParts:    []*genai.Part{{Text: "Beta result ready"}},
			promptTokens:     7,
			completionTokens: 3,
			totalTokens:      10,
		})
		strippedOne := false
		for i := range betaTurn0 {
			if betaTurn0[i]["event_type"] == EventAgentResponse {
				attrsDoc := jsonObjectColumn(t, betaTurn0[i], columnAttributes)
				if adkMap, ok := attrsDoc["adk"].(map[string]any); ok {
					delete(adkMap, "app_name")
					b, _ := json.Marshal(attrsDoc)
					betaTurn0[i][columnAttributes] = string(b)
					betaTurn0[i]["user_id"] = ""
					strippedOne = true
				}
			}
		}
		if !strippedOne {
			t.Fatal("expected to strip adk.app_name from one row in betaTurn0")
		}

		allRows := slices.Concat(aliceTurn0, bobTurn0, aliceTurn1, betaTurn0)
		sessions := executeEvalViewSQL(t, sessionSQL, allRows)
		if len(sessions) != 3 {
			t.Fatalf("len(v_eval_session rows) = %d, want 3 distinct sessions; got %+v", len(sessions), sessions)
		}

		byRoot := make(map[string]evalViewExecResultRow, len(sessions))
		for _, s := range sessions {
			byRoot[s.RootAgent] = s
		}

		alphaSess, ok := byRoot["alpha-agent"]
		if !ok {
			t.Fatalf("missing alpha-agent session in %+v", sessions)
		}
		if alphaSess.UserID == nil || *alphaSess.UserID != "alice" || alphaSess.TurnCount != 2 || !alphaSess.EvalReady || alphaSess.TotalTokens != 28 {
			t.Errorf("alpha-agent session = %+v, want user_id=alice, turn_count=2, eval_ready=true, total_tokens=28", alphaSess)
		}
		alphaAgents, _ := alphaSess.Trace["agents"].(map[string]any)
		if len(alphaAgents) != 1 || alphaAgents["alpha-agent"] == nil {
			t.Errorf("alpha-agent trace.agents = %#v, want only alpha-agent", alphaAgents)
		}

		bobSess, ok := byRoot["bob-agent"]
		if !ok {
			t.Fatalf("missing bob-agent session in %+v", sessions)
		}
		if bobSess.UserID == nil || *bobSess.UserID != "bob" || bobSess.TurnCount != 1 || !bobSess.EvalReady || bobSess.TotalTokens != 15 {
			t.Errorf("bob-agent session = %+v, want user_id=bob, turn_count=1, eval_ready=true, total_tokens=15", bobSess)
		}

		betaSess, ok := byRoot["beta-agent"]
		if !ok {
			t.Fatalf("missing beta-agent session in %+v", sessions)
		}
		if betaSess.UserID == nil || *betaSess.UserID != "alice" || betaSess.TurnCount != 1 || !betaSess.EvalReady || betaSess.TotalTokens != 10 {
			t.Errorf("beta-agent session = %+v, want user_id=alice, turn_count=1, eval_ready=true, total_tokens=10", betaSess)
		}
		betaAgents, _ := betaSess.Trace["agents"].(map[string]any)
		if len(betaAgents) != 1 || betaAgents["beta-agent"] == nil {
			t.Errorf("beta-agent trace.agents = %#v, want only beta-agent", betaAgents)
		}

		turnRows := executeEvalViewSQL(t, turnSQL, allRows)
		if len(turnRows) != 4 {
			t.Fatalf("len(v_eval_turn rows) = %d, want 4", len(turnRows))
		}
		gotTurnIdx := make(map[string]int64, len(turnRows))
		for _, tr := range turnRows {
			gotTurnIdx[tr.InvocationID] = tr.TurnIndex
		}
		wantTurnIdx := map[string]int64{
			"inv-10": 0,
			"inv-30": 1,
			"inv-20": 0,
			"inv-40": 0,
		}
		if diff := cmp.Diff(wantTurnIdx, gotTurnIdx); diff != "" {
			t.Errorf("v_eval_turn turn_index mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("null_app_name_and_null_user_id_join_preserves_agent_configs", func(t *testing.T) {
		calcTool := &declaredTool{
			fakeTool: fakeTool{name: "calc_sum", description: "Adds numbers"},
			decl: &genai.FunctionDeclaration{
				Name:                 "calc_sum",
				Description:          "Adds numbers",
				ParametersJsonSchema: map[string]any{"type": "object"},
			},
		}
		rows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:          "",
			userID:           "",
			sessionID:        "sess-null-scope",
			invocationID:     "inv-null-1",
			agentName:        "calc-agent",
			agentDesc:        "Calculator",
			instruction:      "Compute sums.",
			toolDecl:         calcTool,
			toolCallArgs:     map[string]any{"a": 1, "b": 2},
			toolCallResult:   map[string]any{"sum": 3},
			userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "1+2"}}},
			responseParts:    []*genai.Part{{Text: "3"}},
			promptTokens:     5,
			completionTokens: 2,
			totalTokens:      7,
		})
		for i := range rows {
			delete(rows[i], "user_id") // SQL NULL user_id alongside NULL $.adk.app_name
		}

		sessRows := executeEvalViewSQL(t, sessionSQL, rows)
		if len(sessRows) != 1 {
			t.Fatalf("len(v_eval_session) = %d, want 1", len(sessRows))
		}
		if sessRows[0].UserID != nil {
			t.Errorf("UserID = %v, want nil", *sessRows[0].UserID)
		}
		agentsMap, _ := sessRows[0].Trace["agents"].(map[string]any)
		calcCfg, ok := agentsMap["calc-agent"].(map[string]any)
		if !ok {
			t.Fatalf("trace.agents missing calc-agent when user_id and app_name are NULL: %#v", sessRows[0].Trace["agents"])
		}
		toolsArr, _ := calcCfg["tools"].([]any)
		if len(toolsArr) != 1 {
			t.Fatalf("calc-agent tools = %#v, want 1 tool declaration group (proving NULL-safe agent_configs join)", toolsArr)
		}

		// Also verify PayloadColumnDenylist=["attributes"] where attributes is projected as CAST(NULL AS JSON).
		denyAttrsDefs, err := (Config{
			ProjectID:             "p",
			DatasetID:             "d",
			TableName:             "t",
			PayloadColumnDenylist: []string{columnAttributes},
		}).EvalViewDefinitions()
		if err != nil {
			t.Fatalf("EvalViewDefinitions(deny attributes) error = %v", err)
		}
		denySessRows := executeEvalViewSQL(t, denyAttrsDefs[0].Query, rows)
		if len(denySessRows) != 1 {
			t.Fatalf("len(v_eval_session with attributes denied) = %d, want 1", len(denySessRows))
		}
		denyAgents, _ := denySessRows[0].Trace["agents"].(map[string]any)
		if _, ok := denyAgents["calc-agent"]; !ok {
			t.Errorf("attributes-denied v_eval_session dropped trace.agents: %#v", denyAgents)
		}
		denyTurnRows := executeEvalViewSQL(t, denyAttrsDefs[1].Query, rows)
		if len(denyTurnRows) != 1 {
			t.Fatalf("len(v_eval_turn with attributes denied) = %d, want 1", len(denyTurnRows))
		}
		turnAgents, _ := denyTurnRows[0].Trace["agents"].(map[string]any)
		if _, ok := turnAgents["calc-agent"]; !ok {
			t.Errorf("attributes-denied v_eval_turn dropped trace.agents: %#v", turnAgents)
		}
	})

	t.Run("deduplicates_retried_event_id_and_preserves_empty_event_id_and_prefers_clean_copy", func(t *testing.T) {
		lookupTool := &declaredTool{
			fakeTool: fakeTool{name: "lookup_item", description: "Looks up an item"},
			decl:     &genai.FunctionDeclaration{Name: "lookup_item", Description: "Looks up an item"},
		}
		baseRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:          "shop-app",
			userID:           "u-retry",
			sessionID:        "sess-retry",
			invocationID:     "inv-retry-1",
			agentName:        "shop-agent",
			agentDesc:        "Shop helper",
			instruction:      "Look up items once.",
			toolDecl:         lookupTool,
			toolCallArgs:     map[string]any{"sku": "SKU-9"},
			toolCallResult:   map[string]any{"price": 42},
			userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "price of SKU-9"}}},
			responseParts:    []*genai.Part{{Text: "SKU-9 costs $42"}},
			promptTokens:     15,
			completionTokens: 9,
			totalTokens:      24,
		})

		var withRetries []map[string]any
		for _, r := range baseRows {
			switch r["event_type"] {
			case EventToolStarting:
				// 3 copies with identical event_id (which previously triggered has_tool_loop=true and tool_call_count=3).
				badFmt := cloneEventRow(r)
				badFmt[columnContent] = strconv.Quote(formattingFailedPlaceholder)
				withRetries = append(withRetries, badFmt, cloneEventRow(r), cloneEventRow(r))
			case EventLLMResponse:
				truncCopy := cloneEventRow(r)
				truncCopy["is_truncated"] = true
				withRetries = append(withRetries, truncCopy, cloneEventRow(r), cloneEventRow(r))
			case EventAgentResponse:
				failedUploadCopy := cloneEventRow(r)
				failedUploadCopy[columnContentParts] = []map[string]any{{
					"part_index":   int64(0),
					"mime_type":    "text/plain",
					"storage_mode": "INLINE",
					"text":         uploadFailedText,
				}}
				failedUploadCopy["is_truncated"] = true
				failedFmtCopy := cloneEventRow(r)
				failedFmtCopy[columnContent] = strconv.Quote(formattingFailedPlaceholder)
				delete(failedFmtCopy, columnContentParts)
				withRetries = append(withRetries, failedFmtCopy, failedUploadCopy, cloneEventRow(r))
			default:
				withRetries = append(withRetries, cloneEventRow(r))
			}
		}

		// Second turn in another session where 3 distinct TOOL_STARTING rows have empty/NULL event_id;
		// all 3 must be preserved rather than collapsed into 1.
		emptyIDRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:          "shop-app",
			userID:           "u-empty-id",
			sessionID:        "sess-empty-id",
			invocationID:     "inv-empty-1",
			agentName:        "shop-agent",
			agentDesc:        "Shop helper",
			instruction:      "Look up items.",
			toolDecl:         lookupTool,
			toolCallArgs:     map[string]any{"sku": "A"},
			toolCallResult:   map[string]any{"price": 10},
			userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "check three SKUs"}}},
			responseParts:    []*genai.Part{{Text: "Checked all three"}},
			promptTokens:     10,
			completionTokens: 4,
			totalTokens:      14,
		})
		var toolTemplate map[string]any
		for i := range emptyIDRows {
			if emptyIDRows[i]["event_type"] == EventToolStarting {
				emptyIDRows[i]["event_id"] = ""
				toolTemplate = emptyIDRows[i]
				break
			}
		}
		if toolTemplate == nil {
			t.Fatal("missing TOOL_STARTING template row")
		}
		extraToolNullID := cloneEventRow(toolTemplate)
		delete(extraToolNullID, "event_id")
		extraToolNullID[columnContent] = `{"args":{"sku":"B"},"tool":"lookup_item","tool_origin":"LOCAL"}`
		extraToolEmptyID := cloneEventRow(toolTemplate)
		extraToolEmptyID["event_id"] = ""
		extraToolEmptyID[columnContent] = `{"args":{"sku":"C"},"tool":"lookup_item","tool_origin":"LOCAL"}`
		emptyIDRows = append(emptyIDRows, extraToolNullID, extraToolEmptyID)

		turns := executeEvalViewSQL(t, turnSQL, slices.Concat(withRetries, emptyIDRows))
		if len(turns) != 2 {
			t.Fatalf("len(v_eval_turn) = %d, want 2", len(turns))
		}
		byInv := map[string]evalViewExecResultRow{}
		for _, tr := range turns {
			byInv[tr.InvocationID] = tr
		}

		retryTurn := byInv["inv-retry-1"]
		if retryTurn.ToolCallCount != 1 || retryTurn.MaxConsecutiveIdenticalToolCalls != 1 || retryTurn.HasToolLoop {
			t.Errorf("replayed turn tool metrics = (count=%d, max_consec=%d, loop=%v), want (1, 1, false)",
				retryTurn.ToolCallCount, retryTurn.MaxConsecutiveIdenticalToolCalls, retryTurn.HasToolLoop)
		}
		if retryTurn.TotalPromptTokens != 15 || retryTurn.TotalCompletionTokens != 9 || retryTurn.TotalTokens != 24 {
			t.Errorf("replayed turn tokens = (%d, %d, %d), want (15, 9, 24)",
				retryTurn.TotalPromptTokens, retryTurn.TotalCompletionTokens, retryTurn.TotalTokens)
		}
		if retryTurn.HasTruncatedContent || !retryTurn.EvalReady {
			t.Errorf("replayed turn (HasTruncatedContent=%v, EvalReady=%v), want (false, true)",
				retryTurn.HasTruncatedContent, retryTurn.EvalReady)
		}
		if gotTexts := extractModelResponseTextsFromTrace(t, retryTurn.Trace); !cmp.Equal(gotTexts, []string{"SKU-9 costs $42"}) {
			t.Errorf("replayed turn model texts = %#v, want [\"SKU-9 costs $42\"]", gotTexts)
		}

		emptyTurn := byInv["inv-empty-1"]
		if emptyTurn.ToolCallCount != 3 || emptyTurn.HasToolLoop {
			t.Errorf("empty-event_id turn = (tool_call_count=%d, has_tool_loop=%v), want (3, false)",
				emptyTurn.ToolCallCount, emptyTurn.HasToolLoop)
		}
	})

	t.Run("agent_response_reconstructs_from_content_parts_and_preserves_fallbacks", func(t *testing.T) {
		literalCases := []struct {
			id    string
			parts []*genai.Part
			want  string
		}{
			{id: "inv-delim-call", parts: []*genai.Part{{Text: "label | call: lookup"}}, want: "label | call: lookup"},
			{id: "inv-delim-resp", parts: []*genai.Part{{Text: "summary | resp: lookup"}}, want: "summary | resp: lookup"},
			{id: "inv-delim-other", parts: []*genai.Part{{Text: "note | other"}}, want: "note | other"},
			{id: "inv-delim-exec", parts: []*genai.Part{{Text: "snippet | Executable code (PYTHON): x = 1"}}, want: "snippet | Executable code (PYTHON): x = 1"},
			{id: "inv-delim-res", parts: []*genai.Part{{Text: "out | Code execution result (OUTCOME_OK): 1"}}, want: "out | Code execution result (OUTCOME_OK): 1"},
			{id: "inv-delim-quote", parts: []*genai.Part{{Text: "left' | text: 'right"}}, want: "left' | text: 'right"},
			{id: "inv-multipart", parts: []*genai.Part{{Text: "first | call: one"}, {Text: "second | other"}}, want: "first | call: one\nsecond | other"},
		}

		var rows []map[string]any
		for _, tc := range literalCases {
			rows = append(rows, emitCallbackTurnRows(t, syntheticTurnSpec{
				appName:          "fmt-app",
				userID:           "u-fmt",
				sessionID:        "sess-" + tc.id,
				invocationID:     tc.id,
				agentName:        "fmt-agent",
				agentDesc:        "Formatter test agent",
				instruction:      "Echo text verbatim.",
				userMsg:          &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "go"}}},
				responseParts:    tc.parts,
				promptTokens:     4,
				completionTokens: 4,
				totalTokens:      8,
			})...)
		}

		// Offloaded GCS_REFERENCE text part (e.g. cross-language or offloaded writer row where content.response is "None").
		offloadedRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "fmt-app",
			userID:        "u-fmt",
			sessionID:     "sess-offloaded",
			invocationID:  "inv-offloaded",
			agentName:     "fmt-agent",
			agentDesc:     "Formatter test agent",
			instruction:   "Echo long text.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "long"}}},
			responseParts: []*genai.Part{{Text: "placeholder"}},
		})
		for i := range offloadedRows {
			if offloadedRows[i]["event_type"] == EventAgentResponse {
				offloadedRows[i][columnContent] = `{"response":"None"}`
				offloadedRows[i][columnContentParts] = []map[string]any{{
					"part_index":   int64(0),
					"mime_type":    "text/plain",
					"storage_mode": "GCS_REFERENCE",
					"text":         "Long answer | call: details... [OFFLOADED]",
				}}
			}
		}
		rows = append(rows, offloadedRows...)

		// Thought-only content_parts row: even if content.response contains "text: 'internal'",
		// non-empty content_parts with only thought part_attributes must evaluate response_text to NULL.
		thoughtOnlyRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "fmt-app",
			userID:        "u-fmt",
			sessionID:     "sess-thought-only",
			invocationID:  "inv-thought-only",
			agentName:     "fmt-agent",
			agentDesc:     "Formatter test agent",
			instruction:   "Think silently.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "think"}}},
			responseParts: []*genai.Part{{Text: "placeholder"}},
		})
		for i := range thoughtOnlyRows {
			if thoughtOnlyRows[i]["event_type"] == EventAgentResponse {
				thoughtOnlyRows[i][columnContent] = `{"response":"text: 'internal reasoning'"}`
				thoughtOnlyRows[i][columnContentParts] = []map[string]any{{
					"part_index":      int64(0),
					"mime_type":       "text/plain",
					"storage_mode":    "INLINE",
					"part_attributes": `{"thought":true}`,
					"text":            "internal reasoning",
				}}
			}
		}
		rows = append(rows, thoughtOnlyRows...)

		// FinalResponseToolNames row: empty content_parts, JSON object in content.response.
		finalTool := &declaredTool{
			fakeTool: fakeTool{name: "submit_final_answer", description: "Submits structured answer"},
			decl:     &genai.FunctionDeclaration{Name: "submit_final_answer", Description: "Submits structured answer"},
		}
		finalToolRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:        "fmt-app",
			userID:         "u-fmt",
			sessionID:      "sess-final-tool",
			invocationID:   "inv-final-tool",
			agentName:      "fmt-agent",
			agentDesc:      "Formatter test agent",
			instruction:    "Submit via tool.",
			toolDecl:       finalTool,
			toolCallArgs:   map[string]any{"verdict": "approved", "score": 99},
			toolCallResult: map[string]any{"ok": true},
			userMsg:        &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "grade"}}},
			pluginModifier: func(c *Config) {
				c.FinalResponseToolNames = []string{"submit_final_answer"}
			},
		})
		rows = append(rows, finalToolRows...)

		turnResults := executeEvalViewSQL(t, turnSQL, rows)
		byInv := make(map[string]evalViewExecResultRow, len(turnResults))
		for _, tr := range turnResults {
			byInv[tr.InvocationID] = tr
		}

		for _, tc := range literalCases {
			gotTurn, ok := byInv[tc.id]
			if !ok {
				t.Fatalf("missing turn result for %s", tc.id)
			}
			if !gotTurn.EvalReady {
				t.Errorf("%s EvalReady = false, want true", tc.id)
			}
			if gotTexts := extractModelResponseTextsFromTrace(t, gotTurn.Trace); !cmp.Equal(gotTexts, []string{tc.want}) {
				t.Errorf("%s model response text mismatch (-want +got):\n%s", tc.id, cmp.Diff([]string{tc.want}, gotTexts))
			}
		}

		if offTurn := byInv["inv-offloaded"]; !offTurn.EvalReady {
			t.Errorf("inv-offloaded EvalReady = false, want true")
		} else if gotTexts := extractModelResponseTextsFromTrace(t, offTurn.Trace); !cmp.Equal(gotTexts, []string{"Long answer | call: details... [OFFLOADED]"}) {
			t.Errorf("inv-offloaded texts = %#v", gotTexts)
		}

		if thoughtTurn := byInv["inv-thought-only"]; thoughtTurn.EvalReady {
			t.Errorf("inv-thought-only EvalReady = true, want false (thought-only content_parts must not produce response_text)")
		} else if gotTexts := extractModelResponseTextsFromTrace(t, thoughtTurn.Trace); len(gotTexts) != 0 {
			t.Errorf("inv-thought-only unexpectedly emitted model text: %#v", gotTexts)
		}

		if ftTurn := byInv["inv-final-tool"]; !ftTurn.EvalReady {
			t.Errorf("inv-final-tool EvalReady = false, want true")
		} else if gotTexts := extractModelResponseTextsFromTrace(t, ftTurn.Trace); !cmp.Equal(gotTexts, []string{`{"score":99,"verdict":"approved"}`}) {
			t.Errorf("inv-final-tool texts = %#v, want structured JSON fallback", gotTexts)
		}

		// Also verify PayloadColumnDenylist=["content_parts"] fallback to formatted_response.
		denyPartsDefs, err := (Config{
			ProjectID:             "p",
			DatasetID:             "d",
			TableName:             "t",
			PayloadColumnDenylist: []string{columnContentParts},
		}).EvalViewDefinitions()
		if err != nil {
			t.Fatalf("EvalViewDefinitions(deny content_parts) error = %v", err)
		}
		plainRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "fmt-app",
			userID:        "u-fmt",
			sessionID:     "sess-denied-parts",
			invocationID:  "inv-denied-parts",
			agentName:     "fmt-agent",
			agentDesc:     "Formatter test agent",
			instruction:   "Plain answer.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "weather?"}}},
			responseParts: []*genai.Part{{Text: "It is 20C and sunny."}},
		})
		deniedTurnResults := executeEvalViewSQL(t, denyPartsDefs[1].Query, plainRows)
		if len(deniedTurnResults) != 1 || !deniedTurnResults[0].EvalReady {
			t.Fatalf("content_parts-denied turn results = %+v, want 1 eval_ready turn", deniedTurnResults)
		}
		if gotTexts := extractModelResponseTextsFromTrace(t, deniedTurnResults[0].Trace); !cmp.Equal(gotTexts, []string{"It is 20C and sunny."}) {
			t.Errorf("content_parts-denied fallback texts = %#v, want [\"It is 20C and sunny.\"]", gotTexts)
		}
	})

	t.Run("user_message_reconstructs_from_gcs_reference_and_skips_media_offloaded", func(t *testing.T) {
		extractUserTexts := func(trace map[string]any) []string {
			turnsAny, _ := trace["turns"].([]any)
			var texts []string
			for _, turnAny := range turnsAny {
				tm, _ := turnAny.(map[string]any)
				eventsAny, _ := tm["events"].([]any)
				for _, evAny := range eventsAny {
					em, _ := evAny.(map[string]any)
					cm, _ := em["content"].(map[string]any)
					if cm["role"] != "user" {
						continue
					}
					partsAny, _ := cm["parts"].([]any)
					for _, pAny := range partsAny {
						pm, _ := pAny.(map[string]any)
						if txt, hasTxt := pm["text"].(string); hasTxt {
							texts = append(texts, txt)
						}
					}
				}
			}
			return texts
		}

		// 1. End-to-end callback emission with real GCSOffloader offloading a text user prompt
		// exceeding MaxInlineTextSize, leaving content_parts[].storage_mode="GCS_REFERENCE"
		// and omitting content.text_summary.
		off := &fakeOffloader{}
		longPrompt := "Summarize quarterly metrics for " + strings.Repeat("segment-alpha-", 5)
		e2eRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-gcs",
			sessionID:     "sess-user-e2e-offloaded",
			invocationID:  "inv-user-e2e-offloaded",
			agentName:     "usr-agent",
			agentDesc:     "User offload test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: longPrompt}}},
			responseParts: []*genai.Part{{Text: "Summary ready."}},
			pluginModifier: func(c *Config) {
				c.GCSOffloader = off
				c.ConnectionID = "us.my-connection"
				c.MaxInlineTextSize = 40
				c.BatchFlushIntv = 10 * time.Millisecond
			},
		})
		var sawOffloadedUserRow bool
		var wantUserPreview string
		for _, r := range e2eRows {
			if r["event_type"] == EventUserMessageReceived {
				parts, _ := r[columnContentParts].([]map[string]any)
				if len(parts) == 1 && parts[0]["storage_mode"] == "GCS_REFERENCE" {
					sawOffloadedUserRow = true
					wantUserPreview, _ = parts[0]["text"].(string)
				}
				if strings.Contains(fmt.Sprint(r[columnContent]), "segment-alpha-") {
					t.Fatalf("expected USER_MESSAGE_RECEIVED content summary to omit offloaded text, got %v", r[columnContent])
				}
			}
		}
		if !sawOffloadedUserRow || !strings.HasSuffix(wantUserPreview, "... [OFFLOADED]") {
			t.Fatalf("expected fakeOffloader to offload USER_MESSAGE_RECEIVED text prompt, preview=%q calls=%d", wantUserPreview, len(off.calls))
		}

		e2eTurns := executeEvalViewSQL(t, turnSQL, e2eRows)
		if len(e2eTurns) != 1 || !e2eTurns[0].EvalReady {
			t.Fatalf("e2e offloaded user prompt v_eval_turn = %+v, want 1 eval_ready turn", e2eTurns)
		}
		if gotUser := extractUserTexts(e2eTurns[0].Trace); !cmp.Equal(gotUser, []string{wantUserPreview}) {
			t.Errorf("v_eval_turn user texts = %#v, want [%q]", gotUser, wantUserPreview)
		}
		e2eSessions := executeEvalViewSQL(t, sessionSQL, e2eRows)
		if len(e2eSessions) != 1 || !e2eSessions[0].EvalReady {
			t.Fatalf("e2e offloaded user prompt v_eval_session = %+v, want 1 eval_ready session", e2eSessions)
		}
		if gotUser := extractUserTexts(e2eSessions[0].Trace); !cmp.Equal(gotUser, []string{wantUserPreview}) {
			t.Errorf("v_eval_session user texts = %#v, want [%q]", gotUser, wantUserPreview)
		}

		// 2. Direct SQL row shapes: GCS_REFERENCE text prompt recovers preview, whereas
		// [MEDIA OFFLOADED] placeholders (non-text or text/plain) are excluded.
		gcsRefUserRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-gcs",
			sessionID:     "sess-user-gcs-ref",
			invocationID:  "inv-user-gcs-ref",
			agentName:     "usr-agent",
			agentDesc:     "User offload test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "placeholder"}}},
			responseParts: []*genai.Part{{Text: "Ack."}},
		})
		for i := range gcsRefUserRows {
			if gcsRefUserRows[i]["event_type"] == EventUserMessageReceived {
				gcsRefUserRows[i][columnContent] = "{}"
				gcsRefUserRows[i][columnContentParts] = []map[string]any{{
					"part_index":   int64(0),
					"mime_type":    "text/plain",
					"storage_mode": "GCS_REFERENCE",
					"text":         "Prompt snippet... [OFFLOADED]",
				}}
			}
		}

		mediaUserRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-gcs",
			sessionID:     "sess-user-media-offloaded",
			invocationID:  "inv-user-media-offloaded",
			agentName:     "usr-agent",
			agentDesc:     "User offload test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "placeholder"}}},
			responseParts: []*genai.Part{{Text: "Saw image."}},
		})
		for i := range mediaUserRows {
			if mediaUserRows[i]["event_type"] == EventUserMessageReceived {
				mediaUserRows[i][columnContent] = "{}"
				mediaUserRows[i][columnContentParts] = []map[string]any{{
					"part_index":   int64(0),
					"mime_type":    "image/png",
					"storage_mode": "GCS_REFERENCE",
					"text":         mediaOffloadedText,
				}}
			}
		}

		mediaTextPlaceholderRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-gcs",
			sessionID:     "sess-user-media-text-placeholder",
			invocationID:  "inv-user-media-text-placeholder",
			agentName:     "usr-agent",
			agentDesc:     "User offload test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "placeholder"}}},
			responseParts: []*genai.Part{{Text: "Ignored placeholder."}},
		})
		for i := range mediaTextPlaceholderRows {
			if mediaTextPlaceholderRows[i]["event_type"] == EventUserMessageReceived {
				mediaTextPlaceholderRows[i][columnContent] = "{}"
				mediaTextPlaceholderRows[i][columnContentParts] = []map[string]any{{
					"part_index":   int64(0),
					"mime_type":    "text/plain",
					"storage_mode": "GCS_REFERENCE",
					"text":         mediaOffloadedText,
				}}
			}
		}

		directTurns := executeEvalViewSQL(t, turnSQL, slices.Concat(gcsRefUserRows, mediaUserRows, mediaTextPlaceholderRows))
		byInv := make(map[string]evalViewExecResultRow, len(directTurns))
		for _, tr := range directTurns {
			byInv[tr.InvocationID] = tr
		}
		if gcsTurn := byInv["inv-user-gcs-ref"]; !gcsTurn.EvalReady {
			t.Errorf("inv-user-gcs-ref EvalReady = false, want true")
		} else if gotUser := extractUserTexts(gcsTurn.Trace); !cmp.Equal(gotUser, []string{"Prompt snippet... [OFFLOADED]"}) {
			t.Errorf("inv-user-gcs-ref user texts = %#v, want [\"Prompt snippet... [OFFLOADED]\"]", gotUser)
		}
		for _, id := range []string{"inv-user-media-offloaded", "inv-user-media-text-placeholder"} {
			gotTurn := byInv[id]
			if gotTurn.EvalReady {
				t.Errorf("%s EvalReady = true, want false (media placeholder must not satisfy user message text)", id)
			}
			if gotUser := extractUserTexts(gotTurn.Trace); len(gotUser) != 0 {
				t.Errorf("%s unexpectedly emitted user text: %#v", id, gotUser)
			}
		}

		// 3. ContentFormatter failure on USER_MESSAGE_RECEIVED vs legitimate user messages:
		// (a) Failed ContentFormatter on USER_MESSAGE_RECEIVED emits scalar "[FORMATTER_FAILED]"
		//     with no content_parts -> must NOT satisfy has_user_message (eval_ready=false in
		//     both v_eval_session and v_eval_turn, and 0 user text events in trace).
		// (b) Normal prompt control -> eval_ready=true in both v_eval_session and v_eval_turn.
		// (c) Structured text-summary user message literally containing "[FORMATTER_FAILED]"
		//     (both with content_parts populated and with DisableMultiModalContent=true so only
		//     content.text_summary is written) -> preserved with eval_ready=true.
		// (d) Normal scalar ContentFormatter output on USER_MESSAGE_RECEIVED -> preserved with
		//     eval_ready=true.
		// (e) Failed ContentFormatter on AGENT_STARTING emits scalar "[FORMATTER_FAILED]" ->
		//     must NOT populate trace.agents[...].instruction with "[FORMATTER_FAILED]" and falls
		//     back to attributes.static_instruction when present.
		failedFmtUserRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-fmt-guard",
			sessionID:     "sess-user-fmt-failed",
			invocationID:  "inv-user-fmt-failed",
			agentName:     "usr-agent",
			agentDesc:     "User formatter guard test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "real user question lost to formatter error"}}},
			responseParts: []*genai.Part{{Text: "Model answered anyway."}},
			pluginModifier: func(c *Config) {
				c.Logger = &recordingLogger{}
				c.ContentFormatter = func(payload any, eventType string) (any, error) {
					if eventType == EventUserMessageReceived || eventType == EventAgentStarting {
						return nil, errors.New("formatter failed")
					}
					return payload, nil
				}
			},
		})
		for i := range failedFmtUserRows {
			if failedFmtUserRows[i]["event_type"] == EventAgentStarting {
				var attrs map[string]any
				if rawAttrs, ok := failedFmtUserRows[i][columnAttributes].(string); ok && rawAttrs != "" {
					if err := json.Unmarshal([]byte(rawAttrs), &attrs); err != nil {
						t.Fatalf("json.Unmarshal(AGENT_STARTING attributes) error = %v", err)
					}
				} else {
					attrs = map[string]any{}
				}
				attrs["static_instruction"] = "Fallback static instruction."
				b, err := json.Marshal(attrs)
				if err != nil {
					t.Fatalf("json.Marshal(AGENT_STARTING attributes) error = %v", err)
				}
				failedFmtUserRows[i][columnAttributes] = string(b)
			}
		}
		normalControlRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-fmt-guard",
			sessionID:     "sess-user-fmt-control",
			invocationID:  "inv-user-fmt-control",
			agentName:     "usr-agent",
			agentDesc:     "User formatter guard test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "normal user prompt"}}},
			responseParts: []*genai.Part{{Text: "Normal response."}},
		})
		literalSummaryWithPartsRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-fmt-guard",
			sessionID:     "sess-user-literal-parts",
			invocationID:  "inv-user-literal-parts",
			agentName:     "usr-agent",
			agentDesc:     "User formatter guard test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: formattingFailedPlaceholder}}},
			responseParts: []*genai.Part{{Text: "Explained placeholder."}},
		})
		literalSummaryNoPartsRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-fmt-guard",
			sessionID:     "sess-user-literal-summary",
			invocationID:  "inv-user-literal-summary",
			agentName:     "usr-agent",
			agentDesc:     "User formatter guard test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: formattingFailedPlaceholder}}},
			responseParts: []*genai.Part{{Text: "Explained summary placeholder."}},
			pluginModifier: func(c *Config) {
				c.DisableMultiModalContent = true
			},
		})
		scalarFmtSuccessRows := emitCallbackTurnRows(t, syntheticTurnSpec{
			appName:       "usr-app",
			userID:        "u-fmt-guard",
			sessionID:     "sess-user-scalar-fmt-ok",
			invocationID:  "inv-user-scalar-fmt-ok",
			agentName:     "usr-agent",
			agentDesc:     "User formatter guard test agent",
			instruction:   "Answer user prompts.",
			userMsg:       &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "raw prompt"}}},
			responseParts: []*genai.Part{{Text: "Scalar formatter response."}},
			pluginModifier: func(c *Config) {
				c.ContentFormatter = func(payload any, eventType string) (any, error) {
					if eventType == EventUserMessageReceived {
						return "custom scalar formatted prompt", nil
					}
					return payload, nil
				}
			},
		})

		fmtGuardRows := slices.Concat(
			failedFmtUserRows,
			normalControlRows,
			literalSummaryWithPartsRows,
			literalSummaryNoPartsRows,
			scalarFmtSuccessRows,
		)
		fmtGuardTurns := executeEvalViewSQL(t, turnSQL, fmtGuardRows)
		fmtTurnByInv := make(map[string]evalViewExecResultRow, len(fmtGuardTurns))
		for _, tr := range fmtGuardTurns {
			fmtTurnByInv[tr.InvocationID] = tr
		}
		fmtGuardSessions := executeEvalViewSQL(t, sessionSQL, fmtGuardRows)
		fmtSessByID := make(map[string]evalViewExecResultRow, len(fmtGuardSessions))
		for _, sr := range fmtGuardSessions {
			fmtSessByID[sr.SessionID] = sr
		}

		if failedTurn, ok := fmtTurnByInv["inv-user-fmt-failed"]; !ok {
			t.Fatal("missing v_eval_turn row for inv-user-fmt-failed")
		} else {
			if failedTurn.EvalReady {
				t.Errorf("inv-user-fmt-failed v_eval_turn EvalReady = true, want false when USER_MESSAGE_RECEIVED ContentFormatter failed")
			}
			if gotUser := extractUserTexts(failedTurn.Trace); len(gotUser) != 0 {
				t.Errorf("inv-user-fmt-failed v_eval_turn user texts = %#v, want empty", gotUser)
			}
			turnAgents, _ := failedTurn.Trace["agents"].(map[string]any)
			if usrAg, ok := turnAgents["usr-agent"].(map[string]any); !ok {
				t.Errorf("inv-user-fmt-failed v_eval_turn missing trace.agents[usr-agent]: %#v", turnAgents)
			} else if gotInst, _ := usrAg["instruction"].(string); gotInst != "Fallback static instruction." {
				t.Errorf("inv-user-fmt-failed v_eval_turn trace.agents[usr-agent].instruction = %q, want %q", gotInst, "Fallback static instruction.")
			}
		}
		if failedSess, ok := fmtSessByID["sess-user-fmt-failed"]; !ok {
			t.Fatal("missing v_eval_session row for sess-user-fmt-failed")
		} else {
			if failedSess.EvalReady {
				t.Errorf("sess-user-fmt-failed v_eval_session EvalReady = true, want false when USER_MESSAGE_RECEIVED ContentFormatter failed")
			}
			if gotUser := extractUserTexts(failedSess.Trace); len(gotUser) != 0 {
				t.Errorf("sess-user-fmt-failed v_eval_session user texts = %#v, want empty", gotUser)
			}
			sessAgents, _ := failedSess.Trace["agents"].(map[string]any)
			if usrAg, ok := sessAgents["usr-agent"].(map[string]any); !ok {
				t.Errorf("sess-user-fmt-failed v_eval_session missing trace.agents[usr-agent]: %#v", sessAgents)
			} else if gotInst, _ := usrAg["instruction"].(string); gotInst != "Fallback static instruction." {
				t.Errorf("sess-user-fmt-failed v_eval_session trace.agents[usr-agent].instruction = %q, want %q", gotInst, "Fallback static instruction.")
			}
		}

		for _, tc := range []struct {
			sessID   string
			invID    string
			wantUser string
		}{
			{sessID: "sess-user-fmt-control", invID: "inv-user-fmt-control", wantUser: "normal user prompt"},
			{sessID: "sess-user-literal-parts", invID: "inv-user-literal-parts", wantUser: formattingFailedPlaceholder},
			{sessID: "sess-user-literal-summary", invID: "inv-user-literal-summary", wantUser: formattingFailedPlaceholder},
			{sessID: "sess-user-scalar-fmt-ok", invID: "inv-user-scalar-fmt-ok", wantUser: "custom scalar formatted prompt"},
		} {
			gotTurn, ok := fmtTurnByInv[tc.invID]
			if !ok {
				t.Fatalf("missing v_eval_turn row for %s", tc.invID)
			}
			if !gotTurn.EvalReady {
				t.Errorf("%s v_eval_turn EvalReady = false, want true", tc.invID)
			}
			if gotUser := extractUserTexts(gotTurn.Trace); !cmp.Equal(gotUser, []string{tc.wantUser}) {
				t.Errorf("%s v_eval_turn user texts = %#v, want [%q]", tc.invID, gotUser, tc.wantUser)
			}
			gotSess, ok := fmtSessByID[tc.sessID]
			if !ok {
				t.Fatalf("missing v_eval_session row for %s", tc.sessID)
			}
			if !gotSess.EvalReady {
				t.Errorf("%s v_eval_session EvalReady = false, want true", tc.sessID)
			}
			if gotUser := extractUserTexts(gotSess.Trace); !cmp.Equal(gotUser, []string{tc.wantUser}) {
				t.Errorf("%s v_eval_session user texts = %#v, want [%q]", tc.sessID, gotUser, tc.wantUser)
			}
		}
	})
}
