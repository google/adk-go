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
	"cmp"
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
	"weak"

	bq "cloud.google.com/go/bigquery"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const viewDDLTimeout = 30 * time.Second

// maxBigQueryIdentifierLen is the longest dataset ID, and table or view name
// in bytes, that BigQuery accepts.
const maxBigQueryIdentifierLen = 1024

// ViewDefinition describes a BigQuery view over the agent events table.
type ViewDefinition struct {
	// Name is the unqualified BigQuery view name (for example "v_llm_request"
	// or "v_eval_session").
	Name string
	// EventType is the event_type value filtered by a per-event view (for
	// example [EventLLMRequest]), or empty for multi-event rollups such as the
	// evaluation views.
	EventType string
	// Query is the complete CREATE OR REPLACE VIEW DDL statement for this view.
	Query string
}

// viewCommonColumns are selected by every per-event-type BigQuery view,
// matching _VIEW_COMMON_COLUMNS in adk-python.
var viewCommonColumns = []string{
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

type eventViewSpec struct {
	eventType string
	extraCols []string
}

// eventViewSpecs stores all per-event-type view definitions in deterministic
// slice order matching _EVENT_VIEW_DEFS in adk-python.
var eventViewSpecs = []eventViewSpec{
	{
		eventType: EventUserMessageReceived,
	},
	{
		eventType: EventLLMRequest,
		extraCols: []string{
			"JSON_VALUE(attributes, '$.model') AS model",
			"content AS request_content",
			"JSON_QUERY(attributes, '$.llm_config') AS llm_config",
			"JSON_QUERY(attributes, '$.tools') AS tools",
		},
	},
	{
		eventType: EventLLMResponse,
		extraCols: []string{
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
	},
	{
		eventType: EventLLMError,
		extraCols: []string{
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
	},
	{
		eventType: EventToolStarting,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
		},
	},
	{
		eventType: EventToolCompleted,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.result') AS tool_result",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
			"JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
			"JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
		},
	},
	{
		eventType: EventToolError,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(content, '$.tool_origin') AS tool_origin",
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
	},
	{
		eventType: EventAgentStarting,
		extraCols: []string{
			`COALESCE(JSON_VALUE(content, '$.text_summary'), IF(REGEXP_CONTAINS(JSON_VALUE(content, '$'), r'^(?:<(?:function|bound method) |<.+ at 0x[0-9a-fA-F]+>|functools\.partial\()'), NULL, JSON_VALUE(content, '$'))) AS agent_instruction`,
			"JSON_VALUE(attributes, '$.instruction_source') AS instruction_source",
			"JSON_VALUE(attributes, '$.static_instruction') AS static_instruction",
			"JSON_VALUE(attributes, '$.agent_description') AS agent_description",
			"JSON_VALUE(attributes, '$.model') AS model",
			"JSON_QUERY(attributes, '$.sub_agents') AS sub_agents",
		},
	},
	{
		eventType: EventAgentCompleted,
		extraCols: []string{
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
		},
	},
	{
		eventType: EventAgentError,
		extraCols: []string{
			"CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
			"JSON_VALUE(content, '$.error_traceback') AS error_traceback",
		},
	},
	{
		eventType: EventInvocationStarting,
	},
	{
		eventType: EventInvocationCompleted,
	},
	{
		eventType: EventInvocationError,
		extraCols: []string{
			"JSON_VALUE(content, '$.error_traceback') AS error_traceback",
		},
	},
	{
		eventType: EventStateDelta,
		extraCols: []string{
			"JSON_QUERY(attributes, '$.state_delta') AS state_delta",
		},
	},
	{
		eventType: EventHITLCredentialRequest,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
	},
	{
		eventType: EventHITLConfirmationRequest,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
	},
	{
		eventType: EventHITLInputRequest,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
		},
	},
	{
		eventType: EventA2AInteraction,
		extraCols: []string{
			"content AS response_content",
			`JSON_VALUE(attributes, '$.a2a_metadata."a2a:task_id"') AS a2a_task_id`,
			`JSON_VALUE(attributes, '$.a2a_metadata."a2a:context_id"') AS a2a_context_id`,
			`JSON_QUERY(attributes, '$.a2a_metadata."a2a:request"') AS a2a_request`,
			`JSON_QUERY(attributes, '$.a2a_metadata."a2a:response"') AS a2a_response`,
		},
	},
	{
		eventType: EventAgentResponse,
		extraCols: []string{
			"JSON_VALUE(content, '$.response') AS response_text",
			"JSON_VALUE(attributes, '$.source_event_id') AS source_event_id",
			"JSON_VALUE(attributes, '$.source_event_author') AS source_event_author",
			"JSON_VALUE(attributes, '$.source_event_branch') AS source_event_branch",
		},
	},
	{
		eventType: EventAgentTransfer,
		extraCols: []string{
			"JSON_VALUE(content, '$.from_agent') AS from_agent",
			"JSON_VALUE(content, '$.to_agent') AS to_agent",
			"JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
		},
	},
	{
		// Deliberately differs from adk-python's EVENT_COMPACTION projection
		// (which uses bare CAST(... AS FLOAT64), * 1000000, and TIMESTAMP_MICROS):
		// SAFE_CAST, SAFE_MULTIPLY, and SAFE.TIMESTAMP_MICROS yield NULL instead
		// of failing the query on out-of-range numbers, and the COALESCE fallback
		// parses legacy RFC 3339 strings (including 9-digit nanoseconds). Both
		// projections expose the same column names and BigQuery types.
		eventType: EventEventCompaction,
		extraCols: []string{
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS start_seconds`,
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS end_seconds`,
			`SAFE.TIMESTAMP_MICROS(SAFE_CAST(SAFE_MULTIPLY(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6), 1000000) AS INT64)) AS window_start`,
			`SAFE.TIMESTAMP_MICROS(SAFE_CAST(SAFE_MULTIPLY(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6), 1000000) AS INT64)) AS window_end`,
			"JSON_QUERY(content, '$.compacted_content') AS compacted_content",
		},
	},
	{
		eventType: EventAgentStateCheckpoint,
		extraCols: []string{
			"JSON_QUERY(content, '$.agent_state') AS agent_state",
			"JSON_TYPE(JSON_QUERY(content, '$.agent_state')) AS agent_state_type",
			"SAFE_CAST(JSON_VALUE(content, '$.end_of_agent') AS BOOL) AS end_of_agent",
			"JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
		},
	},
	{
		eventType: EventToolPaused,
		extraCols: []string{
			"JSON_VALUE(content, '$.tool') AS tool_name",
			"JSON_QUERY(content, '$.args') AS tool_args",
			"JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
			"JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
		},
	},
	{
		eventType: EventNodeOutput,
		extraCols: []string{
			"JSON_VALUE(attributes, '$.adk.node.path') AS node_path",
			"JSON_VALUE(attributes, '$.adk.node.run_id') AS node_run_id",
			"JSON_VALUE(attributes, '$.adk.node.parent_run_id') AS node_parent_run_id",
			"content AS output",
		},
	},
	{
		eventType: EventNodeError,
		extraCols: []string{
			"JSON_VALUE(attributes, '$.adk.node.path') AS node_path",
			"JSON_VALUE(attributes, '$.adk.node.run_id') AS node_run_id",
			"JSON_VALUE(attributes, '$.adk.node.parent_run_id') AS node_parent_run_id",
			"JSON_VALUE(content, '$.error_code') AS error_code",
		},
	},
}

// ViewDefinitions returns the CREATE OR REPLACE VIEW statements that the
// plugin runs for c, one per view in deterministic order, honoring
// [Config.ViewPrefix] (empty means "v") and [Config.PayloadColumnDenylist].
// A caller without a full Config can set only the fields that name the views,
// for example Config{ProjectID: p, DatasetID: d, TableName: t}.ViewDefinitions().
// It returns an error if an identifier is invalid or a view name would equal
// the events table name.
func (c Config) ViewDefinitions() ([]ViewDefinition, error) {
	c = applyDefaults(c)
	denied, err := parsePayloadColumnDenylist(c.PayloadColumnDenylist)
	if err != nil {
		return nil, err
	}
	return viewDefinitionsWithDenied(c.ProjectID, c.DatasetID, c.TableName, c.ViewPrefix, denied)
}

func viewDefinitionsWithDenied(projectID, datasetID, tableName, viewPrefix string, deniedColumns map[string]struct{}) ([]ViewDefinition, error) {
	if viewPrefix == "" {
		viewPrefix = defaultViewPrefix
	}
	if err := validateViewIdentifiers(projectID, datasetID, tableName, viewPrefix); err != nil {
		return nil, err
	}

	defs := make([]ViewDefinition, 0, len(eventViewSpecs))
	for _, spec := range eventViewSpecs {
		viewName := viewPrefix + "_" + strings.ToLower(spec.eventType)
		if len(viewName) > maxBigQueryIdentifierLen {
			return nil, fmt.Errorf("BigQuery view name %q is longer than %d bytes; use a shorter view prefix", viewName, maxBigQueryIdentifierLen)
		}
		// Datasets can be case-insensitive, so a view must not differ from the
		// events table only in case either.
		if strings.EqualFold(viewName, tableName) {
			return nil, fmt.Errorf("BigQuery view name %q collides with the events table %q; use another view prefix", viewName, tableName)
		}
		extra := projectViewColumns(spec.extraCols, deniedColumns)
		cols := make([]string, 0, len(viewCommonColumns)+len(extra))
		cols = append(cols, viewCommonColumns...)
		cols = append(cols, extra...)

		query := fmt.Sprintf(
			"CREATE OR REPLACE VIEW `%s.%s.%s` AS\nSELECT\n  %s\nFROM\n  `%s.%s.%s`\nWHERE\n  event_type = '%s'\n",
			projectID, datasetID, viewName,
			strings.Join(cols, ",\n  "),
			projectID, datasetID, tableName,
			spec.eventType,
		)
		defs = append(defs, ViewDefinition{
			Name:      viewName,
			EventType: spec.eventType,
			Query:     query,
		})
	}
	return defs, nil
}

// evalViewRequiredEvents lists the 15 event types required to reconstruct
// AgentData evaluation trajectories and turn completion/pause state in the
// evaluation views, in sorted order matching _EVAL_VIEW_REQUIRED_EVENTS in
// adk-python.
var evalViewRequiredEvents = []string{
	EventAgentResponse,
	EventAgentStarting,
	EventAgentTransfer,
	EventHITLConfirmationRequest,
	EventHITLConfirmationCompleted,
	EventHITLCredentialRequest,
	EventHITLCredentialCompleted,
	EventHITLInputRequest,
	EventHITLInputCompleted,
	EventInvocationCompleted,
	EventLLMRequest,
	EventToolCompleted,
	EventToolPaused,
	EventToolStarting,
	EventUserMessageReceived,
}

// evalViewLegacyDeniedEvents holds the subset of evalViewRequiredEvents that
// are dropped on at least one emission path when "EVENT" is in EventDenylist
// (session/HITL/pause events emitted through logSessionEvent or
// logSessionSpanEvent, plus TOOL_COMPLETED because resumed long-running tool
// calls emit it through logSessionSpanEvent).
var evalViewLegacyDeniedEvents = newSet(
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
)

// evalViewLegacyAlwaysAllowedEvents holds the subset of evalViewRequiredEvents
// whose every active runtime emitter checks shouldLogSessionEvent and is
// therefore covered unconditionally by "EVENT" in EventAllowlist.
// EventAgentResponse is checked separately in validateEvalViewConfig because
// FinalResponseToolNames adds a logEvent emission path that ignores "EVENT".
var evalViewLegacyAlwaysAllowedEvents = newSet(
	EventAgentTransfer,
	EventHITLConfirmationRequest,
	EventHITLConfirmationCompleted,
	EventHITLCredentialRequest,
	EventHITLCredentialCompleted,
	EventHITLInputRequest,
	EventHITLInputCompleted,
	EventToolPaused,
)

func validateEvalViewConfig(c Config, deniedColumns map[string]struct{}, fromInit bool) error {
	prefix := "cannot create eval views"
	if fromInit {
		prefix = "invalid config: CreateEvalViews cannot be used"
	}
	if _, ok := deniedColumns[columnContent]; ok {
		return fmt.Errorf("%s when %q is in PayloadColumnDenylist, because AgentData trace reconstruction requires the %q column", prefix, columnContent, columnContent)
	}
	if len(c.EventDenylist) > 0 {
		denylist := newSet(c.EventDenylist...)
		_, denyLegacy := denylist[legacyEventType]
		var deniedEvents []string
		for _, ev := range evalViewRequiredEvents {
			_, denyEv := denylist[ev]
			_, legacyDenied := evalViewLegacyDeniedEvents[ev]
			if denyEv || (denyLegacy && legacyDenied) {
				deniedEvents = append(deniedEvents, ev)
			}
		}
		if len(deniedEvents) > 0 {
			return fmt.Errorf("%s when required events are in EventDenylist: %v", prefix, deniedEvents)
		}
	}
	if len(c.EventAllowlist) > 0 {
		allowlist := newSet(c.EventAllowlist...)
		_, allowLegacy := allowlist[legacyEventType]
		var missingEvents []string
		for _, ev := range evalViewRequiredEvents {
			_, allowEv := allowlist[ev]
			_, allowedByLegacy := evalViewLegacyAlwaysAllowedEvents[ev]
			if ev == EventAgentResponse && len(c.FinalResponseToolNames) == 0 {
				allowedByLegacy = true
			}
			if !allowEv && !(allowLegacy && allowedByLegacy) {
				missingEvents = append(missingEvents, ev)
			}
		}
		if len(missingEvents) > 0 {
			return fmt.Errorf("%s unless EventAllowlist includes all required events; missing: %v", prefix, missingEvents)
		}
	}
	return nil
}

const evalViewCTEsTemplate = `WITH
  raw_events AS (
    SELECT
      timestamp,
      event_id,
      event_type,
      agent,
      session_id,
      invocation_id,
      user_id,
      content,
      {content_parts_column},
      {attributes_column},
      {latency_ms_column},
      status,
      error_message,
      is_truncated
    FROM
      {full_table_ref}
    WHERE
      session_id IS NOT NULL
      AND session_id != ''
      AND invocation_id IS NOT NULL
      AND invocation_id != ''
  ),
  ranked_events AS (
    SELECT
      timestamp,
      event_id,
      event_type,
      agent,
      session_id,
      invocation_id,
      COALESCE(
        NULLIF(user_id, ''),
        IF(
          MIN(NULLIF(user_id, '')) OVER (PARTITION BY session_id, invocation_id)
            = MAX(NULLIF(user_id, '')) OVER (
              PARTITION BY session_id, invocation_id
            ),
          MAX(NULLIF(user_id, '')) OVER (
            PARTITION BY session_id, invocation_id
          ),
          NULL
        ),
        user_id
      ) AS scoped_user_id,
      NULLIF(JSON_VALUE(attributes, '$.adk.app_name'), '') AS app_name_raw,
      content,
      content_parts,
      attributes,
      latency_ms,
      status,
      error_message,
      is_truncated,
      IF(
        COALESCE(JSON_VALUE(content, '$'), '') = '[FORMATTER_FAILED]', 1, 0
      ) AS formatter_failed_rank,
      IF(
        EXISTS(
          SELECT 1
          FROM UNNEST(content_parts) AS p
          WHERE p.text = '[UPLOAD FAILED]'
        ),
        1,
        0
      ) AS upload_failed_rank,
      IF(
        EXISTS(
          SELECT 1
          FROM UNNEST(content_parts) AS p
          WHERE p.storage_mode = 'GCS_REFERENCE'
        ),
        0,
        1
      ) AS gcs_offload_rank,
      IF(COALESCE(is_truncated, FALSE), 1, 0) AS truncated_rank,
      COALESCE(TO_JSON_STRING(content), '') AS content_tiebreaker
    FROM
      raw_events
  ),
  scoped_events AS (
    SELECT
      *,
      COALESCE(
        app_name_raw,
        IF(
          MIN(app_name_raw) OVER (
            PARTITION BY session_id, scoped_user_id, invocation_id
          ) = MAX(app_name_raw) OVER (
            PARTITION BY session_id, scoped_user_id, invocation_id
          ),
          MAX(app_name_raw) OVER (
            PARTITION BY session_id, scoped_user_id, invocation_id
          ),
          NULL
        )
      ) AS app_name
    FROM
      ranked_events
  ),
  deduped_events AS (
    SELECT
      timestamp,
      event_id,
      event_type,
      agent,
      session_id,
      invocation_id,
      scoped_user_id AS user_id,
      app_name,
      content,
      content_parts,
      attributes,
      latency_ms,
      status,
      error_message,
      is_truncated
    FROM
      scoped_events
    QUALIFY
      event_id IS NULL
      OR event_id = ''
      OR ROW_NUMBER() OVER (
        PARTITION BY
          session_id,
          scoped_user_id,
          app_name,
          invocation_id,
          event_id
        ORDER BY
          formatter_failed_rank ASC,
          upload_failed_rank ASC,
          gcs_offload_rank ASC,
          truncated_rank ASC,
          timestamp DESC,
          content_tiebreaker DESC
      ) = 1
  ),
  typed_events AS (
    SELECT
      *,
      COALESCE(
        agent, JSON_VALUE(attributes, '$.root_agent_name'), 'root_agent'
      ) AS author,
      JSON_VALUE(content, '$.tool') AS tool_name,
      TO_JSON_STRING(JSON_QUERY(content, '$.args')) AS tool_args_json,
      CASE
        WHEN event_type != 'USER_MESSAGE_RECEIVED' THEN NULL
        WHEN COALESCE(ARRAY_LENGTH(content_parts), 0) > 0 THEN (
          SELECT
            STRING_AGG(p.text, '\n' ORDER BY p.part_index)
          FROM
            UNNEST(content_parts) AS p
          WHERE
            p.mime_type = 'text/plain'
            AND COALESCE(p.storage_mode, 'INLINE')
              IN ('INLINE', 'GCS_REFERENCE')
            AND COALESCE(p.part_attributes, '{}') = '{}'
            AND p.text IS NOT NULL
            AND p.text NOT IN (
              '', '[BINARY DATA]', '[MEDIA OFFLOADED]', '[UPLOAD FAILED]'
            )
        )
        WHEN NOT REGEXP_CONTAINS(
          COALESCE(
            JSON_VALUE(content, '$.text_summary'),
            JSON_VALUE(content, '$'),
            ''
          ),
          r'^Function response: '
        ) THEN NULLIF(
          COALESCE(
            JSON_VALUE(content, '$.text_summary'), JSON_VALUE(content, '$')
          ),
          ''
        )
        ELSE NULL
      END AS message_text,
      (
        event_type = 'USER_MESSAGE_RECEIVED'
        AND (
          REGEXP_CONTAINS(
            COALESCE(
              JSON_VALUE(content, '$.text_summary'),
              JSON_VALUE(content, '$'),
              ''
            ),
            r'^Function response: '
          )
          OR EXISTS(
            SELECT 1
            FROM UNNEST(content_parts) AS p
            WHERE
              JSON_QUERY(p.part_attributes, '$.function_response')
              IS NOT NULL
          )
        )
      ) AS is_function_response,
      IF(
        event_type = 'AGENT_RESPONSE',
        COALESCE(
          JSON_VALUE(content, '$.response'),
          IF(
            JSON_TYPE(JSON_QUERY(content, '$.response'))
              IN ('object', 'array'),
            TO_JSON_STRING(JSON_QUERY(content, '$.response')),
            NULL
          )
        ),
        NULL
      ) AS formatted_response,
      (
        event_type = 'LLM_RESPONSE'
        AND (
          JSON_VALUE(attributes, '$.finish_reason') IS NOT NULL
          OR JSON_VALUE(attributes, '$.cache_type') IS NOT NULL
        )
      ) AS is_final_llm_response,
      SAFE_CAST(JSON_VALUE(content, '$.usage.total') AS INT64) AS usage_total,
      SAFE_CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64)
        AS usage_prompt,
      SAFE_CAST(JSON_VALUE(content, '$.usage.completion') AS INT64)
        AS usage_completion,
      SAFE_CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms
    FROM
      deduped_events
  ),
  response_events AS (
    SELECT
      *,
      (
        event_type = 'TOOL_STARTING'
        AND COALESCE(tool_name, '') != 'transfer_to_agent'
      ) AS is_tool_call,
      CASE
        WHEN event_type != 'AGENT_RESPONSE' THEN NULL
        WHEN COALESCE(ARRAY_LENGTH(content_parts), 0) > 0 THEN (
          SELECT
            STRING_AGG(p.text, '\n' ORDER BY p.part_index)
          FROM
            UNNEST(content_parts) AS p
          WHERE
            p.mime_type = 'text/plain'
            AND COALESCE(p.storage_mode, 'INLINE')
              IN ('INLINE', 'GCS_REFERENCE')
            AND COALESCE(p.part_attributes, '{}') = '{}'
            AND p.text IS NOT NULL
            AND p.text NOT IN (
              '', '[BINARY DATA]', '[MEDIA OFFLOADED]', '[UPLOAD FAILED]'
            )
        )
        WHEN
          REGEXP_CONTAINS(formatted_response, r"(?s)^text: '.*'$")
          AND NOT REGEXP_CONTAINS(
            formatted_response,
            r" [|] (?:call: |resp: |Executable code |Code execution result )"
          )
          AND NOT REGEXP_CONTAINS(formatted_response, r" [|] other(?: [|] |$)")
          THEN NULLIF(
            REGEXP_REPLACE(
              REGEXP_REPLACE(formatted_response, r"^text: '|'$", ''),
              r"' [|] text: '",
              CHR(10)
            ),
            ''
          )
        WHEN
          NOT REGEXP_CONTAINS(
            formatted_response,
            r"^(?:call: |resp: |Executable code |Code execution result )"
          )
          AND NOT REGEXP_CONTAINS(
            formatted_response, r"^(?:None$|text: '|other(?: [|] |$))"
          )
          THEN NULLIF(formatted_response, '')
      END AS response_text
    FROM
      typed_events
  ),
  annotated_events AS (
    SELECT
      *,
      (
        COUNTIF(is_final_llm_response)
          OVER (PARTITION BY session_id, user_id, app_name, invocation_id)
      ) > 0 AS turn_has_final_llm_response,
      ROW_NUMBER() OVER (
        PARTITION BY session_id, user_id, app_name, invocation_id, is_tool_call
        ORDER BY timestamp, event_id
      ) - ROW_NUMBER() OVER (
        PARTITION BY
          session_id,
          user_id,
          app_name,
          invocation_id,
          is_tool_call,
          tool_name,
          tool_args_json
        ORDER BY timestamp, event_id
      ) AS tool_call_island
    FROM
      response_events
  ),
  trace_events AS (
    SELECT
      session_id,
      user_id,
      app_name,
      invocation_id,
      timestamp,
      event_id,
      event_type,
      agent,
      JSON_VALUE(attributes, '$.root_agent_name') AS root_agent_name,
      status,
      error_message,
      is_truncated,
      message_text,
      response_text,
      is_function_response,
      is_tool_call,
      IF(is_tool_call, NULLIF(tool_name, ''), NULL) AS invoked_tool_name,
      IF(
        is_tool_call,
        COUNT(*) OVER (
          PARTITION BY
            session_id,
            user_id,
            app_name,
            invocation_id,
            is_tool_call,
            tool_name,
            tool_args_json,
            tool_call_island
        ),
        NULL
      ) AS tool_call_run_length,
      (
        event_type = 'LLM_RESPONSE'
        AND (is_final_llm_response OR NOT turn_has_final_llm_response)
      ) AS is_usage_row,
      usage_total,
      usage_prompt,
      usage_completion,
      total_ms,
      CASE
        WHEN event_type = 'USER_MESSAGE_RECEIVED' THEN 1
        WHEN
          event_type IN (
            'TOOL_STARTING',
            'HITL_CREDENTIAL_REQUEST',
            'HITL_CONFIRMATION_REQUEST',
            'HITL_INPUT_REQUEST'
          )
          THEN 2
        WHEN event_type = 'AGENT_TRANSFER' THEN 3
        WHEN
          event_type IN (
            'TOOL_COMPLETED',
            'HITL_CREDENTIAL_REQUEST_COMPLETED',
            'HITL_CONFIRMATION_REQUEST_COMPLETED',
            'HITL_INPUT_REQUEST_COMPLETED'
          )
          THEN 4
        ELSE 5
      END AS event_seq,
      CASE event_type
        WHEN 'USER_MESSAGE_RECEIVED' THEN 'user'
        WHEN 'AGENT_TRANSFER'
          THEN COALESCE(JSON_VALUE(content, '$.from_agent'), author)
        ELSE author
      END AS event_author,
      CASE
        WHEN event_type = 'USER_MESSAGE_RECEIVED' AND message_text IS NOT NULL
          THEN JSON_OBJECT(
            'role', 'user',
            'parts', JSON_ARRAY(JSON_OBJECT('text', message_text))
          )
        WHEN
          (
            is_tool_call
            AND COALESCE(tool_name, '') NOT IN (
              'adk_request_credential',
              'adk_request_confirmation',
              'adk_request_input'
            )
          )
          OR event_type IN (
            'HITL_CREDENTIAL_REQUEST',
            'HITL_CONFIRMATION_REQUEST',
            'HITL_INPUT_REQUEST'
          )
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_call', JSON_OBJECT(
                  'name', COALESCE(tool_name, ''),
                  'args', IF(
                    JSON_TYPE(JSON_QUERY(content, '$.args')) = 'object',
                    JSON_QUERY(content, '$.args'),
                    JSON_OBJECT()
                  )
                )
              )
            )
          )
        WHEN event_type = 'AGENT_TRANSFER'
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_call', JSON_OBJECT(
                  'name', 'transfer_to_agent',
                  'args', JSON_OBJECT(
                    'agent_name',
                    COALESCE(JSON_VALUE(content, '$.to_agent'), '')
                  )
                )
              )
            )
          )
        WHEN
          (
            event_type = 'TOOL_COMPLETED'
            AND COALESCE(tool_name, '') NOT IN (
              'transfer_to_agent',
              'adk_request_credential',
              'adk_request_confirmation',
              'adk_request_input'
            )
          )
          OR event_type IN (
            'HITL_CREDENTIAL_REQUEST_COMPLETED',
            'HITL_CONFIRMATION_REQUEST_COMPLETED',
            'HITL_INPUT_REQUEST_COMPLETED'
          )
          THEN JSON_OBJECT(
            'role', 'user',
            'parts', JSON_ARRAY(
              JSON_OBJECT(
                'function_response', JSON_OBJECT(
                  'name', COALESCE(tool_name, ''),
                  'response', IF(
                    JSON_TYPE(JSON_QUERY(content, '$.result')) = 'object',
                    JSON_QUERY(content, '$.result'),
                    JSON_OBJECT('result', JSON_QUERY(content, '$.result'))
                  )
                )
              )
            )
          )
        WHEN event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
          THEN JSON_OBJECT(
            'role', 'model',
            'parts', JSON_ARRAY(JSON_OBJECT('text', response_text))
          )
      END AS event_content
    FROM
      annotated_events
  ),
  turn_rollup AS (
    SELECT
      session_id,
      user_id,
      app_name,
      invocation_id,
      ARRAY_AGG(
        root_agent_name IGNORE NULLS ORDER BY timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS first_root_agent_name,
      ARRAY_AGG(
        agent IGNORE NULLS ORDER BY timestamp LIMIT 1
      )[SAFE_OFFSET(0)] AS first_agent,
      MIN(timestamp) AS turn_start_timestamp,
      MAX(timestamp) AS turn_end_timestamp,
      COUNTIF(is_tool_call) AS tool_call_count,
      COUNTIF(event_type = 'TOOL_ERROR') AS tool_error_count,
      COUNTIF(event_type = 'LLM_ERROR') AS llm_error_count,
      COUNTIF(event_type = 'AGENT_ERROR') AS agent_error_count,
      COALESCE(
        LOGICAL_OR(
          status = 'ERROR'
          OR (COALESCE(status, '') != 'OK' AND error_message IS NOT NULL)
          OR (event_type = 'LLM_RESPONSE' AND error_message IS NOT NULL)
        ),
        FALSE
      ) AS has_error,
      LOGICAL_OR(COALESCE(is_truncated, FALSE)) AS has_truncated_content,
      COALESCE(MAX(tool_call_run_length), 0) >= 3 AS has_tool_loop,
      COALESCE(MAX(tool_call_run_length), 0)
        AS max_consecutive_identical_tool_calls,
      COALESCE(
        ARRAY_AGG(
          DISTINCT invoked_tool_name IGNORE NULLS ORDER BY invoked_tool_name
        ),
        ARRAY<STRING>[]
      ) AS distinct_tools_used,
      COALESCE(SUM(IF(is_usage_row, usage_prompt, NULL)), 0)
        AS total_prompt_tokens,
      COALESCE(SUM(IF(is_usage_row, usage_completion, NULL)), 0)
        AS total_completion_tokens,
      COALESCE(SUM(IF(is_usage_row, usage_total, NULL)), 0) AS total_tokens,
      COALESCE(
        MAX(IF(event_type = 'INVOCATION_COMPLETED', total_ms, NULL)),
        MAX(IF(event_type = 'AGENT_COMPLETED', total_ms, NULL))
      ) AS turn_latency_ms,
      (
        COUNTIF(event_type = 'INVOCATION_COMPLETED') > 0
        AND COUNTIF(
          event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
        ) = 0
        AND COUNTIF(
          event_type IN (
            'TOOL_PAUSED',
            'HITL_CREDENTIAL_REQUEST',
            'HITL_CONFIRMATION_REQUEST',
            'HITL_INPUT_REQUEST'
          )
        ) > 0
      ) AS is_paused,
      (
        COUNTIF(event_type = 'INVOCATION_COMPLETED') > 0
        AND COUNTIF(
          event_type = 'AGENT_RESPONSE' AND response_text IS NOT NULL
        ) > 0
      ) AS is_complete,
      COUNTIF(
        event_type = 'USER_MESSAGE_RECEIVED' AND message_text IS NOT NULL
      ) > 0 AS has_user_message,
      COUNTIF(is_function_response) > 0 AS has_function_response,
      ARRAY_AGG(
        IF(
          event_content IS NULL,
          NULL,
          JSON_OBJECT(
            'author', event_author,
            'content', event_content,
            'event_time',
            FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E6SZ', timestamp)
          )
        )
        IGNORE NULLS
        ORDER BY timestamp, event_seq, event_id
      ) AS events
    FROM
      trace_events
    GROUP BY
      session_id,
      user_id,
      app_name,
      invocation_id
  ),
  turns AS (
    SELECT
      *,
      COALESCE(first_root_agent_name, first_agent) AS root_agent,
      ROW_NUMBER() OVER (
        PARTITION BY session_id, user_id, app_name
        ORDER BY turn_start_timestamp, invocation_id
      ) - 1 AS turn_index,
      IF(
        COALESCE(ARRAY_LENGTH(events), 0) > 0, TO_JSON(events), JSON_ARRAY()
      ) AS events_json,
      (
        has_user_message
        AND is_complete
        AND COALESCE(ARRAY_LENGTH(events), 0) > 0
      ) AS eval_ready,
      (
        (
          has_user_message
          OR (
            has_function_response
            AND COALESCE(
              LAG(is_paused) OVER (
                PARTITION BY session_id, user_id, app_name
                ORDER BY turn_start_timestamp, invocation_id
              ),
              FALSE
            )
          )
        )
        AND (
          is_complete
          OR (
            is_paused
            AND COALESCE(
              LEAD(has_function_response) OVER (
                PARTITION BY session_id, user_id, app_name
                ORDER BY turn_start_timestamp, invocation_id
              ),
              FALSE
            )
          )
        )
        AND COALESCE(ARRAY_LENGTH(events), 0) > 0
      ) AS session_turn_eval_ready
    FROM
      turn_rollup
  ),
  agent_rows AS (
    SELECT
      session_id,
      user_id,
      app_name,
      invocation_id,
      agent AS agent_id,
      timestamp,
      IF(
        event_type = 'AGENT_STARTING',
        COALESCE(
          JSON_VALUE(content, '$.text_summary'),
          IF(
            REGEXP_CONTAINS(
              JSON_VALUE(content, '$'),
              r'^(?:<(?:function|bound method) |<.+ at 0x[0-9a-fA-F]+>|functools\.partial\()'
            ),
            NULL,
            JSON_VALUE(content, '$')
          ),
          JSON_VALUE(attributes, '$.static_instruction')
        ),
        NULL
      ) AS instruction,
      IF(
        event_type = 'AGENT_STARTING',
        JSON_VALUE(attributes, '$.agent_description'),
        NULL
      ) AS description,
      IF(
        event_type = 'AGENT_STARTING'
        AND JSON_TYPE(JSON_QUERY(attributes, '$.sub_agents')) = 'array',
        JSON_QUERY(attributes, '$.sub_agents'),
        NULL
      ) AS sub_agents,
      IF(
        event_type = 'LLM_REQUEST'
        AND JSON_TYPE(JSON_QUERY(attributes, '$.tools')) = 'array',
        JSON_QUERY(attributes, '$.tools'),
        NULL
      ) AS tool_declarations
    FROM
      deduped_events
    WHERE
      agent IS NOT NULL
  ),
  agent_latest AS (
    SELECT
      {scope_columns},
      agent_id,
      ARRAY_AGG(
        instruction IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS instruction,
      ARRAY_AGG(
        description IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS description,
      ARRAY_AGG(
        sub_agents IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS sub_agents,
      ARRAY_AGG(
        tool_declarations IGNORE NULLS ORDER BY timestamp DESC LIMIT 1
      )[SAFE_OFFSET(0)] AS tool_declarations
    FROM
      agent_rows
    GROUP BY
      {scope_columns},
      agent_id
  ),
  agent_tool_declarations AS (
    SELECT
      {scope_columns},
      agent_id,
      ARRAY_AGG(
        IF(
          JSON_TYPE(JSON_QUERY(declaration, '$.parameters')) = 'object',
          JSON_OBJECT(
            'name', JSON_VALUE(declaration, '$.name'),
            'description',
            COALESCE(JSON_VALUE(declaration, '$.description'), ''),
            'parameters_json_schema', JSON_QUERY(declaration, '$.parameters')
          ),
          JSON_OBJECT(
            'name', JSON_VALUE(declaration, '$.name'),
            'description',
            COALESCE(JSON_VALUE(declaration, '$.description'), '')
          )
        )
        ORDER BY declaration_offset
      ) AS function_declarations
    FROM
      agent_latest,
      UNNEST(JSON_QUERY_ARRAY(tool_declarations)) AS declaration
        WITH OFFSET AS declaration_offset
    WHERE
      JSON_VALUE(declaration, '$.name') IS NOT NULL
    GROUP BY
      {scope_columns},
      agent_id
  ),
  agent_configs AS (
    SELECT
      {qualified_scope_columns},
      l.agent_id,
      JSON_OBJECT(
        'agent_id', l.agent_id,
        'instruction', COALESCE(l.instruction, ''),
        'description', COALESCE(l.description, ''),
        'sub_agents', COALESCE(l.sub_agents, JSON_ARRAY()),
        'tools', IF(
          d.function_declarations IS NULL,
          JSON_ARRAY(),
          JSON_ARRAY(
            JSON_OBJECT(
              'function_declarations', TO_JSON(d.function_declarations)
            )
          )
        )
      ) AS agent_config
    FROM
      agent_latest AS l
    LEFT JOIN
      agent_tool_declarations AS d
      ON {agent_configs_join_condition}
      AND l.agent_id = d.agent_id
  ),
  agent_maps AS (
    SELECT
      {scope_columns},
      JSON_OBJECT(
        ARRAY_AGG(agent_id ORDER BY agent_id),
        ARRAY_AGG(agent_config ORDER BY agent_id)
      ) AS agents
    FROM
      agent_configs
    GROUP BY
      {scope_columns}
  )`

const evalSessionViewSelectSQL = `,
  session_rollup AS (
    SELECT
      session_id,
      user_id,
      app_name,
      COALESCE(
        ARRAY_AGG(
          first_root_agent_name IGNORE NULLS
          ORDER BY turn_start_timestamp, invocation_id
          LIMIT 1
        )[SAFE_OFFSET(0)],
        ARRAY_AGG(
          first_agent IGNORE NULLS
          ORDER BY turn_start_timestamp, invocation_id
          LIMIT 1
        )[SAFE_OFFSET(0)]
      ) AS root_agent,
      MIN(turn_start_timestamp) AS session_start_timestamp,
      MAX(turn_end_timestamp) AS session_end_timestamp,
      COUNT(DISTINCT invocation_id) AS turn_count,
      SUM(tool_call_count) AS tool_call_count,
      SUM(tool_error_count) AS tool_error_count,
      SUM(llm_error_count) AS llm_error_count,
      SUM(agent_error_count) AS agent_error_count,
      LOGICAL_OR(has_error) AS has_error,
      LOGICAL_OR(has_truncated_content) AS has_truncated_content,
      LOGICAL_OR(has_tool_loop) AS has_tool_loop,
      MAX(max_consecutive_identical_tool_calls)
        AS max_consecutive_identical_tool_calls,
      ARRAY_CONCAT_AGG(distinct_tools_used) AS tools_used_by_turn,
      SUM(total_prompt_tokens) AS total_prompt_tokens,
      SUM(total_completion_tokens) AS total_completion_tokens,
      SUM(total_tokens) AS total_tokens,
      SUM(turn_latency_ms) AS total_latency_ms,
      LOGICAL_AND(session_turn_eval_ready) AS eval_ready,
      ARRAY_AGG(
        JSON_OBJECT(
          'turn_index', turn_index,
          'turn_id', invocation_id,
          'events', events_json
        )
        ORDER BY turn_index
      ) AS turn_traces
    FROM
      turns
    GROUP BY
      session_id,
      user_id,
      app_name
  )
SELECT
  s.session_id AS session_id,
  s.user_id AS user_id,
  s.root_agent AS root_agent,
  s.session_start_timestamp AS session_start_timestamp,
  s.session_end_timestamp AS session_end_timestamp,
  s.turn_count AS turn_count,
  s.tool_call_count AS tool_call_count,
  s.tool_error_count AS tool_error_count,
  s.llm_error_count AS llm_error_count,
  s.agent_error_count AS agent_error_count,
  s.has_error AS has_error,
  s.has_truncated_content AS has_truncated_content,
  s.has_tool_loop AS has_tool_loop,
  s.max_consecutive_identical_tool_calls
    AS max_consecutive_identical_tool_calls,
  ARRAY(
    SELECT DISTINCT tool FROM UNNEST(s.tools_used_by_turn) AS tool ORDER BY tool
  ) AS distinct_tools_used,
  s.total_prompt_tokens AS total_prompt_tokens,
  s.total_completion_tokens AS total_completion_tokens,
  s.total_tokens AS total_tokens,
  s.total_latency_ms AS total_latency_ms,
  s.eval_ready AS eval_ready,
  JSON_OBJECT(
    'agents', COALESCE(m.agents, JSON_OBJECT()),
    'turns', TO_JSON(s.turn_traces)
  ) AS trace
FROM
  session_rollup AS s
LEFT JOIN
  agent_maps AS m
  ON m.session_id = s.session_id
  AND m.user_id IS NOT DISTINCT FROM s.user_id
  AND m.app_name IS NOT DISTINCT FROM s.app_name
`

const evalTurnViewSelectSQL = `
SELECT
  t.session_id AS session_id,
  t.invocation_id AS invocation_id,
  t.turn_index AS turn_index,
  t.user_id AS user_id,
  t.root_agent AS root_agent,
  t.turn_start_timestamp AS turn_start_timestamp,
  t.turn_end_timestamp AS turn_end_timestamp,
  t.tool_call_count AS tool_call_count,
  t.tool_error_count AS tool_error_count,
  t.llm_error_count AS llm_error_count,
  t.agent_error_count AS agent_error_count,
  t.has_error AS has_error,
  t.has_truncated_content AS has_truncated_content,
  t.has_tool_loop AS has_tool_loop,
  t.max_consecutive_identical_tool_calls
    AS max_consecutive_identical_tool_calls,
  t.distinct_tools_used AS distinct_tools_used,
  t.total_prompt_tokens AS total_prompt_tokens,
  t.total_completion_tokens AS total_completion_tokens,
  t.total_tokens AS total_tokens,
  t.turn_latency_ms AS turn_latency_ms,
  t.eval_ready AS eval_ready,
  JSON_OBJECT(
    'agents', COALESCE(m.agents, JSON_OBJECT()),
    'turns', JSON_ARRAY(
      JSON_OBJECT(
        'turn_index', 0,
        'turn_id', t.invocation_id,
        'events', t.events_json
      )
    )
  ) AS trace
FROM
  turns AS t
LEFT JOIN
  agent_maps AS m
  ON m.session_id = t.session_id
  AND m.user_id IS NOT DISTINCT FROM t.user_id
  AND m.app_name IS NOT DISTINCT FROM t.app_name
  AND m.invocation_id = t.invocation_id
`

func evalViewDatasetRef(fullTableID string, deniedColumns map[string]struct{}) (string, error) {
	if _, ok := deniedColumns[columnContent]; ok {
		return "", fmt.Errorf("cannot build eval view SQL when %q is in PayloadColumnDenylist", columnContent)
	}
	idx := strings.LastIndexByte(fullTableID, '.')
	if idx <= 0 || idx == len(fullTableID)-1 {
		return "", fmt.Errorf("fullTableID must be a dataset-qualified table name such as \"project.dataset.table\"; got %q", fullTableID)
	}
	return fullTableID[:idx], nil
}

func buildEvalViewCTEs(fullTableID string, deniedColumns map[string]struct{}, includeInvocation bool) string {
	contentPartsCol := "content_parts"
	if _, ok := deniedColumns[columnContentParts]; ok {
		contentPartsCol = "ARRAY<STRUCT<part_index INT64, mime_type STRING, text STRING, part_attributes STRING, storage_mode STRING>>[] AS content_parts"
	}
	attributesCol := "attributes"
	if _, ok := deniedColumns[columnAttributes]; ok {
		attributesCol = "CAST(NULL AS JSON) AS attributes"
	}
	latencyCol := "latency_ms"
	if _, ok := deniedColumns[columnLatencyMS]; ok {
		latencyCol = "CAST(NULL AS JSON) AS latency_ms"
	}
	scopeColumns := "session_id, user_id, app_name"
	qualifiedScopeColumns := "l.session_id, l.user_id, l.app_name"
	joinCondition := "l.session_id = d.session_id AND l.user_id IS NOT DISTINCT FROM d.user_id AND l.app_name IS NOT DISTINCT FROM d.app_name"
	if includeInvocation {
		scopeColumns += ", invocation_id"
		qualifiedScopeColumns += ", l.invocation_id"
		joinCondition += " AND l.invocation_id = d.invocation_id"
	}
	return strings.NewReplacer(
		"{full_table_ref}", "`"+fullTableID+"`",
		"{content_parts_column}", contentPartsCol,
		"{attributes_column}", attributesCol,
		"{latency_ms_column}", latencyCol,
		"{scope_columns}", scopeColumns,
		"{qualified_scope_columns}", qualifiedScopeColumns,
		"{agent_configs_join_condition}", joinCondition,
	).Replace(evalViewCTEsTemplate)
}

func buildEvalSessionViewSQL(fullTableID, viewPrefix string, deniedColumns map[string]struct{}) (string, error) {
	if viewPrefix == "" {
		viewPrefix = defaultViewPrefix
	}
	datasetRef, err := evalViewDatasetRef(fullTableID, deniedColumns)
	if err != nil {
		return "", err
	}
	viewRef := fmt.Sprintf("%s.%s_eval_session", datasetRef, viewPrefix)
	ctes := buildEvalViewCTEs(fullTableID, deniedColumns, false)
	return "CREATE OR REPLACE VIEW `" + viewRef + "` AS\n" + ctes + evalSessionViewSelectSQL, nil
}

func buildEvalTurnViewSQL(fullTableID, viewPrefix string, deniedColumns map[string]struct{}) (string, error) {
	if viewPrefix == "" {
		viewPrefix = defaultViewPrefix
	}
	datasetRef, err := evalViewDatasetRef(fullTableID, deniedColumns)
	if err != nil {
		return "", err
	}
	viewRef := fmt.Sprintf("%s.%s_eval_turn", datasetRef, viewPrefix)
	ctes := buildEvalViewCTEs(fullTableID, deniedColumns, true)
	return "CREATE OR REPLACE VIEW `" + viewRef + "` AS\n" + ctes + evalTurnViewSelectSQL, nil
}

// EvalViewDefinitions returns the CREATE OR REPLACE VIEW statements for the
// two pre-built agent evaluation views (<ViewPrefix>_eval_session and
// <ViewPrefix>_eval_turn) in deterministic order, honoring [Config.ViewPrefix]
// (empty means "v") and [Config.PayloadColumnDenylist].
//
// Sessions and turn windows are scoped by ADK's composite session identity
// (app_name from attributes $.adk.app_name, user_id, session_id). When
// "attributes" is in [Config.PayloadColumnDenylist], $.adk.app_name is
// unavailable (projected as NULL) and session scoping falls back to
// (user_id, session_id).
//
// A caller without a full Config can set only the fields that name the views,
// for example Config{ProjectID: p, DatasetID: d, TableName: t}.EvalViewDefinitions().
// It returns an error if an identifier is invalid, a view name would equal the
// events table name, or "content" is in [Config.PayloadColumnDenylist].
func (c Config) EvalViewDefinitions() ([]ViewDefinition, error) {
	c = applyDefaults(c)
	denied, err := parsePayloadColumnDenylist(c.PayloadColumnDenylist)
	if err != nil {
		return nil, err
	}
	return evalViewDefinitionsWithDenied(c.ProjectID, c.DatasetID, c.TableName, c.ViewPrefix, denied)
}

func evalViewDefinitionsWithDenied(projectID, datasetID, tableName, viewPrefix string, deniedColumns map[string]struct{}) ([]ViewDefinition, error) {
	if _, ok := deniedColumns[columnContent]; ok {
		return nil, fmt.Errorf("cannot build eval view SQL when %q is in PayloadColumnDenylist", columnContent)
	}
	if viewPrefix == "" {
		viewPrefix = defaultViewPrefix
	}
	if err := validateViewIdentifiers(projectID, datasetID, tableName, viewPrefix); err != nil {
		return nil, err
	}
	fullTableID := fmt.Sprintf("%s.%s.%s", projectID, datasetID, tableName)
	type evalSpec struct {
		suffix   string
		buildSQL func(fullTableID, viewPrefix string, deniedColumns map[string]struct{}) (string, error)
	}
	specs := []evalSpec{
		{suffix: "eval_session", buildSQL: buildEvalSessionViewSQL},
		{suffix: "eval_turn", buildSQL: buildEvalTurnViewSQL},
	}
	defs := make([]ViewDefinition, 0, len(specs))
	for _, spec := range specs {
		viewName := viewPrefix + "_" + spec.suffix
		if len(viewName) > maxBigQueryIdentifierLen {
			return nil, fmt.Errorf("BigQuery view name %q is longer than %d bytes; use a shorter view prefix", viewName, maxBigQueryIdentifierLen)
		}
		if strings.EqualFold(viewName, tableName) {
			return nil, fmt.Errorf("BigQuery view name %q collides with the events table %q; use another view prefix", viewName, tableName)
		}
		query, err := spec.buildSQL(fullTableID, viewPrefix, deniedColumns)
		if err != nil {
			return nil, err
		}
		defs = append(defs, ViewDefinition{
			Name:  viewName,
			Query: query,
		})
	}
	return defs, nil
}

// projectViewColumns drops derived view expressions whose SQL expression
// references a denied payload column (content, attributes, or latency_ms),
// matching _project_view_columns in adk-python.
func projectViewColumns(extraCols []string, deniedColumns map[string]struct{}) []string {
	if len(deniedColumns) == 0 || len(extraCols) == 0 {
		return extraCols
	}
	kept := make([]string, 0, len(extraCols))
	for _, expr := range extraCols {
		if referencesDeniedColumn(expr, deniedColumns) {
			continue
		}
		kept = append(kept, expr)
	}
	return kept
}

func referencesDeniedColumn(expr string, deniedColumns map[string]struct{}) bool {
	sqlPart := expr
	if idx := strings.LastIndex(expr, " AS "); idx >= 0 {
		sqlPart = expr[:idx]
	}
	sqlPart = stripSQLStringLiterals(sqlPart)
	for col := range deniedColumns {
		if containsIdentifierToken(sqlPart, col) {
			return true
		}
	}
	return false
}

// stripSQLStringLiterals replaces the contents of single-quoted SQL string
// literals (such as JSON path strings like '$.content') with empty quotes so
// that column identifier scans only inspect SQL identifiers.
func stripSQLStringLiterals(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inQuote := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if !inQuote {
			b.WriteByte(ch)
			if ch == '\'' {
				inQuote = true
			}
			continue
		}
		if ch == '\\' && i+1 < len(s) {
			i++
			continue
		}
		if ch == '\'' {
			b.WriteByte('\'')
			inQuote = false
		}
	}
	return b.String()
}

// containsIdentifierToken reports whether s contains ident as a standalone
// SQL identifier token (bounded by non-[A-Za-z0-9_] characters).
func containsIdentifierToken(s, ident string) bool {
	if ident == "" {
		return false
	}
	for start := 0; start < len(s); {
		idx := strings.Index(s[start:], ident)
		if idx < 0 {
			return false
		}
		pos := start + idx
		end := pos + len(ident)
		leftOK := pos == 0 || !isIdentChar(s[pos-1])
		rightOK := end == len(s) || !isIdentChar(s[end])
		if leftOK && rightOK {
			return true
		}
		start = pos + 1
	}
	return false
}

func isIdentChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_'
}

func isValidProjectID(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		b := s[i]
		if !isIdentChar(b) && b != '-' && b != '.' && b != ':' {
			return false
		}
	}
	return true
}

func isValidIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		b := s[i]
		if !isIdentChar(b) && b != '-' {
			return false
		}
	}
	return true
}

func isValidDatasetID(s string) bool {
	if s == "" || len(s) > maxBigQueryIdentifierLen {
		return false
	}
	for i := 0; i < len(s); i++ {
		if !isIdentChar(s[i]) {
			return false
		}
	}
	return true
}

func isValidTableName(s string) bool {
	if !utf8.ValidString(s) || strings.TrimSpace(s) == "" || len(s) > maxBigQueryIdentifierLen {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) && !unicode.IsNumber(r) && !unicode.IsMark(r) && !unicode.Is(unicode.Zs, r) && r != '_' && r != '-' {
			return false
		}
	}
	return true
}

func validateViewIdentifiers(projectID, datasetID, tableName, viewPrefix string) error {
	if !isValidProjectID(projectID) {
		return fmt.Errorf("invalid BigQuery project ID %q", projectID)
	}
	if !isValidDatasetID(datasetID) {
		return fmt.Errorf("invalid BigQuery dataset ID %q", datasetID)
	}
	if !isValidTableName(tableName) {
		return fmt.Errorf("invalid BigQuery table name %q", tableName)
	}
	if !isValidIdentifier(viewPrefix) {
		return fmt.Errorf("invalid BigQuery view prefix %q", viewPrefix)
	}
	return nil
}

// viewRunner abstracts DDL execution so that view creation and error handling
// can be tested deterministically without live BigQuery RPCs.
type viewRunner interface {
	RunViewDDL(ctx context.Context, viewName, query string) error
}

type bqViewRunner struct {
	// client is weak because the pluginRegistry keeps the runner for as long
	// as the plugin lives, and the client's transport or credentials can
	// refer to an object that holds the plugin. The plugin keeps the client
	// alive instead.
	client    weak.Pointer[bq.Client]
	projectID string
	datasetID string
}

func (r bqViewRunner) RunViewDDL(ctx context.Context, viewName, query string) error {
	client := r.client.Value()
	if client == nil {
		return errors.New("agentanalytics: BigQuery client is not configured for view creation")
	}
	if r.datasetID != "" && viewName != "" {
		if _, selectBody, ok := strings.Cut(query, " AS\n"); ok {
			var table *bq.Table
			if r.projectID != "" {
				table = client.DatasetInProject(r.projectID, r.datasetID).Table(viewName)
			} else {
				table = client.Dataset(r.datasetID).Table(viewName)
			}
			md, err := table.Metadata(ctx)
			if err == nil && md != nil {
				// CREATE OR REPLACE VIEW cannot replace a table that is not a
				// view, so say so instead of running a job that must fail. An
				// empty type is left for the job to decide.
				if md.Type != "" && md.Type != bq.ViewTable {
					return fmt.Errorf("cannot replace existing BigQuery %s %q with a view", md.Type, viewName)
				}
				if md.Type == bq.ViewTable && strings.TrimSpace(md.ViewQuery) == strings.TrimSpace(selectBody) {
					return nil
				}
			}
		}
	}
	// Canceling ctx stops the request or client wait; a job already accepted by
	// BigQuery is not canceled on the server.
	job, err := client.Query(query).Run(ctx)
	if err != nil {
		return err
	}
	js := job.LastStatus()
	if js == nil || !js.Done() {
		js, err = job.Wait(ctx)
		if err != nil {
			return err
		}
	}
	return js.Err()
}

// syncAnalyticsViews executes CREATE OR REPLACE VIEW for every definition in
// defs. When logf is not nil (background startup path with a non-default
// Logger), failures are logged through logf rather than failing plugin
// initialization.
func syncAnalyticsViews(ctx context.Context, runner viewRunner, defs []ViewDefinition, logf func(format string, args ...any)) error {
	return syncAnalyticsViewsWithRetryHint(ctx, runner, defs, logf, "SyncViews")
}

func syncAnalyticsViewsWithRetryHint(ctx context.Context, runner viewRunner, defs []ViewDefinition, logf func(format string, args ...any), retryMethod string) error {
	if retryMethod == "" {
		retryMethod = "SyncViews"
	}
	var (
		errs    []error
		created int
		failed  int
	)
	for i, def := range defs {
		if err := ctx.Err(); err != nil {
			skippedNames := make([]string, 0, len(defs)-i)
			for _, rem := range defs[i:] {
				skippedNames = append(skippedNames, rem.Name)
			}
			errs = append(errs, fmt.Errorf("view creation canceled before %s (skipped: %s): %w", def.Name, strings.Join(skippedNames, ", "), err))
			break
		}
		ddlCtx, cancel := context.WithTimeout(ctx, viewDDLTimeout)
		err := runner.RunViewDDL(ddlCtx, def.Name, def.Query)
		cancel()
		if err == nil || isConflictError(err) {
			created++
			continue
		}
		// Once ctx is done, the error ends the pass whatever it says: a
		// deadline is not a Canceled error, and a transport need not wrap ctx's
		// error. A per-view timeout leaves ctx alive and is a failure.
		if parentErr := ctx.Err(); parentErr != nil {
			if errors.Is(err, parentErr) {
				errs = append(errs, fmt.Errorf("view creation canceled during %s: %w", def.Name, err))
			} else {
				errs = append(errs, fmt.Errorf("view creation canceled during %s: %w (%v)", def.Name, parentErr, err))
			}
			break
		}
		failed++
		if logf != nil && !errors.Is(err, context.Canceled) && status.Code(err) != codes.Canceled {
			logf("agentanalytics: failed to create view %s: %v", def.Name, err)
		}
		errs = append(errs, fmt.Errorf("failed to create view %s: %w", def.Name, err))
	}
	if logf != nil && !errors.Is(ctx.Err(), context.Canceled) && failed > 0 {
		logf("agentanalytics: view creation finished: %d created, %d failed of %d; call %s to retry", created, failed, len(defs), retryMethod)
	}
	return errors.Join(errs...)
}

var errPluginClosed = errors.New("agentanalytics: plugin is closed")

// SyncViews creates or replaces all per-event-type BigQuery analytics views
// (and the two evaluation views when [Config.CreateEvalViews] is true)
// synchronously using ctx. It runs even when DisableCreateViews is set,
// permanently cancels any in-flight startup view creation pass, and returns
// the joined per-view errors with 409 Conflict / AlreadyExists responses
// ignored. When one view group (per-event or evaluation) fails definition or
// configuration validation while the other group is valid, the valid views
// still run and their result is joined with the group validation error; if no
// valid views remain, SyncViews returns the validation error without
// canceling any in-flight startup pass. Only canceling ctx or closing the
// plugin stops waiting on the current DDL request and stops starting further
// view jobs; exceeding the 30-second per-view timeout stops waiting on and fails
// only that view while the pass continues with the remaining views (and in
// either case, a job already submitted to BigQuery may still finish on the
// server). If Close is called on the plugin, a SyncViews call holding the view
// gate is canceled and Close waits for its current step to return, a SyncViews
// call waiting for the gate returns immediately, and every closed path returns
// an "agentanalytics: plugin is closed" error.
func (p *BigQueryAgentAnalyticsPlugin) SyncViews(ctx context.Context) error {
	if p == nil || p.pluginState == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	// The plugin keeps the BigQuery client alive, and the runner refers to it
	// only weakly.
	defer runtime.KeepAlive(p.Plugin)
	return p.pluginState.SyncViews(ctx)
}

// SyncEvalViews creates or replaces the two pre-built agent evaluation
// BigQuery views (<ViewPrefix>_eval_session and <ViewPrefix>_eval_turn)
// synchronously using ctx. It runs even when [Config.CreateEvalViews] is false.
//
// Configuration and view definitions are validated first; if "content" is in
// [Config.PayloadColumnDenylist], [Config.EventDenylist] or
// [Config.EventAllowlist] filters out a required trace event, or the view
// names fail validation or collide with the events table, SyncEvalViews returns
// an error without canceling any in-flight startup view creation pass. Once
// validated, it permanently cancels any in-flight startup pass and creates or
// replaces only the two evaluation views (ignoring 409 Conflict / AlreadyExists
// responses). If [Config.DisableCreateViews] is false and the canceled startup
// pass had not finished creating the per-event views, call
// [BigQueryAgentAnalyticsPlugin.SyncViews] afterwards to create the remaining
// per-event views. It shares the view gate and Close preemption semantics of
// [BigQueryAgentAnalyticsPlugin.SyncViews].
func (p *BigQueryAgentAnalyticsPlugin) SyncEvalViews(ctx context.Context) error {
	if p == nil || p.pluginState == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	defer runtime.KeepAlive(p.Plugin)
	return p.pluginState.SyncEvalViews(ctx)
}

func (s *pluginState) SyncViews(ctx context.Context) error {
	if s == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	var defs []ViewDefinition
	var errs []error
	if s.defsErr != nil {
		errs = append(errs, s.defsErr)
	} else {
		defs = append(defs, s.defs...)
	}
	if s.createEvalViews {
		if evalErr := cmp.Or(s.evalConfigErr, s.evalDefsErr); evalErr != nil {
			errs = append(errs, evalErr)
		} else {
			defs = append(defs, s.evalDefs...)
		}
	}
	deferredErr := errors.Join(errs...)
	if len(defs) == 0 {
		return s.syncViewsWithDefs(ctx, nil, deferredErr, nil, nil)
	}
	return s.syncViewsWithDefs(ctx, defs, nil, nil, deferredErr)
}

func (s *pluginState) SyncEvalViews(ctx context.Context) error {
	if s == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	return s.syncViewsWithDefs(ctx, s.evalDefs, s.evalDefsErr, s.evalConfigErr, nil)
}

func (s *pluginState) syncViewsWithDefs(ctx context.Context, defs []ViewDefinition, defsErr, configErr, deferredErr error) error {
	if s == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	if !s.enabled {
		return errors.New("agentanalytics: plugin is disabled")
	}
	s.viewsMu.Lock()
	closed, runner := s.closed, s.runner
	s.viewsMu.Unlock()
	if closed {
		return errPluginClosed
	}
	if configErr != nil {
		return configErr
	}
	if runner == nil {
		return errors.New("agentanalytics: BigQuery client is not configured for view creation")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if defsErr != nil {
		return defsErr
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	if s.lifeCtx != nil {
		stop := context.AfterFunc(s.lifeCtx, cancel)
		defer stop()
	}

	if err := s.preemptAsyncViewCreation(ctx); err != nil {
		if s.isClosed() {
			return errPluginClosed
		}
		return err
	}
	if err := s.acquireViewGate(ctx); err != nil {
		if s.isClosed() {
			return errPluginClosed
		}
		return err
	}
	defer s.releaseViewGate()
	if s.isClosed() {
		return errPluginClosed
	}
	syncErr := syncAnalyticsViews(ctx, runner, defs, nil)
	if s.isClosed() {
		return errPluginClosed
	}
	return errors.Join(syncErr, deferredErr)
}

func (s *pluginState) ensureViewGate() chan struct{} {
	s.viewsMu.Lock()
	defer s.viewsMu.Unlock()
	if s.viewGate == nil {
		s.viewGate = make(chan struct{}, 1)
		s.viewGate <- struct{}{}
	}
	return s.viewGate
}

func (s *pluginState) acquireViewGate(ctx context.Context) error {
	gate := s.ensureViewGate()
	select {
	case <-gate:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *pluginState) releaseViewGate() {
	s.ensureViewGate() <- struct{}{}
}

func (s *pluginState) isClosed() bool {
	s.viewsMu.Lock()
	defer s.viewsMu.Unlock()
	return s.closed
}

// startAsyncViewCreation runs a view creation pass for defs in the background,
// logging failures through logf if it is not nil. The pass keeps client
// reachable until it ends, because the runner refers to it only weakly.
func (s *pluginState) startAsyncViewCreation(parentCtx context.Context, defs []ViewDefinition, logf func(format string, args ...any), client *bq.Client) {
	s.startAsyncViewCreationWithRetryHint(parentCtx, defs, logf, "SyncViews", client)
}

func (s *pluginState) startAsyncViewCreationWithRetryHint(parentCtx context.Context, defs []ViewDefinition, logf func(format string, args ...any), retryMethod string, client *bq.Client) {
	viewsCtx, viewsCancel := context.WithCancel(context.WithoutCancel(parentCtx))
	done := make(chan struct{})
	s.viewsMu.Lock()
	s.viewsCancel = viewsCancel
	s.asyncViewsDone = done
	runner := s.runner
	s.viewsMu.Unlock()

	go func() {
		defer runtime.KeepAlive(client)
		defer close(done)
		// viewsCancel keeps parentCtx, whose values can refer to the
		// caller's objects, so the pass forgets it once it ends, before
		// anyone waiting on done goes on.
		defer func() {
			s.viewsMu.Lock()
			s.viewsCancel = nil
			s.asyncViewsDone = nil
			s.viewsMu.Unlock()
		}()
		defer viewsCancel()
		if err := s.acquireViewGate(viewsCtx); err != nil {
			return
		}
		defer s.releaseViewGate()
		if s.isClosed() || viewsCtx.Err() != nil {
			return
		}
		_ = syncAnalyticsViewsWithRetryHint(viewsCtx, runner, defs, logf, retryMethod)
	}()
}

func (s *pluginState) preemptAsyncViewCreation(ctx context.Context) error {
	s.viewsMu.Lock()
	cancel := s.viewsCancel
	done := s.asyncViewsDone
	s.viewsMu.Unlock()

	if cancel != nil {
		cancel()
	}
	if done != nil {
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (s *pluginState) stopAsyncViewCreation() {
	if s == nil {
		return
	}
	s.viewsMu.Lock()
	s.closed = true
	lifeCancel := s.lifeCancel
	cancel := s.viewsCancel
	done := s.asyncViewsDone
	// No new view DDL job is started after Close, and the registry keeps s for
	// as long as the plugin lives. An in-flight pass uses its own runner copy;
	// Close cancels its client wait below and waits for the pass to return
	// (a job already submitted to BigQuery may still finish on the server).
	s.runner = nil
	s.viewsMu.Unlock()

	if lifeCancel != nil {
		lifeCancel()
	}
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
	gate := s.ensureViewGate()
	<-gate
	gate <- struct{}{}
}
