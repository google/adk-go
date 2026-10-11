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
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"

	bq "cloud.google.com/go/bigquery"
)

const viewDDLTimeout = 30 * time.Second

// ViewDefinition describes a BigQuery view over the agent events table for a
// single event type.
type ViewDefinition struct {
	// Name is the unqualified BigQuery view name (for example "v_llm_request").
	Name string
	// EventType is the event_type value filtered by the view (for example [EventLLMRequest]).
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
		eventType: EventEventCompaction,
		extraCols: []string{
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS start_seconds`,
			`COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) AS end_seconds`,
			`TIMESTAMP_MICROS(CAST(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.start_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.start_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) * 1000000 AS INT64)) AS window_start`,
			`TIMESTAMP_MICROS(CAST(COALESCE(SAFE_CAST(JSON_VALUE(content, '$.end_timestamp') AS FLOAT64), UNIX_MICROS(SAFE_CAST(REGEXP_REPLACE(JSON_VALUE(content, '$.end_timestamp'), r'(\.\d{6})\d+', r'\1') AS TIMESTAMP)) / 1e6) * 1000000 AS INT64)) AS window_end`,
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

// ViewDefinitions returns the deterministic list of per-event-type BigQuery
// view definitions for the given project, dataset, table, and view prefix
// without column denylist filtering. For definitions that honor
// [Config.PayloadColumnDenylist], use [ViewDefinitionsForConfig] or
// [Config.ViewDefinitions]. If viewPrefix is empty, "v" is used.
func ViewDefinitions(projectID, datasetID, tableName, viewPrefix string) ([]ViewDefinition, error) {
	return viewDefinitionsWithDenied(projectID, datasetID, tableName, viewPrefix, nil)
}

// ViewDefinitionsForConfig returns the CREATE OR REPLACE VIEW statements that
// the plugin runs for c, with [Config.PayloadColumnDenylist] applied and
// [Config.ViewPrefix] defaulting to "v".
func ViewDefinitionsForConfig(c Config) ([]ViewDefinition, error) {
	c = applyDefaults(c)
	denied, err := parsePayloadColumnDenylist(c.PayloadColumnDenylist)
	if err != nil {
		return nil, err
	}
	return viewDefinitionsWithDenied(c.ProjectID, c.DatasetID, c.TableName, c.ViewPrefix, denied)
}

// ViewDefinitions returns the CREATE OR REPLACE VIEW statements that the
// plugin runs for c, honoring [Config.ViewPrefix] and
// [Config.PayloadColumnDenylist].
func (c Config) ViewDefinitions() ([]ViewDefinition, error) {
	return ViewDefinitionsForConfig(c)
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

func isValidTableName(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-' && r != ' ' {
			return false
		}
	}
	return true
}

func validateViewIdentifiers(projectID, datasetID, tableName, viewPrefix string) error {
	if !isValidProjectID(projectID) {
		return fmt.Errorf("invalid BigQuery project ID %q", projectID)
	}
	if !isValidIdentifier(datasetID) {
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
	client    *bq.Client
	projectID string
	datasetID string
}

func (r bqViewRunner) RunViewDDL(ctx context.Context, viewName, query string) error {
	if r.datasetID != "" && viewName != "" {
		if _, selectBody, ok := strings.Cut(query, " AS\n"); ok {
			var table *bq.Table
			if r.projectID != "" {
				table = r.client.DatasetInProject(r.projectID, r.datasetID).Table(viewName)
			} else {
				table = r.client.Dataset(r.datasetID).Table(viewName)
			}
			md, err := table.Metadata(ctx)
			if err == nil && md != nil && md.Type == bq.ViewTable && strings.TrimSpace(md.ViewQuery) == strings.TrimSpace(selectBody) {
				return nil
			}
		}
	}
	job, err := r.client.Query(query).Run(ctx)
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
// defs. When logErrors is true (background startup path), failures are logged
// via config.Logger rather than failing plugin initialization.
func syncAnalyticsViews(ctx context.Context, runner viewRunner, defs []ViewDefinition, logErrors bool, config Config) error {
	var (
		errs         []error
		created      int
		failed       int
		skippedNames []string
	)
	for i, def := range defs {
		if err := ctx.Err(); err != nil {
			skippedNames = make([]string, 0, len(defs)-i)
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
		if ctx.Err() != nil && errors.Is(err, context.Canceled) {
			skippedNames = append(skippedNames, def.Name)
			for _, rem := range defs[i+1:] {
				skippedNames = append(skippedNames, rem.Name)
			}
			errs = append(errs, fmt.Errorf("view creation canceled during %s: %w", def.Name, err))
			break
		}
		failed++
		if logErrors && !errors.Is(err, context.Canceled) {
			config.logf("agentanalytics: failed to create view %s: %v", def.Name, err)
		}
		errs = append(errs, fmt.Errorf("failed to create view %s: %w", def.Name, err))
	}
	skipped := len(skippedNames)
	if logErrors && !errors.Is(ctx.Err(), context.Canceled) && (failed > 0 || skipped > 0) {
		if skipped > 0 {
			config.logf("agentanalytics: view creation finished: %d created, %d failed, %d skipped of %d (skipped: %s); call SyncViews to retry", created, failed, skipped, len(defs), strings.Join(skippedNames, ", "))
		} else {
			config.logf("agentanalytics: view creation finished: %d created, %d failed, %d skipped of %d; call SyncViews to retry", created, failed, skipped, len(defs))
		}
	}
	return errors.Join(errs...)
}

// SyncViews creates or replaces all per-event-type BigQuery analytics views
// synchronously using ctx. It runs even when CreateViews is disabled,
// permanently cancels any in-flight startup view creation pass, and returns
// the joined per-view errors with 409 Conflict / AlreadyExists responses
// ignored. Calling Close on the plugin cancels an in-flight SyncViews and
// waits for it to return.
func (p *BigQueryAgentAnalyticsPlugin) SyncViews(ctx context.Context) error {
	if p == nil || p.pluginState == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	return p.pluginState.SyncViews(ctx)
}

func (s *pluginState) SyncViews(ctx context.Context) error {
	if s == nil {
		return errors.New("agentanalytics: nil plugin")
	}
	if !s.config.Enabled {
		return errors.New("agentanalytics: plugin is disabled")
	}
	if s.runner == nil {
		return errors.New("agentanalytics: BigQuery client is not configured for view creation")
	}
	if s.isClosed() {
		return errors.New("agentanalytics: plugin is closed")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	defs, err := ViewDefinitionsForConfig(s.config)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	if s.lifeCtx != nil {
		stop := context.AfterFunc(s.lifeCtx, cancel)
		defer stop()
	}

	if err := s.preemptAsyncViewCreation(ctx); err != nil {
		if s.isClosed() {
			return errors.New("agentanalytics: plugin is closed")
		}
		return err
	}
	if err := s.acquireViewGate(ctx); err != nil {
		if s.isClosed() {
			return errors.New("agentanalytics: plugin is closed")
		}
		return err
	}
	defer s.releaseViewGate()
	if s.isClosed() {
		return errors.New("agentanalytics: plugin is closed")
	}
	return syncAnalyticsViews(ctx, s.runner, defs, false, s.config)
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

func (s *pluginState) startAsyncViewCreation(parentCtx context.Context, defs []ViewDefinition) {
	viewsCtx, viewsCancel := context.WithCancel(context.WithoutCancel(parentCtx))
	done := make(chan struct{})
	s.viewsMu.Lock()
	s.viewsCancel = viewsCancel
	s.asyncViewsDone = done
	s.viewsMu.Unlock()

	go func() {
		defer close(done)
		defer viewsCancel()
		if err := s.acquireViewGate(viewsCtx); err != nil {
			return
		}
		defer s.releaseViewGate()
		if s.isClosed() || viewsCtx.Err() != nil {
			return
		}
		_ = syncAnalyticsViews(viewsCtx, s.runner, defs, true, s.config)
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
