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

// Package agentanalytics implements the BigQuery Agent Analytics Plugin.
package agentanalytics

import (
	"cmp"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"reflect"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

const pluginName = "bigquery_agent_analytics"

// Event types written to the event_type column. They match the names written
// by adk-python, so that rows from either language can be queried together.
// This version of the plugin does not emit all of them yet.
const (
	EventUserMessageReceived = "USER_MESSAGE_RECEIVED"
	EventInvocationStarting  = "INVOCATION_STARTING"
	EventInvocationCompleted = "INVOCATION_COMPLETED"
	EventAgentStarting       = "AGENT_STARTING"
	EventAgentCompleted      = "AGENT_COMPLETED"
	EventAgentError          = "AGENT_ERROR"
	EventInvocationError     = "INVOCATION_ERROR"
	EventLLMRequest          = "LLM_REQUEST"
	// EventLLMResponse is written once per model call, on the final
	// non-partial response. Streamed chunks with Partial set only feed
	// latency_ms.time_to_first_token_ms on that final row, and only the
	// first chunk with generated content does.
	EventLLMResponse               = "LLM_RESPONSE"
	EventLLMError                  = "LLM_ERROR"
	EventToolStarting              = "TOOL_STARTING"
	EventToolCompleted             = "TOOL_COMPLETED"
	EventToolError                 = "TOOL_ERROR"
	EventStateDelta                = "STATE_DELTA"
	EventAgentTransfer             = "AGENT_TRANSFER"
	EventEventCompaction           = "EVENT_COMPACTION"
	EventAgentStateCheckpoint      = "AGENT_STATE_CHECKPOINT"
	EventToolPaused                = "TOOL_PAUSED"
	EventHITLCredentialRequest     = "HITL_CREDENTIAL_REQUEST"
	EventHITLConfirmationRequest   = "HITL_CONFIRMATION_REQUEST"
	EventHITLInputRequest          = "HITL_INPUT_REQUEST"
	EventHITLCredentialCompleted   = "HITL_CREDENTIAL_REQUEST_COMPLETED"
	EventHITLConfirmationCompleted = "HITL_CONFIRMATION_REQUEST_COMPLETED"
	EventHITLInputCompleted        = "HITL_INPUT_REQUEST_COMPLETED"
	EventA2AInteraction            = "A2A_INTERACTION"
	EventAgentResponse             = "AGENT_RESPONSE"
	EventNodeOutput                = "NODE_OUTPUT"
	EventNodeError                 = "NODE_ERROR"
)

const (
	// legacyEventType is written for every session event passed to the
	// OnEventCallback.
	// TODO(haiyuan): replace it with the per-event rows that adk-python
	// writes (STATE_DELTA, AGENT_TRANSFER, HITL_*, ...).
	legacyEventType = "EVENT"

	// adkEnvelopeSchemaVersion is the version of the attributes.adk object.
	adkEnvelopeSchemaVersion = "1"

	// unknownToolOrigin is written to content.tool_origin of tool events.
	// TODO(haiyuan): classify tools (LOCAL, MCP, A2A, ...) as adk-python does.
	unknownToolOrigin = "UNKNOWN"

	userAuthor = "user"

	scopeKindNodeRun      = "node_run"
	scopeKindFunctionCall = "function_call"

	adkModulePath = "google.golang.org/adk/v2"
)

// knownEventTypes holds the event types that EventAllowlist and
// EventDenylist accept without a warning: the Event constants and the legacy
// EVENT type.
var knownEventTypes = newSet(
	EventUserMessageReceived, EventInvocationStarting, EventInvocationCompleted, EventInvocationError,
	EventAgentStarting, EventAgentCompleted, EventAgentError,
	EventLLMRequest, EventLLMResponse, EventLLMError,
	EventToolStarting, EventToolCompleted, EventToolError,
	EventStateDelta, EventAgentTransfer, EventEventCompaction, EventAgentStateCheckpoint, EventToolPaused,
	EventHITLCredentialRequest, EventHITLConfirmationRequest, EventHITLInputRequest,
	EventHITLCredentialCompleted, EventHITLConfirmationCompleted, EventHITLInputCompleted,
	EventA2AInteraction, EventAgentResponse, EventNodeOutput, EventNodeError,
	legacyEventType,
)

// NewBigQueryAgentAnalyticsPlugin creates a newly configured analytics plugin with default config.
//
// Register the plugin before any plugin whose callbacks return replacement
// events, model responses, tool results or agent content. ADK runs the
// callbacks of each kind in registration order and stops at the first plugin
// that returns a non-nil result or an error, so a plugin that is registered
// earlier hides what it replaces from this one. This plugin then misses the
// rows of what was hidden and, because it infers failures from the callbacks
// that it saw, it can end a successful run with AGENT_ERROR and
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED.
func NewBigQueryAgentAnalyticsPlugin(
	ctx context.Context,
	projectID string,
	datasetID string,
	tableID string,
) (*baseplugin.Plugin, error) {
	config := DefaultConfig()
	config.ProjectID = projectID
	config.DatasetID = datasetID
	config.TableName = tableID
	return NewBigQueryAgentAnalyticsPluginWithConfig(ctx, config)
}

// NewBigQueryAgentAnalyticsPluginWithConfig creates a newly configured analytics plugin.
//
// Register the plugin before any plugin whose callbacks return replacement
// events, model responses, tool results or agent content. ADK runs the
// callbacks of each kind in registration order and stops at the first plugin
// that returns a non-nil result or an error, so a plugin that is registered
// earlier hides what it replaces from this one. This plugin then misses the
// rows of what was hidden and, because it infers failures from the callbacks
// that it saw, it can end a successful run with AGENT_ERROR and
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED.
func NewBigQueryAgentAnalyticsPluginWithConfig(
	ctx context.Context,
	config Config,
) (*baseplugin.Plugin, error) {
	if !config.Enabled {
		// Return empty plugin wrapped
		return baseplugin.New(baseplugin.Config{Name: pluginName})
	}
	config = applyDefaults(config)
	// Reject invalid settings before any client is created.
	if _, err := validateConfig(config); err != nil {
		return nil, err
	}

	// Set up BigQuery data client for generic operations like ensure table exists
	// We use the normal BigQuery client here
	bqClient, err := bq.NewClient(ctx, config.ProjectID, config.ClientOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create bigquery client: %w", err)
	}
	defer func() {
		if err := bqClient.Close(); err != nil {
			config.logf("failed to close bigquery client: %v", err)
		}
	}()

	// Setup Storage Write Client
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, config.ClientOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to create BigQuery write client: %w", err)
	}

	return NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
}

// NewBigQueryAgentAnalyticsPluginWithClients creates a newly configured analytics plugin using the provided BigQuery clients.
//
// It creates the events table if it does not exist and, unless
// DisableSchemaUpgrade is set, adds any missing columns to an existing table.
// ctx bounds this table setup; background writes use only its values (see
// [NewBatchProcessor]).
//
// The plugin owns writeClient and closes it when the plugin is closed. It
// also closes writeClient right away when config.Enabled is false, and before
// it returns any error, such as an invalid config or a table that cannot be
// read, created or upgraded.
//
// Register the plugin before any plugin whose callbacks return replacement
// events, model responses, tool results or agent content. ADK runs the
// callbacks of each kind in registration order and stops at the first plugin
// that returns a non-nil result or an error, so a plugin that is registered
// earlier hides what it replaces from this one. This plugin then misses the
// rows of what was hidden and, because it infers failures from the callbacks
// that it saw, it can end a successful run with AGENT_ERROR and
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED.
func NewBigQueryAgentAnalyticsPluginWithClients(
	ctx context.Context,
	config Config,
	bqClient *bq.Client,
	writeClient *bqstorage.BigQueryWriteClient,
) (*baseplugin.Plugin, error) {
	if !config.Enabled {
		// The disabled plugin never writes, so it does not keep writeClient.
		closeWriteClient(writeClient, config)
		// Return empty plugin wrapped
		return baseplugin.New(baseplugin.Config{Name: pluginName})
	}
	config = applyDefaults(config)
	deniedColumns, err := validateConfig(config)
	if err != nil {
		closeWriteClient(writeClient, config)
		return nil, err
	}
	validateEventFilterLists(config)

	// Ensure table exists and has all required columns (done once per instance)
	tableSchema := projectEventsSchema(deniedColumns)
	tableRef := bqClient.Dataset(config.DatasetID).Table(config.TableName)
	if err := ensureTable(ctx, tableRef, tableSchema, config); err != nil {
		closeWriteClient(writeClient, config)
		return nil, err
	}

	// Determine destination stream
	streamName := fmt.Sprintf("projects/%s/datasets/%s/tables/%s/streams/_default", config.ProjectID, config.DatasetID, config.TableName)

	processor, err := NewBatchProcessor(ctx, writeClient, streamName, config)
	if err != nil {
		closeWriteClient(writeClient, config)
		return nil, fmt.Errorf("failed to initialize batch processor: %w", err)
	}
	processor.Start()

	traces := newTraceManager()
	traces.logf = config.logf
	p := &analyticsPlugin{
		config:             config,
		processor:          processor,
		allowlist:          newSet(config.EventAllowlist...),
		denylist:           newSet(config.EventDenylist...),
		deniedColumns:      deniedColumns,
		finalResponseTools: newSet(config.FinalResponseToolNames...),
		customMetadata:     newCustomMetadataFilter(config.CustomMetadataAllowlist),
		traces:             traces,
	}

	return baseplugin.New(baseplugin.Config{
		Name: pluginName,

		OnUserMessageCallback: p.onUserMessage,
		BeforeRunCallback:     p.beforeRun,
		AfterRunCallback:      p.afterRun,
		OnEventCallback:       p.onEvent,

		BeforeAgentCallback: p.beforeAgent,
		AfterAgentCallback:  p.afterAgent,

		BeforeModelCallback:  p.beforeModel,
		AfterModelCallback:   p.afterModel,
		OnModelErrorCallback: p.onModelError,

		BeforeToolCallback:  p.beforeTool,
		AfterToolCallback:   p.afterTool,
		OnToolErrorCallback: p.onToolError,

		CloseFunc: func() error {
			processor.Close()
			if err := writeClient.Close(); err != nil {
				return fmt.Errorf("failed to close BigQuery write client: %w", err)
			}
			return nil
		},
	})
}

// ensureTable makes sure the events table exists. A missing table is created
// with every column in schema. An existing table, including one that another
// writer creates concurrently, gains any columns in schema that it lacks.
// With DisableSchemaUpgrade, the schema of an existing table is neither
// inspected nor upgraded, so a table that lacks a column in schema is not
// detected here: the plugin starts, and every write to the table fails. Any
// other failure to read, create or upgrade the table is returned.
func ensureTable(ctx context.Context, tableRef *bq.Table, schema bq.Schema, config Config) error {
	meta, err := tableRef.Metadata(ctx)
	if hasHTTPStatus(err, http.StatusNotFound) {
		config.logf("Table %s not found. Creating it...", config.TableName)
		err = tableRef.Create(ctx, &bq.TableMetadata{
			Schema: schema,
			Labels: map[string]string{
				"adk_schema_version": SchemaVersion,
			},
			// Partition by DAY on timestamp for parity with adk-python and adk-java.
			TimePartitioning: &bq.TimePartitioning{
				Field: timestampField,
				Type:  bq.DayPartitioningType,
			},
			Clustering: &bq.Clustering{
				Fields: config.ClusteringFields,
			},
		})
		if err == nil {
			return nil
		}
		if !hasHTTPStatus(err, http.StatusConflict) {
			return fmt.Errorf("failed to create BigQuery table %s: %w", config.TableName, err)
		}
		// Another writer created the table first, possibly with an older
		// schema; handle it like any existing table.
		if config.DisableSchemaUpgrade {
			return nil
		}
		meta, err = tableRef.Metadata(ctx)
	}
	if err != nil {
		return fmt.Errorf("failed to get BigQuery table %s metadata: %w", config.TableName, err)
	}
	if config.DisableSchemaUpgrade {
		return nil
	}
	if err := upgradeExistingTable(ctx, tableRef, meta, schema); err != nil {
		return fmt.Errorf("failed to upgrade BigQuery table %s schema: %w", config.TableName, err)
	}
	return nil
}

// hasHTTPStatus reports whether err is a BigQuery API error with the given
// HTTP status code.
func hasHTTPStatus(err error, code int) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) && apiErr.Code == code
}

// closeWriteClient closes a write client that the plugin will not use,
// logging any error.
func closeWriteClient(writeClient *bqstorage.BigQueryWriteClient, config Config) {
	if writeClient == nil {
		return
	}
	if err := writeClient.Close(); err != nil {
		config.logf("failed to close BigQuery write client: %v", err)
	}
}

func upgradeExistingTable(ctx context.Context, tableRef *bq.Table, meta *bq.TableMetadata, required bq.Schema) error {
	merged, changed, err := upgradeTableSchema(meta.Schema, required)
	if err != nil || !changed {
		return err
	}
	update := bq.TableMetadataToUpdate{Schema: merged}
	update.SetLabel("adk_schema_version", SchemaVersion)
	_, err = tableRef.Update(ctx, update, meta.ETag)
	if !isPreconditionFailed(err) {
		return err
	}
	latest, metaErr := tableRef.Metadata(ctx)
	if metaErr != nil {
		return err
	}
	merged, changed, err = upgradeTableSchema(latest.Schema, required)
	if err != nil || !changed {
		return err
	}
	retryUpdate := bq.TableMetadataToUpdate{Schema: merged}
	retryUpdate.SetLabel("adk_schema_version", SchemaVersion)
	_, err = tableRef.Update(ctx, retryUpdate, latest.ETag)
	return err
}

func isPreconditionFailed(err error) bool {
	if err == nil {
		return false
	}
	var apiErr *googleapi.Error
	if errors.As(err, &apiErr) && apiErr.Code == http.StatusPreconditionFailed {
		return true
	}
	return strings.Contains(err.Error(), "412")
}

// validateConfig rejects settings that cannot take effect and returns the
// PayloadColumnDenylist as a set.
func validateConfig(c Config) (map[string]struct{}, error) {
	switch {
	case c.BatchSize < 1, c.QueueMaxSize < 1:
		return nil, errors.New("invalid config: BatchSize and QueueMaxSize must be >= 1")
	case c.BatchFlushIntv <= 0, c.ShutdownTimeout <= 0:
		return nil, errors.New("invalid config: BatchFlushIntv and ShutdownTimeout must be > 0")
	case c.MaxContentLen == 0, c.MaxContentLen < -1:
		return nil, errors.New("invalid config: MaxContentLen must be -1 or >= 1")
	case c.RetryConfig.MaxRetries < 0, c.RetryConfig.InitialDelay < 0, c.RetryConfig.Multiplier < 1, c.RetryConfig.MaxDelay < c.RetryConfig.InitialDelay:
		return nil, fmt.Errorf("invalid config: invalid RetryConfig: needs MaxRetries >= 0, InitialDelay >= 0, Multiplier >= 1 and MaxDelay >= InitialDelay, got %+v", c.RetryConfig)
	}
	denied, err := parsePayloadColumnDenylist(c.PayloadColumnDenylist)
	if err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if _, ok := denied[columnAttributes]; ok && len(c.CustomMetadataAllowlist) > 0 {
		return nil, errors.New("invalid config: CustomMetadataAllowlist cannot be used when PayloadColumnDenylist contains \"attributes\"")
	}
	return denied, nil
}

// validateEventFilterLists warns about entries of EventAllowlist and
// EventDenylist that name no known event type and are likely typos.
func validateEventFilterLists(c Config) {
	for _, list := range [][]string{c.EventAllowlist, c.EventDenylist} {
		for _, eventType := range list {
			if _, ok := knownEventTypes[eventType]; !ok {
				c.logf("agentanalytics: unknown event_type %q in filter configuration (possible typo)", eventType)
			}
		}
	}
}

func newSet(items ...string) map[string]struct{} {
	s := make(map[string]struct{}, len(items))
	for _, item := range items {
		s[item] = struct{}{}
	}
	return s
}

// analyticsPlugin holds the state shared by the plugin callbacks.
type analyticsPlugin struct {
	config             Config
	processor          *BatchProcessor
	allowlist          map[string]struct{}
	denylist           map[string]struct{}
	deniedColumns      map[string]struct{}
	finalResponseTools map[string]struct{}
	customMetadata     customMetadataFilter
	traces             *traceManager
}

// eventData carries the event-specific parts of a row.
type eventData struct {
	// span places the row in the trace of its invocation.
	span rowSpan
	// content is the raw payload for the content and content_parts columns.
	content any
	// contentTruncated reports that content was truncated while it was built.
	contentTruncated bool
	// attributes are event-specific attributes, written next to the common
	// ones.
	attributes map[string]any
	// errMsg is written to error_message.
	errMsg string
	// failed sets the status column to "ERROR".
	failed         bool
	functionCallID string
	// sourceEvent is the session event the row describes, if any.
	sourceEvent    *session.Event
	author         string
	customMetadata map[string]any
}

// shouldLogEvent applies EventDenylist, then EventAllowlist if it is set.
func (p *analyticsPlugin) shouldLogEvent(eventType string) bool {
	if _, ok := p.denylist[eventType]; ok {
		return false
	}
	if len(p.allowlist) > 0 {
		_, ok := p.allowlist[eventType]
		return ok
	}
	return true
}

func (p *analyticsPlugin) denied(column string) bool {
	_, ok := p.deniedColumns[column]
	return ok
}

// recoverCallback logs and suppresses panics raised inside plugin callbacks so
// that analytics logging never breaks the agent run.
func (p *analyticsPlugin) recoverCallback(eventType string) {
	if r := recover(); r != nil {
		p.config.logf("agentanalytics: dropped %s event: recovered panic of type %T", eventType, r)
	}
}

// logEvent runs spanFn to update the trace manager and place the row, and if
// eventType passes the event filters, builds and queues the row for writing.
// Logging must never break the agent run, so a panic in spanFn or while the
// row is built drops the row instead.
//
// spanFn always runs before shouldLogEvent so that the spans and latencies of
// the rows that are written stay right when the event filters drop other rows.
func (p *analyticsPlugin) logEvent(ctx context.Context, id identity, eventType string, spanFn func() rowSpan, build func(*eventData)) {
	defer p.recoverCallback(eventType)
	var span rowSpan
	if spanFn != nil {
		span = spanFn()
	}
	if !p.shouldLogEvent(eventType) {
		return
	}
	d := &eventData{span: span}
	if build != nil {
		build(d)
	}
	p.processor.Append(p.buildRow(ctx, id, eventType, d))
}

func (p *analyticsPlugin) onUserMessage(ctx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
	id := identityOf(ctx)
	p.logEvent(ctx, id, EventUserMessageReceived, func() rowSpan {
		return p.traces.invocationSpan(ctx, id, runnerAgentOf(ctx))
	}, func(d *eventData) {
		d.content = msg
		d.author = userAuthor
	})
	return nil, nil
}

func (p *analyticsPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	id := identityOf(ctx)
	p.logEvent(ctx, id, EventInvocationStarting, func() rowSpan {
		return p.traces.invocationSpan(ctx, id, runnerAgentOf(ctx))
	}, nil)
	return nil, nil
}

// afterRun ends the invocation. It writes an AGENT_ERROR row for each agent
// span that the invocation's terminal error is attributed to, then
// INVOCATION_ERROR if the invocation has a terminal error, or else
// INVOCATION_COMPLETED.
//
// Only detectable failures are reported: unrecovered model errors,
// cancellation, agents stopped with an open model or tool call, and open
// agent spans whose activation produced no response. A run that fails after
// producing a response, such as a custom agent that writes an event and then
// returns an error, or an AfterAgentCallback that returns an error, ends with
// INVOCATION_COMPLETED, and so does a run in which a sub-agent fails without
// any callback after an agent that it runs under produced content: no plugin
// callback tells either apart from a successful run whose caller stopped
// reading once it had the response. popUnclosedAgentSpans documents the
// rules.
//
// The rules rely on this plugin seeing the callbacks of the run. ADK stops
// each callback chain at the first plugin that returns a non-nil result or an
// error, so the plugin has to be registered before plugins that return
// replacement events, model responses, tool results or agent content; see
// [NewBigQueryAgentAnalyticsPluginWithClients].
func (p *analyticsPlugin) afterRun(ctx agent.InvocationContext) {
	defer p.recoverCallback(EventInvocationCompleted)
	id := identityOf(ctx)
	unclosed, unrecErr := p.traces.popUnclosedAgentSpans(ctx, id)
	if unrecErr == nil && ctx != nil {
		// A canceled invocation failed even when no span is left open, for
		// example when canceled before any agent started or while post-run
		// session compaction ran.
		unrecErr = orZero(ctx.Err)
	}
	for _, u := range unclosed {
		agentID := id
		if u.agentName != "" {
			agentID.agentName = u.agentName
		}
		agentID.branch = u.branch
		agentID.path = u.path
		p.logEvent(ctx, agentID, EventAgentError, func() rowSpan {
			return u.span
		}, func(d *eventData) {
			d.failed = true
			err := u.err
			if err == nil {
				err = unrecErr
			}
			if err != nil {
				d.errMsg = err.Error()
				d.content = map[string]any{"error_traceback": err.Error()}
			}
			d.author = u.agentName
		})
	}

	if unrecErr != nil || len(unclosed) > 0 {
		p.logEvent(ctx, id, EventInvocationError, func() rowSpan {
			return p.traces.endInvocation(ctx, id)
		}, func(d *eventData) {
			d.failed = true
			if unrecErr != nil {
				d.errMsg = unrecErr.Error()
				d.content = map[string]any{"error_traceback": unrecErr.Error()}
			}
		})
	} else {
		p.logEvent(ctx, id, EventInvocationCompleted, func() rowSpan {
			return p.traces.endInvocation(ctx, id)
		}, nil)
	}

	if p.config.flushOnRunEnd() {
		p.processor.flush() // flush queue synchronously
	}
}

// onEvent writes the row of a session event. The event callback gets the
// agent that the runner runs, so the event is placed by its author, branch
// and node path. A non-partial event with content or a typed output also
// records that its author responded: ADK runs no AfterModelCallback for a
// model call that a BeforeModelCallback answered, so the event is what ends
// such a call, and a custom agent or workflow node may respond with an output
// alone. The function responses of such an event end the tool calls that they
// answer, which an earlier plugin's AfterToolCallback may have hidden from
// this plugin. Streamed partial events end nothing: AfterModelCallback ends
// the model call with the final response, which carries the call's latency.
//
// Because ADK stops an event callback chain at the first plugin that returns an
// event, this callback does not run for events that an earlier-registered
// plugin replaces; see [NewBigQueryAgentAnalyticsPluginWithClients].
func (p *analyticsPlugin) onEvent(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
	defer p.recoverCallback(legacyEventType)
	if ev != nil {
		id := identityOf(ctx)
		evID := id
		if ev.NodeInfo != nil && ev.NodeInfo.Path != "" {
			evID.path = ev.NodeInfo.Path
		}
		if ev.Content != nil || ev.Output != nil {
			if ev.Partial {
				p.traces.recordStreamed(ctx, evID, ev.Author, ev.Branch)
			} else {
				p.traces.completeResponse(ctx, evID, ev.Author, ev.Branch, functionResponseCalls(ev.Content))
			}
		} else if !ev.Partial {
			p.traces.clearLastPartial(ctx, evID)
		}
		p.logEvent(ctx, id, legacyEventType, func() rowSpan {
			return p.traces.eventSpan(ctx, evID, ev.Author, ev.Branch)
		}, func(d *eventData) {
			d.content = ev.Content
			d.attributes = map[string]any{"event_author": ev.Author}
			d.sourceEvent = ev
			d.author = ev.Author
			d.customMetadata = ev.CustomMetadata
		})
		if !ev.Partial && ev.Author != "" && ev.Author != userAuthor && hasTaskCompletedResponse(ev.Content) {
			taskID := evID
			taskID.agentName = ev.Author
			taskID.branch = ev.Branch
			p.logEvent(ctx, taskID, EventAgentCompleted, func() rowSpan {
				return p.traces.popTaskAgentSpan(ctx, taskID)
			}, nil)
		}
	}
	return nil, nil
}

// hasTaskCompletedResponse reports whether c carries a successful
// finish_task function response ("result": "Task completed."), which ends a
// task-mode LLM agent in ADK without running its AfterAgentCallback.
func hasTaskCompletedResponse(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, part := range c.Parts {
		if part == nil || part.FunctionResponse == nil {
			continue
		}
		if part.FunctionResponse.Name == "finish_task" {
			result, ok := part.FunctionResponse.Response["result"].(string)
			return ok && result == "Task completed."
		}
	}
	return false
}

// functionResponseCalls returns the function call ID, or else the tool name,
// of each function response in c: the part of the key that toolKey builds for
// the tool call that the response answers.
func functionResponseCalls(c *genai.Content) []string {
	if c == nil {
		return nil
	}
	var calls []string
	for _, part := range c.Parts {
		if part == nil || part.FunctionResponse == nil {
			continue
		}
		if call := cmp.Or(part.FunctionResponse.ID, part.FunctionResponse.Name); call != "" {
			calls = append(calls, call)
		}
	}
	return calls
}

func (p *analyticsPlugin) beforeAgent(ctx agent.Context) (*genai.Content, error) {
	id := identityOf(ctx)
	p.logEvent(ctx, id, EventAgentStarting, func() rowSpan {
		return p.traces.pushAgentSpan(ctx, id)
	}, func(d *eventData) {
		// The context that ADK gives agent callbacks logs a warning and
		// returns nil from Agent, so the agent is looked up by name in the
		// tree of the agent that the runner runs.
		d.content, d.attributes = agentStartingData(p.traces.agentNamed(ctx, id))
	})
	return nil, nil
}

func (p *analyticsPlugin) afterAgent(ctx agent.Context) (*genai.Content, error) {
	id := identityOf(ctx)
	p.logEvent(ctx, id, EventAgentCompleted, func() rowSpan {
		return p.traces.popAgentSpan(ctx, id)
	}, nil)
	return nil, nil
}

func (p *analyticsPlugin) beforeModel(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	id := identityOf(ctx)
	p.logEvent(ctx, id, EventLLMRequest, func() rowSpan {
		return p.traces.pushLLMSpan(ctx, id)
	}, func(d *eventData) {
		d.content = req
		d.attributes = llmRequestAttributes(req)
	})
	return nil, nil
}

func (p *analyticsPlugin) afterModel(ctx agent.Context, res *model.LLMResponse, err error) (*model.LLMResponse, error) {
	defer p.recoverCallback(EventLLMResponse)
	id := identityOf(ctx)
	if res != nil && res.Partial {
		// A streamed chunk marks the model call and its agent lineage as
		// streaming so a caller that stops reading mid-stream is not mistaken
		// for a failed call, and times the first token when the chunk carries
		// generated content: a leading chunk of usage or role metadata is not
		// the first token. adk-python writes a row for every chunk; the final
		// response alone is written here so that token usage is counted once.
		if hasGeneratedContent(res) {
			p.traces.recordFirstToken(ctx, id)
		} else {
			p.traces.recordStreamed(ctx, id, id.agentName, id.branch)
		}
		return nil, nil
	}
	key := llmKey(id)
	if err != nil {
		p.traces.recordError(ctx, id, key, err)
	} else if res != nil {
		p.traces.clearError(ctx, id, key)
	}
	p.logEvent(ctx, id, EventLLMResponse, func() rowSpan {
		return p.traces.popLLMSpan(ctx, id, res != nil)
	}, func(d *eventData) {
		var errText string
		if err != nil {
			errText = err.Error()
			d.failed = true
		}
		if res != nil {
			if res.Content != nil || res.UsageMetadata != nil {
				d.content = res
			}
			d.attributes = llmResponseAttributes(res)
			d.customMetadata = res.CustomMetadata
			d.errMsg = cmp.Or(errText, res.ErrorMessage)
		} else {
			d.errMsg = errText
		}
	})
	return nil, nil
}

// hasGeneratedContent reports whether res carries content that the model
// generated: a part with text, inline or file data, a function call or
// response, or executable code or its result, the kinds of content that the
// content formatter writes. A chunk of a streamed response that carries only
// metadata, such as token usage or the role of its content, or whose content
// has no part with any of these, is not generated content.
func hasGeneratedContent(res *model.LLMResponse) bool {
	if res == nil || res.Content == nil || len(res.Content.Parts) == 0 {
		return false
	}
	for _, part := range res.Content.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" || part.InlineData != nil || part.FunctionCall != nil ||
			part.FunctionResponse != nil || part.FileData != nil ||
			part.ExecutableCode != nil || part.CodeExecutionResult != nil {
			return true
		}
	}
	return false
}

func (p *analyticsPlugin) onModelError(ctx agent.Context, req *model.LLMRequest, err error) (*model.LLMResponse, error) {
	defer p.recoverCallback(EventLLMError)
	id := identityOf(ctx)
	if err != nil {
		p.traces.recordError(ctx, id, llmKey(id), err)
	}
	p.logEvent(ctx, id, EventLLMError, func() rowSpan {
		return p.traces.popLLMSpan(ctx, id, false)
	}, func(d *eventData) {
		d.failed = true
		if err != nil {
			d.errMsg = err.Error()
		}
		if req != nil && req.Model != "" {
			d.attributes = map[string]any{"model": req.Model}
		}
	})
	return nil, nil
}

func (p *analyticsPlugin) beforeTool(ctx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	p.logEvent(ctx, id, EventToolStarting, func() rowSpan {
		return p.traces.pushToolSpan(ctx, id, toolKey(id, callID, name))
	}, func(d *eventData) {
		d.content = toolContent(name, "args", args)
		var desc string
		var longRunning bool
		if t != nil {
			desc = t.Description()
			longRunning = t.IsLongRunning()
		}
		d.attributes = map[string]any{
			"tool_description": desc,
			"is_long_running":  longRunning,
		}
		d.functionCallID = callID
	})
	return nil, nil
}

func (p *analyticsPlugin) afterTool(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	defer p.recoverCallback(EventToolCompleted)
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	span, closedByError := p.traces.popToolSpan(ctx, id, toolKey(id, callID, name), false)
	if err != nil {
		// OnToolErrorCallback normally owns the TOOL_ERROR row and sets
		// closedByError; when an earlier plugin short-circuited
		// OnToolErrorCallback before onToolError ran, emit TOOL_ERROR here so
		// the failed call still gets its terminal row.
		if !closedByError {
			p.logToolError(ctx, id, func() rowSpan { return span }, name, callID, args, err)
		}
		return nil, nil
	}
	p.logEvent(ctx, id, EventToolCompleted, func() rowSpan { return span }, func(d *eventData) {
		d.content = toolContent(name, "result", result)
		d.functionCallID = callID
	})
	if _, ok := p.finalResponseTools[name]; ok {
		p.logEvent(ctx, id, EventAgentResponse, func() rowSpan {
			return p.traces.eventSpan(ctx, id, id.agentName, id.branch)
		}, func(d *eventData) {
			d.content = map[string]any{"response": args}
			d.attributes = map[string]any{"source_tool": name}
			d.functionCallID = callID
		})
	}
	return nil, nil
}

func (p *analyticsPlugin) onToolError(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
	defer p.recoverCallback(EventToolError)
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	p.logToolError(ctx, id, func() rowSpan {
		span, _ := p.traces.popToolSpan(ctx, id, toolKey(id, callID, name), true)
		return span
	}, name, callID, args, err)
	return nil, nil
}

func (p *analyticsPlugin) logToolError(ctx context.Context, id identity, spanFn func() rowSpan, name, callID string, args map[string]any, err error) {
	p.logEvent(ctx, id, EventToolError, spanFn, func(d *eventData) {
		d.content = toolContent(name, "args", args)
		d.failed = true
		if err != nil {
			d.errMsg = err.Error()
		}
		d.functionCallID = callID
	})
}

// toolName returns the name of t, or "" if t is nil or its Name method
// panics.
func toolName(t tool.Tool) string {
	if t == nil {
		return ""
	}
	return orZero(t.Name)
}

func toolContent(name, payloadKey string, payload map[string]any) map[string]any {
	return map[string]any{
		"tool":        name,
		payloadKey:    payload,
		"tool_origin": unknownToolOrigin,
	}
}

// llmRequestAttributes returns the model, generation settings, labels and
// tool declarations of req.
func llmRequestAttributes(req *model.LLMRequest) map[string]any {
	attrs := map[string]any{}
	if req == nil {
		return attrs
	}
	if req.Model != "" {
		attrs["model"] = req.Model
	}
	if cfg := req.Config; cfg != nil {
		if c := llmConfig(cfg); len(c) > 0 {
			attrs["llm_config"] = c
		}
		if len(cfg.Labels) > 0 {
			attrs["labels"] = cfg.Labels
		}
	}
	if tools := toolDeclarations(req); len(tools) > 0 {
		attrs["tools"] = tools
	}
	return attrs
}

// llmConfig returns the generation settings that adk-python records, plus
// the thinking configuration. Unset settings are left out.
func llmConfig(cfg *genai.GenerateContentConfig) map[string]any {
	c := map[string]any{}
	putIfSet(c, "temperature", cfg.Temperature)
	putIfSet(c, "top_p", cfg.TopP)
	putIfSet(c, "top_k", cfg.TopK)
	if cfg.CandidateCount != 0 {
		c["candidate_count"] = cfg.CandidateCount
	}
	if cfg.MaxOutputTokens != 0 {
		c["max_output_tokens"] = cfg.MaxOutputTokens
	}
	if len(cfg.StopSequences) > 0 {
		c["stop_sequences"] = cfg.StopSequences
	}
	putIfSet(c, "presence_penalty", cfg.PresencePenalty)
	putIfSet(c, "frequency_penalty", cfg.FrequencyPenalty)
	if cfg.ResponseMIMEType != "" {
		c["response_mime_type"] = cfg.ResponseMIMEType
	}
	if cfg.ResponseSchema != nil {
		c["response_schema"] = cfg.ResponseSchema
	}
	putIfSet(c, "seed", cfg.Seed)
	if cfg.ResponseLogprobs {
		c["response_logprobs"] = true
	}
	putIfSet(c, "logprobs", cfg.Logprobs)
	if cfg.ThinkingConfig != nil {
		c["thinking_config"] = cfg.ThinkingConfig
	}
	return c
}

func putIfSet[T any](m map[string]any, key string, v *T) {
	if v != nil {
		m[key] = *v
	}
}

// toolDeclarations describes the tools of req, sorted by name, as
// {name, description, parameters} objects. Descriptions and parameter
// schemas come from the tool or, failing that, from its function
// declaration.
func toolDeclarations(req *model.LLMRequest) []map[string]any {
	if len(req.Tools) == 0 {
		return nil
	}
	configDecls := map[string]*genai.FunctionDeclaration{}
	if req.Config != nil {
		for _, t := range req.Config.Tools {
			if t == nil {
				continue
			}
			for _, fd := range t.FunctionDeclarations {
				if fd != nil && fd.Name != "" {
					configDecls[fd.Name] = fd
				}
			}
		}
	}
	type declarer interface {
		Declaration() *genai.FunctionDeclaration
	}
	decls := make([]map[string]any, 0, len(req.Tools))
	for _, key := range slices.Sorted(maps.Keys(req.Tools)) {
		entry := map[string]any{"name": key}
		t := req.Tools[key]
		if tt, ok := t.(tool.Tool); ok {
			entry["name"] = cmp.Or(tt.Name(), key)
			if desc := tt.Description(); desc != "" {
				entry["description"] = desc
			}
		}
		var fd *genai.FunctionDeclaration
		if d, ok := t.(declarer); ok {
			fd = orZero(d.Declaration)
		}
		if fd == nil {
			fd = configDecls[key]
		}
		if fd != nil {
			if _, ok := entry["description"]; !ok && fd.Description != "" {
				entry["description"] = fd.Description
			}
			switch {
			case fd.ParametersJsonSchema != nil:
				entry["parameters"] = fd.ParametersJsonSchema
			case fd.Parameters != nil:
				entry["parameters"] = fd.Parameters
			}
		}
		decls = append(decls, entry)
	}
	return decls
}

// llmRequestContent builds the LLM_REQUEST content the way adk-python does:
// the role and text summary of each request message, and the summary of the
// system instruction. When withParts is set it also returns the
// content_parts of the messages followed by those of the system instruction.
func llmRequestContent(req *model.LLMRequest, maxLen int, withParts, redact bool) (content map[string]any, parts []map[string]any, contentTruncated, partsTruncated bool) {
	content = map[string]any{}
	add := func(c *genai.Content) string {
		summary, t := contentSummary(c, maxLen, redact)
		contentTruncated = contentTruncated || t
		if withParts {
			ps, pt := formatContentParts(c, maxLen, redact)
			parts = append(parts, ps...)
			partsTruncated = partsTruncated || pt
		}
		return summary
	}
	var prompt []map[string]any
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		role, t := sanitizeAndTruncateText(cmp.Or(c.Role, "unknown"), maxLen, redact)
		contentTruncated = contentTruncated || t
		prompt = append(prompt, map[string]any{"role": role, "content": add(c)})
	}
	if len(prompt) > 0 {
		content["prompt"] = prompt
	}
	if req.Config != nil && req.Config.SystemInstruction != nil {
		content["system_prompt"] = add(req.Config.SystemInstruction)
	}
	return content, parts, contentTruncated, partsTruncated
}

// llmResponseContent builds the LLM_RESPONSE content the way adk-python
// does: the formatted response and the token usage, plus content_parts when
// withParts is set. It returns nil content if res carries neither.
func llmResponseContent(res *model.LLMResponse, maxLen int, withParts, redact bool) (content map[string]any, parts []map[string]any, contentTruncated, partsTruncated bool) {
	content = map[string]any{}
	if res.Content != nil {
		content["response"], contentTruncated = formatResponseContent(res.Content, maxLen, redact)
		if withParts {
			parts, partsTruncated = formatContentParts(res.Content, maxLen, redact)
		}
	}
	if u := res.UsageMetadata; u != nil {
		usage := map[string]any{
			"prompt":     u.PromptTokenCount,
			"completion": u.CandidatesTokenCount,
			"total":      u.TotalTokenCount,
		}
		if u.ThoughtsTokenCount > 0 {
			usage["thoughts"] = u.ThoughtsTokenCount
		}
		if u.CachedContentTokenCount > 0 {
			usage["cached"] = u.CachedContentTokenCount
		}
		content["usage"] = usage
	}
	if len(content) == 0 {
		return nil, parts, false, partsTruncated
	}
	return content, parts, contentTruncated, partsTruncated
}

func llmResponseAttributes(res *model.LLMResponse) map[string]any {
	attrs := map[string]any{
		"partial":       res.Partial,
		"turn_complete": res.TurnComplete,
		"interrupted":   res.Interrupted,
	}
	if res.ModelVersion != "" {
		attrs["model_version"] = res.ModelVersion
	}
	if u := res.UsageMetadata; u != nil {
		usageMeta := map[string]any{
			"prompt_token_count":     u.PromptTokenCount,
			"candidates_token_count": u.CandidatesTokenCount,
			"total_token_count":      u.TotalTokenCount,
		}
		if u.ThoughtsTokenCount > 0 {
			usageMeta["thoughts_token_count"] = u.ThoughtsTokenCount
		}
		if u.CachedContentTokenCount > 0 {
			usageMeta["cached_content_token_count"] = u.CachedContentTokenCount
		}
		if u.ToolUsePromptTokenCount > 0 {
			usageMeta["tool_use_prompt_token_count"] = u.ToolUsePromptTokenCount
		}
		attrs["usage_metadata"] = usageMeta
	}
	if res.FinishReason != "" {
		attrs["finish_reason"] = string(res.FinishReason)
	}
	if res.GroundingMetadata != nil {
		attrs["grounding_metadata"] = res.GroundingMetadata
	}
	if res.CitationMetadata != nil {
		attrs["citation_metadata"] = res.CitationMetadata
	}
	attrs["cache_type"] = cacheType(res.UsageMetadata)
	return attrs
}

// cacheType classifies the cached tokens of a response as "none" without
// cached tokens and "implicit" when cached tokens are present, because ADK Go
// responses carry no explicit context-cache metadata.
func cacheType(u *genai.GenerateContentResponseUsageMetadata) string {
	if u == nil || u.CachedContentTokenCount <= 0 {
		return "none"
	}
	return "implicit"
}

// agentStartingData returns the AGENT_STARTING content and attributes of
// agent a, matching adk-python: the static instruction of an LLM agent as
// content, and its description, model, sub-agent names and, when an
// instruction provider builds the instruction, instruction_source ("provider",
// with nil content). An agent that inherits its model from an ancestor has no
// model attribute, because ADK Go exposes no parent link on agent.Agent.
func agentStartingData(a agent.Agent) (content any, attrs map[string]any) {
	attrs = map[string]any{}
	if a == nil {
		return nil, attrs
	}
	putNonEmpty(attrs, "agent_description", orZero(a.Description))
	var subAgents []string
	for _, sub := range orZero(a.SubAgents) {
		if sub != nil {
			if name := orZero(sub.Name); name != "" {
				subAgents = append(subAgents, name)
			}
		}
	}
	if len(subAgents) > 0 {
		attrs["sub_agents"] = subAgents
	}
	cfg, ok := orZero(func() any { return agentConfig(a) }).(llmagent.Config)
	if !ok {
		return nil, attrs
	}
	if cfg.Model != nil {
		putNonEmpty(attrs, "model", orZero(cfg.Model.Name))
	}
	if cfg.InstructionProvider != nil {
		attrs["instruction_source"] = "provider"
		return nil, attrs
	}
	if cfg.Instruction != "" {
		content = cfg.Instruction
	}
	return content, attrs
}

// agentConfig returns the configuration that ADK keeps in the base agent of
// a, such as the llmagent.Config of an LLM agent, or nil. ADK exposes no
// accessor for it, so it is read with reflection from the exported Config
// field of the base agent's embedded state. An agent that embeds its base
// agent in an exported agent.Agent field, as LLM agents do, is unwrapped
// when its own Config is zero. Agents without a readable Config field yield nil.
func agentConfig(a agent.Agent) any {
	// One unwrapping reaches the base agent of the ADK agents; the bound
	// stops agents that wrap themselves.
	for range 4 {
		v := reflect.ValueOf(a)
		for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
			if v.IsNil() {
				return nil
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return nil
		}
		if f := v.FieldByName("Config"); f.IsValid() && f.CanInterface() && !f.IsZero() {
			return f.Interface()
		}
		inner := v.FieldByName("Agent")
		if !inner.IsValid() || !inner.CanInterface() {
			return nil
		}
		next, ok := inner.Interface().(agent.Agent)
		if !ok {
			return nil
		}
		a = next
	}
	return nil
}

// identity holds the invocation fields of a row.
type identity struct {
	agentName    string
	sessionID    string
	invocationID string
	userID       string
	appName      string
	branch       string
	path         string
	state        session.ReadonlyState
}

// identityOf reads the invocation fields from an ADK context. The context
// accessors panic when the invocation has no agent or session (as in some
// workflow runs), so each field is read on its own and left empty on a
// panic.
func identityOf(ctx context.Context) identity {
	switch c := ctx.(type) {
	case agent.ReadonlyContext:
		return identity{
			agentName:    orZero(c.AgentName),
			sessionID:    orZero(c.SessionID),
			invocationID: orZero(c.InvocationID),
			userID:       orZero(c.UserID),
			appName:      orZero(c.AppName),
			branch:       orZero(c.Branch),
			path:         orZero(func() string { return contextPath(ctx) }),
			state:        orZero(c.ReadonlyState),
		}
	case agent.InvocationContext:
		id := identity{
			invocationID: orZero(c.InvocationID),
			branch:       orZero(c.Branch),
			path:         orZero(func() string { return contextPath(ctx) }),
		}
		if a := orZero(c.Agent); a != nil {
			id.agentName = orZero(a.Name)
		}
		if s := orZero(c.Session); s != nil {
			id.sessionID = orZero(s.ID)
			id.userID = orZero(s.UserID)
			id.appName = orZero(s.AppName)
			id.state = orZero(s.State)
		}
		return id
	default:
		return identity{}
	}
}

type visitedContextStruct struct {
	ptr uintptr
	typ reflect.Type
}

// contextPath returns the execution node path carried by ctx, or "".
// ADK's callbackContextWrapper.Path logs an error and returns "", and
// NewCallbackContext wraps the inner InvocationContext without copying its
// path onto the outer commonContext, so contextPath inspects the wrapped
// context chain directly before falling back to Path() on non-wrapper types.
func contextPath(ctx context.Context) (path string) {
	defer func() {
		if r := recover(); r != nil {
			path = ""
		}
	}()
	queue := []reflect.Value{reflect.ValueOf(ctx)}
	seen := map[visitedContextStruct]bool{}
	for steps := 0; len(queue) > 0 && steps < 512; steps++ {
		v := queue[0]
		queue = queue[1:]
		var ptr uintptr
		for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
			if v.IsNil() {
				v = reflect.Value{}
				break
			}
			if v.Kind() == reflect.Pointer {
				ptr = v.Pointer()
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			continue
		}
		if ptr != 0 {
			key := visitedContextStruct{ptr: ptr, typ: v.Type()}
			if seen[key] {
				continue
			}
			seen[key] = true
		}
		if f := v.FieldByName("path"); f.IsValid() && f.Kind() == reflect.String {
			if s := f.String(); s != "" {
				return s
			}
		}
		if f := v.FieldByName("params"); f.IsValid() {
			for f.Kind() == reflect.Pointer || f.Kind() == reflect.Interface {
				if f.IsNil() {
					f = reflect.Value{}
					break
				}
				f = f.Elem()
			}
			if f.Kind() == reflect.Struct {
				if p := f.FieldByName("Path"); p.IsValid() && p.Kind() == reflect.String {
					if s := p.String(); s != "" {
						return s
					}
				}
			}
		}
		for _, name := range []string{"context", "Context", "ctx", "invocationContext"} {
			if f := v.FieldByName(name); f.IsValid() {
				queue = append(queue, f)
			}
		}
	}
	type pathCarrier interface {
		Path() string
	}
	if p, ok := ctx.(pathCarrier); ok && !strings.HasSuffix(reflect.TypeOf(ctx).String(), "callbackContextWrapper") {
		return orZero(p.Path)
	}
	return ""
}

// orZero returns f(), or the zero value of T if f panics.
func orZero[T any](f func() T) (v T) {
	defer func() { _ = recover() }()
	return f()
}

// runnerAgentOf returns the agent of the context of an invocation callback,
// which is the agent that the runner runs, or nil.
func runnerAgentOf(ctx agent.InvocationContext) agent.Agent {
	if ctx == nil {
		return nil
	}
	return orZero(ctx.Agent)
}

func (p *analyticsPlugin) buildRow(ctx context.Context, id identity, eventType string, d *eventData) map[string]any {
	eventID := newEventID()
	row := map[string]any{
		timestampField:  time.Now(),
		eventIDField:    eventID,
		"event_type":    eventType,
		"invocation_id": id.invocationID,
		"user_id":       id.userID,
	}
	agentName := id.agentName
	if agentName == "" && d.sourceEvent != nil {
		agentName = d.sourceEvent.Author
	}
	if agentName != "" {
		row["agent"] = agentName
	}
	if id.sessionID != "" {
		row["session_id"] = id.sessionID
	}

	putNonEmpty(row, "trace_id", d.span.traceID)
	putNonEmpty(row, "span_id", d.span.spanID)
	putNonEmpty(row, "parent_span_id", d.span.parentSpanID)
	if d.span.latency != nil && !p.denied(columnLatencyMS) {
		// The batch processor writes JSON columns from their JSON text, so
		// the latency object is marshaled here, exactly once.
		if b, err := json.Marshal(d.span.latency); err == nil {
			row[columnLatencyMS] = string(b)
		}
	}
	// The ambient OpenTelemetry span feeds trace_id (via spanFn) and attributes.otel.
	span := trace.SpanContextFromContext(ctx)

	truncated := p.addContent(row, eventType, d)
	if !p.denied(columnAttributes) {
		attrs, t := p.jsonColumn(p.attributes(eventID, id, agentName, span, d))
		row[columnAttributes] = attrs
		truncated = truncated || t
	}

	status := "OK"
	if d.failed {
		status = "ERROR"
	}
	row["status"] = status
	if d.errMsg != "" {
		errMsg := d.errMsg
		if p.config.redactSensitiveKeys() {
			errMsg = redactErrorMessage(errMsg)
		}
		msg, t := truncateString(errMsg, p.config.maxContentLength())
		row["error_message"] = msg
		truncated = truncated || t
	}
	row["is_truncated"] = truncated
	return row
}

// newEventID returns a random UUID as 32 hex digits, the format of
// adk-python's uuid4().hex.
func newEventID() string {
	id := uuid.New()
	return hex.EncodeToString(id[:])
}

// addContent writes the content and content_parts columns of row and
// reports whether anything written was truncated.
func (p *analyticsPlugin) addContent(row map[string]any, eventType string, d *eventData) bool {
	contentDenied := p.denied(columnContent)
	partsDenied := p.denied(columnContentParts)
	if isNil(d.content) || (contentDenied && partsDenied) {
		return false
	}
	payload, errNote, failed := applyContentFormatter(p.config.ContentFormatter, p.config.Logger, d.content, eventType, p.config.DebugContentFormatterErrors)
	if failed {
		if d.attributes == nil {
			d.attributes = map[string]any{}
		}
		d.attributes["formatter_failed"] = true
		if errNote != "" {
			if d.errMsg != "" {
				d.errMsg += "; "
			}
			d.errMsg += errNote
		}
		if !contentDenied {
			row[columnContent] = strconv.Quote(formattingFailedPlaceholder)
		}
		return false
	}
	if isNil(payload) {
		return false
	}

	maxLen := p.config.maxContentLength()
	redact := p.config.redactSensitiveKeys()
	withParts := !partsDenied && p.config.logMultiModalContent()
	content := payload
	contentTruncated := d.contentTruncated
	var parts []map[string]any
	partsTruncated := false
	preFormatted := false
	switch v := payload.(type) {
	case *genai.Content:
		summary, t := contentSummary(v, maxLen, redact)
		content, contentTruncated = map[string]any{"text_summary": summary}, t
		preFormatted = true
		if withParts {
			parts, partsTruncated = formatContentParts(v, maxLen, redact)
		}
	case *model.LLMRequest:
		var t bool
		content, parts, t, partsTruncated = llmRequestContent(v, maxLen, withParts, redact)
		contentTruncated = t
		preFormatted = true
	case *model.LLMResponse:
		var t bool
		content, parts, t, partsTruncated = llmResponseContent(v, maxLen, withParts, redact)
		contentTruncated = t
		preFormatted = true
	}

	truncated := false
	if !contentDenied && !isNil(content) {
		if preFormatted {
			b, err := json.Marshal(content)
			if err != nil {
				row[columnContent] = strconv.Quote(unsupportedObjPlaceholder)
				truncated = true
			} else {
				row[columnContent] = string(b)
				truncated = contentTruncated
			}
		} else {
			s, t := p.jsonColumn(content)
			row[columnContent] = s
			truncated = t || contentTruncated
		}
	}
	if len(parts) > 0 {
		row[columnContentParts] = parts
		truncated = truncated || partsTruncated
	}
	return truncated
}

// isNil reports whether v is nil or holds a nil pointer, map, slice or
// interface.
func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return rv.IsNil()
	default:
		return false
	}
}

// jsonColumn converts v to the JSON text of a JSON column, applying
// truncation and sensitive-key redaction.
func (p *analyticsPlugin) jsonColumn(v any) (string, bool) {
	b, truncated, err := smartTruncate(v, p.config.maxContentLength(), p.config.redactSensitiveKeys())
	if err != nil {
		return strconv.Quote(unsupportedObjPlaceholder), true
	}
	return string(b), truncated
}

// attributes assembles the attributes column: the event-specific attributes
// followed by the common ones.
func (p *analyticsPlugin) attributes(eventID string, id identity, agentName string, span trace.SpanContext, d *eventData) map[string]any {
	attrs := make(map[string]any, len(d.attributes)+6)
	maps.Copy(attrs, d.attributes)
	attrs["adk"] = p.adkEnvelope(eventID, id, agentName, d)
	putNonEmpty(attrs, "root_agent_name", d.span.rootAgentName)
	if len(p.config.CustomTags) > 0 {
		attrs["custom_tags"] = p.config.CustomTags
	}
	if p.config.logSessionMetadata() {
		attrs["session_metadata"] = sessionMetadata(id)
	}
	if p.config.EnableOTelCorrelation && span.IsValid() {
		attrs["otel"] = map[string]any{
			"span_id":  span.SpanID().String(),
			"trace_id": span.TraceID().String(),
		}
	}
	if md := p.customMetadata.apply(d.customMetadata); md != nil {
		attrs["custom_metadata"] = md
	}
	return attrs
}

// adkEnvelope builds attributes.adk, which identifies the ADK objects a row
// belongs to. It shares schema_version, app_name, function_call_id, and the
// session-event fields (source_event_id, branch, node, scope, route) with
// adk-python's envelope, and includes Go-specific extension fields
// (adk_version when available, event_id, invocation_id, session_id, user_id,
// author, isolation_scope, artifact_delta, skip_summarization, escalate, and
// node.message_as_output/output_for).
func (p *analyticsPlugin) adkEnvelope(eventID string, id identity, agentName string, d *eventData) map[string]any {
	adk := map[string]any{
		"schema_version": adkEnvelopeSchemaVersion,
		"event_id":       eventID,
	}
	putNonEmpty(adk, "adk_version", adkVersion())
	putNonEmpty(adk, "app_name", id.appName)
	putNonEmpty(adk, "invocation_id", id.invocationID)
	putNonEmpty(adk, "function_call_id", d.functionCallID)
	if p.config.logSessionMetadata() {
		putNonEmpty(adk, "session_id", id.sessionID)
		putNonEmpty(adk, "user_id", id.userID)
	}
	putNonEmpty(adk, "author", cmp.Or(d.author, agentName))
	if ev := d.sourceEvent; ev != nil {
		addSourceEvent(adk, ev)
	}
	return adk
}

func addSourceEvent(adk map[string]any, ev *session.Event) {
	putNonEmpty(adk, "source_event_id", ev.ID)
	putNonEmpty(adk, "branch", ev.Branch)
	if ni := ev.NodeInfo; ni != nil {
		runID, parentRunID := parseNodePath(ni.Path)
		node := map[string]any{
			"path":          ni.Path,
			"run_id":        runID,
			"parent_run_id": parentRunID,
		}
		if ni.MessageAsOutput {
			node["message_as_output"] = true
		}
		if len(ni.OutputFor) > 0 {
			node["output_for"] = ni.OutputFor
		}
		adk["node"] = node
	}
	if ev.IsolationScope != "" {
		adk["isolation_scope"] = ev.IsolationScope
		adk["scope"] = deriveScope(ev.IsolationScope)
	}
	if len(ev.Routes) > 0 {
		adk["route"] = ev.Routes
	}
	if len(ev.Actions.ArtifactDelta) > 0 {
		adk["artifact_delta"] = ev.Actions.ArtifactDelta
	}
	if ev.Actions.SkipSummarization {
		adk["skip_summarization"] = true
	}
	if ev.Actions.Escalate {
		adk["escalate"] = true
	}
}

func putNonEmpty(m map[string]any, key, value string) {
	if value != "" {
		m[key] = value
	}
}

// parseNodePath returns the run ID of the last segment of a node path,
// "parent/name@run_id", and the run ID of the segment before it, as
// adk-python's NodeInfo does. parentRunID is nil if there is no parent
// segment or it has no run ID.
func parseNodePath(path string) (runID string, parentRunID any) {
	if path == "" {
		return "", nil
	}
	segments := strings.Split(path, "/")
	runID, _ = segmentRunID(segments[len(segments)-1])
	if len(segments) < 2 {
		return runID, nil
	}
	if parent, ok := segmentRunID(segments[len(segments)-2]); ok {
		return runID, parent
	}
	return runID, nil
}

func segmentRunID(segment string) (string, bool) {
	i := strings.LastIndex(segment, "@")
	if i < 0 {
		return "", false
	}
	return segment[i+1:], true
}

// deriveScope classifies an isolation scope as adk-python does: a scope
// whose last path segment has the node form name@run_id is a node run, and
// any other scope is a function call ID.
func deriveScope(scope string) map[string]any {
	kind := scopeKindFunctionCall
	if last := scope[strings.LastIndex(scope, "/")+1:]; strings.Contains(last, "@") {
		kind = scopeKindNodeRun
	}
	return map[string]any{"id": scope, "kind": kind}
}

// sessionMetadata returns attributes.session_metadata.
func sessionMetadata(id identity) map[string]any {
	md := map[string]any{
		"session_id": id.sessionID,
		"app_name":   id.appName,
		"user_id":    id.userID,
	}
	if state := orZero(func() map[string]any { return stateSnapshot(id.state) }); len(state) > 0 {
		md["state"] = state
	}
	return md
}

func stateSnapshot(s session.ReadonlyState) map[string]any {
	if s == nil {
		return nil
	}
	state := map[string]any{}
	for k, v := range s.All() {
		state[k] = v
	}
	return state
}

// adkVersion returns the version of the ADK module linked into the binary,
// or "" if build info has no usable version.
var adkVersion = sync.OnceValue(func() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	return adkVersionFromBuildInfo(info)
})

func adkVersionFromBuildInfo(info *debug.BuildInfo) string {
	if info == nil {
		return ""
	}
	if info.Main.Path == adkModulePath {
		if v := strings.TrimPrefix(info.Main.Version, "v"); v != "" && v != "(devel)" {
			return v
		}
	}
	for _, dep := range info.Deps {
		if dep.Path != adkModulePath {
			continue
		}
		if dep.Replace != nil && dep.Replace.Version != "" {
			dep = dep.Replace
		}
		if v := strings.TrimPrefix(dep.Version, "v"); v != "" && v != "(devel)" {
			return v
		}
	}
	return ""
}
