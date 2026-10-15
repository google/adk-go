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
	EventUserMessageReceived       = "USER_MESSAGE_RECEIVED"
	EventInvocationStarting        = "INVOCATION_STARTING"
	EventInvocationCompleted       = "INVOCATION_COMPLETED"
	EventAgentStarting             = "AGENT_STARTING"
	EventAgentCompleted            = "AGENT_COMPLETED"
	EventLLMRequest                = "LLM_REQUEST"
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
// EventDenylist accept without a warning: the Event constants, the legacy
// EVENT type and the error events that only adk-python writes.
var knownEventTypes = newSet(
	EventUserMessageReceived, EventInvocationStarting, EventInvocationCompleted,
	EventAgentStarting, EventAgentCompleted,
	EventLLMRequest, EventLLMResponse, EventLLMError,
	EventToolStarting, EventToolCompleted, EventToolError,
	EventStateDelta, EventAgentTransfer, EventEventCompaction, EventAgentStateCheckpoint, EventToolPaused,
	EventHITLCredentialRequest, EventHITLConfirmationRequest, EventHITLInputRequest,
	EventHITLCredentialCompleted, EventHITLConfirmationCompleted, EventHITLInputCompleted,
	EventA2AInteraction, EventAgentResponse, EventNodeOutput, EventNodeError,
	legacyEventType, "AGENT_ERROR", "INVOCATION_ERROR",
)

// NewBigQueryAgentAnalyticsPlugin creates a newly configured analytics plugin with default config.
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

	p := &analyticsPlugin{
		config:             config,
		processor:          processor,
		allowlist:          newSet(config.EventAllowlist...),
		denylist:           newSet(config.EventDenylist...),
		deniedColumns:      deniedColumns,
		finalResponseTools: newSet(config.FinalResponseToolNames...),
		customMetadata:     newCustomMetadataFilter(config.CustomMetadataAllowlist),
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
	loggedToolErrors   sync.Map
}

// eventData carries the event-specific parts of a row.
type eventData struct {
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

// logEvent builds a row for eventType and queues it for writing. Logging
// must never break the agent run, so a panic while the row is built drops
// the row instead.
func (p *analyticsPlugin) logEvent(ctx context.Context, eventType string, build func(*eventData)) {
	if !p.shouldLogEvent(eventType) {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			p.config.logf("agentanalytics: dropped %s event: recovered panic of type %T", eventType, r)
		}
	}()
	d := &eventData{}
	if build != nil {
		build(d)
	}
	p.processor.Append(p.buildRow(ctx, eventType, d))
}

func (p *analyticsPlugin) onUserMessage(ctx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
	p.logEvent(ctx, EventUserMessageReceived, func(d *eventData) {
		d.content = msg
		d.author = userAuthor
	})
	return nil, nil
}

func (p *analyticsPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	p.logEvent(ctx, EventInvocationStarting, nil)
	return nil, nil
}

func (p *analyticsPlugin) afterRun(ctx agent.InvocationContext) {
	id := identityOf(ctx)
	p.loggedToolErrors.Range(func(k, _ any) bool {
		if key, ok := k.(toolErrorIdentity); ok && key.invocationID == id.invocationID && key.sessionID == id.sessionID {
			p.loggedToolErrors.Delete(k)
		}
		return true
	})
	p.logEvent(ctx, EventInvocationCompleted, nil)
	if p.config.flushOnRunEnd() {
		p.processor.flush() // flush queue synchronously
	}
}

func (p *analyticsPlugin) onEvent(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
	if ev != nil {
		p.logEvent(ctx, legacyEventType, func(d *eventData) {
			d.content = ev.Content
			d.attributes = map[string]any{"event_author": ev.Author}
			d.sourceEvent = ev
			d.author = ev.Author
			d.customMetadata = ev.CustomMetadata
		})
	}
	return nil, nil
}

func (p *analyticsPlugin) beforeAgent(ctx agent.Context) (*genai.Content, error) {
	p.logEvent(ctx, EventAgentStarting, nil)
	return nil, nil
}

func (p *analyticsPlugin) afterAgent(ctx agent.Context) (*genai.Content, error) {
	p.logEvent(ctx, EventAgentCompleted, nil)
	return nil, nil
}

func (p *analyticsPlugin) beforeModel(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	p.logEvent(ctx, EventLLMRequest, func(d *eventData) {
		d.content = req
		d.attributes = llmRequestAttributes(req)
	})
	return nil, nil
}

func (p *analyticsPlugin) afterModel(ctx agent.Context, res *model.LLMResponse, err error) (*model.LLMResponse, error) {
	if res != nil && res.Partial {
		// TODO(haiyuan): log partial responses, with time to first token, as
		// adk-python does, without counting their usage twice.
		return nil, nil
	}
	p.logEvent(ctx, EventLLMResponse, func(d *eventData) {
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

func (p *analyticsPlugin) onModelError(ctx agent.Context, req *model.LLMRequest, err error) (*model.LLMResponse, error) {
	p.logEvent(ctx, EventLLMError, func(d *eventData) {
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
	p.logEvent(ctx, EventToolStarting, func(d *eventData) {
		d.content = toolContent(toolName(t), "args", args)
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
		d.functionCallID = ctx.FunctionCallID()
	})
	return nil, nil
}

func (p *analyticsPlugin) afterTool(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	name := toolName(t)
	_, alreadyLogged := p.loggedToolErrors.LoadAndDelete(newToolErrorIdentity(ctx, name))
	if err != nil {
		if !alreadyLogged {
			p.emitToolError(ctx, name, args, err)
		}
		return nil, nil
	}
	p.logEvent(ctx, EventToolCompleted, func(d *eventData) {
		d.content = toolContent(name, "result", result)
		d.functionCallID = ctx.FunctionCallID()
	})
	if _, ok := p.finalResponseTools[name]; ok {
		p.logEvent(ctx, EventAgentResponse, func(d *eventData) {
			d.content = map[string]any{"response": args}
			d.attributes = map[string]any{"source_tool": name}
			d.functionCallID = ctx.FunctionCallID()
		})
	}
	return nil, nil
}

func (p *analyticsPlugin) onToolError(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
	name := toolName(t)
	p.loggedToolErrors.Store(newToolErrorIdentity(ctx, name), struct{}{})
	p.emitToolError(ctx, name, args, err)
	return nil, nil
}

func (p *analyticsPlugin) emitToolError(ctx agent.Context, name string, args map[string]any, err error) {
	p.logEvent(ctx, EventToolError, func(d *eventData) {
		d.content = toolContent(name, "args", args)
		d.failed = true
		if err != nil {
			d.errMsg = err.Error()
		}
		d.functionCallID = ctx.FunctionCallID()
	})
}

type toolErrorIdentity struct {
	sessionID    string
	invocationID string
	branch       string
	callID       string
	toolName     string
}

func newToolErrorIdentity(ctx agent.Context, name string) toolErrorIdentity {
	return toolErrorIdentity{
		sessionID:    orZero(ctx.SessionID),
		invocationID: orZero(ctx.InvocationID),
		branch:       orZero(ctx.Branch),
		callID:       orZero(ctx.FunctionCallID),
		toolName:     name,
	}
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
	return attrs
}

// identity holds the invocation fields of a row.
type identity struct {
	agentName    string
	sessionID    string
	invocationID string
	userID       string
	appName      string
	branch       string
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
			state:        orZero(c.ReadonlyState),
		}
	case agent.InvocationContext:
		id := identity{
			invocationID: orZero(c.InvocationID),
			branch:       orZero(c.Branch),
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

// orZero returns f(), or the zero value of T if f panics.
func orZero[T any](f func() T) (v T) {
	defer func() { _ = recover() }()
	return f()
}

func (p *analyticsPlugin) buildRow(ctx context.Context, eventType string, d *eventData) map[string]any {
	id := identityOf(ctx)
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

	// Add trace info dynamically resolving the context span
	span := trace.SpanContextFromContext(ctx)
	if span.IsValid() {
		row["trace_id"] = span.TraceID().String()
		row["span_id"] = span.SpanID().String()
	}

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
