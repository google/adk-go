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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"weak"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
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
const (
	EventUserMessageReceived = "USER_MESSAGE_RECEIVED"
	EventInvocationStarting  = "INVOCATION_STARTING"
	EventInvocationCompleted = "INVOCATION_COMPLETED"
	EventInvocationError     = "INVOCATION_ERROR"
	EventAgentStarting       = "AGENT_STARTING"
	EventAgentCompleted      = "AGENT_COMPLETED"
	EventAgentError          = "AGENT_ERROR"
	EventLLMRequest          = "LLM_REQUEST"
	// EventLLMResponse is written once per model call, on the final
	// non-partial response. Streamed chunks with Partial set only feed
	// latency_ms.time_to_first_token_ms on that final row, and only the
	// first chunk with generated content does.
	EventLLMResponse  = "LLM_RESPONSE"
	EventLLMError     = "LLM_ERROR"
	EventToolStarting = "TOOL_STARTING"
	// EventToolCompleted is written when a tool call finishes without an error
	// (or with a tool confirmation request/rejection). It is not written for
	// a call whose failure OnToolErrorCallback already recorded as TOOL_ERROR
	// (for example, when an agent OnToolErrorCallback or a later plugin
	// recovers the error), nor for a call whose result is classified as a
	// failure (see Config.ToolResultClassifier), such as one answered by
	// plugin/retryandreflect.
	EventToolCompleted = "TOOL_COMPLETED"
	// EventToolError is written when a tool call fails, including when the
	// failure is recovered by an agent OnToolErrorCallback, a later plugin, or
	// plugin/retryandreflect (except when plugin/retryandreflect is registered
	// first and a call to a non-FunctionTool, or to a missing tool that no
	// BeforeToolCallback answered, fails: ADK then skips AfterToolCallback and
	// plugin/retryandreflect answers in OnToolErrorCallback before this plugin
	// runs; if a BeforeToolCallback answers a missing-tool lookup with a
	// non-nil error, ADK dispatches the error and after-tool callbacks so
	// TOOL_ERROR is still written), and when a tool returns a result that is
	// classified as a failure (see Config.ToolResultClassifier). An
	// AfterToolCallback that runs after OnToolErrorCallback already recorded
	// the failure adds no second row. When plugin/retryandreflect is
	// registered before this plugin, any tool result carrying the reserved
	// response_type "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN" is classified
	// as TOOL_ERROR before Config.ToolResultClassifier runs; its error_message
	// comes from error_details or error_type unless Config.ContentFormatter is
	// set or "content" is in Config.PayloadColumnDenylist, in which case a
	// fixed fallback diagnostic is written instead.
	EventToolError     = "TOOL_ERROR"
	EventStateDelta    = "STATE_DELTA"
	EventAgentTransfer = "AGENT_TRANSFER"
	// EventEventCompaction is emitted when session.EventActions.Compaction is
	// populated (or when session.Event.CustomMetadata carries "adk_compaction" on
	// ADK versions prior to typed EventActions.Compaction).
	EventEventCompaction = "EVENT_COMPACTION"
	// EventAgentStateCheckpoint is emitted when session.Event.CustomMetadata
	// carries "adk_agent_state" or "agent_state" (or terminal "end_of_agent" /
	// "adk_end_of_agent" == true), mirroring adk-python's checkpoint rows while
	// session.EventActions does not yet expose a typed agent-state field.
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
	// legacyEventType is accepted in EventAllowlist and EventDenylist as an
	// alias matching all session-event rows (STATE_DELTA, AGENT_RESPONSE, etc.)
	// for backwards compatibility with filters written before the per-signal
	// session event types.
	legacyEventType = "EVENT"

	// adkEnvelopeSchemaVersion is the version of the attributes.adk object.
	adkEnvelopeSchemaVersion = "1"

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
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED. In particular, the
// plugin tells that a model call is streaming from the partial events that
// its OnEventCallback sees, so if an earlier plugin replaces a partial event,
// a caller that stops reading mid-stream is reported as a failure.
//
// Tool errors handled by plugin/retryandreflect are recognized from its
// reserved response_type ("ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN") before
// Config.ToolResultClassifier runs and logged as TOOL_ERROR in either
// registration order, except when plugin/retryandreflect is registered first
// for a call to a tool that is not a FunctionTool or to a missing tool that no
// BeforeToolCallback answered: ADK skips AfterToolCallback for those calls and
// plugin/retryandreflect answers in OnToolErrorCallback, so this plugin never
// receives the error and logs no outcome row. (If a BeforeToolCallback answers
// a missing-tool lookup with a non-nil error, ADK runs the error and
// after-tool callbacks, so TOOL_ERROR is written.) When Config.ContentFormatter
// is set or "content" is in Config.PayloadColumnDenylist, the TOOL_ERROR row
// of a plugin/retryandreflect recovery payload writes a fixed fallback
// error_message rather than copying error_details or error_type from the map.
// Recoveries by other plugins registered before this plugin emit
// TOOL_COMPLETED when AfterToolCallback runs (such as BeforeToolCallback map
// answers or FunctionTool OnToolErrorCallback map recoveries, unless classified
// as a failure by Config.ToolResultClassifier), whereas OnToolErrorCallback map
// recoveries on a missing tool (that no BeforeToolCallback answered) or a
// non-FunctionTool skip AfterToolCallback and write no outcome row.
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
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED. In particular, the
// plugin tells that a model call is streaming from the partial events that
// its OnEventCallback sees, so if an earlier plugin replaces a partial event,
// a caller that stops reading mid-stream is reported as a failure.
//
// Tool errors handled by plugin/retryandreflect are recognized from its
// reserved response_type ("ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN") before
// Config.ToolResultClassifier runs and logged as TOOL_ERROR in either
// registration order, except when plugin/retryandreflect is registered first
// for a call to a tool that is not a FunctionTool or to a missing tool that no
// BeforeToolCallback answered: ADK skips AfterToolCallback for those calls and
// plugin/retryandreflect answers in OnToolErrorCallback, so this plugin never
// receives the error and logs no outcome row. (If a BeforeToolCallback answers
// a missing-tool lookup with a non-nil error, ADK runs the error and
// after-tool callbacks, so TOOL_ERROR is written.) When Config.ContentFormatter
// is set or "content" is in Config.PayloadColumnDenylist, the TOOL_ERROR row
// of a plugin/retryandreflect recovery payload writes a fixed fallback
// error_message rather than copying error_details or error_type from the map.
// Recoveries by other plugins registered before this plugin emit
// TOOL_COMPLETED when AfterToolCallback runs (such as BeforeToolCallback map
// answers or FunctionTool OnToolErrorCallback map recoveries, unless classified
// as a failure by Config.ToolResultClassifier), whereas OnToolErrorCallback map
// recoveries on a missing tool (that no BeforeToolCallback answered) or a
// non-FunctionTool skip AfterToolCallback and write no outcome row.
func NewBigQueryAgentAnalyticsPluginWithConfig(
	ctx context.Context,
	config Config,
) (*baseplugin.Plugin, error) {
	if !config.Enabled {
		return newDisabledPlugin()
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

	// Setup Storage Write Client
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, config.ClientOptions...)
	if err != nil {
		if closeErr := closeBQClient(bqClient); closeErr != nil {
			config.logf("failed to close bigquery client: %v", closeErr)
		}
		return nil, fmt.Errorf("failed to create BigQuery write client: %w", err)
	}

	return newBigQueryAgentAnalyticsPlugin(ctx, config, bqClient, writeClient, true)
}

var closeBQClient = func(c *bq.Client) error {
	return c.Close()
}

type dropStatsProvider interface {
	DropStats() DropStats
}

// pluginRegistry maps weak pointers of constructed ADK plugins to their
// internal pluginState so callers of NewBigQueryAgentAnalyticsPlugin* can inspect
// drop stats and sync views via AsBigQueryAgentAnalyticsPlugin both before and
// after Close without pinning unclosed plugins in memory.
var pluginRegistry sync.Map // map[weak.Pointer[baseplugin.Plugin]]*pluginState

func registerPlugin(adkPlugin *baseplugin.Plugin, state *pluginState) {
	if adkPlugin == nil {
		return
	}
	key := weak.Make(adkPlugin)
	pluginRegistry.Store(key, state)
	runtime.AddCleanup(adkPlugin, func(k weak.Pointer[baseplugin.Plugin]) {
		pluginRegistry.Delete(k)
	}, key)
}

func newDisabledPlugin() (*baseplugin.Plugin, error) {
	adkPlugin, err := baseplugin.New(baseplugin.Config{
		Name: pluginName,
		CloseFunc: func() error {
			return nil
		},
	})
	if err != nil {
		return nil, err
	}
	registerPlugin(adkPlugin, &pluginState{enabled: false, closed: true})
	return adkPlugin, nil
}

// pluginState holds the operational state of a constructed BigQuery Agent
// Analytics plugin independently of the public [baseplugin.Plugin]. It is the
// value of the plugin's pluginRegistry entry, which a global map holds until
// the plugin is collected, so nothing it refers to may lead back to the
// plugin. It therefore keeps no Config, whose Logger or ContentFormatter can
// refer to an object that holds the plugin, and no batchProcessor, which
// keeps the Config. Its view runner refers to the BigQuery client, which can
// refer to the caller's objects, only weakly, while the plugin itself keeps
// the client alive. The startup view pass forgets its context when it ends.
type pluginState struct {
	counters *dropCounters
	// enabled is false for a disabled plugin, which neither writes rows nor
	// creates views.
	enabled bool
	// defs and defsErr are the views of the plugin's Config, or why it has
	// none, from [Config.ViewDefinitions].
	defs            []ViewDefinition
	defsErr         error
	evalDefs        []ViewDefinition
	evalDefsErr     error
	evalConfigErr   error
	createEvalViews bool
	runner          viewRunner
	viewGate        chan struct{}
	viewsMu         sync.Mutex
	lifeCtx         context.Context
	lifeCancel      context.CancelFunc
	viewsCancel     context.CancelFunc
	asyncViewsDone  chan struct{}
	closed          bool
}

// BigQueryAgentAnalyticsPlugin wraps an ADK [baseplugin.Plugin] and exposes
// BigQuery Agent Analytics operational metrics and helpers such as [DropStats]
// and [BigQueryAgentAnalyticsPlugin.SyncViews].
type BigQueryAgentAnalyticsPlugin struct {
	*baseplugin.Plugin
	*pluginState
	stats dropStatsProvider
}

// DropStats returns cumulative event-drop and formatter-failure counts.
// TotalDrops sums locally discarded events and writes whose success could not
// be confirmed (QueueFullDrops + AppendFailureDrops + RowErrorDrops +
// ShutdownDrops); it can include rows committed before an acknowledgment was
// lost, so it is an upper bound. FormatterFailureDrops separately counts
// payload replacement or omission, which does not itself discard a row.
func (p *BigQueryAgentAnalyticsPlugin) DropStats() DropStats {
	if p == nil || p.stats == nil {
		return DropStats{}
	}
	return p.stats.DropStats()
}

// AsBigQueryAgentAnalyticsPlugin returns the [BigQueryAgentAnalyticsPlugin]
// associated with p if p was constructed by this package. It remains available
// after p.Close() so callers holding p can inspect final [DropStats] (including
// any losses during the shutdown drain); the registry entry is released
// automatically when p becomes unreachable.
func AsBigQueryAgentAnalyticsPlugin(p *baseplugin.Plugin) (*BigQueryAgentAnalyticsPlugin, bool) {
	if p == nil {
		return nil, false
	}
	v, ok := pluginRegistry.Load(weak.Make(p))
	if !ok {
		return nil, false
	}
	state, ok := v.(*pluginState)
	if !ok {
		return nil, false
	}
	return &BigQueryAgentAnalyticsPlugin{
		Plugin:      p,
		pluginState: state,
		stats:       state.counters,
	}, true
}

// NewBigQueryAgentAnalyticsPluginWithClients creates a plugin that writes rows
// through writeClient and manages the table and views through bqClient.
//
// When bqClient is not nil, it creates the events table if it does not exist
// and, unless DisableSchemaUpgrade is set, adds any missing columns to an
// existing table. Any failure to read, create, or upgrade the table fails
// construction and closes writeClient. Passing bqClient == nil skips table
// inspection, creation, and schema upgrades for pre-provisioned append-only
// deployments. ctx bounds this table setup, creating the Cloud Storage client
// via storage.NewClient when GCSBucketName is set (with GCSOffloader nil and
// content_parts logging enabled), and, when ExactlyOnceDelivery is enabled, the
// creation of the write stream. Background writes, stream rotations, and Close
// use only the values carried by ctx, so canceling ctx after construction
// neither stops nor drains the plugin; only closing the plugin does (see
// [Config.ShutdownTimeout]). If config.ProjectID is empty or
// bigquery.DetectProjectID, the project of bqClient is used for the table, the
// views and the write stream; when config.ProjectID names another project,
// bqClient checks, creates and upgrades the events table (and views) in
// config.ProjectID rather than bqClient.Project().
//
// bqClient is also used after construction, by background view creation and by
// [BigQueryAgentAnalyticsPlugin.SyncViews], so it must stay open and usable
// until the returned plugin is closed. The plugin owns writeClient and closes
// it when the plugin is closed. It also closes writeClient right away when
// config.Enabled is false, and before it returns any error, such as an invalid
// config or a table that cannot be read, created or upgraded. bqClient and a
// supplied GCSOffloader remain caller-owned, and the caller is responsible for
// closing them. Closing the plugin is idempotent across concurrent and repeated
// calls: shutdown runs once, converts any panic on the calling goroutine in a
// cleanup step into an error so remaining cleanup still executes, and returns
// the cached error result on every call.
//
// Register the plugin before any plugin whose callbacks return replacement
// events, model responses, tool results or agent content. ADK runs the
// callbacks of each kind in registration order and stops at the first plugin
// that returns a non-nil result or an error, so a plugin that is registered
// earlier hides what it replaces from this one. This plugin then misses the
// rows of what was hidden and, because it infers failures from the callbacks
// that it saw, it can end a successful run with AGENT_ERROR and
// INVOCATION_ERROR rows instead of INVOCATION_COMPLETED. In particular, the
// plugin tells that a model call is streaming from the partial events that
// its OnEventCallback sees, so if an earlier plugin replaces a partial event,
// a caller that stops reading mid-stream is reported as a failure.
//
// Tool errors handled by plugin/retryandreflect are recognized from its
// reserved response_type ("ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN") before
// Config.ToolResultClassifier runs and logged as TOOL_ERROR in either
// registration order, except when plugin/retryandreflect is registered first
// for a call to a tool that is not a FunctionTool or to a missing tool that no
// BeforeToolCallback answered: ADK skips AfterToolCallback for those calls and
// plugin/retryandreflect answers in OnToolErrorCallback, so this plugin never
// receives the error and logs no outcome row. (If a BeforeToolCallback answers
// a missing-tool lookup with a non-nil error, ADK runs the error and
// after-tool callbacks, so TOOL_ERROR is written.) When Config.ContentFormatter
// is set or "content" is in Config.PayloadColumnDenylist, the TOOL_ERROR row
// of a plugin/retryandreflect recovery payload writes a fixed fallback
// error_message rather than copying error_details or error_type from the map.
// Recoveries by other plugins registered before this plugin emit
// TOOL_COMPLETED when AfterToolCallback runs (such as BeforeToolCallback map
// answers or FunctionTool OnToolErrorCallback map recoveries, unless classified
// as a failure by Config.ToolResultClassifier), whereas OnToolErrorCallback map
// recoveries on a missing tool (that no BeforeToolCallback answered) or a
// non-FunctionTool skip AfterToolCallback and write no outcome row.
func NewBigQueryAgentAnalyticsPluginWithClients(
	ctx context.Context,
	config Config,
	bqClient *bq.Client,
	writeClient *bqstorage.BigQueryWriteClient,
) (*baseplugin.Plugin, error) {
	return newBigQueryAgentAnalyticsPlugin(ctx, config, bqClient, writeClient, false)
}

func newBigQueryAgentAnalyticsPlugin(
	ctx context.Context,
	config Config,
	bqClient *bq.Client,
	writeClient *bqstorage.BigQueryWriteClient,
	ownBQClient bool,
) (*baseplugin.Plugin, error) {
	adkPlugin, _, err := newBigQueryAgentAnalyticsPluginWithTraces(ctx, config, bqClient, writeClient, ownBQClient)
	return adkPlugin, err
}

func newBigQueryAgentAnalyticsPluginWithTraces(
	ctx context.Context,
	config Config,
	bqClient *bq.Client,
	writeClient *bqstorage.BigQueryWriteClient,
	ownBQClient bool,
) (*baseplugin.Plugin, *traceManager, error) {
	closeClients := func() {
		closeWriteClient(writeClient, config)
		if ownBQClient && bqClient != nil {
			if closeErr := closeBQClient(bqClient); closeErr != nil {
				config.logf("failed to close bigquery client: %v", closeErr)
			}
		}
	}
	if !config.Enabled {
		// The disabled plugin never writes, so it does not keep the clients.
		closeClients()
		adkPlugin, err := newDisabledPlugin()
		return adkPlugin, nil, err
	}
	config = applyDefaults(config)
	deniedColumns, err := validateConfig(config)
	if err != nil {
		closeClients()
		return nil, nil, err
	}
	if writeClient == nil {
		return nil, nil, errors.New("agentanalytics: writeClient must not be nil when plugin is enabled")
	}
	validateEventFilterLists(config)

	// The plugin owns, and closes, an offloader that it builds itself. One
	// that the caller supplies stays the caller's. When content_parts is
	// disabled or denied, nothing can be offloaded, so no Cloud Storage client
	// is created.
	_, partsDenied := deniedColumns[columnContentParts]
	var ownedOffloader *gcsOffloader
	if config.GCSOffloader == nil && config.GCSBucketName != "" {
		if partsDenied || !config.logMultiModalContent() {
			config.logf("agentanalytics: GCSBucketName %q is ignored because content_parts logging is disabled", config.GCSBucketName)
		} else {
			gcsOpts := config.GCSClientOptions
			if len(gcsOpts) == 0 {
				gcsOpts = config.ClientOptions
			}
			o, err := newGCSOffloader(ctx, config.GCSBucketName, gcsOpts...)
			if err != nil {
				closeClients()
				return nil, nil, fmt.Errorf("failed to create GCS offloader: %w", err)
			}
			ownedOffloader = o
			config.GCSOffloader = o
		}
	}
	closeOwnedResources := func() {
		closeClients()
		if ownedOffloader != nil {
			if closeErr := ownedOffloader.Close(); closeErr != nil {
				config.logf("failed to close GCS offloader: %v", closeErr)
			}
		}
	}

	// A client made for bq.DetectProjectID, or a ProjectID left empty, knows the
	// project that the table, the views and the write stream belong to.
	if bqClient != nil && (config.ProjectID == "" || config.ProjectID == bq.DetectProjectID) {
		config.ProjectID = bqClient.Project()
	}

	// Keep the schema callbacks in locals and clear them on config so long-lived
	// copies (the batch processor, trace logger method value, and analyticsPlugin)
	// do not retain anything the closures capture once startup finishes.
	onSchemaReady, onSchemaError := config.OnSchemaReady, config.OnSchemaError
	config.OnSchemaReady = nil
	config.OnSchemaError = nil

	// Ensure table exists and optionally upgrade schema (done once per instance).
	// OnSchemaError hears of a table that cannot be read, created or upgraded;
	// OnSchemaReady hears of a ready table once the views are ready too (below).
	tableChecked := false
	if bqClient != nil {
		ds := bqClient.Dataset(config.DatasetID)
		if config.ProjectID != "" {
			// Check the table in the project that the rows and views go to.
			ds = bqClient.DatasetInProject(config.ProjectID, config.DatasetID)
		}
		tableRef := ds.Table(config.TableName)
		if err := ensureTableAndSchema(ctx, bqTableAdapter{table: tableRef}, config, deniedColumns); err != nil {
			notifySchemaError(onSchemaError, err, config.logf)
			closeOwnedResources()
			return nil, nil, err
		}
		tableChecked = true
	}

	// Determine destination stream: default stream unless ExactlyOnceDelivery
	// is enabled, in which case newBatchProcessor creates a COMMITTED stream.
	streamName := ""
	if !config.ExactlyOnceDelivery {
		streamName = fmt.Sprintf("projects/%s/datasets/%s/tables/%s/streams/_default", config.ProjectID, config.DatasetID, config.TableName)
	}

	processor, err := newBatchProcessor(ctx, writeClient, streamName, config)
	if err != nil {
		closeOwnedResources()
		return nil, nil, fmt.Errorf("failed to initialize batch processor: %w", err)
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

	lifeCtx, lifeCancel := context.WithCancel(context.Background())
	defs, defsErr := config.ViewDefinitions()
	evalDefs, evalDefsErr := config.EvalViewDefinitions()
	evalConfigErr := validateEvalViewConfig(config, deniedColumns, false)
	state := &pluginState{
		counters:        processor.counters,
		enabled:         true,
		defs:            defs,
		defsErr:         defsErr,
		evalDefs:        evalDefs,
		evalDefsErr:     evalDefsErr,
		evalConfigErr:   evalConfigErr,
		createEvalViews: config.CreateEvalViews,
		viewGate:        make(chan struct{}, 1),
		lifeCtx:         lifeCtx,
		lifeCancel:      lifeCancel,
	}
	state.viewGate <- struct{}{}
	if bqClient != nil {
		state.runner = bqViewRunner{client: weak.Make(bqClient), projectID: config.ProjectID, datasetID: config.DatasetID}
	}

	var adkPlugin *baseplugin.Plugin
	adkPlugin, err = baseplugin.New(baseplugin.Config{
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

		CloseFunc: sync.OnceValue(func() (err error) {
			var closeErrs []error
			defer func() {
				// The view runner refers to bqClient only weakly, so this function,
				// which the plugin holds, keeps the client alive for the plugin.
				runtime.KeepAlive(bqClient)
				err = errors.Join(closeErrs...)
			}()
			runCloseStep := func(step string, fn func() error) {
				defer func() {
					if r := recover(); r != nil {
						panicType := fmt.Sprintf("%T", r)
						closeErrs = append(closeErrs, fmt.Errorf("agentanalytics: %s panicked of type %s", step, panicType))
					}
				}()
				if stepErr := fn(); stepErr != nil {
					closeErrs = append(closeErrs, stepErr)
				}
			}
			runCloseStep("stop async view creation", func() error {
				state.stopAsyncViewCreation()
				return nil
			})
			// The processor uploads while it flushes, so it goes first.
			runCloseStep("close batch processor", func() error {
				processor.Close()
				return nil
			})
			if ownedOffloader != nil {
				runCloseStep("close GCS offloader", func() error {
					if err := ownedOffloader.Close(); err != nil {
						return fmt.Errorf("failed to close GCS offloader: %w", err)
					}
					return nil
				})
			}
			if writeClient != nil {
				runCloseStep("close BigQuery write client", func() error {
					if err := writeClient.Close(); err != nil {
						return fmt.Errorf("failed to close BigQuery write client: %w", err)
					}
					return nil
				})
			}
			if ownBQClient && bqClient != nil {
				runCloseStep("close BigQuery client", func() error {
					if err := closeBQClient(bqClient); err != nil {
						return fmt.Errorf("failed to close BigQuery client: %w", err)
					}
					return nil
				})
			}
			return nil
		}),
	})
	if err != nil {
		processor.Close()
		closeOwnedResources()
		return nil, nil, err
	}

	traces.selfPlugin = weak.Make(adkPlugin)
	registerPlugin(adkPlugin, state)

	if (config.createViews() || config.CreateEvalViews) && state.runner != nil {
		logf := config.viewLogf()
		var startupDefs []ViewDefinition
		hasPerEventViews := false
		viewValidationFailed := false
		if config.createViews() {
			if defsErr != nil {
				viewValidationFailed = true
				if logf != nil {
					logf("agentanalytics: skipping view creation: %v", defsErr)
				}
			} else {
				startupDefs = append(startupDefs, defs...)
				hasPerEventViews = true
			}
		}
		if config.CreateEvalViews {
			if evalErr := cmp.Or(evalConfigErr, evalDefsErr); evalErr != nil {
				viewValidationFailed = true
				if logf != nil {
					logf("agentanalytics: skipping view creation: %v", evalErr)
				}
			} else {
				startupDefs = append(startupDefs, evalDefs...)
			}
		}
		if len(startupDefs) > 0 {
			retryMethod := "SyncViews"
			if !hasPerEventViews {
				retryMethod = "SyncEvalViews"
			}
			readyFn := onSchemaReady
			if viewValidationFailed {
				// One requested view group failed validation, so views are not
				// ready as a whole; do not invoke OnSchemaReady.
				readyFn = nil
			}
			// Pass only the non-default Logger hook so the goroutine does not
			// retain Config. The pass calls OnSchemaReady once every view is
			// created, and forgets the callback when it ends.
			state.startAsyncViewCreationWithRetryHint(ctx, startupDefs, logf, retryMethod, bqClient, readyFn)
		}
	} else if tableChecked {
		// No view pass runs, so the schema is ready once the table is.
		notifySchemaReady(onSchemaReady, config.logf)
	}
	return adkPlugin, traces, nil
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
		// The fields are listed one by one because %+v would also print the
		// blank field that prevents unkeyed RetryConfig literals.
		rc := c.RetryConfig
		return nil, fmt.Errorf("invalid config: invalid RetryConfig: needs MaxRetries >= 0, InitialDelay >= 0, Multiplier >= 1 and MaxDelay >= InitialDelay, got {MaxRetries:%v InitialDelay:%v MaxDelay:%v Multiplier:%v}", rc.MaxRetries, rc.InitialDelay, rc.MaxDelay, rc.Multiplier)
	}
	denied, err := parsePayloadColumnDenylist(c.PayloadColumnDenylist)
	if err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if _, ok := denied[columnAttributes]; ok && len(c.CustomMetadataAllowlist) > 0 {
		return nil, errors.New("invalid config: CustomMetadataAllowlist cannot be used when PayloadColumnDenylist contains \"attributes\"")
	}
	if c.ViewPrefix != "" && !isValidIdentifier(c.ViewPrefix) {
		return nil, fmt.Errorf("invalid config: invalid ViewPrefix %q", c.ViewPrefix)
	}
	if c.CreateEvalViews {
		if err := validateEvalViewConfig(c, denied, true); err != nil {
			return nil, err
		}
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
	processor          *batchProcessor
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
	// parts holds pre-formatted content_parts entries (e.g. for AGENT_RESPONSE).
	parts []map[string]any
	// partsTruncated reports that parts were truncated while they were built.
	partsTruncated bool
	// attributes are event-specific attributes, written next to the common
	// ones.
	attributes map[string]any
	// adkExtras holds extra fields merged into attributes.adk (e.g. pause_kind).
	adkExtras map[string]any
	// errMsg is written to error_message. Callbacks that set it also set
	// failed.
	errMsg string
	// failed sets the status column to "ERROR".
	failed         bool
	functionCallID string
	// sourceEvent is the session event the row describes, if any.
	sourceEvent    *session.Event
	author         string
	customMetadata map[string]any
	// customMetadataScrubbed reports that customMetadata was already run
	// through scrubA2ACustomMetadata.
	customMetadataScrubbed bool
	// customMetadataLost holds the keys of customMetadata whose values lost
	// content to a safety cap or placeholder when scrubbed (see
	// scrubA2ACustomMetadata).
	customMetadataLost map[string]bool
	// timestamp overrides span.timestamp when non-zero. Live callbacks leave
	// it zero and set span.timestamp from the trace manager's clock; a
	// mid-run purged activation (see purgeActivationLocked) sets it to
	// lastActivity while span.timestamp is zero, so its AGENT_ERROR row is
	// timestamped before the next attempt's AGENT_STARTING; end-of-run rows
	// from popUnclosedAgentSpans leave both zero so buildRow writes time.Now().
	timestamp time.Time
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

// shouldLogSessionEvent applies EventDenylist and EventAllowlist for rows
// derived from session.Event, preserving the legacy "EVENT" filter name as
// an alias for the session-event family while keeping per-type denylist
// precedence.
func (p *analyticsPlugin) shouldLogSessionEvent(eventType string) bool {
	if _, ok := p.denylist[legacyEventType]; ok {
		return false
	}
	if _, ok := p.denylist[eventType]; ok {
		return false
	}
	if len(p.allowlist) > 0 {
		_, allowLegacy := p.allowlist[legacyEventType]
		_, allowType := p.allowlist[eventType]
		return allowLegacy || allowType
	}
	return true
}

func (p *analyticsPlugin) denied(column string) bool {
	_, ok := p.deniedColumns[column]
	return ok
}

// suppressesToolPayload reports whether content controls omit or transform raw
// tool result payloads so payload fields must not be copied verbatim into
// error_message.
func (p *analyticsPlugin) suppressesToolPayload() bool {
	return p.denied(columnContent) || p.config.ContentFormatter != nil
}

func (p *analyticsPlugin) recordPanic(format string, args ...any) {
	if p.config.Enabled && p.processor != nil {
		p.processor.recordAppendFailureDrop(1)
	}
	safeLogf(p.config.logf, format, args...)
}

// recoverCallback logs and suppresses panics raised inside plugin callbacks so
// that analytics logging never breaks the agent run.
func (p *analyticsPlugin) recoverCallback(eventType string) {
	if r := recover(); r != nil {
		panicType := fmt.Sprintf("%T", r)
		p.recordPanic("agentanalytics: dropped %s event: recovered panic of type %s", eventType, panicType)
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
	p.writeEvent(ctx, id, eventType, span, build)
}

func (p *analyticsPlugin) logSessionEvent(ctx context.Context, id identity, eventType string, span rowSpan, build func(*eventData)) {
	if !p.shouldLogSessionEvent(eventType) {
		return
	}
	p.writeEvent(ctx, id, eventType, span, build)
}

// logSessionSpanEvent runs spanFn before consulting shouldLogSessionEvent, so
// session-event rows emitted outside OnEventCallback (such as user-message HITL
// completions and resumed TOOL_COMPLETED rows) honor the legacy "EVENT" filter
// alias while keeping trace manager span side effects intact.
func (p *analyticsPlugin) logSessionSpanEvent(ctx context.Context, id identity, eventType string, spanFn func() rowSpan, build func(*eventData)) {
	defer p.recoverCallback(eventType)
	var span rowSpan
	if spanFn != nil {
		span = spanFn()
	}
	if !p.shouldLogSessionEvent(eventType) {
		return
	}
	p.writeEvent(ctx, id, eventType, span, build)
}

func (p *analyticsPlugin) writeEvent(ctx context.Context, id identity, eventType string, span rowSpan, build func(*eventData)) {
	defer p.recoverCallback(eventType)
	d := &eventData{span: span}
	if build != nil {
		build(d)
	}
	p.processor.Append(p.buildRow(ctx, id, eventType, d))
}

func (p *analyticsPlugin) onUserMessage(ctx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
	id := identityOf(ctx)
	var span rowSpan
	p.logEvent(ctx, id, EventUserMessageReceived, func() rowSpan {
		span = p.traces.invocationSpan(ctx, id, runnerAgentOf(ctx))
		return span
	}, func(d *eventData) {
		d.content = msg
		d.author = userAuthor
	})
	p.emitUserMessageSignalEvents(ctx, id, msg, span)
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
// cancellation, agents stopped with an open model or tool call, open agent
// spans whose activation produced no response, and spans whose callback
// context ended with a cause other than context.Canceled or bare
// context.DeadlineExceeded (such as a parallel agent's errgroup cause when a
// branch fails). Unless a parallel branch's Run yields an error (leaving its
// own span open with the errgroup cause) or another failure of a parallel
// branch cancels a still-running sibling with the branch's error, a run that
// fails after producing a response, such as a custom agent outside a parallel
// branch that writes an event and then returns an error, or an
// AfterAgentCallback that returns an error when no parallel sibling is still
// open, ends with INVOCATION_COMPLETED, and so does a run in which a sub-agent
// fails without any callback after an agent that it runs under produced
// content: no plugin callback tells either apart from a successful run whose
// caller stopped reading once it had the response. Conversely, a custom agent
// that handles a sub-agent's silent error without writing a final event of its
// own on the same branch and node path while the ancestor spans between it and
// the unresponded sub-agent are still open (reading an intermediate sub-agent
// such as a sequential agent to its end closes that intermediate span and
// leaves its unresponded child beyond completeResponse's open-span walk;
// BeforeAgentCallback answers mark the sub-agent's own InvocationContext as
// ended and do not depend on OnEventCallback), or that handles a parallel
// agent's error (unless every branch had responded, no canceled sibling
// recorded a collateral context.Canceled error, and the custom agent reads the
// parallel agent to its end and then writes a final event of its own), is
// reported as failed; see traceManager.popUnclosedAgentSpans for the full
// failure precedence, including three RetryConfig shapes that still end
// INVOCATION_ERROR after the retry succeeds (a function node with RetryConfig
// whose function runs an agent, an agent node with RetryConfig inside a nested
// workflow.NewWorkflowNode, and a workflow retry whose earlier attempt relies
// on endRestartedNodeRuns when the caller stops reading before the workflow
// agent ends).
//
// The rules rely on this plugin seeing the callbacks of the run. ADK stops
// each callback chain at the first plugin that returns a non-nil result or an
// error, so the plugin has to be registered before plugins that return
// replacement events, model responses, tool results or agent content; see
// [NewBigQueryAgentAnalyticsPluginWithClients].
func (p *analyticsPlugin) afterRun(ctx agent.InvocationContext) {
	defer p.recoverCallback(EventInvocationCompleted)
	id := identityOf(ctx)
	target := p.traces.captureRunEndTarget(id.invocationID)
	var terminalEnded, flushed bool
	// Register the run-end flush before endInvocationSafe so LIFO defer order
	// runs endInvocationSafe first while still running flushRunEndSafe even if
	// afterRun panics before reaching the terminal logEvent.
	defer func() {
		if p.config.flushOnRunEnd() && !flushed {
			p.flushRunEndSafe()
		}
	}()
	defer func() {
		if !terminalEnded {
			p.endInvocationSafe(ctx, id, target)
		}
	}()
	unclosed, unrecErr := p.traces.popUnclosedAgentSpans(ctx, id)
	target = p.traces.refreshRunEndTarget(id.invocationID, target)
	if unrecErr == nil && ctx != nil {
		// A canceled invocation failed even when no span is left open, for
		// example when canceled before any agent started or while post-run
		// session compaction ran.
		unrecErr = orZero(ctx.Err)
	}
	p.logAgentErrors(ctx, id, unclosed, unrecErr)

	if unrecErr != nil || len(unclosed) > 0 {
		p.logEvent(ctx, id, EventInvocationError, func() rowSpan {
			span := p.traces.endInvocationTarget(ctx, id, &target)
			terminalEnded = true
			return span
		}, func(d *eventData) {
			d.failed = true
			if unrecErr != nil {
				d.errMsg = unrecErr.Error()
				d.content = map[string]any{"error_traceback": unrecErr.Error()}
			}
		})
	} else {
		p.logEvent(ctx, id, EventInvocationCompleted, func() rowSpan {
			span := p.traces.endInvocationTarget(ctx, id, &target)
			terminalEnded = true
			return span
		}, nil)
	}
	if !terminalEnded {
		terminalEnded = true
		p.endInvocationSafe(ctx, id, target)
	}

	if p.config.flushOnRunEnd() {
		flushed = true
		p.flushRunEndSafe() // flush queue synchronously under ShutdownTimeout
	}
}

func (p *analyticsPlugin) endInvocationSafe(ctx context.Context, id identity, target runEndTarget) {
	defer func() {
		if r := recover(); r != nil {
			if id.invocationID != "" && p.traces != nil {
				func() {
					defer func() { _ = recover() }()
					p.traces.forceEndInvocationTarget(id, target)
				}()
			}
			panicType := fmt.Sprintf("%T", r)
			p.recordPanic("agentanalytics: endInvocation fallback: recovered panic of type %s", panicType)
		}
	}()
	if p.traces != nil {
		p.traces.endInvocationRetry(ctx, id, target)
	}
}

func (p *analyticsPlugin) flushRunEndSafe() {
	defer func() {
		if r := recover(); r != nil {
			panicType := fmt.Sprintf("%T", r)
			p.recordPanic("agentanalytics: flushRunEnd fallback: recovered panic of type %s", panicType)
		}
	}()
	if p.processor != nil {
		p.processor.flushRunEnd()
	}
}

// logAgentErrors writes an AGENT_ERROR row for each activation in unclosed of
// the invocation that id belongs to, with the activation's own error, or else
// with err.
func (p *analyticsPlugin) logAgentErrors(ctx context.Context, id identity, unclosed []unclosedAgentSpan, err error) {
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
			d.timestamp = u.timestamp
			agentErr := u.err
			if agentErr == nil {
				agentErr = err
			}
			if agentErr != nil {
				d.errMsg = agentErr.Error()
				d.content = map[string]any{"error_traceback": agentErr.Error()}
			}
			d.author = u.agentName
		})
	}
}

// logPurgedAgents writes the AGENT_ERROR rows of the activations that a
// callback of the invocation that id belongs to removed because ADK handled
// their failure itself (see invocationTrace.purgeActivationLocked): such an
// activation gets no AfterAgentCallback, and its failure does not fail the
// invocation.
func (p *analyticsPlugin) logPurgedAgents(ctx context.Context, id identity, purged []unclosedAgentSpan) {
	p.logAgentErrors(ctx, id, purged, nil)
}

// onEvent writes the rows of a session event. The event callback gets the
// agent that the runner runs, so the event is placed by its author, branch
// and node path. A non-partial event with content or a typed output also
// records that its author responded: ADK runs no AfterModelCallback for a
// model call that a BeforeModelCallback answered, so the event is what ends
// such a call, and a custom agent or workflow node may respond with an output
// alone. The function responses of such an event end the tool calls that they
// answer, which an earlier plugin's AfterToolCallback may have hidden from
// this plugin. Streamed partial events end nothing: AfterModelCallback ends
// the model call with the final response, which carries the call's latency.
// A partial event, with or without content, records instead that its author
// is streaming, so that a caller that stops reading on it is not taken for a
// failure, and one with content or a typed output also records that its
// author responded (see recordStreamed).
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
		if ev.Partial {
			p.traces.recordStreamed(ctx, evID, ev.Author, ev.Branch, ev.Content != nil || ev.Output != nil)
		} else if ev.Content != nil || ev.Output != nil {
			p.traces.completeResponseWithActions(ctx, evID, ev.Author, ev.Branch, functionResponseCalls(ev.Content), &ev.Actions)
		} else {
			p.traces.clearLastPartial(ctx, evID)
		}
		if workflowName, failed, ok := nodeCompletion(ev); ok {
			p.traces.recordNodeCompletion(ctx, evID, workflowName, ev.NodeInfo.Path, ev.Branch, failed)
		}
		p.emitSessionEventSignals(ctx, ev)
		if !ev.Partial && ev.Author != "" && ev.Author != userAuthor {
			taskID := evID
			taskID.agentName = ev.Author
			taskID.branch = ev.Branch
			taskOut := classifyTaskResponsesWithPolicy(ev.Content, p.suppressesToolPayload())
			evDispatch := dispatchMapsOf(&ev.Actions)
			if len(taskOut.retiredCallIDs) > 0 || len(taskOut.overruledEndingCallIDs) > 0 {
				p.traces.retireFailedTaskAttemptsForEvent(ctx, taskID, taskOut.retiredCallIDs, taskOut.overruledEndingCallIDs, taskOut.leadingActionCallID, evDispatch, taskOut.mergedResponse, ev.Timestamp)
			}
			// ADK ends a task-mode LLM agent on the first finish_task
			// FunctionResponse when it carries the completion sentinel, so a
			// first response that also reports a tool result error ends the
			// activation with AGENT_ERROR instead of AGENT_COMPLETED, while a
			// non-ending first response keeps the task running even if a later
			// part in the same event carries the sentinel.
			if taskOut.ends {
				if span, ok := p.traces.popTaskAgentSpanForEvent(ctx, taskID, taskOut.endingCallIDs, taskOut.leadingActionCallID, evDispatch, taskOut.failed, taskOut.mergedResponse, ev.Timestamp); ok {
					if taskOut.failed {
						p.logAgentErrors(ctx, taskID, []unclosedAgentSpan{{
							span:      span,
							agentName: taskID.agentName,
							branch:    taskID.branch,
							path:      taskID.path,
							err:       errors.New(taskFailedPrefix + taskOut.errMsg),
						}}, nil)
					} else {
						p.logEvent(ctx, taskID, EventAgentCompleted, func() rowSpan { return span }, nil)
					}
				}
			}
		}
	}
	return nil, nil
}

// hasEventCreationProvenance is a heuristic that reports whether ev looks like
// it was created via session.NewEvent rather than left without an ID until
// runner fallback timestamping. It is NOT a strict proof of creation-time vs
// delivery-time stamping: a custom agent can still construct
// &session.Event{ID: platform.NewUUID(ctx)} with a zero Timestamp that
// runner.go later stamps at delivery time, when a custom
// platform.WithUUIDProvider is active on ctx any non-empty ev.ID passes, and
// stored events re-emitted or forwarded from external session storage carry
// whatever ID and Timestamp that storage assigned.
func hasEventCreationProvenance(ctx context.Context, ev *session.Event) bool {
	if ev == nil || ev.ID == "" {
		return false
	}
	return hasNewEventUUID(ev.ID) || contextHasCustomUUIDProvider(ctx)
}

// hasEventCreationProvenance is the package-level heuristic for an event ev
// that OnEventCallback delivers with ctx and the identity id read from it,
// except that an ID that is not a UUIDv4 passes only if the agent activations
// that created ev ran with a custom platform.UUIDProvider, as recorded for the
// invocation in ev.InvocationID, or in id when ev has none (see
// traceManager.eventUUIDProvider), rather than if ctx carries one: ctx is the
// runner's context, so it neither shows a provider that a custom agent
// installed for a sub-agent nor a nil one that turned the runner's provider
// off. When ev.InvocationID differs from id.invocationID and tm does not track
// ev.InvocationID, a tracked id.invocationID fails closed rather than letting
// a stale or foreign ev.InvocationID fall back to ctx; ctx decides only when
// neither invocation is tracked. As with the package-level helper, stored
// events re-emitted or forwarded from external session storage carry whatever
// ID and Timestamp external storage assigned (the inherent boundary of
// in-process UUID-provider tracking).
func (tm *traceManager) hasEventCreationProvenance(ctx context.Context, id identity, ev *session.Event) bool {
	if ev == nil || ev.ID == "" {
		return false
	}
	if hasNewEventUUID(ev.ID) {
		return true
	}
	var evNodePath string
	if ev.NodeInfo != nil {
		evNodePath = ev.NodeInfo.Path
	}
	if ev.InvocationID != "" && ev.InvocationID != id.invocationID {
		// A custom agent that yields the events of a nested runner hands on
		// events of the nested invocation, whose activations are recorded
		// under its ID when that runner shares this plugin.
		if custom, tracked := tm.eventUUIDProvider(ev.InvocationID, ev.Author, ev.Branch, evNodePath); tracked {
			return custom
		}
		if tm.isTrackedInvocation(id.invocationID) {
			return false
		}
		return contextHasCustomUUIDProvider(ctx)
	}
	path := cmp.Or(evNodePath, id.path)
	if custom, tracked := tm.eventUUIDProviderFor(cmp.Or(ev.InvocationID, id.invocationID), ev.Author, ev.Branch, path, id.agentName, id.branch); tracked {
		return custom
	}
	return contextHasCustomUUIDProvider(ctx)
}

// hasNewEventUUID reports whether id has the canonical 36-byte RFC 4122 UUIDv4
// format produced by session.NewEvent's default platform.NewUUID generator.
func hasNewEventUUID(id string) bool {
	if len(id) != 36 {
		return false
	}
	for i := range 36 {
		c := id[i]
		switch i {
		case 8, 13, 18, 23:
			if c != '-' {
				return false
			}
		case 14:
			if c != '4' {
				return false
			}
		case 19:
			switch c {
			case '8', '9', 'a', 'b', 'A', 'B':
			default:
				return false
			}
		default:
			if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
				return false
			}
		}
	}
	return true
}

// nodeCompletion reports whether ev is the event that ADK's workflow
// scheduler writes when a node run ends without an output, which the
// scheduler marks with a node outcome or a completed flag in its custom
// metadata, and if so the name of the workflow agent whose scheduler ran the
// node and whether the run failed or was canceled (see
// traceManager.recordNodeCompletion). The scheduler records its own node
// path, the parent of the event's node path, under its workflow agent's name
// (see workflowBasePathPrefix), as it does on the events of the nodes that it
// runs, which therefore also carry the base paths of enclosing workflows. An
// event whose node path has no unique such workflow is not attributed. The
// record that the scheduler writes when a resumed run ends with an output
// carries adk.workflow.node_resume_completed instead, and is not counted.
func nodeCompletion(ev *session.Event) (workflowName string, failed, ok bool) {
	if ev == nil || ev.Partial || ev.NodeInfo == nil || ev.NodeInfo.Path == "" {
		return "", false, false
	}
	outcome, hasOutcome := ev.CustomMetadata[workflowNodeOutcomeKey].(string)
	completed, _ := ev.CustomMetadata[workflowNodeCompletedKey].(bool)
	if !hasOutcome && !completed {
		return "", false, false
	}
	var parent string
	if slash := strings.LastIndexByte(ev.NodeInfo.Path, '/'); slash >= 0 {
		parent = ev.NodeInfo.Path[:slash]
	}
	for key, value := range ev.CustomMetadata {
		name, isBasePath := strings.CutPrefix(key, workflowBasePathPrefix)
		if base, isString := value.(string); !isBasePath || !isString || base != parent {
			continue
		}
		if workflowName != "" {
			return "", false, false
		}
		workflowName = name
	}
	if workflowName == "" {
		return "", false, false
	}
	return workflowName, outcome == workflowNodeFailedOutcome || outcome == workflowNodeCancelOutcome, true
}

// finishTaskToolName is the name of the tool with which a task-mode LLM agent
// completes its task.
const finishTaskToolName = "finish_task"

// taskFailedPrefix starts the error message of the AGENT_ERROR row of a
// task-mode LLM agent activation that a finish_task response with a classified
// tool result error ended (see taskEndResult).
const taskFailedPrefix = "finish_task ended the task with a tool result error: "

// isTaskEndingResult reports whether ADK ends a task-mode LLM agent on the
// finish_task response map resp, which it does when resp carries the
// completion sentinel ("result": "Task completed."), whether or not resp also
// reports a tool result error (see taskEndResult).
func isTaskEndingResult(resp map[string]any) bool {
	res, ok := resp["result"].(string)
	return ok && res == "Task completed."
}

// taskEndResult reports whether ADK ends a task-mode LLM agent on the
// finish_task response map resp, which it does when resp carries the
// completion sentinel ("result": "Task completed."), and whether resp also
// carries a classified tool result error (such as top-level boolean isError or
// is_error), with its diagnostic. A task that ends with such an error has
// failed rather than completed.
func taskEndResult(resp map[string]any) (ends, failed bool, errMsg string) {
	return taskEndResultWithPolicy(resp, false)
}

func taskEndResultWithPolicy(resp map[string]any, suppressPayload bool) (ends, failed bool, errMsg string) {
	if !isTaskEndingResult(resp) {
		return false, false, ""
	}
	if msg, ok := retryAnswerError(resp, suppressPayload); ok {
		return true, true, msg
	}
	errMsg, failed = mcpResultError(resp)
	return true, failed, errMsg
}

// taskResponseOutcome summarizes the finish_task function responses in a
// session event's content in a single pass. ADK's isFinishTaskSuccessFR
// inspects only the first finish_task FunctionResponse in an event:
//
//   - If the first finish_task response carries the completion sentinel, ends
//     is true, failed and errMsg come from taskEndResult on that first
//     response, endingCallIDs holds the non-empty call IDs of all ending
//     finish_task responses in part order (with endingCallID set to the first
//     such call ID) so a trailing observed call ID can still resolve the
//     activation when an earlier ending response had an empty or unobserved
//     ID, and retiredCallIDs holds the non-empty call IDs of any subsequent
//     non-ending finish_task responses in the same event.
//   - If the first finish_task response does not carry the completion sentinel,
//     ends is false (ADK keeps the task running), non-ending responses with
//     non-empty call IDs are added to retiredCallIDs, and any subsequent
//     sentinel-bearing responses with non-empty call IDs behind the failed
//     first one are added to overruledEndingCallIDs so their observed ending
//     attempts are retired even if the failed first response had an empty ID.
type taskResponseOutcome struct {
	ends                   bool
	failed                 bool
	mergedResponse         bool
	errMsg                 string
	leadingActionCallID    string
	endingCallID           string
	endingCallIDs          []string
	retiredCallIDs         []string
	overruledEndingCallIDs []string
}

func classifyTaskResponses(c *genai.Content) taskResponseOutcome {
	return classifyTaskResponsesWithPolicy(c, false)
}

func classifyTaskResponsesWithPolicy(c *genai.Content, suppressPayload bool) (out taskResponseOutcome) {
	if c == nil {
		return out
	}
	seenFirst := false
	funcRespCount := 0
	for _, part := range c.Parts {
		if part == nil || part.FunctionResponse == nil {
			continue
		}
		fr := part.FunctionResponse
		funcRespCount++
		if funcRespCount == 1 && fr.Name == finishTaskToolName && fr.ID != "" {
			out.leadingActionCallID = fr.ID
		}
		if fr.Name != finishTaskToolName {
			continue
		}
		if !seenFirst {
			seenFirst = true
			out.ends, out.failed, out.errMsg = taskEndResultWithPolicy(fr.Response, suppressPayload)
			if out.ends {
				out.endingCallID = fr.ID
				if fr.ID != "" {
					out.endingCallIDs = append(out.endingCallIDs, fr.ID)
				}
			} else if fr.ID != "" {
				out.retiredCallIDs = append(out.retiredCallIDs, fr.ID)
			}
			continue
		}
		if fr.ID == "" {
			continue
		}
		if !isTaskEndingResult(fr.Response) {
			out.retiredCallIDs = append(out.retiredCallIDs, fr.ID)
		} else if out.ends {
			if out.endingCallID == "" {
				out.endingCallID = fr.ID
			}
			out.endingCallIDs = append(out.endingCallIDs, fr.ID)
		} else {
			out.overruledEndingCallIDs = append(out.overruledEndingCallIDs, fr.ID)
		}
	}
	out.mergedResponse = funcRespCount > 1
	return out
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

// beforeAgent writes the AGENT_STARTING row of an agent. When a coordinator
// starts again after the task-mode sub-agent that it delegated to completed,
// it first writes the AGENT_COMPLETED row of the activation that the
// delegation interrupted, which ADK ends without an AfterAgentCallback.
func (p *analyticsPlugin) beforeAgent(ctx agent.Context) (*genai.Content, error) {
	defer p.recoverCallback(EventAgentStarting)
	id := identityOf(ctx)
	if span, ok := p.traces.popDelegatingAgentSpan(ctx, id); ok {
		p.logEvent(ctx, id, EventAgentCompleted, func() rowSpan { return span }, nil)
	}
	// pushAgentSpan runs before the AGENT_STARTING row so that logPurgedAgents
	// can write the AGENT_ERROR rows of the previous attempt before the new
	// attempt's row, with their own timestamps (see unclosedAgentSpan).
	span, purged := p.traces.pushAgentSpan(ctx, id)
	p.logPurgedAgents(ctx, id, purged)
	p.logEvent(ctx, id, EventAgentStarting, func() rowSpan {
		return span
	}, func(d *eventData) {
		// The context that ADK gives agent callbacks logs a warning and
		// returns nil from Agent, so the agent is looked up by name in the
		// tree of the agent that the runner runs.
		d.content, d.attributes = agentStartingData(p.traces.agentNamed(ctx, id))
	})
	return nil, nil
}

func (p *analyticsPlugin) afterAgent(ctx agent.Context) (*genai.Content, error) {
	defer p.recoverCallback(EventAgentCompleted)
	id := identityOf(ctx)
	// A workflow agent ends only after every node run that its scheduler
	// started, so the failed attempts of nodes that it started again are
	// purged here, before its AGENT_COMPLETED row (see endRestartedNodeRuns).
	p.logPurgedAgents(ctx, id, p.traces.endRestartedNodeRuns(ctx, id))
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
		// A streamed chunk times the first token when it carries generated
		// content: a leading chunk of usage or role metadata is not the first
		// token. It does not mark the call as streaming: agent
		// AfterModelCallbacks run after this one and may still reject the
		// chunk, so onEvent does that for the partial event that ADK emits once
		// the chunk is accepted. adk-python writes a row for every chunk; the
		// final response alone is written here so that token usage is counted
		// once.
		if hasGeneratedContent(res) {
			p.traces.recordFirstToken(ctx, id)
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
	defer p.recoverCallback(EventToolStarting)
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	if name == finishTaskToolName {
		p.traces.noteTaskStart(ctx, id, callID)
	}
	p.logEvent(ctx, id, EventToolStarting, func() rowSpan {
		return p.traces.pushToolSpan(ctx, id, toolKey(id, callID, name))
	}, func(d *eventData) {
		origin := p.resolveToolOrigin(ctx, id, t, args)
		d.content = toolContent(name, "args", args, origin)
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

const (
	// reflectAndRetryResponseType matches the unexported response_type marker
	// emitted by plugin/retryandreflect (and adk-python's ReflectAndRetryToolPlugin)
	// when it recovers a tool error into retry guidance. Kept here because
	// plugin/agentanalytics requires google.golang.org/adk/v2 v2.0.0.
	reflectAndRetryResponseType         = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
	reflectAndRetryFallbackErrorMessage = "Tool raised an error that ReflectAndRetryToolPlugin handled."
)

// retryAnswerError reports whether result is a recovery payload produced by
// plugin/retryandreflect, returning the extracted error message and true.
// Only exact string values on "error_details" and "error_type" are accepted;
// non-string values are ignored and fall back to
// reflectAndRetryFallbackErrorMessage. When suppressPayload is true (because
// ContentFormatter is set or "content" is in PayloadColumnDenylist), it returns
// reflectAndRetryFallbackErrorMessage directly rather than copying fields from
// the recovery map into error_message.
func retryAnswerError(result map[string]any, suppressPayload bool) (string, bool) {
	rt, ok := result["response_type"].(string)
	if !ok || rt != reflectAndRetryResponseType {
		return "", false
	}
	if suppressPayload {
		return reflectAndRetryFallbackErrorMessage, true
	}
	for _, k := range []string{"error_details", "error_type"} {
		if s, ok := result[k].(string); ok && s != "" {
			return s, true
		}
	}
	return reflectAndRetryFallbackErrorMessage, true
}

// isToolConfirmationOutcome reports whether err is a tool confirmation request
// or rejection. Both outcomes complete the tool call in adk-python rather than
// logging TOOL_ERROR. If inspecting err panics (for example, Unwrap on a typed
// nil *fs.PathError), it reports false so the caller records an ordinary tool
// failure.
func isToolConfirmationOutcome(err error) bool {
	if err == nil {
		return false
	}
	return orZero(func() bool {
		return errors.Is(err, tool.ErrConfirmationRequired) || errors.Is(err, tool.ErrConfirmationRejected)
	})
}

func (p *analyticsPlugin) afterTool(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	defer p.recoverCallback(EventToolCompleted)
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	span, purged, closedByError := p.traces.popToolSpan(ctx, id, toolKey(id, callID, name), false)
	// An agent that the call ran as a tool and that failed ends with it.
	p.logPurgedAgents(ctx, id, purged)
	// A tool that waits for or is rejected by the user's confirmation fails
	// with tool.ErrConfirmationRequired or tool.ErrConfirmationRejected before
	// it runs. adk-python returns the confirmation outcome as the tool's
	// result, so, as there, the call completes with the error that ADK sends
	// the model as its result.
	confirmationOutcome := isToolConfirmationOutcome(err)
	if name == finishTaskToolName {
		// Record every finish_task attempt on the calling activation before
		// error returns or classification run. ADK emits a FunctionResponse
		// carrying callID even when AfterToolCallback receives err != nil or a
		// confirmation outcome, and a later plugin or agent-level
		// AfterToolCallback can promote a failure into a completion (or rewrite
		// a completion into an error) before that FunctionResponse arrives.
		// Non-completion FunctionResponse events retire their own attempt in
		// onEvent so reused call IDs cannot claim or retire another run's
		// attempt. Whether an attempt ended the task, and failed it, follows
		// the result that ADK sees (see taskEndResult), not the verdict of
		// Config.ToolResultClassifier, which decides only this call's row.
		var ends, failed bool
		if err == nil && !confirmationOutcome {
			ends, failed, _ = taskEndResult(result)
		}
		p.traces.noteTaskAttempt(ctx, id, callID, ends, failed)
	}
	if closedByError {
		// onToolError already wrote TOOL_ERROR; even when an agent
		// OnToolErrorCallback or a later plugin recovered or replaced the error,
		// suppress TOOL_COMPLETED and AGENT_RESPONSE.
		return nil, nil
	}
	if err != nil && !confirmationOutcome {
		// When an earlier plugin short-circuited OnToolErrorCallback before
		// onToolError ran and passed the error through, emit TOOL_ERROR here.
		p.logToolError(ctx, id, t, func() rowSpan { return span }, name, callID, args, err)
		return nil, nil
	}
	if confirmationOutcome {
		// The tool has not run. ADK sends the model the confirmation outcome
		// as the call's result, as adk-python does, so the row carries it,
		// and the classifier sees it, as it does there.
		result = map[string]any{"error": err.Error()}
	}
	if msg, failed := p.classifyToolResultError(ctx, t, args, result); failed {
		// This includes the answer of a plugin/retryandreflect registered
		// before this plugin, which short-circuits OnToolErrorCallback and
		// returns a recovery payload with err == nil: it is logged as
		// TOOL_ERROR, without TOOL_COMPLETED and AGENT_RESPONSE.
		p.logToolError(ctx, id, t, func() rowSpan { return span }, name, callID, args, errors.New(msg))
		return nil, nil
	}
	p.logEvent(ctx, id, EventToolCompleted, func() rowSpan { return span }, func(d *eventData) {
		origin := p.resolveToolOrigin(ctx, id, t, args)
		d.content = toolContent(name, "result", result, origin)
		d.functionCallID = callID
	})
	if confirmationOutcome {
		// The tool has not run, so a final response tool has no response
		// yet: AGENT_RESPONSE is written when the confirmed call runs.
		return nil, nil
	}
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
	if isToolConfirmationOutcome(err) {
		// The call waits for or was rejected by the user's confirmation and has
		// not failed. Its span stays open for the TOOL_COMPLETED row of afterTool.
		return nil, nil
	}
	id := identityOf(ctx)
	callID := orZero(ctx.FunctionCallID)
	name := toolName(t)
	if name == finishTaskToolName {
		p.traces.noteTaskStart(ctx, id, callID)
	}
	span, purged, _ := p.traces.popToolSpan(ctx, id, toolKey(id, callID, name), true)
	// An agent that the call ran as a tool and that failed ends with it.
	p.logPurgedAgents(ctx, id, purged)
	p.logToolError(ctx, id, t, func() rowSpan {
		return span
	}, name, callID, args, err)
	return nil, nil
}

const mcpToolErrorMessage = "Tool returned an MCP result with isError=true."

// classifiedToolErrorMessage is the error_message of a result that
// Config.ToolResultClassifier records as TOOL_ERROR without a message of its
// own. It matches adk-python, whose classifier is tool_result_classifier, so
// that rows from either language can be queried together.
const classifiedToolErrorMessage = "tool_result_classifier classified the tool result as an error."

func (p *analyticsPlugin) logToolError(ctx context.Context, id identity, t tool.Tool, spanFn func() rowSpan, name, callID string, args map[string]any, err error) {
	p.logEvent(ctx, id, EventToolError, spanFn, func(d *eventData) {
		origin := p.resolveToolOrigin(ctx, id, t, args)
		d.content = toolContent(name, "args", args, origin)
		d.failed = true
		if err != nil {
			// An error whose Error method returns "" or panics, as a typed nil
			// pointer to a type with a value receiver Error method does, is
			// recorded by its bare type name to match adk-python.
			d.errMsg = cmp.Or(orZero(err.Error), errorTypeName(err))
		}
		d.functionCallID = callID
	})
}

// errorTypeName returns the bare type name of err, matching adk-python's
// type(err).__name__ fallback when Error() is empty or panics.
func errorTypeName(err error) string {
	return valueTypeName(err)
}

// valueTypeName returns the bare type name of v after unwrapping pointers,
// stopping on self-referential pointer chains such as `type P *P` and falling
// back to t.String() for unnamed types.
func valueTypeName(v any) string {
	t := reflect.TypeOf(v)
	if t == nil {
		return ""
	}
	var seen map[reflect.Type]bool
	for t.Kind() == reflect.Pointer {
		elem := t.Elem()
		if elem == t || seen[elem] {
			break
		}
		if seen == nil {
			seen = make(map[reflect.Type]bool)
		}
		seen[t] = true
		t = elem
	}
	return cmp.Or(t.Name(), t.String())
}

func mcpResultError(result map[string]any) (string, bool) {
	if len(result) == 0 {
		return "", false
	}
	for _, k := range []string{"isError", "is_error"} {
		if v, ok := result[k].(bool); ok && v {
			return mcpToolErrorMessage, true
		}
	}
	return "", false
}

// classifyToolResultError returns the error_message of the TOOL_ERROR row of a
// result that a tool returned without an error, and true, or false to record
// the result as TOOL_COMPLETED. A plugin/retryandreflect recovery payload is
// recorded as TOOL_ERROR first (honoring content controls on error_message),
// matching adk-python regardless of registration order or classifier verdicts.
// For other results, Config.ToolResultClassifier decides next: its
// ToolResultError verdict records the row, with the classifier's ErrorMessage,
// or classifiedToolErrorMessage when that is empty, as the error_message, and
// its ToolResultOK verdict completes the call, whatever the result holds. When
// the classifier is nil, abstains, panics or returns a decision that is not
// defined, mcpResultError decides with the remaining built-in rules.
func (p *analyticsPlugin) classifyToolResultError(ctx agent.Context, t tool.Tool, args, result map[string]any) (string, bool) {
	if msg, ok := retryAnswerError(result, p.suppressesToolPayload()); ok {
		return msg, true
	}
	if p.config.ToolResultClassifier != nil {
		switch verdict := p.runToolResultClassifier(ctx, t, args, result); verdict.Decision {
		case ToolResultOK:
			return "", false
		case ToolResultError:
			return cmp.Or(verdict.ErrorMessage, classifiedToolErrorMessage), true
		}
	}
	return mcpResultError(result)
}

// runToolResultClassifier calls Config.ToolResultClassifier and returns its
// verdict, or an abstaining verdict when the classifier panicked or returned
// a decision that is not defined, so that the call is recorded as if the
// classifier abstained. Warnings omit the tool name (which can be an
// unregistered model-hallucinated call answered by a BeforeToolCallback) and
// format only the panic type string or numeric decision so neither the tool
// name nor the raw panic value reaches Config.Logger.
func (p *analyticsPlugin) runToolResultClassifier(ctx agent.Context, t tool.Tool, args, result map[string]any) (verdict ToolResultVerdict) {
	defer func() {
		if r := recover(); r != nil {
			verdict = ToolResultVerdict{}
			safeLogf(p.config.logf, "agentanalytics: ToolResultClassifier failed: recovered panic of type %s; recording the call as if the classifier abstained", fmt.Sprintf("%T", r))
		}
	}()
	verdict = p.config.ToolResultClassifier(ctx, t, args, result)
	switch verdict.Decision {
	case ToolResultAbstain, ToolResultOK, ToolResultError:
		return verdict
	}
	safeLogf(p.config.logf, "agentanalytics: ToolResultClassifier returned an unusable verdict %d; recording the call as if the classifier abstained", int(verdict.Decision))
	return ToolResultVerdict{}
}

// toolName returns the name of t, or "" if t is nil or its Name method
// panics.
func toolName(t tool.Tool) string {
	if t == nil {
		return ""
	}
	return orZero(t.Name)
}

func toolContent(name, payloadKey string, payload map[string]any, origin string) map[string]any {
	return map[string]any{
		"tool":        name,
		payloadKey:    payload,
		"tool_origin": cmp.Or(origin, toolOriginUnknown),
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
// system instruction. Credentials in the summaries are redacted if redact is
// set; the caller always says so, so that no call can leave them unredacted by
// omission. If parts is not nil, it also collects the content_parts of the
// messages followed by those of the system instruction, redacted as the
// options of parts say. The content ordinal of a message is its index in
// req.Contents, which a nil message keeps to itself, and that of the system
// instruction is len(req.Contents). The ordinals keep the objects that hold
// the parts of one request apart.
func llmRequestContent(req *model.LLMRequest, maxLen int, redact bool, parts *partsCollector) (content map[string]any, contentTruncated bool) {
	content, contentTruncated, _ = buildLLMRequestContent(req, maxLen, redact, false, parts)
	return content, contentTruncated
}

func buildLLMRequestContent(req *model.LLMRequest, maxLen int, redact, contentDenied bool, parts *partsCollector) (content map[string]any, contentTruncated bool, builder *contentSummaryBuilder) {
	b := &contentSummaryBuilder{
		isLLMRequest:  true,
		contentDenied: contentDenied,
		maxLen:        maxLen,
		redact:        redact,
	}
	for i, c := range req.Contents {
		if c == nil {
			continue
		}
		role, t := sanitizeAndTruncateText(cmp.Or(c.Role, "unknown"), maxLen, redact)
		b.roleTruncated = b.roleTruncated || t
		msg := newMessageSummary(c, i, maxLen, redact)
		msg.role = role
		msg.hasRole = true
		b.messages = append(b.messages, msg)
		if parts != nil {
			parts.addContent(c, i)
		}
	}
	if req.Config != nil && req.Config.SystemInstruction != nil {
		ord := len(req.Contents)
		msg := newMessageSummary(req.Config.SystemInstruction, ord, maxLen, redact)
		msg.isSystem = true
		b.messages = append(b.messages, msg)
		if parts != nil {
			parts.addContent(req.Config.SystemInstruction, ord)
		}
	}
	if parts != nil {
		for _, u := range parts.pending {
			if u.isText {
				b.omitTextFromSummary(u.contentOrd, u.partIdx)
			}
		}
	}
	content, contentTruncated = b.buildContent()
	b.needsRebuild = false
	return content, contentTruncated, b
}

// buildLLMResponseContent builds the LLM_RESPONSE content: the formatted
// response and the token usage, as in adk-python, plus content_parts when
// parts is not nil, which adk-python does not build for responses. It returns
// nil content if res carries neither.
func buildLLMResponseContent(res *model.LLMResponse, maxLen int, redact, contentDenied bool, parts *partsCollector) (content map[string]any, contentTruncated bool, builder *contentSummaryBuilder) {
	b := &contentSummaryBuilder{
		isLLMResponse: true,
		contentDenied: contentDenied,
		maxLen:        maxLen,
		redact:        redact,
	}
	if res.Content != nil {
		b.messages = []*messageSummary{newResponseMessageSummary(res.Content, 0, maxLen, redact)}
		if parts != nil {
			parts.addContent(res.Content, 0)
			// Go differs from adk-python here. adk-python formats the response
			// into content.response before it parses the content, so the parser
			// neither builds content_parts nor offloads the response text. Go
			// builds content_parts for res.Content, and leaves the text of every
			// part that is offloaded out of content.response: a response that is
			// a single text above MaxInlineTextSize is written as
			// {"response":""}, and the LLM_RESPONSE view that adk-python defines
			// over the table, which reads $.response, shows an empty string for
			// it. The omission keeps a batch of large responses below the size
			// limit of an AppendRows request. If the upload fails, the truncated
			// text is restored.
			for _, u := range parts.pending {
				if u.isText {
					b.omitTextFromSummary(u.contentOrd, u.partIdx)
				}
			}
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
		b.usage = usage
	}
	content, contentTruncated = b.buildContent()
	b.needsRebuild = false
	if len(content) == 0 {
		return nil, false, nil
	}
	return content, contentTruncated, b
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
// with nil content). ADK Go neither inherits models from ancestor agents nor
// exposes a parent link on agent.Agent, so only a's own configured Model is
// recorded.
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

func isADKStructPkgType(owner reflect.Type) bool {
	return isADKStructPkg(owner.PkgPath())
}

// agentConfig returns the configuration that ADK keeps in the base agent of
// a, such as the llmagent.Config of an LLM agent, or nil. ADK exposes no
// accessor for it, so it is read with reflection from the exported Config
// field of the base agent's embedded ADK state. User decorators embedding an
// agent.Agent (directly or through nested anonymous structs or non-nil struct
// pointers, without holding concurrent-swap locks over Agent) are unwrapped
// safely; Config is read only when declared on an ADK struct so mutable user
// fields named Config are never touched.
func agentConfig(a agent.Agent) any {
	for range 8 {
		var seen map[visitedContextStruct]bool
		v, ok := derefContextStructValueLazy(reflect.ValueOf(a), &seen, true)
		if !ok {
			return nil
		}
		if f, owner := promotedStructFieldOwned(v, "Config", isADKStructPkgType); f.IsValid() && isADKStructPkg(owner.PkgPath()) && f.CanInterface() && !f.IsZero() {
			return f.Interface()
		}
		inner, _ := promotedStructField(v, "Agent")
		if !inner.IsValid() || !inner.CanInterface() {
			return nil
		}
		next, ok := reflect.TypeAssert[agent.Agent](inner)
		if !ok {
			return nil
		}
		a = next
	}
	return nil
}

type agentAfterCBLayout struct {
	configIndex  []int
	afterCBIndex []int
	agentIndex   []int
}

var agentAfterCBLayouts sync.Map // map[reflect.Type]agentAfterCBLayout

func fieldPathExported(t reflect.Type, idx []int) bool {
	if len(idx) == 0 {
		return false
	}
	for _, i := range idx {
		if t.Kind() == reflect.Pointer {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct {
			return false
		}
		sf := t.Field(i)
		if !sf.IsExported() {
			return false
		}
		t = sf.Type
	}
	return true
}

func lookupAgentAfterCBLayout(t reflect.Type) agentAfterCBLayout {
	if cached, ok := agentAfterCBLayouts.Load(t); ok {
		return cached.(agentAfterCBLayout)
	}
	var layout agentAfterCBLayout
	if sf, ok := t.FieldByName("Config"); ok && fieldPathExported(t, sf.Index) {
		layout.configIndex = sf.Index
	}
	if sf, ok := t.FieldByName("afterAgentCallbacks"); ok {
		layout.afterCBIndex = sf.Index
	}
	if sf, ok := t.FieldByName("Agent"); ok && fieldPathExported(t, sf.Index) {
		layout.agentIndex = sf.Index
	}
	actual, _ := agentAfterCBLayouts.LoadOrStore(t, layout)
	return actual.(agentAfterCBLayout)
}

func safeFieldByIndex(v reflect.Value, idx []int) (reflect.Value, bool) {
	if len(idx) == 0 {
		return reflect.Value{}, false
	}
	for _, i := range idx {
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				return reflect.Value{}, false
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return reflect.Value{}, false
		}
		v = v.Field(i)
	}
	return v, true
}

// agentHasAfterAgentCallbacks reports whether a configures (or conservatively
// may configure, when a's concrete layout cannot be unwrapped to ADK's base
// agent within the bounded hop limit) agent-level AfterAgentCallbacks, which
// ADK invokes after plugins' AfterAgentCallback hooks finish.
func agentHasAfterAgentCallbacks(a agent.Agent) (res bool) {
	if a == nil {
		return false
	}
	defer func() {
		if r := recover(); r != nil {
			res = true
		}
	}()
	for hop := range 4 {
		v := reflect.ValueOf(a)
		for v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
			if v.IsNil() {
				return hop > 0
			}
			v = v.Elem()
		}
		if v.Kind() != reflect.Struct {
			return true
		}
		vt := v.Type()
		layout := lookupAgentAfterCBLayout(vt)
		if vt.PkgPath() == agentPkgPath && vt.Name() == "agent" {
			if len(layout.afterCBIndex) == 0 {
				return true
			}
			cb, ok := safeFieldByIndex(v, layout.afterCBIndex)
			if !ok || cb.Kind() != reflect.Slice {
				return true
			}
			return cb.Len() > 0
		}
		if f, ok := safeFieldByIndex(v, layout.configIndex); ok && f.CanInterface() {
			if cfg, ok := reflect.TypeAssert[llmagent.Config](f); ok && len(cfg.AfterAgentCallbacks) > 0 {
				return true
			}
			if cfg, ok := reflect.TypeAssert[*llmagent.Config](f); ok && cfg != nil && len(cfg.AfterAgentCallbacks) > 0 {
				return true
			}
		}
		if len(layout.afterCBIndex) > 0 {
			cb, ok := safeFieldByIndex(v, layout.afterCBIndex)
			if !ok || cb.Kind() != reflect.Slice || cb.Len() > 0 {
				return true
			}
		}
		inner, ok := safeFieldByIndex(v, layout.agentIndex)
		if !ok || !inner.CanInterface() {
			return true
		}
		next, ok := reflect.TypeAssert[agent.Agent](inner)
		if !ok || next == nil {
			return true
		}
		a = next
	}
	return true
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

// directStructField returns the direct field of struct v named name, or the
// zero reflect.Value if v has no direct field with that name. Looking up
// t.Field(i) / v.Field(i) directly avoids reflect.Value.FieldByName, which
// chases promoted fields and panics on nil embedded struct pointers.
func directStructField(v reflect.Value, name string) reflect.Value {
	t := v.Type()
	for i := range t.NumField() {
		if t.Field(i).Name == name {
			return v.Field(i)
		}
	}
	return reflect.Value{}
}

// promotedStructField looks up a field named name on struct v or promoted
// through nested anonymous structs and non-nil struct pointers (up to 8
// embedding levels), returning the matching field value and the struct type
// that declares it, or the zero Value when absent or ambiguous at the shallowest
// matching depth. Walking anonymous fields level by level matches Go's selector
// promotion rules without calling reflect.Value.FieldByName, which panics when
// an unrelated nil embedded struct pointer shares or precedes the lookup.
func promotedStructField(v reflect.Value, name string) (reflect.Value, reflect.Type) {
	return promotedStructFieldOwned(v, name, nil)
}

func directStructFieldIndex(t reflect.Type, name string) (int, bool) {
	for i := range t.NumField() {
		if t.Field(i).Name == name {
			return i, true
		}
	}
	return -1, false
}

func findPromotedFieldPath(t reflect.Type, name string, maxDepth int, allowOwner func(reflect.Type) bool) ([]int, reflect.Type, bool) {
	type typeNode struct {
		typ  reflect.Type
		path []int
	}
	minDepth := map[reflect.Type]int{t: 0}
	curLevel := []typeNode{{typ: t}}
	for depth := range maxDepth {
		nextDepth := depth + 1
		var nextLevel []typeNode
		levelCount := make(map[reflect.Type]int)
		matches := 0
		var matchPath []int
		var matchOwner reflect.Type
		for _, node := range curLevel {
			for i := range node.typ.NumField() {
				sf := node.typ.Field(i)
				if !sf.Anonymous {
					continue
				}
				var subType reflect.Type
				switch sf.Type.Kind() {
				case reflect.Struct:
					subType = sf.Type
				case reflect.Pointer:
					if sf.Type.Elem().Kind() != reflect.Struct {
						continue
					}
					subType = sf.Type.Elem()
				default:
					continue
				}
				if prev, seen := minDepth[subType]; seen && prev < nextDepth {
					continue
				}
				if allowOwner == nil || allowOwner(subType) {
					if idx, ok := directStructFieldIndex(subType, name); ok {
						matches++
						if matches > 1 {
							return nil, nil, false
						}
						matchPath = make([]int, len(node.path)+2)
						copy(matchPath, node.path)
						matchPath[len(node.path)] = i
						matchPath[len(node.path)+1] = idx
						matchOwner = subType
					}
				}
				levelCount[subType]++
				if levelCount[subType] <= 2 {
					minDepth[subType] = nextDepth
					nextPath := make([]int, len(node.path)+1)
					copy(nextPath, node.path)
					nextPath[len(node.path)] = i
					nextLevel = append(nextLevel, typeNode{typ: subType, path: nextPath})
				}
			}
		}
		if matches == 1 {
			return matchPath, matchOwner, true
		}
		if len(nextLevel) == 0 {
			break
		}
		curLevel = nextLevel
	}
	return nil, nil, false
}

func promotedStructFieldOwned(v reflect.Value, name string, allowOwner func(reflect.Type) bool) (reflect.Value, reflect.Type) {
	if allowOwner == nil || allowOwner(v.Type()) {
		if f := directStructField(v, name); f.IsValid() {
			return f, v.Type()
		}
	}
	const maxPromoDepth = 8
	hasAnonStruct := false
	vt := v.Type()
	for sf := range vt.Fields() {
		if sf.Anonymous && (sf.Type.Kind() == reflect.Struct || (sf.Type.Kind() == reflect.Pointer && sf.Type.Elem().Kind() == reflect.Struct)) {
			hasAnonStruct = true
			break
		}
	}
	if !hasAnonStruct {
		return reflect.Value{}, nil
	}
	path, owner, ok := findPromotedFieldPath(vt, name, maxPromoDepth, allowOwner)
	if !ok {
		return reflect.Value{}, nil
	}
	cur := v
	for _, idx := range path[:len(path)-1] {
		cur = cur.Field(idx)
		if cur.Kind() == reflect.Pointer {
			if cur.IsNil() {
				return reflect.Value{}, nil
			}
			cur = cur.Elem()
		}
	}
	return cur.Field(path[len(path)-1]), owner
}

func isNilContextField(v reflect.Value) bool {
	for v.IsValid() && v.Kind() == reflect.Interface {
		if v.IsNil() {
			return true
		}
		v = v.Elem()
	}
	if !v.IsValid() {
		return true
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Map, reflect.Chan, reflect.Func:
		return v.IsNil()
	default:
		return false
	}
}

var contextPathReaderOnce = sync.OnceValue(func() func(context.Context) string {
	p := any(agent.Promote(nil))
	if r, ok := p.(interface{ ADKContextPath(context.Context) string }); ok {
		return r.ADKContextPath
	}
	return nil
})

// contextPath returns the execution node path carried by ctx, or "".
// It first queries ADK's private path key via ctx.Value (when running against an
// ADK core that exposes it), then falls back to allowlisted ADK and Go stdlib
// context layouts and Path() on non-wrapper types.
func contextPath(ctx context.Context) (path string) {
	defer func() {
		if r := recover(); r != nil {
			path = ""
		}
	}()
	if ctx == nil {
		return ""
	}
	if read := contextPathReaderOnce(); read != nil {
		if s := read(ctx); s != "" {
			return s
		}
	}
	var q contextStructQueue
	q.push(reflect.ValueOf(ctx))
	var seen map[visitedContextStruct]bool
	for steps := range 512 {
		item, ok := q.pop()
		if !ok {
			break
		}
		v, ok := derefContextStructValueLazy(item, &seen, steps >= 32)
		if !ok {
			continue
		}
		vt := v.Type()
		switch vt.PkgPath() {
		case agentPkgPath:
			if vt.Name() == "commonContext" {
				if f := directStructField(v, "path"); f.IsValid() && f.Kind() == reflect.String {
					if s := f.String(); s != "" {
						return s
					}
				}
			}
		case internalContextPkgPath:
			if vt.Name() == "InvocationContext" {
				if pf := directStructField(v, "params"); pf.IsValid() {
					if p, ok := derefContextStructValueLazy(pf, &seen, steps >= 32); ok {
						if f := directStructField(p, "Path"); f.IsValid() && f.Kind() == reflect.String {
							if s := f.String(); s != "" {
								return s
							}
						}
					}
				}
			}
		}
		c1, c2 := adkContextChildren(v)
		q.push(c1)
		q.push(c2)
	}
	type pathCarrier interface {
		Path() string
	}
	if p, ok := ctx.(pathCarrier); ok {
		vt := reflect.TypeOf(ctx)
		if !(vt != nil && vt.Kind() == reflect.Pointer && vt.Elem().PkgPath() == agentPkgPath &&
			(vt.Elem().Name() == "callbackContextWrapper" || vt.Elem().Name() == "commonContext" || vt.Elem().Name() == "toolContextWrapper")) {
			return p.Path()
		}
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
	rowTS := time.Now()
	if !d.timestamp.IsZero() {
		rowTS = d.timestamp
	} else if !d.span.timestamp.IsZero() {
		rowTS = d.span.timestamp
	}
	row := map[string]any{
		timestampField:  rowTS,
		eventIDField:    eventID,
		"event_type":    eventType,
		"invocation_id": id.invocationID,
		"user_id":       id.userID,
	}
	var built bool
	defer func() {
		if !built && p.processor != nil {
			p.processor.releaseRowUploads(row)
		}
	}()
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
	attrsTruncated := false
	if !p.denied(columnAttributes) {
		attrsMap, metaTrunc := p.attributes(eventID, id, agentName, span, d)
		attrs, t := p.jsonColumn(attrsMap)
		row[columnAttributes] = attrs
		attrsTruncated = t || metaTrunc
		truncated = truncated || attrsTruncated
	}

	status := "OK"
	if d.failed {
		status = "ERROR"
	}
	row["status"] = status
	errMsgTruncated := false
	if d.errMsg != "" {
		errMsg := d.errMsg
		if p.config.redactSensitiveKeys() {
			errMsg = redactErrorMessage(errMsg)
		}
		msg, t := truncateString(errMsg, p.config.maxContentLength())
		row["error_message"] = msg
		errMsgTruncated = t
		truncated = truncated || t
	}
	if builder, ok := row[summaryBuilderKey].(*contentSummaryBuilder); ok {
		builder.otherTruncated = builder.otherTruncated || attrsTruncated || errMsgTruncated
	}
	row["is_truncated"] = truncated
	built = true
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
//
// Everything that goes into row is detached from the caller's messages. The
// row is handed to the batch processor, which writes it on another goroutine
// while the caller may already be changing the messages. Payloads that are to
// be offloaded travel with the row as copies under rawPartsKey. Synthetic
// events such as AGENT_RESPONSE that pre-populate d.parts in events.go when
// no ContentFormatter is set copy those pre-formatted entries directly and
// stay inline up to MaxContentLen rather than re-entering parts.addContent.
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
		if p.processor != nil {
			p.processor.recordFormatterFailureDrop()
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
	var parts *partsCollector
	if !partsDenied && p.config.logMultiModalContent() {
		parts = p.newPartsCollector(maxLen)
		defer func() {
			if len(parts.pending) > 0 {
				row[rawPartsKey] = parts.pending
			}
		}()
	}
	content := payload
	contentTruncated := d.contentTruncated
	preFormatted := false
	var builder *contentSummaryBuilder
	switch v := payload.(type) {
	case *genai.Content:
		if parts != nil {
			parts.addContent(v, 0)
		}
		builder = &contentSummaryBuilder{
			contentDenied: contentDenied,
			maxLen:        maxLen,
			redact:        redact,
			messages:      []*messageSummary{newMessageSummary(v, 0, maxLen, redact)},
		}
		if parts != nil {
			for _, u := range parts.pending {
				if u.isText {
					builder.omitTextFromSummary(u.contentOrd, u.partIdx)
				}
			}
		}
		content, contentTruncated = builder.buildContent()
		builder.needsRebuild = false
		preFormatted = true
	case *model.LLMRequest:
		content, contentTruncated, builder = buildLLMRequestContent(v, maxLen, redact, contentDenied, parts)
		preFormatted = true
	case *model.LLMResponse:
		content, contentTruncated, builder = buildLLMResponseContent(v, maxLen, redact, contentDenied, parts)
		preFormatted = true
	default:
		if parts != nil && len(d.parts) > 0 && p.config.ContentFormatter == nil {
			parts.parts = append(parts.parts, d.parts...)
			parts.truncated = parts.truncated || d.partsTruncated
			parts.nonUploadTruncated = parts.nonUploadTruncated || d.partsTruncated
		}
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
	if parts != nil && len(parts.parts) > 0 {
		row[columnContentParts] = parts.parts
		truncated = truncated || parts.truncated
		if len(parts.pending) > 0 {
			row[rawPartsKey] = parts.pending
			if builder != nil && parts.hasPendingText() {
				builder.otherTruncated = parts.nonUploadTruncated
				row[summaryBuilderKey] = builder
			}
		}
	}
	return truncated
}

// newPartsCollector returns the collector of the content_parts of one row.
// Offloading to Cloud Storage is on if the plugin has an offloader.
func (p *analyticsPlugin) newPartsCollector(maxLen int) *partsCollector {
	opts := partsOptions{
		maxLen:       maxLen,
		redact:       p.config.redactSensitiveKeys(),
		connectionID: p.config.ConnectionID,
	}
	if p.config.GCSOffloader != nil {
		opts.offload = true
		opts.inlineLimit = p.config.MaxInlineTextSize
		if p.processor != nil {
			opts.reserveBytes = p.processor.tryReserveUploadBytes
		}
	}
	return &partsCollector{opts: opts}
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
// followed by the common ones, plus whether scrubbing lost content of an
// entry that custom_metadata keeps to a safety cap or placeholder (see
// eventData.customMetadataLost).
func (p *analyticsPlugin) attributes(eventID string, id identity, agentName string, span trace.SpanContext, d *eventData) (map[string]any, bool) {
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
	metaTrunc := false
	if md := p.customMetadata.apply(d.customMetadata); md != nil {
		attrs["custom_metadata"] = md
		for k := range d.customMetadataLost {
			if _, ok := md[k]; ok {
				metaTrunc = true
				break
			}
		}
	}
	return attrs, metaTrunc
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
	for k, v := range d.adkExtras {
		if _, ok := adk[k]; !ok {
			adk[k] = v
		}
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
	return maps.Collect(s.All())
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
