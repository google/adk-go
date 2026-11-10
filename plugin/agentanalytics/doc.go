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

// Package agentanalytics provides the BigQuery Agent Analytics plugin for ADK Go.
//
// The plugin captures agent invocations, sub-agent activations, model requests
// and responses, tool executions, session state deltas, transfers, human-in-the-loop
// (HITL) pauses and completions, A2A interactions, state checkpoints, compactions,
// and workflow node events, and streams them asynchronously to BigQuery via the
// BigQuery Storage Write API.
//
// # Quick Start and Lifecycle
//
// Construct a plugin with [NewBigQueryAgentAnalyticsPlugin],
// [NewBigQueryAgentAnalyticsPluginWithConfig], or
// [NewBigQueryAgentAnalyticsPluginWithClients], register it in
// runner.PluginConfig.Plugins, and close it during application shutdown so
// queued events are flushed before exit:
//
//	ctx := context.Background()
//	bqPlugin, err := agentanalytics.NewBigQueryAgentAnalyticsPlugin(
//		ctx, "my-gcp-project", "agent_analytics", "agent_events",
//	)
//	if err != nil {
//		return err
//	}
//	defer bqPlugin.Close()
//
// Register this plugin before any plugin whose callbacks return replacement
// events, model responses, tool results, or agent content. ADK invokes plugin
// callbacks in registration order and stops at the first non-nil replacement or
// error, so an earlier short-circuiting plugin hides those callbacks from
// analytics tracking.
//
// Use [AsBigQueryAgentAnalyticsPlugin] to inspect cumulative [DropStats] or to
// run [BigQueryAgentAnalyticsPlugin.SyncViews] synchronously during deployment.
//
// # Schema, Views, and Storage Offloading
//
// By default, initialization verifies or creates the target BigQuery table
// using [EventsSchema] (partitioned by UTC day on timestamp and clustered by
// event_type, agent, and user_id), applies additive column upgrades unless
// [Config.DisableSchemaUpgrade] is set, and creates 25 per-event SQL views
// (such as v_llm_request, v_llm_response, and v_tool_completed; see
// [Config.ViewDefinitions]) in the background unless [Config.DisableCreateViews]
// is set. Pre-provisioned append-only deployments that lack table-read
// permissions can pass a nil BigQuery client to
// [NewBigQueryAgentAnalyticsPluginWithClients].
//
// Automatic credential and sensitive-key redaction is enabled by default unless
// [Config.DisableSensitiveKeyRedaction] is set, and string payloads are bounded
// by [Config.MaxContentLen] (with joined [EventLLMResponse] summaries in
// content.response bounded at MaxContentLen + 4096 runes and function call or
// response names in response summaries bounded at 256 runes) unless
// [Config.CaptureFullMessageContent] is set.
// Cloud Storage offloading requires content-parts logging (enabled by
// [DefaultConfig] or by setting [Config.LogMultiModalContent] to true, and not
// suppressed by [Config.DisableMultiModalContent] or by denying "content_parts"
// in [Config.PayloadColumnDenylist]). When content-parts logging is active,
// setting [Config.GCSBucketName] or [Config.GCSOffloader] offloads inline binary
// blobs and text parts exceeding [Config.MaxInlineTextSize] (default 32 KiB) or
// [Config.MaxContentLen] to Cloud Storage at batch flush time and records
// BigQuery ObjectRef entries in content_parts.
//
// # Behavioral Differences from adk-python
//
// Rows written by this package share the same table schema ([SchemaVersion] "2")
// and event_type names as adk-python's BigQuery Agent Analytics plugin so cross-language
// fleets can share one events table. Intentional behavioral differences from
// adk-python v2.13.0 are documented below for cross-language SQL authors and
// operators.
//
// Rows written and view projections:
//
//   - One [EventLLMResponse] row per streamed model call: Go emits LLM_RESPONSE
//     only for the final non-partial response, while partial chunks with
//     generated content record latency_ms.time_to_first_token_ms on that final
//     row. adk-python also writes an LLM_RESPONSE row for every partial chunk,
//     so v_llm_response row counts and summed token counts differ on streamed
//     calls. (Pinned by TestAfterModelCallback_PartialChunksDoNotEachLogUsage in
//     bigquery_agent_analytics_plugin_test.go and
//     TestTraceManager_StreamingTTFTAndSingleMarshaledLatency in
//     trace_manager_test.go).
//   - Inferred [EventAgentError] and [EventInvocationError] rows: Because ADK
//     Go does not yet expose plugin-level agent or run error callbacks, the
//     plugin infers AGENT_ERROR and INVOCATION_ERROR from open agent spans and
//     unrecovered callback or context errors. End-of-run AGENT_ERROR rows carry
//     the run-end timestamp rather than the instant of failure (activations
//     purged mid-run carry their last callback timestamp), content.error_traceback
//     holds the error's Error() text rather than a generated stack traceback,
//     and an unclosed agent span with no recorded error records a synthetic
//     `agent "<name>" execution failed before completion` message. (Pinned by
//     TestRunner_FailedModelCall_EmitsAgentErrorAndInvocationError,
//     TestRunner_BeforeModelCallbackError_EmitsAgentErrorAndInvocationError,
//     TestRunner_AgentCompletedAndEndOfRunErrorTimestampsAreNotBackdated, and
//     TestTraceManager_PurgedActivationLatencyFromLastActivity in
//     trace_manager_test.go, and
//     TestDocParity_ErrorTracebackAndUnclosedSpanSyntheticError in doc_test.go).
//   - attributes.adk.function_call_id across tool lifecycle rows: In addition
//     to [EventToolPaused], Go stamps attributes.adk.function_call_id on
//     [EventToolStarting], [EventToolCompleted], and [EventToolError] so
//     concurrent tool calls can be correlated across their lifecycle. For a
//     paused long-running tool, Go writes one TOOL_COMPLETED row for the
//     initial pending result and a second TOOL_COMPLETED row (with
//     attributes.adk.pause_kind set to "tool") when the resumed function
//     response arrives, both carrying the same function_call_id; SQL joins from
//     TOOL_PAUSED to TOOL_COMPLETED on function_call_id should filter on
//     pause_kind = 'tool' to match one resumed row per pause. (Pinned by
//     TestCallbacks_WriteRows in bigquery_agent_analytics_plugin_test.go,
//     TestSessionEvents_EmitRichLifecycleTypes in events_test.go, and
//     TestDocParity_PausedLongRunningToolWritesTwoToolCompletedRows in
//     doc_test.go).
//   - [EventAgentResponse] on final events carrying function responses: Go's
//     final-response check excludes FunctionCall parts, LongRunningToolIDs, and
//     compaction events, and accepts final events (including events with
//     SkipSummarization) even when FunctionResponse parts are present. This
//     emits AGENT_RESPONSE for ADK Go's agenttool, whose final SkipSummarization
//     event pairs the sub-agent's FunctionResponse with its answer text, whereas
//     adk-python skips events that contain function responses. (Pinned by
//     TestAgentResponse_SkipsThoughtOnlyAndPartialEvents in events_test.go and
//     TestDocParity_AgentResponseExcludesLongRunningToolsAndCompaction in
//     doc_test.go).
//   - Offloaded [EventLLMResponse] text omits inline content.response: When
//     Cloud Storage offloading is active, Go builds content_parts for
//     LLM_RESPONSE and omits offloaded text parts from content.response to keep
//     serialized batches under Write API request limits. A single offloaded text
//     response therefore writes {"response":""} (an empty string in
//     v_llm_response.response) while storing the preview and object_ref in
//     content_parts; if the upload fails, truncated inline text is restored.
//     adk-python pre-formats LLM_RESPONSE into a dictionary that its content
//     parser does not offload, keeping text inline up to max_content_length.
//     (Pinned by TestGCSOffloading_OmitsOffloadedTextFromContentSummary in
//     gcs_offloader_test.go and
//     TestDocParity_OffloadedSingleTextLLMResponseWritesEmptyResponseString in
//     doc_test.go).
//   - Invocation parent_span_id under an ambient OpenTelemetry span: When a run
//     starts inside an active OpenTelemetry span, Go adopts that trace_id and
//     makes the invocation span a child of the caller's span (setting
//     parent_span_id on invocation-span rows to the caller's span ID) so
//     multiple invocations sharing a server or request span remain separate span
//     trees. Because the caller's span has no row in the BigQuery table, root-row
//     queries in Go cannot rely solely on parent_span_id IS NULL when ambient
//     tracing is active. When no ambient span is active at run start, rows
//     written before ADK opens its first agent span ([EventUserMessageReceived]
//     and [EventInvocationStarting]) keep the plugin's initial trace_id while
//     subsequent rows adopt ADK's trace_id. (Pinned by
//     TestTraceManager_AmbientOTelTraceIDInherited and
//     TestTraceManager_AdoptFirstAgentOTelSpanWhenCallerHasNoSpan in
//     trace_manager_test.go, and TestAttributes_OptionalSections in
//     bigquery_agent_analytics_plugin_test.go).
//   - Invocation-scoped "temp:" keys in [EventStateDelta]: Go strips keys with
//     the "temp:" scratch prefix (after trimming leading whitespace) from
//     state_delta and emits no STATE_DELTA row when a delta contains only
//     "temp:" keys, whereas adk-python writes "temp:" keys in STATE_DELTA.
//     (Pinned by TestSessionEvents_EmitRichLifecycleTypes in events_test.go and
//     TestDocParity_StateDeltaTrimsLeadingWhitespaceBeforeTempPrefix in
//     doc_test.go).
//   - "EVENT" alias in [Config.EventAllowlist] and [Config.EventDenylist]: Go
//     treats "EVENT" as an umbrella alias matching all session-event-derived
//     rows (with EventDenylist taking precedence), whereas adk-python matches
//     concrete event type names literally. (Pinned by
//     TestSessionEvents_LegacyEventFiltering in events_test.go).
//   - Session-event rows skipped for partial events: Go ignores session events
//     with Partial == true when emitting session-event-derived rows, avoiding
//     duplicate STATE_DELTA, TOOL_PAUSED, and HITL_* rows during streaming;
//     adk-python checks partial only for NODE_* and AGENT_RESPONSE rows.
//     (Pinned by TestSSEStreaming_PauseAndStateDeltaEmittedOnce,
//     TestAgentResponse_SkipsThoughtOnlyAndPartialEvents, and
//     TestSessionEvents_EmitRichLifecycleTypes in events_test.go).
//   - [EventHITLInputRequest] from Event.RequestedInput: Alongside
//     adk_request_input function-call parts, Go emits HITL_INPUT_REQUEST when
//     session.Event.RequestedInput is non-nil and no adk_request_input call is
//     present on the event, whereas adk-python emits HITL rows solely from
//     function-call parts. (Pinned by TestSessionEvents_EmitRichLifecycleTypes in
//     events_test.go and TestDocParity_HITLInputRequestFromRequestedInput in
//     doc_test.go).
//
// Payload shape, metadata keys, and operational defaults:
//
//   - Go-only keys in attributes.adk: Go adds adk_version, event_id,
//     invocation_id, session_id and user_id (unless [Config.DisableSessionMetadata]
//     is set), author, isolation_scope, artifact_delta, skip_summarization,
//     escalate, and (on workflow node events) node.message_as_output and
//     node.output_for. (Pinned by TestCallbacks_WriteRows and
//     TestAttributes_OptionalSections in bigquery_agent_analytics_plugin_test.go,
//     and TestDocParity_ADKNodeAttributesAndOmissions in doc_test.go).
//   - attributes.adk.node omitted when NodeInfo is nil: Go writes
//     attributes.adk.node only when session.Event.NodeInfo is non-nil, so
//     JSON_VALUE(attributes, '$.adk.node.path') evaluates to NULL on
//     non-workflow rows rather than "" as in adk-python. (Pinned by
//     TestDocParity_ADKNodeAttributesAndOmissions in doc_test.go).
//   - render_ui_widgets and rewind_before_invocation_id omitted: ADK Go's
//     session.Event and session.EventActions do not define these fields, so
//     attributes.adk never includes them. (Pinned by
//     TestDocParity_ADKNodeAttributesAndOmissions in doc_test.go).
//   - attributes.usage_metadata shape: On [EventLLMResponse], Go writes
//     snake_case prompt_token_count, candidates_token_count, and
//     total_token_count, plus thoughts_token_count, cached_content_token_count,
//     and tool_use_prompt_token_count when greater than zero, omitting
//     per-modality breakdowns and explicit null fields. (Pinned by
//     TestCallbacks_WriteRows in bigquery_agent_analytics_plugin_test.go,
//     TestViews_JSONPathsMatchEmittedPayloads in views_test.go, and
//     TestDocParity_UsageMetadataAndContentUsageShape in doc_test.go).
//   - content.usage values: On [EventLLMResponse], Go writes integer prompt,
//     completion, and total counts (0 when omitted by the model because genai
//     token counts are int32), plus thoughts and cached when greater than zero;
//     adk-python writes only prompt, completion, and total with null for
//     omitted counts. (Pinned by TestCallbacks_WriteRows in
//     bigquery_agent_analytics_plugin_test.go and
//     TestDocParity_UsageMetadataAndContentUsageShape in doc_test.go).
//   - Tool parameter schema casing in [EventLLMRequest]: Go serializes
//     structured genai.Schema tool parameters with google.golang.org/genai's
//     camelCase JSON tags (such as anyOf and maxItems), whereas adk-python
//     serializes Pydantic schemas with snake_case field names. (Pinned by
//     TestDocParity_LLMRequestToolSchemaCamelCaseAndDeterministicOrder in
//     doc_test.go).
//   - Deterministic tool order in [EventLLMRequest]: Because model.LLMRequest.Tools
//     in ADK Go is a map, Go sorts tool declarations alphabetically by name
//     rather than preserving list insertion order. (Pinned by
//     TestLLMRequestAttributes_Tools in bigquery_agent_analytics_plugin_test.go
//     and TestDocParity_LLMRequestToolSchemaCamelCaseAndDeterministicOrder in
//     doc_test.go).
//   - Function call arguments in content_parts: For FunctionCall parts, Go
//     writes both function_name and function_call (including redacted and
//     truncated arguments) in content_parts[].part_attributes, whereas
//     adk-python v2.13.0 writes only function_name. (Pinned by
//     TestStructuredContentPartsAndSanitizedTextInCallbacks in
//     bigquery_agent_analytics_plugin_test.go,
//     TestFormatContentParts_StructuredParts in json_formatter_test.go, and
//     TestDocParity_ContentPartsFunctionCallArgsTruncated in doc_test.go).
//   - Model attribute on [EventAgentStarting]: ADK Go requires each LLM agent
//     to configure its own Model and exposes no parent-agent link on
//     agent.Agent, so attributes.model records only the agent's directly
//     configured model rather than walking ancestor agents. (Pinned by
//     TestAgentStartingData and TestAgentStarting_LooksUpAgentInRunnerTree in
//     trace_manager_test.go).
//   - [ContentFormatter] invocation contract: Go skips [Config.ContentFormatter]
//     on rows whose payload content is nil and passes the live Go value before
//     column formatting (such as *model.LLMRequest on [EventLLMRequest] and
//     *model.LLMResponse on [EventLLMResponse]); adk-python calls
//     content_formatter on every row and passes the pre-formatted
//     {"response", "usage"} dictionary on LLM_RESPONSE. (Pinned by
//     TestContentFormatter in bigquery_agent_analytics_plugin_test.go,
//     TestGCSOffloading_ContentFormatterRunsBeforeTheUpload in
//     gcs_offloader_test.go, and
//     TestDocParity_ContentFormatterReceivesTypedModelPayloadsAndSkipsNilContent
//     in doc_test.go).
//   - latency_ms.time_to_first_token_ms on [EventLLMError]: Go records
//     time_to_first_token_ms alongside total_ms on LLM_ERROR whenever a
//     streamed chunk with generated content arrived before the model call
//     failed, whereas adk-python records only total_ms. (Pinned by
//     TestRunner_SSEStreaming_ModelCallEndKeepsRequestSpanAndLatency in
//     trace_manager_test.go).
//   - Default batch size and initial retry delay: [DefaultConfig] sets
//     BatchSize to 100 (paired with automatic 8 MiB / 9 MiB AppendRows request
//     splitting) and RetryConfig.InitialDelay to 100ms, whereas adk-python
//     defaults batch_size to 1 and initial_delay to 1.0s; see [Config] and
//     [RetryConfig] for Cloud Storage object naming/metadata and
//     [Config.ExactlyOnceDelivery] stream-rotation details. (Pinned by
//     TestDocParity_DefaultConfigBatchSizeAndRetryDelay in doc_test.go and
//     TestBatchProcessor_SplitsOversizedBatchAcrossRequests in
//     batch_processor_test.go).
//   - [DropStats] structure: Go exposes [DropStats] as a typed struct of four
//     category counters (QueueFullDrops, AppendFailureDrops,
//     FormatterFailureDrops, ShutdownDrops) plus TotalDrops via
//     [BigQueryAgentAnalyticsPlugin.DropStats], whereas adk-python's
//     get_drop_stats() returns a dictionary of per-reason counts. (Pinned by
//     TestBigQueryAgentAnalyticsPlugin_DropStats in
//     bigquery_agent_analytics_plugin_test.go, and TestBatchProcessor_DropStats
//     and TestBatchProcessor_CloseTimeoutCountsAllLostRowsInDropStats in
//     batch_processor_test.go).
//   - Response summary and function-name bounds: When
//     [Config.CaptureFullMessageContent] is false, Go caps FunctionCall and
//     FunctionResponse names in response summaries at 256 runes with redaction
//     and caps joined multi-part response summaries at [Config.MaxContentLen] +
//     4096 runes before appending "...[TRUNCATED]"; [EventLLMResponse] writes
//     that summary directly to content.response, whereas [EventAgentResponse]
//     (in content.response) and [EventEventCompaction] also undergo column-level
//     [Config.MaxContentLen] string truncation, and
//     [Config.CaptureFullMessageContent] lifts both length caps while preserving
//     redaction. adk-python interpolates function names verbatim and applies no
//     joined-length cap after joining part summaries. (Pinned by
//     TestResponseMessageSummaryLengthCap and
//     TestResponseMessageSummaryFunctionNameCap in json_formatter_test.go, and
//     TestDocParity_ResponseSummaryAndFunctionNameBounds in doc_test.go).
package agentanalytics
