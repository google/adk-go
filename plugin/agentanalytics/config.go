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
	"log"
	"time"

	"google.golang.org/api/option"
)

const (
	// SchemaVersion is the version of the events table schema written by this
	// plugin and stored under the schemaVersionKey table label. It matches the
	// version written by adk-python so that writers in either language agree on
	// the table schema version.
	SchemaVersion = "2"
	// schemaVersionKey is the BigQuery table label key that records the schema
	// version.
	schemaVersionKey = "adk_schema_version"
)

const (
	defaultTableName               = "agent_events"
	defaultMaxInlineTextSize       = 32 * 1024
	defaultGCSMaxConcurrentUploads = 8
	defaultViewPrefix              = "v"
)

// Logger interface to use standard log operations natively.
type Logger interface {
	Printf(format string, v ...any)
	Println(v ...any)
}

// RetryConfig holds retry parameters for operations like BigQuery Appends.
// An all-zero RetryConfig{} is populated from [DefaultConfig] (MaxRetries: 3,
// InitialDelay: 100ms, MaxDelay: 10s, Multiplier: 2.0); to disable retries,
// set MaxRetries: 0 alongside a non-zero InitialDelay, MaxDelay, or Multiplier.
//
// MaxRetries applies to each AppendRows request. A batch that is too large for
// one request is split into several that are sent in order (see [Config]); a
// request that fails is resent on its own, up to MaxRetries times (plus at most
// one resend after a stream rotation when [Config.ExactlyOnceDelivery] is
// enabled), and the requests that BigQuery already confirmed are not resent. A
// status that a retry cannot fix, such as PermissionDenied or InvalidArgument,
// ends the write of the batch without further retries when opening the stream
// fails with it or BigQuery reports it in an AppendRowsResponse, except that
// when a response without AppendResult carries a nil or InvalidArgument status
// and valid in-range AppendRowsResponse.RowErrors (and not after an ambiguous
// send under ExactlyOnceDelivery), the rejected rows are dropped under
// DropStats.RowErrorDrops and the surviving rows are resent up to 8 passes
// per chunk. When a non-retryable status instead ends the stream, the server
// may not have read the request, so while retries remain the request is resent
// on a fresh stream: always if an earlier request opened that stream, and once
// per request if it was opened for the request, after which the status ends the
// write. Each such resend uses up one retry and its backoff.
//
// Retries never extend a write beyond Config.ShutdownTimeout. Each batch
// write made by a background flush, including its retries and the backoff
// between them, ends when ShutdownTimeout expires, while the writes made by a
// run-end flush, and those made while Close drains the queue, share a single
// ShutdownTimeout. If a run-end flush times out while writing to BigQuery, the
// rows of the in-flight batch that BigQuery has not confirmed are deferred for
// at most one retry in the next flush (background, run-end, or Close); the
// rows of the split requests that it confirmed before then are not resent, and
// neither are the rows of a request that BigQuery may have committed without
// confirming it (see [Config]), which count in DropStats.AppendFailureDrops.
// This provides at-least-once delivery; if
// the backend already committed the cut-off write, duplicates under the same
// event_id can occur (and may differ if uploads cut short on the first attempt
// succeed on retry; queries deduplicating by event_id should prefer rows where
// storage_mode is GCS_REFERENCE or text is not [UPLOAD FAILED]). Unfinished
// batches during background or Close flushes are dropped directly.
type RetryConfig struct {
	// _ prevents unkeyed literals, which adding a field would break.
	_            struct{}
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// ContentFormatter transforms or redacts an event payload before it is
// truncated and written to the content column. The payload is the live value
// the agent is using (for example *model.LLMRequest on EventLLMRequest or
// *model.LLMResponse on EventLLMResponse, where content_parts is populated from
// the formatted *model.LLMResponse when returned), so a formatter must return a
// new value instead of modifying it in place. If it returns an error or panics,
// the content is replaced with "[FORMATTER_FAILED]" so that unformatted data is
// never written.
type ContentFormatter func(payload any, eventType string) (any, error)

// DropStats reports cumulative event-drop and formatter-failure counts across
// the plugin's event processing pipeline.
// TotalDrops sums locally discarded events and writes whose success could not
// be confirmed (QueueFullDrops + AppendFailureDrops + RowErrorDrops +
// ShutdownDrops); it can include rows committed before an acknowledgment was
// lost, so it is an upper bound. FormatterFailureDrops separately counts
// payload replacement or omission, which does not itself discard a row.
type DropStats struct {
	// _ prevents unkeyed literals, which adding a field would break.
	_ struct{}
	// QueueFullDrops counts events dropped because QueueMaxSize was reached.
	QueueFullDrops int64 `json:"queue_full_drops"`
	// AppendFailureDrops counts events dropped because a plugin callback or row
	// construction panicked (a panic in both afterRun and one of its cleanup
	// fallbacks can count the same run more than once), a batch write was
	// skipped before sending because its write deadline had already expired or
	// its desynchronized committed stream could not be replaced (including
	// while stream rotation is in cooldown), or a BigQuery Storage Write append
	// was rejected by the server or exhausted its retries. Because fallback
	// panics in afterRun can increment alongside the callback panic and an
	// unknown last-attempt outcome (sent, but no response) may in fact have
	// been written, this is an upper bound.
	AppendFailureDrops int64 `json:"append_failure_drops"`
	// RowErrorDrops counts individual rows rejected by BigQuery Storage Write
	// AppendRowsResponse.RowErrors and isolated from their batch so the
	// remaining valid rows could still be written.
	RowErrorDrops int64 `json:"row_error_drops"`
	// FormatterFailureDrops counts events whose payload was replaced with
	// "[FORMATTER_FAILED]" (or omitted from content_parts) because
	// ContentFormatter returned an error or panicked; this does not itself
	// discard a row.
	FormatterFailureDrops int64 `json:"formatter_failure_drops"`
	// ShutdownDrops counts events logged by plugin callbacks after Close is
	// called, or queued events dropped during Close because ShutdownTimeout
	// expired before their batch could be written.
	ShutdownDrops int64 `json:"shutdown_drops"`
	// TotalDrops sums locally discarded events and writes whose success could
	// not be confirmed (QueueFullDrops + AppendFailureDrops + RowErrorDrops +
	// ShutdownDrops).
	TotalDrops int64 `json:"total_drops"`
}

// GCSUploadRequest describes one payload to upload to Cloud Storage.
type GCSUploadRequest struct {
	// _ prevents unkeyed literals, which adding a field would break.
	_ struct{}
	// Path is the object name within the bucket. When Config.GCSPathPrefix is
	// non-empty after trimming leading and trailing '/', Path begins with the
	// effective prefix followed by "/" — either the '/'-trimmed
	// Config.GCSPathPrefix verbatim, or its rewritten "<base>~<sha256>" form
	// when that '/'-trimmed prefix contains '~', control/line-separator
	// characters, or invalid UTF-8, starts with the reserved
	// ".well-known/acme-challenge" segments, or exceeds 750 bytes (see
	// Config.GCSPathPrefix). Every condition is checked after the trimming, so
	// a 752-byte "/" + 750 bytes + "/" does not exceed the limit.
	Path string
	// Data is the raw payload to store in the object.
	Data []byte
	// ContentType is the MIME type to set on the stored object.
	ContentType string
}

// GCSContentOffloader uploads multimodal or oversized text payloads to Cloud
// Storage and returns the gs:// URI of the stored object.
//
// The plugin calls UploadContent from several goroutines at the same time.
// GCSUploadRequest.Path is the object name within the bucket and already begins
// with the effective prefix derived from Config.GCSPathPrefix: the '/'-trimmed
// prefix verbatim when valid, or its rewritten "<base>~<sha256>" form when the
// '/'-trimmed prefix contains '~', control/line-separator characters, or
// invalid UTF-8, starts with the reserved ".well-known/acme-challenge"
// segments, or exceeds 750 bytes (see Config.GCSPathPrefix), so raw and
// rewritten prefix namespaces are strictly disjoint. If UploadContent fails, a
// text part keeps its inline text (truncated to MaxContentLen) in the BigQuery
// row, while a binary part's text is set to "[UPLOAD FAILED]" and its raw bytes
// are not stored inline.
// UploadContent must return soon after ctx is done: the plugin waits for the
// uploads of a batch before writing it, and Close waits for that write.
type GCSContentOffloader interface {
	UploadContent(ctx context.Context, req GCSUploadRequest) (gcsURI string, err error)
}

// Config represents settings mapping to adk-python's BigQueryLoggerConfig.
//
// GCSClientOptions, MaxInlineTextSize, GCSMaxConcurrentUploads,
// GCSUploadTimeout, GCSMaxPendingUploadBytes, and GCSOffloader have no
// counterpart in BigQueryLoggerConfig. adk-python declares
// exactly_once_delivery and creates a COMMITTED write stream when enabled, but
// differs in handling. After OutOfRange, NotFound, or an unexpected
// AlreadyExists, Go replaces the stream (at most once per batch) and resends
// the request that failed within the same append (the whole batch, unless it
// was split into several requests; see below), with two exceptions: if the
// stream was rotated by an earlier batch and has not committed yet, Go drops
// the batch and pauses rotation; if an earlier attempt of the request had an
// unknown outcome, OutOfRange or NotFound, or a response that reports
// RowErrors, makes Go drop the batch, to prevent duplicate delivery, and
// replace the stream before the next batch (pausing rotation if the stream
// had not committed yet). adk-python drops the batch after OutOfRange,
// NotFound, or an unexpected AlreadyExists (treating AlreadyExists after an
// attempt with an unknown outcome as confirmation) and replaces the stream
// before the next one. On
// an RPC-level receive error after a successful Send (an
// AppendRowsResponse.Error carrying one of these codes is a server
// rejection), Go treats Unavailable, DeadlineExceeded, Canceled, Aborted,
// Internal, and Unknown as unknown outcomes, whereas adk-python also treats
// ResourceExhausted (TooManyRequests) as one. After an RPC-level
// InvalidArgument or FailedPrecondition, Go replaces the stream only if an
// earlier attempt of the request had an unknown outcome, whereas adk-python
// also replaces it when the error is not a schema mismatch. DropStats has five
// per-reason counters, whereas adk-python's get_drop_stats() returns ten.
// adk-python uploads while it parses an event, before the row is queued, with a
// Cloud Storage client built from the credentials of the plugin, a fixed 32 KiB
// inline text limit and the parallelism of its thread pool. This plugin queues
// payloads with rows and uploads them at batch flush time, bounds upload
// concurrency and memory, supports customizable offloaders, exposes atomic
// DropStats counters, and optionally writes through a COMMITTED stream with
// explicit offsets when ExactlyOnceDelivery is enabled. ExactlyOnceDelivery
// handles offset desynchronization (rotating to a fresh stream without
// consuming append retry attempts) and distinguishes ambiguous transport errors
// from definite server rejections.
//
// Go splits a batch whose rows are estimated at more than 8 MiB into several
// AppendRows requests, and halves a multi-row request that serializes to more
// than 9 MiB, so that a multi-row request stays under the Write API's size
// limit; adk-python sends every batch as one request. The requests of a batch
// are sent in order, each with its own retries (see [RetryConfig]). The first
// request that fails ends the batch: the requests that BigQuery confirmed
// before it are written and are not resent, and only the rows that BigQuery
// did not confirm count as dropped or, when a run-end flush times out, are
// deferred. After an AppendRowsResponse.Error, Go resends the request while
// retries remain if the code is Unavailable, ResourceExhausted, Internal,
// DeadlineExceeded, or Aborted, whereas adk-python resends it only for
// DeadlineExceeded, Internal, or Unavailable, and drops the batch for any
// other code. When AppendRowsResponse.RowErrors reports valid in-range row
// indices without AppendResult alongside a nil or InvalidArgument status (and
// not after an ambiguous send under ExactlyOnceDelivery), Go isolates the
// rejected rows under DropStats.RowErrorDrops and resends surviving rows up to
// 8 passes per chunk, counting any remaining unconfirmed rows under
// AppendFailureDrops, whereas adk-python drops the entire batch when the
// status is InvalidArgument and treats a response with no status set as
// committed without inspecting row_errors. A response that carries both
// AppendResult and RowErrors leaves the rows of its request possibly
// committed, as RowErrors after an ambiguous send do, so Go counts those rows
// under AppendFailureDrops and never sends them again, not even when a run-end
// flush times out.
//
// Zero numeric and duration fields (MaxContentLen, BatchSize, BatchFlushIntv,
// ShutdownTimeout, QueueMaxSize, and an unset RetryConfig) are filled from
// [DefaultConfig] when a plugin is constructed.
//
// Use [DefaultConfig] or set LogMultiModalContent: true to enable
// content_parts (DisableMultiModalContent overrides it). Session metadata
// logging, redaction of sensitive keys and credentials, run-end flushing,
// additive schema upgrades, and per-event view creation are enabled by default;
// DisableSessionMetadata, DisableSensitiveKeyRedaction, DisableFlushOnRunEnd,
// DisableSchemaUpgrade, and DisableCreateViews opt out of each, while
// CreateEvalViews opts in to creating the pre-built evaluation views. Created and
// upgraded tables carry label adk_schema_version="2" ([SchemaVersion]), while
// attributes.adk.schema_version on each row is "1". Recursive payload
// sanitization is bounded by maxSanitizeDepth=50 and maxSanitizeNodes=100000.
//
// While concurrent activations of the same agent on the same branch and node
// path are open, an agent that one of them runs is placed under that
// activation's span (parent_span_id): the activation whose context, or a
// context derived from it, was passed to the agent's Run, whichever goroutine
// runs the agent. If the context does not show the activation that it derives
// from, as when it was derived through a context type that does not embed its
// parent as an exported Context field, goroutines decide instead: a child agent
// that runs on an activation's goroutine, or on a goroutine that the activation
// starts directly, is placed under that activation, and one launched across an
// intermediate goroutine that runs no ADK callback is placed under the nearest
// open span enclosing the concurrent activations that hold an open span its
// branch and node path fit (or the invocation span if none encloses them).
type Config struct {
	// _ prevents unkeyed literals, which adding a field would break.
	_ struct{}

	Enabled        bool
	ProjectID      string
	DatasetID      string
	TableName      string
	MaxContentLen  int
	BatchSize      int
	BatchFlushIntv time.Duration
	QueueMaxSize   int
	ClientOptions  []option.ClientOption

	// Fields to cluster the table by.
	ClusteringFields []string

	// CustomTags are written to attributes.custom_tags on every event.
	CustomTags map[string]any

	// Whether to log multi-modal content in content_parts. Overridden when
	// DisableMultiModalContent is true.
	LogMultiModalContent bool

	// ShutdownTimeout bounds each batch write made by a background flush,
	// including all retries and the backoff between them (see
	// [RetryConfig]), and separately bounds a whole run-end flush (see
	// DisableFlushOnRunEnd) and the BigQuery writes of the drain of queued
	// rows during Close. The drain starts once a background write that is
	// already in flight has finished. In a run-end flush, the rows of an
	// in-flight batch that BigQuery has not confirmed when the timeout expires
	// are deferred for at most one retry in the next flush (background,
	// run-end, or Close); the rows of the batch's requests that BigQuery
	// confirmed first are written, not retried. Rows whose write remains
	// unfinished after retrying or when the timeout expires during Close are
	// dropped.
	// (Duplicates under the same event_id can occur and may differ if cut-short
	// uploads succeed on retry; prefer rows with GCS_REFERENCE storage_mode).
	// Close logs each batch it drops.
	//
	// Cloud Storage uploads (see GCSBucketName) do not use up the time that
	// the writes of the drain have: the uploads of all the queued rows share
	// half a ShutdownTimeout, after which the parts that are not uploaded fall
	// back (see GCSUploadTimeout). Close can therefore block for up to about
	// two and a half times ShutdownTimeout. In a run-end flush, the uploads of
	// each fresh batch get at most half of what is left of the ShutdownTimeout
	// of the flush, and those of a deferred batch that it retries get at most
	// a quarter of what is left (leaving three quarters for the BigQuery
	// write, which cannot be deferred again). If this limit, or Close
	// canceling the uploads, has ended the uploads of a fresh batch that is
	// then deferred, the next flush retries the uploads of the batch that
	// returned an error, unless the batch paused the uploads (see
	// GCSUploadTimeout).
	ShutdownTimeout time.Duration

	// Injected logger.
	Logger Logger

	// Retry configuration for appending rows.
	RetryConfig RetryConfig

	// DisableMultiModalContent stops content_parts from being written,
	// overriding LogMultiModalContent.
	DisableMultiModalContent bool

	// DisableSessionMetadata stops session_id, app_name, user_id and the
	// session state from being written to attributes.session_metadata.
	DisableSessionMetadata bool

	// CaptureFullMessageContent turns off length truncation of text values.
	// Sensitive-key redaction still applies.
	CaptureFullMessageContent bool

	// DisableSensitiveKeyRedaction turns off all automatic redaction, not only
	// the redaction of sensitive keys. When true, values under credential-like
	// keys (for example api_key, password or token) and "temp:" state keys,
	// credentials in URIs, Authorization headers and Bearer or Basic
	// credentials, sensitive keys in JSON embedded in text, and credentials in
	// error_message are all written unredacted. When false, key names match
	// regardless of case and of "_", "-" or ".", so apiKey, APIKey, api-key and
	// api.key are redacted like api_key. A compound name that ends in a
	// credential name, such as DB_PASSWORD, dbPassword or
	// spring.datasource.password, is redacted too, and so is a credential name
	// or "temp:" key behind "user:" or "app:" state prefixes, stacked or not,
	// such as user:api_key, app:user:token or user:temp:x. The "." and
	// compound-name matching apply to map keys, to the keys of query,
	// fragment and ";" parameter pairs of a URI, and to keys in text. They do
	// not apply to URI path segments: only a segment that is a credential
	// name itself, whatever its case and "_" or "-", or a "temp:" key, alone
	// or behind "user:" or "app:" prefixes, redacts the segment after it, as
	// in https://h/api_key/abc123, while https://h/api.key/abc123 and
	// https://h/DB_PASSWORD/abc123 keep abc123, as file names and route slugs
	// such as /private.key/download or /reset-password/confirm stay as they
	// are. Cursor and tokenizer names that end in "token" are not redacted,
	// in snake, camel or upper case: next_token, prev_token and
	// previous_token; page_token, sync_token, continuation_token and
	// resume_token, alone or after next, prev, previous, start or newStart,
	// as nextPageToken, startPageToken or newStartPageToken; and pad_token,
	// eos_token, eot_token, bos_token, unk_token, cls_token, sep_token and
	// mask_token. Plural count names such as input_tokens and output_tokens
	// are also kept because they end in "tokens", whereas singular or
	// flag-style compound names that end in a sensitive credential suffix,
	// such as input_token, output_token, has_password or is_secret, are
	// redacted regardless of value type. Redaction also applies to declared
	// Go types whose type name matches a sensitive key (generic or not): a
	// non-struct value, or a struct whose value or pointer type has a
	// MarshalJSON or MarshalText method, redacts whole as "[REDACTED]" with
	// truncated=true; any other struct shows its exported fields, with the
	// sensitive ones redacted, and omits unexported fields; and map keys of
	// such a type collapse to "[REDACTED]":"[REDACTED]" with
	// truncated=true (for example, map[SyntaxToken]int{1: 12, 2: 9} formats as
	// {"[REDACTED]":"[REDACTED]"} with truncated=true). The adk-python plugin
	// matches exact key names only.
	DisableSensitiveKeyRedaction bool

	// ContentFormatter, if set, is applied to each event's raw payload before
	// the content column is built. The evaluation views (see CreateEvalViews)
	// need it to keep the content keys that they read.
	ContentFormatter ContentFormatter
	// DebugContentFormatterErrors includes error-shape metadata (such as error
	// byte length) in the warning logged when ContentFormatter fails, without
	// logging raw error text.
	DebugContentFormatterErrors bool

	// EventAllowlist, if non-empty, limits logging to the listed event types.
	// The legacy umbrella name "EVENT" matches session-event emitters
	// (STATE_DELTA, AGENT_TRANSFER, TOOL_PAUSED, HITL_*, OnEventCallback
	// AGENT_RESPONSE, and resumed TOOL_COMPLETED); it does not match primary
	// TOOL_COMPLETED rows or FinalResponseToolNames AGENT_RESPONSE rows.
	EventAllowlist []string
	// EventDenylist stops the listed event types from being logged. It takes
	// precedence over EventAllowlist, including when "EVENT" is listed in
	// EventAllowlist. The legacy umbrella name "EVENT" suppresses
	// session-event emitters (including resumed TOOL_COMPLETED rows, though not
	// primary TOOL_COMPLETED or FinalResponseToolNames AGENT_RESPONSE rows).
	EventDenylist []string

	// PayloadColumnDenylist names payload columns that are omitted from newly
	// created tables, excluded from the Arrow writer schema, and never written.
	// Allowed values are "content", "content_parts", "attributes" and
	// "latency_ms"; any other value is a configuration error.
	PayloadColumnDenylist []string

	// CustomMetadataAllowlist selects the model response and event
	// CustomMetadata keys written to attributes.custom_metadata. An entry
	// ending in "*" matches keys by prefix; other entries match exactly.
	// EventAgentStateCheckpoint rows read their reserved checkpoint keys
	// directly from CustomMetadata, while attributes.custom_metadata still
	// includes only allowlisted keys.
	CustomMetadataAllowlist []string

	// FinalResponseToolNames lists tools whose successful completion is also
	// logged as an AGENT_RESPONSE event carrying the tool arguments.
	FinalResponseToolNames []string

	// DisableFlushOnRunEnd stops the synchronous best-effort flush when an
	// invocation completes. When false (the default), AfterRunCallback flushes
	// queued events, skipping the flush if a background flush is already
	// running. A single ShutdownTimeout bounds the whole flush; rows that are
	// still queued (or an in-flight batch deferred after a write timeout) when
	// it expires are left for a later flush.
	DisableFlushOnRunEnd bool

	// DisableSchemaUpgrade stops the plugin from adding missing columns to an
	// existing table during initialization; when bqClient is non-nil, table
	// metadata is still read to check whether the table exists, and a missing
	// table is still created. The schema of an existing table is then neither
	// diffed nor upgraded, so the table must already have every column that the
	// plugin writes: every column of [EventsSchema], including event_id, except
	// those in PayloadColumnDenylist. Otherwise construction succeeds, but every
	// AppendRows call fails with "Input schema has more fields than BigQuery
	// schema" and every row is dropped. For pre-provisioned append-only
	// deployments without bigquery.tables.get permission, pass bqClient == nil
	// to [NewBigQueryAgentAnalyticsPluginWithClients]. When false (the default),
	// missing columns are added, which needs the bigquery.tables.update
	// permission; a table that already has every column is not updated.
	DisableSchemaUpgrade bool

	// EnableOTelCorrelation also writes the ambient OpenTelemetry trace and
	// span IDs to attributes.otel.
	EnableOTelCorrelation bool

	// GCSBucketName enables offloading of large text and inline binary parts
	// to the named Cloud Storage bucket. The plugin creates the Cloud Storage
	// client from GCSClientOptions (or ClientOptions when GCSClientOptions is
	// empty; shared options must be service-neutral) and closes it when the
	// plugin is closed.
	//
	// Unlike adk-python, which uses the host's local date in object names,
	// omits "size" from object_ref.details.gcs_metadata on all offloaded
	// parts (both text and binary), and records "text/plain" as the
	// content_type of offloaded text, this plugin uses the UTC date of the
	// row's timestamp in object names, records "size" on both text and binary
	// parts, and records "text/plain; charset=utf-8" for offloaded text
	// (while keeping content_parts[].mime_type as "text/plain"); queries
	// filtering on $.gcs_metadata.content_type in object_ref.details see
	// "text/plain; charset=utf-8" for Go rows and "text/plain" for Python
	// rows.
	GCSBucketName string
	// GCSClientOptions configures the Cloud Storage client created for
	// GCSBucketName. When empty, ClientOptions is used.
	GCSClientOptions []option.ClientOption
	// GCSPathPrefix is prepended, followed by "/", to the names of offloaded
	// objects. Leading and trailing slashes are trimmed, so "/logs/" and
	// "logs" are the same prefix and a prefix of only slashes adds nothing.
	// The trimmed prefix is used verbatim when it is valid UTF-8 of at most
	// 750 bytes; does not contain '~' (reserved as the rewrite hash
	// delimiter); has no control character (U+0000 to U+001F and U+007F to
	// U+009F, which include CR, LF and U+0085) and no Unicode line or
	// paragraph separator (U+2028 or U+2029); and its first two "/"-separated
	// segments are not, case-insensitively, ".well-known" and "acme-challenge",
	// which Cloud Storage reserves (whole segments are compared, so
	// ".well-known/acme-challenge-x" is kept verbatim). Valid characters and
	// segments such as '\', spaces, '=', Unicode text, empty inner segments
	// ("//"), and "." or ".." segments are kept unchanged.
	//
	// Any other prefix is rewritten to "<base>~<sha256>", and a warning with
	// the rewrite reason and byte lengths (without logging prefix text) is
	// logged to Logger when the plugin is constructed: each
	// '~', each byte that is not valid UTF-8, and each of those control or
	// line/paragraph-separator characters is replaced with '_', a leading
	// ".well-known" segment before "acme-challenge" becomes "_well-known", the
	// result is cut to at most 402 bytes on a UTF-8 rune boundary, and "~"
	// followed by the 64 lowercase hex digits of the SHA-256 digest of the
	// trimmed prefix is appended (at most 467 rewritten prefix bytes total),
	// so distinct trimmed prefixes keep distinct destinations, the folder part
	// of "<prefix>/<YYYY-MM-DD>/<trace_id>/" fits the 512-byte folder limit on
	// hierarchical-namespace Cloud Storage buckets, assembled object names stay
	// within Cloud Storage's 1,024-byte limit, and raw and rewritten prefix
	// namespaces are strictly disjoint. Because '~' is reserved, passing a
	// rewritten "<base>~<sha256>" output back into GCSPathPrefix rewrites it
	// again rather than aliasing the original rewritten prefix.
	GCSPathPrefix string
	// ConnectionID is the BigQuery connection recorded as the authorizer of
	// offloaded objects, in the "location.connection_id" form, for example
	// "us.my-connection".
	ConnectionID string
	// MaxInlineTextSize is the size in bytes above which a text part is
	// offloaded. A text part that MaxContentLen would cut is offloaded as
	// well. It has no effect unless GCSBucketName or GCSOffloader is set.
	// Values <= 0 mean 32 KiB.
	MaxInlineTextSize int
	// GCSMaxConcurrentUploads bounds parallel uploads per batch. Uploads for a
	// batch run during flush before the batch is appended to BigQuery, so
	// sustained offload throughput is bounded by roughly
	// GCSMaxConcurrentUploads / upload_latency. Values <= 0 mean 8.
	GCSMaxConcurrentUploads int
	// GCSUploadTimeout bounds how long a batch waits for its Cloud Storage
	// uploads. The parts whose uploads are not finished by then fall back: a
	// text part keeps its inline text (truncated to MaxContentLen), and a
	// binary part's text is set to "[UPLOAD FAILED]", its raw bytes not being
	// stored inline. When the uploads of a batch time out without any upload
	// succeeding, or when an in-flight upload times out during the batch timeout
	// (partial outage backoff), the plugin stops calling the offloader for
	// another GCSUploadTimeout: every batch in that time falls back at once, so
	// that an outage does not make each batch wait in turn. In a run-end flush
	// (see ShutdownTimeout), which gives the uploads of each fresh batch half of
	// its remaining budget (and those of a deferred batch that it retries at
	// most a quarter), that deadline pauses the uploads too on a fresh batch
	// when it ends them before GCSUploadTimeout, provided that the batch
	// started with at least a quarter of ShutdownTimeout left, and either no
	// upload in the batch succeeded or the deadline cut short an upload that
	// had been running for at least three quarters of the batch's upload time,
	// even if other uploads in the batch succeeded. An upload that a custom
	// offloader fails with a deadline error of its own, such as a per-request
	// timeout, before that deadline passes is not cut short by it, however
	// long it ran. A fresh batch's upload time
	// can thus be as short as an eighth of ShutdownTimeout, so a single upload
	// that is still running after three thirty-seconds of ShutdownTimeout
	// (about 0.94s by default) can pause the uploads, even if storage is
	// healthy and only that upload is slow: a ShutdownTimeout smaller than
	// eight times the latency of an upload (about eleven times if it waited a
	// quarter of the batch's upload time for a free slot) may pause them.
	// An upload cut short after running for less than three
	// quarters of the batch's upload time (for example after waiting for a
	// free slot behind other uploads) does not pause the uploads when another
	// upload in the batch succeeded, and neither does a budget during Close.
	// Caller cancellation and an already-expired deadline do not pause the
	// uploads either. The run-end deadline of a retried deferred batch, or of
	// a batch that started with less than a quarter of ShutdownTimeout left,
	// does not pause them either, though the batch's own GCSUploadTimeout still
	// does when that ends its uploads first. A pause that a run-end flush starts
	// lasts GCSUploadTimeout as well, so it can outlast the flush: until it ends,
	// background flushes, later run-end flushes and Close also fall back
	// without calling the offloader.
	// Values <= 0 mean 10s.
	GCSUploadTimeout time.Duration
	// GCSMaxPendingUploadBytes bounds the total bytes held in memory by
	// pending Cloud Storage uploads across queued rows. When reserving an
	// upload would exceed this limit, the part is not uploaded and falls back
	// as if its upload had failed. The limit counts each payload once, but an
	// upload in flight holds more with the built-in offloader: a []byte copy
	// of a text payload and the Cloud Storage writer's chunk buffer, which is
	// as large as the payload, up to 16 MiB. In the worst case, the memory of
	// text uploads is therefore about three times this limit, and that of
	// binary uploads about twice. Values <= 0 mean 256 MiB.
	GCSMaxPendingUploadBytes int64
	// GCSOffloader overrides the offloader built from GCSBucketName. The
	// plugin does not close an offloader that the caller supplies.
	GCSOffloader GCSContentOffloader

	// DisableCreateViews stops the plugin from creating or replacing the 25
	// pre-built per-event BigQuery views (see [Config.ViewDefinitions]), named
	// "<ViewPrefix>_<lower-case event type>" (for example v_tool_starting), in
	// the dataset of the events table. When false (the default) and the
	// plugin has a BigQuery client, the views are created in the background
	// once the table is ready, with one CREATE OR REPLACE VIEW job per view;
	// this needs bigquery.jobs.create and bigquery.tables.create/update, and
	// replaces existing views of the same names, while a table of the same
	// name that is not a view is left alone and its view reported as failed.
	// Closing the plugin stops starting further view jobs and cancels the
	// client wait on an in-flight request, though a job already submitted to
	// BigQuery may still finish on the server. Background failures are
	// reported only when a non-default Logger is set;
	// [BigQueryAgentAnalyticsPlugin.SyncViews] creates the views synchronously
	// and returns the error, even when DisableCreateViews is true. Deployments
	// that start many instances may set DisableCreateViews and call SyncViews
	// once at deploy time.
	DisableCreateViews bool
	// CreateEvalViews creates or replaces the two pre-built agent evaluation
	// BigQuery views (see [Config.EvalViewDefinitions]), named
	// "<ViewPrefix>_eval_session" and "<ViewPrefix>_eval_turn" (for example
	// v_eval_session and v_eval_turn), in the dataset of the events table.
	// When true (the default is false) and the plugin has a BigQuery client,
	// the views are created in the background once the table is ready, after
	// the per-event views when DisableCreateViews is false or on their own
	// when DisableCreateViews is true; [BigQueryAgentAnalyticsPlugin.SyncEvalViews]
	// creates or replaces the two evaluation views synchronously and returns
	// the error even when CreateEvalViews is false, and
	// [BigQueryAgentAnalyticsPlugin.SyncViews] also includes them when
	// CreateEvalViews is true.
	//
	// Both views reconstruct Vertex AI GenAI Evaluation AgentData trajectories
	// (trace) and per-turn / per-session guardrail metrics from the content
	// column and require USER_MESSAGE_RECEIVED, AGENT_STARTING, LLM_REQUEST,
	// TOOL_STARTING, TOOL_COMPLETED, TOOL_PAUSED, AGENT_TRANSFER,
	// HITL_CREDENTIAL_REQUEST, HITL_CONFIRMATION_REQUEST, HITL_INPUT_REQUEST,
	// HITL_CREDENTIAL_REQUEST_COMPLETED, HITL_CONFIRMATION_REQUEST_COMPLETED,
	// HITL_INPUT_REQUEST_COMPLETED, AGENT_RESPONSE, and INVOCATION_COMPLETED
	// events. Enabling CreateEvalViews when "content" is in
	// PayloadColumnDenylist or when EventDenylist / EventAllowlist filters out
	// any active emitter of these required events (including "EVENT" in
	// EventAllowlist without explicit AGENT_RESPONSE when FinalResponseToolNames
	// is non-empty, or "EVENT" in EventDenylist) is a configuration error.
	//
	// A custom ContentFormatter must keep the content keys that the views read
	// ($.text_summary, $.response, $.tool, $.args, and $.result, as well as
	// $.from_agent and $.to_agent of AGENT_TRANSFER rows and $.usage of
	// LLM_RESPONSE rows). Setting ContentFormatter also stops AGENT_RESPONSE
	// rows from writing content_parts, so the views then take the reply text
	// from $.response alone.
	//
	// A user message with a text part offloaded to Cloud Storage (see
	// GCSBucketName and MaxInlineTextSize) has only a 200-rune preview of that
	// part in the table, so the views reconstruct no user text for the
	// message: its turn is not eval_ready, and neither is its session.
	//
	// Because the evaluation views aggregate per session or turn, filtering on
	// session_start_timestamp or turn_start_timestamp does not prune the
	// events table's timestamp partitions.
	CreateEvalViews bool
	// ViewPrefix is the name prefix of the created views. Empty means "v".
	// It may contain only letters, digits, '_' and '-'. Give each events
	// table that shares a dataset its own prefix.
	ViewPrefix string
	// ExactlyOnceDelivery writes through a COMMITTED write stream with
	// explicit offsets instead of the default stream (for which event_id
	// remains the row-level deduplication key), so that retrying an append
	// whose outcome is unknown does not write the rows twice. Only ambiguous
	// transport errors (e.g. Unavailable, DeadlineExceeded) mark a send as
	// ambiguous; definite server rejections (e.g. InvalidArgument) leave
	// the offset in sync. When an offset desynchronization occurs (e.g.
	// OutOfRange, NotFound), the plugin rotates to a fresh COMMITTED stream
	// at offset 0 without burning an append retry attempt. When a rotation
	// fails, or a freshly rotated stream is desynchronized again before it
	// commits a batch, rotation pauses for 30 seconds: batches flushed
	// during the pause are dropped without calling CreateWriteStream and
	// counted in DropStats.AppendFailureDrops, or in DropStats.ShutdownDrops
	// during Close once its write budget is used up. A
	// failed rotation does not pause rotation if the batch write reached it
	// with less than half of ShutdownTimeout (and less than 30 seconds) left
	// and the write's deadline ended the CreateWriteStream call; the next
	// batch retries the rotation instead.
	// Replacing a desynchronized stream consumes CreateWriteStream quota and
	// finalizes the retired stream in the background. Close waits for
	// retired-stream finalizations that are still running and finalizes the
	// active stream only within the remaining ShutdownTimeout write budget: it
	// cancels the finalizations still running when the budget runs out, and
	// skips the active stream's if draining queued rows already exhausted
	// ShutdownTimeout.
	ExactlyOnceDelivery bool
}

// DefaultConfig returns the default configuration for the agent analytics plugin.
func DefaultConfig() Config {
	return Config{
		Enabled:                  true,
		MaxContentLen:            500 * 1024,
		DatasetID:                "agent_analytics",
		TableName:                defaultTableName,
		ClusteringFields:         []string{"event_type", "agent", "user_id"},
		LogMultiModalContent:     true,
		BatchSize:                100,
		BatchFlushIntv:           time.Second,
		ShutdownTimeout:          10 * time.Second,
		QueueMaxSize:             10000,
		CustomTags:               make(map[string]any),
		Logger:                   log.Default(), //nolint:forbidigo // pre-slog call site
		MaxInlineTextSize:        defaultMaxInlineTextSize,
		GCSMaxConcurrentUploads:  defaultGCSMaxConcurrentUploads,
		GCSUploadTimeout:         gcsUploadTimeout,
		GCSMaxPendingUploadBytes: defaultMaxPendingUploadBytes,
		ViewPrefix:               defaultViewPrefix,
		RetryConfig: RetryConfig{
			MaxRetries:   3,
			InitialDelay: 100 * time.Millisecond,
			MaxDelay:     10 * time.Second,
			Multiplier:   2.0,
		},
	}
}

// applyDefaults fills in zero values that have a non-zero default.
func applyDefaults(c Config) Config {
	defaults := DefaultConfig()
	if c.DatasetID == "" {
		c.DatasetID = defaults.DatasetID
	}
	if c.TableName == "" {
		c.TableName = defaults.TableName
	}
	if c.MaxContentLen == 0 {
		c.MaxContentLen = defaults.MaxContentLen
	}
	if c.BatchSize == 0 {
		c.BatchSize = defaults.BatchSize
	}
	if c.BatchFlushIntv == 0 {
		c.BatchFlushIntv = defaults.BatchFlushIntv
	}
	if c.ShutdownTimeout == 0 {
		c.ShutdownTimeout = defaults.ShutdownTimeout
	}
	if c.QueueMaxSize == 0 {
		c.QueueMaxSize = defaults.QueueMaxSize
	}
	if c.RetryConfig == (RetryConfig{}) {
		c.RetryConfig = defaults.RetryConfig
	} else {
		if c.RetryConfig.InitialDelay == 0 {
			c.RetryConfig.InitialDelay = defaults.RetryConfig.InitialDelay
		}
		if c.RetryConfig.MaxDelay == 0 {
			c.RetryConfig.MaxDelay = defaults.RetryConfig.MaxDelay
		}
		if c.RetryConfig.Multiplier == 0 {
			c.RetryConfig.Multiplier = defaults.RetryConfig.Multiplier
		}
	}
	if c.MaxInlineTextSize <= 0 {
		c.MaxInlineTextSize = defaultMaxInlineTextSize
	}
	if c.GCSMaxConcurrentUploads <= 0 {
		c.GCSMaxConcurrentUploads = defaultGCSMaxConcurrentUploads
	}
	if c.GCSUploadTimeout <= 0 {
		c.GCSUploadTimeout = gcsUploadTimeout
	}
	if c.GCSMaxPendingUploadBytes <= 0 {
		c.GCSMaxPendingUploadBytes = defaultMaxPendingUploadBytes
	}
	if c.ViewPrefix == "" {
		c.ViewPrefix = defaultViewPrefix
	}
	return c
}

func (c Config) logMultiModalContent() bool {
	return c.LogMultiModalContent && !c.DisableMultiModalContent
}

func (c Config) logSessionMetadata() bool {
	return !c.DisableSessionMetadata
}

func (c Config) redactSensitiveKeys() bool {
	return !c.DisableSensitiveKeyRedaction
}

func (c Config) flushOnRunEnd() bool {
	return !c.DisableFlushOnRunEnd
}

func (c Config) createViews() bool {
	return !c.DisableCreateViews
}

// maxContentLength returns the text truncation limit; -1 means no limit.
func (c Config) maxContentLength() int {
	if c.CaptureFullMessageContent {
		return -1
	}
	return c.MaxContentLen
}

// logf logs through c.Logger when one is configured.
func (c Config) logf(format string, args ...any) {
	if c.Logger != nil {
		c.Logger.Printf(format, args...)
	}
}

// viewLogf returns c.Logger.Printf when the caller set a non-default Logger, or
// nil when Logger is nil or DefaultConfig's log.Default(), so background view
// creation never writes diagnostics to the process-wide standard logger.
func (c Config) viewLogf() func(format string, args ...any) {
	if c.Logger == nil || c.Logger == log.Default() { //nolint:forbidigo // comparing against DefaultConfig's pre-slog Logger value
		return nil
	}
	return c.Logger.Printf
}
