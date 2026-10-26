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
// Retries never extend a write beyond Config.ShutdownTimeout. Each batch
// write made by a background flush, including its retries and the backoff
// between them, ends when ShutdownTimeout expires, while the writes made by a
// run-end flush, and those made while Close drains the queue, share a single
// ShutdownTimeout. If a run-end flush times out while writing to BigQuery, the
// in-flight batch is deferred for at most one retry in the next flush
// (background, run-end, or Close). This provides at-least-once delivery; if
// the backend already committed the cut-off write, duplicates under the same
// event_id can occur (and may differ if uploads cut short on the first attempt
// succeed on retry; queries deduplicating by event_id should prefer rows where
// storage_mode is GCS_REFERENCE or text is not [UPLOAD FAILED]). Unfinished
// batches during background or Close flushes are dropped directly.
type RetryConfig struct {
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
// be confirmed (QueueFullDrops + AppendFailureDrops + ShutdownDrops); it can
// include rows committed before an acknowledgment was lost, so it is an upper
// bound. FormatterFailureDrops separately counts payload replacement or
// omission, which does not itself discard a row.
type DropStats struct {
	// QueueFullDrops counts events dropped because QueueMaxSize was reached.
	QueueFullDrops int64 `json:"queue_full_drops"`
	// AppendFailureDrops counts events dropped because a plugin callback or row
	// construction panicked (each recovered callback panic counts as at most 1
	// failed event write; counts are approximate under rare logger panics), a
	// batch write was skipped before sending because its write deadline had
	// already expired or its desynchronized committed stream could not be
	// replaced (including while stream rotation is in cooldown), or a BigQuery
	// Storage Write append was rejected by the server or exhausted its retries.
	// If the last attempt's outcome was unknown (sent, but no response), the
	// rows may in fact have been written, so this is an upper bound.
	AppendFailureDrops int64 `json:"append_failure_drops"`
	// FormatterFailureDrops counts events whose payload was replaced with
	// "[FORMATTER_FAILED]" (or omitted from content_parts) because
	// ContentFormatter returned an error or panicked; this does not itself
	// discard a row.
	FormatterFailureDrops int64 `json:"formatter_failure_drops"`
	// ShutdownDrops counts events rejected because Append was called after Close,
	// or queued events dropped during Close because ShutdownTimeout expired
	// before their batch could be written.
	ShutdownDrops int64 `json:"shutdown_drops"`
	// TotalDrops sums locally discarded events and writes whose success could
	// not be confirmed (QueueFullDrops + AppendFailureDrops + ShutdownDrops).
	TotalDrops int64 `json:"total_drops"`
}

// GCSUploadRequest describes one payload to upload to Cloud Storage.
type GCSUploadRequest struct {
	// Path is the object name within the bucket; it already includes
	// Config.GCSPathPrefix.
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
// GCSUploadRequest.Path is the object name within the bucket and already
// includes Config.GCSPathPrefix. If UploadContent fails, a text part keeps its
// inline text (truncated to MaxContentLen) in the BigQuery row, while a binary
// part's text is set to "[UPLOAD FAILED]" and its raw bytes are not stored
// inline.
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
// the batch within the same append, with two exceptions: if the stream was
// rotated by an earlier batch and has not committed yet, Go drops the batch
// and pauses rotation; if an earlier attempt of the batch had an unknown
// outcome, OutOfRange or NotFound makes Go drop the batch, to prevent
// duplicate delivery, and replace the stream before the next batch (pausing
// rotation if the stream had not committed yet). adk-python drops the batch
// after any of the three and replaces the stream before the next one. On an
// RPC-level receive error after a successful Send (an AppendRowsResponse.Error
// carrying one of these codes is a server rejection), Go treats Unavailable,
// DeadlineExceeded, Canceled, Aborted, Internal, and Unknown as unknown
// outcomes, whereas adk-python also treats ResourceExhausted (TooManyRequests)
// as one. After an RPC-level InvalidArgument or FailedPrecondition, Go
// replaces the stream only if an earlier attempt of the batch had an unknown
// outcome, whereas adk-python also replaces it when the error is not a schema
// mismatch. DropStats has four counters, whereas adk-python's get_drop_stats()
// returns ten per-reason counts. adk-python uploads while it parses an event,
// before the row is queued, with a Cloud Storage client built from the
// credentials of the plugin, a fixed 32 KiB inline text limit and the
// parallelism of its thread pool. This plugin queues payloads with rows and
// uploads them at batch flush time, bounds upload concurrency and memory,
// supports customizable offloaders, exposes atomic DropStats counters, and
// optionally writes through a COMMITTED stream with explicit offsets when
// ExactlyOnceDelivery is enabled. ExactlyOnceDelivery handles offset
// desynchronization (rotating to a fresh stream without consuming append retry
// attempts) and distinguishes ambiguous transport errors from definite server
// rejections.
//
// Zero numeric and duration fields (MaxContentLen, BatchSize, BatchFlushIntv,
// ShutdownTimeout, QueueMaxSize, and an unset RetryConfig) are filled from
// [DefaultConfig] when a plugin or batch processor is constructed.
//
// Use [DefaultConfig] or set LogMultiModalContent: true to enable
// content_parts (DisableMultiModalContent overrides it). Session metadata
// logging, redaction of sensitive keys and credentials, run-end flushing,
// additive schema upgrades, and per-event view creation are enabled by default;
// DisableSessionMetadata, DisableSensitiveKeyRedaction, DisableFlushOnRunEnd,
// DisableSchemaUpgrade, and DisableCreateViews opt out of each. Created and
// upgraded tables carry label adk_schema_version="2" ([SchemaVersion]), while
// attributes.adk.schema_version on each row is "1". Recursive payload
// sanitization is bounded by maxSanitizeDepth=50 and maxSanitizeNodes=100000.
type Config struct {
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
	// already in flight has finished. In a run-end flush, an in-flight batch
	// that times out is deferred for at most one retry in the next flush
	// (background, run-end, or Close); rows whose write remains unfinished
	// after retrying or when the timeout expires during Close are dropped.
	// (Duplicates under the same event_id can occur and may differ if cut-short
	// uploads succeed on retry; prefer rows with GCS_REFERENCE storage_mode).
	// Close logs each batch it drops.
	//
	// Cloud Storage uploads (see GCSBucketName) do not use up the time that
	// the writes of the drain have: the uploads of all the queued rows share
	// half a ShutdownTimeout, after which the parts that are not uploaded fall
	// back (see GCSUploadTimeout). Close can therefore block for up to about
	// two and a half times ShutdownTimeout. In a run-end flush, the uploads of
	// each batch get at most half of what is left of the ShutdownTimeout of
	// the flush.
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
	// regardless of case and of "_" or "-", so apiKey, APIKey and api-key are
	// redacted like api_key.
	DisableSensitiveKeyRedaction bool

	// ContentFormatter, if set, is applied to each event's raw payload before
	// the content column is built.
	ContentFormatter ContentFormatter
	// DebugContentFormatterErrors includes error-shape metadata (such as error
	// byte length) in the warning logged when ContentFormatter fails, without
	// logging raw error text.
	DebugContentFormatterErrors bool

	// EventAllowlist, if non-empty, limits logging to the listed event types.
	// The legacy umbrella name "EVENT" matches all session-event rows
	// (such as STATE_DELTA, AGENT_TRANSFER, and AGENT_RESPONSE).
	EventAllowlist []string
	// EventDenylist stops the listed event types from being logged. It takes
	// precedence over EventAllowlist, including when "EVENT" is listed in
	// EventAllowlist. The legacy umbrella name "EVENT" suppresses all
	// session-event rows.
	EventDenylist []string

	// PayloadColumnDenylist names payload columns that are omitted from newly
	// created tables, excluded from the Arrow writer schema, and never written.
	// Allowed values are "content", "content_parts", "attributes" and
	// "latency_ms"; any other value is a configuration error.
	PayloadColumnDenylist []string

	// CustomMetadataAllowlist selects the model response and event
	// CustomMetadata keys written to attributes.custom_metadata. An entry
	// ending in "*" matches keys by prefix; other entries match exactly.
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
	// GCSPathPrefix is prepended to the names of offloaded objects.
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
	// that an outage does not make each batch wait in turn. A run-end deadline
	// stops uploads for the rest of that flush if an upload ran for at least
	// three quarters of its batch's upload window before timing out, even if
	// other uploads succeeded or earlier batches completed. This duration is
	// bounded below by ShutdownTimeout/8 and above by 3*ShutdownTimeout/8
	// (1.25s and 3.75s by default). A window shorter than ShutdownTimeout/8
	// cannot trigger this stop. A run-end deadline triggers the shared cooldown
	// only after at least 3*ShutdownTimeout/8 of upload time. Time waiting for a
	// concurrent upload slot or cleaning up after the deadline does not count.
	// Set ShutdownTimeout high enough for normal upload latency: a shorter
	// run-end window may treat a slow but healthy upload as stalled. Caller
	// cancellation, an already-expired deadline, or a deadline during Close
	// does not pause uploads.
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
