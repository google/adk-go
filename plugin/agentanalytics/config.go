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
	// plugin. It matches the version written by adk-python and is stamped as the
	// adk_schema_version label on created BigQuery tables.
	SchemaVersion = "2"
)

const (
	defaultTableName               = "agent_events"
	defaultMaxInlineTextSize       = 32 * 1024
	defaultGCSMaxConcurrentUploads = 8
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
// GCSUploadTimeout, GCSMaxPendingUploadBytes and GCSOffloader have no
// counterpart in BigQueryLoggerConfig. adk-python uploads while it parses an
// event, before the row is queued, with a Cloud Storage client built from the
// credentials of the plugin, a fixed 32 KiB inline text limit and the
// parallelism of its thread pool. This plugin queues the payloads with their
// rows and uploads them when a batch is flushed, before the batch is appended
// to BigQuery. It therefore bounds the uploads of a batch (their parallelism,
// their time, and the memory of the payloads that wait for them), and lets the
// Cloud Storage client and the offloader be configured or replaced.
//
// Zero numeric and duration fields (MaxContentLen, BatchSize, BatchFlushIntv,
// ShutdownTimeout, QueueMaxSize, and an unset RetryConfig) are filled from
// [DefaultConfig] when a plugin or batch processor is constructed.
//
// Use [DefaultConfig] or set LogMultiModalContent: true to enable
// content_parts (DisableMultiModalContent overrides it). Session metadata
// logging, redaction of sensitive keys and credentials, and run-end flushing
// are enabled by default; DisableSessionMetadata, DisableSensitiveKeyRedaction,
// and DisableFlushOnRunEnd opt out of each. Created and upgraded tables carry
// label adk_schema_version="2" ([SchemaVersion]), while
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
	// existing table during initialization; a missing table is still created.
	// The schema of an existing table is then neither inspected nor upgraded,
	// so the table must already have every column that the plugin writes:
	// every column of [EventsSchema], including event_id, except those in
	// PayloadColumnDenylist. Otherwise construction succeeds, but every
	// AppendRows call fails with "Input schema has more fields than BigQuery
	// schema" and every row is dropped. When false (the default), missing
	// columns are added, which needs the bigquery.tables.update permission; a
	// table that already has every column is not updated.
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
	// that an outage does not make each batch wait in turn. The deadline
	// of the first upload batch of a run-end flush (see ShutdownTimeout) pauses
	// the uploads when it is still fresh (at least three quarters of
	// ShutdownTimeout left) and ends the batch's uploads before GCSUploadTimeout
	// without any upload in the batch succeeding (a run-end flush allocates half
	// of its remaining budget to storage uploads, so a ShutdownTimeout smaller
	// than twice the typical upload latency may pause uploads if storage does
	// not finish within that slice), whereas a batch where at least one upload
	// succeeded, an already-depleted budget (less than three quarters
	// remaining), or a budget during Close does not pause.
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
