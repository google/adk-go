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
	// plugin and stored under the SchemaVersionKey table label. It matches the
	// version written by adk-python so that writers in either language agree on
	// the table schema version.
	SchemaVersion = "2"
	// SchemaVersionKey is the BigQuery table label key that records the schema
	// version.
	SchemaVersionKey = "adk_schema_version"
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
// ShutdownTimeout. If the attempts and backoff allowed by a RetryConfig take
// longer than that, retrying stops when the timeout expires and the batch is
// dropped.
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
	// AppendFailureDrops counts events dropped because row construction
	// panicked or a BigQuery Storage Write append was rejected by the server or
	// exhausted its retries. If the last attempt's outcome was unknown (sent,
	// but no response), the rows may in fact have been written, so this is an
	// upper bound.
	AppendFailureDrops int64 `json:"append_failure_drops"`
	// FormatterFailureDrops counts events whose payload was replaced with
	// "[FORMATTER_FAILED]" (or omitted from content_parts) because
	// ContentFormatter returned an error or panicked; this does not itself
	// discard a row.
	FormatterFailureDrops int64 `json:"formatter_failure_drops"`
	// ShutdownDrops counts events rejected because Append was called after Close.
	ShutdownDrops int64 `json:"shutdown_drops"`
	// TotalDrops sums locally discarded events and writes whose success could
	// not be confirmed (QueueFullDrops + AppendFailureDrops + ShutdownDrops);
	// it can include rows committed before an acknowledgment was lost, so this
	// is an upper bound.
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
// includes Config.GCSPathPrefix. If UploadContent fails, the payload is kept
// inline in the BigQuery row. UploadContent must return soon after ctx is
// done: the plugin waits for the uploads of a batch before writing it, and
// Close waits for that write.
type GCSContentOffloader interface {
	UploadContent(ctx context.Context, req GCSUploadRequest) (gcsURI string, err error)
}

// Config represents settings mapping to adk-python's BigQueryLoggerConfig.
//
// Zero numeric and duration fields (MaxContentLen, BatchSize, BatchFlushIntv,
// ShutdownTimeout, QueueMaxSize, and an unset RetryConfig) are filled from
// [DefaultConfig] when a plugin or batch processor is constructed.
//
// Use [DefaultConfig] or set LogMultiModalContent: true to enable
// content_parts (DisableMultiModalContent overrides it). Session metadata
// logging, redaction of sensitive keys and credentials, run-end flushing, and
// additive schema upgrades are enabled by default; DisableSessionMetadata,
// DisableSensitiveKeyRedaction, DisableFlushOnRunEnd, and
// DisableAutoSchemaUpgrade (or DisableSchemaUpgrade) opt out of each. Created
// and upgraded tables carry label adk_schema_version="2" ([SchemaVersion]),
// while attributes.adk.schema_version on each row is "1". Recursive payload
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
	// DisableFlushOnRunEnd) and the whole drain of queued rows during Close.
	// Rows whose write is still unfinished when the timeout expires are
	// dropped.
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
	// still queued when it expires are left for a later flush.
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
	// uploads before the remaining parts of that batch fall back to inline
	// storage. Values <= 0 mean 10s.
	GCSUploadTimeout time.Duration
	// GCSMaxPendingUploadBytes bounds the total bytes held in memory by
	// pending Cloud Storage uploads across queued rows. When reserving an
	// upload would exceed this limit, the part stays inline instead.
	// Values <= 0 mean 256 MiB.
	GCSMaxPendingUploadBytes int64
	// GCSOffloader overrides the offloader built from GCSBucketName. The
	// plugin does not close an offloader that the caller supplies.
	GCSOffloader GCSContentOffloader

	// DisableAutoSchemaUpgrade turns off the additive-only schema check at
	// startup. When false (the default), the plugin appends any missing columns
	// as NULLABLE and updates the adk_schema_version table label. Upgrading
	// requires bigquery.tables.update permission when columns are added; a 403
	// on initial metadata inspection logs a warning and proceeds assuming the
	// table exists, whereas an incompatible existing schema or failed update
	// returns an error. If all columns already match, no metadata write is
	// issued.
	DisableAutoSchemaUpgrade bool

	// ExactlyOnceDelivery writes through a COMMITTED write stream with
	// explicit offsets instead of the default stream (for which event_id
	// remains the row-level deduplication key), so that retrying an append
	// whose outcome is unknown does not write the rows twice. It only
	// deduplicates retries made by this process and is not lossless: a batch
	// is still dropped, and counted in DropStats, when its retries or stream
	// rotation fail. Replacing a desynchronized stream consumes
	// CreateWriteStream quota. The stream is finalized on Close.
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

func (c Config) autoSchemaUpgrade() bool {
	return !c.DisableAutoSchemaUpgrade && !c.DisableSchemaUpgrade
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
