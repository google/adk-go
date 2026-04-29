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
	"bytes"

	bq "cloud.google.com/go/bigquery"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/ipc"
)

// EventsSchema returns the BigQuery schema for the events table.
func EventsSchema() bq.Schema {
	return bq.Schema{
		{Name: "timestamp", Type: bq.TimestampFieldType, Required: true, Description: "The UTC timestamp when the event occurred. Used for ordering events within a session."},
		{Name: "event_type", Type: bq.StringFieldType, Required: false, Description: "The category of the event (e.g., 'LLM_REQUEST', 'TOOL_CALL', 'AGENT_RESPONSE'). Helps in filtering specific types of interactions."},
		{Name: "agent", Type: bq.StringFieldType, Required: false, Description: "The name of the agent that generated this event. Useful for multi-agent systems."},
		{Name: "session_id", Type: bq.StringFieldType, Required: false, Description: "A unique identifier for the entire conversation session. Used to group all events belonging to a single user interaction."},
		{Name: "invocation_id", Type: bq.StringFieldType, Required: false, Description: "A unique identifier for a single turn or execution within a session. Groups related events like LLM request and response."},
		{Name: "user_id", Type: bq.StringFieldType, Required: false, Description: "The identifier of the end-user participating in the session, if available."},
		{Name: "trace_id", Type: bq.StringFieldType, Required: false, Description: "OpenTelemetry trace ID for distributed tracing across services."},
		{Name: "span_id", Type: bq.StringFieldType, Required: false, Description: "OpenTelemetry span ID for this specific operation."},
		{Name: "parent_span_id", Type: bq.StringFieldType, Required: false, Description: "OpenTelemetry parent span ID to reconstruct the operation hierarchy."},
		{Name: "content", Type: bq.JSONFieldType, Required: false, Description: "The primary payload of the event, stored as a JSON string. The structure depends on the event_type (e.g., prompt text for LLM_REQUEST, tool output for TOOL_RESPONSE)."},
		{
			Name:        "content_parts",
			Type:        bq.RecordFieldType,
			Repeated:    true,
			Description: "For multi-modal events, contains a list of content parts (text, images, etc.).",
			Schema: bq.Schema{
				{Name: "mime_type", Type: bq.StringFieldType, Required: false, Description: "The MIME type of the content part (e.g., 'text/plain', 'image/png')."},
				{Name: "uri", Type: bq.StringFieldType, Required: false, Description: "The URI of the content part if stored externally (e.g., GCS bucket path)."},
				{
					Name:        "object_ref",
					Type:        bq.RecordFieldType,
					Required:    false,
					Description: "The ObjectRef of the content part if stored externally.",
					Schema: bq.Schema{
						{Name: "uri", Type: bq.StringFieldType, Required: false, Description: "The URI of the object."},
						{Name: "version", Type: bq.StringFieldType, Required: false, Description: "The version of the object."},
						{Name: "authorizer", Type: bq.StringFieldType, Required: false, Description: "The authorizer for the object."},
						{Name: "details", Type: bq.JSONFieldType, Required: false, Description: "Additional details about the object."},
					},
				},
				{Name: "text", Type: bq.StringFieldType, Required: false, Description: "The raw text content if the part is text-based."},
				{Name: "part_index", Type: bq.IntegerFieldType, Required: false, Description: "The zero-based index of this part within the content."},
				{Name: "part_attributes", Type: bq.StringFieldType, Required: false, Description: "Additional metadata for this content part as a JSON object (serialized to string)."},
				{Name: "storage_mode", Type: bq.StringFieldType, Required: false, Description: "Indicates how the content part is stored (e.g., 'INLINE', 'GCS_REFERENCE', 'EXTERNAL_URI')."},
			},
		},
		{Name: "attributes", Type: bq.JSONFieldType, Required: false, Description: "A JSON object containing arbitrary key-value pairs for additional event metadata. Includes enrichment fields like 'root_agent_name' (turn orchestration), 'model' (request model), 'model_version' (response version), and 'usage_metadata' (detailed token counts)."},
		{Name: "latency_ms", Type: bq.JSONFieldType, Required: false, Description: "A JSON object containing latency measurements, such as 'total_ms' and 'time_to_first_token_ms'."},
		{Name: "status", Type: bq.StringFieldType, Required: false, Description: "The outcome of the event, typically 'OK' or 'ERROR'."},
		{Name: "error_message", Type: bq.StringFieldType, Required: false, Description: "Detailed error message if the status is 'ERROR'."},
		{Name: "is_truncated", Type: bq.BooleanFieldType, Required: false, Description: "Boolean flag indicating if the 'content' field was truncated because it exceeded the maximum allowed size."},
	}
}

// ArrowSchema returns the Arrow schema for the events table.
func ArrowSchema() *arrow.Schema {
	return arrow.NewSchema(
		[]arrow.Field{
			{Name: "timestamp", Type: arrow.FixedWidthTypes.Timestamp_us, Nullable: false},
			{Name: "event_type", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "agent", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "session_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "invocation_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "user_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "trace_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "span_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "parent_span_id", Type: arrow.BinaryTypes.String, Nullable: true},
			{
				Name:     "content",
				Type:     arrow.BinaryTypes.String,
				Nullable: true,
				Metadata: arrow.NewMetadata([]string{"ARROW:extension:name"}, []string{"google:sqlType:json"}),
			},
			{
				Name: "content_parts",
				Type: arrow.ListOfField(arrow.Field{
					Name: "element",
					Type: arrow.StructOf(
						arrow.Field{Name: "mime_type", Type: arrow.BinaryTypes.String, Nullable: true},
						arrow.Field{Name: "uri", Type: arrow.BinaryTypes.String, Nullable: true},
						arrow.Field{
							Name: "object_ref",
							Type: arrow.StructOf(
								arrow.Field{Name: "uri", Type: arrow.BinaryTypes.String, Nullable: true},
								arrow.Field{Name: "version", Type: arrow.BinaryTypes.String, Nullable: true},
								arrow.Field{Name: "authorizer", Type: arrow.BinaryTypes.String, Nullable: true},
								arrow.Field{
									Name:     "details",
									Type:     arrow.BinaryTypes.String,
									Nullable: true,
									Metadata: arrow.NewMetadata([]string{"ARROW:extension:name"}, []string{"google:sqlType:json"}),
								},
							),
							Nullable: true,
						},
						arrow.Field{Name: "text", Type: arrow.BinaryTypes.String, Nullable: true},
						arrow.Field{Name: "part_index", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
						arrow.Field{Name: "part_attributes", Type: arrow.BinaryTypes.String, Nullable: true},
						arrow.Field{Name: "storage_mode", Type: arrow.BinaryTypes.String, Nullable: true},
					),
					Nullable: true,
				}),
				Nullable: true,
			},
			{
				Name:     "attributes",
				Type:     arrow.BinaryTypes.String,
				Nullable: true,
				Metadata: arrow.NewMetadata([]string{"ARROW:extension:name"}, []string{"google:sqlType:json"}),
			},
			{
				Name:     "latency_ms",
				Type:     arrow.BinaryTypes.String,
				Nullable: true,
				Metadata: arrow.NewMetadata([]string{"ARROW:extension:name"}, []string{"google:sqlType:json"}),
			},
			{Name: "status", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "error_message", Type: arrow.BinaryTypes.String, Nullable: true},
			{Name: "is_truncated", Type: arrow.FixedWidthTypes.Boolean, Nullable: true},
		},
		nil,
	)
}

// SerializedArrowSchema returns the serialized Arrow schema for the events table.
func SerializedArrowSchema() ([]byte, error) {
	schema := ArrowSchema()
	var buf bytes.Buffer
	wr := ipc.NewWriter(&buf, ipc.WithSchema(schema))
	// We only need the schema itself, so we close cleanly
	err := wr.Close()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
