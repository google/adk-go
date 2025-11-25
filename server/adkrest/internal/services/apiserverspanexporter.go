// Copyright 2025 Google LLC
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

package services

import (
	"context"
	"strings"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// APIServerSpanExporter is a custom SpanExporter that stores relevant span data.
// Stores attributes of specific spans (call_llm, send_data, execute_tool) keyed by `gcp.vertex.agent.event_id`.
// This is used for debugging individual events.
// APIServerSpanExporter implements sdktrace.SpanExporter interface.
type APIServerSpanExporter struct {
	traceDict map[string]map[string]string
}

// NewAPIServerSpanExporter returns a APIServerSpanExporter instance
func NewAPIServerSpanExporter() *APIServerSpanExporter {
	return &APIServerSpanExporter{
		traceDict: make(map[string]map[string]string),
	}
}

// GetTraceDict returns stored trace informations
func (s *APIServerSpanExporter) GetTraceDict() map[string]map[string]string {
	return s.traceDict
}

// ExportSpans implements custom export function for sdktrace.SpanExporter.
func (s *APIServerSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	for _, span := range spans {
		if span.Name() == "call_llm" || span.Name() == "send_data" || strings.HasPrefix(span.Name(), "execute_tool") {
			spanAttributes := span.Attributes()
			attributes := make(map[string]string)
			for _, attribute := range spanAttributes {
				key := string(attribute.Key)
				attributes[key] = attribute.Value.AsString()
			}
			attributes["trace_id"] = span.SpanContext().TraceID().String()
			attributes["span_id"] = span.SpanContext().SpanID().String()
			if eventID, ok := attributes["gcp.vertex.agent.event_id"]; ok {
				s.traceDict[eventID] = attributes
			}
		}
	}
	return nil
}

// Shutdown is a function that sdktrace.SpanExporter has, should close the span exporter connections.
// Since APIServerSpanExporter holds only in-memory dictionary, no additional logic required.
func (s *APIServerSpanExporter) Shutdown(ctx context.Context) error {
	return nil
}

var _ sdktrace.SpanExporter = (*APIServerSpanExporter)(nil)
