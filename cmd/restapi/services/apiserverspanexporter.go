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
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/attribute"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type APIServerSpanExporter struct {
	traceDict map[string]map[string]any
}

func NewAPIServerSpanExporter() *APIServerSpanExporter {
	return &APIServerSpanExporter{
		traceDict: make(map[string]map[string]any),
	}
}

func (s *APIServerSpanExporter) GetTraceDict() map[string]map[string]any {
	return s.traceDict
}

func (s *APIServerSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	fmt.Printf("ExportSpans: %v\n", spans)
	for _, span := range spans {
		if span.Name() == "call_llm" || span.Name() == "send_data" || strings.HasPrefix(span.Name(), "execute_tool") {
			spanAttributes := span.Attributes()
			attributes := make(map[string]any)
			for _, attribute := range spanAttributes {
				key := string(attribute.Key)
				attributes[key] = attribute.Value
			}
			attributes["trace_id"] = span.SpanContext().TraceID()
			attributes["span_id"] = span.SpanContext().SpanID()
			if event, ok := attributes["gcp.vertex.agent.event_id"]; ok {
				eventID, ok := event.(attribute.Value)
				if !ok {
					continue
				}
				s.traceDict[eventID.AsString()] = attributes
			}
		}
	}
	return nil
}

func (s *APIServerSpanExporter) Shutdown(ctx context.Context) error {
	return nil
}

var _ sdktrace.SpanExporter = (*APIServerSpanExporter)(nil)
