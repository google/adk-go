package services

import (
	"context"
	"fmt"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type APIServerSpanExporter struct {
	traceDict map[string]map[string]any
}

func (s *APIServerSpanExporter) GetTraceDict() map[string]map[string]any {
	return s.traceDict
}

func (s *APIServerSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	fmt.Printf("ExportSpans: %v\n", spans)
	for _, span := range spans {
		fmt.Printf("span: %v\n", span)
		fmt.Printf("span name: %v\n", span.Name())
		// if span.Name() == "call_llm" || span.Name() == "send_data" || strings.HasPrefix(span.Name(), "execute_tool") {
		spanAttributes := span.Attributes()
		attributes := make(map[string]any)
		for _, attribute := range spanAttributes {
			key := string(attribute.Key)
			attributes[key] = attribute.Value
		}
		fmt.Printf("attributes: %v\n", attributes)
		attributes["trace_id"] = span.SpanContext().TraceID()
		attributes["span_id"] = span.SpanContext().SpanID()
		if event, ok := attributes["gcp.vertex.agent.event_id"]; ok {
			eventID, ok := event.(string)
			if !ok {
				// fmt.Printf()
				continue
			}
			s.traceDict[eventID] = attributes
		}
		// }
	}
	return nil
}

func (s *APIServerSpanExporter) Shutdown(ctx context.Context) error {
	return nil
}

var _ sdktrace.SpanExporter = (*APIServerSpanExporter)(nil)
