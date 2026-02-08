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

package telemetry

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
)

func TestTelemetrySmoke(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	ctx := t.Context()

	// Initialize telemetry.
	projectID := "test-project-id"
	serviceName := "test-service"
	serviceVersion := "1.2.3"
	r, err := resource.New(ctx, resource.WithAttributes(
		semconv.ServiceNameKey.String(serviceName),
		semconv.ServiceVersionKey.String(serviceVersion),
	))
	if err != nil {
		t.Fatalf("failed to create resource: %v", err)
	}
	service, err := New(t.Context(),
		WithSpanProcessors(sdktrace.NewSimpleSpanProcessor(exporter)),
		WithResourceProjectID(projectID),
		WithResource(r),
	)
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := service.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	service.SetGlobalOtelProviders()

	// Create test tracer.
	tracer := otel.Tracer("test-tracer")
	spanName := "test-span"

	_, span := tracer.Start(ctx, spanName, trace.WithSpanKind(trace.SpanKindServer))
	span.End()

	if err := service.TraceProvider().ForceFlush(context.Background()); err != nil {
		t.Fatalf("failed to flush spans: %v", err)
	}

	// Check exporter contains the span.
	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	gotSpan := spans[0]
	if gotSpan.Name != spanName {
		t.Errorf("got span name %q, want %q", gotSpan.Name, spanName)
	}
	gotProjectID, gotServiceName, gotServiceVersion := extractResourceAttributes(gotSpan.Resource)
	if gotProjectID != projectID {
		t.Errorf("want 'gcp.project_id' attribute %q, got %q", projectID, gotProjectID)
	}
	if gotServiceName != serviceName {
		t.Errorf("want 'service.name' attribute %q, got %q", serviceName, gotServiceName)
	}
	if gotServiceVersion != serviceVersion {
		t.Errorf("want 'service.version' attribute %q, got %q", serviceVersion, gotServiceVersion)
	}

	if err := service.Shutdown(context.WithoutCancel(ctx)); err != nil {
		t.Errorf("telemetry.Shutdown() failed: %v", err)
	}
	if len(exporter.GetSpans()) != 0 {
		t.Errorf("expected no spans after shutdown, got %d", len(exporter.GetSpans()))
	}
}

func TestTelemetryCustomProvider(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(exporter)),
	)
	ctx := t.Context()

	// Initialize telemetry with custom provider.
	service, err := New(t.Context(), WithTracerProvider(tp))
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := service.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	service.SetGlobalOtelProviders()

	// Create test tracer and span.
	tracer := otel.Tracer("test-tracer")
	spanName := "test-span"
	_, span := tracer.Start(ctx, spanName)
	span.End()

	if err := service.TraceProvider().ForceFlush(context.Background()); err != nil {
		t.Fatalf("failed to flush spans: %v", err)
	}

	// Verify span was exported.
	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	if spans[0].Name != spanName {
		t.Errorf("got span name %q, want %q", spans[0].Name, spanName)
	}
}

func extractResourceAttributes(res *resource.Resource) (projectID, serviceName, serviceVersion string) {
	for _, attr := range res.Attributes() {
		switch attr.Key {
		case "gcp.project_id":
			projectID = attr.Value.AsString()
		case semconv.ServiceNameKey:
			serviceName = attr.Value.AsString()
		case semconv.ServiceVersionKey:
			serviceVersion = attr.Value.AsString()
		}
	}
	return
}
