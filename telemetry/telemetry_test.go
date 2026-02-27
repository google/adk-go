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
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/oauth2/google"
)

const (
	resourceProject = "resource-project"
	quotaProject    = "quota-project"
)

func TestTelemetrySmoke(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	logExporter := &inMemoryLogExporter{}
	ctx := t.Context()

	// Initialize telemetry.
	serviceName := "test-service"
	serviceVersion := "1.2.3"
	r, err := resource.New(ctx, resource.WithAttributes(
		semconv.ServiceNameKey.String(serviceName),
		semconv.ServiceVersionKey.String(serviceVersion),
	))
	if err != nil {
		t.Fatalf("failed to create resource: %v", err)
	}
	providers, err := New(t.Context(),
		WithSpanProcessors(sdktrace.NewSimpleSpanProcessor(exporter)),
		WithLogRecordProcessors(sdklog.NewSimpleProcessor(logExporter)),
		WithGcpResourceProject(resourceProject),
		WithGcpQuotaProject(quotaProject),
		WithResource(r),
	)
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	providers.SetGlobalOtelProviders()

	// Create test tracer.
	tracer := otel.Tracer("test-tracer")
	spanName := "test-span"

	_, span := tracer.Start(ctx, spanName, trace.WithSpanKind(trace.SpanKindServer))
	span.End()

	// Create test logger and log.
	logger := providers.LoggerProvider.Logger("test-logger")
	logBody := "test-log"

	var record log.Record
	record.SetBody(log.StringValue(logBody))
	logger.Emit(ctx, record)

	if err := providers.TracerProvider.ForceFlush(context.Background()); err != nil {
		t.Fatalf("failed to flush spans: %v", err)
	}
	if err := providers.LoggerProvider.ForceFlush(context.Background()); err != nil {
		t.Fatalf("failed to flush logs: %v", err)
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
	gotResourceProject, gotServiceName, gotServiceVersion := extractResourceAttributes(gotSpan.Resource)
	if gotResourceProject != resourceProject {
		t.Errorf("want 'gcp.project_id' attribute %q, got %q", resourceProject, gotResourceProject)
	}
	if gotServiceName != serviceName {
		t.Errorf("want 'service.name' attribute %q, got %q", serviceName, gotServiceName)
	}
	if gotServiceVersion != serviceVersion {
		t.Errorf("want 'service.version' attribute %q, got %q", serviceVersion, gotServiceVersion)
	}

	// Check exporter contains the log.
	if len(logExporter.records) != 1 {
		t.Fatalf("got %d log records, want 1", len(logExporter.records))
	}
	gotLog := logExporter.records[0]
	if gotLog.Body().AsString() != logBody {
		t.Errorf("got log body %q, want %q", gotLog.Body().AsString(), logBody)
	}

	if err := providers.Shutdown(context.WithoutCancel(ctx)); err != nil {
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
	unusedExporter := tracetest.NewInMemoryExporter()
	ctx := t.Context()

	// Initialize telemetry with custom provider.
	providers, err := New(t.Context(),
		WithTracerProvider(tp),
		WithSpanProcessors(sdktrace.NewSimpleSpanProcessor(unusedExporter)),
	)
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	providers.SetGlobalOtelProviders()

	// Create test tracer and span.
	tracer := otel.Tracer("test-tracer")
	spanName := "test-span"
	_, span := tracer.Start(ctx, spanName)
	span.End()

	if err := providers.TracerProvider.ForceFlush(context.Background()); err != nil {
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

	// Unused exporter should not have any spans.
	if len(unusedExporter.GetSpans()) != 0 {
		t.Fatalf("got %d spans, want 0", len(unusedExporter.GetSpans()))
	}
}

func TestTelemetryCustomLoggerProvider(t *testing.T) {
	logExporter := &inMemoryLogExporter{}
	lp := sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExporter)),
	)
	unusedLogExporter := &inMemoryLogExporter{}
	ctx := t.Context()

	// Initialize telemetry with custom logger provider.
	providers, err := New(t.Context(),
		WithLoggerProvider(lp),
		WithLogRecordProcessors(sdklog.NewSimpleProcessor(unusedLogExporter)),
	)
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	providers.SetGlobalOtelProviders()

	// Create test logger and emit.
	logger := providers.LoggerProvider.Logger("test-logger")
	logBody := "test-log"

	var record log.Record
	record.SetBody(log.StringValue(logBody))
	logger.Emit(ctx, record)

	if err := providers.LoggerProvider.ForceFlush(context.Background()); err != nil {
		t.Fatalf("failed to flush logs: %v", err)
	}

	// Verify log was exported.
	if len(logExporter.records) != 1 {
		t.Fatalf("got %d logs, want 1", len(logExporter.records))
	}
	if logExporter.records[0].Body().AsString() != logBody {
		t.Errorf("got log body %q, want %q", logExporter.records[0].Body().AsString(), logBody)
	}

	// Unused exporter should not have any logs.
	if len(unusedLogExporter.records) != 0 {
		t.Fatalf("got %d logs, want 0", len(unusedLogExporter.records))
	}
}

func extractResourceAttributes(res *resource.Resource) (string, string, string) {
	var projectID string
	var serviceName string
	var serviceVersion string

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

	return projectID, serviceName, serviceVersion
}

func TestResolveResourceProject(t *testing.T) {
	testCases := []struct {
		name        string
		opts        []Option
		envVar      string
		wantProject string
		wantErr     bool
	}{
		{
			name: "project from options",
			opts: []Option{
				WithOtelToCloud(true),
				WithGcpResourceProject("option-project"),
				WithGoogleCredentials(&google.Credentials{ProjectID: "cred-project"}),
			},
			envVar:      "env-project",
			wantProject: "option-project",
		},
		{
			name: "project from credentials",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "cred-project"}),
			},
			envVar:      "env-project",
			wantProject: "cred-project",
		},
		{
			name: "project from env var",
			opts: []Option{
				WithOtelToCloud(true),
			},
			envVar:      "env-project",
			wantProject: "env-project",
		},
		{
			name: "no project",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{}),
			},
			wantErr: true,
		},
		{
			name: "no project no credentials",
			opts: []Option{
				WithOtelToCloud(true),
			},
			wantErr: true,
		},
		{
			name: "env var whitespace",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{}),
			},
			envVar:  " ",
			wantErr: true,
		},
		{
			name: "option project whitespace",
			opts: []Option{
				WithOtelToCloud(true),
				WithGcpResourceProject(" "),
			},
			wantErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Always set the environment variable to avoid flakiness from ambient GOOGLE_CLOUD_PROJECT.
			t.Setenv("GOOGLE_CLOUD_PROJECT", tc.envVar)

			cfg, err := configFromOpts(tc.opts...)
			if err != nil {
				t.Fatalf("configFromOpts() unexpected error: %v", err)
			}

			gotProject, err := resolveGcpResourceProject(cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("resolveGcpResourceProject() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}

			if gotProject != tc.wantProject {
				t.Errorf("resolveGcpResourceProject() got = %v, want %v", gotProject, tc.wantProject)
			}
		})
	}
}

func TestResolveQuotaProject(t *testing.T) {
	testCases := []struct {
		name        string
		opts        []Option
		envVar      string
		wantProject string
		wantErr     bool
	}{
		{
			name: "project from options",
			opts: []Option{
				WithOtelToCloud(true),
				WithGcpQuotaProject("option-project"),
				WithGoogleCredentials(&google.Credentials{ProjectID: "cred-project"}),
			},
			envVar:      "env-project",
			wantProject: "option-project",
		},
		{
			name: "project from credentials",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "cred-project"}),
			},
			envVar:      "env-project",
			wantProject: "cred-project",
		},
		{
			name: "project from env var",
			opts: []Option{
				WithOtelToCloud(true),
			},
			envVar:      "env-project",
			wantProject: "env-project",
		},
		{
			name: "no project",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{}),
			},
			wantErr: true,
		},
		{
			name: "no project no credentials",
			opts: []Option{
				WithOtelToCloud(true),
			},
			wantErr: true,
		},
		{
			name: "no project and otelToCloud disabled",
			opts: []Option{
				WithOtelToCloud(false),
				WithGoogleCredentials(&google.Credentials{}),
			},
			wantProject: "",
		},
		{
			name: "env var whitespace",
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{}),
			},
			envVar:  " ",
			wantErr: true,
		},
		{
			name: "option project whitespace",
			opts: []Option{
				WithOtelToCloud(true),
				WithGcpQuotaProject(" "),
			},
			wantErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Always set the environment variable to avoid flakiness from ambient GOOGLE_CLOUD_PROJECT.
			t.Setenv("GOOGLE_CLOUD_PROJECT", tc.envVar)

			cfg, err := configFromOpts(tc.opts...)
			if err != nil {
				t.Fatalf("configFromOpts() unexpected error: %v", err)
			}

			gotProject, err := resolveGcpQuotaProject(cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("resolveGcpQuotaProject() error = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}

			if gotProject != tc.wantProject {
				t.Errorf("resolveGcpQuotaProject() got = %v, want %v", gotProject, tc.wantProject)
			}
		})
	}
}

type inMemoryLogExporter struct {
	records []sdklog.Record
}

func (e *inMemoryLogExporter) Export(_ context.Context, records []sdklog.Record) error {
	e.records = append(e.records, records...)
	return nil
}

func (e *inMemoryLogExporter) Shutdown(context.Context) error   { return nil }
func (e *inMemoryLogExporter) ForceFlush(context.Context) error { return nil }

type envVars struct {
	OTEL_EXPORTER_OTLP_ENDPOINT        string
	OTEL_EXPORTER_OTLP_TRACES_ENDPOINT string
	OTEL_EXPORTER_OTLP_LOGS_ENDPOINT   string
}

func TestConfigureExporters(t *testing.T) {
	testCases := []struct {
		name    string
		envVars envVars
		opts    []Option
		// The client address is nested deep inside the http client of the exporter, which is nested in a processor.
		// Accessing it via reflection is too brittle. The best thing we can do is a smoke test, which checks the number of created processors.
		wantSpanProcessors int
		wantLogProcessors  int
	}{
		{
			name:               "no processors",
			envVars:            envVars{},
			wantSpanProcessors: 0,
			wantLogProcessors:  0,
		},
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			wantSpanProcessors: 1,
			wantLogProcessors:  1,
		},
		{
			name: "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://localhost:4318/v1/traces",
			},
			wantSpanProcessors: 1,
			wantLogProcessors:  0,
		},
		{
			name: "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: "http://localhost:4318/v1/logs",
			},
			wantSpanProcessors: 0,
			wantLogProcessors:  1,
		},
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT and otel_to_cloud",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
			},
			wantSpanProcessors: 2,
			wantLogProcessors:  1,
		},
		{
			name: "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT and otel_to_cloud",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://localhost:4318/v1/traces",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
			},
			wantSpanProcessors: 2,
			wantLogProcessors:  0,
		},
		{
			name: "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT and otel_to_cloud",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: "http://localhost:4318/v1/logs",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
			},
			wantSpanProcessors: 1,
			wantLogProcessors:  1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_ENDPOINT)
			t.Setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
			t.Setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
			// Set the quota project needed to configure GCP exporters.
			t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
			ctx := t.Context()
			cfg, err := configure(ctx, tc.opts...)
			if err != nil {
				t.Fatalf("configure() unexpected error: %v", err)
			}
			spanProcessors, logProcessors, err := configureExporters(ctx, cfg)
			if err != nil {
				t.Fatalf("configureExporters() unexpected error: %v", err)
			}
			if len(spanProcessors) != tc.wantSpanProcessors {
				t.Errorf("got %d span processors, want %d", len(spanProcessors), tc.wantSpanProcessors)
			}
			if len(logProcessors) != tc.wantLogProcessors {
				t.Errorf("got %d log processors, want %d", len(logProcessors), tc.wantLogProcessors)
			}
		})
	}
}
