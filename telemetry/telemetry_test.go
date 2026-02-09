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
	"golang.org/x/oauth2/google"
)

const (
	resourceProject = "resource-project"
	quotaProject    = "quota-project"
)

func TestTelemetrySmoke(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
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
	service, err := New(t.Context(),
		WithSpanProcessors(sdktrace.NewSimpleSpanProcessor(exporter)),
		WithGcpResourceProject(resourceProject),
		WithGcpQuotaProject(quotaProject),
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

	if err := service.TracerProvider().ForceFlush(context.Background()); err != nil {
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
	service, err := New(t.Context(),
		WithTracerProvider(tp),
		WithGcpResourceProject(resourceProject),
		WithGcpQuotaProject(quotaProject),
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

	// Create test tracer and span.
	tracer := otel.Tracer("test-tracer")
	spanName := "test-span"
	_, span := tracer.Start(ctx, spanName)
	span.End()

	if err := service.TracerProvider().ForceFlush(context.Background()); err != nil {
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

func TestResolveResourceProject(t *testing.T) {
	testCases := []struct {
		name        string
		cfg         *config
		envVar      string
		wantProject string
		wantErr     bool
	}{
		{
			name: "project from config",
			cfg: &config{
				oTelToCloud:        true,
				gcpResourceProject: "config-project",
				googleCredentials:  &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:      "env-project",
			wantProject: "config-project",
		},
		{
			name: "project from credentials",
			cfg: &config{
				oTelToCloud:       true,
				googleCredentials: &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:      "env-project",
			wantProject: "cred-project",
		},
		{
			name: "project from env var",
			cfg: &config{
				oTelToCloud: true,
			},
			envVar:      "env-project",
			wantProject: "env-project",
		},
		{
			name: "no project",
			cfg: &config{
				oTelToCloud:       true,
				googleCredentials: &google.Credentials{},
			},
			wantErr: true,
		},
		{
			name: "no project and otelToCloud disabled",
			cfg: &config{
				oTelToCloud: false,
			},
			wantProject: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.envVar != "" {
				t.Setenv("GOOGLE_CLOUD_PROJECT", tc.envVar)
			}

			gotProject, err := resolveGcpResourceProject(tc.cfg)
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
		cfg         *config
		envVar      string
		wantProject string
		wantErr     bool
	}{
		{
			name: "project from config",
			cfg: &config{
				oTelToCloud:       true,
				gcpQuotaProject:   "config-project",
				googleCredentials: &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:      "env-project",
			wantProject: "config-project",
		},
		{
			name: "project from credentials",
			cfg: &config{
				oTelToCloud:       true,
				googleCredentials: &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:      "env-project",
			wantProject: "cred-project",
		},
		{
			name: "project from env var",
			cfg: &config{
				oTelToCloud: true,
			},
			envVar:      "env-project",
			wantProject: "env-project",
		},
		{
			name: "no project",
			cfg: &config{
				oTelToCloud:       true,
				googleCredentials: &google.Credentials{},
			},
			wantErr: true,
		},
		{
			name: "no project and otelToCloud disabled",
			cfg: &config{
				oTelToCloud: false,
			},
			wantProject: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.envVar != "" {
				t.Setenv("GOOGLE_CLOUD_PROJECT", tc.envVar)
			}

			gotProject, err := resolveGcpQuotaProject(tc.cfg)
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
