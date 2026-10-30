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
	"errors"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	resourceProject = "resource-project"
	quotaProject    = "quota-project"
)

func TestTelemetrySmoke(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	logExporter := &inMemoryLogExporter{}
	reader := sdkmetric.NewManualReader()
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
		WithMetricReaders(reader),
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
	record.SetBody(attribute.StringValue(logBody))
	logger.Emit(ctx, record)

	// Record through the global meter provider, as ADK instrumentation does.
	counter, err := otel.Meter("test-meter").Int64Counter("test-counter")
	if err != nil {
		t.Fatalf("failed to create counter: %v", err)
	}
	counter.Add(ctx, 1)

	if err := providers.TracerProvider.ForceFlush(t.Context()); err != nil {
		t.Fatalf("failed to flush spans: %v", err)
	}
	if err := providers.LoggerProvider.ForceFlush(t.Context()); err != nil {
		t.Fatalf("failed to flush logs: %v", err)
	}

	// Check the reader sees the metric, with the configured resource.
	var rm metricdata.ResourceMetrics
	if err := reader.Collect(ctx, &rm); err != nil {
		t.Fatalf("failed to collect metrics: %v", err)
	}
	if len(rm.ScopeMetrics) != 1 {
		t.Fatalf("got %d scope metrics, want 1", len(rm.ScopeMetrics))
	}
	if got, _, _ := extractResourceAttributes(rm.Resource); got != resourceProject {
		t.Errorf("metrics: want 'gcp.project_id' attribute %q, got %q", resourceProject, got)
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
	if err := reader.Collect(ctx, &rm); !errors.Is(err, sdkmetric.ErrReaderShutdown) {
		t.Errorf("Collect() after Shutdown error = %v, want %v", err, sdkmetric.ErrReaderShutdown)
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

	if err := providers.TracerProvider.ForceFlush(t.Context()); err != nil {
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
	record.SetBody(attribute.StringValue(logBody))
	logger.Emit(ctx, record)

	if err := providers.LoggerProvider.ForceFlush(t.Context()); err != nil {
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

func TestResolveResourceServiceInstanceID(t *testing.T) {
	ctx := t.Context()
	custom := resource.NewSchemaless(semconv.ServiceInstanceID("custom-instance"))

	testCases := []struct {
		name string
		opts []Option
		want *regexp.Regexp
	}{
		{
			// adk-python's `<uuid4 hex>-<pid>` form.
			name: "generated per process",
			want: regexp.MustCompile(`^[0-9a-f]{32}-` + strconv.Itoa(os.Getpid()) + `$`),
		},
		{
			name: "overridden by the config resource",
			opts: []Option{WithResource(custom)},
			want: regexp.MustCompile(`^custom-instance$`),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := configFromOpts(tc.opts...)
			if err != nil {
				t.Fatalf("configFromOpts() unexpected error: %v", err)
			}
			r, err := resolveResource(ctx, cfg)
			if err != nil {
				t.Fatalf("resolveResource() unexpected error: %v", err)
			}
			got, ok := r.Set().Value(semconv.ServiceInstanceIDKey)
			if !ok {
				t.Fatal("resource has no 'service.instance.id' attribute")
			}
			if !tc.want.MatchString(got.AsString()) {
				t.Errorf("got 'service.instance.id' %q, want a match for %v", got.AsString(), tc.want)
			}
		})
	}
}

func TestTelemetryCustomMeterProvider(t *testing.T) {
	mp := sdkmetric.NewMeterProvider()
	providers, err := New(t.Context(), WithMeterProvider(mp), WithMetricReaders(sdkmetric.NewManualReader()))
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.WithoutCancel(t.Context())); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})
	if providers.MeterProvider != mp {
		t.Errorf("got MeterProvider %p, want the one passed to WithMeterProvider %p", providers.MeterProvider, mp)
	}
}

func TestTelemetryNoReadersNoMeterProvider(t *testing.T) {
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
	t.Setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "")
	providers, err := New(t.Context())
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	if providers.MeterProvider != nil {
		t.Errorf("got MeterProvider %v, want nil without readers", providers.MeterProvider)
	}
	// No typed nil is installed globally.
	otel.SetMeterProvider(sdkmetric.NewMeterProvider())
	providers.SetGlobalOtelProviders()
	if mp, ok := otel.GetMeterProvider().(*sdkmetric.MeterProvider); !ok || mp == nil {
		t.Errorf("otel.GetMeterProvider() = %#v, want the previous provider kept", otel.GetMeterProvider())
	}
}

func TestOtlpEnvMetricShutdownDoesNotExport(t *testing.T) {
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", srv.URL)
	t.Setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "")
	ctx := t.Context()

	providers, err := New(ctx)
	if err != nil {
		t.Fatalf("failed to create telemetry: %v", err)
	}
	if providers.MeterProvider == nil {
		t.Fatal("MeterProvider is nil, want one for OTEL_EXPORTER_OTLP_ENDPOINT")
	}
	counter, err := providers.MeterProvider.Meter("test-meter").Int64Counter("test-counter")
	if err != nil {
		t.Fatalf("failed to create counter: %v", err)
	}
	counter.Add(ctx, 1)
	if err := providers.Shutdown(ctx); err != nil {
		t.Fatalf("telemetry.Shutdown() failed: %v", err)
	}
	if got := requests.Load(); got != 0 {
		t.Errorf("got %d export requests on Shutdown, want 0", got)
	}
}

// recordingTransport captures every request and answers it with an empty success.
type recordingTransport struct {
	mu       sync.Mutex
	requests []*http.Request
}

func (rt *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.requests = append(rt.requests, req)
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       http.NoBody,
		Request:    req,
	}, nil
}

func TestGcpExporters(t *testing.T) {
	transport := &recordingTransport{}
	// oauth2.NewClient wraps the transport of the client found in the context.
	ctx := context.WithValue(t.Context(), oauth2.HTTPClient, &http.Client{Transport: transport})
	creds := &google.Credentials{
		ProjectID:   resourceProject,
		TokenSource: oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "test-token"}),
	}
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
	t.Setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
	t.Setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "")
	t.Setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "")

	cfg, err := configure(ctx,
		WithOtelToCloud(true),
		WithGoogleCredentials(creds),
		WithGcpQuotaProject(quotaProject),
	)
	if err != nil {
		t.Fatalf("configure() unexpected error: %v", err)
	}
	if len(cfg.metricReaders) != 1 {
		t.Fatalf("got %d metric readers, want 1", len(cfg.metricReaders))
	}
	// The literal, not the constant: Cloud Monitoring's floor is what the test pins.
	if got := readerInterval(t, cfg.metricReaders[0]); got != 5*time.Second {
		t.Errorf("got metric export interval %v, want 5s", got)
	}

	providers, err := newInternal(cfg)
	if err != nil {
		t.Fatalf("newInternal() unexpected error: %v", err)
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.WithoutCancel(ctx)); err != nil {
			t.Errorf("telemetry.Shutdown() failed: %v", err)
		}
	})

	_, span := providers.TracerProvider.Tracer("test-tracer").Start(ctx, "test-span")
	span.End()
	counter, err := providers.MeterProvider.Meter("test-meter").Int64Counter("test-counter")
	if err != nil {
		t.Fatalf("failed to create counter: %v", err)
	}
	counter.Add(ctx, 1)
	if err := providers.TracerProvider.ForceFlush(ctx); err != nil {
		t.Fatalf("failed to flush spans: %v", err)
	}
	if err := providers.MeterProvider.ForceFlush(ctx); err != nil {
		t.Fatalf("failed to flush metrics: %v", err)
	}

	// Requests per endpoint; both exporters share the client, so headers are checked on each.
	requests := func() map[string][]http.Header {
		transport.mu.Lock()
		defer transport.mu.Unlock()
		got := make(map[string][]http.Header)
		for _, req := range transport.requests {
			got[req.URL.String()] = append(got[req.URL.String()], req.Header)
		}
		return got
	}
	got := requests()
	for _, endpoint := range []string{gcpTracesEndpoint, gcpMetricsEndpoint} {
		headers, ok := got[endpoint]
		if !ok {
			t.Errorf("no request to %s, got %v", endpoint, slices.Sorted(maps.Keys(got)))
			continue
		}
		if len(headers) != 1 {
			t.Errorf("%s: got %d requests, want 1", endpoint, len(headers))
		}
		for _, header := range headers {
			if header.Get("Authorization") != "Bearer test-token" {
				t.Errorf("%s: Authorization header does not carry the credentials' token", endpoint)
			}
			if header.Get("x-goog-user-project") != quotaProject {
				t.Errorf("%s: got x-goog-user-project %q, want %q", endpoint, header.Get("x-goog-user-project"), quotaProject)
			}
		}
	}

	// Shutdown exports no further metrics.
	if err := providers.Shutdown(ctx); err != nil {
		t.Fatalf("telemetry.Shutdown() failed: %v", err)
	}
	if got := len(requests()[gcpMetricsEndpoint]); got != 1 {
		t.Errorf("got %d metric requests after Shutdown, want still 1", got)
	}
}

// readerInterval reads a periodic reader's interval off its logging view,
// the only place the SDK exposes it.
func readerInterval(t *testing.T, r sdkmetric.Reader) time.Duration {
	t.Helper()
	ml, ok := r.(interface{ MarshalLog() any })
	if !ok {
		t.Fatalf("reader %T has no MarshalLog", r)
	}
	field := reflect.ValueOf(ml.MarshalLog()).FieldByName("Interval")
	if !field.IsValid() {
		t.Fatalf("MarshalLog() of %T has no Interval field", r)
	}
	interval, ok := field.Interface().(time.Duration)
	if !ok {
		t.Fatalf("MarshalLog() Interval of %T is %T, want time.Duration", r, field.Interface())
	}
	return interval
}

// recordingMetricExporter counts calls; the exported data is irrelevant.
type recordingMetricExporter struct {
	mu        sync.Mutex
	exports   int
	shutdowns int
}

func (e *recordingMetricExporter) Temporality(k sdkmetric.InstrumentKind) metricdata.Temporality {
	return sdkmetric.DefaultTemporalitySelector(k)
}

func (e *recordingMetricExporter) Aggregation(k sdkmetric.InstrumentKind) sdkmetric.Aggregation {
	return sdkmetric.DefaultAggregationSelector(k)
}

func (e *recordingMetricExporter) Export(context.Context, *metricdata.ResourceMetrics) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exports++
	return nil
}

func (e *recordingMetricExporter) ForceFlush(context.Context) error { return nil }

func (e *recordingMetricExporter) Shutdown(context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.shutdowns++
	return nil
}

func (e *recordingMetricExporter) counts() (exports, shutdowns int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.exports, e.shutdowns
}

func TestPeriodicReader(t *testing.T) {
	ctx := t.Context()
	exporter := &recordingMetricExporter{}
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(newPeriodicReader(exporter, sdkmetric.WithInterval(time.Hour))))
	counter, err := mp.Meter("test-meter").Int64Counter("test-counter")
	if err != nil {
		t.Fatalf("failed to create counter: %v", err)
	}

	// An instrument that has not recorded yields an empty collection, which
	// is not exported.
	if err := mp.ForceFlush(ctx); err != nil {
		t.Fatalf("MeterProvider.ForceFlush() failed: %v", err)
	}
	if exports, _ := exporter.counts(); exports != 0 {
		t.Errorf("got %d exports before any recording, want 0", exports)
	}

	counter.Add(ctx, 1)
	if err := mp.ForceFlush(ctx); err != nil {
		t.Fatalf("MeterProvider.ForceFlush() failed: %v", err)
	}
	if exports, _ := exporter.counts(); exports != 1 {
		t.Errorf("got %d exports after ForceFlush, want 1", exports)
	}
	if err := mp.Shutdown(ctx); err != nil {
		t.Fatalf("MeterProvider.Shutdown() failed: %v", err)
	}
	if exports, shutdowns := exporter.counts(); exports != 1 || shutdowns != 1 {
		t.Errorf("after Shutdown got %d exports, %d exporter shutdowns; want 1, 1", exports, shutdowns)
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
	OTEL_EXPORTER_OTLP_ENDPOINT         string
	OTEL_EXPORTER_OTLP_TRACES_ENDPOINT  string
	OTEL_EXPORTER_OTLP_METRICS_ENDPOINT string
	OTEL_EXPORTER_OTLP_LOGS_ENDPOINT    string
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
		wantMetricReaders  int
	}{
		{
			name:               "no processors",
			envVars:            envVars{},
			wantSpanProcessors: 0,
			wantLogProcessors:  0,
			wantMetricReaders:  0,
		},
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			wantSpanProcessors: 1,
			wantLogProcessors:  1,
			wantMetricReaders:  1,
		},
		{
			name: "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://localhost:4318/v1/traces",
			},
			wantSpanProcessors: 1,
			wantLogProcessors:  0,
			wantMetricReaders:  0,
		},
		{
			name: "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: "http://localhost:4318/v1/metrics",
			},
			wantSpanProcessors: 0,
			wantLogProcessors:  0,
			wantMetricReaders:  1,
		},
		{
			name: "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: "http://localhost:4318/v1/logs",
			},
			wantSpanProcessors: 0,
			wantLogProcessors:  1,
			wantMetricReaders:  0,
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
			wantMetricReaders:  2,
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
			wantMetricReaders:  1,
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
			wantMetricReaders:  1,
		},
		// Nothing is built for a supplied provider: it would ignore it, and an
		// unregistered periodic reader reports an error on every tick.
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT, otel_to_cloud and WithTracerProvider",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
				WithTracerProvider(sdktrace.NewTracerProvider()),
			},
			wantSpanProcessors: 0,
			wantLogProcessors:  1,
			wantMetricReaders:  2,
		},
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT, otel_to_cloud and WithLoggerProvider",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
				WithLoggerProvider(sdklog.NewLoggerProvider()),
			},
			wantSpanProcessors: 2,
			wantLogProcessors:  0,
			wantMetricReaders:  2,
		},
		{
			name: "OTEL_EXPORTER_OTLP_ENDPOINT, otel_to_cloud and WithMeterProvider",
			envVars: envVars{
				OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4318",
			},
			opts: []Option{
				WithOtelToCloud(true),
				WithGoogleCredentials(&google.Credentials{ProjectID: "test-project"}),
				WithMeterProvider(sdkmetric.NewMeterProvider()),
			},
			wantSpanProcessors: 2,
			wantLogProcessors:  1,
			wantMetricReaders:  0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_ENDPOINT)
			t.Setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
			t.Setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
			t.Setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", tc.envVars.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
			// Set the quota project needed to configure GCP exporters.
			t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
			ctx := t.Context()
			cfg, err := configure(ctx, tc.opts...)
			if err != nil {
				t.Fatalf("configure() unexpected error: %v", err)
			}
			shutdownReaders(t, cfg.metricReaders)
			got, err := configureExporters(ctx, cfg)
			if err != nil {
				t.Fatalf("configureExporters() unexpected error: %v", err)
			}
			shutdownReaders(t, got.metricReaders)
			if len(got.spanProcessors) != tc.wantSpanProcessors {
				t.Errorf("got %d span processors, want %d", len(got.spanProcessors), tc.wantSpanProcessors)
			}
			if len(got.logProcessors) != tc.wantLogProcessors {
				t.Errorf("got %d log processors, want %d", len(got.logProcessors), tc.wantLogProcessors)
			}
			if len(got.metricReaders) != tc.wantMetricReaders {
				t.Errorf("got %d metric readers, want %d", len(got.metricReaders), tc.wantMetricReaders)
			}
		})
	}
}

// shutdownReaders stops the goroutines of periodic readers the test created
// but never registered with a provider.
func shutdownReaders(t *testing.T, readers []sdkmetric.Reader) {
	t.Helper()
	t.Cleanup(func() {
		for _, r := range readers {
			if err := r.Shutdown(context.WithoutCancel(t.Context())); err != nil {
				t.Errorf("reader Shutdown() failed: %v", err)
			}
		}
	})
}
