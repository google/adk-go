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
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/contrib/detectors/gcp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploghttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	gcpTracesEndpoint  = "https://telemetry.googleapis.com/v1/traces"
	gcpMetricsEndpoint = "https://telemetry.googleapis.com/v1/metrics"

	gcpMetricExportInterval = 5 * time.Second
)

func configure(ctx context.Context, opts ...Option) (*config, error) {
	cfg, err := configFromOpts(opts...)
	if err != nil {
		return nil, err
	}

	if cfg.oTelToCloud {
		// Load ADC if no credentials are provided in the config.
		if cfg.googleCredentials == nil {
			cfg.googleCredentials, err = google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
			if err != nil {
				return nil, fmt.Errorf("failed to find default credentials: %w", err)
			}
		}
		quotaProject, err := resolveGcpQuotaProject(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve GCP quota project: %w", err)
		}
		cfg.gcpQuotaProject = quotaProject
		resourceProject, err := resolveGcpResourceProject(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve GCP resource project: %w", err)
		}
		cfg.gcpResourceProject = resourceProject
	}

	cfg.resource, err = resolveResource(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve resource: %w", err)
	}

	exp, err := configureExporters(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to configure exporters: %w", err)
	}
	cfg.spanProcessors = append(cfg.spanProcessors, exp.spanProcessors...)
	cfg.logProcessors = append(cfg.logProcessors, exp.logProcessors...)
	cfg.metricReaders = append(cfg.metricReaders, exp.metricReaders...)
	return cfg, nil
}

func configFromOpts(opts ...Option) (*config, error) {
	cfg := &config{}

	for _, opt := range opts {
		if err := opt.apply(cfg); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}
	return cfg, nil
}

func newInternal(cfg *config) (*Providers, error) {
	tp := initTracerProvider(cfg)
	lp := initLoggerProvider(cfg)
	mp := initMeterProvider(cfg)

	return &Providers{
		TracerProvider: tp,
		LoggerProvider: lp,
		MeterProvider:  mp,
	}, nil
}

// resolveGcpQuotaProject determines the quota project for telemetry export in the following order:
// 1. config.gcpQuotaProject, if present.
// 2. Project ID from credentials, if present.
// 3. GOOGLE_CLOUD_PROJECT environment variable.
// Returns the quota project or error if the quota project cannot be determined.
func resolveGcpQuotaProject(cfg *config) (string, error) {
	return resolveProject(cfg.gcpQuotaProject, cfg.googleCredentials, cfg.oTelToCloud, "quota")
}

// resolveGcpResourceProject determines the resource project for telemetry export in the following order:
// 1. config.gcpResourceProject, if present.
// 2. Project ID from credentials, if present.
// 3. GOOGLE_CLOUD_PROJECT environment variable.
// Returns the resource project or error if the resource project cannot be determined.
func resolveGcpResourceProject(cfg *config) (string, error) {
	return resolveProject(cfg.gcpResourceProject, cfg.googleCredentials, cfg.oTelToCloud, "resource")
}

func resolveProject(configuredProject string, creds *google.Credentials, requireProject bool, projectType string) (string, error) {
	configuredProject = strings.TrimSpace(configuredProject)
	if configuredProject != "" {
		return configuredProject, nil
	}
	if creds != nil && creds.ProjectID != "" {
		return creds.ProjectID, nil
	}
	// The project was always empty during testing, even though it was set in ADC JSON file.
	// Using fallback to env variable to resolve the project as a workaround.
	project := strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_PROJECT"))
	if requireProject && project == "" {
		return "", fmt.Errorf("telemetry.googleapis.com requires setting the %s project. Refer to telemetry.config for the available options to set the %s project", projectType, projectType)
	}
	return project, nil
}

// resolveResource creates a new resource with attributes specified in the following order (later attributes override earlier ones):
//  1. A `service.instance.id` unique to this process.
//  2. [resource.Default()] populates the resource labels from environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
//  3. GCP related attributes when otelToCloud is enabled or gcpResourceProject is set - `gcp.project_id` attribute needed by Cloud Trace and other attributes provided by [gcp.NewDetector()].
//  4. GCP detector adds runtime attributes if ADK runs on one of supported platforms (e.g. GCE, GKE, CloudRun).
//  5. Resource from config, if present.
func resolveResource(ctx context.Context, cfg *config) (*resource.Resource, error) {
	r, err := resource.Merge(resource.NewSchemaless(semconv.ServiceInstanceID(serviceInstanceID())), resource.Default())
	if err != nil {
		return nil, fmt.Errorf("failed to merge default resource: %w", err)
	}

	opts := []resource.Option{
		resource.WithAttributes(
			attribute.Key("gcp.project_id").String(cfg.gcpResourceProject),
		),
	}
	if cfg.oTelToCloud {
		// Add GCP specific detectors.
		opts = append(opts, resource.WithDetectors(gcp.NewDetector()))
	}
	gcpResource, err := resource.New(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCP resource: %w", err)
	}
	// Merge with the default resource.
	r, err = resource.Merge(r, gcpResource)
	if err != nil {
		return nil, fmt.Errorf("failed to merge default and GCP resources: %w", err)
	}
	// Lastly, merge with the resource from config.
	if cfg.resource != nil {
		r, err = resource.Merge(r, cfg.resource)
		if err != nil {
			return nil, fmt.Errorf("failed to merge with config resource: %w", err)
		}
	}
	return r, nil
}

// serviceInstanceID returns an id unique to this process, in the same
// `<uuid hex>-<pid>` form as ADK Python.
func serviceInstanceID() string {
	u := uuid.New()
	return hex.EncodeToString(u[:]) + "-" + strconv.Itoa(os.Getpid())
}

type exporters struct {
	spanProcessors []sdktrace.SpanProcessor
	logProcessors  []sdklog.Processor
	metricReaders  []sdkmetric.Reader
}

func (e *exporters) add(other exporters) {
	e.spanProcessors = append(e.spanProcessors, other.spanProcessors...)
	e.logProcessors = append(e.logProcessors, other.logProcessors...)
	e.metricReaders = append(e.metricReaders, other.metricReaders...)
}

// ownProviders says which providers [New] builds itself, as opposed to taking
// from [WithTracerProvider], [WithLoggerProvider] or [WithMeterProvider].
type ownProviders struct {
	tracer, logger, meter bool
}

// configureExporters initializes OTel exporters from environment variables and otelToCloud.
func configureExporters(ctx context.Context, cfg *config) (exporters, error) {
	own := ownProviders{
		tracer: cfg.tracerProvider == nil,
		logger: cfg.loggerProvider == nil,
		meter:  cfg.meterProvider == nil,
	}
	exp, err := otlpExporters(ctx, own)
	if err != nil {
		return exporters{}, err
	}
	if cfg.oTelToCloud {
		gcp, err := gcpExporters(ctx, cfg, own)
		if err != nil {
			return exporters{}, err
		}
		exp.add(gcp)
	}
	return exp, nil
}

func otlpExporters(ctx context.Context, own ownProviders) (exporters, error) {
	endpoint := strings.TrimSpace(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
	endpointSet := func(signal string) bool {
		return endpoint != "" || strings.TrimSpace(os.Getenv("OTEL_EXPORTER_OTLP_"+signal+"_ENDPOINT")) != ""
	}

	var exp exporters
	if own.tracer && endpointSet("TRACES") {
		exporter, err := otlptracehttp.New(ctx)
		if err != nil {
			return exporters{}, fmt.Errorf("failed to create OTLP HTTP exporter: %w", err)
		}
		exp.spanProcessors = append(exp.spanProcessors, sdktrace.NewBatchSpanProcessor(exporter))
	}
	if own.meter && endpointSet("METRICS") {
		exporter, err := otlpmetrichttp.New(ctx)
		if err != nil {
			return exporters{}, fmt.Errorf("failed to create OTLP HTTP metric exporter: %w", err)
		}
		exp.metricReaders = append(exp.metricReaders, newPeriodicReader(exporter))
	}
	if own.logger && endpointSet("LOGS") {
		exporter, err := otlploghttp.New(ctx)
		if err != nil {
			return exporters{}, fmt.Errorf("failed to create OTLP HTTP log exporter: %w", err)
		}
		exp.logProcessors = append(exp.logProcessors, sdklog.NewBatchProcessor(exporter))
	}
	return exp, nil
}

func gcpExporters(ctx context.Context, cfg *config, own ownProviders) (exporters, error) {
	client := oauth2.NewClient(ctx, cfg.googleCredentials.TokenSource)
	headers := map[string]string{"x-goog-user-project": cfg.gcpQuotaProject}

	var exp exporters
	if own.tracer {
		exporter, err := otlptracehttp.New(ctx,
			otlptracehttp.WithHTTPClient(client),
			otlptracehttp.WithEndpointURL(gcpTracesEndpoint),
			otlptracehttp.WithHeaders(headers))
		if err != nil {
			return exporters{}, fmt.Errorf("failed to create GCP span exporter: %w", err)
		}
		exp.spanProcessors = append(exp.spanProcessors, sdktrace.NewBatchSpanProcessor(exporter))
	}
	if own.meter {
		exporter, err := otlpmetrichttp.New(ctx,
			otlpmetrichttp.WithHTTPClient(client),
			otlpmetrichttp.WithEndpointURL(gcpMetricsEndpoint),
			otlpmetrichttp.WithHeaders(headers))
		if err != nil {
			return exporters{}, fmt.Errorf("failed to create GCP metric exporter: %w", err)
		}
		exp.metricReaders = append(exp.metricReaders, newPeriodicReader(exporter, sdkmetric.WithInterval(gcpMetricExportInterval)))
	}
	// TODO log exporter
	return exp, nil
}

func initTracerProvider(cfg *config) *sdktrace.TracerProvider {
	if cfg.tracerProvider != nil {
		return cfg.tracerProvider
	}
	if len(cfg.spanProcessors) == 0 {
		return nil
	}
	opts := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(cfg.resource),
	}
	for _, p := range cfg.spanProcessors {
		opts = append(opts, sdktrace.WithSpanProcessor(p))
	}
	tp := sdktrace.NewTracerProvider(opts...)

	return tp
}

func initLoggerProvider(cfg *config) *sdklog.LoggerProvider {
	if cfg.loggerProvider != nil {
		return cfg.loggerProvider
	}
	if len(cfg.logProcessors) == 0 {
		return nil
	}
	opts := []sdklog.LoggerProviderOption{
		sdklog.WithResource(cfg.resource),
	}
	for _, p := range cfg.logProcessors {
		opts = append(opts, sdklog.WithProcessor(p))
	}
	lp := sdklog.NewLoggerProvider(opts...)

	return lp
}

func initMeterProvider(cfg *config) *sdkmetric.MeterProvider {
	if cfg.meterProvider != nil {
		return cfg.meterProvider
	}
	if len(cfg.metricReaders) == 0 {
		return nil
	}
	opts := []sdkmetric.Option{
		sdkmetric.WithResource(cfg.resource),
	}
	for _, r := range cfg.metricReaders {
		opts = append(opts, sdkmetric.WithReader(r))
	}
	mp := sdkmetric.NewMeterProvider(opts...)

	return mp
}
