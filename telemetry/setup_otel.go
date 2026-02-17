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
	"fmt"
	"os"

	"go.opentelemetry.io/contrib/detectors/gcp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

func configure(ctx context.Context, opts ...Option) (*config, error) {
	cfg := &config{}

	for _, opt := range opts {
		if err := opt.apply(cfg); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	var err error
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

	spanProcessors, err := configureExporters(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to configure exporters: %w", err)
	}
	cfg.spanProcessors = append(cfg.spanProcessors, spanProcessors...)

	return cfg, nil
}

func newInternal(cfg *config) (*Providers, error) {
	tp, err := initTracerProvider(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tracer provider: %w", err)
	}

	lp := initLoggerProvider(cfg)
	// TODO(#479) init meter provider

	return &Providers{
		TracerProvider: tp,
		LoggerProvider: lp,
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
	if configuredProject != "" {
		return configuredProject, nil
	}
	if creds != nil && creds.ProjectID != "" {
		return creds.ProjectID, nil
	}
	// The project was always empty during testing, even though it was set in ADC JSON file.
	// Using fallback to env variable to resolve the project as a workaround.
	project, ok := os.LookupEnv("GOOGLE_CLOUD_PROJECT")
	if !ok && requireProject {
		return "", fmt.Errorf("telemetry.googleapis.com requires setting the %s project. Refer to telemetry.config for the available options to set the %s project", projectType, projectType)
	}
	return project, nil
}

// resolveResource creates a new resource with attributes specified in the following order (later attributes override earlier ones):
//  1. [resource.Default()] populates the resource labels from environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
//  2. GCP related attributes when otelToCloud is enabled or gcpResourceProject is set - `gcp.project_id` attribute needed by Cloud Trace and other attributes provided by [gcp.NewDetector()].
//  3. GCP detector adds runtime attributes if ADK runs on one of supported platforms (e.g. GCE, GKE, CloudRun).
//  4. Resource from config, if present.
func resolveResource(ctx context.Context, cfg *config) (*resource.Resource, error) {
	r := resource.Default()

	opts := []resource.Option{
		resource.WithAttributes(
			attribute.Key("gcp.project_id").String(cfg.gcpResourceProject),
		),
	}
	if cfg.oTelToCloud {
		// Add GCP specific detectors.
		opts = append(opts, resource.WithDetectors(gcp.NewDetector()))
	}
	var err error
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

// configureExporters initializes OTel exporters from environment variables and otelToCloud.
func configureExporters(ctx context.Context, cfg *config) ([]sdktrace.SpanProcessor, error) {
	var spanProcessors []sdktrace.SpanProcessor

	_, otelEndpointExists := os.LookupEnv("OTEL_EXPORTER_OTLP_ENDPOINT")
	_, otelTracesEndpointExists := os.LookupEnv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
	if otelEndpointExists || otelTracesEndpointExists {
		exporter, err := otlptracehttp.New(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to create OTLP HTTP exporter: %w", err)
		}
		spanProcessors = append(spanProcessors, sdktrace.NewBatchSpanProcessor(
			exporter,
		))
	}
	if cfg.oTelToCloud {
		spanExporter, err := newGcpSpanExporter(ctx, cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to create GCP span exporter: %w", err)
		}
		spanProcessors = append(spanProcessors, sdktrace.NewBatchSpanProcessor(spanExporter))
	}
	return spanProcessors, nil
}

func initTracerProvider(cfg *config) (*sdktrace.TracerProvider, error) {
	if cfg.tracerProvider != nil {
		return cfg.tracerProvider, nil
	}
	if len(cfg.spanProcessors) == 0 {
		return nil, nil
	}
	opts := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(cfg.resource),
	}
	for _, p := range cfg.spanProcessors {
		opts = append(opts, sdktrace.WithSpanProcessor(p))
	}
	tp := sdktrace.NewTracerProvider(opts...)

	return tp, nil
}

// TODO(#479) finish the implementation and add the default exporter if env vars are set.
func initLoggerProvider(cfg *config) *sdklog.LoggerProvider {
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

func newGcpSpanExporter(ctx context.Context, cfg *config) (sdktrace.SpanExporter, error) {
	client := oauth2.NewClient(ctx, cfg.googleCredentials.TokenSource)
	return otlptracehttp.New(ctx,
		otlptracehttp.WithHTTPClient(client),
		otlptracehttp.WithEndpointURL("https://telemetry.googleapis.com/v1/traces"),
		// Pass the quota project id in headers to fix auth errors.
		// https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds
		otlptracehttp.WithHeaders(map[string]string{
			"x-goog-user-project": cfg.gcpQuotaProject,
		}))
}
