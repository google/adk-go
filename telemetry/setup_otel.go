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

// Package telemetry implements the open telemetry in ADK.
package telemetry

import (
	"context"
	"fmt"
	"log"
	"os"

	"go.opentelemetry.io/contrib/detectors/gcp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

func newInternal(cfg *config) (*telemetryService, error) {
	tp, err := initTracerProvider(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tracer provider: %w", err)
	}
	// TODO(#479) init logger provider
	// TODO(#479) init meter provider

	return &telemetryService{
		tp: tp,
	}, nil
}

type telemetryService struct {
	tp *sdktrace.TracerProvider
}

func (t *telemetryService) TraceProvider() *sdktrace.TracerProvider {
	return t.tp
}

func (t *telemetryService) Shutdown(ctx context.Context) error {
	if t.tp != nil {
		return t.tp.Shutdown(ctx)
	}
	return nil
}

func (t *telemetryService) SetGlobalOtelProviders() {
	if t.tp != nil {
		otel.SetTracerProvider(t.tp)
	}
}

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
			cfg.googleCredentials, err = applicationDefaultCredentials(ctx)
			if err != nil {
				return nil, fmt.Errorf("failed to get application default credentials: %w", err)
			}
		}
	}

	if err := cfg.resolveQuotaProject(); err != nil {
		return nil, err
	}

	if err := cfg.resolveResourceProject(); err != nil {
		return nil, err
	}

	cfg.resource, err = resolveResource(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve resource: %w", err)
	}

	spanProcessors, err := configureProcessors(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to configure processors: %w", err)
	}
	cfg.spanProcessors = append(cfg.spanProcessors, spanProcessors...)

	return cfg, nil
}

// resolveQuotaProject determines the quota project for telemetry export in the following order:
// 1. Quota project from config, if present.
// 2. Project ID from credentials, if present.
// 3. GOOGLE_CLOUD_PROJECT environment variable.
// Returns an error if the quota project cannot be determined.
func (c *config) resolveQuotaProject() error {
	if c.quotaProject != "" {
		return nil
	}
	if c.googleCredentials != nil && c.googleCredentials.ProjectID != "" {
		c.quotaProject = c.googleCredentials.ProjectID
	} else {
		// The quota project wasn't set in credentials during testing, even when it's set in ADC JSON file.
		// Using fallback to env variable to resolve the quota project as a workaround.
		projectID, ok := os.LookupEnv("GOOGLE_CLOUD_PROJECT")
		if !ok {
			log.Println("telemetry.googleapis.com requires setting the quota project. Refer to telemetry.config for the available options to set the quota project")
			return nil
		}
		c.quotaProject = projectID
	}
	return nil
}

// resolveResourceProject determines the resource project for telemetry export in the following order:
// 1. Resource project from config, if present.
// 2. Project ID from credentials, if present.
// 3. GOOGLE_CLOUD_PROJECT environment variable.
// Returns an error if the resource project cannot be determined.
func (c *config) resolveResourceProject() error {
	if c.resourceProject != "" {
		return nil
	}
	if c.googleCredentials != nil && c.googleCredentials.ProjectID != "" {
		c.resourceProject = c.googleCredentials.ProjectID
	} else {
		// The resource project wasn't set in credentials during testing, even when it's set in ADC JSON file.
		// Using fallback to env variable to resolve the resource project as a workaround.
		projectID, ok := os.LookupEnv("GOOGLE_CLOUD_PROJECT")
		if !ok {
			log.Println("telemetry.googleapis.com requires setting the resource project. Refer to telemetry.config for the available options to set the resource project")
			return nil
		}
		c.resourceProject = projectID
	}
	return nil
}

// resolveResource creates a new resource with attributes specified in the following order (later attributes override earlier ones):
//  1. [resource.Default()] populates the resource labels from environment variables like OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
//  2. [projectIDDetector](.detector.go) populates `gcp.project_id` attribute needed by Cloud Trace.
//  3. GCP detector adds runtime attributes if ADK runs on one of supported platforms (e.g. GCE, GKE, CloudRun).
//  4. Resource from config, if present.
func resolveResource(ctx context.Context, cfg *config) (*resource.Resource, error) {
	// Add GCP specific detectors.
	gcpResource, err := resource.New(
		ctx,
		resource.WithDetectors(gcp.NewDetector()),
		resource.WithAttributes(
			attribute.Key("gcp.project_id").String(cfg.resourceProject),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCP resource: %w", err)
	}

	// Merge with the default resource.
	merged, err := resource.Merge(resource.Default(), gcpResource)
	if err != nil {
		return nil, fmt.Errorf("failed to merge default and GCP resources: %w", err)
	}
	// Lastly, merge with the resource from config.
	if cfg.resource != nil {
		merged, err = resource.Merge(merged, cfg.resource)
		if err != nil {
			return nil, fmt.Errorf("failed to merge with config resource: %w", err)
		}
	}
	return merged, nil
}

// configureProcessors initializes OTel exporters from environment variables
func configureProcessors(ctx context.Context, cfg *config) ([]sdktrace.SpanProcessor, error) {
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

func applicationDefaultCredentials(ctx context.Context) (*google.Credentials, error) {
	adc, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("failed to find default credentials: %w", err)
	}
	return adc, nil
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

func newGcpSpanExporter(ctx context.Context, cfg *config) (sdktrace.SpanExporter, error) {
	client := oauth2.NewClient(ctx, cfg.googleCredentials.TokenSource)
	return otlptracehttp.New(ctx,
		otlptracehttp.WithHTTPClient(client),
		otlptracehttp.WithEndpointURL("https://telemetry.googleapis.com/v1/traces"),
		// Pass the quota project id in headers to fix auth errors.
		// https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds
		otlptracehttp.WithHeaders(map[string]string{
			"x-goog-user-project": cfg.quotaProject,
		}))
}
