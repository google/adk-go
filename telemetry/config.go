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
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"golang.org/x/oauth2/google"
)

type config struct {
	// Enables/disables telemetry export to GCP.
	oTelToCloud bool

	// gcpResourceProject is used as the gcp.project.id resource attribute.
	// If it's empty, the project will be read from ADC or GOOGLE_CLOUD_PROJECT env variable.
	gcpResourceProject string

	// gcpQuotaProject is used as the quota project for the telemetry export.
	// If it's empty, the project will be read from ADC or GOOGLE_CLOUD_PROJECT env variable.
	gcpQuotaProject string

	// googleCredentials override the application default credentials.
	googleCredentials *google.Credentials

	// resource customizes the OTel resource. It will be merged with default detectors.
	resource *resource.Resource
	// spanProcessors registers additional span processors, e.g. for custom span exporters.
	spanProcessors []sdktrace.SpanProcessor

	// logProcessors registers additional log processors, e.g. for custom log exporters.
	logProcessors []sdklog.Processor

	// tracerProvider overrides the default TracerProvider.
	tracerProvider *sdktrace.TracerProvider
}

// Option configures adk telemetry.
type Option interface {
	apply(*config) error
}

type optionFunc func(*config) error

func (fn optionFunc) apply(cfg *config) error {
	return fn(cfg)
}

// WithOtelToCloud enables/disables exporting telemetry to GCP.
func WithOtelToCloud(value bool) Option {
	return optionFunc(func(cfg *config) error {
		cfg.oTelToCloud = value
		return nil
	})
}

// WithGcpResourceProject sets the gcp.project.id resource attribute.
func WithGcpResourceProject(project string) Option {
	return optionFunc(func(cfg *config) error {
		cfg.gcpResourceProject = project
		return nil
	})
}

// WithGcpQuotaProject sets the quota project for the telemetry export.
func WithGcpQuotaProject(projectID string) Option {
	return optionFunc(func(cfg *config) error {
		cfg.gcpQuotaProject = projectID
		return nil
	})
}

// WithResource configures the OTel resource.
func WithResource(r *resource.Resource) Option {
	return optionFunc(func(cfg *config) error {
		cfg.resource = r
		return nil
	})
}

// WithGoogleCredentials overrides the application default credentials.
func WithGoogleCredentials(c *google.Credentials) Option {
	return optionFunc(func(cfg *config) error {
		cfg.googleCredentials = c
		return nil
	})
}

// WithSpanProcessors registers additional span processors.
func WithSpanProcessors(p ...sdktrace.SpanProcessor) Option {
	return optionFunc(func(cfg *config) error {
		cfg.spanProcessors = append(cfg.spanProcessors, p...)
		return nil
	})
}

// WithLogRecordProcessors registers additional log processors.
func WithLogRecordProcessors(p ...sdklog.Processor) Option {
	return optionFunc(func(cfg *config) error {
		cfg.logProcessors = append(cfg.logProcessors, p...)
		return nil
	})
}

// WithTracerProvider overrides the default TracerProvider with preconfigured instance.
func WithTracerProvider(tp *sdktrace.TracerProvider) Option {
	return optionFunc(func(cfg *config) error {
		cfg.tracerProvider = tp
		return nil
	})
}
