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

// Package telemetry contains OpenTelemetry related functionality for ADK.
package telemetry

import (
	"context"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	internal "google.golang.org/adk/internal/telemetry"
)

// Service wraps all telemetry providers and implements functions for telemetry lifecycle management.
type Service interface {
	// SetGlobalOtelProviders registers the configured providers as the global OTel providers.
	SetGlobalOtelProviders()

	// TracerProvider returns the configured TracerProvider or nil.
	TracerProvider() *sdktrace.TracerProvider

	// Shutdown shuts down underlying OTel providers.
	Shutdown(ctx context.Context) error
}

// New initializes a new telemetry service and underlying providers: TraceProvider, LogProvider, and MeterProvider.
// Options can be used to customize the defaults, e.g. use custom credentials, add SpanProcessors, or use preconfigured TraceProvider.
// Telemetry providers have to be registered in the global OTel providers either manually or via [SetGlobalOtelProviders].
//
// # Usage
//
//	 import (
//		"context"
//		"log"
//		"time"
//
//		"go.opentelemetry.io/otel/sdk/resource"
//		semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
//		"google.golang.org/adk/telemetry"
//	 )
//
//	 func main() {
//			ctx := context.Background()
//			res, err := resource.New(ctx,
//				resource.WithAttributes(
//					semconv.ServiceNameKey.String("my-service"),
//					semconv.ServiceVersionKey.String("1.0.0"),
//				),
//			)
//			if err != nil {
//				log.Fatalf("failed to create resource: %v", err)
//			}
//
//			telemetryService, err := telemetry.New(ctx,
//				telemetry.WithOtelToCloud(true),
//				telemetry.WithResource(res),
//			)
//			if err != nil {
//				log.Fatal(err)
//			}
//			defer func() {
//				shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
//				defer cancel()
//				if err := telemetryService.Shutdown(shutdownCtx); err != nil {
//					log.Printf("telemetry shutdown failed: %v", err)
//				}
//			}()
//			telemetryService.SetGlobalOtelProviders()
//
//			tp := telemetryService.TracerProvider()
//			instrumentedlib.SetTracerProvider(tp) // Set TracerProvider manually if your lib doesn't use the global provider.
//
//			// app code
//		}
//
// The caller must call [Shutdown] method to gracefully shut down the underlying telemetry and release resources.
func New(ctx context.Context, opts ...Option) (Service, error) {
	cfg, err := configure(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return newInternal(cfg)
}

// RegisterLocalSpanProcessor registers the span processor to local trace provider instance.
// Any processor should be registered BEFORE any of the events are emitted, otherwise
// the registration will be ignored.
// In addition to the RegisterLocalSpanProcessor function, global trace provider configs
// are respected.
//
// Deprecated: Configure processors via [Option]s passed to [New]. TODO(#479) remove this together with local tracer provider.
func RegisterLocalSpanProcessor(processor sdktrace.SpanProcessor) {
	internal.AddSpanProcessor(processor)
}
