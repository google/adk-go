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

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	internal "google.golang.org/adk/internal/telemetry"
)

// Providers wraps all telemetry providers and provides [Shutdown] function.
type Providers struct {
	// TracerProvider is the configured TracerProvider or nil.
	TracerProvider *sdktrace.TracerProvider
}

// Shutdown shuts down underlying OTel providers.
func (t *Providers) Shutdown(ctx context.Context) error {
	if t.TracerProvider != nil {
		return t.TracerProvider.Shutdown(ctx)
	}
	return nil
}

// SetGlobalOtelProviders registers the configured providers as the global OTel providers.
func (t *Providers) SetGlobalOtelProviders() {
	if t.TracerProvider != nil {
		otel.SetTracerProvider(t.TracerProvider)
	}
}

// New initializes telemetry providers: TraceProvider, LogProvider, and MeterProvider.
// Options can be used to customize the defaults, e.g. use custom credentials, add SpanProcessors, or use preconfigured TraceProvider.
// Telemetry providers have to be registered in the global OTel providers either manually or via [Providers.SetGlobalOtelProviders].
// If your library doesn't use the global providers, you can use the providers directly and pass them to the instrumented libraries.
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
//			telemetryProviders, err := telemetry.New(ctx,
//				telemetry.WithOtelToCloud(true),
//				telemetry.WithResource(res),
//			)
//			if err != nil {
//				log.Fatal(err)
//			}
//			defer func() {
//				shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
//				defer cancel()
//				if err := telemetryProviders.Shutdown(shutdownCtx); err != nil {
//					log.Printf("telemetry shutdown failed: %v", err)
//				}
//			}()
//			telemetryProviders.SetGlobalOtelProviders()
//
//			tp := telemetryProviders.TracerProvider
//			instrumentedlib.SetTracerProvider(tp) // Set TracerProvider manually if your lib doesn't use the global provider.
//
//			// app code
//		}
//
// The caller must call [Providers.Shutdown] method to gracefully shut down the underlying telemetry and release resources.
func New(ctx context.Context, opts ...Option) (*Providers, error) {
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
