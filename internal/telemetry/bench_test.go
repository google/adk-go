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
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// BenchmarkStartSpan starts and ends one span the way every ADK span starts,
// tracer lookup included, from one goroutine and from GOMAXPROCS goroutines.
// "global" is the process default before an SDK is installed; "sdk" records
// every span, as a traced server does.
func BenchmarkStartSpan(b *testing.B) {
	for _, tc := range []struct {
		name string
		tp   trace.TracerProvider
	}{
		{"global", otel.GetTracerProvider()},
		{"sdk", sdktrace.NewTracerProvider()},
	} {
		b.Run(tc.name+"/serial", func(b *testing.B) {
			OverrideTracerForTesting(b, tc.tp)
			ctx := b.Context()
			b.ReportAllocs()
			for b.Loop() {
				_, span := tracer().Start(ctx, "generate_content stub")
				span.End()
			}
		})
		b.Run(tc.name+"/parallel", func(b *testing.B) {
			OverrideTracerForTesting(b, tc.tp)
			ctx := b.Context()
			b.ReportAllocs()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_, span := tracer().Start(ctx, "generate_content stub")
					span.End()
				}
			})
		})
	}
}
