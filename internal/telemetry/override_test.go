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
	"sync"
	"testing"

	"go.opentelemetry.io/otel/log"
	lognoop "go.opentelemetry.io/otel/log/noop"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// A goroutine an earlier test left running can start spans, emit logs and
// read the capture mode while a later test swaps the providers or calls
// ApplyEnv. Under -race this fails unless all three are synchronized.
func TestOverrideConcurrentWithUse(t *testing.T) {
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			_, span := StartTrace(context.Background(), "background")
			span.End()
			_ = otelLogger()
			_ = contentCapture()
		}
	})
	for range 100 {
		t.Run("override", func(t *testing.T) {
			OverrideTracerForTesting(t, noop.NewTracerProvider())
			OverrideLoggerForTesting(t, lognoop.NewLoggerProvider())
			ApplyEnv()
		})
	}
	close(stop)
	wg.Wait()
}

// Each override is undone when its test ends, restoring the provider before
// it, down to the default.
func TestOverrideRestoresPrevious(t *testing.T) {
	outerTP, innerTP := sdktrace.NewTracerProvider(), sdktrace.NewTracerProvider()
	outerLP, innerLP := sdklog.NewLoggerProvider(), sdklog.NewLoggerProvider()
	check := func(t *testing.T, when string, wantTP trace.TracerProvider, wantLP log.LoggerProvider) {
		t.Helper()
		if tracerProvider() != wantTP {
			t.Errorf("%s: tracerProvider() is not the expected provider", when)
		}
		if loggerProvider() != wantLP {
			t.Errorf("%s: loggerProvider() is not the expected provider", when)
		}
	}
	t.Run("outer", func(t *testing.T) {
		OverrideTracerForTesting(t, outerTP)
		OverrideLoggerForTesting(t, outerLP)
		t.Run("inner", func(t *testing.T) {
			OverrideTracerForTesting(t, innerTP)
			OverrideLoggerForTesting(t, innerLP)
			check(t, "inside the inner override", innerTP, innerLP)
		})
		check(t, "after the inner test", outerTP, outerLP)
	})
	check(t, "after the outer test", defaultTracerProvider, defaultLoggerProvider)
}
