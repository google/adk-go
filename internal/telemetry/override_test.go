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
	metricnoop "go.opentelemetry.io/otel/metric/noop"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// A goroutine an earlier test left running can start spans, emit logs, record
// metrics and read the capture mode while a later test swaps the providers or
// calls ApplyEnv. Under -race this fails unless all four are synchronized.
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
			inst().inferenceDuration.Record(context.Background(), 1)
			_ = contentCapture()
		}
	})
	for range 100 {
		t.Run("override", func(t *testing.T) {
			OverrideTracerForTesting(t, noop.NewTracerProvider())
			OverrideLoggerForTesting(t, lognoop.NewLoggerProvider())
			OverrideMeterForTesting(t, metricnoop.NewMeterProvider())
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
	outerMP, innerMP := sdkmetric.NewMeterProvider(), sdkmetric.NewMeterProvider()
	var outerInst, innerInst *instruments
	check := func(t *testing.T, when string, wantTP trace.TracerProvider, wantLP log.LoggerProvider, wantInst *instruments) {
		t.Helper()
		if tracerProvider() != wantTP {
			t.Errorf("%s: tracerProvider() is not the expected provider", when)
		}
		if loggerProvider() != wantLP {
			t.Errorf("%s: loggerProvider() is not the expected provider", when)
		}
		if inst() != wantInst {
			t.Errorf("%s: inst() is not the expected instrument set", when)
		}
	}
	t.Run("outer", func(t *testing.T) {
		OverrideTracerForTesting(t, outerTP)
		OverrideLoggerForTesting(t, outerLP)
		OverrideMeterForTesting(t, outerMP)
		outerInst = inst()
		t.Run("inner", func(t *testing.T) {
			OverrideTracerForTesting(t, innerTP)
			OverrideLoggerForTesting(t, innerLP)
			OverrideMeterForTesting(t, innerMP)
			innerInst = inst()
			check(t, "inside the inner override", innerTP, innerLP, innerInst)
		})
		check(t, "after the inner test", outerTP, outerLP, outerInst)
	})
	check(t, "after the outer test", defaultTracerProvider, defaultLoggerProvider, &defaultInstruments)
}
