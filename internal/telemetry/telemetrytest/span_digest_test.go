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

package telemetrytest

import (
	"testing"

	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestBuildDigest(t *testing.T) {
	traceID := trace.TraceID{1}
	span := func(name string, id, parent byte) tracetest.SpanStub {
		s := tracetest.SpanStub{
			Name:        name,
			SpanContext: trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: trace.SpanID{id}}),
		}
		if parent != 0 {
			s.Parent = trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: trace.SpanID{parent}})
		}
		return s
	}
	logIn := func(spanID byte) sdklog.Record {
		var r sdklog.Record
		r.SetEventName("event")
		r.SetSpanID(trace.SpanID{spanID})
		return r
	}

	tests := []struct {
		name    string
		spans   tracetest.SpanStubs
		logs    []sdklog.Record
		wantErr bool
	}{
		{
			name:  "log under a span",
			spans: tracetest.SpanStubs{span("root", 1, 0), span("child", 2, 1)},
			logs:  []sdklog.Record{logIn(2)},
		},
		{
			name:    "log outside any span",
			spans:   tracetest.SpanStubs{span("root", 1, 0)},
			logs:    []sdklog.Record{logIn(9)},
			wantErr: true,
		},
		{
			name:    "no root span",
			wantErr: true,
		},
		{
			name:    "two root spans",
			spans:   tracetest.SpanStubs{span("a", 1, 0), span("b", 2, 0)},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildDigest(tt.spans, tt.logs)
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Errorf("BuildDigest() error = %v, want error: %t", err, tt.wantErr)
			}
		})
	}
}
