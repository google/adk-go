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
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel/log"
	sdklog "go.opentelemetry.io/otel/sdk/log"
)

// LogDigest is a deterministic snapshot of one OTel log record:
// the event name, attributes, and the body decoded to a Go value.
type LogDigest struct {
	EventName  string
	Attributes map[string]any
	Body       any // JSON-normalized: int64 values are returned as float64
}

func buildLogDigest(r *sdklog.Record) *LogDigest {
	attrs := map[string]any{}
	r.WalkAttributes(func(kv log.KeyValue) bool {
		attrs[kv.Key] = logValueToAny(kv.Value)
		return true
	})
	return &LogDigest{
		EventName:  r.EventName(),
		Attributes: attrs,
		Body:       jsonRoundTrip(logValueToAny(r.Body())),
	}
}

// jsonRoundTrip marshals v to JSON and unmarshals the result back
// into a fresh any. This normalises:
//   - numeric types (int64, float64 → float64),
//   - string-valued bodies (returned as plain Go strings),
//   - structured bodies (returned as map[string]any / []any).
//
// On marshal/unmarshal failure the input value is returned
// unchanged; cmp.Diff will then surface the unconverted shape and
// the test author can fix the body rendering.
func jsonRoundTrip(v any) any {
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return v
	}
	return out
}

// InMemoryLogExporter is a minimal in-memory log record sink for tests.
type InMemoryLogExporter struct {
	mu      sync.Mutex
	records []sdklog.Record
}

// NewInMemoryLogExporter returns an empty exporter ready for use with sdklog.NewSimpleProcessor.
func NewInMemoryLogExporter() *InMemoryLogExporter { return &InMemoryLogExporter{} }

// Export implements sdklog.Exporter.
func (e *InMemoryLogExporter) Export(_ context.Context, records []sdklog.Record) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, r := range records {
		// Records are pooled by some processors; clone so the
		// stored copy is stable.
		e.records = append(e.records, r.Clone())
	}
	return nil
}

// Shutdown implements sdklog.Exporter.
func (e *InMemoryLogExporter) Shutdown(_ context.Context) error { return nil }

// ForceFlush implements sdklog.Exporter.
func (e *InMemoryLogExporter) ForceFlush(_ context.Context) error { return nil }

// Records returns a snapshot of the records collected so far, in emit order.
func (e *InMemoryLogExporter) Records() []sdklog.Record {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]sdklog.Record, len(e.records))
	copy(out, e.records)
	return out
}

// logValueToAny converts an OTel log.Value to a plain Go value
// (string, int64, float64, bool, []any, map[string]any).
func logValueToAny(v log.Value) any {
	switch v.Kind() {
	case log.KindBool:
		return v.AsBool()
	case log.KindFloat64:
		return v.AsFloat64()
	case log.KindInt64:
		return v.AsInt64()
	case log.KindString:
		return v.AsString()
	case log.KindBytes:
		return v.AsBytes()
	case log.KindSlice:
		s := v.AsSlice()
		out := make([]any, len(s))
		for i, e := range s {
			out[i] = logValueToAny(e)
		}
		return out
	case log.KindMap:
		m := v.AsMap()
		out := make(map[string]any, len(m))
		for _, kv := range m {
			out[kv.Key] = logValueToAny(kv.Value)
		}
		return out
	case log.KindEmpty:
		return nil
	default:
		return fmt.Sprintf("<unknown log.Kind %v>", v.Kind())
	}
}
