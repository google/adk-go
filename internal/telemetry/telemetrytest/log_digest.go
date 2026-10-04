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
	"sync"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"

	"google.golang.org/adk/v2/internal/telemetry"
)

// LogDigest is a deterministic snapshot of one log record.
type LogDigest struct {
	EventName  string         `json:"event_name"`
	Body       any            `json:"body"`
	Attributes map[string]any `json:"attributes"`
}

func buildLogDigest(r *sdklog.Record) *LogDigest {
	attrs := map[string]any{}
	r.WalkAttributes(func(kv attribute.KeyValue) bool {
		attrs[string(kv.Key)] = normalizeAttribute(string(kv.Key), telemetry.FromLogValue(kv.Value))
		return true
	})
	return &LogDigest{
		EventName:  r.EventName(),
		Body:       telemetry.FromLogValue(r.Body()),
		Attributes: attrs,
	}
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
