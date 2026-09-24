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
	"cmp"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/internal/telemetry"
)

// present stands in for a value that cannot be pinned, such as a generated id,
// so a golden asserts only that it is there. Same literal as adk-python's
// tests/unittests/telemetry/functional/_digests.py.
const present = "PRESENT"

// nonDeterministicAttributes are attributes whose value changes run to run.
var nonDeterministicAttributes = map[string]bool{
	"gcp.vertex.agent.event_id":      true,
	"gen_ai.tool.call.id":            true,
	"gen_ai.conversation.id":         true,
	"gcp.vertex.agent.invocation_id": true,
}

// jsonStringAttributes are span attributes holding a JSON string. A golden
// shows one parsed, wrapped as {jsonStringKey: value}. adk-python's
// _digests.py parses them in place, but Go can also record these attributes as
// structured values, and the wrapper keeps the two encodings apart.
var jsonStringAttributes = map[string]bool{
	"gen_ai.input.messages":           true,
	"gen_ai.output.messages":          true,
	"gen_ai.system_instructions":      true,
	"gcp.vertex.agent.tool_call_args": true,
	"gcp.vertex.agent.tool_response":  true,
}

const jsonStringKey = "JSON_STRING"

// Digest is what a golden compares of one scenario run: the span tree, with
// each log record attached to the span it was emitted under. Of a span it
// keeps the name, attributes and status code; of a log record, the event name,
// body and attributes.
type Digest struct {
	RootSpan *SpanDigest `json:"root_span"`
}

// SpanDigest is a deterministic snapshot of one span.
type SpanDigest struct {
	Name       string         `json:"name"`
	Attributes map[string]any `json:"attributes"`
	// Status is "UNSET", "ERROR" or "OK".
	Status   string        `json:"status"`
	Children []*SpanDigest `json:"children"`
	Logs     []*LogDigest  `json:"logs"`
}

type spanWithMeta struct {
	digest    *SpanDigest
	parentID  trace.SpanID
	startTime int64
}

// BuildDigest collects spans and log records into one tree. The scenario must
// produce exactly one root span, and emit every log record under a span.
func BuildDigest(t *testing.T, spans tracetest.SpanStubs, logs []sdklog.Record) *Digest {
	t.Helper()
	ordered := make([]*spanWithMeta, 0, len(spans))
	bySpanID := make(map[trace.SpanID]*spanWithMeta, len(spans))
	for _, s := range spans {
		d := &spanWithMeta{
			digest: &SpanDigest{
				Name:       s.Name,
				Attributes: normalizeSpanAttributes(s.Attributes),
				Status:     strings.ToUpper(s.Status.Code.String()),
				Children:   []*SpanDigest{},
				Logs:       []*LogDigest{},
			},
			parentID:  s.Parent.SpanID(),
			startTime: s.StartTime.UnixNano(),
		}
		ordered = append(ordered, d)
		bySpanID[s.SpanContext.SpanID()] = d
	}
	for _, r := range logs {
		s, ok := bySpanID[r.SpanID()]
		if !ok {
			t.Fatalf("log record %q was emitted outside any collected span", r.EventName())
		}
		s.digest.Logs = append(s.digest.Logs, buildLogDigest(&r))
	}
	roots := linkAndSort(ordered, bySpanID)
	if len(roots) != 1 {
		t.Fatalf("expected exactly 1 root span, got %d", len(roots))
	}
	return &Digest{RootSpan: roots[0]}
}

// linkAndSort links each span to its parent, orders siblings by start time
// (name as tiebreaker), and returns the spans whose parent was not collected.
func linkAndSort(ordered []*spanWithMeta, bySpanID map[trace.SpanID]*spanWithMeta) []*SpanDigest {
	slices.SortStableFunc(ordered, func(a, b *spanWithMeta) int {
		return cmp.Or(cmp.Compare(a.startTime, b.startTime), cmp.Compare(a.digest.Name, b.digest.Name))
	})
	var roots []*SpanDigest
	for _, s := range ordered {
		if parent, ok := bySpanID[s.parentID]; ok {
			parent.digest.Children = append(parent.digest.Children, s.digest)
		} else {
			roots = append(roots, s.digest)
		}
	}
	return roots
}

func normalizeSpanAttributes(attrs []attribute.KeyValue) map[string]any {
	out := make(map[string]any, len(attrs))
	for _, kv := range attrs {
		key := string(kv.Key)
		value := telemetry.FromLogValue(kv.Value)
		if s, ok := value.(string); ok && jsonStringAttributes[key] {
			// Not every value is JSON: a legacy tool response can be
			// "<not specified>", which is kept as it is.
			var parsed any
			if err := json.Unmarshal([]byte(s), &parsed); err == nil {
				value = map[string]any{jsonStringKey: parsed}
			}
		}
		out[key] = normalizeAttribute(key, value)
	}
	return out
}

func normalizeAttribute(key string, value any) any {
	if nonDeterministicAttributes[key] {
		return present
	}
	return value
}
