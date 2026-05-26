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
	"sort"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// PRESENT is the sentinel used to mark attributes whose value is
// non-deterministic across runs but whose presence must still be
// asserted (e.g. invocation_id, session_id).
const PRESENT = "<PRESENT>"

// nonDeterministicSpanAttributes are span attributes whose VALUE
// varies run-to-run. [BuildDigests] replaces these with [PRESENT]
// so the comparison shape is stable while still asserting the key
// is set.
var nonDeterministicSpanAttributes = map[string]bool{
	"gcp.vertex.agent.event_id":       true,
	"gen_ai.tool.call.id":             true,
	"gen_ai.conversation.id":          true,
	"gcp.vertex.agent.invocation_id":  true,
	"gcp.vertex.agent.session_id":     true,
	"gcp.vertex.agent.tool_call_args": true, // contains generated arg ids
	"gcp.vertex.agent.tool_response":  true, // contains generated event ids
}

// SpanDigest is a deterministic snapshot of a span: name, normalised
// attributes, child spans, and the log records emitted while the
// span was active.
type SpanDigest struct {
	Name       string
	Attributes map[string]any
	Children   []*SpanDigest
	Logs       []*LogDigest
}

type spanWithMeta struct {
	digest    *SpanDigest
	spanID    trace.SpanID
	parentID  trace.SpanID
	startTime int64
}

// BuildDigests collects the spans and log records into a single
// tree, attaching each log to the SpanDigest of the span
// it was emitted under. The scenario MUST produce exactly one root span.
func BuildDigests(t *testing.T, spans tracetest.SpanStubs, logs []sdklog.Record) *SpanDigest {
	t.Helper()
	digests := make([]*spanWithMeta, 0, len(spans))
	for _, s := range spans {
		digests = append(digests, buildSpanDigest(s))
	}
	digests = attachLogs(digests, logs)
	roots := linkAndSort(digests)
	if len(roots) != 1 {
		t.Fatalf("expected exactly 1 root span, got %d", len(roots))
	}
	return roots[0]
}

func buildSpanDigest(s tracetest.SpanStub) *spanWithMeta {
	return &spanWithMeta{
		digest: &SpanDigest{
			Name:       s.Name,
			Attributes: normaliseSpanAttributes(s.Attributes),
		},
		spanID:    s.SpanContext.SpanID(),
		parentID:  s.Parent.SpanID(),
		startTime: s.StartTime.UnixNano(),
	}
}

// attachLogs populates the Logs slice of each digest whose SpanID
// matches a log record's SpanID. Log iteration order is emit
// order, so the resulting per-span Logs slices end up
// chronological. Logs not associated with any collected span
// (e.g. emitted at module init) are dropped silently.
func attachLogs(digests []*spanWithMeta, logs []sdklog.Record) []*spanWithMeta {
	bySpanID := make(map[trace.SpanID]*SpanDigest, len(digests))
	for _, d := range digests {
		bySpanID[d.spanID] = d.digest
	}
	for _, r := range logs {
		if d, ok := bySpanID[r.SpanID()]; ok {
			d.Logs = append(d.Logs, buildLogDigest(&r))
		}
	}
	return digests
}

// linkAndSort assembles the parent→children adjacency, recursively
// sorts each level by start time (name as tiebreaker), assigns the
// sorted children to the SpanDigest.Children slices, and returns
// the roots (spans whose parent was not collected).
func linkAndSort(digests []*spanWithMeta) []*SpanDigest {
	bySpanID := make(map[trace.SpanID]*spanWithMeta, len(digests))
	for _, sw := range digests {
		bySpanID[sw.spanID] = sw
	}
	childrenOf := map[*spanWithMeta][]*spanWithMeta{}
	var roots []*spanWithMeta
	for _, sw := range digests {
		if parent, ok := bySpanID[sw.parentID]; ok {
			childrenOf[parent] = append(childrenOf[parent], sw)
		} else {
			roots = append(roots, sw)
		}
	}
	less := func(a, b *spanWithMeta) bool {
		if a.startTime != b.startTime {
			return a.startTime < b.startTime
		}
		return a.digest.Name < b.digest.Name
	}
	var assignSorted func(sw *spanWithMeta)
	assignSorted = func(sw *spanWithMeta) {
		children := childrenOf[sw]
		if len(children) == 0 {
			return
		}
		sort.SliceStable(children, func(i, j int) bool { return less(children[i], children[j]) })
		sw.digest.Children = make([]*SpanDigest, len(children))
		for i, c := range children {
			sw.digest.Children[i] = c.digest
			assignSorted(c)
		}
	}
	sort.SliceStable(roots, func(i, j int) bool { return less(roots[i], roots[j]) })
	out := make([]*SpanDigest, len(roots))
	for i, r := range roots {
		assignSorted(r)
		out[i] = r.digest
	}
	return out
}

// normaliseSpanAttributes converts the OTel attribute slice into a
// map[string]any with non-deterministic values collapsed to
// [PRESENT].
func normaliseSpanAttributes(attrs []attribute.KeyValue) map[string]any {
	out := make(map[string]any, len(attrs))
	for _, kv := range attrs {
		key := string(kv.Key)
		if nonDeterministicSpanAttributes[key] {
			out[key] = PRESENT
			continue
		}
		out[key] = kv.Value.AsInterface()
	}
	return out
}
