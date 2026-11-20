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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"google.golang.org/adk/v2/internal/telemetry"
)

// MetricPoint is a deterministic snapshot of one metric data point.
type MetricPoint struct {
	Attributes map[string]any `json:"attributes"`
	// Value is a counter's total or a histogram's sum. It is "PRESENT" for a
	// timing, whose value changes run to run, and for a kind of aggregation
	// the digest does not read.
	Value any `json:"value"`
}

// buildMetricPoints groups collected data points by metric name, ordered by
// the JSON encoding of their attributes, as adk-python's
// tests/unittests/telemetry/functional/_digests.py does. It reads the values
// and attributes of sums and histograms, the kinds ADK records. Any other kind
// of aggregation shows up as one point without attributes, valued "PRESENT",
// so that a golden still records it. The result is empty, not nil, when
// nothing was recorded.
func buildMetricPoints(rm metricdata.ResourceMetrics) map[string][]MetricPoint {
	grouped := map[string][]MetricPoint{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			// Durations and chunk timings measure the wall clock.
			timing := strings.HasSuffix(m.Name, ".duration") || strings.HasSuffix(m.Name, "_chunk")
			for _, p := range dataPoints(m.Data) {
				if timing {
					p.Value = present
				}
				grouped[m.Name] = append(grouped[m.Name], p)
			}
		}
	}
	for _, points := range grouped {
		slices.SortFunc(points, func(a, b MetricPoint) int {
			return cmp.Compare(canonicalJSON(a.Attributes), canonicalJSON(b.Attributes))
		})
	}
	return grouped
}

func dataPoints(data metricdata.Aggregation) []MetricPoint {
	var out []MetricPoint
	switch d := data.(type) {
	case metricdata.Sum[int64]:
		for _, p := range d.DataPoints {
			out = append(out, MetricPoint{normalizeMetricAttributes(p.Attributes), p.Value})
		}
	case metricdata.Sum[float64]:
		for _, p := range d.DataPoints {
			out = append(out, MetricPoint{normalizeMetricAttributes(p.Attributes), p.Value})
		}
	case metricdata.Histogram[int64]:
		for _, p := range d.DataPoints {
			out = append(out, MetricPoint{normalizeMetricAttributes(p.Attributes), p.Sum})
		}
	case metricdata.Histogram[float64]:
		for _, p := range d.DataPoints {
			out = append(out, MetricPoint{normalizeMetricAttributes(p.Attributes), p.Sum})
		}
	default:
		out = append(out, MetricPoint{Attributes: map[string]any{}, Value: present})
	}
	return out
}

func normalizeMetricAttributes(set attribute.Set) map[string]any {
	out := make(map[string]any, set.Len())
	for _, kv := range set.ToSlice() {
		out[string(kv.Key)] = normalizeAttribute(string(kv.Key), telemetry.FromLogValue(kv.Value))
	}
	return out
}

// canonicalJSON encodes a value with map keys sorted, as encoding/json does.
func canonicalJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		// Attribute values are strings, numbers, booleans and slices of them,
		// which always encode.
		panic(err)
	}
	return string(b)
}
