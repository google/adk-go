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

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

func TestBuildMetricPoints(t *testing.T) {
	set := func(kv ...attribute.KeyValue) attribute.Set { return attribute.NewSet(kv...) }
	rm := metricdata.ResourceMetrics{ScopeMetrics: []metricdata.ScopeMetrics{{Metrics: []metricdata.Metrics{
		{Name: "tokens", Data: metricdata.Sum[int64]{DataPoints: []metricdata.DataPoint[int64]{
			{Attributes: set(attribute.String("m", "text")), Value: 7},
			{Attributes: set(attribute.String("m", "image")), Value: 3},
		}}},
		{Name: "per_call", Data: metricdata.Histogram[int64]{DataPoints: []metricdata.HistogramDataPoint[int64]{
			{Attributes: set(attribute.Bool("b", true)), Sum: 12},
		}}},
		{Name: "op.duration", Data: metricdata.Histogram[float64]{DataPoints: []metricdata.HistogramDataPoint[float64]{
			{Attributes: set(attribute.String("error.type", "429")), Sum: 0.25},
		}}},
		{Name: "time_to_first_chunk", Data: metricdata.Histogram[float64]{DataPoints: []metricdata.HistogramDataPoint[float64]{
			{Attributes: set(), Sum: 0.5},
		}}},
		{Name: "ratio", Data: metricdata.Sum[float64]{DataPoints: []metricdata.DataPoint[float64]{
			{Attributes: set(attribute.String("gcp.vertex.agent.invocation_id", "abc")), Value: 1.5},
		}}},
		{Name: "level", Data: metricdata.Gauge[int64]{DataPoints: []metricdata.DataPoint[int64]{
			{Attributes: set(attribute.String("m", "text")), Value: 4},
		}}},
	}}}}

	want := map[string][]MetricPoint{
		"tokens": {
			{Attributes: map[string]any{"m": "image"}, Value: int64(3)},
			{Attributes: map[string]any{"m": "text"}, Value: int64(7)},
		},
		"per_call":            {{Attributes: map[string]any{"b": true}, Value: int64(12)}},
		"op.duration":         {{Attributes: map[string]any{"error.type": "429"}, Value: present}},
		"time_to_first_chunk": {{Attributes: map[string]any{}, Value: present}},
		"ratio":               {{Attributes: map[string]any{"gcp.vertex.agent.invocation_id": present}, Value: 1.5}},
		"level":               {{Attributes: map[string]any{}, Value: present}},
	}
	if diff := cmp.Diff(want, buildMetricPoints(rm)); diff != "" {
		t.Errorf("buildMetricPoints() mismatch (-want +got):\n%s", diff)
	}
}

func TestBuildMetricPoints_NothingRecorded(t *testing.T) {
	got := buildMetricPoints(metricdata.ResourceMetrics{})
	if got == nil || len(got) != 0 {
		t.Errorf("buildMetricPoints() = %#v, want an empty map", got)
	}
}
