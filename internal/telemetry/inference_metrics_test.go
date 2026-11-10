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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

// Cases from https://github.com/google/adk-python/blob/16898f30c9a5c6b2ae1ecf099424afff8fd28e77/tests/unittests/telemetry/test_metrics.py#L613-L653
// plus corner cases it lacks. Its zero-token case expects zero under unknown,
// which the counters add for an empty split.
func TestSplitByModality(t *testing.T) {
	details := func(counts ...any) []*genai.ModalityTokenCount {
		var out []*genai.ModalityTokenCount
		for i := 0; i < len(counts); i += 2 {
			out = append(out, &genai.ModalityTokenCount{Modality: genai.MediaModality(counts[i].(string)), TokenCount: int32(counts[i+1].(int))})
		}
		return out
	}
	for _, tc := range []struct {
		name    string
		total   int64
		details []*genai.ModalityTokenCount
		want    map[string]int64
	}{
		{"no details", 25, nil, map[string]int64{"unknown": 25}},
		{"details sum to total", 20, details("TEXT", 12, "IMAGE", 8), map[string]int64{"text": 12, "image": 8}},
		{"remainder", 25, details("TEXT", 12, "AUDIO", 8), map[string]int64{"text": 12, "audio": 8, "unknown": 5}},
		{"video and document", 30, details("VIDEO", 10, "DOCUMENT", 20), map[string]int64{"video": 10, "document": 20}},
		{"unspecified is text", 10, details("TEXT", 3, "TEXT", 2, "MODALITY_UNSPECIFIED", 5), map[string]int64{"text": 10}},
		{"zero modality dropped", 8, details("TEXT", 8, "IMAGE", 0), map[string]int64{"text": 8}},
		{"no modality skipped", 8, details("", 3, "TEXT", 5), map[string]int64{"text": 5, "unknown": 3}},
		{"details exceed total", 5, details("TEXT", 8), map[string]int64{"text": 8}},
		{"zero total", 0, nil, map[string]int64{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, splitByModality(tc.total, tc.details)); diff != "" {
				t.Errorf("splitByModality() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
