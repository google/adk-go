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
		want    modalitySplit
	}{
		{"no details", 25, nil, modalitySplit{unknown: 25}},
		{"details sum to total", 20, details("TEXT", 12, "IMAGE", 8), modalitySplit{text: 12, image: 8}},
		{"remainder", 25, details("TEXT", 12, "AUDIO", 8), modalitySplit{text: 12, audio: 8, unknown: 5}},
		{"video and document", 30, details("VIDEO", 10, "DOCUMENT", 20), modalitySplit{video: 10, document: 20}},
		{"unspecified is unknown", 10, details("TEXT", 3, "TEXT", 2, "MODALITY_UNSPECIFIED", 5), modalitySplit{text: 5, unknown: 5}},
		{"unrecognized is unknown", 10, details("TEXT", 4, "HOLOGRAM", 6), modalitySplit{text: 4, unknown: 6}},
		{"lowercase modality", 10, details("text", 10), modalitySplit{text: 10}},
		{"zero modality", 8, details("TEXT", 8, "IMAGE", 0), modalitySplit{text: 8}},
		{"no modality skipped", 8, details("", 3, "TEXT", 5), modalitySplit{text: 5, unknown: 3}},
		{"details exceed total", 5, details("TEXT", 8), modalitySplit{text: 8}},
		{"zero total", 0, nil, modalitySplit{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, splitByModality(tc.total, tc.details), cmp.AllowUnexported(modalitySplit{})); diff != "" {
				t.Errorf("splitByModality() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
