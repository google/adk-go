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

package version

import (
	"net/http"
	"regexp"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// The MCP tests compare against TrackingHeader and the Gemini test checks only
// substrings, so only this test pins its form.
func TestTrackingHeader(t *testing.T) {
	re := regexp.MustCompile(`^google-adk/` + regexp.QuoteMeta(Version) + ` gl-go/[0-9]`)
	if got := TrackingHeader(); !re.MatchString(got) {
		t.Errorf("TrackingHeader() = %q, want a match for %s", got, re)
	}
}

func TestWithTrackingHeaders(t *testing.T) {
	adk := TrackingHeader()
	tests := []struct {
		name string
		in   http.Header
		want http.Header
	}{
		{
			name: "nil",
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "caller tokens follow ADK labels",
			in:   http.Header{"User-Agent": {"my-app/1.0"}, "X-Goog-Api-Client": {"my-lib/2.0"}},
			want: http.Header{"User-Agent": {adk + " my-app/1.0"}, "X-Goog-Api-Client": {adk + " my-lib/2.0"}},
		},
		{
			name: "non-canonical keys are merged into one in sorted key order",
			in:   http.Header{"user-agent": {"a/1"}, "User-Agent": {"b/2"}, "USER-AGENT": {"c/3"}, "uSer-agEnt": {"d/4"}, "x-goog-api-client": {"e/5"}},
			want: http.Header{"User-Agent": {adk + " c/3 b/2 d/4 a/1"}, "X-Goog-Api-Client": {adk + " e/5"}},
		},
		{
			name: "duplicate tokens are dropped",
			in:   http.Header{"User-Agent": {"my-app/1.0 " + adk, "my-app/1.0"}},
			want: http.Header{"User-Agent": {adk + " my-app/1.0"}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "a stale ADK label is kept after the current one",
			in:   http.Header{"X-Goog-Api-Client": {"my-lib/2.0 google-adk/0.0.1"}},
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk + " my-lib/2.0 google-adk/0.0.1"}},
		},
		{
			name: "empty values add nothing",
			in:   http.Header{"User-Agent": {"", "  "}},
			want: http.Header{"User-Agent": {adk}, "X-Goog-Api-Client": {adk}},
		},
		{
			name: "other headers are kept",
			in:   http.Header{"X-Custom": {"v"}},
			want: http.Header{"User-Agent": {adk}, "X-Custom": {"v"}, "X-Goog-Api-Client": {adk}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			orig := tt.in.Clone()
			// Repeated, since map order would make an unsorted merge pass by chance.
			for range 20 {
				got := WithTrackingHeaders(tt.in)
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Fatalf("WithTrackingHeaders() mismatch (-want +got):\n%s", diff)
				}
			}
			if diff := cmp.Diff(orig, tt.in); diff != "" {
				t.Errorf("WithTrackingHeaders() modified its input (-want +got):\n%s", diff)
			}
		})
	}
}
