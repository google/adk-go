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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// googleAPI is a destination that gets the hosting runtime suffix.
const googleAPI = "https://aiplatform.googleapis.com/v1"

// runtimeEnvVars are the variables that select a runtime suffix. The names
// and the suffixes in these tests are literals because services parse them.
var runtimeEnvVars = []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "K_SERVICE", "CLOUD_RUN_JOB", "KUBERNETES_SERVICE_HOST"}

// clearRuntimeEnv hides the runtime the test itself runs on, which may be one
// of the runtimes ADK detects.
func clearRuntimeEnv(t *testing.T) {
	t.Helper()
	for _, name := range runtimeEnvVars {
		t.Setenv(name, "")
	}
}

// adkToken returns the first token of a tracking header value.
func adkToken(t *testing.T, header string) string {
	t.Helper()
	token, _, _ := strings.Cut(header, " ")
	return token
}

// The MCP tests compare against TrackingHeader and the Gemini test checks only
// substrings, so only this test pins its form.
func TestTrackingHeader(t *testing.T) {
	clearRuntimeEnv(t)
	re := regexp.MustCompile(`^google-adk/` + regexp.QuoteMeta(Version) + ` gl-go/[0-9][^ ]*$`)
	for _, dest := range []string{"", googleAPI, "https://mcp.example/mcp"} {
		if got := TrackingHeader(dest); !re.MatchString(got) {
			t.Errorf("TrackingHeader(%q) = %q, want a match for %s", dest, got, re)
		}
	}
}

func TestTrackingHeaderRuntimeSuffix(t *testing.T) {
	tests := []struct {
		name string
		set  []string
		want string
	}{
		{name: "Agent Engine", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID"}, want: "+remote_reasoning_engine"},
		{name: "Cloud Run service", set: []string{"K_SERVICE"}, want: "+cloud_run"},
		{name: "Cloud Run job", set: []string{"CLOUD_RUN_JOB"}, want: "+cloud_run"},
		{name: "Cloud Run service and job", set: []string{"K_SERVICE", "CLOUD_RUN_JOB"}, want: "+cloud_run"},
		{name: "Kubernetes", set: []string{"KUBERNETES_SERVICE_HOST"}, want: "+k8s"},
		{name: "Cloud Run wins over Kubernetes", set: []string{"K_SERVICE", "KUBERNETES_SERVICE_HOST"}, want: "+cloud_run"},
		{name: "Cloud Run job wins over Kubernetes", set: []string{"CLOUD_RUN_JOB", "KUBERNETES_SERVICE_HOST"}, want: "+cloud_run"},
		{name: "Agent Engine wins over Cloud Run", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "K_SERVICE"}, want: "+remote_reasoning_engine"},
		{name: "Agent Engine wins over Kubernetes", set: []string{"GOOGLE_CLOUD_AGENT_ENGINE_ID", "KUBERNETES_SERVICE_HOST"}, want: "+remote_reasoning_engine"},
		{name: "Agent Engine wins over all", set: runtimeEnvVars, want: "+remote_reasoning_engine"},
		{name: "no runtime"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearRuntimeEnv(t)
			for _, name := range tt.set {
				t.Setenv(name, "some-value")
			}
			want := "google-adk/" + Version + tt.want
			if got := adkToken(t, TrackingHeader(googleAPI)); got != want {
				t.Errorf("TrackingHeader(%q) starts with %q, want %q", googleAPI, got, want)
			}
			// The value of a variable is never sent, only the suffix.
			if got := TrackingHeader(googleAPI); strings.Contains(got, "some-value") {
				t.Errorf("TrackingHeader(%q) = %q, which holds the value of an environment variable", googleAPI, got)
			}
			// A destination that is not a Google API never gets a suffix.
			for _, dest := range []string{"", "https://mcp.example/mcp"} {
				if got, want := adkToken(t, TrackingHeader(dest)), "google-adk/"+Version; got != want {
					t.Errorf("TrackingHeader(%q) starts with %q, want %q", dest, got, want)
				}
			}
		})
	}
}

// Detection is not cached, so a variable set after the first call counts.
func TestTrackingHeaderReadsEnvironmentOnEveryCall(t *testing.T) {
	clearRuntimeEnv(t)
	if got, want := adkToken(t, TrackingHeader(googleAPI)), "google-adk/"+Version; got != want {
		t.Fatalf("TrackingHeader(%q) starts with %q, want %q", googleAPI, got, want)
	}
	t.Setenv("CLOUD_RUN_JOB", "my-job")
	if got, want := adkToken(t, TrackingHeader(googleAPI)), "google-adk/"+Version+"+cloud_run"; got != want {
		t.Errorf("TrackingHeader(%q) starts with %q, want %q", googleAPI, got, want)
	}
}

// The table holds the rows of adk-python's test_is_googleapis_url, and more.
func TestTrackingHeaderDestination(t *testing.T) {
	clearRuntimeEnv(t)
	t.Setenv("K_SERVICE", "svc")
	tests := []struct {
		dest       string
		wantSuffix bool
	}{
		{dest: "https://aiplatform.googleapis.com/v1", wantSuffix: true},
		{dest: "https://aiplatform.mtls.googleapis.com/v1", wantSuffix: true},
		{dest: "https://us-central1-aiplatform.googleapis.com/", wantSuffix: true},
		{dest: "https://aiplatform.us.rep.googleapis.com/", wantSuffix: true},
		{dest: "https://BigQuery.GoogleAPIs.com/v1", wantSuffix: true},
		{dest: "HTTPS://bigquery.googleapis.com/v1", wantSuffix: true},
		{dest: "https://bigquery.googleapis.com:443/v1", wantSuffix: true},
		{dest: "https://bigquery.googleapis.com", wantSuffix: true},
		{dest: "https://\u0130.googleapis.com/v1", wantSuffix: true},
		// No such host exists, but it has the suffix, as in adk-python.
		{dest: "https://.googleapis.com/v1", wantSuffix: true},
		// A backslash outside the authority is fine.
		{dest: `https://bigquery.googleapis.com/v1?q=a\b`, wantSuffix: true},
		{dest: "http://aiplatform.googleapis.com/v1"},
		{dest: "wss://generativelanguage.googleapis.com/"},
		{dest: "https://notgoogleapis.com/v1"},
		{dest: "https://googleapis.com.evil.com/v1"},
		{dest: "https://example.com/v1"},
		{dest: "https://example.com/bigquery.googleapis.com"},
		{dest: "https://example.com/?u=https://bigquery.googleapis.com"},
		// The apex is not a service endpoint.
		{dest: "https://googleapis.com/v1"},
		{dest: "https://googleapis.com@evil.com/v1"},
		{dest: "https://user@bigquery.googleapis.com/v1"},
		{dest: "https://user:pw@bigquery.googleapis.com/v1"},
		{dest: `https://evil.com\.googleapis.com/v1`},
		{dest: `https://evil.com\@bigquery.googleapis.com/v1`},
		{dest: "bigquery.googleapis.com/v1"},
		{dest: "//bigquery.googleapis.com/v1"},
		{dest: "https://bigquery.googleapis.com./v1"},
		{dest: "https://%zz.googleapis.com/v1"},
		// "İ" and "ſ" are not the ASCII letters that case mapping turns them
		// into, so these are names of other domains.
		{dest: "https://x.googleap\u0130s.com/v1"},
		{dest: "https://x.googleapi\u017f.com/v1"},
		{dest: ""},
	}
	for _, tt := range tests {
		want := "google-adk/" + Version
		if tt.wantSuffix {
			want += "+cloud_run"
		}
		if got := adkToken(t, TrackingHeader(tt.dest)); got != want {
			t.Errorf("TrackingHeader(%q) starts with %q, want %q", tt.dest, got, want)
		}
	}
}

func TestWithTrackingHeaders(t *testing.T) {
	adk := TrackingHeader("")
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
				got := WithTrackingHeaders(tt.in, "")
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

func TestWithTrackingHeadersRuntimeSuffix(t *testing.T) {
	clearRuntimeEnv(t)
	t.Setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
	plain, tagged := TrackingHeader(""), TrackingHeader(googleAPI)
	if want := "google-adk/" + Version + "+k8s"; adkToken(t, tagged) != want {
		t.Fatalf("TrackingHeader(%q) = %q, want it to start with %q", googleAPI, tagged, want)
	}
	in := http.Header{"User-Agent": {"my-app/1.0"}}
	tests := []struct {
		dest string
		want http.Header
	}{
		{dest: googleAPI, want: http.Header{"User-Agent": {tagged + " my-app/1.0"}, "X-Goog-Api-Client": {tagged}}},
		{dest: "https://mcp.example/mcp", want: http.Header{"User-Agent": {plain + " my-app/1.0"}, "X-Goog-Api-Client": {plain}}},
		{dest: "", want: http.Header{"User-Agent": {plain + " my-app/1.0"}, "X-Goog-Api-Client": {plain}}},
	}
	for _, tt := range tests {
		if diff := cmp.Diff(tt.want, WithTrackingHeaders(in, tt.dest)); diff != "" {
			t.Errorf("WithTrackingHeaders(_, %q) mismatch (-want +got):\n%s", tt.dest, diff)
		}
	}
}
