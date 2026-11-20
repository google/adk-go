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
	"maps"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"slices"
	"strings"
)

// TrackingHeader returns the client labels ADK puts at the front of the
// user-agent and x-goog-api-client headers so that services can attribute
// traffic to ADK Go, for example "google-adk/2.6.0 gl-go/1.26.5". Other tokens
// may follow them in the header. It has the same form as the
// labels adk-python builds in google/adk/utils/_client_labels_utils.py.
//
// dest is the URL the request goes to. When it is a Google API endpoint and
// the process runs on a runtime ADK recognizes, the ADK label names that
// runtime, as in "google-adk/2.6.0+cloud_run". Any other dest, including "",
// gets the label without a suffix.
func TrackingHeader(dest string) string {
	return strings.Join(trackingTokens(dest), " ")
}

var languageToken = "gl-go/" + strings.TrimPrefix(runtime.Version(), "go")

func trackingTokens(dest string) []string {
	adk := "google-adk/" + Version
	if tag := runtimeTag(); tag != "" && isGoogleAPI(dest) {
		adk += "+" + tag
	}
	return []string{adk, languageToken}
}

// runtimeTag returns the tag of the runtime hosting the process, or "" when
// none is recognized. The tags and the order match adk-python's
// _detect_runtime_tag. Only whether a variable is set to a non-empty value is
// read, never the value, and nothing is cached.
func runtimeTag() string {
	switch {
	// Most specific first: the Agent Engine runtime may set K_SERVICE too, and
	// Knative on a cluster sets KUBERNETES_SERVICE_HOST as well.
	case os.Getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID") != "":
		return "remote_reasoning_engine"
	// K_SERVICE is set for Cloud Run services, Cloud Functions (2nd gen) and
	// Knative, CLOUD_RUN_JOB for Cloud Run jobs.
	case os.Getenv("K_SERVICE") != "" || os.Getenv("CLOUD_RUN_JOB") != "":
		return "cloud_run"
	// Kubernetes sets it in pods on any cluster, not only on GKE.
	case os.Getenv("KUBERNETES_SERVICE_HOST") != "":
		return "k8s"
	}
	return ""
}

// isGoogleAPI reports whether dest is an https URL of a host under
// googleapis.com. It follows adk-python's is_googleapis_url, and also rejects
// a URL that url.Parse does. A URL with userinfo is rejected, because HTTP
// clients disagree on which host it names.
func isGoogleAPI(dest string) bool {
	u, err := url.Parse(dest)
	if err != nil {
		return false
	}
	const suffix = ".googleapis.com"
	host := u.Hostname()
	// Compared as bytes of the same length rather than after strings.ToLower,
	// which turns letters such as "İ" into ASCII ones.
	i := len(host) - len(suffix)
	return u.Scheme == "https" && u.User == nil && i >= 0 && strings.EqualFold(host[i:], suffix)
}

// WithTrackingHeaders returns a copy of h whose user-agent and
// x-goog-api-client headers hold ADK's tracking labels followed by the tokens
// h held under any casing of those names, without duplicates. It is the rule
// adk-python's merge_tracking_headers applies. dest is the URL the request
// goes to, as for [TrackingHeader]. h is not modified.
func WithTrackingHeaders(h http.Header, dest string) http.Header {
	merged := h.Clone()
	if merged == nil {
		merged = make(http.Header)
	}
	adk := trackingTokens(dest)
	for _, name := range []string{"User-Agent", "X-Goog-Api-Client"} {
		tokens := slices.Clone(adk)
		// Sorted, so differently cased keys merge in the same order every time.
		for _, key := range slices.Sorted(maps.Keys(merged)) {
			if !strings.EqualFold(key, name) {
				continue
			}
			for _, v := range merged[key] {
				for tok := range strings.FieldsSeq(v) {
					if !slices.Contains(tokens, tok) {
						tokens = append(tokens, tok)
					}
				}
			}
			delete(merged, key)
		}
		merged.Set(name, strings.Join(tokens, " "))
	}
	return merged
}
