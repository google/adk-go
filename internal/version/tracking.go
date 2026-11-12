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
	"runtime"
	"slices"
	"strings"
)

// AgentNameLabel is the request label under which the agent flow records the
// name of the agent that makes a model call, so that billing reports can be
// split per agent. The flow sets it in GenerateContentConfig.Labels unless the
// request already holds the key. Model connectors that cannot send labels
// ignore this key.
const AgentNameLabel = "adk_agent_name"

// TrackingHeader returns the client labels ADK puts at the front of the
// user-agent and x-goog-api-client headers so that services can attribute
// traffic to ADK Go, for example "google-adk/2.6.0 gl-go/1.26.5". Other tokens
// may follow them in the header. It has the same form as the
// labels adk-python builds in google/adk/utils/_client_labels_utils.py.
func TrackingHeader() string {
	return trackingHeader
}

var trackingHeader = "google-adk/" + Version + " gl-go/" + strings.TrimPrefix(runtime.Version(), "go")

var trackingTokens = strings.Fields(trackingHeader)

// WithTrackingHeaders returns a copy of h whose user-agent and
// x-goog-api-client headers hold ADK's tracking labels followed by the tokens
// h held under any casing of those names, without duplicates. It is the rule
// adk-python's merge_tracking_headers applies. h is not modified.
func WithTrackingHeaders(h http.Header) http.Header {
	merged := h.Clone()
	if merged == nil {
		merged = make(http.Header)
	}
	for _, name := range []string{"User-Agent", "X-Goog-Api-Client"} {
		tokens := slices.Clone(trackingTokens)
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
