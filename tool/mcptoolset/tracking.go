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

package mcptoolset

import (
	"net/http"
	"slices"
	"strings"

	"google.golang.org/adk/v2/internal/version"
)

// trackingTransport puts the ADK client labels at the front of the user-agent
// and x-goog-api-client headers of every request, so an MCP server can tell
// ADK traffic from other clients. A value already on the request when it gets
// here is kept after the ADK tokens, without duplicates, the rule adk-python's
// merge_tracking_headers applies to its MCP headers. go-sdk sets neither header
// itself, and base runs later, so a header base sets is not merged.
type trackingTransport struct {
	base http.RoundTripper
}

// adkTokens is shared by every request, so RoundTrip clones it before appending.
var adkTokens = strings.Fields(version.TrackingHeader())

func (t *trackingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for _, name := range []string{"User-Agent", "X-Goog-Api-Client"} {
		tokens := slices.Clone(adkTokens)
		for _, v := range req.Header.Values(name) {
			for _, tok := range strings.Fields(v) {
				if !slices.Contains(tokens, tok) {
					tokens = append(tokens, tok)
				}
			}
		}
		req.Header.Set(name, strings.Join(tokens, " "))
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(req)
}
