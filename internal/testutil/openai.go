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

package testutil

import (
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/internal/httprr"
)

// openAIRequestHeaders and openAIResponseHeaders are the only headers an OpenAI
// recording keeps, besides the framing httprr writes itself (Host,
// Content-Length, Connection). Everything else is dropped: the API key, the
// SDK's version and platform headers, the organization and project openai-go
// reads from OPENAI_ORG_ID and OPENAI_PROJECT_ID, and the response headers that
// identify the recording account (organization, project, rate limits, cookies,
// request IDs). An allowlist drops a header the SDK or the API adds later as
// well.
var (
	openAIRequestHeaders  = map[string]bool{"Accept": true, "Content-Type": true}
	openAIResponseHeaders = map[string]bool{"Content-Type": true}
)

// NewOpenAITransport returns a transport that records OpenAI traffic to rrfile,
// or replays it from there, for an openai-go client.
func NewOpenAITransport(rrfile string) (http.RoundTripper, error) {
	rr, err := httprr.Open(rrfile, scrubOpenAIResponse{http.DefaultTransport})
	if err != nil {
		return nil, fmt.Errorf("httprr.Open(%q) failed: %w", rrfile, err)
	}
	rr.ScrubReq(scrubOpenAIRequest)
	return rr, nil
}

func scrubOpenAIRequest(req *http.Request) error {
	keepHeaders(req.Header, openAIRequestHeaders)
	return nil
}

// scrubOpenAIResponse wraps the real transport, so it runs only when recording.
// It drops response headers before httprr writes them, so a recording run sees
// the same response a replay does. It refuses an error or redirect response
// rather than letting httprr write it: an error body can name the account, as a
// rate-limit error names the organization, and openai-go retries some errors
// under the same scrubbed key, which would leave the failure in the recording
// next to the success that replaced it.
type scrubOpenAIResponse struct {
	base http.RoundTripper
}

func (s scrubOpenAIResponse) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := s.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("refusing to record an OpenAI response with status %d: an error body can identify the account", resp.StatusCode)
	}
	keepHeaders(resp.Header, openAIResponseHeaders)
	return resp, nil
}

func keepHeaders(h http.Header, keep map[string]bool) {
	for name := range h {
		if !keep[name] {
			delete(h, name)
		}
	}
}
