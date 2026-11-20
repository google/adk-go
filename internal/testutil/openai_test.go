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
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestScrubOpenAIRequest_KeepsOnlyPayloadHeaders(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("OpenAI-Organization", "org-secret")
	req.Header.Set("OpenAI-Project", "proj_secret")
	req.Header.Set("User-Agent", "OpenAI/Go 1.2.3")
	req.Header.Set("X-Stainless-Runtime-Version", "go1.2.3")

	if err := scrubOpenAIRequest(req); err != nil {
		t.Fatal(err)
	}

	want := http.Header{
		"Accept":       {"application/json"},
		"Content-Type": {"application/json"},
	}
	if diff := cmp.Diff(want, req.Header); diff != "" {
		t.Errorf("scrubbed request headers mismatch (-want +got):\n%s", diff)
	}
}

func TestScrubOpenAIResponse_KeepsOnlyContentType(t *testing.T) {
	base := roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header: http.Header{
				"Content-Type":                 {"application/json"},
				"Openai-Organization":          {"org-secret"},
				"Openai-Project":               {"proj_secret"},
				"Set-Cookie":                   {"__cf_bm=secret"},
				"X-Ratelimit-Remaining-Tokens": {"149999"},
				"X-Request-Id":                 {"req_secret"},
			},
		}, nil
	})
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := scrubOpenAIResponse{base}.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}

	want := http.Header{"Content-Type": {"application/json"}}
	if diff := cmp.Diff(want, resp.Header); diff != "" {
		t.Errorf("scrubbed response headers mismatch (-want +got):\n%s", diff)
	}
}

func TestScrubOpenAIResponse_RefusesFailures(t *testing.T) {
	for _, status := range []int{http.StatusTemporaryRedirect, http.StatusUnauthorized, http.StatusTooManyRequests, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := &closeRecorder{Reader: strings.NewReader(`{"error":{}}`)}
			base := roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: status, Body: body}, nil
			})
			req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
			if err != nil {
				t.Fatal(err)
			}

			resp, err := scrubOpenAIResponse{base}.RoundTrip(req)
			if err == nil {
				t.Errorf("RoundTrip() = %v, nil; want an error for status %d", resp, status)
			}
			if !body.closed {
				t.Error("RoundTrip() left the refused response body open")
			}
		})
	}
}

type closeRecorder struct {
	io.Reader
	closed bool
}

func (c *closeRecorder) Close() error {
	c.closed = true
	return nil
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
