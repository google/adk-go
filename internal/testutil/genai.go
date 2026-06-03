// Copyright 2025 Google LLC
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
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/httprr"
)

// NewGeminiTransport returns the genai.ClientConfig configured for record and replay.
func NewGeminiTransport(rrfile string) (http.RoundTripper, error) {
	rr, err := httprr.Open(rrfile, http.DefaultTransport)
	if err != nil {
		return nil, fmt.Errorf("httprr.Open(%q) failed: %w", rrfile, err)
	}
	rr.ScrubReq(scrubGeminiRequest)
	return rr, nil
}

func scrubGeminiRequest(req *http.Request) error {
	delete(req.Header, "x-goog-api-key")    // genai does not canonicalize
	req.Header.Del("X-Goog-Api-Key")        // in case it starts
	delete(req.Header, "x-goog-api-client") // contains version numbers
	req.Header.Del("X-Goog-Api-Client")
	delete(req.Header, "user-agent") // contains google-genai-sdk and gl-go version numbers
	req.Header.Del("User-Agent")

	if ctype := req.Header.Get("Content-Type"); ctype == "application/json" || strings.HasPrefix(ctype, "application/json;") {
		// Canonicalize JSON body.
		// google.golang.org/protobuf/internal/encoding.json
		// goes out of its way to randomize the JSON encodings
		// of protobuf messages by adding or not adding spaces
		// after commas. Derandomize by compacting the JSON.
		b := req.Body.(*httprr.Body)
		var buf bytes.Buffer
		if err := json.Compact(&buf, b.Data); err == nil {
			b.Data = buf.Bytes()
		}
	}
	return nil
}

// NewGeminiTestClientConfig returns the genai.ClientConfig configured for record and replay.
func NewGeminiTestClientConfig(t *testing.T, rrfile string) *genai.ClientConfig {
	t.Helper()
	rr, err := NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	apiKey := ""
	if recording, _ := httprr.Recording(rrfile); !recording {
		apiKey = "fakekey"
	}
	return &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: rr},
		APIKey:     apiKey,
	}
}
