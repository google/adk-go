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

package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/adk-go/internal/httprr"
	"google.golang.org/genai"
)

//go:generate go test -httprecord=TestNewGeminiModel

func TestNewGeminiModel(t *testing.T) {
	ctx := t.Context()
	modelName := "gemini-2.0-flash"

	cfg := newGeminiTestClientConfig(t, filepath.Join("testdata", t.Name()+".httprr"))
	m, err := NewGeminiModel(ctx, modelName, cfg)
	if err != nil {
		t.Fatalf("NewGeminiModel(%q) failed: %v", modelName, err)
	}
	if got, want := m.Name(), modelName; got != want {
		t.Errorf("model Name = %q, want %q", got, want)
	}

	readResponse := func(s adk.LLMResponseStream) (string, error) {
		var answer string
		for resp, err := range s {
			if err != nil {
				return answer, err
			}
			if resp.Content == nil || len(resp.Content.Parts) == 0 {
				return answer, fmt.Errorf("encountered an empty response: %v", resp)
			}
			answer += resp.Content.Parts[0].Text
		}
		return answer, nil
	}

	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			s := m.GenerateContent(ctx, &adk.LLMRequest{
				Model:    m, // TODO: strange. What happens if this doesn't match m?
				Contents: genai.Text("What is the capital of France?"),
			}, stream)
			answer, err := readResponse(s)
			if err != nil || !strings.Contains(strings.ToLower(answer), "paris") {
				t.Errorf("GenerateContent(stream=%v)=(%q, %v), want ('.*paris.*', nil)", stream, answer, err)
			}
		})
	}
}

// newGeminiTestClientConfig returns the genai.ClientConfig configured for record and replay.
func newGeminiTestClientConfig(t *testing.T, rrfile string) *genai.ClientConfig {
	t.Helper()
	rr, err := httprr.Open(rrfile, http.DefaultTransport)
	if err != nil {
		t.Fatalf("httprr.Open(%q) failed: %v", rrfile, err)
	}
	rr.ScrubReq(Scrub)

	// When operating in replay mode (!inRecordingMode), supply APIKey
	// because google.golang.com/genai.NewClient expects API key to be set
	// https://github.com/googleapis/go-genai/blob/f2244624b33ed6ecb5a14dddcad004bcc09b8e6b/client.go#L260-L262
	inRecordingMode, err := httprr.Recording(rrfile)
	if err != nil {
		t.Fatalf("httprr.Recording(%q) failed: %v", rrfile, err)
	}
	apiKey := ""
	if !inRecordingMode {
		apiKey = "fakeAPIKey"
	}

	return &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: rr},
		APIKey:     apiKey,
	}
}

func Scrub(req *http.Request) error {
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
