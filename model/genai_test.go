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
	replayTrace := filepath.Join("testdata", t.Name()+".httprr")
	cfg := newGeminiTestClientConfig(t, replayTrace)

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
	rr, err := httprr.NewGeminiTransportForTesting(rrfile)
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
