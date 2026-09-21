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

package gemini

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/auth"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestStripReplayedFunctionCallIDs(t *testing.T) {
	tests := []struct {
		name     string
		contents []*genai.Content
		want     []*genai.Content
	}{
		{
			name: "clears a genuine model-issued function call id",
			contents: []*genai.Content{
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_213331", Name: "getWeather"}}}},
			},
			want: []*genai.Content{
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "", Name: "getWeather"}}}},
			},
		},
		{
			name: "leaves function response id untouched",
			contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "call_213331", Name: "getWeather"}}}},
			},
			want: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "call_213331", Name: "getWeather"}}}},
			},
		},
		{
			name: "leaves a function call with no id untouched",
			contents: []*genai.Content{
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "getWeather"}}}},
			},
			want: []*genai.Content{
				{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "getWeather"}}}},
			},
		},
		{
			name: "tolerates nil contents, a nil content, and nil/non-functionCall parts",
			contents: []*genai.Content{
				nil,
				{Role: "user", Parts: []*genai.Part{nil, {Text: "hello"}}},
			},
			want: []*genai.Content{
				nil,
				{Role: "user", Parts: []*genai.Part{nil, {Text: "hello"}}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stripReplayedFunctionCallIDs(tt.contents)
			if diff := cmp.Diff(tt.want, tt.contents); diff != "" {
				t.Errorf("stripReplayedFunctionCallIDs() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// staticTokenProvider avoids falling back to Application Default
// Credentials, which would otherwise hit the network during client
// construction.
type staticTokenProvider struct{}

func (staticTokenProvider) Token(context.Context) (*auth.Token, error) {
	return &auth.Token{Value: "fake-token", Expiry: time.Now().Add(time.Hour)}, nil
}

// capturingTransport records the first request body it sees, then refuses
// to send it.
type capturingTransport struct {
	body []byte
}

func (c *capturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		b, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		c.body = b
	}
	return nil, errors.New("capturingTransport: refusing to send, this test is offline by design")
}

func TestGenerateContent_StripsFunctionCallIDOnlyForVertex(t *testing.T) {
	tests := []struct {
		name       string
		backend    genai.Backend
		wantIDKept bool
	}{
		{name: "vertex_strips_the_id", backend: genai.BackendVertexAI, wantIDKept: false},
		{name: "gemini_api_keeps_the_id", backend: genai.BackendGeminiAPI, wantIDKept: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transport := &capturingTransport{}
			cfg := &genai.ClientConfig{
				Backend:    tt.backend,
				HTTPClient: &http.Client{Transport: transport},
			}
			// Vertex AI and the Gemini API backend take disjoint auth config.
			if tt.backend == genai.BackendVertexAI {
				cfg.Project = "fake-project"
				cfg.Location = "us-central1"
				cfg.Credentials = auth.NewCredentials(&auth.CredentialsOptions{
					TokenProvider: staticTokenProvider{},
				})
			} else {
				cfg.APIKey = "fake-api-key"
			}

			m, err := NewModel(t.Context(), "gemini-2.0-flash", cfg)
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_213331", Name: "getWeather"}}}},
				},
			}

			sawErr := false
			for _, err := range m.GenerateContent(t.Context(), req, false) {
				if err != nil {
					sawErr = true
				}
			}
			if !sawErr {
				t.Fatal("GenerateContent() succeeded; want an error from capturingTransport")
			}

			if transport.body == nil {
				t.Fatal("request never reached the transport; nothing captured")
			}
			gotIDKept := strings.Contains(string(transport.body), `"id":"call_213331"`)
			if gotIDKept != tt.wantIDKept {
				t.Errorf("outgoing request body retained the function call id = %v, want %v; body: %s", gotIDKept, tt.wantIDKept, transport.body)
			}
		})
	}
}
