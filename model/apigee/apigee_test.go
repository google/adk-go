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

package apigee

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

const proxyURL = "https://test.apigee.net"

// roundTripFunc is an adapter to allow the use of ordinary functions as http.RoundTrippers.
type roundTripFunc func(req *http.Request) (*http.Response, error)

// RoundTrip executes the round trip.
func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// newTestClient returns an *http.Client with the Transport replaced by the provided roundTripFunc.
func newTestClient(fn roundTripFunc) *http.Client {
	return &http.Client{
		Transport: fn,
	}
}

func TestNewModelWithValidModelStrings(t *testing.T) {
	validModelStrings := []string{
		"apigee/gemini-1.5-flash",
		"apigee/v1/gemini-1.5-flash",
		"apigee/vertex_ai/gemini-1.5-flash",
		"apigee/gemini/v1/gemini-1.5-flash",
		"apigee/vertex_ai/v1beta/gemini-1.5-flash",
	}
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	for _, modelName := range validModelStrings {
		t.Run(modelName, func(t *testing.T) {
			if strings.Contains(modelName, "vertex_ai") {
				t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "test-location")
			} else {
				if err := os.Unsetenv("GOOGLE_CLOUD_PROJECT"); err != nil {
					t.Errorf("failed to unset GOOGLE_CLOUD_PROJECT: %v", err)
				}
				if err := os.Unsetenv("GOOGLE_CLOUD_LOCATION"); err != nil {
					t.Errorf("failed to unset GOOGLE_CLOUD_LOCATION: %v", err)
				}
			}
			client := newTestClient(func(req *http.Request) (*http.Response, error) {
				// Check if the request URL is what we expect
				if req.URL.String() != "https://www.google.com" {
					t.Errorf("Unexpected URL: got %s, want https://www.google.com", req.URL.String())
					return nil, fmt.Errorf("unexpected URL: %s", req.URL.String())
				}

				// Return a mock response
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("<html><body>Mock Google</body></html>")),
					Header:     http.Header{"Content-Type": []string{"text/html"}},
				}, nil
			})

			_, err := NewModel(ctx, modelName, WithProxyURL(proxyURL), WithHTTPClient(client))
			if err != nil {
				t.Errorf("NewModel(%q) returned an unexpected error: %v", modelName, err)
			}
		})
	}
}

func TestNewModelWithInvalidModelStrings(t *testing.T) {
	invalidModelStrings := []string{
		"apigee/openai/v1/gpt",
		"apigee/",
		"apigee",
		"gemini-pro",
		"apigee/vertex_ai/v1/model/extra",
		"apigee/unknown/model",
	}
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	for _, modelName := range invalidModelStrings {
		t.Run(modelName, func(t *testing.T) {
			_, err := NewModel(ctx, modelName, WithProxyURL(proxyURL))
			if err == nil {
				t.Errorf("NewModel(%q) did not return an error for invalid model string", modelName)
			}
		})
	}
}

func TestParseModelName(t *testing.T) {
	testCases := []struct {
		name      string
		modelName string
		vertexEnv string
		want      *modelInfo
		wantErr   bool
	}{
		{
			name:      "simple",
			modelName: "apigee/gemini-1.5-flash",
			vertexEnv: "",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "", isVertexAI: false},
		},
		{
			name:      "simple vertex env",
			modelName: "apigee/gemini-1.5-flash",
			vertexEnv: "true",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "", isVertexAI: true},
		},
		{
			name:      "v1",
			modelName: "apigee/v1/gemini-1.5-flash",
			vertexEnv: "",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "v1", isVertexAI: false},
		},
		{
			name:      "vertex",
			modelName: "apigee/vertex_ai/gemini-1.5-flash",
			vertexEnv: "",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "", isVertexAI: true},
		},
		{
			name:      "gemini v1",
			modelName: "apigee/gemini/v1/gemini-1.5-flash",
			vertexEnv: "",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "v1", isVertexAI: false},
		},
		{
			name:      "vertex v1beta",
			modelName: "apigee/vertex_ai/v1beta/gemini-1.5-flash",
			vertexEnv: "",
			want:      &modelInfo{modelID: "gemini-1.5-flash", apiVersion: "v1beta", isVertexAI: true},
		},
		{
			name:      "invalid openai",
			modelName: "apigee/openai/v1/gpt",
			wantErr:   true,
		},
		{
			name:      "invalid trailing slash",
			modelName: "apigee/",
			wantErr:   true,
		},
		{
			name:      "invalid extra parts",
			modelName: "apigee/vertex_ai/v1/model/extra",
			wantErr:   true,
		},
		{
			name:      "invalid unknown",
			modelName: "apigee/unknown/model",
			wantErr:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.vertexEnv != "" {
				t.Setenv(googleGenaiUseVertexAIEnvVar, tc.vertexEnv)
			} else {
				if err := os.Unsetenv(googleGenaiUseVertexAIEnvVar); err != nil {
					t.Errorf("failed to unset %s: %v", googleGenaiUseVertexAIEnvVar, err)
				}
			}
			got, err := parseModelName(tc.modelName)
			if (err != nil) != tc.wantErr {
				t.Errorf("parseModelName(%q) error = %v, wantErr %v", tc.modelName, err, tc.wantErr)
				return
			}
			if !tc.wantErr {
				if !cmp.Equal(got, tc.want, cmp.AllowUnexported(modelInfo{})) {
					t.Errorf("parseModelName(%q) = %+v, want %+v", tc.modelName, got, tc.want)
				}
			}
		})
	}
}

func TestNewModelWithCustomHeaders(t *testing.T) {
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	headers := http.Header{}
	headers.Set("X-Custom-Header", "custom-value")
	_, err := NewModel(ctx, "apigee/gemini-1.5-flash", WithProxyURL(proxyURL), WithCustomHeaders(headers))
	if err != nil {
		t.Fatalf("NewModel() returned an unexpected error: %v", err)
	}
}

func TestNewModelWithoutProxyURL(t *testing.T) {
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	if err := os.Unsetenv(apigeeProxyURLEnvVar); err != nil {
		t.Errorf("failed to unset %s: %v", apigeeProxyURLEnvVar, err)
	}
	_, err := NewModel(ctx, "apigee/gemini-1.5-flash")
	if err == nil {
		t.Errorf("NewModel() did not return an error when proxy URL is not set")
	}

	t.Setenv(apigeeProxyURLEnvVar, "https://env.proxy.url")
	_, err = NewModel(ctx, "apigee/gemini-1.5-flash")
	if err != nil {
		t.Fatalf("NewModel() returned an unexpected error: %v", err)
	}
}

func TestNewModelVertexMissingProjectOrLocation(t *testing.T) {
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	t.Setenv(googleGenaiUseVertexAIEnvVar, "true")
	if err := os.Unsetenv(projectEnvVar); err != nil {
		t.Errorf("failed to unset %s: %v", projectEnvVar, err)
	}
	if err := os.Unsetenv(locationEnvVar); err != nil {
		t.Errorf("failed to unset %s: %v", locationEnvVar, err)
	}
	_, err := NewModel(ctx, "apigee/gemini-1.5-flash", WithProxyURL(proxyURL))
	if err == nil || !strings.Contains(err.Error(), projectEnvVar) {
		t.Errorf("NewModel() with vertex enabled but no project env var should fail")
	}

	t.Setenv(projectEnvVar, "test-project")
	_, err = NewModel(ctx, "apigee/gemini-1.5-flash", WithProxyURL(proxyURL))
	if err == nil || !strings.Contains(err.Error(), locationEnvVar) {
		t.Errorf("NewModel() with vertex enabled but no location env var should fail")
	}
}

// test GenerateContent
func TestGenerateContent(t *testing.T) {
	ctx := context.Background()
	t.Setenv("GOOGLE_API_KEY", "test-key")
	t.Setenv(googleGenaiUseVertexAIEnvVar, "true")
	t.Setenv(projectEnvVar, "test-project")
	t.Setenv(locationEnvVar, "test-location")
	client := newTestClient(func(req *http.Request) (*http.Response, error) {
		// Check if the request URL is what we expect
		if req.URL.String() != "https://test.apigee.net/v1/models/gemini-1.5-flash:generateContent" && req.URL.String() != "https://test.apigee.net/v1/models/gemini-1.5-flash:streamGenerateContent?alt=sse" {
			t.Errorf("Unexpected URL: got %s, want https://test.apigee.net/v1/models/gemini-1.5-flash:generateContent or https://test.apigee.net/v1/models/gemini-1.5-flash:streamGenerateContent?alt=sse", req.URL.String())
			return nil, fmt.Errorf("unexpected URL: %s", req.URL.String())
		}

		// Return a mock response
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Paris\"}]}}]}")),
			Header:     http.Header{"Content-Type": []string{"application/json"}},
		}, nil
	})
	apigeeModel, err := NewModel(ctx, "apigee/gemini/v1/gemini-1.5-flash", WithProxyURL(proxyURL), WithHTTPClient(client))
	if err != nil {
		t.Fatalf("NewModel() returned an unexpected error: %v", err)
	}
	req := &model.LLMRequest{
		Contents: genai.Text("What is the capital of France? One word."),
		Config: &genai.GenerateContentConfig{
			Temperature: new(float32),
		},
	}

	responses := apigeeModel.GenerateContent(ctx, req, false)
	for resp, err := range responses {
		if err != nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Errorf("GenerateContent() returned an unexpected error or empty response: %v", err)
		}
	}
}
