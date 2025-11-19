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

package gemini

import (
	"fmt"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
)

//go:generate go test -httprecord=testdata/.*\.httprr

func TestModel_Generate(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		req       *model.LLMRequest
		want      *model.LLMResponse
		wantErr   bool
	}{
		{
			name:      "ok",
			modelName: "gemini-2.0-flash",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France? One word."),
				Config: &genai.GenerateContentConfig{
					Temperature: new(float32),
				},
			},
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("Paris\n", genai.RoleModel),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					CandidatesTokenCount:    2,
					CandidatesTokensDetails: []*genai.ModalityTokenCount{{Modality: "TEXT", TokenCount: 2}},
					PromptTokenCount:        10,
					PromptTokensDetails:     []*genai.ModalityTokenCount{{Modality: "TEXT", TokenCount: 10}},
					TotalTokenCount:         12,
				},
				FinishReason: "STOP",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")

			testModel, err := NewModel(t.Context(), tt.modelName, newGeminiTestClientConfig(t, httpRecordFilename))
			if err != nil {
				t.Fatal(err)
			}

			for got, err := range testModel.GenerateContent(t.Context(), tt.req, false) {
				if (err != nil) != tt.wantErr {
					t.Errorf("Model.Generate() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreFields(model.LLMResponse{}, "AvgLogprobs")); diff != "" {
					t.Errorf("Model.Generate() = %v, want %v\ndiff(-want +got):\n%v", got, tt.want, diff)
				}
			}
		})
	}
}

func TestModel_GenerateStream(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		req       *model.LLMRequest
		want      string
		wantErr   bool
	}{
		{
			name:      "ok",
			modelName: "gemini-2.0-flash",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France? One word."),
				Config: &genai.GenerateContentConfig{
					Temperature: new(float32),
				},
			},
			want: "Paris\n",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")

			model, err := NewModel(t.Context(), tt.modelName, newGeminiTestClientConfig(t, httpRecordFilename))
			if err != nil {
				t.Fatal(err)
			}

			// Transforms the stream into strings, concatenating the text value of the response parts
			got, err := readResponse(model.GenerateContent(t.Context(), tt.req, true))
			if (err != nil) != tt.wantErr {
				t.Errorf("Model.GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if diff := cmp.Diff(tt.want, got.PartialText); diff != "" {
				t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.PartialText, tt.want, diff)
			}
			// Since we are expecting GenerateStream to aggregate partial events, the text should be the same
			if diff := cmp.Diff(tt.want, got.FinalText); diff != "" {
				t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.FinalText, tt.want, diff)
			}
		})
	}
}

func TestModel_TrackingHeaders(t *testing.T) {
	t.Run("verifies_headers_are_set", func(t *testing.T) {
		httpRecordFilename := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".httprr")

		baseTransport, err := testutil.NewGeminiTransport(httpRecordFilename)
		if err != nil {
			t.Fatal(err)
		}

		headersChecked := false
		interceptor := &headerInterceptor{
			base: baseTransport,
			check: func(req *http.Request) {
				headersChecked = true
				// Verify that standard tracking headers are present.
				// The exact expected values for these may need adjustment based on
				// the specific implementation of the tracking logic.
				if ua := req.Header.Get("User-Agent"); !strings.Contains(ua, "google-adk/") || !strings.Contains(ua, "gl-go/") {
					t.Errorf("User-Agent header should contain both 'google-adk/' and 'gl-go/', but got: %q", ua)
				}
				if xgac := req.Header.Get("x-goog-api-client"); !strings.Contains(xgac, "google-adk/") || !strings.Contains(xgac, "gl-go/") {
					t.Errorf("x-goog-api-client header should contain both 'google-adk/' and 'gl-go/', but got: %q", xgac)
				}
			},
		}

		apiKey := ""
		if recording, _ := httprr.Recording(httpRecordFilename); !recording {
			apiKey = "fakekey"
		}

		cfg := &genai.ClientConfig{
			HTTPClient: &http.Client{Transport: interceptor},
			APIKey:     apiKey,
		}

		geminiModel, err := NewModel(t.Context(), "gemini-2.0-flash", cfg)
		if err != nil {
			t.Fatal(err)
		}

		// Trigger a request to fire the interceptor.
		// We don't strictly care about the success of the call, only that it was attempted with headers.
		req := &model.LLMRequest{Contents: genai.Text("ping")}
		for _, err := range geminiModel.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Logf("GenerateContent finished with error (expected if no recording exists): %v", err)
			}
		}

		if !headersChecked {
			t.Error("HTTP request was not intercepted; headers not verified")
		}
	})
}

// newGeminiTestClientConfig returns the genai.ClientConfig configured for record and replay.
func newGeminiTestClientConfig(t *testing.T, rrfile string) *genai.ClientConfig {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
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

// TextResponse holds the concatenated text from a response stream,
// separated into partial and final parts.
type TextResponse struct {
	// PartialText is the full text concatenated from all partial (streaming) responses.
	PartialText string
	// FinalText is the full text concatenated from all final (non-partial) responses.
	FinalText string
}

// readResponse transforms a sequence into a TextResponse, concatenating the text value of the response parts
// depending on the readPartial value it will only concatenate the text of partial events or the text of non partial events
func readResponse(s iter.Seq2[*model.LLMResponse, error]) (TextResponse, error) {
	var partialBuilder, finalBuilder strings.Builder
	var result TextResponse

	for resp, err := range s {
		if err != nil {
			// Return what we have so far, along with the error.
			result.PartialText = partialBuilder.String()
			result.FinalText = finalBuilder.String()
			return result, err
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return result, fmt.Errorf("encountered an empty response: %v", resp)
		}

		text := resp.Content.Parts[0].Text
		if resp.Partial {
			partialBuilder.WriteString(text)
		} else {
			finalBuilder.WriteString(text)
		}
	}

	result.PartialText = partialBuilder.String()
	result.FinalText = finalBuilder.String()
	return result, nil
}

// headerInterceptor is a http.RoundTripper that executes a check function on the request
// before delegating to the base transport.
type headerInterceptor struct {
	base  http.RoundTripper
	check func(*http.Request)
}

func (h *headerInterceptor) RoundTrip(req *http.Request) (*http.Response, error) {
	if h.check != nil {
		h.check(req)
	}
	if h.base == nil {
		return http.DefaultTransport.RoundTrip(req)
	}
	return h.base.RoundTrip(req)
}
