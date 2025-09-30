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
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

//go:generate go test -httprecord=testdata/.*\.httprr

func TestModel_Generate(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		req       *llm.Request
		want      *llm.Response
		wantErr   bool
	}{
		{
			name:      "ok",
			modelName: "gemini-2.0-flash",
			req: &llm.Request{
				Contents: genai.Text("What is the capital of France? One word."),
				GenerateConfig: &genai.GenerateContentConfig{
					Temperature: new(float32),
				},
			},
			want: &llm.Response{
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

			model, err := NewModel(t.Context(), tt.modelName, newGeminiTestClientConfig(t, httpRecordFilename))
			if err != nil {
				t.Fatal(err)
			}

			got, err := model.Generate(t.Context(), tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("Model.Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreFields(llm.Response{}, "AvgLogprobs")); diff != "" {
				t.Errorf("Model.Generate() = %v, want %v\ndiff(-want +got):\n%v", got, tt.want, diff)
			}
		})
	}
}

func TestModel_GenerateStream(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		req       *llm.Request
		want      string
		wantErr   bool
	}{
		{
			name:      "ok",
			modelName: "gemini-2.0-flash",
			req: &llm.Request{
				Contents: genai.Text("What is the capital of France? One word."),
				GenerateConfig: &genai.GenerateContentConfig{
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

			gotPartial, err := readResponsePartial(model.GenerateStream(t.Context(), tt.req))
			if (err != nil) != tt.wantErr {
				t.Errorf("Model.GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, gotPartial); diff != "" {
				t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", gotPartial, tt.want, diff)
			}

			gotNonPartial, err := readResponseNonPartial(model.GenerateStream(t.Context(), tt.req))
			if (err != nil) != tt.wantErr {
				t.Errorf("Model.GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, gotNonPartial); diff != "" {
				t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", gotNonPartial, tt.want, diff)
			}
		})
	}
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

func readResponsePartial(s iter.Seq2[*llm.Response, error]) (string, error) {
	var answer string
	for resp, err := range s {
		if err != nil {
			return answer, err
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return answer, fmt.Errorf("encountered an empty response: %v", resp)
		}
		if resp.Partial {
			answer += resp.Content.Parts[0].Text
		}
	}
	return answer, nil
}

func readResponseNonPartial(s iter.Seq2[*llm.Response, error]) (string, error) {
	var answer string
	for resp, err := range s {
		if err != nil {
			return answer, err
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return answer, fmt.Errorf("encountered an empty response: %v", resp)
		}
		if !resp.Partial {
			answer += resp.Content.Parts[0].Text
		}
	}
	return answer, nil
}
