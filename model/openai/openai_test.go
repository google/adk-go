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

package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// mockOpenAIResponse creates a standard OpenAI chat completion response.
func mockOpenAIResponse(content string, finishReason string) openAIResponse {
	return openAIResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openAIChoice{
			{
				Index: 0,
				Message: &openAIMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &openAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
}

// mockToolCallResponse creates an OpenAI response with tool calls.
func mockToolCallResponse(name string, args map[string]any) openAIResponse {
	argsJSON, _ := json.Marshal(args)
	return openAIResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openAIChoice{
			{
				Index: 0,
				Message: &openAIMessage{
					Role: "assistant",
					ToolCalls: []openAIToolCall{
						{
							ID:   "call_test123",
							Type: "function",
							Function: openAIFunctionCall{
								Name:      name,
								Arguments: string(argsJSON),
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: &openAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
}

// newTestServer creates a mock HTTP server that returns the given response.
func newTestServer(t *testing.T, response any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("expected /chat/completions, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
}

// newStreamingTestServer creates a mock HTTP server for streaming responses.
func newStreamingTestServer(t *testing.T, chunks []string, finalContent string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		// Send chunks
		for i, chunk := range chunks {
			data := openAIResponse{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							Content: chunk,
						},
					},
				},
			}
			jsonData, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()

			// Last chunk includes finish_reason
			if i == len(chunks)-1 {
				finalData := openAIResponse{
					ID:    "chatcmpl-test",
					Model: "test-model",
					Choices: []openAIChoice{
						{
							Index:        0,
							Delta:        &openAIMessage{},
							FinishReason: "stop",
						},
					},
					Usage: &openAIUsage{
						PromptTokens:     10,
						CompletionTokens: 5,
						TotalTokens:      15,
					},
				}
				jsonData, _ := json.Marshal(finalData)
				fmt.Fprintf(w, "data: %s\n\n", jsonData)
				flusher.Flush()
			}
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

// newTestModel creates a model connected to the test server.
func newTestModel(t *testing.T, server *httptest.Server) model.LLM {
	t.Helper()
	llm, err := NewModel(context.Background(), "test-model", &ClientConfig{
		APIKey:     "test-api-key",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	return llm
}

func TestModel_Generate(t *testing.T) {
	tests := []struct {
		name     string
		req      *model.LLMRequest
		response openAIResponse
		want     *model.LLMResponse
		wantErr  bool
	}{
		{
			name: "simple_text",
			req: &model.LLMRequest{
				Contents: genai.Text("What is 2+2?"),
				Config: &genai.GenerateContentConfig{
					Temperature: float32Ptr(0),
				},
			},
			response: mockOpenAIResponse("4", "stop"),
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "4"}},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 5,
					TotalTokenCount:      15,
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
		{
			name: "with_system_instruction",
			req: &model.LLMRequest{
				Contents: genai.Text("Tell me a joke"),
				Config: &genai.GenerateContentConfig{
					SystemInstruction: genai.NewContentFromText("You are a pirate.", "system"),
					Temperature:       float32Ptr(0.7),
				},
			},
			response: mockOpenAIResponse("Arrr, why did the pirate go to school? To improve his arrrticulation!", "stop"),
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Arrr, why did the pirate go to school? To improve his arrrticulation!"}},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 5,
					TotalTokenCount:      15,
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer(t, tt.response)
			defer server.Close()

			llm := newTestModel(t, server)

			for got, err := range llm.GenerateContent(t.Context(), tt.req, false) {
				if (err != nil) != tt.wantErr {
					t.Errorf("GenerateContent() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(genai.Content{}, genai.Part{})); diff != "" {
					t.Errorf("GenerateContent() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestModel_GenerateStream(t *testing.T) {
	tests := []struct {
		name    string
		req     *model.LLMRequest
		chunks  []string
		want    string
		wantErr bool
	}{
		{
			name: "streaming_text",
			req: &model.LLMRequest{
				Contents: genai.Text("Count from 1 to 5"),
				Config: &genai.GenerateContentConfig{
					Temperature: float32Ptr(0),
				},
			},
			chunks: []string{"1", ", 2", ", 3", ", 4", ", 5"},
			want:   "1, 2, 3, 4, 5",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newStreamingTestServer(t, tt.chunks, tt.want)
			defer server.Close()

			llm := newTestModel(t, server)

			var partialText strings.Builder
			for resp, err := range llm.GenerateContent(t.Context(), tt.req, true) {
				if (err != nil) != tt.wantErr {
					t.Errorf("GenerateContent() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if resp.Partial && len(resp.Content.Parts) > 0 {
					partialText.WriteString(resp.Content.Parts[0].Text)
				}
			}

			if got := partialText.String(); got != tt.want {
				t.Errorf("GenerateContent() streaming = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestModel_FunctionCalling(t *testing.T) {
	tests := []struct {
		name         string
		req          *model.LLMRequest
		response     openAIResponse
		wantFuncName string
		wantArgs     map[string]any
		wantErr      bool
	}{
		{
			name: "function_call",
			req: &model.LLMRequest{
				Contents: genai.Text("What's the weather in Paris?"),
				Config: &genai.GenerateContentConfig{
					Temperature: float32Ptr(0),
					Tools: []*genai.Tool{
						{
							FunctionDeclarations: []*genai.FunctionDeclaration{
								{
									Name:        "get_weather",
									Description: "Get the current weather for a location",
									Parameters: &genai.Schema{
										Type: genai.TypeObject,
										Properties: map[string]*genai.Schema{
											"location": {
												Type:        genai.TypeString,
												Description: "The city name",
											},
										},
										Required: []string{"location"},
									},
								},
							},
						},
					},
				},
			},
			response:     mockToolCallResponse("get_weather", map[string]any{"location": "Paris"}),
			wantFuncName: "get_weather",
			wantArgs:     map[string]any{"location": "Paris"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer(t, tt.response)
			defer server.Close()

			llm := newTestModel(t, server)

			for resp, err := range llm.GenerateContent(t.Context(), tt.req, false) {
				if (err != nil) != tt.wantErr {
					t.Errorf("GenerateContent() error = %v, wantErr %v", err, tt.wantErr)
					return
				}

				// Find function call in parts
				var foundCall *genai.FunctionCall
				for _, part := range resp.Content.Parts {
					if part.FunctionCall != nil {
						foundCall = part.FunctionCall
						break
					}
				}

				if foundCall == nil {
					t.Fatal("expected function call in response")
				}
				if foundCall.Name != tt.wantFuncName {
					t.Errorf("FunctionCall.Name = %q, want %q", foundCall.Name, tt.wantFuncName)
				}
				if diff := cmp.Diff(tt.wantArgs, foundCall.Args); diff != "" {
					t.Errorf("FunctionCall.Args mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestModel_ImageAnalysis(t *testing.T) {
	server := newTestServer(t, mockOpenAIResponse("This image shows a plate of scones.", "stop"))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "image/jpeg",
							Data:     []byte("fake-image-data"),
						},
					},
					{Text: "What do you see in this image?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if len(resp.Content.Parts) == 0 {
			t.Fatal("expected response parts")
		}
		if !strings.Contains(resp.Content.Parts[0].Text, "scones") {
			t.Errorf("expected response to contain 'scones', got %q", resp.Content.Parts[0].Text)
		}
	}
}

func TestModel_AudioAnalysis(t *testing.T) {
	server := newTestServer(t, mockOpenAIResponse("The audio contains a discussion about Pixel phones.", "stop"))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "audio/mpeg",
							Data:     []byte("fake-audio-data"),
						},
					},
					{Text: "What is being said in this audio?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if len(resp.Content.Parts) == 0 {
			t.Fatal("expected response parts")
		}
		if !strings.Contains(resp.Content.Parts[0].Text, "Pixel") {
			t.Errorf("expected response to contain 'Pixel', got %q", resp.Content.Parts[0].Text)
		}
	}
}

func TestModel_VideoAnalysis(t *testing.T) {
	server := newTestServer(t, mockOpenAIResponse("The video shows a demonstration of the Pixel 8 phone.", "stop"))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "video/mp4",
							Data:     []byte("fake-video-data"),
						},
					},
					{Text: "What is happening in this video?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if len(resp.Content.Parts) == 0 {
			t.Fatal("expected response parts")
		}
		if !strings.Contains(resp.Content.Parts[0].Text, "Pixel 8") {
			t.Errorf("expected response to contain 'Pixel 8', got %q", resp.Content.Parts[0].Text)
		}
	}
}

func TestModel_PDFAnalysis(t *testing.T) {
	server := newTestServer(t, mockOpenAIResponse("This PDF document is about machine learning research.", "stop"))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "application/pdf",
							Data:     []byte("fake-pdf-data"),
						},
					},
					{Text: "What is this PDF document about?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0.2),
		},
	}

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if len(resp.Content.Parts) == 0 {
			t.Fatal("expected response parts")
		}
		if !strings.Contains(resp.Content.Parts[0].Text, "machine learning") {
			t.Errorf("expected response to contain 'machine learning', got %q", resp.Content.Parts[0].Text)
		}
	}
}

func TestModel_Name(t *testing.T) {
	server := newTestServer(t, mockOpenAIResponse("test", "stop"))
	defer server.Close()

	llm := newTestModel(t, server)

	if got := llm.Name(); got != "test-model" {
		t.Errorf("Name() = %q, want %q", got, "test-model")
	}
}

func TestModel_ErrorHandling(t *testing.T) {
	// Test server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": {"message": "Invalid request"}}`))
	}))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: genai.Text("test"),
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err == nil {
			t.Error("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "400") {
			t.Errorf("expected error to contain '400', got %v", err)
		}
	}
}

func TestNewModel_MissingConfig(t *testing.T) {
	// Test without API key
	_, err := NewModel(context.Background(), "test-model", &ClientConfig{
		BaseURL: "http://localhost",
	})
	if err == nil {
		t.Error("expected error for missing API key")
	}

	// Test without base URL
	_, err = NewModel(context.Background(), "test-model", &ClientConfig{
		APIKey: "test-key",
	})
	if err == nil {
		t.Error("expected error for missing base URL")
	}
}

// Helper function
func float32Ptr(f float32) *float32 {
	return &f
}
