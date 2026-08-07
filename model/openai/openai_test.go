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
	"os"
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
	// Save original env vars
	origAPIKey := os.Getenv("OPENAI_API_KEY")
	origBaseURL := os.Getenv("OPENAI_BASE_URL")
	defer func() {
		os.Setenv("OPENAI_API_KEY", origAPIKey)
		os.Setenv("OPENAI_BASE_URL", origBaseURL)
	}()

	// Test without API key
	os.Unsetenv("OPENAI_API_KEY")
	os.Setenv("OPENAI_BASE_URL", "http://localhost")
	_, err := NewModel(context.Background(), "test-model", &ClientConfig{
		BaseURL: "http://localhost",
	})
	if err == nil {
		t.Error("expected error for missing API key")
	}

	// Test without base URL
	os.Setenv("OPENAI_API_KEY", "test-key")
	os.Unsetenv("OPENAI_BASE_URL")
	_, err = NewModel(context.Background(), "test-model", &ClientConfig{
		APIKey: "test-key",
	})
	if err == nil {
		t.Error("expected error for missing base URL")
	}
}

func TestConvertFunctionDeclaration(t *testing.T) {
	tests := []struct {
		name string
		fn   *genai.FunctionDeclaration
		want openAITool
	}{
		{
			name: "with_ParametersJsonSchema_map",
			fn: &genai.FunctionDeclaration{
				Name:        "get_weather",
				Description: "Get weather for a location",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []any{"location"},
				},
			},
			want: openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        "get_weather",
					Description: "Get weather for a location",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "City name",
							},
						},
						"required": []any{"location"},
					},
				},
			},
		},
		{
			name: "with_Parameters_legacy",
			fn: &genai.FunctionDeclaration{
				Name:        "calculate",
				Description: "Calculate something",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"x": {
							Type:        genai.TypeNumber,
							Description: "First number",
						},
						"y": {
							Type:        genai.TypeNumber,
							Description: "Second number",
						},
					},
					Required: []string{"x", "y"},
				},
			},
			want: openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        "calculate",
					Description: "Calculate something",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"x": map[string]any{
								"type":        "number",
								"description": "First number",
							},
							"y": map[string]any{
								"type":        "number",
								"description": "Second number",
							},
						},
						"required": []string{"x", "y"},
					},
				},
			},
		},
		{
			name: "no_parameters",
			fn: &genai.FunctionDeclaration{
				Name:        "get_time",
				Description: "Get current time",
			},
			want: openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        "get_time",
					Description: "Get current time",
					Parameters:  map[string]any{},
				},
			},
		},
		{
			name: "prefers_ParametersJsonSchema_over_Parameters",
			fn: &genai.FunctionDeclaration{
				Name:        "test_tool",
				Description: "Test tool with both schemas",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"new_param": map[string]any{"type": "string"},
					},
				},
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"old_param": {Type: genai.TypeString},
					},
				},
			},
			want: openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        "test_tool",
					Description: "Test tool with both schemas",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"new_param": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertFunctionDeclaration(tt.fn)
			if diff := cmp.Diff(tt.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("convertFunctionDeclaration() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTryConvertJsonSchema(t *testing.T) {
	tests := []struct {
		name   string
		schema any
		want   map[string]any
	}{
		{
			name: "already_map",
			schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"field": map[string]any{"type": "string"}},
			},
			want: map[string]any{
				"type":       "object",
				"properties": map[string]any{"field": map[string]any{"type": "string"}},
			},
		},
		{
			name: "struct_via_json",
			schema: struct {
				Type       string                 `json:"type"`
				Properties map[string]interface{} `json:"properties"`
			}{
				Type: "object",
				Properties: map[string]interface{}{
					"name": map[string]interface{}{"type": "string"},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
			},
		},
		{
			name:   "invalid_type",
			schema: make(chan int), // Cannot be marshaled
			want:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tryConvertJsonSchema(tt.schema)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("tryConvertJsonSchema() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertLegacyParameters(t *testing.T) {
	tests := []struct {
		name   string
		schema *genai.Schema
		want   map[string]any
	}{
		{
			name: "with_properties_and_required",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"name": {
						Type:        genai.TypeString,
						Description: "User name",
					},
					"age": {
						Type:        genai.TypeInteger,
						Description: "User age",
					},
				},
				Required: []string{"name"},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "User name",
					},
					"age": map[string]any{
						"type":        "integer",
						"description": "User age",
					},
				},
				"required": []string{"name"},
			},
		},
		{
			name: "empty_properties",
			schema: &genai.Schema{
				Type: genai.TypeObject,
			},
			want: map[string]any{
				"type": "object",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertLegacyParameters(tt.schema)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("convertLegacyParameters() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestModel_StreamingToolCalls(t *testing.T) {
	// Test server that streams tool calls with Index field
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		// Stream multiple tool calls with explicit Index values
		chunks := []openAIResponse{
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							ToolCalls: []openAIToolCall{
								{
									Index: intPtr(0),
									ID:    "call_abc123",
									Type:  "function",
									Function: openAIFunctionCall{
										Name:      "get_weather",
										Arguments: "",
									},
								},
							},
						},
					},
				},
			},
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							ToolCalls: []openAIToolCall{
								{
									Index: intPtr(0),
									Function: openAIFunctionCall{
										Arguments: `{"location":`,
									},
								},
							},
						},
					},
				},
			},
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							ToolCalls: []openAIToolCall{
								{
									Index: intPtr(0),
									Function: openAIFunctionCall{
										Arguments: ` "Paris"}`,
									},
								},
							},
						},
					},
				},
			},
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index:        0,
						Delta:        &openAIMessage{},
						FinishReason: "tool_calls",
					},
				},
				Usage: &openAIUsage{
					PromptTokens:     15,
					CompletionTokens: 8,
					TotalTokens:      23,
				},
			},
		}

		for _, chunk := range chunks {
			jsonData, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: genai.Text("What's the weather in Paris?"),
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0),
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get weather for a location",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"location": {Type: genai.TypeString},
								},
							},
						},
					},
				},
			},
		},
	}

	var finalResp *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if !resp.Partial {
			finalResp = resp
		}
	}

	if finalResp == nil {
		t.Fatal("expected final response")
	}

	// Find function call in parts
	var foundCall *genai.FunctionCall
	for _, part := range finalResp.Content.Parts {
		if part.FunctionCall != nil {
			foundCall = part.FunctionCall
			break
		}
	}

	if foundCall == nil {
		t.Fatal("expected function call in final response")
	}
	if foundCall.Name != "get_weather" {
		t.Errorf("FunctionCall.Name = %q, want %q", foundCall.Name, "get_weather")
	}

	expectedArgs := map[string]any{"location": "Paris"}
	if diff := cmp.Diff(expectedArgs, foundCall.Args); diff != "" {
		t.Errorf("FunctionCall.Args mismatch (-want +got):\n%s", diff)
	}

	if foundCall.ID != "call_abc123" {
		t.Errorf("FunctionCall.ID = %q, want %q", foundCall.ID, "call_abc123")
	}
}

func TestModel_StreamingMultipleToolCalls(t *testing.T) {
	// Test server that streams multiple tool calls with different indices
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		chunks := []openAIResponse{
			// First tool call starts
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							ToolCalls: []openAIToolCall{
								{
									Index: intPtr(0),
									ID:    "call_1",
									Type:  "function",
									Function: openAIFunctionCall{
										Name:      "get_weather",
										Arguments: `{"location":"Tokyo"}`,
									},
								},
							},
						},
					},
				},
			},
			// Second tool call starts
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Delta: &openAIMessage{
							ToolCalls: []openAIToolCall{
								{
									Index: intPtr(1),
									ID:    "call_2",
									Type:  "function",
									Function: openAIFunctionCall{
										Name:      "get_time",
										Arguments: `{"timezone":"JST"}`,
									},
								},
							},
						},
					},
				},
			},
			// Finish
			{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index:        0,
						Delta:        &openAIMessage{},
						FinishReason: "tool_calls",
					},
				},
			},
		}

		for _, chunk := range chunks {
			jsonData, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: genai.Text("Get weather in Tokyo and current time"),
		Config: &genai.GenerateContentConfig{
			Temperature: float32Ptr(0),
		},
	}

	var finalResp *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if !resp.Partial {
			finalResp = resp
		}
	}

	if finalResp == nil {
		t.Fatal("expected final response")
	}

	// Should have 2 function calls
	var functionCalls []*genai.FunctionCall
	for _, part := range finalResp.Content.Parts {
		if part.FunctionCall != nil {
			functionCalls = append(functionCalls, part.FunctionCall)
		}
	}

	if len(functionCalls) != 2 {
		t.Fatalf("expected 2 function calls, got %d", len(functionCalls))
	}

	// Verify first call
	if functionCalls[0].Name != "get_weather" {
		t.Errorf("functionCalls[0].Name = %q, want %q", functionCalls[0].Name, "get_weather")
	}
	if functionCalls[0].ID != "call_1" {
		t.Errorf("functionCalls[0].ID = %q, want %q", functionCalls[0].ID, "call_1")
	}

	// Verify second call
	if functionCalls[1].Name != "get_time" {
		t.Errorf("functionCalls[1].Name = %q, want %q", functionCalls[1].Name, "get_time")
	}
	if functionCalls[1].ID != "call_2" {
		t.Errorf("functionCalls[1].ID = %q, want %q", functionCalls[1].ID, "call_2")
	}
}

func TestModel_EmptyToolCallFiltering(t *testing.T) {
	// Test that empty tool calls are filtered out
	tests := []struct {
		name     string
		response openAIResponse
		wantLen  int
	}{
		{
			name: "filters_empty_tool_call",
			response: openAIResponse{
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
									ID:   "",
									Type: "",
									Function: openAIFunctionCall{
										Name:      "",
										Arguments: "",
									},
								},
								{
									ID:   "call_valid",
									Type: "function",
									Function: openAIFunctionCall{
										Name:      "valid_function",
										Arguments: `{"arg": "value"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			},
			wantLen: 1,
		},
		{
			name: "keeps_valid_tool_calls",
			response: openAIResponse{
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
									ID:   "call_1",
									Type: "function",
									Function: openAIFunctionCall{
										Name:      "func1",
										Arguments: `{}`,
									},
								},
								{
									ID:   "call_2",
									Type: "function",
									Function: openAIFunctionCall{
										Name:      "func2",
										Arguments: `{"x": 1}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer(t, tt.response)
			defer server.Close()

			llm := newTestModel(t, server)

			req := &model.LLMRequest{
				Contents: genai.Text("test"),
			}

			for resp, err := range llm.GenerateContent(t.Context(), req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() error = %v", err)
				}

				var functionCalls []*genai.FunctionCall
				for _, part := range resp.Content.Parts {
					if part.FunctionCall != nil {
						functionCalls = append(functionCalls, part.FunctionCall)
					}
				}

				if len(functionCalls) != tt.wantLen {
					t.Errorf("expected %d function calls, got %d", tt.wantLen, len(functionCalls))
				}
			}
		})
	}
}

func TestBuildFinalResponse_EmptyToolCallFiltering(t *testing.T) {
	m := &openAIModel{
		modelName: "test-model",
	}

	tests := []struct {
		name      string
		toolCalls []openAIToolCall
		wantLen   int
	}{
		{
			name: "filters_all_empty",
			toolCalls: []openAIToolCall{
				{ID: "", Function: openAIFunctionCall{Name: "", Arguments: ""}},
				{ID: "", Function: openAIFunctionCall{Name: "", Arguments: ""}},
			},
			wantLen: 0,
		},
		{
			name: "filters_mixed",
			toolCalls: []openAIToolCall{
				{ID: "", Function: openAIFunctionCall{Name: "", Arguments: ""}},
				{ID: "call_1", Function: openAIFunctionCall{Name: "valid", Arguments: `{"x": 1}`}},
			},
			wantLen: 1,
		},
		{
			name: "keeps_all_valid",
			toolCalls: []openAIToolCall{
				{ID: "call_1", Function: openAIFunctionCall{Name: "func1", Arguments: `{}`}},
				{ID: "call_2", Function: openAIFunctionCall{Name: "func2", Arguments: `{}`}},
			},
			wantLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := m.buildFinalResponse("", tt.toolCalls, nil, "stop")

			var functionCalls []*genai.FunctionCall
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					functionCalls = append(functionCalls, part.FunctionCall)
				}
			}

			if len(functionCalls) != tt.wantLen {
				t.Errorf("expected %d function calls, got %d", tt.wantLen, len(functionCalls))
			}
		})
	}
}

// TestExtractTexts tests the extractTexts function with various input types
func TestExtractTexts(t *testing.T) {
	tests := []struct {
		name  string
		input any
		want  []*genai.Part
	}{
		{
			name:  "nil_input",
			input: nil,
			want:  nil,
		},
		{
			name:  "string_input",
			input: "This is reasoning content",
			want: []*genai.Part{
				{Text: "This is reasoning content", Thought: true},
			},
		},
		{
			name:  "empty_string",
			input: "",
			want:  nil,
		},
		{
			name:  "array_of_strings",
			input: []any{"First thought", "Second thought", ""},
			want: []*genai.Part{
				{Text: "First thought", Thought: true},
				{Text: "Second thought", Thought: true},
			},
		},
		{
			name: "map_with_text_key",
			input: map[string]any{
				"text": "Extracted from map",
			},
			want: []*genai.Part{
				{Text: "Extracted from map", Thought: true},
			},
		},
		{
			name: "map_with_content_key",
			input: map[string]any{
				"content": "Content field",
			},
			want: []*genai.Part{
				{Text: "Content field", Thought: true},
			},
		},
		{
			name: "map_with_reasoning_key",
			input: map[string]any{
				"reasoning": "Reasoning text",
			},
			want: []*genai.Part{
				{Text: "Reasoning text", Thought: true},
			},
		},
		{
			name: "map_with_reasoning_content_key",
			input: map[string]any{
				"reasoning_content": "Reasoning content text",
			},
			want: []*genai.Part{
				{Text: "Reasoning content text", Thought: true},
			},
		},
		{
			name: "map_with_multiple_keys",
			input: map[string]any{
				"text":    "Text value",
				"content": "Content value",
				"other":   "Should be ignored",
			},
			want: []*genai.Part{
				{Text: "Text value", Thought: true},
				{Text: "Content value", Thought: true},
			},
		},
		{
			name: "nested_array_with_maps",
			input: []any{
				map[string]any{"text": "First"},
				map[string]any{"content": "Second"},
				"Direct string",
			},
			want: []*genai.Part{
				{Text: "First", Thought: true},
				{Text: "Second", Thought: true},
				{Text: "Direct string", Thought: true},
			},
		},
		{
			name: "map_with_non_string_values",
			input: map[string]any{
				"text":  123,
				"other": "ignored",
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var parts []*genai.Part
			extractTexts(tt.input, &parts)
			if diff := cmp.Diff(tt.want, parts, cmpopts.IgnoreUnexported(genai.Part{})); diff != "" {
				t.Errorf("extractTexts() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestExtractReasoningParts tests the extractReasoningParts function
func TestExtractReasoningParts(t *testing.T) {
	tests := []struct {
		name             string
		reasoningContent any
		want             []*genai.Part
	}{
		{
			name:             "nil_content",
			reasoningContent: nil,
			want:             nil,
		},
		{
			name:             "string_content",
			reasoningContent: "Let me think about this",
			want: []*genai.Part{
				{Text: "Let me think about this", Thought: true},
			},
		},
		{
			name: "array_of_reasoning",
			reasoningContent: []any{
				"First reasoning step",
				"Second reasoning step",
			},
			want: []*genai.Part{
				{Text: "First reasoning step", Thought: true},
				{Text: "Second reasoning step", Thought: true},
			},
		},
		{
			name: "map_with_reasoning",
			reasoningContent: map[string]any{
				"reasoning": "Deep thought process",
			},
			want: []*genai.Part{
				{Text: "Deep thought process", Thought: true},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractReasoningParts(tt.reasoningContent)
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(genai.Part{})); diff != "" {
				t.Errorf("extractReasoningParts() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestParseToolCallsFromText tests the parseToolCallsFromText function
func TestParseToolCallsFromText(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		wantCallCount int
		wantCalls     []openAIToolCall
		wantRemainder string
	}{
		{
			name:          "empty_text",
			text:          "",
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: "",
		},
		{
			name:          "no_tool_calls",
			text:          "This is just regular text without any JSON",
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: "This is just regular text without any JSON",
		},
		{
			name:          "single_tool_call",
			text:          `Use the tool: {"name": "get_weather", "arguments": {"location": "Paris"}}`,
			wantCallCount: 1,
			wantCalls: []openAIToolCall{
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "get_weather",
						Arguments: `{"location":"Paris"}`,
					},
				},
			},
			wantRemainder: "Use the tool:",
		},
		{
			name:          "multiple_tool_calls",
			text:          `First: {"name": "func1", "arguments": {"a": 1}} then {"name": "func2", "arguments": {"b": 2}}`,
			wantCallCount: 2,
			wantCalls: []openAIToolCall{
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "func1",
						Arguments: `{"a":1}`,
					},
				},
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "func2",
						Arguments: `{"b":2}`,
					},
				},
			},
			wantRemainder: "First:  then",
		},
		{
			name:          "tool_call_with_id",
			text:          `{"id": "call_123", "name": "test_func", "arguments": {}}`,
			wantCallCount: 1,
			wantCalls: []openAIToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "test_func",
						Arguments: `{}`,
					},
				},
			},
			wantRemainder: "",
		},
		{
			name:          "arguments_as_string",
			text:          `{"name": "stringify", "arguments": "{\"key\": \"value\"}"}`,
			wantCallCount: 1,
			wantCalls: []openAIToolCall{
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "stringify",
						Arguments: `{"key": "value"}`,
					},
				},
			},
			wantRemainder: "",
		},
		{
			name:          "arguments_as_object",
			text:          `{"name": "objectify", "arguments": {"nested": {"deep": "value"}}}`,
			wantCallCount: 1,
			wantCalls: []openAIToolCall{
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "objectify",
						Arguments: `{"nested":{"deep":"value"}}`,
					},
				},
			},
			wantRemainder: "",
		},
		{
			name:          "invalid_json_object",
			text:          `{"not_a_tool": "call"} regular text`,
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: `{"not_a_tool": "call"} regular text`,
		},
		{
			name:          "missing_name_field",
			text:          `{"arguments": {"x": 1}}`,
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: `{"arguments": {"x": 1}}`,
		},
		{
			name:          "missing_arguments_field",
			text:          `{"name": "no_args"}`,
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: `{"name": "no_args"}`,
		},
		{
			name:          "malformed_json",
			text:          `{invalid json} some text`,
			wantCallCount: 0,
			wantCalls:     nil,
			wantRemainder: `{invalid json} some text`,
		},
		{
			name:          "json_in_middle_of_text",
			text:          `Before {"name": "middle", "arguments": {"pos": "center"}} after`,
			wantCallCount: 1,
			wantCalls: []openAIToolCall{
				{
					Type: "function",
					Function: openAIFunctionCall{
						Name:      "middle",
						Arguments: `{"pos":"center"}`,
					},
				},
			},
			wantRemainder: "Before  after",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, remainder := parseToolCallsFromText(tt.text)

			if len(calls) != tt.wantCallCount {
				t.Errorf("parseToolCallsFromText() got %d calls, want %d", len(calls), tt.wantCallCount)
			}

			if remainder != tt.wantRemainder {
				t.Errorf("parseToolCallsFromText() remainder = %q, want %q", remainder, tt.wantRemainder)
			}

			if tt.wantCalls != nil {
				for i, wantCall := range tt.wantCalls {
					if i >= len(calls) {
						t.Fatalf("Missing call at index %d", i)
					}
					if calls[i].Type != wantCall.Type {
						t.Errorf("Call[%d].Type = %q, want %q", i, calls[i].Type, wantCall.Type)
					}
					if calls[i].Function.Name != wantCall.Function.Name {
						t.Errorf("Call[%d].Function.Name = %q, want %q", i, calls[i].Function.Name, wantCall.Function.Name)
					}
					if calls[i].Function.Arguments != wantCall.Function.Arguments {
						t.Errorf("Call[%d].Function.Arguments = %q, want %q", i, calls[i].Function.Arguments, wantCall.Function.Arguments)
					}
					if wantCall.ID != "" && calls[i].ID != wantCall.ID {
						t.Errorf("Call[%d].ID = %q, want %q", i, calls[i].ID, wantCall.ID)
					}
					if wantCall.ID == "" && calls[i].ID == "" {
						t.Errorf("Call[%d].ID should be generated but is empty", i)
					}
				}
			}
		})
	}
}

// TestModel_ResponseWithReasoningContent tests handling of reasoning content in responses
func TestModel_ResponseWithReasoningContent(t *testing.T) {
	tests := []struct {
		name             string
		response         openAIResponse
		wantThoughtCount int
		wantThoughtText  []string
	}{
		{
			name: "string_reasoning_content",
			response: openAIResponse{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Message: &openAIMessage{
							Role:             "assistant",
							Content:          "The answer is 42",
							ReasoningContent: "Let me think... I need to calculate this carefully.",
						},
						FinishReason: "stop",
					},
				},
			},
			wantThoughtCount: 1,
			wantThoughtText:  []string{"Let me think... I need to calculate this carefully."},
		},
		{
			name: "array_reasoning_content",
			response: openAIResponse{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Message: &openAIMessage{
							Role:    "assistant",
							Content: "Final answer",
							ReasoningContent: []any{
								"First step of reasoning",
								"Second step of reasoning",
							},
						},
						FinishReason: "stop",
					},
				},
			},
			wantThoughtCount: 2,
			wantThoughtText:  []string{"First step of reasoning", "Second step of reasoning"},
		},
		{
			name: "map_reasoning_content",
			response: openAIResponse{
				ID:    "chatcmpl-test",
				Model: "test-model",
				Choices: []openAIChoice{
					{
						Index: 0,
						Message: &openAIMessage{
							Role:    "assistant",
							Content: "Result",
							ReasoningContent: map[string]any{
								"text": "Thought process here",
							},
						},
						FinishReason: "stop",
					},
				},
			},
			wantThoughtCount: 1,
			wantThoughtText:  []string{"Thought process here"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer(t, tt.response)
			defer server.Close()

			llm := newTestModel(t, server)

			req := &model.LLMRequest{
				Contents: genai.Text("test"),
			}

			for resp, err := range llm.GenerateContent(context.Background(), req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() error = %v", err)
				}

				var thoughtParts []*genai.Part
				for _, part := range resp.Content.Parts {
					if part.Thought {
						thoughtParts = append(thoughtParts, part)
					}
				}

				if len(thoughtParts) != tt.wantThoughtCount {
					t.Errorf("got %d thought parts, want %d", len(thoughtParts), tt.wantThoughtCount)
				}

				for i, wantText := range tt.wantThoughtText {
					if i >= len(thoughtParts) {
						t.Fatalf("Missing thought part at index %d", i)
					}
					if thoughtParts[i].Text != wantText {
						t.Errorf("ThoughtPart[%d].Text = %q, want %q", i, thoughtParts[i].Text, wantText)
					}
				}
			}
		})
	}
}

// TestModel_StreamingWithReasoningContent tests streaming responses with reasoning
func TestModel_StreamingWithReasoningContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		chunk1 := openAIResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openAIChoice{
				{
					Index: 0,
					Delta: &openAIMessage{
						Content: "Answer: ",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk1)
		fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		chunk2 := openAIResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openAIChoice{
				{
					Index: 0,
					Delta: &openAIMessage{
						Content: "42",
					},
				},
			},
		}
		jsonData, _ = json.Marshal(chunk2)
		fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		finalChunk := openAIResponse{
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
				PromptTokens:     5,
				CompletionTokens: 3,
				TotalTokens:      8,
			},
		}
		jsonData, _ = json.Marshal(finalChunk)
		fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: genai.Text("What is the answer?"),
	}

	var finalResp *model.LLMResponse
	partialCount := 0
	for resp, err := range llm.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if resp.Partial {
			partialCount++
		} else {
			finalResp = resp
		}
	}

	if partialCount == 0 {
		t.Error("expected at least one partial response")
	}

	if finalResp == nil {
		t.Fatal("expected final response")
	}

	if finalResp.UsageMetadata == nil {
		t.Error("expected usage metadata in final response")
	}
}

// TestModel_StreamingNoFinishReason tests fallback when stream ends without FinishReason
func TestModel_StreamingNoFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}

		chunk := openAIResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openAIChoice{
				{
					Index: 0,
					Delta: &openAIMessage{
						Content: "Hello",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()
	}))
	defer server.Close()

	llm := newTestModel(t, server)

	req := &model.LLMRequest{
		Contents: genai.Text("test"),
	}

	var finalResp *model.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if !resp.Partial {
			finalResp = resp
		}
	}

	if finalResp == nil {
		t.Fatal("expected final response even without explicit finish_reason")
	}

	if finalResp.FinishReason != genai.FinishReasonStop {
		t.Errorf("expected finish reason 'stop', got %v", finalResp.FinishReason)
	}
}

// TestConvertContent tests the convertContent function with various content types
func TestConvertContent(t *testing.T) {
	m := &openAIModel{modelName: "test-model"}

	tests := []struct {
		name    string
		content *genai.Content
		want    []openAIMessage
		wantErr bool
	}{
		{
			name:    "nil_content",
			content: nil,
			want:    nil,
			wantErr: false,
		},
		{
			name: "empty_parts",
			content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{},
			},
			want:    nil,
			wantErr: false,
		},
		{
			name: "text_only",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Hello"},
				},
			},
			want: []openAIMessage{
				{
					Role:    "user",
					Content: "Hello",
				},
			},
			wantErr: false,
		},
		{
			name: "multiple_text_parts",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Hello"},
					{Text: "World"},
				},
			},
			want: []openAIMessage{
				{
					Role:    "user",
					Content: "Hello\nWorld",
				},
			},
			wantErr: false,
		},
		{
			name: "model_role_converts_to_assistant",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: "Response"},
				},
			},
			want: []openAIMessage{
				{
					Role:    "assistant",
					Content: "Response",
				},
			},
			wantErr: false,
		},
		{
			name: "function_response",
			content: &genai.Content{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "call_123",
							Name: "get_weather",
							Response: map[string]any{
								"temperature": 72,
								"condition":   "sunny",
							},
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role:       "tool",
					Content:    `{"condition":"sunny","temperature":72}`,
					ToolCallID: "call_123",
				},
			},
			wantErr: false,
		},
		{
			name: "function_call",
			content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_456",
							Name: "search",
							Args: map[string]any{"query": "weather"},
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role: "assistant",
					ToolCalls: []openAIToolCall{
						{
							ID:   "call_456",
							Type: "function",
							Function: openAIFunctionCall{
								Name:      "search",
								Arguments: `{"query":"weather"}`,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "inline_image_data",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "image/jpeg",
							Data:     []byte("fake-image"),
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role: "user",
					Content: []map[string]any{
						{
							"type": "image_url",
							"image_url": map[string]any{
								"url": "data:image/jpeg;base64,ZmFrZS1pbWFnZQ==",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "inline_text_data",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "text/plain",
							Data:     []byte("text content"),
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role:    "user",
					Content: "text content",
				},
			},
			wantErr: false,
		},
		{
			name: "file_data_with_uri",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{
						FileData: &genai.FileData{
							FileURI: "file-123",
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role: "user",
					Content: []map[string]any{
						{
							"type": "file",
							"file": map[string]any{
								"file_id": "file-123",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "mixed_text_and_image",
			content: &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What's in this image?"},
					{
						InlineData: &genai.Blob{
							MIMEType: "image/png",
							Data:     []byte("image-data"),
						},
					},
				},
			},
			want: []openAIMessage{
				{
					Role: "user",
					Content: []map[string]any{
						{
							"type": "text",
							"text": "What's in this image?",
						},
						{
							"type": "image_url",
							"image_url": map[string]any{
								"url": "data:image/png;base64,aW1hZ2UtZGF0YQ==",
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.convertContent(tt.content)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertContent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			if len(got) != len(tt.want) {
				t.Fatalf("convertContent() got %d messages, want %d", len(got), len(tt.want))
			}

			for i := range tt.want {
				if tt.want[i].Role == "tool" && tt.want[i].ToolCallID == "" {
					if got[i].ToolCallID == "" {
						t.Errorf("Message[%d].ToolCallID should be generated but is empty", i)
					}
					tt.want[i].ToolCallID = got[i].ToolCallID
				}

				if len(tt.want[i].ToolCalls) > 0 && tt.want[i].ToolCalls[0].ID == "" {
					if len(got[i].ToolCalls) == 0 || got[i].ToolCalls[0].ID == "" {
						t.Errorf("Message[%d].ToolCalls[0].ID should be generated but is empty", i)
					}
					if len(got[i].ToolCalls) > 0 {
						tt.want[i].ToolCalls[0].ID = got[i].ToolCalls[0].ID
					}
				}

				if diff := cmp.Diff(tt.want[i], got[i], cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("Message[%d] mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

// TestExtractTextFromContent tests the extractTextFromContent function
func TestExtractTextFromContent(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
	}{
		{
			name:    "nil_content",
			content: nil,
			want:    "",
		},
		{
			name: "empty_parts",
			content: &genai.Content{
				Parts: []*genai.Part{},
			},
			want: "",
		},
		{
			name: "single_text_part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Hello"},
				},
			},
			want: "Hello",
		},
		{
			name: "multiple_text_parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Line 1"},
					{Text: "Line 2"},
					{Text: "Line 3"},
				},
			},
			want: "Line 1\nLine 2\nLine 3",
		},
		{
			name: "mixed_parts_with_non_text",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Text 1"},
					{InlineData: &genai.Blob{MIMEType: "image/jpeg", Data: []byte("img")}},
					{Text: "Text 2"},
				},
			},
			want: "Text 1\nText 2",
		},
		{
			name: "no_text_parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/jpeg", Data: []byte("img")}},
				},
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextFromContent(tt.content)
			if got != tt.want {
				t.Errorf("extractTextFromContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestMapFinishReason tests the mapFinishReason function
func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		name   string
		reason string
		want   genai.FinishReason
	}{
		{
			name:   "stop",
			reason: "stop",
			want:   genai.FinishReasonStop,
		},
		{
			name:   "length",
			reason: "length",
			want:   genai.FinishReasonMaxTokens,
		},
		{
			name:   "tool_calls",
			reason: "tool_calls",
			want:   genai.FinishReasonStop,
		},
		{
			name:   "function_call",
			reason: "function_call",
			want:   genai.FinishReasonStop,
		},
		{
			name:   "content_filter",
			reason: "content_filter",
			want:   genai.FinishReasonSafety,
		},
		{
			name:   "unknown",
			reason: "some_unknown_reason",
			want:   genai.FinishReasonOther,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapFinishReason(tt.reason)
			if got != tt.want {
				t.Errorf("mapFinishReason(%q) = %v, want %v", tt.reason, got, tt.want)
			}
		})
	}
}

// TestBuildUsageMetadata tests the buildUsageMetadata function
func TestBuildUsageMetadata(t *testing.T) {
	tests := []struct {
		name  string
		usage *openAIUsage
		want  *genai.GenerateContentResponseUsageMetadata
	}{
		{
			name:  "nil_usage",
			usage: nil,
			want:  nil,
		},
		{
			name: "basic_usage",
			usage: &openAIUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
				TotalTokenCount:      15,
			},
		},
		{
			name: "with_cached_tokens",
			usage: &openAIUsage{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
				PromptTokensDetails: &promptTokensDetails{
					CachedTokens: 30,
				},
			},
			want: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        100,
				CandidatesTokenCount:    50,
				TotalTokenCount:         150,
				CachedContentTokenCount: 30,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildUsageMetadata(tt.usage)
			if diff := cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(genai.GenerateContentResponseUsageMetadata{})); diff != "" {
				t.Errorf("buildUsageMetadata() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Helper functions
func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
