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

package minimax

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// roundTripFunc is an adapter to allow the use of ordinary functions as
// http.RoundTrippers.
type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func newTestClient(fn roundTripFunc) *http.Client {
	return &http.Client{Transport: fn}
}

func newMockResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
}

func TestNewModel_MissingAPIKey(t *testing.T) {
	t.Setenv(apiKeyEnvVar, "")
	_, err := NewModel("MiniMax-M2.7")
	if err == nil {
		t.Error("NewModel() should return an error when API key is missing")
	}
}

func TestNewModel_WithAPIKey(t *testing.T) {
	m, err := NewModel("MiniMax-M2.7", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}
	if m.Name() != "MiniMax-M2.7" {
		t.Errorf("Name() = %q, want %q", m.Name(), "MiniMax-M2.7")
	}
}

func TestNewModel_FromEnvVar(t *testing.T) {
	t.Setenv(apiKeyEnvVar, "env-key")
	m, err := NewModel("MiniMax-M2.7")
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}
	if m.Name() != "MiniMax-M2.7" {
		t.Errorf("Name() = %q, want %q", m.Name(), "MiniMax-M2.7")
	}
}

func TestGenerateContent_SimpleText(t *testing.T) {
	responseBody := `{
		"choices": [{
			"message": {"role": "assistant", "content": "Hello, world!"},
			"finish_reason": "stop"
		}]
	}`
	client := newTestClient(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", req.Method)
		}
		if !strings.HasSuffix(req.URL.Path, "/v1/chat/completions") {
			t.Errorf("unexpected path: %s", req.URL.Path)
		}
		if req.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected auth header: %s", req.Header.Get("Authorization"))
		}
		return newMockResponse(responseBody), nil
	})

	m, err := NewModel("MiniMax-M2.7", WithAPIKey("test-key"), WithHTTPClient(client))
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("Hello"),
	}

	var got *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}

	if got == nil || got.Content == nil || len(got.Content.Parts) == 0 {
		t.Fatal("GenerateContent() returned empty response")
	}
	if got.Content.Parts[0].Text != "Hello, world!" {
		t.Errorf("got text %q, want %q", got.Content.Parts[0].Text, "Hello, world!")
	}
}

func TestGenerateContent_WithSystemInstruction(t *testing.T) {
	var capturedBody []byte
	client := newTestClient(func(req *http.Request) (*http.Response, error) {
		capturedBody, _ = io.ReadAll(req.Body)
		return newMockResponse(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`), nil
	})

	m, err := NewModel("MiniMax-M2.7", WithAPIKey("test-key"), WithHTTPClient(client))
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("Hi"),
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are helpful.", "user"),
		},
	}

	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
	}

	var sentReq openAIRequest
	if err := json.Unmarshal(capturedBody, &sentReq); err != nil {
		t.Fatalf("failed to parse sent request: %v", err)
	}
	if len(sentReq.Messages) == 0 || sentReq.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got: %+v", sentReq.Messages)
	}
}

func TestGenerateContent_WithTools(t *testing.T) {
	responseBody := `{
		"choices": [{
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_123",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"location\":\"Paris\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}]
	}`
	client := newTestClient(func(_ *http.Request) (*http.Response, error) {
		return newMockResponse(responseBody), nil
	})

	m, err := NewModel("MiniMax-M2.7", WithAPIKey("test-key"), WithHTTPClient(client))
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("What is the weather in Paris?"),
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get the current weather for a location.",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"location": {Type: genai.TypeString, Description: "The city name"},
						},
						Required: []string{"location"},
					},
				}},
			}},
		},
	}

	var got *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}

	if got == nil || got.Content == nil || len(got.Content.Parts) == 0 {
		t.Fatal("expected function call in response")
	}
	fc := got.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall in response part")
	}
	if fc.Name != "get_weather" {
		t.Errorf("FunctionCall.Name = %q, want %q", fc.Name, "get_weather")
	}
	if fc.Args["location"] != "Paris" {
		t.Errorf("FunctionCall.Args[location] = %v, want Paris", fc.Args["location"])
	}
}

func TestConvertRole(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"user", "user"},
		{"model", "assistant"},
		{"system", "system"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := convertRole(tc.in); got != tc.want {
			t.Errorf("convertRole(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestConvertContents_FunctionResponse(t *testing.T) {
	contents := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "Call the function"},
			},
		},
		{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_abc",
						Name: "my_func",
						Args: map[string]any{"x": 1.0},
					},
				},
			},
		},
		{
			Role: "user",
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "call_abc",
						Name:     "my_func",
						Response: map[string]any{"result": "done"},
					},
				},
			},
		},
	}

	msgs, err := convertContents(contents, nil)
	if err != nil {
		t.Fatalf("convertContents() unexpected error: %v", err)
	}

	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" {
		t.Errorf("msgs[0].Role = %q, want user", msgs[0].Role)
	}
	if msgs[1].Role != "assistant" {
		t.Errorf("msgs[1].Role = %q, want assistant", msgs[1].Role)
	}
	if len(msgs[1].ToolCalls) != 1 {
		t.Errorf("msgs[1].ToolCalls length = %d, want 1", len(msgs[1].ToolCalls))
	}
	if msgs[2].Role != "tool" {
		t.Errorf("msgs[2].Role = %q, want tool", msgs[2].Role)
	}
	if msgs[2].ToolCallID != "call_abc" {
		t.Errorf("msgs[2].ToolCallID = %q, want call_abc", msgs[2].ToolCallID)
	}
}

func TestConvertSchema(t *testing.T) {
	schema := &genai.Schema{
		Type:        genai.TypeObject,
		Description: "A test schema",
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString, Description: "Name field"},
			"age":  {Type: genai.TypeInteger},
		},
		Required: []string{"name"},
	}

	got := convertSchema(schema)

	if got.Type != "object" {
		t.Errorf("Type = %q, want object", got.Type)
	}
	if got.Description != "A test schema" {
		t.Errorf("Description = %q, want %q", got.Description, "A test schema")
	}
	if len(got.Properties) != 2 {
		t.Errorf("Properties count = %d, want 2", len(got.Properties))
	}
	if !cmp.Equal(got.Required, []string{"name"}) {
		t.Errorf("Required = %v, want [name]", got.Required)
	}
}

func TestConvertSchema_Nil(t *testing.T) {
	if got := convertSchema(nil); got != nil {
		t.Errorf("convertSchema(nil) = %v, want nil", got)
	}
}

func TestBuildRequest_TemperatureClamp(t *testing.T) {
	cases := []struct {
		name     string
		input    float32
		wantZero bool
	}{
		{"zero clamped to 1.0", 0.0, false},
		{"negative clamped to 1.0", -0.5, false},
		{"above max clamped to 1.0", 1.5, false},
		{"valid value preserved", 0.7, false},
	}

	m := &minimaxModel{name: "MiniMax-M2.7", apiKey: "key", baseURL: DefaultBaseURL, httpClient: http.DefaultClient}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			temp := tc.input
			req := &model.LLMRequest{
				Contents: genai.Text("hi"),
				Config: &genai.GenerateContentConfig{
					Temperature: &temp,
				},
			}
			apiReq, err := m.buildRequest(req, false)
			if err != nil {
				t.Fatalf("buildRequest() unexpected error: %v", err)
			}
			if apiReq.Temperature == nil {
				t.Fatal("Temperature should not be nil")
			}
			if *apiReq.Temperature <= 0 || *apiReq.Temperature > 1.0 {
				t.Errorf("Temperature = %f, must be in (0.0, 1.0]", *apiReq.Temperature)
			}
		})
	}
}

func TestGenerateContent_APIError(t *testing.T) {
	client := newTestClient(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       io.NopCloser(strings.NewReader(`{"error":"unauthorized"}`)),
		}, nil
	})

	m, err := NewModel("MiniMax-M2.7", WithAPIKey("bad-key"), WithHTTPClient(client))
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}

	req := &model.LLMRequest{Contents: genai.Text("Hi")}
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err == nil {
			t.Error("expected an error for 401 response")
		}
	}
}

func TestGenerateContent_CustomBaseURL(t *testing.T) {
	var capturedURL string
	client := newTestClient(func(req *http.Request) (*http.Response, error) {
		capturedURL = req.URL.String()
		return newMockResponse(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`), nil
	})

	m, err := NewModel("MiniMax-M2.7",
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.example.com/api"),
		WithHTTPClient(client),
	)
	if err != nil {
		t.Fatalf("NewModel() unexpected error: %v", err)
	}

	req := &model.LLMRequest{Contents: genai.Text("Hi")}
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
	}

	want := "https://custom.example.com/api/v1/chat/completions"
	if capturedURL != want {
		t.Errorf("URL = %q, want %q", capturedURL, want)
	}
}
