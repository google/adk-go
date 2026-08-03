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

package openaimodel

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestModel_Generate(t *testing.T) {
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprint(w, `{"id":"resp_123","model":"test-model","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`); err != nil {
			t.Errorf("failed to write mock response: %v", err)
		}
	}))
	defer server.Close()

	clientCfg := &ClientConfig{
		APIKey:     "test",
		BaseURL:    server.URL + "/v1",
		HTTPClient: server.Client(),
	}

	ctx := t.Context()
	llm, err := NewModel(ctx, openai.ChatModelGPT4oMini, clientCfg)
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("World?", genai.RoleUser)},
	}
	var text string
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			text += resp.Content.Parts[0].Text
		}
	}
	if diff := cmp.Diff("hello", text); diff != "" {
		t.Fatalf("response text mismatch (-want +got):\n%s", diff)
	}
}

func TestModel_GenerateStream_Metadata(t *testing.T) {
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		events := []string{
			`{"type": "response.created", "response": {"id": "resp_stream_123", "model": "stream-model"}}`,
			`{"type": "response.output_text.delta", "delta": "chunk1"}`,
			`{"type": "response.completed", "response": {"id": "resp_stream_123", "model": "stream-model", "usage": {"total_tokens": 10}}}`,
			`[DONE]`,
		}

		for _, evt := range events {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", evt)
		}
	}))
	defer server.Close()

	clientCfg := &ClientConfig{
		APIKey:     "test",
		BaseURL:    server.URL + "/v1",
		HTTPClient: server.Client(),
	}

	ctx := t.Context()
	llm, err := NewModel(ctx, openai.ChatModelGPT4oMini, clientCfg)
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("World?", genai.RoleUser)},
	}

	var chunks int
	var finalResp *model.LLMResponse
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() stream err = %v", err)
		}
		chunks++
		if resp.CustomMetadata["openai_response_id"] != "resp_stream_123" {
			t.Errorf("expected chunk to have openai_response_id='resp_stream_123', got %v", resp.CustomMetadata["openai_response_id"])
		}
		if resp.CustomMetadata["openai_model"] != "stream-model" {
			t.Errorf("expected chunk to have openai_model='stream-model', got %v", resp.CustomMetadata["openai_model"])
		}
		finalResp = resp
	}

	// Expect the partial chunk and the final aggregated response
	if chunks != 2 {
		t.Errorf("expected 2 chunks from stream, got %d", chunks)
	}
	if finalResp == nil || finalResp.UsageMetadata == nil {
		t.Fatal("expected final stream response to have UsageMetadata, got nil")
	}
	if finalResp.UsageMetadata.TotalTokenCount != 10 {
		t.Errorf("expected final UsageMetadata.TotalTokenCount=10, got %d", finalResp.UsageMetadata.TotalTokenCount)
	}
}

// newLocalhostServer starts httptest.Server bound to IPv4 loopback since some sandboxes forbid IPv6 listeners.
func newLocalhostServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(handler)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on IPv4 loopback: %v", err)
	}
	server.Listener = ln
	server.Start()
	return server
}

func TestModel_ValidateModelNameInput(t *testing.T) {
	clientCfg := ClientConfig{APIKey: "test"}
	_, err := NewModel(t.Context(), "", &clientCfg)
	if !errors.Is(err, ErrModelNameRequired) {
		t.Fatalf("NewModel() err = %v, want %v", err, ErrModelNameRequired)
	}
}
