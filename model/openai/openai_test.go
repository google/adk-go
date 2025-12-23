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
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestModel_Generate(t *testing.T) {
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprint(w, `{"id":"resp_123","model":"test-model","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`); err != nil {
			t.Fatalf("failed to write mock response: %v", err)
		}
	}))
	defer server.Close()

	client := openai.NewClient(
		option.WithAPIKey("test"),
		option.WithHTTPClient(server.Client()),
		option.WithBaseURL(server.URL+"/v1"),
	)

	llm, err := NewModel(t.Context(), openai.ChatModelGPT4oMini, client)
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("World?", genai.RoleUser)},
	}
	var text string
	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
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

func TestModel_ValidateInputs(t *testing.T) {
	t.Parallel()
	client := openai.NewClient(option.WithAPIKey("test"))

	tests := []struct {
		name      string
		modelName openai.ChatModel
		client    openai.Client
		wantErr   error
	}{
		{
			name:      "missing model name",
			modelName: "",
			client:    client,
			wantErr:   ErrModelNameRequired,
		},
		{
			name:      "missing client",
			modelName: openai.ChatModelGPT4o,
			client:    openai.Client{},
			wantErr:   ErrClientRequired,
		},
	}

	ctx := context.Background()

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewModel(ctx, tt.modelName, tt.client)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("NewModel() err = %v, want %v", err, tt.wantErr)
			}
		})
	}
}
