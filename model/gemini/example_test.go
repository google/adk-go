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

package gemini_test

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
)

// cannedTransport answers every HTTP request with the same JSON body, so the
// client never opens a connection.
type cannedTransport struct{ body string }

func (t cannedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(t.body)),
		Request:    req,
	}, nil
}

// A candidate that has no content parts and a finish reason other than STOP
// becomes an error response: ErrorCode is the finish reason and ErrorMessage
// the finish message. The response keeps the usage metadata but carries no
// content and no grounding metadata, even when the candidate had some.
//
// The example uses the Vertex AI backend because the genai client's Gemini
// API decoder does not copy finishMessage, so ErrorMessage stays empty there.
func ExampleNewModel_errorResponse() {
	const blocked = `{
	  "candidates": [{
	    "finishReason": "SAFETY",
	    "finishMessage": "Blocked by the safety filter.",
	    "groundingMetadata": {"webSearchQueries": ["weather in Paris"]}
	  }],
	  "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7}
	}`
	ctx := context.Background()
	llm, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    "my-project",
		Location:   "us-central1",
		HTTPClient: &http.Client{Transport: cannedTransport{body: blocked}},
	})
	if err != nil {
		log.Fatal(err)
	}
	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("Weather in Paris?", genai.RoleUser)}}
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("ErrorCode: %s\n", resp.ErrorCode)
		fmt.Printf("ErrorMessage: %s\n", resp.ErrorMessage)
		fmt.Printf("Content: %v, GroundingMetadata: %v\n", resp.Content, resp.GroundingMetadata)
		fmt.Printf("prompt tokens: %d\n", resp.UsageMetadata.PromptTokenCount)
	}
	// Output:
	// ErrorCode: SAFETY
	// ErrorMessage: Blocked by the safety filter.
	// Content: <nil>, GroundingMetadata: <nil>
	// prompt tokens: 7
}
