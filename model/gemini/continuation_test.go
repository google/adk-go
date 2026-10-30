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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// fakeContinuationAPI answers each request by its continuation token, "" for
// the first request: generateContent with the token's first candidate,
// streamGenerateContent with all of them, each response reporting usage of 10
// prompt and 2 candidate tokens.
type fakeContinuationAPI struct {
	byToken map[string][]map[string]any

	mu       sync.Mutex
	tokens   []string
	contents []string
	bodies   []string
}

func (f *fakeContinuationAPI) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var req struct {
		ContinuationToken string `json:"continuationToken"`
		Contents          []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"contents"`
	}
	_ = json.Unmarshal(body, &req)
	token, _ := base64.StdEncoding.DecodeString(req.ContinuationToken)
	var contents []string
	for _, c := range req.Contents {
		var text strings.Builder
		for _, p := range c.Parts {
			text.WriteString(p.Text)
		}
		contents = append(contents, c.Role+": "+text.String())
	}
	f.mu.Lock()
	f.tokens = append(f.tokens, string(token))
	f.contents = append(f.contents, strings.Join(contents, " | "))
	f.bodies = append(f.bodies, string(body))
	f.mu.Unlock()

	chunks := f.byToken[string(token)]
	if len(chunks) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error": {"code": 400, "message": "unknown continuation token", "status": "INVALID_ARGUMENT"}}`)
		return
	}
	usage := map[string]any{"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}
	if strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
		w.Header().Set("Content-Type", "text/event-stream")
		for i, c := range chunks {
			resp := map[string]any{"candidates": []any{c}}
			if i == len(chunks)-1 {
				resp["usageMetadata"] = usage
			}
			data, _ := json.Marshal(resp)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		}
		return
	}
	data, _ := json.Marshal(map[string]any{"candidates": []any{chunks[0]}, "usageMetadata": usage})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

func candidate(text, finishReason, token string) map[string]any {
	c := map[string]any{"content": map[string]any{"role": "model", "parts": []any{map[string]any{"text": text}}}}
	if finishReason != "" {
		c["finishReason"] = finishReason
	}
	if token != "" {
		c["continuationToken"] = base64.StdEncoding.EncodeToString([]byte(token))
	}
	return c
}

// newContinuationModel returns the model on api.
func newContinuationModel(t *testing.T, api *fakeContinuationAPI) model.LLM {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(api.serve))
	t.Cleanup(srv.Close)
	m, err := NewModel(t.Context(), "m", &genai.ClientConfig{
		APIKey:      "key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m
}

const continuationPrompt = "Count to three."

func continuationRequest() *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText(continuationPrompt, genai.RoleUser)}}
}

// checkRequests checks the continuation tokens of the requests the model
// sent, in order, and that each carried the original contents unchanged.
func checkRequests(t *testing.T, api *fakeContinuationAPI, wantTokens []string) {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	if diff := cmp.Diff(wantTokens, api.tokens); diff != "" {
		t.Errorf("continuation tokens of the requests sent (-want +got):\n%s", diff)
	}
	for i, c := range api.contents {
		if want := "user: " + continuationPrompt; c != want {
			t.Errorf("request %d contents = %q, want %q", i, c, want)
		}
	}
}

func TestContinuation_PreservesClientProviderOnEveryRequest(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two", "STOP", "")},
			}}
			srv := httptest.NewServer(http.HandlerFunc(api.serve))
			t.Cleanup(srv.Close)
			providerCalls := 0
			llm, err := NewModel(t.Context(), "m", &genai.ClientConfig{
				APIKey:  "key",
				Backend: genai.BackendGeminiAPI,
				HTTPOptions: genai.HTTPOptions{
					BaseURL: srv.URL,
					ExtrasRequestProvider: func(body map[string]any) map[string]any {
						providerCalls++
						body["clientMarker"] = "preserved"
						contents := mapsFromSlice(body["contents"])
						parts := mapsFromSlice(contents[0]["parts"])
						contents[0]["parts"] = append(parts, map[string]any{"thoughtSignature": "provider-signature"})
						return body
					},
				},
			})
			if err != nil {
				t.Fatalf("NewModel: %v", err)
			}
			for _, err := range llm.GenerateContent(t.Context(), continuationRequest(), stream) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
			}

			api.mu.Lock()
			bodies := append([]string(nil), api.bodies...)
			api.mu.Unlock()
			if got, want := providerCalls, 2; got != want {
				t.Fatalf("client provider calls = %d, want %d", got, want)
			}
			if got, want := len(bodies), 2; got != want {
				t.Fatalf("request count = %d, want %d", got, want)
			}
			for i, requestBody := range bodies {
				var body map[string]any
				if err := json.Unmarshal([]byte(requestBody), &body); err != nil {
					t.Fatalf("request %d: json.Unmarshal() error = %v", i, err)
				}
				if got := body["clientMarker"]; got != "preserved" {
					t.Errorf("request %d: client provider marker = %#v, want preserved", i, got)
				}
				contents := mapsFromSlice(body["contents"])
				parts := mapsFromSlice(contents[0]["parts"])
				providerPart := parts[len(parts)-1]
				if got, ok := providerPart["text"]; !ok || got != "" {
					t.Errorf("request %d: provider part text = %#v, present = %t; want present empty text", i, got, ok)
				}
			}
		})
	}
}

// TestContinuation_GenerateResumedBySDK checks that a generation the model
// pauses with CONTINUATION is resumed by the genai SDK's automatic
// continuation, which resends the original contents with the token and
// returns the responses merged into one.
func TestContinuation_GenerateResumedBySDK(t *testing.T) {
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {candidate("one ", "CONTINUATION", "t1")},
		"t1": {candidate("two", "STOP", "")},
	}}
	var got *model.LLMResponse
	for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		got = resp
	}
	want := &model.LLMResponse{
		Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one "}, {Text: "two"}}},
		FinishReason:  genai.FinishReasonStop,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
	}
	if diff := cmp.Diff(want, &model.LLMResponse{Content: got.Content, FinishReason: got.FinishReason, UsageMetadata: got.UsageMetadata}); diff != "" {
		t.Errorf("response (-want +got):\n%s", diff)
	}
	checkRequests(t, api, []string{"", "t1"})
}

// TestContinuation_GenerateStreamResumedBySDK checks that a streamed
// generation the model pauses with CONTINUATION goes on in the same stream,
// without the paused chunk completing the turn.
func TestContinuation_GenerateStreamResumedBySDK(t *testing.T) {
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {candidate("one ", "", ""), candidate("two ", "CONTINUATION", "t1")},
		"t1": {candidate("three", "STOP", "")},
	}}
	var partials []string
	var final *model.LLMResponse
	for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
			continue
		}
		desc := resp.Content.Parts[0].Text
		if resp.TurnComplete {
			desc += " (turn complete)"
		}
		partials = append(partials, desc)
	}
	if diff := cmp.Diff([]string{"one ", "two ", "three (turn complete)"}, partials); diff != "" {
		t.Errorf("partial responses (-want +got):\n%s", diff)
	}
	if final == nil {
		t.Fatal("no final response")
	}
	want := &model.LLMResponse{
		Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one two three"}}},
		FinishReason:  genai.FinishReasonStop,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
	}
	if diff := cmp.Diff(want, &model.LLMResponse{Content: final.Content, FinishReason: final.FinishReason, UsageMetadata: final.UsageMetadata}); diff != "" {
		t.Errorf("final response (-want +got):\n%s", diff)
	}
	checkRequests(t, api, []string{"", "t1"})
}

// TestContinuation_AutomaticContinuationOff checks that a request that turns
// off automatic continuation reaches the SDK with it off, so a pause comes
// back as it is.
func TestContinuation_AutomaticContinuationOff(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two", "STOP", "")},
			}}
			req := continuationRequest()
			req.Config = &genai.GenerateContentConfig{AutomaticContinuation: genai.Ptr(false)}
			var final *model.LLMResponse
			for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), req, stream) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
				if !resp.Partial {
					final = resp
				}
			}
			if final == nil {
				t.Fatal("no final response")
			}
			if final.FinishReason != genai.FinishReasonContinuation {
				t.Errorf("finish reason = %q, want %q", final.FinishReason, genai.FinishReasonContinuation)
			}
			checkRequests(t, api, []string{""})
		})
	}
}

// TestContinuation_GenerateStreamKeepsMetadata checks that the final response
// of a resumed stream keeps the last grounding and citation metadata any
// chunk reported, even when the last request reports none.
func TestContinuation_GenerateStreamKeepsMetadata(t *testing.T) {
	withMetadata := func(c map[string]any, query, uri string) map[string]any {
		if query != "" {
			c["groundingMetadata"] = map[string]any{"webSearchQueries": []any{query}}
		}
		if uri != "" {
			c["citationMetadata"] = map[string]any{"citationSources": []any{map[string]any{"uri": uri}}}
		}
		return c
	}
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {withMetadata(candidate("one ", "CONTINUATION", "t1"), "q1", "https://example.com/a")},
		"t1": {withMetadata(candidate("two ", "CONTINUATION", "t2"), "q2", "https://example.com/b")},
		"t2": {candidate("three", "STOP", "")},
	}}
	var final *model.LLMResponse
	for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if !resp.Partial {
			final = resp
		}
	}
	if final == nil {
		t.Fatal("no final response")
	}
	if diff := cmp.Diff(&genai.GroundingMetadata{WebSearchQueries: []string{"q2"}}, final.GroundingMetadata); diff != "" {
		t.Errorf("final grounding metadata (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(&genai.CitationMetadata{Citations: []*genai.Citation{{URI: "https://example.com/b"}}}, final.CitationMetadata); diff != "" {
		t.Errorf("final citation metadata (-want +got):\n%s", diff)
	}
	checkRequests(t, api, []string{"", "t1", "t2"})
}

// TestContinuation_MaxTokensNotResumed checks that a MAX_TOKENS stop that
// carries a token is returned as it is, even after a CONTINUATION pause was
// resumed, since resuming it makes the API return a 400.
func TestContinuation_MaxTokensNotResumed(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two", "MAX_TOKENS", "t2")},
			}}
			var final *model.LLMResponse
			for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), stream) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
				if !resp.Partial {
					final = resp
				}
			}
			if final == nil {
				t.Fatal("no final response")
			}
			if final.FinishReason != genai.FinishReasonMaxTokens {
				t.Errorf("finish reason = %q, want %q", final.FinishReason, genai.FinishReasonMaxTokens)
			}
			checkRequests(t, api, []string{"", "t1"})
		})
	}
}
