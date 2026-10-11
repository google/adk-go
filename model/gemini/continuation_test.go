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
// prompt and 2 candidate tokens. With always set, it answers every request
// with a CONTINUATION stop and a new token.
type fakeContinuationAPI struct {
	byToken map[string][]map[string]any
	always  bool
	// failures answers the first requests with a token with HTTP 503, as many
	// times as it says.
	failures map[string]int

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
	n := len(f.tokens)
	fail := f.failures[string(token)] > 0
	if fail {
		f.failures[string(token)]--
	}
	f.mu.Unlock()
	if fail {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, `{"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}`)
		return
	}

	chunks := f.byToken[string(token)]
	if f.always {
		chunks = []map[string]any{candidate("x", "CONTINUATION", fmt.Sprint("t", n))}
		if n > maxResumes+1 {
			// Finish, so that a model resuming past maxResumes fails the test
			// rather than hanging it.
			chunks = []map[string]any{candidate("x", "STOP", "")}
		}
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

// newContinuationModel returns the model on api, with retry, when set, as the
// client's retry options.
func newContinuationModel(t *testing.T, api *fakeContinuationAPI, retry ...*genai.HTTPRetryOptions) model.LLM {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(api.serve))
	t.Cleanup(srv.Close)
	cc := &genai.ClientConfig{
		APIKey:      "key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	}
	if len(retry) > 0 {
		cc.HTTPOptions.RetryOptions = retry[0]
	}
	m, err := NewModel(t.Context(), "m", cc)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m
}

const continuationPrompt = "Count to three."

func continuationRequest() *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText(continuationPrompt, genai.RoleUser)}}
}

// checkRequests checks the requests the model sent, in order: their
// continuation tokens, and the output so far that each resend carried after
// the original contents, "" for none.
func checkRequests(t *testing.T, api *fakeContinuationAPI, wantTokens, wantOutputSoFar []string) {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	if diff := cmp.Diff(wantTokens, api.tokens); diff != "" {
		t.Errorf("continuation tokens of the requests sent (-want +got):\n%s", diff)
	}
	var want []string
	for _, out := range wantOutputSoFar {
		c := "user: " + continuationPrompt
		if out != "" {
			c += " | model: " + out
		}
		want = append(want, c)
	}
	if diff := cmp.Diff(want, api.contents); diff != "" {
		t.Errorf("contents of the requests sent (-want +got):\n%s", diff)
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

func TestContinuation_Generate(t *testing.T) {
	tests := []struct {
		name            string
		byToken         map[string][]map[string]any
		want            *model.LLMResponse
		wantTokens      []string
		wantOutputSoFar []string
	}{
		{
			name: "resumed until finished",
			byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two ", "CONTINUATION", "t2")},
				"t2": {candidate("three", "STOP", "")},
			},
			want: &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one two three"}}},
				FinishReason:  genai.FinishReasonStop,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 30, CandidatesTokenCount: 6, TotalTokenCount: 36},
			},
			wantTokens:      []string{"", "t1", "t2"},
			wantOutputSoFar: []string{"", "one ", "one two "},
		},
		{
			name: "same token twice",
			byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two", "CONTINUATION", "t1")},
			},
			want: &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one two"}}},
				FinishReason:  genai.FinishReasonContinuation,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
			},
			wantTokens:      []string{"", "t1"},
			wantOutputSoFar: []string{"", "one "},
		},
		{
			// The empty pause is no error, since an earlier request produced
			// the output.
			name: "gave up on an empty pause",
			byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {pausedWith("t1")},
			},
			want: &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one "}}},
				FinishReason:  genai.FinishReasonContinuation,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
			},
			wantTokens:      []string{"", "t1"},
			wantOutputSoFar: []string{"", "one "},
		},
		{
			name:    "maxOutputTokens reached",
			byToken: map[string][]map[string]any{"": {candidate("one ", "MAX_TOKENS", "t1")}},
			want: &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one "}}},
				FinishReason:  genai.FinishReasonMaxTokens,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 2, TotalTokenCount: 12},
			},
			wantTokens:      []string{""},
			wantOutputSoFar: []string{""},
		},
		{
			name:    "paused without a token",
			byToken: map[string][]map[string]any{"": {candidate("one ", "CONTINUATION", "")}},
			want: &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one "}}},
				FinishReason:  genai.FinishReasonContinuation,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 2, TotalTokenCount: 12},
			},
			wantTokens:      []string{""},
			wantOutputSoFar: []string{""},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: tc.byToken}
			var got *model.LLMResponse
			for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), false) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
				got = resp
			}
			got = &model.LLMResponse{Content: got.Content, ErrorCode: got.ErrorCode, FinishReason: got.FinishReason, UsageMetadata: got.UsageMetadata, CustomMetadata: got.CustomMetadata}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("GenerateContent(stream=false) (-want +got):\n%s", diff)
			}
			checkRequests(t, api, tc.wantTokens, tc.wantOutputSoFar)
		})
	}
}

func TestContinuation_GenerateStream(t *testing.T) {
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
	// The chunk that is resumed must not complete the turn.
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
	got := &model.LLMResponse{Content: final.Content, FinishReason: final.FinishReason, UsageMetadata: final.UsageMetadata, CustomMetadata: final.CustomMetadata}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("final response (-want +got):\n%s", diff)
	}
	checkRequests(t, api, []string{"", "t1"}, []string{"", "one two "})
}

// TestContinuation_GenerateStreamGivesUp checks that a streamed generation the
// model keeps paused with the same token ends with CONTINUATION, since only a
// pause that is resumed has its finish reason cleared.
func TestContinuation_GenerateStreamGivesUp(t *testing.T) {
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {candidate("one ", "CONTINUATION", "t1")},
		"t1": {candidate("two", "CONTINUATION", "t1")},
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
	if final == nil || final.FinishReason != genai.FinishReasonContinuation {
		t.Fatalf("final response %+v, want one with finish reason CONTINUATION", final)
	}
	checkRequests(t, api, []string{"", "t1"}, []string{"", "one "})
}

// pausedWith returns a candidate that pauses with token after parts.
func pausedWith(token string, parts ...map[string]any) map[string]any {
	ps := make([]any, len(parts))
	for i, p := range parts {
		ps[i] = p
	}
	return map[string]any{
		"content":           map[string]any{"role": "model", "parts": ps},
		"finishReason":      "CONTINUATION",
		"continuationToken": base64.StdEncoding.EncodeToString([]byte(token)),
	}
}

// resentOutput returns the parts of the output so far that the second request
// carried, as JSON values.
func resentOutput(t *testing.T, api *fakeContinuationAPI) []any {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	if len(api.bodies) != 2 {
		t.Fatalf("sent %d requests, want 2", len(api.bodies))
	}
	var resend struct {
		Contents []struct {
			Parts []any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal([]byte(api.bodies[1]), &resend); err != nil || len(resend.Contents) != 2 {
		t.Fatalf("resend %s, want the original contents and the output so far", api.bodies[1])
	}
	return resend.Contents[1].Parts
}

// TestContinuation_ResendCopiesParts checks that a resend carries the parts as
// the model sent them, not as a consumer changed them in the meantime: the flow
// gives a function call an ID, and an after-model callback may edit its
// arguments or a part's data.
func TestContinuation_ResendCopiesParts(t *testing.T) {
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"": {pausedWith("t1",
			map[string]any{"functionCall": map[string]any{"name": "lookup", "args": map[string]any{
				"q": "a", "secret": "s", "filter": map[string]any{"region": "eu"},
			}}},
			map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": "AQID"}},
		)},
		"t1": {candidate("done", "STOP", "")},
	}}
	for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if resp.Content == nil {
			continue
		}
		for _, p := range resp.Content.Parts {
			if fc := p.FunctionCall; fc != nil {
				fc.ID = "client-id"
				delete(fc.Args, "secret")
				if filter, ok := fc.Args["filter"].(map[string]any); ok {
					filter["region"] = "us"
				}
			}
			if b := p.InlineData; b != nil && len(b.Data) > 0 {
				b.Data[0] = 9
			}
		}
	}
	want := []any{
		map[string]any{"functionCall": map[string]any{"name": "lookup", "args": map[string]any{
			"q": "a", "secret": "s", "filter": map[string]any{"region": "eu"},
		}}},
		map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": "AQID"}},
	}
	if diff := cmp.Diff(want, resentOutput(t, api)); diff != "" {
		t.Errorf("output so far in the resend (-want +got):\n%s", diff)
	}
}

// TestContinuation_PauseClosingParts checks the parts a paused stream can end
// with: the empty ones, which genai encodes as {} or {"thought": true} and the
// API rejects, are left out of the resend, and the signature-only one joins
// onto the text it closes, in the resend and in the final response of both
// modes.
func TestContinuation_PauseClosingParts(t *testing.T) {
	sig := base64.StdEncoding.EncodeToString([]byte("sig"))
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"": {pausedWith("t1",
					map[string]any{"text": "one "},
					map[string]any{"text": "", "thoughtSignature": sig},
					map[string]any{"text": ""},
					map[string]any{"text": "", "thought": true},
				)},
				"t1": {candidate("two", "STOP", "")},
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
			want := []any{map[string]any{"text": "one ", "thoughtSignature": sig}}
			if diff := cmp.Diff(want, resentOutput(t, api)); diff != "" {
				t.Errorf("output so far in the resend (-want +got):\n%s", diff)
			}
			if final == nil {
				t.Fatal("no final response")
			}
			wantContent := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
				{Text: "one two", ThoughtSignature: []byte("sig")},
			}}
			if diff := cmp.Diff(wantContent, final.Content); diff != "" {
				t.Errorf("final content (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContinuation_LeavesRequestUnchanged checks that resuming a generation
// does not write the token or the resend's retry options into the caller's
// request.
func TestContinuation_LeavesRequestUnchanged(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {candidate("two", "STOP", "")},
			}}
			req := continuationRequest()
			req.Config = &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{}}
			for _, err := range newContinuationModel(t, api).GenerateContent(t.Context(), req, stream) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
			}
			if diff := cmp.Diff(continuationRequest().Contents, req.Contents); diff != "" {
				t.Errorf("request contents after the call (-want +got):\n%s", diff)
			}
			if req.Config.ContinuationToken != nil {
				t.Errorf("request config has continuation token %q after the call, want none", req.Config.ContinuationToken)
			}
			if req.Config.HTTPOptions.RetryOptions != nil {
				t.Errorf("request config has retry options %+v after the call, want none", req.Config.HTTPOptions.RetryOptions)
			}
			checkRequests(t, api, []string{"", "t1"}, []string{"", "one "})
		})
	}
}

// TestContinuation_BlockedAfterResume checks that a resumed generation that
// ends blocked reports the block as its finish reason, with the output
// generated before it, and the error code a single response with that output
// gets: none in unary mode, where only a response without content gets one,
// and the last chunk's in streaming mode, as the aggregator gives any stream.
func TestContinuation_BlockedAfterResume(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {candidate("one ", "CONTINUATION", "t1")},
				"t1": {{"finishReason": "SAFETY"}},
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
			want := &model.LLMResponse{
				Content:       &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "one "}}},
				FinishReason:  genai.FinishReasonSafety,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
			}
			if stream {
				want.ErrorCode = string(genai.FinishReasonSafety)
			}
			got := &model.LLMResponse{Content: final.Content, ErrorCode: final.ErrorCode, FinishReason: final.FinishReason, UsageMetadata: final.UsageMetadata}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("final response (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContinuation_KeepsMetadata checks that a resumed generation keeps the
// last grounding and citation metadata a request reported, though the last
// request reports none.
func TestContinuation_KeepsMetadata(t *testing.T) {
	withMetadata := func(c map[string]any, query, uri string) map[string]any {
		if query != "" {
			c["groundingMetadata"] = map[string]any{"webSearchQueries": []any{query}}
		}
		if uri != "" {
			c["citationMetadata"] = map[string]any{"citationSources": []any{map[string]any{"uri": uri}}}
		}
		return c
	}
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
				"":   {withMetadata(candidate("one ", "CONTINUATION", "t1"), "q1", "https://example.com/a")},
				"t1": {withMetadata(candidate("two ", "CONTINUATION", "t2"), "q2", "")},
				"t2": {candidate("three", "STOP", "")},
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
			if diff := cmp.Diff(&genai.GroundingMetadata{WebSearchQueries: []string{"q2"}}, final.GroundingMetadata); diff != "" {
				t.Errorf("final grounding metadata (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(&genai.CitationMetadata{Citations: []*genai.Citation{{URI: "https://example.com/a"}}}, final.CitationMetadata); diff != "" {
				t.Errorf("final citation metadata (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAppendParts(t *testing.T) {
	sig1, sig2 := []byte("sig1"), []byte("sig2")
	call := &genai.FunctionCall{Name: "lookup"}
	tests := []struct {
		name        string
		parts, next []*genai.Part
		want        []*genai.Part
	}{
		{
			name:  "text joined across calls",
			parts: []*genai.Part{{Text: "a"}},
			next:  []*genai.Part{{Text: "b"}, {Text: "c"}},
			want:  []*genai.Part{{Text: "abc"}},
		},
		{
			name: "thoughts and answer kept apart",
			next: []*genai.Part{{Text: "a", Thought: true}, {Text: "b", Thought: true}, {Text: "c"}, {Text: "d", Thought: true}},
			want: []*genai.Part{{Text: "ab", Thought: true}, {Text: "c"}, {Text: "d", Thought: true}},
		},
		{
			name: "later signature wins",
			next: []*genai.Part{{Text: "a", ThoughtSignature: sig1}, {Text: "b", ThoughtSignature: sig2}},
			want: []*genai.Part{{Text: "ab", ThoughtSignature: sig2}},
		},
		{
			name: "earlier signature kept when the later part has none",
			next: []*genai.Part{{Text: "a", ThoughtSignature: sig1}, {Text: "b"}},
			want: []*genai.Part{{Text: "ab", ThoughtSignature: sig1}},
		},
		{
			name: "signature part closing each pause joins onto its text",
			next: []*genai.Part{
				{Text: "a"},
				{ThoughtSignature: sig1},
				{Text: "b"},
				{ThoughtSignature: sig2},
				{Text: "c"},
			},
			want: []*genai.Part{{Text: "abc", ThoughtSignature: sig2}},
		},
		{
			name: "signature part joins onto thoughts too",
			next: []*genai.Part{{Text: "a", Thought: true}, {Thought: true, ThoughtSignature: sig1}, {Text: "b"}},
			want: []*genai.Part{{Text: "a", Thought: true, ThoughtSignature: sig1}, {Text: "b"}},
		},
		{
			name: "function call not joined, signed or not",
			next: []*genai.Part{{Text: "a"}, {FunctionCall: call}, {Text: "b"}, {FunctionCall: call, ThoughtSignature: sig1}},
			want: []*genai.Part{{Text: "a"}, {FunctionCall: call}, {Text: "b"}, {FunctionCall: call, ThoughtSignature: sig1}},
		},
		{
			name: "signature part after no text kept",
			next: []*genai.Part{{FunctionCall: call}, {ThoughtSignature: sig1}, {Text: "a"}},
			want: []*genai.Part{{FunctionCall: call}, {ThoughtSignature: sig1}, {Text: "a"}},
		},
		{
			name: "empty parts dropped",
			next: []*genai.Part{nil, {}, {Thought: true}, {Text: "a"}, {}, {Text: "b"}},
			want: []*genai.Part{{Text: "ab"}},
		},
		{
			// Decoded from {"text":"","thoughtSignature":""}, which would
			// otherwise be resent as {}.
			name: "part with an empty signature dropped",
			next: []*genai.Part{{ThoughtSignature: []byte{}}, {Text: "a"}, {ThoughtSignature: []byte{}}, {Text: "b"}},
			want: []*genai.Part{{Text: "ab"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, appendParts(tc.parts, tc.next)); diff != "" {
				t.Errorf("appendParts() (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAddUsage(t *testing.T) {
	text := func(n int32) *genai.ModalityTokenCount {
		return &genai.ModalityTokenCount{Modality: genai.MediaModalityText, TokenCount: n}
	}
	image := func(n int32) *genai.ModalityTokenCount {
		return &genai.ModalityTokenCount{Modality: genai.MediaModalityImage, TokenCount: n}
	}
	total := &genai.GenerateContentResponseUsageMetadata{
		CacheTokensDetails:         []*genai.ModalityTokenCount{text(1)},
		CachedContentTokenCount:    1,
		CandidatesTokenCount:       2,
		CandidatesTokensDetails:    []*genai.ModalityTokenCount{text(2)},
		PromptTokenCount:           10,
		PromptTokensDetails:        []*genai.ModalityTokenCount{text(8), image(2)},
		ThoughtsTokenCount:         3,
		ToolUsePromptTokenCount:    4,
		ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{text(4)},
		TotalTokenCount:            19,
		TrafficType:                genai.TrafficTypeOnDemand,
	}
	next := &genai.GenerateContentResponseUsageMetadata{
		CacheTokensDetails:         []*genai.ModalityTokenCount{image(5)},
		CachedContentTokenCount:    5,
		CandidatesTokenCount:       6,
		CandidatesTokensDetails:    []*genai.ModalityTokenCount{text(6)},
		PromptTokenCount:           20,
		PromptTokensDetails:        []*genai.ModalityTokenCount{image(5), text(15)},
		ThoughtsTokenCount:         7,
		ToolUsePromptTokenCount:    8,
		ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{text(8)},
		TotalTokenCount:            41,
		TrafficType:                genai.TrafficTypeProvisionedThroughput,
	}
	want := &genai.GenerateContentResponseUsageMetadata{
		CacheTokensDetails:         []*genai.ModalityTokenCount{text(1), image(5)},
		CachedContentTokenCount:    6,
		CandidatesTokenCount:       8,
		CandidatesTokensDetails:    []*genai.ModalityTokenCount{text(8)},
		PromptTokenCount:           30,
		PromptTokensDetails:        []*genai.ModalityTokenCount{text(23), image(7)},
		ThoughtsTokenCount:         10,
		ToolUsePromptTokenCount:    12,
		ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{text(12)},
		TotalTokenCount:            60,
		TrafficType:                genai.TrafficTypeProvisionedThroughput,
	}
	if diff := cmp.Diff(want, addUsage(total, next)); diff != "" {
		t.Errorf("addUsage() (-want +got):\n%s", diff)
	}
	if got := addUsage(nil, next); got != next {
		t.Errorf("addUsage(nil, next) = %+v, want next", got)
	}
	if got := addUsage(total, nil); got != total {
		t.Errorf("addUsage(total, nil) = %+v, want total", got)
	}
}

func TestContinuation_StopsAtMaxResumes(t *testing.T) {
	api := &fakeContinuationAPI{always: true}
	var got *model.LLMResponse
	for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		got = resp
	}
	if n := len(api.tokens); n != maxResumes+1 {
		t.Errorf("sent %d requests, want %d", n, maxResumes+1)
	}
	if got.FinishReason != genai.FinishReasonContinuation {
		t.Errorf("FinishReason = %q, want CONTINUATION, since the generation was left paused", got.FinishReason)
	}
}

// fastContinuationRetry shortens the backoff between retries for a test.
func fastContinuationRetry(t *testing.T) {
	t.Helper()
	saved := continuationRetry
	continuationRetry = &genai.HTTPRetryOptions{
		Attempts:     saved.Attempts,
		InitialDelay: genai.Ptr(0.001),
		MaxDelay:     genai.Ptr(0.001),
	}
	t.Cleanup(func() { continuationRetry = saved })
}

// TestContinuation_RetriesResumingRequest checks that a request resuming a
// generation is retried after a transient failure, without starting the
// generation over, and that the first request is not.
func TestContinuation_RetriesResumingRequest(t *testing.T) {
	fastContinuationRetry(t)
	chunks := map[string][]map[string]any{
		"":   {candidate("one ", "CONTINUATION", "t1")},
		"t1": {candidate("two", "STOP", "")},
	}
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			api := &fakeContinuationAPI{byToken: chunks, failures: map[string]int{"t1": 2}}
			var text string
			var final *model.LLMResponse
			for resp, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), stream) {
				if err != nil {
					t.Fatalf("GenerateContent: %v", err)
				}
				if !resp.Partial {
					final = resp
				}
			}
			if final != nil && final.Content != nil {
				for _, p := range final.Content.Parts {
					text += p.Text
				}
			}
			if text != "one two" {
				t.Errorf("final text = %q, want %q", text, "one two")
			}
			checkRequests(t, api, []string{"", "t1", "t1", "t1"}, []string{"", "one ", "one ", "one "})
		})
	}

	t.Run("first request", func(t *testing.T) {
		api := &fakeContinuationAPI{byToken: chunks, failures: map[string]int{"": 1}}
		for _, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), false) {
			if err == nil {
				t.Error("GenerateContent succeeded after a 503 on the first request, want the error")
			}
		}
		checkRequests(t, api, []string{""}, []string{""})
	})

	t.Run("gives up", func(t *testing.T) {
		api := &fakeContinuationAPI{byToken: chunks, failures: map[string]int{"t1": 100}}
		var gotErr error
		for _, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), false) {
			gotErr = err
		}
		if gotErr == nil {
			t.Error("GenerateContent succeeded with every retry failing, want the error")
		}
		if n := len(api.tokens); n != 1+int(*continuationRetry.Attempts) {
			t.Errorf("sent %d requests, want 1 plus %d attempts", n, *continuationRetry.Attempts)
		}
	})

	t.Run("client's retry options", func(t *testing.T) {
		api := &fakeContinuationAPI{byToken: chunks, failures: map[string]int{"t1": 1}}
		var gotErr error
		for _, err := range newContinuationModel(t, api, &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](1)}).GenerateContent(t.Context(), continuationRequest(), false) {
			gotErr = err
		}
		if gotErr == nil {
			t.Error("GenerateContent succeeded after a 503 with the client allowing one attempt, want the error")
		}
		checkRequests(t, api, []string{"", "t1"}, []string{"", "one "})
	})

	t.Run("caller's retry options", func(t *testing.T) {
		api := &fakeContinuationAPI{byToken: chunks, failures: map[string]int{"t1": 1}}
		req := continuationRequest()
		req.Config = &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](1)}}}
		var gotErr error
		for _, err := range newContinuationModel(t, api).GenerateContent(t.Context(), req, false) {
			gotErr = err
		}
		if gotErr == nil {
			t.Error("GenerateContent succeeded after a 503 with the caller allowing one attempt, want the error")
		}
		checkRequests(t, api, []string{"", "t1"}, []string{"", "one "})
	})
}
