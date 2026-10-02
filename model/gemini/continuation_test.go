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
			got = &model.LLMResponse{Content: got.Content, FinishReason: got.FinishReason, UsageMetadata: got.UsageMetadata, CustomMetadata: got.CustomMetadata}
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

// TestContinuation_ResendCopiesParts checks that a resend carries the parts as
// the model sent them, not as a consumer changed them in the meantime, as the
// flow does when it gives a function call an ID.
func TestContinuation_ResendCopiesParts(t *testing.T) {
	call := map[string]any{"content": map[string]any{"role": "model", "parts": []any{
		map[string]any{"functionCall": map[string]any{"name": "lookup", "args": map[string]any{}}},
	}}, "finishReason": "CONTINUATION", "continuationToken": base64.StdEncoding.EncodeToString([]byte("t1"))}
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {call},
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
			if p.FunctionCall != nil {
				p.FunctionCall.ID = "client-id"
			}
		}
	}
	api.mu.Lock()
	defer api.mu.Unlock()
	if len(api.bodies) != 2 {
		t.Fatalf("sent %d requests, want 2", len(api.bodies))
	}
	if !strings.Contains(api.bodies[1], `"lookup"`) || strings.Contains(api.bodies[1], "client-id") {
		t.Errorf("the resend carried %s, want the function call without the ID set after it arrived", api.bodies[1])
	}
}

// TestContinuation_ResendDropsEmptyParts checks that a resend leaves out a part
// with no text and no data but a thought signature, such as the one a stream
// can end with: genai encodes it as a part with no data, which the API rejects.
func TestContinuation_ResendDropsEmptyParts(t *testing.T) {
	end := map[string]any{"content": map[string]any{"role": "model", "parts": []any{
		map[string]any{"thoughtSignature": base64.StdEncoding.EncodeToString([]byte("sig"))},
	}}, "finishReason": "CONTINUATION", "continuationToken": base64.StdEncoding.EncodeToString([]byte("t1"))}
	api := &fakeContinuationAPI{byToken: map[string][]map[string]any{
		"":   {candidate("one ", "", ""), end},
		"t1": {candidate("two", "STOP", "")},
	}}
	for _, err := range newContinuationModel(t, api).GenerateContent(t.Context(), continuationRequest(), true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	api.mu.Lock()
	defer api.mu.Unlock()
	if len(api.bodies) != 2 {
		t.Fatalf("sent %d requests, want 2", len(api.bodies))
	}
	var resend struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal([]byte(api.bodies[1]), &resend); err != nil || len(resend.Contents) != 2 {
		t.Fatalf("resend %s, want the original contents and the output so far", api.bodies[1])
	}
	for _, p := range resend.Contents[1].Parts {
		if p["text"] == nil {
			t.Errorf("the resend's output so far has part %v, want only text parts", p)
		}
	}
}

// TestContinuation_BlockedAfterResume checks that a resumed generation that
// ends blocked reports the block reason as its error code, with the output
// generated before it.
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
				ErrorCode:     string(genai.FinishReasonSafety),
				FinishReason:  genai.FinishReasonSafety,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 20, CandidatesTokenCount: 4, TotalTokenCount: 24},
			}
			got := &model.LLMResponse{Content: final.Content, ErrorCode: final.ErrorCode, FinishReason: final.FinishReason, UsageMetadata: final.UsageMetadata}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("final response (-want +got):\n%s", diff)
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
			name: "first signature kept",
			next: []*genai.Part{{Text: "a", ThoughtSignature: sig1}, {Text: "b", ThoughtSignature: sig2}},
			want: []*genai.Part{{Text: "ab", ThoughtSignature: sig1}},
		},
		{
			name: "later signature taken when the first part has none",
			next: []*genai.Part{{Text: "a"}, {Text: "b", ThoughtSignature: sig2}},
			want: []*genai.Part{{Text: "ab", ThoughtSignature: sig2}},
		},
		{
			name: "function call not joined",
			next: []*genai.Part{{Text: "a"}, {FunctionCall: call}, {Text: "b"}},
			want: []*genai.Part{{Text: "a"}, {FunctionCall: call}, {Text: "b"}},
		},
		{
			name: "empty parts dropped",
			next: []*genai.Part{nil, {}, {Thought: true}, {ThoughtSignature: sig1}, {Text: "a"}},
			want: []*genai.Part{{Text: "a"}},
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
