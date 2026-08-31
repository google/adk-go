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
	"strings"
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
		text += allText(resp.Content)
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

// Synthetic Responses-API stream events shared by the streaming tests, carrying
// only the fields those tests assert on.
const (
	evCreated   = `{"type":"response.created","response":{"id":"resp_1","model":"stream-model"}}`
	evDelta1    = `{"type":"response.output_text.delta","delta":"hel"}`
	evDelta2    = `{"type":"response.output_text.delta","delta":"lo"}`
	evCompleted = `{"type":"response.completed","response":` + bodyCompleted + `}`
	evMaxTokens = `{"type":"response.incomplete","response":{"id":"resp_1","model":"stream-model","incomplete_details":{"reason":"max_output_tokens"}}}`
	evFiltered  = `{"type":"response.incomplete","response":{"id":"resp_1","model":"stream-model","incomplete_details":{"reason":"content_filter"}}}`
	evFailed    = `{"type":"response.failed","response":{"id":"resp_1","model":"stream-model","error":{"message":"upstream exploded"}}}`

	// The blocking-mode body carrying exactly the output evCompleted does, so
	// the two paths can be compared on the same model output.
	bodyCompleted = `{"id":"resp_1","model":"stream-model","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7},"output":[{"type":"message","content":[{"type":"output_text","text":"hello","logprobs":[{"token":"hello","logprob":-0.5,"top_logprobs":[{"token":"hello","logprob":-0.5}]}]}]}]}`
)

// runStream drives the model over a synthetic SSE stream and collects everything
// it emits, so a test can assert on the shape of the whole turn.
func runStream(t *testing.T, events ...string) ([]*model.LLMResponse, error) {
	t.Helper()
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, evt := range append(events, "[DONE]") {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", evt)
		}
	}))
	defer server.Close()
	return collectResponses(t, server, true)
}

// runBlocking is runStream's non-streaming counterpart, for parity assertions.
func runBlocking(t *testing.T, body string) ([]*model.LLMResponse, error) {
	t.Helper()
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, body)
	}))
	defer server.Close()
	return collectResponses(t, server, false)
}

func collectResponses(t *testing.T, server *httptest.Server, stream bool) ([]*model.LLMResponse, error) {
	t.Helper()
	ctx := t.Context()
	llm, err := NewModel(ctx, openai.ChatModelGPT4oMini, &ClientConfig{
		APIKey:     "test",
		BaseURL:    server.URL + "/v1",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("World?", genai.RoleUser)},
	}
	var got []*model.LLMResponse
	var firstErr error
	for resp, err := range llm.GenerateContent(ctx, req, stream) {
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		got = append(got, resp)
	}
	return got, firstErr
}

// assertTurnShape checks the invariant the bug broke — the last response closes
// the turn and no other claims to — and returns it. Asserting on positions
// rather than a count keeps it stable if the aggregator changes how much it
// emits.
func assertTurnShape(t *testing.T, got []*model.LLMResponse) *model.LLMResponse {
	t.Helper()
	if len(got) == 0 {
		t.Fatal("stream emitted no responses, want at least the aggregated turn")
	}
	var terminal []int
	for i, resp := range got {
		if resp.TurnComplete {
			terminal = append(terminal, i)
		}
	}
	if diff := cmp.Diff([]int{len(got) - 1}, terminal); diff != "" {
		t.Errorf("indices of responses with TurnComplete (-want +got):\n%s", diff)
	}
	for i, resp := range got[:len(got)-1] {
		if resp.FinishReason != "" {
			t.Errorf("partial response %d has FinishReason %q, want none", i, resp.FinishReason)
		}
		if !resp.Partial {
			t.Errorf("partial response %d has Partial = false, want true", i)
		}
	}
	final := got[len(got)-1]
	if final.Partial {
		t.Error("final response has Partial = true, want false")
	}
	return final
}

// TestModel_GenerateStream_TurnComplete pins the end of a streamed turn: the last
// response is the only terminal one, and it carries the finish reason and
// logprobs the same output reports without streaming.
func TestModel_GenerateStream_TurnComplete(t *testing.T) {
	helloLogprobs := &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{{Token: "hello", LogProbability: -0.5}},
		TopCandidates: []*genai.LogprobsResultTopCandidates{
			{Candidates: []*genai.LogprobsResultCandidate{{Token: "hello", LogProbability: -0.5}}},
		},
	}

	tests := []struct {
		name             string
		events           []string
		wantFinishReason genai.FinishReason
		wantLogprobs     *genai.LogprobsResult
		wantModelVersion string
	}{
		{
			name:             "completed",
			events:           []string{evCreated, evDelta1, evDelta2, evCompleted},
			wantFinishReason: genai.FinishReasonStop,
			wantLogprobs:     helloLogprobs,
			wantModelVersion: "stream-model",
		},
		{
			name:             "truncated by max output tokens",
			events:           []string{evCreated, evDelta1, evDelta2, evMaxTokens},
			wantFinishReason: genai.FinishReasonMaxTokens,
			wantModelVersion: "stream-model",
		},
		{
			name:             "cut short by the content filter",
			events:           []string{evCreated, evDelta1, evDelta2, evFiltered},
			wantFinishReason: genai.FinishReasonSafety,
			wantModelVersion: "stream-model",
		},
		{
			// The model never said why it stopped, so report it as unknown
			// rather than invent a clean stop.
			name:             "stream ends without a terminal response",
			events:           []string{evCreated, evDelta1, evDelta2},
			wantFinishReason: genai.FinishReasonUnspecified,
			wantModelVersion: "stream-model",
		},
		{
			// No response object at all, so no model name to read off one.
			name:             "no response object at all",
			events:           []string{evDelta1, evDelta2},
			wantFinishReason: genai.FinishReasonUnspecified,
		},
		{
			// A provider that batches its output streams no deltas at all.
			name:             "output batched onto the terminal event",
			events:           []string{evCreated, evCompleted},
			wantFinishReason: genai.FinishReasonStop,
			wantLogprobs:     helloLogprobs,
			wantModelVersion: "stream-model",
		},
		{
			// A "completed" after a truncation must not relabel it a clean stop.
			name:             "first terminal event wins",
			events:           []string{evCreated, evDelta1, evDelta2, evMaxTokens, evCompleted},
			wantFinishReason: genai.FinishReasonMaxTokens,
			wantModelVersion: "stream-model",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := runStream(t, tc.events...)
			if err != nil {
				t.Fatalf("GenerateContent() stream err = %v", err)
			}
			final := assertTurnShape(t, got)

			if final.FinishReason != tc.wantFinishReason {
				t.Errorf("final FinishReason = %q, want %q", final.FinishReason, tc.wantFinishReason)
			}
			if diff := cmp.Diff(tc.wantLogprobs, final.LogprobsResult); diff != "" {
				t.Errorf("final LogprobsResult mismatch (-want +got):\n%s", diff)
			}
			if final.ModelVersion != tc.wantModelVersion {
				t.Errorf("final ModelVersion = %q, want %q", final.ModelVersion, tc.wantModelVersion)
			}
			if text := allText(final.Content); text != "hello" {
				t.Errorf("final text = %q, want %q", text, "hello")
			}
		})
	}
}

// TestModel_GenerateStream_MatchesBlocking pins the parity the bug cost: one
// model output reports the same terminal fields whether or not it streamed.
func TestModel_GenerateStream_MatchesBlocking(t *testing.T) {
	streamed, err := runStream(t, evCreated, evDelta1, evDelta2, evCompleted)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	blocking, err := runBlocking(t, bodyCompleted)
	if err != nil {
		t.Fatalf("blocking err = %v", err)
	}
	if len(blocking) != 1 {
		t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
	}
	got, want := assertTurnShape(t, streamed), blocking[0]

	if got.FinishReason != want.FinishReason {
		t.Errorf("FinishReason: streamed %q, blocking %q", got.FinishReason, want.FinishReason)
	}
	if diff := cmp.Diff(want.LogprobsResult, got.LogprobsResult); diff != "" {
		t.Errorf("LogprobsResult mismatch (-blocking +streamed):\n%s", diff)
	}
	if diff := cmp.Diff(want.UsageMetadata, got.UsageMetadata); diff != "" {
		t.Errorf("UsageMetadata mismatch (-blocking +streamed):\n%s", diff)
	}
	if got.ModelVersion != want.ModelVersion {
		t.Errorf("ModelVersion: streamed %q, blocking %q", got.ModelVersion, want.ModelVersion)
	}
	if gotText, wantText := allText(got.Content), allText(want.Content); gotText != wantText {
		t.Errorf("text: streamed %q, blocking %q", gotText, wantText)
	}
}

// TestModel_GenerateStream_TruncatedLeavesUsageUnset pins that a stream cut off
// before its terminal event reports no usage rather than zero usage, which would
// understate a turn that did real work.
func TestModel_GenerateStream_TruncatedLeavesUsageUnset(t *testing.T) {
	got, err := runStream(t, evCreated, evDelta1, evDelta2)
	if err != nil {
		t.Fatalf("GenerateContent() stream err = %v", err)
	}
	if final := assertTurnShape(t, got); final.UsageMetadata != nil {
		t.Errorf("final UsageMetadata = %+v, want nil", final.UsageMetadata)
	}
}

// TestModel_GenerateStream_NoOutputItems pins that a terminal response carrying
// nothing usable fails the call, as blocking does, rather than closing the turn
// with a silently empty answer.
func TestModel_GenerateStream_NoOutputItems(t *testing.T) {
	got, err := runStream(t, evCreated, evFiltered)
	if !errors.Is(err, ErrNoOutputItems) {
		t.Errorf("streaming err = %v, want %v", err, ErrNoOutputItems)
	}
	if len(got) != 0 {
		t.Errorf("stream emitted %d responses, want none", len(got))
	}
	const filteredBody = `{"id":"resp_1","model":"stream-model","incomplete_details":{"reason":"content_filter"}}`
	if _, err := runBlocking(t, filteredBody); !errors.Is(err, ErrNoOutputItems) {
		t.Errorf("blocking err = %v, want %v", err, ErrNoOutputItems)
	}
}

// TestModel_GenerateStream_Failed pins that a stream that errors out ends the
// turn with that error, never with a response claiming the turn completed.
func TestModel_GenerateStream_Failed(t *testing.T) {
	got, err := runStream(t, evCreated, evDelta1, evFailed)
	if err == nil {
		t.Fatal("streaming err = nil, want the upstream failure")
	}
	if !strings.Contains(err.Error(), "upstream exploded") {
		t.Errorf("streaming err = %v, want it to name the upstream failure", err)
	}
	for i, resp := range got {
		if resp.TurnComplete {
			t.Errorf("response %d has TurnComplete = true, want the error to end the turn", i)
		}
	}
}

// TestModel_GenerateStream_FunctionCall pins that a streamed tool call closes its
// turn the way a streamed message does.
func TestModel_GenerateStream_FunctionCall(t *testing.T) {
	const (
		evItemAdded = `{"type":"response.output_item.added","item":{"id":"item_1","call_id":"call_1","name":"get_weather"}}`
		evArgsDone  = `{"type":"response.function_call_arguments.done","item_id":"item_1","arguments":"{\"city\":\"SF\"}"}`
	)
	got, err := runStream(t, evCreated, evItemAdded, evArgsDone, evCompleted)
	if err != nil {
		t.Fatalf("GenerateContent() stream err = %v", err)
	}
	final := assertTurnShape(t, got)
	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonStop)
	}
	var calls []*genai.FunctionCall
	if final.Content != nil {
		for _, part := range final.Content.Parts {
			if part.FunctionCall != nil {
				calls = append(calls, part.FunctionCall)
			}
		}
	}
	want := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}}}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("final function calls mismatch (-want +got):\n%s", diff)
	}
}

func allText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var text string
	for _, part := range content.Parts {
		text += part.Text
	}
	return text
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
