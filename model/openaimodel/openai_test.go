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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
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
	evError     = `{"type":"error","message":"stream blew up"}`
	evReasoning = `{"type":"response.reasoning_text.delta","delta":"thinking"}`
	// A terminal event omitting the response object the schema marks required.
	evBareCompleted = `{"type":"response.completed"}`
	// Arguments the SSE decoder cannot finish reading, so the stream itself
	// fails rather than an event in it.
	evTruncated = `{"type":"response.output_text.delta","delta":`
	// A call the aggregator drops for want of a name, no
	// "response.output_item.added" having introduced the item.
	evArgsDoneUnnamed = `{"type":"response.function_call_arguments.done","item_id":"item_1","arguments":"{\"city\":\"SF\"}"}`

	// The blocking-mode body carrying exactly the output evCompleted does, so
	// the two paths can be compared on the same model output.
	bodyCompleted = `{"id":"resp_1","model":"stream-model","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7},"output":[{"type":"message","content":[{"type":"output_text","text":"hello","logprobs":[{"token":"hello","logprob":-0.5,"top_logprobs":[{"token":"hello","logprob":-0.5}]}]}]}]}`
	// The blocking-mode bodies for the streams asserted on above.
	bodyFiltered = `{"id":"resp_1","model":"stream-model","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`
	bodyToolCall = `{"id":"resp_1","model":"stream-model","output":[{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}`
)

// incompleteEvent builds a "response.incomplete" whose response carries the
// given fields ahead of the model's partial output.
func incompleteEvent(fields string) string {
	return `{"type":"response.incomplete","response":{"id":"resp_1","model":"stream-model",` + fields +
		`"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}}`
}

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
		{
			// Nor must a stray "created", whose response has no reason of its
			// own and would otherwise read as a clean stop.
			name:             "trailing response.created does not reopen the turn",
			events:           []string{evCreated, evDelta1, evDelta2, evMaxTokens, evCreated},
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

// TestModel_GenerateStream_IncompleteWithoutReason pins that an event whose
// name declares the turn incomplete never closes it as a clean stop. openai-go
// marks incomplete_details required but not its reason, so every shape below is
// schema-legal.
func TestModel_GenerateStream_IncompleteWithoutReason(t *testing.T) {
	tests := []struct {
		name             string
		fields           string
		wantMessage      string
		wantFinishReason genai.FinishReason
	}{
		{
			name:             "empty incomplete_details",
			fields:           `"status":"incomplete","incomplete_details":{},`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			name:             "incomplete_details omitted",
			fields:           `"status":"incomplete",`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			name:             "incomplete_details null",
			fields:           `"status":"incomplete","incomplete_details":null,`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			name:             "reason empty",
			fields:           `"status":"incomplete","incomplete_details":{"reason":""},`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			// Neither a status nor a reason: only the event's own name is left
			// to say the turn was cut short.
			name:             "status omitted too",
			fields:           ``,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			// No status, but an incomplete_details object the provider did
			// send — the one shape the blocking path can read too.
			name:             "empty incomplete_details, no status",
			fields:           `"incomplete_details":{},`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "incomplete",
		},
		{
			// An unmapped reason still flattens to OTHER, leaving the
			// provider's wording the only account of what happened.
			name:             "unrecognized reason",
			fields:           `"status":"incomplete","incomplete_details":{"reason":"something_new"},`,
			wantFinishReason: genai.FinishReasonOther,
			wantMessage:      "something_new",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := runStream(t, evCreated, evDelta1, evDelta2, incompleteEvent(tc.fields))
			if err != nil {
				t.Fatalf("GenerateContent() stream err = %v", err)
			}
			final := assertTurnShape(t, got)
			if final.FinishReason != tc.wantFinishReason {
				t.Errorf("final FinishReason = %q, want %q", final.FinishReason, tc.wantFinishReason)
			}
			assertFinishSignal(t, final, tc.wantMessage)
		})
	}
}

// TestIncompleteWithoutReason_MatchesBlocking pins that the two paths agree on a
// truncated turn wherever the payload itself carries the news. Only a payload
// reporting nothing at all can differ, since there only streaming has the
// event's name to read.
func TestIncompleteWithoutReason_MatchesBlocking(t *testing.T) {
	tests := []struct {
		name        string
		fields      string
		want        genai.FinishReason
		wantMessage string
	}{
		{name: "status incomplete", fields: `"status":"incomplete",`, want: genai.FinishReasonOther, wantMessage: "incomplete"},
		{name: "empty incomplete_details", fields: `"incomplete_details":{},`, want: genai.FinishReasonOther, wantMessage: "incomplete"},
		{name: "empty reason", fields: `"incomplete_details":{"reason":""},`, want: genai.FinishReasonOther, wantMessage: "incomplete"},
		{name: "a mapped reason", fields: `"incomplete_details":{"reason":"max_output_tokens"},`, want: genai.FinishReasonMaxTokens},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			event := incompleteEvent(tc.fields)
			streamed, err := runStream(t, evCreated, evDelta1, evDelta2, event)
			if err != nil {
				t.Fatalf("streaming err = %v", err)
			}
			// incompleteEvent wraps exactly this body.
			body := strings.TrimSuffix(strings.TrimPrefix(event, `{"type":"response.incomplete","response":`), `}`)
			blocking, err := runBlocking(t, body)
			if err != nil {
				t.Fatalf("blocking err = %v", err)
			}
			if len(blocking) != 1 {
				t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
			}
			for name, got := range map[string]*model.LLMResponse{
				"streamed": assertTurnShape(t, streamed),
				"blocking": blocking[0],
			} {
				if got.FinishReason != tc.want {
					t.Errorf("%s FinishReason = %q, want %q", name, got.FinishReason, tc.want)
				}
				// The wording has to match too, not just the reason.
				if msg, _ := got.CustomMetadata[FinishMessageKey].(string); msg != tc.wantMessage {
					t.Errorf("%s CustomMetadata[%q] = %q, want %q", name, FinishMessageKey, msg, tc.wantMessage)
				}
			}
		})
	}
}

// assertFinishSignal pins where a turn that ended badly but still carries an
// answer says why: as metadata, never in the error fields, which tool/agenttool
// and server/adka2a both read as a failed turn.
func assertFinishSignal(t *testing.T, resp *model.LLMResponse, wantMessage string) {
	t.Helper()
	if !carriesContent(resp) {
		t.Fatal("response carries no content, so this is not the case being asserted")
	}
	if resp.ErrorCode != "" || resp.ErrorMessage != "" {
		t.Errorf("response has ErrorCode %q / ErrorMessage %q, want both empty on a turn that carries an answer",
			resp.ErrorCode, resp.ErrorMessage)
	}
	got, _ := resp.CustomMetadata[FinishMessageKey].(string)
	if got != wantMessage {
		t.Errorf("CustomMetadata[%q] = %q, want %q", FinishMessageKey, got, wantMessage)
	}
}

// TestBlockedTurnSurfacesBlockReason pins that a content-filtered turn reaches
// the caller with something to gate on: metadata beside a SAFETY finish reason
// when it still carries an answer, and ErrorCode — where model/gemini reports a
// blocked prompt — when it does not.
func TestBlockedTurnSurfacesBlockReason(t *testing.T) {
	t.Run("with an answer to read", func(t *testing.T) {
		streamed, err := runStream(t, evCreated, evDelta1, evDelta2, evFiltered)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		blocking, err := runBlocking(t, bodyFiltered)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if len(blocking) != 1 {
			t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
		}
		for name, got := range map[string]*model.LLMResponse{
			"streamed": assertTurnShape(t, streamed),
			"blocking": blocking[0],
		} {
			t.Run(name, func(t *testing.T) {
				if got.FinishReason != genai.FinishReasonSafety {
					t.Errorf("FinishReason = %q, want %q", got.FinishReason, genai.FinishReasonSafety)
				}
				assertFinishSignal(t, got, "content_filter")
			})
		}
	})

	t.Run("with nothing to read", func(t *testing.T) {
		// The deltas produce a call the aggregator drops, so the turn is
		// blocked with no content at all — the case ErrorCode is for.
		got, err := runStream(t, evCreated, evArgsDoneUnnamed, evFiltered)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		if carriesContent(final) {
			t.Fatalf("final carries content %+v, so this is not the case being asserted", final.Content)
		}
		if final.FinishReason != genai.FinishReasonSafety {
			t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonSafety)
		}
		if want := string(genai.BlockedReasonSafety); final.ErrorCode != want {
			t.Errorf("final ErrorCode = %q, want %q", final.ErrorCode, want)
		}
		if want := "content_filter"; final.ErrorMessage != want {
			t.Errorf("final ErrorMessage = %q, want %q", final.ErrorMessage, want)
		}
	})
}

// TestUnfinishedTurnWithNothingToReadReportsErrorCode pins the arm of
// attachFinishSignal that sets ErrorCode, and so makes tool/agenttool abort the
// run and server/adka2a fail the task: a turn that ended for a reason this
// package does not map, carrying nothing to read. main reports a clean STOP
// with neither field for the same stream.
func TestUnfinishedTurnWithNothingToReadReportsErrorCode(t *testing.T) {
	// The unnamed call is dropped in aggregation, so the turn carries nothing,
	// and the event names neither a reason nor a status. It still needs a
	// response object: one without never latches.
	const evBareIncomplete = `{"type":"response.incomplete","response":{"id":"resp_1","model":"stream-model"}}`
	got, err := runStream(t, evCreated, evArgsDoneUnnamed, evBareIncomplete)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	final := assertTurnShape(t, got)
	if carriesContent(final) {
		t.Fatalf("final carries content %+v, so this is not the case being asserted", final.Content)
	}
	if final.FinishReason != genai.FinishReasonOther {
		t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonOther)
	}
	if want := string(genai.FinishReasonOther); final.ErrorCode != want {
		t.Errorf("final ErrorCode = %q, want %q", final.ErrorCode, want)
	}
	if want := "incomplete"; final.ErrorMessage != want {
		t.Errorf("final ErrorMessage = %q, want %q", final.ErrorMessage, want)
	}
}

// TestModel_GenerateStream_LogprobsDescribeTheAnswer pins that logprobs are only
// reported against the text they belong to. A streamed turn takes its content
// from the deltas and its logprobs from the terminal event, which a provider
// may restate differently.
func TestModel_GenerateStream_LogprobsDescribeTheAnswer(t *testing.T) {
	tests := []struct {
		name         string
		events       []string
		wantText     string
		wantLogprobs bool
	}{
		{
			name:     "deltas stop short of the terminal output",
			events:   []string{evCreated, evDelta1, evCompleted},
			wantText: "hel",
		},
		{
			name:     "delta arrives after the terminal output and disagrees",
			events:   []string{evCreated, evCompleted, `{"type":"response.output_text.delta","delta":"other"}`},
			wantText: "other",
		},
		{
			// Reasoning is not part of the answer the logprobs describe.
			name:         "thoughts do not count against the answer",
			events:       []string{evCreated, evReasoning, evDelta1, evDelta2, evCompleted},
			wantText:     "thinkinghello",
			wantLogprobs: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := runStream(t, tc.events...)
			if err != nil {
				t.Fatalf("GenerateContent() stream err = %v", err)
			}
			final := assertTurnShape(t, got)
			if text := allText(final.Content); text != tc.wantText {
				t.Errorf("final text = %q, want %q", text, tc.wantText)
			}
			if gotLogprobs := final.LogprobsResult != nil; gotLogprobs != tc.wantLogprobs {
				t.Errorf("final LogprobsResult = %+v, want any = %v: they describe %q",
					final.LogprobsResult, tc.wantLogprobs, "hello")
			}
		})
	}
}

// TestModel_GenerateStream_TerminalEventWithoutResponse pins that an event
// omitting the response object the schema marks required does not close the
// turn. AsResponse* discards its unmarshal error and returns a zero value, so
// latching one would lose what a later, well-formed event reports.
func TestModel_GenerateStream_TerminalEventWithoutResponse(t *testing.T) {
	t.Run("a well-formed event later in the turn still wins", func(t *testing.T) {
		got, err := runStream(t, evCreated, evDelta1, evDelta2, evBareCompleted, evCompleted)
		if err != nil {
			t.Fatalf("GenerateContent() stream err = %v", err)
		}
		final := assertTurnShape(t, got)
		if final.ModelVersion != "stream-model" {
			t.Errorf("final ModelVersion = %q, want %q", final.ModelVersion, "stream-model")
		}
		if final.UsageMetadata == nil || final.UsageMetadata.TotalTokenCount != 7 {
			t.Errorf("final UsageMetadata = %+v, want the terminal event's 7 total tokens", final.UsageMetadata)
		}
		if final.FinishReason != genai.FinishReasonStop {
			t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonStop)
		}
	})

	t.Run("alone it says nothing about why the turn ended", func(t *testing.T) {
		got, err := runStream(t, evCreated, evDelta1, evDelta2, evBareCompleted)
		if err != nil {
			t.Fatalf("GenerateContent() stream err = %v", err)
		}
		final := assertTurnShape(t, got)
		if final.FinishReason != genai.FinishReasonUnspecified {
			t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonUnspecified)
		}
		if final.UsageMetadata != nil {
			t.Errorf("final UsageMetadata = %+v, want nil", final.UsageMetadata)
		}
	})
}

// TestModel_GenerateStream_EmptyAggregateUsesTerminalEvent pins that a turn the
// aggregator emptied is rebuilt from the terminal event rather than closed as a
// clean stop with nothing in it: a streamed function call with no name is
// dropped in aggregation, and blocking returns it for the same body.
func TestModel_GenerateStream_EmptyAggregateUsesTerminalEvent(t *testing.T) {
	t.Run("a tool call", func(t *testing.T) {
		const evCompletedToolCall = `{"type":"response.completed","response":` + bodyToolCall + `}`
		streamed, err := runStream(t, evCreated, evArgsDoneUnnamed, evCompletedToolCall)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		blocking, err := runBlocking(t, bodyToolCall)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if len(blocking) != 1 {
			t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
		}
		want := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}}}
		if diff := cmp.Diff(want, functionCalls(blocking[0])); diff != "" {
			t.Fatalf("blocking function calls mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, streamed))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	// Text has no adoption path of its own, so a provider that batches its
	// whole message onto the terminal event relies on the rebuild.
	t.Run("a whole message the deltas never carried", func(t *testing.T) {
		streamed, err := runStream(t, evCreated, evArgsDoneUnnamed, evCompleted)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, streamed)
		if text := allText(final.Content); text != "hello" {
			t.Errorf("streamed text = %q, want the terminal event's %q", text, "hello")
		}
		if final.LogprobsResult == nil {
			t.Error("streamed LogprobsResult = nil, want the terminal event's logprobs for that text")
		}
	})
}

// TestModel_GenerateStream_CreatedOnly pins what a stream that announces a
// response and then produces nothing yields: no turn at all. Blocking fails
// with ErrNoOutputItems on the same body, an asymmetry this test records rather
// than endorses.
func TestModel_GenerateStream_CreatedOnly(t *testing.T) {
	got, err := runStream(t, evCreated)
	if err != nil {
		t.Fatalf("GenerateContent() stream err = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("stream emitted %d responses, want none", len(got))
	}
	if _, err := runBlocking(t, `{"id":"resp_1","model":"stream-model"}`); !errors.Is(err, ErrNoOutputItems) {
		t.Errorf("blocking err = %v, want %v", err, ErrNoOutputItems)
	}
}

// closeSpy counts Close calls on the response bodies it sees, so a test can
// observe whether the upstream stream was torn down.
type closeSpy struct {
	base   http.RoundTripper
	closed atomic.Int64
}

func (s *closeSpy) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := s.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	resp.Body = spyBody{ReadCloser: resp.Body, spy: s}
	return resp, nil
}

type spyBody struct {
	io.ReadCloser
	spy *closeSpy
}

func (b spyBody) Close() error {
	b.spy.closed.Add(1)
	return b.ReadCloser.Close()
}

// TestModel_GenerateStream_StopsWhenTheConsumerStops pins the iter.Seq2
// contract: a consumer that breaks out of the range loop ends the sequence and
// the upstream stream is closed. The language enforces the first half; the
// deferred stream.Close is ours, and leaks a connection per abandoned turn.
func TestModel_GenerateStream_StopsWhenTheConsumerStops(t *testing.T) {
	// The handler blocks after the first delta, so a consumer that breaks does
	// so mid-turn with the response still open.
	release := make(chan struct{})
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, evt := range []string{evCreated, evDelta1} {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", evt)
		}
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer server.Close()
	defer close(release)

	client := server.Client()
	spy := &closeSpy{base: client.Transport}
	client.Transport = spy

	ctx := t.Context()
	llm, err := NewModel(ctx, openai.ChatModelGPT4oMini, &ClientConfig{
		APIKey:     "test",
		BaseURL:    server.URL + "/v1",
		HTTPClient: client,
	})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("World?", genai.RoleUser)},
	}
	var got int
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("GenerateContent() stream err = %v", err)
		}
		got++
		if resp.TurnComplete {
			t.Fatal("turn completed before the consumer stopped, so the break is untested")
		}
		break
	}
	if got != 1 {
		t.Fatalf("consumer saw %d responses before breaking, want 1", got)
	}
	if n := spy.closed.Load(); n != 1 {
		t.Errorf("upstream body closed %d times after the consumer broke, want 1", n)
	}
}

// TestModel_GenerateStream_OutputlessTerminalEventKeepsTheReason pins that a
// turn the aggregator emptied still reports why it ended when the terminal
// event has nothing to rebuild from — the shape a truncated turn arrives in.
func TestModel_GenerateStream_OutputlessTerminalEventKeepsTheReason(t *testing.T) {
	// evArgsDoneUnnamed is dropped in aggregation, so the turn exists but holds
	// nothing; evMaxTokens carries no output.
	got, err := runStream(t, evCreated, evArgsDoneUnnamed, evMaxTokens)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	final := assertTurnShape(t, got)
	if final.FinishReason != genai.FinishReasonMaxTokens {
		t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonMaxTokens)
	}

	// With nothing in the aggregator either, there is no turn to report and the
	// call fails the way blocking does on the same body.
	if _, err := runStream(t, evCreated, evMaxTokens); !errors.Is(err, ErrNoOutputItems) {
		t.Errorf("streaming err = %v, want %v", err, ErrNoOutputItems)
	}
}

// TestModel_GenerateStream_TerminalEventWithEmptyResponse pins that an event
// carrying an empty response object is no more terminal than one omitting it:
// both decode to the same zero value.
func TestModel_GenerateStream_TerminalEventWithEmptyResponse(t *testing.T) {
	const evEmptyResponse = `{"type":"response.completed","response":{}}`
	got, err := runStream(t, evCreated, evDelta1, evDelta2, evEmptyResponse, evCompleted)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	final := assertTurnShape(t, got)
	if final.ModelVersion != "stream-model" {
		t.Errorf("final ModelVersion = %q, want %q", final.ModelVersion, "stream-model")
	}
	if final.UsageMetadata == nil || final.UsageMetadata.TotalTokenCount != 7 {
		t.Errorf("final UsageMetadata = %+v, want the well-formed event's 7 total tokens", final.UsageMetadata)
	}
}

// TestModel_GenerateStream_TerminalEventWithoutID pins the other half of
// carriesResponse: a response that decoded but omits "id" still closes the turn.
// The guard is aimed at the zero value, not at every response the schema would
// fault.
func TestModel_GenerateStream_TerminalEventWithoutID(t *testing.T) {
	t.Run("output batched onto the terminal event still reaches the caller", func(t *testing.T) {
		const evIdlessBatched = `{"type":"response.completed","response":{"model":"stream-model","status":"completed",` +
			`"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}}`
		got, err := runStream(t, evIdlessBatched)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		if text := allText(final.Content); text != "hello" {
			t.Errorf("streamed text = %q, want the terminal event's %q", text, "hello")
		}
		if final.FinishReason != genai.FinishReasonStop {
			t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonStop)
		}
	})

	t.Run("a turn cut short still reports why", func(t *testing.T) {
		const evIdlessFiltered = `{"type":"response.incomplete","response":{"model":"stream-model","status":"incomplete",` +
			`"incomplete_details":{"reason":"content_filter"},` +
			`"output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}}`
		got, err := runStream(t, evCreated, evDelta1, evDelta2, evIdlessFiltered)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		if final.FinishReason != genai.FinishReasonSafety {
			t.Errorf("final FinishReason = %q, want %q", final.FinishReason, genai.FinishReasonSafety)
		}
	})
}

// TestModel_GenerateStream_AdoptsTerminalToolCalls pins that a tool call the
// terminal event carries reaches the caller even when the deltas already built
// the turn's text. Losing it makes a turn that called a tool read as a plain
// answer.
func TestModel_GenerateStream_AdoptsTerminalToolCalls(t *testing.T) {
	const bodyTextAndCall = `{"id":"resp_1","model":"stream-model","output":[` +
		`{"type":"message","content":[{"type":"output_text","text":"hello"}]},` +
		`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}`
	const evTextAndCall = `{"type":"response.completed","response":` + bodyTextAndCall + `}`

	streamed, err := runStream(t, evCreated, evDelta1, evDelta2, evArgsDoneUnnamed, evTextAndCall)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	blocking, err := runBlocking(t, bodyTextAndCall)
	if err != nil {
		t.Fatalf("blocking err = %v", err)
	}
	if len(blocking) != 1 {
		t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
	}
	final := assertTurnShape(t, streamed)
	if text := allText(final.Content); text != "hello" {
		t.Errorf("streamed text = %q, want the deltas' %q", text, "hello")
	}
	want := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}}}
	if diff := cmp.Diff(want, functionCalls(blocking[0])); diff != "" {
		t.Fatalf("blocking function calls mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, functionCalls(final)); diff != "" {
		t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
	}

	t.Run("a call the deltas already carry is not duplicated", func(t *testing.T) {
		const evNamedArgsDone = `{"type":"response.output_item.added","item":{"id":"item_1","type":"function_call","name":"get_weather","call_id":"call_1"}}`
		got, err := runStream(t, evCreated, evNamedArgsDone, evArgsDoneUnnamed, evTextAndCall)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("unusable arguments fail the call as they do unstreamed", func(t *testing.T) {
		const bodyBadArgs = `{"id":"resp_1","model":"stream-model","output":[` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]},` +
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{"}]}`
		_, err := runStream(t, evCreated, evDelta1, evDelta2, `{"type":"response.completed","response":`+bodyBadArgs+`}`)
		if !errors.Is(err, ErrFunctionCallArgs) {
			t.Errorf("streaming err = %v, want %v", err, ErrFunctionCallArgs)
		}
		if _, err := runBlocking(t, bodyBadArgs); !errors.Is(err, ErrFunctionCallArgs) {
			t.Errorf("blocking err = %v, want %v", err, ErrFunctionCallArgs)
		}
	})
}

// TestModel_GenerateStream_TerminalToolCallsAreAuthoritative pins that the
// terminal event's calls stand in for what streamed rather than adding to it.
//
// The streamed call ID falls back to the item ID when
// "response.output_item.added" carries no "call_id", so reconciling the two
// lists dispatches one tool intent twice for any provider that omits the field.
func TestModel_GenerateStream_TerminalToolCallsAreAuthoritative(t *testing.T) {
	const (
		evAddedNameOnly = `{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","name":"get_weather"}}`
		evAddedOtherID  = `{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","name":"get_weather","call_id":"call_A"}}`
		evArgsDoneNamed = `{"type":"response.function_call_arguments.done","item_id":"fc_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"}`
		evArgsDoneSF    = `{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"city\":\"SF\"}"}`
		callSF          = `{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}`
		evOneCall       = `{"type":"response.completed","response":{"id":"resp_1","model":"stream-model",` +
			`"status":"completed","output":[` + callSF + `]}}`
	)
	wantSF := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}}}

	// The three shapes in which the streamed ID cannot match the terminal
	// item's, each of which appended a second copy of the one call.
	tests := []struct {
		name   string
		events []string
	}{
		{"the added event carries no call_id", []string{evCreated, evAddedNameOnly, evArgsDoneSF, evOneCall}},
		{"no added event, the name arrives on the done event", []string{evCreated, evArgsDoneNamed, evOneCall}},
		{"the added event and the terminal item disagree", []string{evCreated, evAddedOtherID, evArgsDoneSF, evOneCall}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := runStream(t, tc.events...)
			if err != nil {
				t.Fatalf("streaming err = %v", err)
			}
			if diff := cmp.Diff(wantSF, functionCalls(assertTurnShape(t, got))); diff != "" {
				t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
			}
		})
	}

	// Two calls stream and the terminal event names one. Whether that is the
	// provider's count or the turn being cut off is what the event's name says.
	const (
		evAdded1 = `{"type":"response.output_item.added","item":{"id":"f1","type":"function_call","name":"get_weather","call_id":"call_1"}}`
		evAdded2 = `{"type":"response.output_item.added","item":{"id":"f2","type":"function_call","name":"get_weather","call_id":"call_2"}}`
		evArgs1  = `{"type":"response.function_call_arguments.done","item_id":"f1","arguments":"{\"city\":\"SF\"}"}`
		evArgs2  = `{"type":"response.function_call_arguments.done","item_id":"f2","arguments":"{\"city\":\"NY\"}"}`
	)
	twoStreamed := []string{evCreated, evAdded1, evArgs1, evAdded2, evArgs2}

	t.Run("a completed response states the turn's whole output", func(t *testing.T) {
		got, err := runStream(t, append(slices.Clone(twoStreamed), evOneCall)...)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		if diff := cmp.Diff(wantSF, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("an incomplete response may name fewer than streamed", func(t *testing.T) {
		const evCutShort = `{"type":"response.incomplete","response":{"id":"resp_1","model":"stream-model",` +
			`"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[` + callSF + `]}}`
		got, err := runStream(t, append(slices.Clone(twoStreamed), evCutShort)...)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		// The shorter list is the truncation showing, not the provider's count:
		// replacing would lose a call the model did make.
		want := []*genai.FunctionCall{
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
			{Name: "get_weather", ID: "call_2", Args: map[string]any{"city": "NY"}},
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("the event's calls keep the place the turn's calls held", func(t *testing.T) {
		// Replacing must not reorder: appending would move the call past text
		// it preceded, where blocking reports it first for the same body.
		const evCallThenText = `{"type":"response.completed","response":{"id":"resp_1","model":"stream-model",` +
			`"status":"completed","output":[` + callSF + `,` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}}`
		got, err := runStream(t, evCreated, evAddedNameOnly, evArgsDoneSF, evDelta1, evDelta2, evCallThenText)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		if diff := cmp.Diff([]string{"call", "text"}, partKinds(assertTurnShape(t, got))); diff != "" {
			t.Errorf("part order mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("a turn that interleaved calls with text stays interleaved", func(t *testing.T) {
		// Two calls straddling the text. Each stated call has one of the turn's
		// own to stand in for, so it takes that one's place: gathering them at
		// the first would report a shape neither path gives for this body.
		const body = `{"id":"resp_1","model":"stream-model","status":"completed","output":[` + callSF + `,` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]},` +
			`{"type":"function_call","name":"get_weather","call_id":"call_2","arguments":"{\"city\":\"NY\"}"}]}`
		streamed, err := runStream(t, evCreated, evAdded1, evArgs1, evDelta1, evDelta2, evAdded2, evArgs2,
			`{"type":"response.completed","response":`+body+`}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, streamed)
		blocked, err := runBlocking(t, body)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if diff := cmp.Diff(partKinds(blocked[len(blocked)-1]), partKinds(final)); diff != "" {
			t.Errorf("streaming and blocking disagree on part order (-blocking +streaming):\n%s", diff)
		}
		// Still the event's own calls, in the event's order.
		want := []*genai.FunctionCall{
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
			{Name: "get_weather", ID: "call_2", Args: map[string]any{"city": "NY"}},
		}
		if diff := cmp.Diff(want, functionCalls(final)); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("a rebuilt turn keeps the shape blocking gives the same body", func(t *testing.T) {
		// Nothing survives aggregation, so the turn is rebuilt from the event
		// as blocking builds it and adoption then runs over that rebuild.
		// Appending reordered the parts the rebuild had just taken from it.
		const body = `{"id":"resp_1","model":"stream-model","status":"completed","output":[` + callSF + `,` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`
		streamed, err := runStream(t, evCreated, `{"type":"response.completed","response":`+body+`}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		blocked, err := runBlocking(t, body)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		want := partKinds(blocked[len(blocked)-1])
		if diff := cmp.Diff(want, partKinds(assertTurnShape(t, streamed))); diff != "" {
			t.Errorf("streaming and blocking disagree on part order (-blocking +streaming):\n%s", diff)
		}
	})

	t.Run("the event's order is the turn's, not the order they streamed in", func(t *testing.T) {
		// The event lists the same two calls the other way round. The flow
		// dispatches tools in part order, so following the stream here would
		// run them in an order the provider did not state.
		const evReversed = `{"type":"response.completed","response":{"id":"resp_1","model":"stream-model",` +
			`"status":"completed","output":[{"type":"function_call","name":"get_weather","call_id":"call_2","arguments":"{\"city\":\"NY\"}"},` +
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}}`
		got, err := runStream(t, append(slices.Clone(twoStreamed), evReversed)...)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		want := []*genai.FunctionCall{
			{Name: "get_weather", ID: "call_2", Args: map[string]any{"city": "NY"}},
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("a call the aggregator dropped keeps the event's order", func(t *testing.T) {
		// A call with no name anywhere is dropped in aggregation, so the turn
		// would report only the later one. Taking the event's list whole
		// recovers the first and puts it back ahead of the second.
		const (
			evUnnamedArgs = `{"type":"response.function_call_arguments.done","item_id":"f0","arguments":"{\"n\":1}"}`
			evBothInOrder = `{"type":"response.completed","response":{"id":"resp_1","model":"stream-model",` +
				`"status":"completed","output":[{"type":"function_call","name":"first","call_id":"call_0","arguments":"{\"n\":1}"},` +
				`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}}`
		)
		got, err := runStream(t, evCreated, evUnnamedArgs, evAdded1, evArgs1, evBothInOrder)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		want := []*genai.FunctionCall{
			{Name: "first", ID: "call_0", Args: map[string]any{"n": float64(1)}},
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
		}
		streamed := functionCalls(assertTurnShape(t, got))
		if diff := cmp.Diff(want, streamed); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
		// The same body blocking, since the event's list is what both report.
		blocked, err := runBlocking(t, `{"id":"resp_1","model":"stream-model","status":"completed",`+
			`"output":[{"type":"function_call","name":"first","call_id":"call_0","arguments":"{\"n\":1}"},`+
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}`)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if diff := cmp.Diff(functionCalls(blocked[len(blocked)-1]), streamed); diff != "" {
			t.Errorf("streaming and blocking disagree on the turn's calls (-blocking +streaming):\n%s", diff)
		}
	})

	t.Run("an unusable item fails the turn even beside the call it restates", func(t *testing.T) {
		// Taking the event's list converts every item, so a payload blocking
		// rejects is not excused by the deltas having built the same call
		// usably. Falling back to the streamed copy would report a turn
		// blocking fails, for the same bytes.
		const badArgs = `{"id":"resp_1","model":"stream-model","status":"completed",` +
			`"output":[{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{"}]}`
		if _, err := runStream(t, evCreated, evAdded1, evArgs1,
			`{"type":"response.completed","response":`+badArgs+`}`); !errors.Is(err, ErrFunctionCallArgs) {
			t.Errorf("streaming err = %v, want %v", err, ErrFunctionCallArgs)
		}
		if _, err := runBlocking(t, badArgs); !errors.Is(err, ErrFunctionCallArgs) {
			t.Errorf("blocking err = %v, want %v", err, ErrFunctionCallArgs)
		}
	})

	t.Run("a field the item leaves unstated keeps what streamed", func(t *testing.T) {
		// The event is authoritative for which calls a turn made, not for every
		// field of each. An item naming a call but carrying no arguments says
		// nothing about them; taking the blank dispatches the tool with no
		// arguments and no error.
		withArgs := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}}}
		noArgs := []*genai.FunctionCall{{Name: "get_weather", ID: "call_1", Args: map[string]any{}}}
		tests := []struct {
			name string
			item string
			want []*genai.FunctionCall
		}{
			{
				name: "arguments omitted",
				item: `{"type":"function_call","name":"get_weather","call_id":"call_1"}`,
				want: withArgs,
			},
			{
				name: "arguments null",
				item: `{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":null}`,
				want: withArgs,
			},
			{
				name: "name omitted, leaving a call nothing can dispatch",
				item: `{"type":"function_call","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}`,
				want: withArgs,
			},
			{
				// The other half: a stated field is never second-guessed. "{}"
				// is a call that takes none, and restoring over it would report
				// arguments the provider withdrew.
				name: "empty arguments are stated, not unstated",
				item: `{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{}"}`,
				want: noArgs,
			},
			{
				// The provider saying the same thing badly, but still saying it.
				name: "an empty string is stated too",
				item: `{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":""}`,
				want: noArgs,
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got, err := runStream(t, evCreated, evAdded1, evArgs1,
					`{"type":"response.completed","response":{"id":"resp_1","model":"stream-model",`+
						`"status":"completed","output":[`+tc.item+`]}}`)
				if err != nil {
					t.Fatalf("streaming err = %v", err)
				}
				if diff := cmp.Diff(tc.want, functionCalls(assertTurnShape(t, got))); diff != "" {
					t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("a call that streamed before the text is still reported before it", func(t *testing.T) {
		// The event states more calls than the turn holds, so they cannot each
		// take a slot and are placed as a group — where the turn's own call
		// sat, not at the end behind text that call streamed ahead of.
		const body = `{"id":"resp_1","model":"stream-model","status":"completed","output":[` +
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"},` +
			`{"type":"function_call","name":"get_weather","call_id":"call_2","arguments":"{\"city\":\"NY\"}"},` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`
		got, err := runStream(t, evCreated, evAdded1, evArgs1, evDelta1, evDelta2,
			`{"type":"response.completed","response":`+body+`}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		if diff := cmp.Diff([]string{"call", "call", "text"}, partKinds(final)); diff != "" {
			t.Errorf("part order mismatch (-want +got):\n%s", diff)
		}
		blocked, err := runBlocking(t, body)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if diff := cmp.Diff(partKinds(blocked[len(blocked)-1]), partKinds(final)); diff != "" {
			t.Errorf("streaming and blocking disagree on part order (-blocking +streaming):\n%s", diff)
		}
	})

	t.Run("an unstated field is restored when the event states more calls than streamed", func(t *testing.T) {
		// The counts differ, so the calls are placed as a group rather than each
		// taking a slot. Restoring still has to happen: the call IDs agree, and
		// declining would dispatch call_1 with no arguments while the deltas
		// that carried them sit right there.
		got, err := runStream(t, evCreated, evAdded1, evArgs1,
			`{"type":"response.completed","response":{"id":"resp_1","model":"stream-model","status":"completed",`+
				`"output":[{"type":"function_call","name":"get_weather","call_id":"call_1"},`+
				`{"type":"function_call","name":"get_weather","call_id":"call_2","arguments":"{\"city\":\"NY\"}"}]}}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		want := []*genai.FunctionCall{
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
			{Name: "get_weather", ID: "call_2", Args: map[string]any{"city": "NY"}},
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("each call is restored from the one it restates, not the one in its place", func(t *testing.T) {
		// The event lists the same two calls the other way round and states
		// arguments for neither. Pairing by slot would silently hand each the
		// arguments the model wrote for the other; the call ID says which.
		got, err := runStream(t, evCreated, evAdded1, evArgs1, evAdded2, evArgs2,
			`{"type":"response.completed","response":{"id":"resp_1","model":"stream-model","status":"completed",`+
				`"output":[{"type":"function_call","name":"get_weather","call_id":"call_2"},`+
				`{"type":"function_call","name":"get_weather","call_id":"call_1"}]}}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		want := []*genai.FunctionCall{
			{Name: "get_weather", ID: "call_2", Args: map[string]any{"city": "NY"}},
			{Name: "get_weather", ID: "call_1", Args: map[string]any{"city": "SF"}},
		}
		if diff := cmp.Diff(want, functionCalls(assertTurnShape(t, got))); diff != "" {
			t.Errorf("streamed function calls mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("the event places a call no aggregated turn held", func(t *testing.T) {
		// A nameless call is dropped in aggregation while the text survives, so
		// the turn holds no call and no position is a call's. The event's own
		// order is then the only one either path has, and it puts the call
		// first.
		const body = `{"id":"resp_1","model":"stream-model","status":"completed","output":[` +
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"},` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]}]}`
		got, err := runStream(t, evCreated,
			`{"type":"response.function_call_arguments.done","item_id":"f0","arguments":"{\"city\":\"SF\"}"}`,
			evDelta1, evDelta2, `{"type":"response.completed","response":`+body+`}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		if diff := cmp.Diff([]string{"call", "text"}, partKinds(final)); diff != "" {
			t.Errorf("part order mismatch (-want +got):\n%s", diff)
		}
		blocked, err := runBlocking(t, body)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if diff := cmp.Diff(partKinds(blocked[len(blocked)-1]), partKinds(final)); diff != "" {
			t.Errorf("streaming and blocking disagree on part order (-blocking +streaming):\n%s", diff)
		}
	})

	t.Run("a call the event states after its text stays after it", func(t *testing.T) {
		// The other half: an event leading with a message leaves the calls at
		// the end, so placement follows the event rather than always fronting.
		const body = `{"id":"resp_1","model":"stream-model","status":"completed","output":[` +
			`{"type":"message","content":[{"type":"output_text","text":"hello"}]},` +
			`{"type":"function_call","name":"get_weather","call_id":"call_1","arguments":"{\"city\":\"SF\"}"}]}`
		got, err := runStream(t, evCreated,
			`{"type":"response.function_call_arguments.done","item_id":"f0","arguments":"{\"city\":\"SF\"}"}`,
			evDelta1, evDelta2, `{"type":"response.completed","response":`+body+`}`)
		if err != nil {
			t.Fatalf("streaming err = %v", err)
		}
		final := assertTurnShape(t, got)
		blocked, err := runBlocking(t, body)
		if err != nil {
			t.Fatalf("blocking err = %v", err)
		}
		if diff := cmp.Diff(partKinds(blocked[len(blocked)-1]), partKinds(final)); diff != "" {
			t.Errorf("streaming and blocking disagree on part order (-blocking +streaming):\n%s", diff)
		}
	})
}

// partKinds names each part of a turn in order, for asserting on the shape a
// caller reads rather than on the parts themselves.
func partKinds(resp *model.LLMResponse) []string {
	if resp == nil || resp.Content == nil {
		return nil
	}
	kinds := make([]string, 0, len(resp.Content.Parts))
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			kinds = append(kinds, "call")
			continue
		}
		kinds = append(kinds, "text")
	}
	return kinds
}

// TestModel_GenerateStream_RefusalLogprobsMatchBlocking pins that a refusal
// counts towards the answer the logprobs describe: convertOutputItems turns one
// into a text part like any other, so skipping it would have the two paths
// disagree over whether the logprobs match.
func TestModel_GenerateStream_RefusalLogprobsMatchBlocking(t *testing.T) {
	const bodyRefusal = `{"id":"resp_1","model":"stream-model","output":[{"type":"message","content":[` +
		`{"type":"output_text","text":"hello","logprobs":[{"token":"hello","logprob":-0.5}]},` +
		`{"type":"refusal","refusal":"I cannot"}]}]}`

	streamed, err := runStream(t, evCreated, `{"type":"response.completed","response":`+bodyRefusal+`}`)
	if err != nil {
		t.Fatalf("streaming err = %v", err)
	}
	blocking, err := runBlocking(t, bodyRefusal)
	if err != nil {
		t.Fatalf("blocking err = %v", err)
	}
	if len(blocking) != 1 {
		t.Fatalf("blocking emitted %d responses, want 1", len(blocking))
	}
	for name, got := range map[string]*model.LLMResponse{
		"streamed": assertTurnShape(t, streamed),
		"blocking": blocking[0],
	} {
		if text := allText(got.Content); text != "helloI cannot" {
			t.Errorf("%s text = %q, want %q", name, text, "helloI cannot")
		}
		if got.LogprobsResult == nil {
			t.Errorf("%s LogprobsResult = nil, want the logprobs for the text it carries", name)
		}
	}
}

func functionCalls(resp *model.LLMResponse) []*genai.FunctionCall {
	if resp == nil || resp.Content == nil {
		return nil
	}
	var calls []*genai.FunctionCall
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			calls = append(calls, part.FunctionCall)
		}
	}
	return calls
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

// TestModel_GenerateStream_ErrorsEndTheTurn pins that every way a stream can go
// wrong ends the turn with that error rather than a response claiming the turn
// completed — one case per source of error the streaming path has.
func TestModel_GenerateStream_ErrorsEndTheTurn(t *testing.T) {
	tests := []struct {
		name    string
		events  []string
		wantErr string
	}{
		{
			name:    "response.failed",
			events:  []string{evCreated, evDelta1, evFailed},
			wantErr: "upstream exploded",
		},
		{
			name:    "error event",
			events:  []string{evCreated, evDelta1, evError},
			wantErr: "stream blew up",
		},
		{
			name:    "translator rejects the event",
			events:  []string{evCreated, `{"type":"response.function_call_arguments.done","item_id":"i1","name":"f","arguments":"{"}`},
			wantErr: "parse streamed function args",
		},
		{
			name:    "the stream itself fails",
			events:  []string{evCreated, evDelta1, evTruncated},
			wantErr: "unexpected end of JSON input",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := runStream(t, tc.events...)
			if err == nil {
				t.Fatalf("streaming err = nil, want one naming %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("streaming err = %v, want it to name %q", err, tc.wantErr)
			}
			for i, resp := range got {
				if resp.TurnComplete {
					t.Errorf("response %d has TurnComplete = true, want the error to end the turn", i)
				}
			}
		})
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
