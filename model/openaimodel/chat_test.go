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
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// chatRig serves one canned body and records the path and body it was asked
// for, which is how the tests below prove which endpoint was called.
type chatRig struct {
	server   *httptest.Server
	paths    []string
	requests []string
}

func newChatRig(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) *chatRig {
	t.Helper()
	rig := &chatRig{}
	rig.server = newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// ReadAll rather than one Read, which may return part of the body.
		body, _ := io.ReadAll(r.Body)
		rig.paths = append(rig.paths, r.URL.Path)
		rig.requests = append(rig.requests, string(body))
		handler(w, r)
	}))
	t.Cleanup(rig.server.Close)
	return rig
}

func (r *chatRig) model(t *testing.T) model.LLM {
	t.Helper()
	llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
		APIKey:     "test",
		BaseURL:    r.server.URL + "/v1",
		HTTPClient: r.server.Client(),
		API:        APIChatCompletions,
	})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	return llm
}

func chatJSON(body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, body)
	}
}

// chatSSE serves the frames as a Chat Completions stream, closed the way the
// API closes one.
func chatSSE(frames ...string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, frame := range frames {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}
}

func askChat(t *testing.T, llm model.LLM, stream bool) ([]*model.LLMResponse, error) {
	t.Helper()
	req := &model.LLMRequest{Contents: []*genai.Content{
		genai.NewContentFromText("weather?", genai.RoleUser),
	}}
	var (
		got      []*model.LLMResponse
		firstErr error
	)
	for resp, err := range llm.GenerateContent(t.Context(), req, stream) {
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

func TestNewModel_SelectsAPI(t *testing.T) {
	tests := []struct {
		name     string
		api      API
		wantPath string
	}{
		{name: "zero value keeps Responses", api: "", wantPath: "/v1/responses"},
		{name: "explicit Responses", api: APIResponses, wantPath: "/v1/responses"},
		{name: "chat completions", api: APIChatCompletions, wantPath: "/v1/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rig := newChatRig(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
			})
			llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
				APIKey:     "test",
				BaseURL:    rig.server.URL + "/v1",
				HTTPClient: rig.server.Client(),
				API:        tt.api,
			})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			_, _ = askChat(t, llm, false)
			if len(rig.paths) == 0 {
				t.Fatal("no request reached the server")
			}
			if rig.paths[0] != tt.wantPath {
				t.Errorf("path = %q, want %q", rig.paths[0], tt.wantPath)
			}
		})
	}
}

func TestNewModel_RejectsUnknownAPI(t *testing.T) {
	_, err := NewModel(t.Context(), "m", &ClientConfig{APIKey: "k", API: "grpc"})
	if !errors.Is(err, ErrUnsupportedAPI) {
		t.Fatalf("err = %v, want %v", err, ErrUnsupportedAPI)
	}
	if !strings.Contains(err.Error(), "grpc") {
		t.Errorf("err = %v, want it to name the value", err)
	}
}

func TestChatModel_GenerateContent_NilRequest(t *testing.T) {
	rig := newChatRig(t, chatJSON(`{}`))
	for _, err := range rig.model(t).GenerateContent(t.Context(), nil, false) {
		if !errors.Is(err, ErrRequestNil) {
			t.Fatalf("err = %v, want %v", err, ErrRequestNil)
		}
		return
	}
	t.Fatal("no response yielded")
}

func TestChatModel_GenerateContent_Text(t *testing.T) {
	rig := newChatRig(t, chatJSON(`{"id":"c1","model":"gpt-4o-mini",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"sunny"}}],
		"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
	got, err := askChat(t, rig.model(t), false)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("responses = %d, want 1", len(got))
	}
	if text := responseText(got[0]); text != "sunny" {
		t.Errorf("text = %q, want sunny", text)
	}
	if got[0].CustomMetadata["openai_response_id"] != "c1" {
		t.Errorf("response id metadata = %v", got[0].CustomMetadata["openai_response_id"])
	}
	// Recorded as a plain string, which is what this key has always been
	// documented to carry.
	if _, ok := got[0].CustomMetadata["openai_model"].(string); !ok {
		t.Errorf("model metadata = %T, want string", got[0].CustomMetadata["openai_model"])
	}
}

func TestChatModel_GenerateContent_ServerError(t *testing.T) {
	rig := newChatRig(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
	})
	if _, err := askChat(t, rig.model(t), false); err == nil {
		t.Fatal("err = nil, want the transport failure surfaced")
	}
}

func TestChatModel_GenerateStream_TextDeltas(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hail "}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"at -7 C"}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if len(got) < 2 {
		t.Fatalf("responses = %d, want partials plus a final", len(got))
	}
	final := got[len(got)-1]
	if !final.TurnComplete {
		t.Error("last response does not close the turn")
	}
	if text := responseText(final); text != "hail at -7 C" {
		t.Errorf("final text = %q", text)
	}
	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want STOP", final.FinishReason)
	}
	// Usage arrives only on the trailing chunk, and only because the request
	// asked for it.
	if final.UsageMetadata == nil || final.UsageMetadata.TotalTokenCount != 12 {
		t.Errorf("usage = %#v, want 12 total tokens", final.UsageMetadata)
	}
	if !strings.Contains(rig.requests[0], `"include_usage":true`) {
		t.Errorf("request did not ask for streamed usage: %s", rig.requests[0])
	}
}

// TestChatModel_GenerateStream_ToolArgumentsAcrossChunks covers the shape this
// endpoint has no event for: arguments arrive in fragments and the call is
// complete only once the stream ends.
func TestChatModel_GenerateStream_ToolArgumentsAcrossChunks(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Lisbon\"}"}}]}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	final := got[len(got)-1]
	var call *genai.FunctionCall
	for _, part := range final.Content.Parts {
		if part.FunctionCall != nil {
			call = part.FunctionCall
		}
	}
	if call == nil {
		t.Fatalf("no function call on the final response: %#v", final.Content.Parts)
	}
	if call.Name != "get_weather" || call.ID != "call_1" {
		t.Errorf("call = %#v", call)
	}
	if call.Args["city"] != "Lisbon" {
		t.Errorf("args = %#v, want the fragments reassembled", call.Args)
	}
}

func TestChatModel_GenerateStream_RefusalDeltas(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I cannot "}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"refusal":"help"}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	if text := responseText(got[len(got)-1]); text != "I cannot help" {
		t.Errorf("final text = %q, want the refusal as text", text)
	}
}

func TestChatModel_GenerateStream_EmptyStream(t *testing.T) {
	rig := newChatRig(t, chatSSE())
	_, err := askChat(t, rig.model(t), true)
	if !errors.Is(err, ErrNoChoices) {
		t.Fatalf("err = %v, want %v", err, ErrNoChoices)
	}
}

// TestChatModel_GenerateStream_UsageIsTheLatestReport covers a provider that
// resends the running usage on every chunk. Summing those reports would count
// the prompt once per chunk.
func TestChatModel_GenerateStream_UsageIsTheLatestReport(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hail "}}],"usage":{"prompt_tokens":8,"completion_tokens":1,"total_tokens":9}}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"at -7 C"}}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	usage := got[len(got)-1].UsageMetadata
	if usage == nil || usage.PromptTokenCount != 8 || usage.CandidatesTokenCount != 4 || usage.TotalTokenCount != 12 {
		t.Errorf("usage = %#v, want the last report: 8 prompt, 4 candidates, 12 total", usage)
	}
}

// TestChatModel_GenerateStream_UnparseableCallFailsAsBlocking pins that a turn
// whose streamed text survived still fails when its tool call cannot be read,
// because only the snapshot states the calls and blocking rejects the same
// body.
func TestChatModel_GenerateStream_UnparseableCallFailsAsBlocking(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"looking"}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	))
	_, err := askChat(t, rig.model(t), true)
	if !errors.Is(err, ErrFunctionCallArgs) {
		t.Fatalf("err = %v, want %v", err, ErrFunctionCallArgs)
	}
}

// TestChatModel_GenerateStream_EmptySnapshotKeepsStreamedText covers a delta
// the accumulator refuses, here for a choice index past its bound, while the
// text it carried still reached the caller as a partial.
func TestChatModel_GenerateStream_EmptySnapshotKeepsStreamedText(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":500,"delta":{"role":"assistant","content":"sunny"}}]}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err != nil {
		t.Fatalf("GenerateContent() err = %v", err)
	}
	final := got[len(got)-1]
	if text := responseText(final); text != "sunny" {
		t.Errorf("final text = %q, want the streamed text", text)
	}
	// Nothing states why the turn ended, which must not read as a clean stop.
	if final.FinishReason != genai.FinishReasonUnspecified {
		t.Errorf("finish reason = %v, want UNSPECIFIED", final.FinishReason)
	}
}

func TestChatModel_GenerateStream_ErrorMidStream(t *testing.T) {
	rig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hail "}}]}`,
		`{"error":{"message":"overloaded","type":"server_error"}}`,
	))
	got, err := askChat(t, rig.model(t), true)
	if err == nil {
		t.Fatal("err = nil, want the stream error surfaced")
	}
	if len(got) == 0 || responseText(got[0]) != "hail " {
		t.Errorf("responses = %d, want the partial that streamed before the error", len(got))
	}
	for _, resp := range got {
		if resp.TurnComplete {
			t.Error("a response closed the turn, but the stream failed")
		}
	}
}

// TestChatModel_GenerateStream_EarlyBreakClosesTheStream pins that a consumer
// leaving the range releases the connection rather than leaving the provider
// streaming into a reader that is gone.
func TestChatModel_GenerateStream_EarlyBreakClosesTheStream(t *testing.T) {
	released := make(chan struct{})
	rig := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hail "}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(released)
	})
	for resp, err := range rig.model(t).GenerateContent(t.Context(), toolReq(nil), true) {
		if err != nil {
			t.Fatalf("GenerateContent() err = %v", err)
		}
		if resp.Partial {
			break
		}
	}
	select {
	case <-released:
	case <-time.After(5 * time.Second):
		t.Fatal("the connection was still open after the consumer stopped")
	}
}

func TestChatModel_GenerateStream_HonoursTimeout(t *testing.T) {
	rig := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	timeout := time.Nanosecond
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}},
	}
	var gotErr error
	for _, err := range rig.model(t).GenerateContent(t.Context(), req, true) {
		if err != nil {
			gotErr = err
		}
	}
	if !errors.Is(gotErr, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want a deadline error", gotErr)
	}
}

// TestChatModel_GenerateStream_MatchesBlocking is the guard that keeps the two
// paths from drifting: the same turn must read the same whether it streamed.
func TestChatModel_GenerateStream_MatchesBlocking(t *testing.T) {
	blockingRig := newChatRig(t, chatJSON(`{"id":"c","model":"m",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"looking",
		"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lisbon\"}"}}]}}],
		"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}`))
	streamRig := newChatRig(t, chatSSE(
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"looking"}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lisbon\"}"}}]}}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4,"total_tokens":12}}`,
	))

	blocking, err := askChat(t, blockingRig.model(t), false)
	if err != nil {
		t.Fatalf("blocking err = %v", err)
	}
	streamed, err := askChat(t, streamRig.model(t), true)
	if err != nil {
		t.Fatalf("streamed err = %v", err)
	}

	want, got := blocking[0], streamed[len(streamed)-1]
	if responseText(want) != responseText(got) {
		t.Errorf("text: blocking %q, streamed %q", responseText(want), responseText(got))
	}
	if want.FinishReason != got.FinishReason {
		t.Errorf("finish reason: blocking %v, streamed %v", want.FinishReason, got.FinishReason)
	}
	if want.UsageMetadata.TotalTokenCount != got.UsageMetadata.TotalTokenCount {
		t.Errorf("usage: blocking %d, streamed %d",
			want.UsageMetadata.TotalTokenCount, got.UsageMetadata.TotalTokenCount)
	}
	wantCall, gotCall := onlyCall(t, want), onlyCall(t, got)
	if wantCall.Name != gotCall.Name || wantCall.ID != gotCall.ID {
		t.Errorf("call: blocking %#v, streamed %#v", wantCall, gotCall)
	}
	if wantCall.Args["city"] != gotCall.Args["city"] {
		t.Errorf("args: blocking %#v, streamed %#v", wantCall.Args, gotCall.Args)
	}
}

func TestChatModel_HonoursTimeout(t *testing.T) {
	rig := newChatRig(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	timeout := time.Nanosecond
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}},
	}
	var gotErr error
	for _, err := range rig.model(t).GenerateContent(t.Context(), req, false) {
		if err != nil {
			gotErr = err
		}
	}
	if !errors.Is(gotErr, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want a deadline error", gotErr)
	}
}

func onlyCall(t *testing.T, resp *model.LLMResponse) *genai.FunctionCall {
	t.Helper()
	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			return part.FunctionCall
		}
	}
	t.Fatalf("no function call on %#v", resp.Content.Parts)
	return nil
}

func responseText(resp *model.LLMResponse) string {
	if resp == nil || resp.Content == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range resp.Content.Parts {
		if part != nil && !part.Thought {
			b.WriteString(part.Text)
		}
	}
	return b.String()
}
