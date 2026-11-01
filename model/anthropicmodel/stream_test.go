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

package anthropicmodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// generateStreamingReference is generateStreaming with every event passed
// straight to anthropic.Message.Accumulate.
func generateStreamingReference(
	ctx context.Context,
	m *anthropicModel,
	params anthropic.MessageNewParams,
	reqOpts []option.RequestOption,
	unwrapOutput bool,
	yield func(*model.LLMResponse, error) bool,
) {
	stream := m.client.Messages.NewStreaming(ctx, params, slices.Clone(reqOpts)...)
	defer func() {
		_ = stream.Close()
	}()

	var acc anthropic.Message
	toolInputBuffers := make(map[int64]*strings.Builder)

	for stream.Next() {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			yield(nil, classifyError(err))
			return
		}

		switch event.Type {
		case "content_block_start":
			switch event.ContentBlock.Type {
			case "text":
				if !unwrapOutput && event.ContentBlock.Text != "" {
					if !yieldPartialPart(yield, string(acc.Model), genai.NewPartFromText(event.ContentBlock.Text)) {
						return
					}
				}
			case "thinking":
				if event.ContentBlock.Thinking != "" {
					if !yieldPartialPart(yield, string(acc.Model), &genai.Part{
						Thought: true,
						Text:    event.ContentBlock.Thinking,
					}) {
						return
					}
				}
			case "tool_use":
				toolInputBuffers[event.Index] = &strings.Builder{}
			}

		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				if !unwrapOutput && event.Delta.Text != "" {
					if !yieldPartialPart(yield, string(acc.Model), genai.NewPartFromText(event.Delta.Text)) {
						return
					}
				}
			case "thinking_delta":
				if event.Delta.Thinking != "" {
					if !yieldPartialPart(yield, string(acc.Model), &genai.Part{
						Thought: true,
						Text:    event.Delta.Thinking,
					}) {
						return
					}
				}
			case "input_json_delta":
				if event.Delta.PartialJSON != "" {
					buf, ok := toolInputBuffers[event.Index]
					if !ok {
						buf = &strings.Builder{}
						toolInputBuffers[event.Index] = buf
					}
					buf.WriteString(event.Delta.PartialJSON)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		yield(nil, classifyError(err))
		return
	}
	if err := ctx.Err(); err != nil {
		yield(nil, err)
		return
	}

	if acc.StopReason != anthropic.StopReasonMaxTokens && acc.StopReason != anthropic.StopReasonModelContextWindowExceeded {
		for idx, buf := range toolInputBuffers {
			if idx >= 0 && int(idx) < len(acc.Content) && buf.Len() > 0 {
				acc.Content[idx].Input = json.RawMessage(buf.String())
			}
		}
	}

	finalResp, err := convertMessageToLLMResponse(&acc, unwrapOutput)
	if err != nil {
		yield(nil, err)
		return
	}
	yield(finalResp, nil)
}

const (
	evMessageStart = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`
	evMessageStop  = `{"type":"message_stop"}`
)

func jsonQuote(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func evTextStart(i int, text string) string {
	return fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":%s}}`, i, jsonQuote(text))
}

func evThinkingStart(i int) string {
	return fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"thinking","thinking":"","signature":""}}`, i)
}

func evToolStart(i int, id, name string) string {
	return fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":%s,"name":%s,"input":{}}}`, i, jsonQuote(id), jsonQuote(name))
}

func evDelta(i int, deltaType, field, value string) string {
	return fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":%s,%s:%s}}`, i, jsonQuote(deltaType), jsonQuote(field), jsonQuote(value))
}

func evText(i int, s string) string      { return evDelta(i, "text_delta", "text", s) }
func evThinking(i int, s string) string  { return evDelta(i, "thinking_delta", "thinking", s) }
func evSignature(i int, s string) string { return evDelta(i, "signature_delta", "signature", s) }
func evInput(i int, s string) string     { return evDelta(i, "input_json_delta", "partial_json", s) }

func evCitation(i int, cited string) string {
	return fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":%s,"document_index":0,"document_title":"Doc","start_char_index":0,"end_char_index":%d}}}`, i, jsonQuote(cited), len(cited))
}

func evStop(i int) string {
	return fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, i)
}

func evMessageDelta(stopReason string, outputTokens int) string {
	return fmt.Sprintf(`{"type":"message_delta","delta":{"stop_reason":%s,"stop_sequence":null},"usage":{"output_tokens":%d}}`, jsonQuote(stopReason), outputTokens)
}

// sseBody renders events as a server-sent event stream. An event that starts
// with "event:" is used verbatim; any other event is a JSON object sent under
// the name in its "type" field.
func sseBody(events []string) string {
	var b strings.Builder
	for _, ev := range events {
		if strings.HasPrefix(ev, "event:") {
			b.WriteString(ev)
			continue
		}
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(ev), &head); err != nil {
			panic(fmt.Sprintf("invalid test event %s: %v", ev, err))
		}
		b.WriteString("event: " + head.Type + "\ndata: " + ev + "\n\n")
	}
	return b.String()
}

type streamCase struct {
	name   string
	events []string
}

// streamCases is the corpus both equivalence tests run over.
func streamCases() []streamCase {
	large := func(delta func(int, string) string, start string, extra ...string) []string {
		events := []string{evMessageStart, start}
		chunk := strings.Repeat("lorem ipsum ", 5)[:50]
		for range 2000 {
			events = append(events, delta(0, chunk))
		}
		events = append(events, extra...)
		return append(events, evStop(0), evMessageDelta("end_turn", 2000), evMessageStop)
	}
	return []streamCase{
		{"plain_text", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "Hello"), evText(0, ", "), evText(0, "world"), evText(0, "!"),
			evStop(0), evMessageDelta("end_turn", 5), evMessageStop,
		}},
		{"text_in_block_start", []string{
			evMessageStart, evTextStart(0, "Hi"), evText(0, " there"), evText(0, "."),
			evStop(0), evMessageDelta("end_turn", 3), evMessageStop,
		}},
		{"thinking_signature_then_text", []string{
			evMessageStart, evThinkingStart(0), evThinking(0, "Let me "), evThinking(0, "think."), evSignature(0, "sig_"), evSignature(0, "abc"), evStop(0),
			evTextStart(1, ""), evText(1, "4"), evText(1, "2"), evStop(1),
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":12,"cache_creation_input_tokens":7,"cache_read_input_tokens":5,"output_tokens":30,"output_tokens_details":{"thinking_tokens":18}}}`,
			evMessageStop,
		}},
		{"redacted_thinking", []string{
			evMessageStart, `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"enc_blob"}}`, evStop(0),
			evTextStart(1, ""), evText(1, "ok"), evStop(1), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"text_tool_use_text", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "Checking"), evText(0, " weather."), evStop(0),
			evToolStart(1, "toolu_1", "get_weather"), evInput(1, `{"city":`), evInput(1, `"Warsaw"`), evInput(1, `,"days":3}`), evStop(1),
			evTextStart(2, ""), evText(2, "Done"), evStop(2), evMessageDelta("tool_use", 20), evMessageStop,
		}},
		{"tool_use_without_input_deltas", []string{
			evMessageStart, evToolStart(0, "toolu_2", "get_time"), evStop(0), evMessageDelta("tool_use", 4), evMessageStop,
		}},
		{"citations", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "The sky "), evCitation(0, "the sky is blue"), evText(0, "is blue"),
			evCitation(0, "blue"), evText(0, "."), evStop(0), evMessageDelta("end_turn", 6), evMessageStop,
		}},
		{"empty_deltas", []string{
			evMessageStart, evTextStart(0, ""), evText(0, ""), evText(0, "a"), evText(0, ""), evStop(0),
			evThinkingStart(1), evThinking(1, ""), evThinking(1, "b"), evThinking(1, ""), evStop(1),
			evTextStart(2, ""), evText(2, ""), evStop(2), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"unicode_split_across_deltas", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "Zaż"), evText(0, "ółć "), evText(0, "😀"), evText(0, "e"), evText(0, "\u0301"),
			evText(0, "中"), evText(0, "文"),
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\ud83d"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\ude00"}}`,
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"bad \xff\xfe utf8\"}}\n\n",
			evStop(0), evThinkingStart(1), evThinking(1, "ł"), evThinking(1, "ą"), evStop(1), evMessageDelta("end_turn", 9), evMessageStop,
		}},
		{"json_escapes", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "<b>"), evText(0, "&amp;"), evText(0, `"quoted"`), evText(0, `back\slash`),
			evText(0, "new\nline\ttab\x00nul"), evStop(0), evMessageDelta("end_turn", 6), evMessageStop,
		}},
		{"error_event_mid_stream", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "partial "), evText(0, "answer"),
			"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
		}},
		{"malformed_event_mid_stream", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "partial"),
			"event: content_block_delta\ndata: {\"type\":\n\n",
		}},
		{"text_delta_index_out_of_range", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "a"), evText(3, "b"), evStop(0), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"thinking_delta_before_any_block", []string{
			evMessageStart, evThinking(0, "x"), evMessageDelta("end_turn", 1), evMessageStop,
		}},
		{"negative_delta_index", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "a"), evText(-1, "b"), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"block_start_index_gap", []string{
			evMessageStart, evTextStart(1, "x"), evText(1, "y"), evMessageDelta("end_turn", 1), evMessageStop,
		}},
		{"block_stop_index_out_of_range_with_pending_text", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "a"), evText(0, "b"), evStop(4), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"truncated_without_message_stop", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "cut "), evText(0, "off"),
		}},
		{"truncated_after_message_delta", []string{
			evMessageStart, evThinkingStart(0), evThinking(0, "hmm"), evThinking(0, "..."), evMessageDelta("end_turn", 2),
		}},
		{"max_tokens_with_cut_tool_use", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "Let me "), evText(0, "check."), evStop(0),
			evToolStart(1, "toolu_3", "get_weather"), evInput(1, `{"city": "Wars`), evStop(1), evMessageDelta("max_tokens", 20), evMessageStop,
		}},
		{"deltas_after_block_stop", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "a"), evStop(0), evText(0, "late"), evThinking(0, "later"),
			evMessageDelta("end_turn", 3), evMessageStop,
		}},
		{"second_message_start", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "discarded"), evThinkingStart(1), evThinking(1, "gone"),
			evMessageStart, evTextStart(0, ""), evText(0, "kept"), evStop(0), evMessageDelta("end_turn", 1), evMessageStop,
		}},
		{"mismatched_delta_types", []string{
			evMessageStart, evTextStart(0, ""), evThinking(0, "t0"), evText(0, "x0"),
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x1","thinking":"ignored"}}`, evStop(0),
			evThinkingStart(1), evText(1, "x1"), evThinking(1, "t1"),
			`{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"t2","text":"ignored"}}`, evStop(1),
			evMessageDelta("end_turn", 4), evMessageStop,
		}},
		{"interleaved_blocks", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "a"), evThinkingStart(1), evThinking(1, "b"), evText(0, "c"),
			evToolStart(2, "toolu_4", "noop"), evThinking(1, "d"), evInput(2, `{}`), evStop(0), evThinking(1, "e"), evStop(1), evStop(2),
			evMessageDelta("tool_use", 5), evMessageStop,
		}},
		{"unknown_types_and_pings", []string{
			evMessageStart, "event: ping\ndata: {\"type\": \"ping\"}\n\n", evTextStart(0, ""), evText(0, "a"),
			"event: message_delta\ndata: {\"type\":\"bogus\"}\n\n",
			`{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","text":"ignored","thinking":"ignored"}}`,
			"event: future_event\ndata: {\"type\":\"future_event\"}\n\n",
			evText(0, "b"), "event: ping\ndata: {\"type\": \"ping\"}\n\n", evStop(0), evMessageDelta("end_turn", 2), evMessageStop,
		}},
		{"empty_message", []string{evMessageStart, evMessageDelta("end_turn", 0), evMessageStop}},
		{"stop_sequence", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "one two"), evStop(0),
			`{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"END"},"usage":{"output_tokens":3}}`, evMessageStop,
		}},
		{"refusal", []string{
			evMessageStart, evTextStart(0, ""), evText(0, "I can't"), evText(0, " help."), evStop(0), evMessageDelta("refusal", 3), evMessageStop,
		}},
		{"output_schema_wrapper", []string{
			evMessageStart, evTextStart(0, `{"res`), evText(0, `ult": {"city"`), evText(0, `: "Warsaw", `), evText(0, `"temp_c": 18}}`),
			evStop(0), evMessageDelta("end_turn", 8), evMessageStop,
		}},
		{"large_text", large(evText, evTextStart(0, ""))},
		{"large_thinking", large(evThinking, evThinkingStart(0), evSignature(0, "sig_large"))},
	}
}

func sseResponse(body string, req *http.Request) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}
}

// TestAccumulateMatchesMessageAccumulate checks that accumulate and flushDeltas
// build exactly the message that anthropic.Message.Accumulate builds, with the
// same errors, and that the message plus the pending deltas equals it after
// every event.
func TestAccumulateMatchesMessageAccumulate(t *testing.T) {
	for _, tc := range streamCases() {
		t.Run(tc.name, func(t *testing.T) {
			stream := ssestream.NewStream[anthropic.MessageStreamEventUnion](ssestream.NewDecoder(sseResponse(sseBody(tc.events), nil)), nil)
			var want, got anthropic.Message
			pending := make(map[int64]*blockDeltas)
			for i := 0; stream.Next(); i++ {
				event := stream.Current()
				wantErr := want.Accumulate(event)
				gotErr := accumulate(&got, event, pending)
				if fmt.Sprint(gotErr) != fmt.Sprint(wantErr) {
					t.Fatalf("event %d: accumulate() error = %v, want %v", i, gotErr, wantErr)
				}
				if wantErr != nil {
					return
				}
				withPending := got
				withPending.Content = slices.Clone(got.Content)
				for idx, d := range pending {
					withPending.Content[idx].Text += d.text.String()
					withPending.Content[idx].Thinking += d.thinking.String()
				}
				if !reflect.DeepEqual(withPending, want) {
					t.Fatalf("event %d: accumulated message plus pending deltas differs:\ngot  %s\nwant %s", i, messageDump(withPending), messageDump(want))
				}
			}
			flushDeltas(&got, pending)
			if len(pending) != 0 {
				t.Errorf("flushDeltas() left %d pending blocks", len(pending))
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("accumulated message differs:\ngot  %s\nwant %s", messageDump(got), messageDump(want))
			}
		})
	}
}

func messageDump(msg anthropic.Message) string {
	return fmt.Sprintf("raw=%s content=%+v", msg.RawJSON(), msg.Content)
}

type streamStep struct {
	Resp *model.LLMResponse
	Err  string
}

// describeError renders everything a caller can observe about err: its type,
// message, and the sentinel errors it matches.
func describeError(err error) string {
	if err == nil {
		return ""
	}
	s := fmt.Sprintf("%T: %v", err, err)
	for _, target := range []error{
		context.Canceled, context.DeadlineExceeded, ErrNoContent, ErrInvalidConfig, ErrUnsupportedToolType,
		ErrUnsupportedContent, ErrAuthentication, ErrBadRequest, ErrRateLimited, ErrOverloaded, ErrAPIStatus,
	} {
		if errors.Is(err, target) {
			s += fmt.Sprintf(" [is %q]", target)
		}
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		s += fmt.Sprintf(" %+v", *apiErr)
	}
	return s
}

type streamFunc func(ctx context.Context, yield func(*model.LLMResponse, error) bool)

// collectSteps runs gen and records what it yields. The consumer cancels the
// context after cancelAfter values and stops after stopAfter values; zero
// disables either.
func collectSteps(gen streamFunc, stopAfter, cancelAfter int) []streamStep {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var steps []streamStep
	gen(ctx, func(resp *model.LLMResponse, err error) bool {
		steps = append(steps, streamStep{Resp: resp, Err: describeError(err)})
		if len(steps) == cancelAfter {
			cancel()
		}
		return len(steps) != stopAfter
	})
	return steps
}

// TestGenerateStreamingMatchesReference checks that generateStreaming yields
// the same partial and final responses and errors as
// generateStreamingReference, also when the consumer stops early or cancels.
func TestGenerateStreamingMatchesReference(t *testing.T) {
	for _, tc := range streamCases() {
		t.Run(tc.name, func(t *testing.T) {
			body := sseBody(tc.events)
			llm, err := NewModel(t.Context(), "claude-sonnet-4-5-20250929", ClientConfig{
				APIKey: "test-key",
				HTTPClient: &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
					return sseResponse(body, r), nil
				})},
			})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}
			m := llm.(*anthropicModel)
			params, reqOpts, _, err := m.buildMessageParams(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")})
			if err != nil {
				t.Fatalf("buildMessageParams() error = %v", err)
			}
			for _, unwrap := range []bool{false, true} {
				got := streamFunc(func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
					m.generateStreaming(ctx, params, reqOpts, unwrap, yield)
				})
				want := streamFunc(func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
					generateStreamingReference(ctx, m, params, reqOpts, unwrap, yield)
				})
				full := collectSteps(want, 0, 0)
				if diff := cmp.Diff(full, collectSteps(got, 0, 0)); diff != "" {
					t.Fatalf("unwrap=%v: generateStreaming() mismatch (-want +got):\n%s", unwrap, diff)
				}
				for _, k := range []int{1, 2, len(full) - 1} {
					if k < 1 || k >= len(full) {
						continue
					}
					if diff := cmp.Diff(collectSteps(want, k, 0), collectSteps(got, k, 0)); diff != "" {
						t.Errorf("unwrap=%v, stop after %d: generateStreaming() mismatch (-want +got):\n%s", unwrap, k, diff)
					}
					if diff := cmp.Diff(collectSteps(want, 0, k), collectSteps(got, 0, k)); diff != "" {
						t.Errorf("unwrap=%v, cancel after %d: generateStreaming() mismatch (-want +got):\n%s", unwrap, k, diff)
					}
				}
			}
		})
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
