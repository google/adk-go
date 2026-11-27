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

package completions

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// truncationTurn is one assistant turn, sent either as a blocking body or as
// a stream.
type truncationTurn struct {
	text         string
	args         []string // one get_weather call per entry
	finishReason string
}

func (turn truncationTurn) toolCalls(indexed bool) string {
	calls := make([]string, len(turn.args))
	for i, args := range turn.args {
		quoted, _ := json.Marshal(args)
		index := ""
		if indexed {
			index = fmt.Sprintf(`"index":%d,`, i)
		}
		calls[i] = fmt.Sprintf(`{%s"id":"call_%d","type":"function","function":{"name":"get_weather","arguments":%s}}`, index, i, quoted)
	}
	return "[" + strings.Join(calls, ",") + "]"
}

func (turn truncationTurn) serve(stream bool) func(w http.ResponseWriter, r *http.Request) {
	text, _ := json.Marshal(turn.text)
	if !stream {
		return serveJSON(fmt.Sprintf(`{"id":"c","model":"m","choices":[{"index":0,"finish_reason":%q,"message":{"role":"assistant","content":%s,"tool_calls":%s}}]}`,
			turn.finishReason, text, turn.toolCalls(false)))
	}
	return serveSSE(
		fmt.Sprintf(`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":%s}}]}`, text),
		fmt.Sprintf(`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":%s}}]}`, turn.toolCalls(true)),
		fmt.Sprintf(`{"id":"c","model":"m","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":%q}]}`, turn.finishReason),
	)
}

// TestModel_TruncatedToolCalls covers a turn the token limit cut off inside a
// call's arguments: its calls are dropped, its text is kept, and a turn left
// with nothing reports MAX_TOKENS as an error. Each case runs blocking and
// streamed.
func TestModel_TruncatedToolCalls(t *testing.T) {
	tests := []struct {
		name          string
		turn          truncationTurn
		wantText      string
		wantCalls     int
		wantFinish    genai.FinishReason
		wantErrorCode string
		wantErr       error
	}{
		{
			name:       "empty arguments, text kept",
			turn:       truncationTurn{text: "Checking.", args: []string{""}, finishReason: "length"},
			wantText:   "Checking.",
			wantFinish: genai.FinishReasonMaxTokens,
		},
		{
			name:          "partial arguments, nothing left",
			turn:          truncationTurn{args: []string{`{"city":"Lis`}, finishReason: "length"},
			wantFinish:    genai.FinishReasonMaxTokens,
			wantErrorCode: string(genai.FinishReasonMaxTokens),
		},
		{
			name:       "a complete call goes with the unfinished one",
			turn:       truncationTurn{text: "Checking.", args: []string{`{"city":"Lisbon"}`, `{"city":"Por`}, finishReason: "length"},
			wantText:   "Checking.",
			wantFinish: genai.FinishReasonMaxTokens,
		},
		{
			name:       "complete arguments are kept",
			turn:       truncationTurn{args: []string{`{"city":"Lisbon"}`}, finishReason: "length"},
			wantCalls:  1,
			wantFinish: genai.FinishReasonMaxTokens,
		},
		{
			name:       "empty arguments outside a length stop still run",
			turn:       truncationTurn{args: []string{""}, finishReason: "tool_calls"},
			wantCalls:  1,
			wantFinish: genai.FinishReasonStop,
		},
		{
			name:    "partial arguments outside a length stop still fail",
			turn:    truncationTurn{args: []string{`{"city":"Lis`}, finishReason: "tool_calls"},
			wantErr: shared.ErrFunctionCallArgs,
		},
	}
	for _, tt := range tests {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tt.name, stream), func(t *testing.T) {
				got, err := ask(t, newTestRig(t, tt.turn.serve(stream)).model(t), stream)
				if tt.wantErr != nil {
					if !errors.Is(err, tt.wantErr) {
						t.Fatalf("err = %v, want %v", err, tt.wantErr)
					}
					return
				}
				if err != nil {
					t.Fatalf("GenerateContent() err = %v", err)
				}
				final := got[len(got)-1]
				if text := responseText(final); text != tt.wantText {
					t.Errorf("text = %q, want %q", text, tt.wantText)
				}
				if calls := countCalls(final); calls != tt.wantCalls {
					t.Errorf("function calls = %d, want %d", calls, tt.wantCalls)
				}
				if final.FinishReason != tt.wantFinish {
					t.Errorf("finish reason = %v, want %v", final.FinishReason, tt.wantFinish)
				}
				if final.ErrorCode != tt.wantErrorCode {
					t.Errorf("error code = %q, want %q", final.ErrorCode, tt.wantErrorCode)
				}
				if (final.ErrorMessage != "") != (tt.wantErrorCode != "") {
					t.Errorf("error message = %q, want one only with an error code", final.ErrorMessage)
				}
			})
		}
	}
}

// TestConvertCompletion_TruncatedTurnLeavesOtherCallsAlone pins what a
// length stop does not count as an unfinished call.
func TestConvertCompletion_TruncatedTurnLeavesOtherCallsAlone(t *testing.T) {
	t.Run("accumulator padding", func(t *testing.T) {
		// A stream that skipped tool-call index 0 leaves an empty entry there.
		resp := decodeCompletion(t, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","tool_calls":[
			{"type":"function","function":{"name":"","arguments":""}},
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lisbon\"}"}}]}}]}`)
		got, err := convertCompletion(resp)
		if err != nil {
			t.Fatalf("convertCompletion() err = %v", err)
		}
		parts := got.Candidates[0].Content.Parts
		if len(parts) != 1 || parts[0].FunctionCall == nil || parts[0].FunctionCall.ID != "call_1" {
			t.Errorf("parts = %#v, want the call_1 call", parts)
		}
	})
	t.Run("unsupported call type", func(t *testing.T) {
		// Such a turn fails whatever its finish reason, rather than being
		// passed off as one that only lost its calls.
		resp := decodeCompletion(t, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","tool_calls":[
			{"id":"call_0","type":"custom","custom":{"name":"grep","input":"x"}},
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}`)
		if _, err := convertCompletion(resp); !errors.Is(err, shared.ErrUnsupportedOutputItemType) {
			t.Fatalf("err = %v, want %v", err, shared.ErrUnsupportedOutputItemType)
		}
	})
}

func countCalls(resp *model.LLMResponse) int {
	n := 0
	if resp.Content != nil {
		for _, part := range resp.Content.Parts {
			if part != nil && part.FunctionCall != nil {
				n++
			}
		}
	}
	return n
}
