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

package completions

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
)

// decodeCompletion builds a ChatCompletion from the JSON a provider would send,
// so the tests exercise the SDK's own decoding rather than hand-built structs.
func decodeCompletion(t *testing.T, body string) *openai.ChatCompletion {
	t.Helper()
	var resp openai.ChatCompletion
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("unmarshal completion: %v", err)
	}
	return &resp
}

func TestConvertCompletion_Text(t *testing.T) {
	resp := decodeCompletion(t, `{
		"id":"chatcmpl-1","model":"gpt-4o-mini","object":"chat.completion",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hail at -7 C"}}],
		"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}
	}`)
	got, err := convertCompletion(resp)
	if err != nil {
		t.Fatalf("convertCompletion() err = %v", err)
	}
	if got.ResponseID != "chatcmpl-1" || got.ModelVersion != "gpt-4o-mini" {
		t.Errorf("id/model = %q/%q", got.ResponseID, got.ModelVersion)
	}
	cand := got.Candidates[0]
	if len(cand.Content.Parts) != 1 || cand.Content.Parts[0].Text != "hail at -7 C" {
		t.Errorf("parts = %#v", cand.Content.Parts)
	}
	if cand.Content.Role != string(genai.RoleModel) {
		t.Errorf("role = %q, want model", cand.Content.Role)
	}
	if cand.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want STOP", cand.FinishReason)
	}
	if got.UsageMetadata.PromptTokenCount != 11 || got.UsageMetadata.CandidatesTokenCount != 5 {
		t.Errorf("usage = %#v", got.UsageMetadata)
	}
}

func TestConvertCompletion_Refusal(t *testing.T) {
	resp := decodeCompletion(t, `{
		"id":"c","model":"m","choices":[{"index":0,"finish_reason":"stop",
		"message":{"role":"assistant","content":"","refusal":"I cannot help with that"}}]}`)
	got, err := convertCompletion(resp)
	if err != nil {
		t.Fatalf("convertCompletion() err = %v", err)
	}
	// Flattened to text, matching what the Responses path does with a refusal
	// content block, so one refusal reads the same on either endpoint.
	parts := got.Candidates[0].Content.Parts
	if len(parts) != 1 || parts[0].Text != "I cannot help with that" {
		t.Errorf("parts = %#v, want the refusal as text", parts)
	}
}

func TestConvertCompletion_ToolCalls(t *testing.T) {
	resp := decodeCompletion(t, `{
		"id":"c","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{
			"role":"assistant","content":"",
			"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lisbon\"}"}}]
		}}]}`)
	got, err := convertCompletion(resp)
	if err != nil {
		t.Fatalf("convertCompletion() err = %v", err)
	}
	parts := got.Candidates[0].Content.Parts
	if len(parts) != 1 || parts[0].FunctionCall == nil {
		t.Fatalf("parts = %#v, want one function call", parts)
	}
	call := parts[0].FunctionCall
	if call.Name != "get_weather" || call.ID != "call_1" || call.Args["city"] != "Lisbon" {
		t.Errorf("call = %#v", call)
	}
	// A turn that ends by calling a tool stopped cleanly.
	if got.Candidates[0].FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want STOP", got.Candidates[0].FinishReason)
	}
}

func TestConvertCompletion_ParallelToolCalls(t *testing.T) {
	resp := decodeCompletion(t, `{
		"id":"c","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{
			"role":"assistant","content":"working on it",
			"tool_calls":[
				{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},
				{"id":"b","type":"function","function":{"name":"g","arguments":"{\"n\":3}"}}
			]}}]}`)
	got, err := convertCompletion(resp)
	if err != nil {
		t.Fatalf("convertCompletion() err = %v", err)
	}
	parts := got.Candidates[0].Content.Parts
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want text plus two calls", len(parts))
	}
	if parts[0].Text != "working on it" {
		t.Errorf("first part = %#v, want the text", parts[0])
	}
	if parts[1].FunctionCall.ID != "a" || parts[2].FunctionCall.ID != "b" {
		t.Errorf("call order = %q,%q, want a,b", parts[1].FunctionCall.ID, parts[2].FunctionCall.ID)
	}
	// An empty argument payload is a call that takes none, not a nil map.
	if parts[1].FunctionCall.Args == nil || len(parts[1].FunctionCall.Args) != 0 {
		t.Errorf("args = %#v, want an empty map", parts[1].FunctionCall.Args)
	}
}

// TestConvertCompletion_EmptyToolArguments covers the two ways a provider
// says a call takes no arguments besides "{}": leaving them out, and JSON null.
// Either must reach the tool as an empty map rather than a nil one.
func TestConvertCompletion_EmptyToolArguments(t *testing.T) {
	for _, args := range []string{``, `null`} {
		t.Run(fmt.Sprintf("%q", args), func(t *testing.T) {
			raw, err := json.Marshal(args)
			if err != nil {
				t.Fatal(err)
			}
			resp := decodeCompletion(t, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{
				"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":`+string(raw)+`}}]}}]}`)
			got, err := convertCompletion(resp)
			if err != nil {
				t.Fatalf("convertCompletion() err = %v", err)
			}
			call := got.Candidates[0].Content.Parts[0].FunctionCall
			if call == nil || call.Args == nil || len(call.Args) != 0 {
				t.Errorf("call = %#v, want an empty, non-nil argument map", call)
			}
		})
	}
}

func TestConvertCompletion_UnparseableToolArguments(t *testing.T) {
	resp := decodeCompletion(t, `{
		"id":"c","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{
			"role":"assistant","tool_calls":[{"id":"call_9","type":"function",
			"function":{"name":"get_weather","arguments":"{not json"}}]}}]}`)
	_, err := convertCompletion(resp)
	if !errors.Is(err, openaicommon.ErrFunctionCallArgs) {
		t.Fatalf("err = %v, want %v", err, openaicommon.ErrFunctionCallArgs)
	}
	// Conversion aborts on the first bad call, so the error has to identify it.
	if !strings.Contains(err.Error(), "get_weather") || !strings.Contains(err.Error(), "call_9") {
		t.Errorf("err = %v, want it to name the call", err)
	}
}

func TestConvertCompletion_Empty(t *testing.T) {
	tests := []struct {
		name string
		resp *openai.ChatCompletion
		want error
	}{
		{name: "nil", resp: nil, want: openaicommon.ErrEmptyResponse},
		{name: "no choices", resp: decodeCompletion(t, `{"id":"c","model":"m","choices":[]}`), want: openaicommon.ErrNoChoices},
		{
			name: "choice with nothing to read",
			resp: decodeCompletion(t, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":""}}]}`),
			want: openaicommon.ErrNoTextOrToolContent,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := convertCompletion(tt.resp); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestFinishReason(t *testing.T) {
	tests := []struct {
		reason string
		want   genai.FinishReason
	}{
		{reason: "stop", want: genai.FinishReasonStop},
		{reason: "tool_calls", want: genai.FinishReasonStop},
		{reason: "function_call", want: genai.FinishReasonStop},
		{reason: "length", want: genai.FinishReasonMaxTokens},
		{reason: "content_filter", want: genai.FinishReasonSafety},
		// Silence is not a clean stop: a caller that retries on anything but
		// STOP must not accept a partial answer as final.
		{reason: "", want: genai.FinishReasonUnspecified},
		{reason: "something_new", want: genai.FinishReasonOther},
	}
	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			if got := finishReason(tt.reason); got != tt.want {
				t.Errorf("finishReason(%q) = %v, want %v", tt.reason, got, tt.want)
			}
		})
	}
}

func TestConvertUsage(t *testing.T) {
	resp := decodeCompletion(t, `{"id":"c","model":"m",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"x"}}],
		"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,
			"prompt_tokens_details":{"cached_tokens":40},
			"completion_tokens_details":{"reasoning_tokens":7}}}`)
	got := convertUsage(resp.Usage)
	// The field names differ from the Responses API's on every count, so this
	// pins the mapping rather than the arithmetic.
	if got.PromptTokenCount != 100 || got.CandidatesTokenCount != 20 || got.TotalTokenCount != 120 {
		t.Errorf("totals = %#v", got)
	}
	if got.CachedContentTokenCount != 40 {
		t.Errorf("cached = %d, want 40", got.CachedContentTokenCount)
	}
	if got.ThoughtsTokenCount != 7 {
		t.Errorf("reasoning = %d, want 7", got.ThoughtsTokenCount)
	}
}

func TestConvertLogprobs(t *testing.T) {
	resp := decodeCompletion(t, `{"id":"c","model":"m","choices":[{"index":0,"finish_reason":"stop",
		"message":{"role":"assistant","content":"hi"},
		"logprobs":{"content":[{"token":"hi","logprob":-0.25,
			"top_logprobs":[{"token":"hi","logprob":-0.25},{"token":"yo","logprob":-2.5}]}]}}]}`)
	got := convertLogprobs(resp.Choices[0].Logprobs)
	if got == nil || len(got.ChosenCandidates) != 1 {
		t.Fatalf("logprobs = %#v", got)
	}
	if got.ChosenCandidates[0].Token != "hi" {
		t.Errorf("chosen token = %q", got.ChosenCandidates[0].Token)
	}
	if len(got.TopCandidates) != 1 || len(got.TopCandidates[0].Candidates) != 2 {
		t.Errorf("top candidates = %#v", got.TopCandidates)
	}
}

func TestConvertLogprobs_Absent(t *testing.T) {
	if got := convertLogprobs(openai.ChatCompletionChoiceLogprobs{}); got != nil {
		t.Errorf("logprobs = %#v, want nil when the provider sent none", got)
	}
}
