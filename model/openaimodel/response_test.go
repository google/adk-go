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
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

func TestConvertResponse_Text(t *testing.T) {
	resp := &responses.Response{
		ID:    "resp-1",
		Model: "gpt-test",
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "hello"},
				},
			},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  5,
			OutputTokens: 2,
			TotalTokens:  7,
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	if got.Candidates == nil || got.Candidates[0].Content.Parts[0].Text != "hello" {
		t.Fatalf("unexpected candidate contents: %+v", got.Candidates)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.PromptTokenCount != 5 {
		t.Fatalf("usage metadata missing: %+v", got.UsageMetadata)
	}
}

func TestConvertResponse_Refusal(t *testing.T) {
	resp := &responses.Response{
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "refusal", Refusal: "nope"},
				},
			},
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	part := got.Candidates[0].Content.Parts[0]
	if diff := cmp.Diff("nope", part.Text); diff != "" {
		t.Fatalf("refusal mismatch (-want +got):\n%s", diff)
	}
}

func TestConvertResponse_NoOutput(t *testing.T) {
	_, err := convertResponse(&responses.Response{})
	if err == nil {
		t.Fatalf("expected error for empty output")
	}
}

func TestConvertResponse_Logprobs(t *testing.T) {
	tests := []struct {
		name       string
		logprobs   []responses.ResponseOutputTextLogprob
		wantResult *genai.LogprobsResult
	}{
		{
			name:       "empty config",
			logprobs:   nil,
			wantResult: nil,
		},
		{
			name: "fully specified",
			logprobs: []responses.ResponseOutputTextLogprob{
				{
					Token:   "hel",
					Logprob: -0.1,
					TopLogprobs: []responses.ResponseOutputTextLogprobTopLogprob{
						{Token: "hel", Logprob: -0.1},
						{Token: "hi", Logprob: -2.3},
					},
				},
				{
					Token:   "lo",
					Logprob: -0.2,
				},
			},
			wantResult: &genai.LogprobsResult{
				ChosenCandidates: []*genai.LogprobsResultCandidate{
					{Token: "hel", LogProbability: -0.1},
					{Token: "lo", LogProbability: -0.2},
				},
				TopCandidates: []*genai.LogprobsResultTopCandidates{
					{
						Candidates: []*genai.LogprobsResultCandidate{
							{Token: "hel", LogProbability: -0.1},
							{Token: "hi", LogProbability: -2.3},
						},
					},
					{
						Candidates: nil,
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := &responses.Response{
				ID:    "resp-1",
				Model: "gpt-test",
				Output: []responses.ResponseOutputItemUnion{
					{
						Type: "message",
						Content: []responses.ResponseOutputMessageContentUnion{
							{
								Type:     "output_text",
								Text:     "hello",
								Logprobs: tc.logprobs,
							},
						},
					},
				},
			}
			got, err := convertResponse(resp)
			if err != nil {
				t.Fatalf("convertResponse() err = %v", err)
			}
			if got.Candidates == nil {
				t.Fatalf("expected candidates")
			}
			cand := got.Candidates[0]
			if diff := cmp.Diff(tc.wantResult, cand.LogprobsResult); diff != "" {
				t.Errorf("LogprobsResult mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertResponse_IncompleteDetails(t *testing.T) {
	resp := &responses.Response{
		ID:    "resp-1",
		Model: "gpt-test",
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "hello"},
				},
			},
		},
		IncompleteDetails: responses.ResponseIncompleteDetails{
			Reason: "max_output_tokens",
		},
	}
	got, err := convertResponse(resp)
	if err != nil {
		t.Fatalf("convertResponse() err = %v", err)
	}
	if got.PromptFeedback != nil {
		t.Errorf("expected PromptFeedback to be nil, got: %+v", got.PromptFeedback)
	}
	if got.Candidates == nil {
		t.Fatalf("expected candidates")
	}
	if got.Candidates[0].FinishReason != genai.FinishReasonMaxTokens {
		t.Errorf("expected FinishReasonMaxTokens, got: %v", got.Candidates[0].FinishReason)
	}
}

func TestConvertFunctionCall(t *testing.T) {
	tests := []struct {
		name     string
		call     responses.ResponseOutputItemUnion
		wantErr  bool
		wantID   string
		wantName string
		wantArg  string
	}{
		{
			name: "valid",
			call: responses.ResponseOutputItemUnion{
				CallID:    "call-1",
				Name:      "test_fn",
				Arguments: responses.ResponseOutputItemUnionArguments{OfString: `{"arg":"val"}`},
			},
			wantID:   "call-1",
			wantName: "test_fn",
			wantArg:  "val",
		},
		{
			name: "bad json",
			call: responses.ResponseOutputItemUnion{
				CallID:    "call-1",
				Name:      "test_fn",
				Arguments: responses.ResponseOutputItemUnionArguments{OfString: `{bad`},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertFunctionCall(tc.call)
			if (err != nil) != tc.wantErr {
				t.Fatalf("convertFunctionCall() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if got == nil {
				t.Fatalf("expected result, got nil")
			}
			if got.FunctionCall.ID != tc.wantID || got.FunctionCall.Name != tc.wantName {
				t.Fatalf("unexpected fields: %+v", got)
			}
			if got.FunctionCall.Args["arg"] != tc.wantArg {
				t.Fatalf("unexpected args: %+v", got.FunctionCall.Args)
			}
		})
	}
}

// TestConvertFunctionCall_DecodedArguments exercises the arguments field
// through the SDK's real UnmarshalJSON path. Constructing
// ResponseOutputItemUnionArguments as a struct literal bypasses that decoding
// entirely, so it cannot tell which union arm a given wire payload populates.
func TestConvertFunctionCall_DecodedArguments(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		wantArgs map[string]any
		wantErr  bool
	}{
		{
			name:     "string arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":"{\"location\":\"Paris\"}"}`,
			wantArgs: map[string]any{"location": "Paris"},
		},
		{
			// OpenAI-compatible endpoints commonly send a bare JSON object
			// here rather than a string. These arguments must not be dropped.
			name:     "object arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":{"location":"Paris"}}`,
			wantArgs: map[string]any{"location": "Paris"},
		},
		{
			name:     "empty object arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":{}}`,
			wantArgs: map[string]any{},
		},
		{
			name:     "empty string arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":""}`,
			wantArgs: map[string]any{},
		},
		{
			name:     "null arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":null}`,
			wantArgs: map[string]any{},
		},
		{
			name:     "absent arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c"}`,
			wantArgs: map[string]any{},
		},
		{
			// A JSON string holding the literal null means "no arguments",
			// matching a null in the bare arm. It must not yield a nil map:
			// a tool wrapper writing to one would panic.
			name:     "null string arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":"null"}`,
			wantArgs: map[string]any{},
		},
		{
			// Both arms must follow encoding/json's last-one-wins rule, as
			// every other JSON consumer in the package does. The SDK decodes
			// with gjson, which keeps the first occurrence, so decoding the
			// bare arm from anything but the original bytes would resolve
			// this to /tmp/safe and make the two arms disagree.
			name:     "duplicate keys in object arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":{"path":"/tmp/safe","path":"/etc/shadow"}}`,
			wantArgs: map[string]any{"path": "/etc/shadow"},
		},
		{
			name:     "duplicate keys in string arguments",
			raw:      `{"type":"function_call","name":"f","call_id":"c","arguments":"{\"path\":\"/tmp/safe\",\"path\":\"/etc/shadow\"}"}`,
			wantArgs: map[string]any{"path": "/etc/shadow"},
		},
		{
			// A non-object payload must surface an error rather than degrade
			// into an empty argument map.
			name:    "array arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":[1,2]}`,
			wantErr: true,
		},
		{
			name:    "number arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":42}`,
			wantErr: true,
		},
		{
			name:    "bool arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":true}`,
			wantErr: true,
		},
		{
			name:    "non-object string arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":"[1,2]"}`,
			wantErr: true,
		},
		{
			// Out of range for float64. The SDK's decoder turns this into
			// +Inf, so it must be rejected from the original bytes rather
			// than reported as a re-encoding failure.
			name:    "out of range number in object arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":{"x":1e400}}`,
			wantErr: true,
		},
		{
			name:    "malformed string arguments",
			raw:     `{"type":"function_call","name":"f","call_id":"c","arguments":"{not json"}`,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var item responses.ResponseOutputItemUnion
			if err := json.Unmarshal([]byte(tc.raw), &item); err != nil {
				t.Fatalf("json.Unmarshal(%s) err = %v", tc.raw, err)
			}
			got, err := convertFunctionCall(item)
			if (err != nil) != tc.wantErr {
				t.Fatalf("convertFunctionCall() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				if !errors.Is(err, ErrFunctionCallArgs) {
					t.Errorf("error = %v, want errors.Is(err, ErrFunctionCallArgs)", err)
				}
				return
			}
			if got.FunctionCall.Args == nil {
				t.Errorf("Args = nil, want non-nil (a nil map panics on write)")
			}
			if diff := cmp.Diff(tc.wantArgs, got.FunctionCall.Args); diff != "" {
				t.Errorf("Args mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestConvertFunctionCall_HandBuiltArguments covers values constructed as struct
// literals rather than decoded from the wire, as callers do in their own tests.
// These carry no JSON metadata and no original bytes, so neither arm can be
// discriminated by presence alone.
func TestConvertFunctionCall_HandBuiltArguments(t *testing.T) {
	tests := []struct {
		name     string
		args     responses.ResponseOutputItemUnionArguments
		wantArgs map[string]any
	}{
		{
			name:     "string arm",
			args:     responses.ResponseOutputItemUnionArguments{OfString: `{"location":"Paris"}`},
			wantArgs: map[string]any{"location": "Paris"},
		},
		{
			name:     "bare arm",
			args:     responses.ResponseOutputItemUnionArguments{OfResponseToolSearchCallArguments: map[string]any{"location": "Paris"}},
			wantArgs: map[string]any{"location": "Paris"},
		},
		{
			name:     "zero value",
			args:     responses.ResponseOutputItemUnionArguments{},
			wantArgs: map[string]any{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := convertFunctionCall(responses.ResponseOutputItemUnion{Arguments: tc.args})
			if err != nil {
				t.Fatalf("convertFunctionCall() err = %v", err)
			}
			if diff := cmp.Diff(tc.wantArgs, got.FunctionCall.Args); diff != "" {
				t.Errorf("Args mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestConvertResponse_BadFunctionCallArgs covers what the convertFunctionCall
// tests structurally cannot, by going through the whole conversion:
// convertOutputItems stops at the first error, so a single undecodable
// arguments payload discards the entire response — the model's text is lost
// with it, rather than the response degrading to its remaining parts. The
// sentinel must also survive that wrapping chain, and the error must name the
// offending call, since nothing else survives to identify it.
func TestConvertResponse_BadFunctionCallArgs(t *testing.T) {
	const raw = `{"id":"r","model":"m","output":[
		{"type":"message","content":[{"type":"output_text","text":"hello"}]},
		{"type":"function_call","name":"write_file","call_id":"call-1","arguments":[1,2]}]}`

	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("json.Unmarshal() err = %v", err)
	}
	got, err := convertResponse(&resp)
	if err == nil {
		t.Fatalf("convertResponse() = %+v, want error", got)
	}
	// The preceding text part does not survive.
	if got != nil {
		t.Errorf("convertResponse() = %+v, want nil alongside the error", got)
	}
	if !errors.Is(err, ErrFunctionCallArgs) {
		t.Errorf("error = %v, want errors.Is(err, ErrFunctionCallArgs)", err)
	}
	for _, want := range []string{`write_file`, `call-1`} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want it to mention %q", err, want)
		}
	}
}

func TestFinishReason(t *testing.T) {
	tests := []struct {
		reason string
		want   genai.FinishReason
	}{
		{"stop", genai.FinishReasonOther},
		{"max_output_tokens", genai.FinishReasonMaxTokens},
		{"content_filter", genai.FinishReasonSafety},
		{"", genai.FinishReasonStop},
		{"other", genai.FinishReasonOther},
	}
	for _, tc := range tests {
		t.Run(tc.reason, func(t *testing.T) {
			resp := &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: tc.reason,
				},
			}
			got := finishReason(resp)
			if got != tc.want {
				t.Errorf("finishReason(%q) = %v, want %v", tc.reason, got, tc.want)
			}
		})
	}

	t.Run("nil response", func(t *testing.T) {
		if got := finishReason(nil); got != genai.FinishReasonUnspecified {
			t.Errorf("finishReason(nil) = %v, want Unspecified", got)
		}
	})
}

func TestConvertOutputItems(t *testing.T) {
	tests := []struct {
		name    string
		items   []responses.ResponseOutputItemUnion
		want    []*genai.Part
		wantErr error
	}{
		{
			name: "valid items",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "output_text", Text: "text1"},
						{Type: "refusal", Refusal: "nope"},
					},
				},
				{
					Type:      "function_call",
					CallID:    "call-1",
					Name:      "fn",
					Arguments: responses.ResponseOutputItemUnionArguments{OfString: `{}`},
				},
				{
					Type: "reasoning",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Text: "thought1"},
					},
					Summary: []responses.ResponseReasoningItemSummary{
						{Text: "summary1"},
					},
				},
			},
			want: []*genai.Part{
				{Text: "text1"},
				{Text: "nope"},
				{
					FunctionCall: &genai.FunctionCall{
						Name: "fn",
						ID:   "call-1",
						Args: map[string]any{},
					},
				},
				{Text: "thought1", Thought: true},
				{Text: "summary1", Thought: true},
			},
		},
		{
			name:    "empty items",
			items:   nil,
			wantErr: ErrNoOutputItems,
		},
		{
			name: "invalid type",
			items: []responses.ResponseOutputItemUnion{
				{Type: "invalid"},
			},
			wantErr: ErrUnsupportedOutputItemType,
		},
		{
			name: "invalid message content type",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "invalid"},
					},
				},
			},
			wantErr: ErrUnsupportedMessageContentType,
		},
		{
			name: "empty message content",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{Type: "output_text", Text: ""}, // Empty text is skipped
					},
				},
			},
			wantErr: ErrNoTextOrToolContent,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parts, err := convertOutputItems(tc.items)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("convertOutputItems() error = %v, wantErr %v", err, tc.wantErr)
			}
			if len(parts) != len(tc.want) {
				t.Fatalf("expected %d parts, got %d", len(tc.want), len(parts))
			}
			if diff := cmp.Diff(tc.want, parts); diff != "" {
				t.Errorf("convertOutputItems() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPromptFeedback(t *testing.T) {
	tests := []struct {
		name string
		resp *responses.Response
		want *genai.GenerateContentResponsePromptFeedback
	}{
		{
			name: "content filter",
			resp: &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: "content_filter",
				},
			},
			want: &genai.GenerateContentResponsePromptFeedback{
				BlockReason:        genai.BlockedReasonSafety,
				BlockReasonMessage: "content_filter",
			},
		},
		{
			name: "max_output_tokens",
			resp: &responses.Response{
				IncompleteDetails: responses.ResponseIncompleteDetails{
					Reason: "max_output_tokens",
				},
			},
			want: nil,
		},
		{
			name: "nil response",
			resp: nil,
			want: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := promptFeedback(tc.resp)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("promptFeedback() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertLogprobs(t *testing.T) {
	tests := []struct {
		name  string
		items []responses.ResponseOutputItemUnion
		want  *genai.LogprobsResult
	}{
		{
			name:  "empty items",
			items: nil,
			want:  nil,
		},
		{
			name: "fully specified",
			items: []responses.ResponseOutputItemUnion{
				{
					Type: "message",
					Content: []responses.ResponseOutputMessageContentUnion{
						{
							Type: "output_text",
							Logprobs: []responses.ResponseOutputTextLogprob{
								{
									Token:   "hel",
									Logprob: -0.1,
									TopLogprobs: []responses.ResponseOutputTextLogprobTopLogprob{
										{Token: "hel", Logprob: -0.1},
										{Token: "hi", Logprob: -2.3},
									},
								},
								{
									Token:   "lo",
									Logprob: -0.2,
								},
							},
						},
					},
				},
			},
			want: &genai.LogprobsResult{
				ChosenCandidates: []*genai.LogprobsResultCandidate{
					{Token: "hel", LogProbability: -0.1},
					{Token: "lo", LogProbability: -0.2},
				},
				TopCandidates: []*genai.LogprobsResultTopCandidates{
					{
						Candidates: []*genai.LogprobsResultCandidate{
							{Token: "hel", LogProbability: -0.1},
							{Token: "hi", LogProbability: -2.3},
						},
					},
					{
						Candidates: nil,
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := convertLogprobs(tc.items)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("convertLogprobs() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
