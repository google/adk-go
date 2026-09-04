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
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func TestBuildOpenAIParams_Text(t *testing.T) {
	req := &model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			genai.NewContentFromText("ping", genai.RoleUser),
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	if got, want := string(params.Model), "gpt-4o-mini"; got != want {
		t.Fatalf("Model mismatch got=%q want=%q", got, want)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("unexpected input items: %+v", items)
	}
	textParts := items[0].OfMessage.Content.OfInputItemContentList
	if len(textParts) != 1 {
		t.Fatalf("unexpected message parts: %+v", textParts)
	}
	if got, want := textParts[0].OfInputText.Text, "ping"; got != want {
		t.Fatalf("text mismatch got=%q want=%q", got, want)
	}
}

// TestBuildOpenAIParams_MultiTurnAssistantUsesOutputText guards that a replayed
// assistant turn is serialized as an output message with content type
// "output_text". Sending "input_text" for the assistant role makes the OpenAI
// Responses API reject every multi-turn request with HTTP 400 from the second
// message onward.
func TestBuildOpenAIParams_MultiTurnAssistantUsesOutputText(t *testing.T) {
	req := &model.LLMRequest{
		Model: "gpt-4o-mini",
		Contents: []*genai.Content{
			genai.NewContentFromText("hi", genai.RoleUser),
			genai.NewContentFromText("hello there", genai.RoleModel),
			genai.NewContentFromText("can you code", genai.RoleUser),
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}

	items := params.Input.OfInputItemList
	if len(items) != 3 {
		t.Fatalf("got %d input items, want 3: %+v", len(items), items)
	}

	// User turns remain easy input messages using input_text.
	if items[0].OfMessage == nil || items[2].OfMessage == nil {
		t.Fatalf("user turns should be easy input messages: %+v", items)
	}
	if got := items[0].OfMessage.Content.OfInputItemContentList[0].OfInputText.Type; got != constant.InputText("input_text") {
		t.Errorf("user content type = %q, want input_text", got)
	}

	// The assistant turn must be an output message using output_text.
	out := items[1].OfOutputMessage
	if out == nil {
		t.Fatalf("assistant turn should be an output message, got %+v", items[1])
	}
	if len(out.Content) != 1 || out.Content[0].OfOutputText == nil {
		t.Fatalf("assistant output message content malformed: %+v", out.Content)
	}
	if got, want := out.Content[0].OfOutputText.Text, "hello there"; got != want {
		t.Errorf("assistant text = %q, want %q", got, want)
	}
	// Verify the wire format OpenAI actually receives. Asserting the marshalled
	// JSON rather than the structs is what makes these checks meaningful: Type
	// elides its zero value to "output_text", ID is dropped when empty, and
	// Status is `omitzero`, so none of the three is observable on the struct.
	raw, err := json.Marshal(items[1])
	if err != nil {
		t.Fatalf("marshal assistant item: %v", err)
	}
	if !strings.Contains(string(raw), `"output_text"`) {
		t.Errorf("assistant item JSON missing output_text: %s", raw)
	}
	if strings.Contains(string(raw), `"input_text"`) {
		t.Errorf("assistant item JSON must not contain input_text: %s", raw)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal assistant item: %v", err)
	}
	// OpenAI mints message IDs on output; a replayed turn has none to echo back
	// and must omit the field rather than invent one, or the request is rejected.
	if v, ok := wire["id"]; ok {
		t.Errorf("assistant item must not carry an id, got %v: %s", v, raw)
	}
	if got := wire["status"]; got != "completed" {
		t.Errorf("assistant item status = %v, want completed: %s", got, raw)
	}
	if got := wire["role"]; got != "assistant" {
		t.Errorf("assistant item role = %v, want assistant: %s", got, raw)
	}
}

// TestBuildOpenAIParams_ItemOrdering pins the order in which a model turn's
// parts become input items. Text is buffered and flushed by convertContents
// immediately before a function call or response is appended; dropping that
// flush does not lose the text but does emit it after the call, silently
// reordering the history. Assistant text became a third item kind with the
// output_text fix, so the ordering needs a guard.
func TestBuildOpenAIParams_ItemOrdering(t *testing.T) {
	call := &genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "call_1"}}
	tests := []struct {
		name  string
		parts []*genai.Part
		want  []string
	}{
		{
			name:  "text before call is flushed first",
			parts: []*genai.Part{{Text: "Let me check."}, call},
			want:  []string{"output_message", "function_call"},
		},
		{
			name:  "text after call is flushed last",
			parts: []*genai.Part{call, {Text: "Checking now."}},
			want:  []string{"function_call", "output_message"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{Contents: []*genai.Content{
				{Role: string(genai.RoleModel), Parts: tc.parts},
			}}
			params, err := buildOpenAIParams("fallback", req)
			if err != nil {
				t.Fatalf("buildOpenAIParams() err = %v", err)
			}
			var got []string
			for _, item := range params.Input.OfInputItemList {
				switch {
				case item.OfOutputMessage != nil:
					got = append(got, "output_message")
				case item.OfMessage != nil:
					got = append(got, "message")
				case item.OfFunctionCall != nil:
					got = append(got, "function_call")
				case item.OfFunctionCallOutput != nil:
					got = append(got, "function_call_output")
				default:
					got = append(got, "unknown")
				}
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("item order = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildOpenAIParams_FunctionCall(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleModel),
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "lookup", Args: map[string]any{"city": "Paris"}}},
					{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"temp": 72}}},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	var call *responses.ResponseFunctionToolCallParam
	var response *responses.ResponseInputItemFunctionCallOutputParam
	for _, item := range params.Input.OfInputItemList {
		switch {
		case item.OfFunctionCall != nil:
			call = item.OfFunctionCall
		case item.OfFunctionCallOutput != nil:
			response = item.OfFunctionCallOutput
		}
	}
	if call == nil || response == nil {
		t.Fatalf("missing function call/response in %+v", params.Input.OfInputItemList)
		return
	}
	if call.CallID == "" || !response.CallID.Valid() {
		t.Fatalf("call IDs must be populated: call=%+v response=%+v", call, response)
		return
	}
	if call.CallID != response.CallID.Value {
		t.Fatalf("call IDs mismatch: %q vs %q", call.CallID, response.CallID.Value)
	}
}

func TestBuildOpenAIParams_JSONSchema(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("respond JSON", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"answer": {Type: genai.TypeString},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	if params.Text.Format.OfJSONSchema == nil {
		t.Fatalf("expected json schema format, got: %+v", params.Text.Format)
	}
	if got := params.Text.Format.OfJSONSchema.Schema["type"]; got != "object" {
		t.Fatalf("schema mismatch got=%v", got)
	}
}

func TestBuildOpenAIParams_UnsupportedPart(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{Data: []byte{0x1}}},
				},
			},
		},
	}
	if _, err := buildOpenAIParams("fallback", req); err == nil {
		t.Fatalf("expected error for inline data part")
	}
}

func TestCallTrackerNewFunctionResponse_UnknownCallID(t *testing.T) {
	tracker := callTracker{pending: []string{"call-1"}}
	fr := &genai.FunctionResponse{
		Name:     "lookup",
		ID:       "call-missing",
		Response: map[string]any{"ok": true},
	}
	if _, err := tracker.newFunctionResponse(fr); err == nil || !strings.Contains(err.Error(), "unknown or already completed") {
		t.Fatalf("expected error for unknown call id, got %v", err)
	}
	if len(tracker.pending) != 1 || tracker.pending[0] != "call-1" {
		t.Fatalf("pending calls should remain untouched, got %+v", tracker.pending)
	}
}

func TestApplyGenerationConfig(t *testing.T) {
	topK := float32(5)
	p := float32(0.5)
	temp := float32(0.8)
	topP := float32(0.9)
	logprobs := int32(2)

	tests := []struct {
		name       string
		cfg        *genai.GenerateContentConfig
		wantErr    error
		wantParams *responses.ResponseNewParams
	}{
		{
			name: "nil config",
			cfg:  nil,
		},
		{
			name:    "TopK not supported",
			cfg:     &genai.GenerateContentConfig{TopK: &topK},
			wantErr: ErrTopKNotSupported,
		},
		{
			name:    "StopSequences not supported",
			cfg:     &genai.GenerateContentConfig{StopSequences: []string{"stop"}},
			wantErr: ErrStopSequencesNotSupported,
		},
		{
			name:    "Multiple candidates not supported",
			cfg:     &genai.GenerateContentConfig{CandidateCount: 2},
			wantErr: ErrMultipleCandidatesNotSupported,
		},
		{
			name:    "Penalties not supported",
			cfg:     &genai.GenerateContentConfig{FrequencyPenalty: &p},
			wantErr: ErrPenaltiesNotSupported,
		},
		{
			name:    "Labels not supported",
			cfg:     &genai.GenerateContentConfig{Labels: map[string]string{"a": "b"}},
			wantErr: ErrLabelsNotSupported,
		},
		{
			name:    "Safety settings not supported",
			cfg:     &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{}}},
			wantErr: ErrSafetySettingsNotSupported,
		},
		{
			name:    "Unsupported MIME type",
			cfg:     &genai.GenerateContentConfig{ResponseMIMEType: "image/png"},
			wantErr: ErrUnsupportedMIMEType,
		},
		{
			name: "success fully configured",
			cfg: &genai.GenerateContentConfig{
				Temperature:       &temp,
				TopP:              &topP,
				MaxOutputTokens:   100,
				ResponseLogprobs:  true,
				Logprobs:          &logprobs,
				SystemInstruction: genai.NewContentFromText("sys", "system"),
				ResponseMIMEType:  "application/json",
				ResponseSchema:    &genai.Schema{Type: genai.TypeObject},
			},
			wantParams: &responses.ResponseNewParams{
				Temperature:     param.NewOpt(float64(float32(temp))),
				TopP:            param.NewOpt(float64(float32(topP))),
				MaxOutputTokens: param.NewOpt(int64(100)),
				TopLogprobs:     param.NewOpt(int64(int32(logprobs))),
				Include:         []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs},
				Instructions:    param.NewOpt("sys"),
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
							Name:   "adk_response",
							Strict: param.NewOpt(true),
							Type:   constant.JSONSchema("json_schema"),
							Schema: map[string]any{
								"type": "object",
							},
						},
					},
				},
			},
		},
		{
			name: "success application/json without schema falls back to json_object",
			cfg: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
			wantParams: &responses.ResponseNewParams{
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONObject: &shared.ResponseFormatJSONObjectParam{
							Type: constant.JSONObject("json_object"),
						},
					},
				},
			},
		},
		{
			name: "success logprobs only",
			cfg: &genai.GenerateContentConfig{
				ResponseLogprobs: true,
			},
			wantParams: &responses.ResponseNewParams{
				TopLogprobs: param.NewOpt(int64(1)),
				Include:     []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			err := applyGenerationConfig(params, tc.cfg)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("applyGenerationConfig() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantParams != nil && !reflect.DeepEqual(params, tc.wantParams) {
				t.Errorf("applyGenerationConfig() params = %+v, want %+v", params, tc.wantParams)
			}
		})
	}
}

// Each case sets exactly one previously-dropped field and expects the request
// to fail naming it.
func TestApplyGenerationConfigRejectsUnsupportedFields(t *testing.T) {
	tests := []struct {
		field string
		cfg   *genai.GenerateContentConfig
	}{
		{"Seed", &genai.GenerateContentConfig{Seed: genai.Ptr(int32(7))}},
		{"RoutingConfig", &genai.GenerateContentConfig{RoutingConfig: &genai.GenerationConfigRoutingConfig{}}},
		{"ModelSelectionConfig", &genai.GenerateContentConfig{ModelSelectionConfig: &genai.ModelSelectionConfig{}}},
		{"CachedContent", &genai.GenerateContentConfig{CachedContent: "cached"}},
		{"ResponseModalities", &genai.GenerateContentConfig{ResponseModalities: []string{"AUDIO"}}},
		{"MediaResolution", &genai.GenerateContentConfig{MediaResolution: genai.MediaResolutionLow}},
		{"SpeechConfig", &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}},
		{"AudioTimestamp", &genai.GenerateContentConfig{AudioTimestamp: true}},
		{"ImageConfig", &genai.GenerateContentConfig{ImageConfig: &genai.ImageConfig{}}},
		{"EnableEnhancedCivicAnswers", &genai.GenerateContentConfig{EnableEnhancedCivicAnswers: genai.Ptr(true)}},
		{"ModelArmorConfig", &genai.GenerateContentConfig{ModelArmorConfig: &genai.ModelArmorConfig{}}},
		{"AudioTranscriptionConfig", &genai.GenerateContentConfig{AudioTranscriptionConfig: &genai.AudioTranscriptionConfig{}}},
	}

	for _, tc := range tests {
		t.Run(tc.field, func(t *testing.T) {
			err := applyGenerationConfig(&responses.ResponseNewParams{}, tc.cfg)
			if !errors.Is(err, ErrUnsupportedConfigField) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("applyGenerationConfig() error = %q, want it to name %q", err, tc.field)
			}
		})
	}
}

// ThinkingConfig is honored rather than rejected, mapping onto effort-based
// reasoning. Summary rides on IncludeThoughts alone: it requires a verified
// OpenAI organization, so sending it unasked would fail every reasoning call an
// unverified org makes.
func TestApplyGenerationConfigThinkingConfig(t *testing.T) {
	tests := []struct {
		name     string
		thinking *genai.ThinkingConfig
		want     shared.ReasoningParam
	}{
		{"minimal level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal}, shared.ReasoningParam{Effort: shared.ReasoningEffortMinimal}},
		{"low level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, shared.ReasoningParam{Effort: shared.ReasoningEffortLow}},
		{"medium level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium}, shared.ReasoningParam{Effort: shared.ReasoningEffortMedium}},
		{"high level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}, shared.ReasoningParam{Effort: shared.ReasoningEffortHigh}},
		// Explicitly unspecified is distinct from unset, and resolves to medium.
		{"unspecified level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelUnspecified}, shared.ReasoningParam{Effort: shared.ReasoningEffortMedium}},
		// Responses has no token budget, so only zero/non-zero survives. Zero is
		// none rather than minimal: minimal is the least thinking, not none of it.
		{"zero budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(0))}, shared.ReasoningParam{Effort: shared.ReasoningEffortNone}},
		{"positive budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(2048))}, shared.ReasoningParam{Effort: shared.ReasoningEffortMedium}},
		// -1 is genai's "you decide", and the way to say that to Responses is to
		// send no effort at all rather than to pick one on the caller's behalf.
		{"dynamic budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(dynamicThinkingBudget))}, shared.ReasoningParam{}},
		{"dynamic budget with thoughts", &genai.ThinkingConfig{
			ThinkingBudget:  genai.Ptr(int32(dynamicThinkingBudget)),
			IncludeThoughts: true,
		}, shared.ReasoningParam{Summary: shared.ReasoningSummaryAuto}},
		// A level wins over a budget, so an explicit MINIMAL still means minimal
		// even alongside the zero budget that would otherwise mean none.
		{"level wins over budget", &genai.ThinkingConfig{
			ThinkingLevel:  genai.ThinkingLevelMinimal,
			ThinkingBudget: genai.Ptr(int32(0)),
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortMinimal}},
		// IncludeThoughts is what asks for summaries, and only it.
		{"thoughts with level", &genai.ThinkingConfig{
			ThinkingLevel:   genai.ThinkingLevelHigh,
			IncludeThoughts: true,
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortHigh, Summary: shared.ReasoningSummaryAuto}},
		{"thoughts with budget", &genai.ThinkingConfig{
			ThinkingBudget:  genai.Ptr(int32(2048)),
			IncludeThoughts: true,
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortMedium, Summary: shared.ReasoningSummaryAuto}},
		// Thoughts alone leave the effort to the model rather than inventing one.
		{"thoughts alone", &genai.ThinkingConfig{IncludeThoughts: true}, shared.ReasoningParam{Summary: shared.ReasoningSummaryAuto}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, &genai.GenerateContentConfig{ThinkingConfig: tc.thinking}); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
			}
			if !reflect.DeepEqual(params.Reasoning, tc.want) {
				t.Errorf("params.Reasoning = %+v, want %+v", params.Reasoning, tc.want)
			}
		})
	}
}

// A summary costs a verified organization, so no caller gets one without
// asking. This is the regression guard for the unconditional summary that made
// every translated ThinkingConfig a 400.
func TestApplyGenerationConfigOmitsReasoningSummaryUnlessAsked(t *testing.T) {
	tests := []struct {
		name     string
		thinking *genai.ThinkingConfig
	}{
		{"high level", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}},
		{"zero budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(0))}},
		{"positive budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(4096))}},
		{"dynamic budget", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(dynamicThinkingBudget))}},
		{"level with thoughts off", &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow, IncludeThoughts: false}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, &genai.GenerateContentConfig{ThinkingConfig: tc.thinking}); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
			}
			if params.Reasoning.Summary != "" {
				t.Errorf("applyGenerationConfig() set Summary = %q, want it unset", params.Reasoning.Summary)
			}
		})
	}
}

// A budget below the dynamic sentinel means nothing in genai and cannot be
// translated, so it is named rather than rounded into some effort.
func TestApplyGenerationConfigRejectsNegativeThinkingBudget(t *testing.T) {
	tests := []struct {
		name     string
		thinking *genai.ThinkingConfig
	}{
		{"budget alone", &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(-2))}},
		// A level wins over a budget, and must not carry a nonsense one past
		// unmentioned on its way through.
		{"budget behind a winning level", &genai.ThinkingConfig{
			ThinkingLevel:  genai.ThinkingLevelLow,
			ThinkingBudget: genai.Ptr(int32(-2)),
		}},
		{"budget behind thoughts", &genai.ThinkingConfig{
			ThinkingBudget:  genai.Ptr(int32(-100)),
			IncludeThoughts: true,
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
				ThinkingConfig: tc.thinking,
			})
			if !errors.Is(err, ErrUnsupportedConfigField) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
			}
			if !strings.Contains(err.Error(), "ThinkingBudget") {
				t.Errorf("applyGenerationConfig() error = %q, want it to name ThinkingBudget", err)
			}
		})
	}
}

// THINKING_LEVEL_UNSPECIFIED is the caller declining to name a level, so it
// stands in for medium only when no budget says something more specific. Alone
// it still means "think", which is why it is not simply treated as unset.
func TestApplyGenerationConfigUnspecifiedLevelYieldsToABudget(t *testing.T) {
	tests := []struct {
		name     string
		thinking *genai.ThinkingConfig
		want     shared.ReasoningParam
	}{
		{"unspecified alone", &genai.ThinkingConfig{
			ThinkingLevel: genai.ThinkingLevelUnspecified,
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortMedium}},
		{"unspecified yields to zero budget", &genai.ThinkingConfig{
			ThinkingLevel:  genai.ThinkingLevelUnspecified,
			ThinkingBudget: genai.Ptr(int32(0)),
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortNone}},
		{"unspecified yields to dynamic budget", &genai.ThinkingConfig{
			ThinkingLevel:  genai.ThinkingLevelUnspecified,
			ThinkingBudget: genai.Ptr(int32(dynamicThinkingBudget)),
		}, shared.ReasoningParam{}},
		// A named level is a choice, so it keeps winning.
		{"named level still wins over zero budget", &genai.ThinkingConfig{
			ThinkingLevel:  genai.ThinkingLevelHigh,
			ThinkingBudget: genai.Ptr(int32(0)),
		}, shared.ReasoningParam{Effort: shared.ReasoningEffortHigh}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, &genai.GenerateContentConfig{ThinkingConfig: tc.thinking}); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
			}
			if !reflect.DeepEqual(params.Reasoning, tc.want) {
				t.Errorf("params.Reasoning = %+v, want %+v", params.Reasoning, tc.want)
			}
		})
	}
}

// A thinking level genai grows later must be named in an error, not lowercased
// into an effort string the API rejects with a message pointing nowhere useful.
func TestApplyGenerationConfigRejectsUnknownThinkingLevel(t *testing.T) {
	err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevel("EXHAUSTIVE")},
	})
	if !errors.Is(err, ErrUnsupportedConfigField) {
		t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
	}
	if !strings.Contains(err.Error(), "EXHAUSTIVE") {
		t.Errorf("applyGenerationConfig() error = %q, want it to name the level", err)
	}
}

// Every level genai declares today maps to an effort, so a caller setting a
// valid level gets reasoning rather than an error.
//
// Go cannot enumerate the constants of a string enum, so this list is written
// out and cannot notice a level genai adds later. It does not need to: an
// unmapped level is rejected by name at runtime, which is a diagnosable failure
// rather than the silent drop this package exists to avoid. The length check
// below only catches the opposite drift — an entry added to the map that is not
// a level, or one this list forgot.
func TestReasoningEffortsCoverEveryThinkingLevel(t *testing.T) {
	levels := []genai.ThinkingLevel{
		genai.ThinkingLevelUnspecified,
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh,
	}
	for _, level := range levels {
		if _, ok := reasoningEfforts[level]; !ok {
			t.Errorf("reasoningEfforts is missing genai.ThinkingLevel %q", level)
		}
	}
	if len(reasoningEfforts) != len(levels) {
		t.Errorf("reasoningEfforts has %d entries, want %d: it gained one this test does not list", len(reasoningEfforts), len(levels))
	}

	// And the mapping is reachable end to end, not just present in the map: a
	// map entry nothing reads would pass the loop above and still drop the level.
	for _, level := range levels {
		params := &responses.ResponseNewParams{}
		cfg := &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: level}}
		if err := applyGenerationConfig(params, cfg); err != nil {
			t.Errorf("applyGenerationConfig(ThinkingLevel %q) error = %v, want nil", level, err)
			continue
		}
		if params.Reasoning.Effort != reasoningEfforts[level] {
			t.Errorf("ThinkingLevel %q produced effort %q, want %q", level, params.Reasoning.Effort, reasoningEfforts[level])
		}
	}
}

// A ThinkingConfig with nothing set asks for nothing, so nothing is sent and
// nothing is dropped. IncludeThoughts false in particular is a request this
// package satisfies — by not including thoughts — rather than one it cannot
// honor, so erroring would fail a caller who got precisely what they asked for.
func TestApplyGenerationConfigAcceptsEmptyThinkingConfig(t *testing.T) {
	tests := []struct {
		name     string
		thinking *genai.ThinkingConfig
	}{
		{"zero value", &genai.ThinkingConfig{}},
		{"thoughts explicitly off", &genai.ThinkingConfig{IncludeThoughts: false}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, &genai.GenerateContentConfig{ThinkingConfig: tc.thinking}); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
			}
			if !reflect.DeepEqual(params.Reasoning, shared.ReasoningParam{}) {
				t.Errorf("params.Reasoning = %+v, want zero: nothing was asked for", params.Reasoning)
			}
		})
	}
}

// Reasoning stays unset when the caller asks for nothing, so non-reasoning
// models are not sent a reasoning block they would reject.
func TestApplyGenerationConfigOmitsReasoningByDefault(t *testing.T) {
	params := &responses.ResponseNewParams{}
	if err := applyGenerationConfig(params, &genai.GenerateContentConfig{}); err != nil {
		t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
	}
	if !reflect.DeepEqual(params.Reasoning, shared.ReasoningParam{}) {
		t.Errorf("params.Reasoning = %+v, want zero", params.Reasoning)
	}
}

// A field can be translated and still be handed a value that is not. These are
// the values that would otherwise slip through the presence check, since the
// field they belong to is one the package does support.
func TestApplyGenerationConfigRejectsUntranslatableValues(t *testing.T) {
	tests := []struct {
		name  string
		cfg   *genai.GenerateContentConfig
		names string
	}{
		// Logprobs only sizes the list ResponseLogprobs asks for.
		{"orphan Logprobs", &genai.GenerateContentConfig{Logprobs: genai.Ptr(int32(5))}, "Logprobs"},
		{"negative MaxOutputTokens", &genai.GenerateContentConfig{MaxOutputTokens: -1}, "MaxOutputTokens"},
		{"negative CandidateCount", &genai.GenerateContentConfig{CandidateCount: -1}, "CandidateCount"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := applyGenerationConfig(&responses.ResponseNewParams{}, tc.cfg)
			if !errors.Is(err, ErrUnsupportedConfigField) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
			}
			if !strings.Contains(err.Error(), tc.names) {
				t.Errorf("applyGenerationConfig() error = %q, want it to name %q", err, tc.names)
			}
		})
	}
}

// The values either side of the rejected ones still work, so tightening the
// check did not turn ordinary configs into errors.
func TestApplyGenerationConfigAcceptsBoundaryValues(t *testing.T) {
	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
	}{
		{"unset MaxOutputTokens", &genai.GenerateContentConfig{}},
		{"positive MaxOutputTokens", &genai.GenerateContentConfig{MaxOutputTokens: 1}},
		{"unset CandidateCount", &genai.GenerateContentConfig{}},
		// Zero and one both mean the single candidate Responses returns.
		{"single CandidateCount", &genai.GenerateContentConfig{CandidateCount: 1}},
		{"Logprobs alongside ResponseLogprobs", &genai.GenerateContentConfig{
			ResponseLogprobs: true,
			Logprobs:         genai.Ptr(int32(5)),
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := applyGenerationConfig(&responses.ResponseNewParams{}, tc.cfg); err != nil {
				t.Errorf("applyGenerationConfig() error = %v, want nil", err)
			}
		})
	}
}

// Presence, not value: an off value still means the caller expected the knob to
// be wired up. Sniffing for no-op values would need re-deciding per new field.
func TestApplyGenerationConfigRejectsExplicitOff(t *testing.T) {
	err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
		EnableEnhancedCivicAnswers: genai.Ptr(false),
	})
	if !errors.Is(err, ErrUnsupportedConfigField) {
		t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
	}
	if !strings.Contains(err.Error(), "EnableEnhancedCivicAnswers") {
		t.Errorf("applyGenerationConfig() error = %q, want it to name EnableEnhancedCivicAnswers", err)
	}
}

// Every tier genai names has a Responses equivalent, so setting one is honored
// rather than refused. OpenAI offers tiers genai cannot name; that is fine, the
// mapping only has to be total in this direction.
func TestApplyGenerationConfigServiceTier(t *testing.T) {
	tests := []struct {
		tier genai.ServiceTier
		want responses.ResponseNewParamsServiceTier
	}{
		{genai.ServiceTierFlex, responses.ResponseNewParamsServiceTierFlex},
		{genai.ServiceTierPriority, responses.ResponseNewParamsServiceTierPriority},
		// genai's "standard" is what OpenAI calls "default".
		{genai.ServiceTierStandard, responses.ResponseNewParamsServiceTierDefault},
		// Explicitly declining to choose is what auto means.
		{genai.ServiceTierUnspecified, responses.ResponseNewParamsServiceTierAuto},
	}

	for _, tc := range tests {
		t.Run(string(tc.tier), func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, &genai.GenerateContentConfig{ServiceTier: tc.tier}); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
			}
			if params.ServiceTier != tc.want {
				t.Errorf("params.ServiceTier = %q, want %q", params.ServiceTier, tc.want)
			}
		})
	}

	// A tier genai adds later is named rather than passed through as a string
	// the API would refuse for reasons the caller cannot act on.
	err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
		ServiceTier: genai.ServiceTier("platinum"),
	})
	if !errors.Is(err, ErrUnsupportedConfigField) {
		t.Fatalf("unknown tier: error = %v, want %v", err, ErrUnsupportedConfigField)
	}
	if !strings.Contains(err.Error(), "platinum") {
		t.Errorf("unknown tier: error = %q, want it to name the tier", err)
	}
}

// serviceTiers must cover every tier genai declares, or a caller setting a
// valid one gets an error instead of the tier they asked for.
func TestServiceTiersCoverEveryGenaiTier(t *testing.T) {
	tiers := []genai.ServiceTier{
		genai.ServiceTierUnspecified,
		genai.ServiceTierStandard,
		genai.ServiceTierFlex,
		genai.ServiceTierPriority,
	}
	for _, tier := range tiers {
		if _, ok := serviceTiers[tier]; !ok {
			t.Errorf("serviceTiers is missing genai.ServiceTier %q", tier)
		}
	}
	if len(serviceTiers) != len(tiers) {
		t.Errorf("serviceTiers has %d entries, want %d: it gained one this test does not list", len(serviceTiers), len(tiers))
	}
}

// A positive Timeout is the one part of HTTPOptions that crosses to Responses:
// a duration means the same thing to any HTTP client and can carry nothing
// sensitive. Headers cannot, which the tests below cover.
func TestRequestOptionsHonorsTimeout(t *testing.T) {
	timeout := 45 * time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
	if err := applyGenerationConfig(&responses.ResponseNewParams{}, cfg); err != nil {
		t.Fatalf("applyGenerationConfig() error = %v, want nil: a timeout is honored", err)
	}
	if got := len(requestOptions(cfg)); got != 1 {
		t.Errorf("requestOptions() produced %d options, want 1", got)
	}
}

// openai-go reads a zero timeout as "no deadline", so forwarding a non-positive
// one would lift the caller's bound instead of applying it.
func TestApplyGenerationConfigRejectsNonPositiveTimeout(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		timeout := d
		err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{Timeout: &timeout},
		})
		if !errors.Is(err, ErrUnsupportedConfigField) {
			t.Fatalf("timeout %v: error = %v, want %v", d, err, ErrUnsupportedConfigField)
		}
		if !strings.Contains(err.Error(), "Timeout") {
			t.Errorf("timeout %v: error = %q, want it to name Timeout", d, err)
		}
	}
}

// The reason no header crosses, pinned against the SDK rather than asserted in
// prose: openai-go records any case-insensitive Authorization as an override and
// then skips attaching the configured API key, so forwarding a caller's header
// would replace the real credential rather than accompany it. WithHeaderAdd is
// no safer than WithHeader here, despite the name.
//
// This drives openai-go directly. If a future version stops treating the header
// as an override, this test fails and Headers can be reconsidered — until then
// it documents why applyGenerationConfig refuses them.
func TestRequestOptionsCannotDisplaceTheAPIKey(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","object":"response","status":"completed","output":[]}`))
	}))
	defer srv.Close()

	client := openai.NewClient(option.WithAPIKey("real-key"), option.WithBaseURL(srv.URL))
	params := responses.ResponseNewParams{Model: shared.ResponsesModel("m")}

	// Only the header the stub saw matters here, never the decoded response.
	_, _ = client.Responses.New(context.Background(), params)
	if got != "Bearer real-key" {
		t.Fatalf("baseline Authorization = %q, want the configured key", got)
	}
	_, _ = client.Responses.New(context.Background(), params,
		option.WithHeaderAdd("Authorization", "Bearer caller"))
	if got == "Bearer real-key" {
		t.Fatal("openai-go now keeps the configured key alongside a caller Authorization header; " +
			"the reason HTTPOptions.Headers is rejected no longer holds, so revisit it")
	}
	if got != "Bearer caller" {
		t.Errorf("Authorization = %q, want the caller header to have displaced the key", got)
	}
}

// A caller header must never reach the wire, whatever it is called: the whole
// field is refused, so there is no name-based hole to slip through.
func TestApplyGenerationConfigRejectsHTTPOptionsHeaders(t *testing.T) {
	headers := []http.Header{
		{"Authorization": []string{"Bearer caller"}},
		{"authorization": []string{"Bearer lowercase"}},
		{"X-Goog-Api-Key": []string{"gemini-key"}},
		{"X-Trace-Id": []string{"harmless"}},
		{}, // present but empty is still a header map the caller allocated
	}
	for _, h := range headers {
		err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{Headers: h},
		})
		if !errors.Is(err, ErrUnsupportedConfigField) {
			t.Fatalf("headers %v: error = %v, want %v", h, err, ErrUnsupportedConfigField)
		}
		if !strings.Contains(err.Error(), "Headers") {
			t.Errorf("headers %v: error = %q, want it to name Headers", h, err)
		}
	}
}

// The rest of HTTPOptions describes the Gemini wire format, which is not the
// request being sent, so it is named rather than quietly doing nothing.
func TestApplyGenerationConfigRejectsGeminiShapedHTTPOptions(t *testing.T) {
	retry := int32(3)
	tests := []struct {
		field string
		opts  *genai.HTTPOptions
	}{
		{"BaseURL", &genai.HTTPOptions{BaseURL: "https://example.test"}},
		{"BaseURLResourceScope", &genai.HTTPOptions{BaseURLResourceScope: genai.ResourceScope("global")}},
		{"APIVersion", &genai.HTTPOptions{APIVersion: "v1beta"}},
		{"ExtraBody", &genai.HTTPOptions{ExtraBody: map[string]any{"k": "v"}}},
		{"RetryOptions", &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: &retry}}},
	}

	for _, tc := range tests {
		t.Run(tc.field, func(t *testing.T) {
			err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
				HTTPOptions: tc.opts,
			})
			if !errors.Is(err, ErrUnsupportedConfigField) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("applyGenerationConfig() error = %q, want it to name %q", err, tc.field)
			}
		})
	}
}

// An empty HTTPOptions asks for no transport change, which is satisfiable by
// doing nothing — the same rule the empty ThinkingConfig follows.
func TestRequestOptionsEmptyHTTPOptions(t *testing.T) {
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{}}
	if err := applyGenerationConfig(&responses.ResponseNewParams{}, cfg); err != nil {
		t.Fatalf("applyGenerationConfig() error = %v, want nil", err)
	}
	if opts := requestOptions(cfg); len(opts) != 0 {
		t.Errorf("requestOptions() produced %d options, want none", len(opts))
	}
	if opts := requestOptions(nil); len(opts) != 0 {
		t.Errorf("requestOptions(nil) produced %d options, want none", len(opts))
	}
}

// unsupportedHTTPOptionFields must name real fields, or an entry guards nothing.
func TestUnsupportedHTTPOptionFieldsAreAccountedFor(t *testing.T) {
	honored := map[string]bool{"Timeout": true}
	rejected := make(map[string]bool, len(unsupportedHTTPOptionFields))
	for _, field := range unsupportedHTTPOptionFields {
		rejected[field.name] = true
	}

	optType := reflect.TypeOf(genai.HTTPOptions{})
	for i := range optType.NumField() {
		field := optType.Field(i)
		if field.PkgPath != "" {
			continue // unexported, not settable by callers
		}
		if !honored[field.Name] && !rejected[field.Name] {
			t.Errorf("genai.HTTPOptions.%s is neither honored nor rejected: it would be silently ignored", field.Name)
		}
	}
	for name := range rejected {
		if _, ok := optType.FieldByName(name); !ok {
			t.Errorf("unsupportedHTTPOptionFields lists %q, which genai.HTTPOptions no longer has", name)
		}
	}
}

// A slice can tell an explicit empty from unset, so it should: asking for no
// modalities at all is still a request this package cannot honor.
func TestApplyGenerationConfigRejectsEmptyResponseModalities(t *testing.T) {
	err := applyGenerationConfig(&responses.ResponseNewParams{}, &genai.GenerateContentConfig{
		ResponseModalities: []string{},
	})
	if !errors.Is(err, ErrUnsupportedConfigField) {
		t.Fatalf("applyGenerationConfig() error = %v, want %v", err, ErrUnsupportedConfigField)
	}
	if !strings.Contains(err.Error(), "ResponseModalities") {
		t.Errorf("applyGenerationConfig() error = %q, want it to name ResponseModalities", err)
	}
}

// Every named error is checked before the sentinel, so an errors.Is call site
// that worked before ErrUnsupportedConfigField existed still works when the
// caller happens to set one of the newly-rejected fields as well.
func TestApplyGenerationConfigKeepsNamedErrorPrecedence(t *testing.T) {
	topK := float32(5)
	// Each case pairs a named error with a field that returns the sentinel; the
	// named error must win. Logprobs is here because it is rejected from within
	// the translation order rather than from the trailing sweep, which is
	// exactly where a sentinel can get in front of a named error by accident.
	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
		want error
	}{
		{"TopK over Seed", &genai.GenerateContentConfig{
			TopK: &topK,
			Seed: genai.Ptr(int32(7)),
		}, ErrTopKNotSupported},
		{"Labels over orphan Logprobs", &genai.GenerateContentConfig{
			Logprobs: genai.Ptr(int32(5)),
			Labels:   map[string]string{"team": "search"},
		}, ErrLabelsNotSupported},
		{"SafetySettings over orphan Logprobs", &genai.GenerateContentConfig{
			Logprobs:       genai.Ptr(int32(5)),
			SafetySettings: []*genai.SafetySetting{{Category: genai.HarmCategoryHarassment}},
		}, ErrSafetySettingsNotSupported},
		{"MIME type over orphan Logprobs", &genai.GenerateContentConfig{
			Logprobs:         genai.Ptr(int32(5)),
			ResponseMIMEType: "text/csv",
		}, ErrUnsupportedMIMEType},
		{"StopSequences over ThinkingConfig", &genai.GenerateContentConfig{
			StopSequences:  []string{"STOP"},
			ThinkingConfig: &genai.ThinkingConfig{},
		}, ErrStopSequencesNotSupported},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := applyGenerationConfig(&responses.ResponseNewParams{}, tc.cfg)
			if !errors.Is(err, tc.want) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, tc.want)
			}
		})
	}
}

// Guards against the bug returning as genai grows fields: every exported field
// must be translated, covered by a named error, or in unsupportedConfigFields.
// When this fails, add the new field to whichever of the three it belongs in —
// not to this test alone, which would only re-hide the drop.
func TestGenerateContentConfigFieldsAreAccountedFor(t *testing.T) {
	// Fields applyGenerationConfig, convertTools or convertToolChoice translate.
	translated := map[string]bool{
		"SystemInstruction":  true,
		"Temperature":        true,
		"TopP":               true,
		"MaxOutputTokens":    true,
		"ResponseLogprobs":   true,
		"Logprobs":           true,
		"ResponseMIMEType":   true,
		"ResponseSchema":     true,
		"ResponseJsonSchema": true,
		"ThinkingConfig":     true,
		"ServiceTier":        true,
		"HTTPOptions":        true,
		"Tools":              true,
		"ToolConfig":         true,
	}
	// Fields rejected with their own error, predating ErrUnsupportedConfigField.
	namedError := map[string]bool{
		"TopK":             true,
		"StopSequences":    true,
		"CandidateCount":   true,
		"FrequencyPenalty": true,
		"PresencePenalty":  true,
		"Labels":           true,
		"SafetySettings":   true,
	}
	rejected := make(map[string]bool, len(unsupportedConfigFields))
	for _, field := range unsupportedConfigFields {
		rejected[field.name] = true
	}

	cfgType := reflect.TypeOf(genai.GenerateContentConfig{})
	for i := range cfgType.NumField() {
		name := cfgType.Field(i).Name
		if cfgType.Field(i).PkgPath != "" {
			continue // unexported, not settable by callers
		}
		if !translated[name] && !namedError[name] && !rejected[name] {
			t.Errorf("genai.GenerateContentConfig.%s is neither translated nor rejected: it would be silently ignored", name)
		}
	}

	// Reverse drift: a field renamed upstream leaves a stale entry guarding nothing.
	for name := range rejected {
		if _, ok := cfgType.FieldByName(name); !ok {
			t.Errorf("unsupportedConfigFields lists %q, which genai.GenerateContentConfig no longer has", name)
		}
	}
}

// ThinkingConfig is the one nested type this package claims to read in full, so
// a field added to it must be translated rather than dropped like the parts of
// Tools and the schema types that doc.go still excludes from the guarantee.
func TestThinkingConfigFieldsAreAccountedFor(t *testing.T) {
	read := map[string]bool{
		"IncludeThoughts": true,
		"ThinkingBudget":  true,
		"ThinkingLevel":   true,
	}
	cfgType := reflect.TypeOf(genai.ThinkingConfig{})
	for i := range cfgType.NumField() {
		field := cfgType.Field(i)
		if field.PkgPath != "" {
			continue // unexported, not settable by callers
		}
		if !read[field.Name] {
			t.Errorf("genai.ThinkingConfig.%s is not read by applyThinkingConfig: it would be silently ignored", field.Name)
		}
	}
}

func TestFlattenContentText(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
		wantErr bool
	}{
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
		{
			name: "valid text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "part1"},
					nil,
					{Text: "part2"},
				},
			},
			want: "part1\npart2",
		},
		{
			name: "non-text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "fn"}},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			txt, err := flattenContentText(tc.content)
			if (err != nil) != tc.wantErr {
				t.Fatalf("flattenContentText() error = %v, wantErr %v", err, tc.wantErr)
			}
			if txt != tc.want {
				t.Fatalf("flattenContentText() = %q, want %q", txt, tc.want)
			}
		})
	}
}

func TestNormalizeSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  any
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "nil schema",
			schema:  nil,
			wantErr: true,
		},
		{
			name:   "map schema",
			schema: map[string]any{"type": "object"},
			want:   map[string]any{"type": "object"},
		},
		{
			name: "struct schema",
			schema: struct {
				Type string `json:"type"`
			}{Type: "array"},
			want: map[string]any{"type": "array"},
		},
		{
			name:    "invalid schema",
			schema:  func() {}, // unmarshalable
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("normalizeSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && got["type"] != tc.want["type"] {
				t.Fatalf("normalizeSchema() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNormalizeRole(t *testing.T) {
	tests := []struct {
		role    genai.Role
		want    responses.EasyInputMessageRole
		wantErr bool
	}{
		{"", responses.EasyInputMessageRoleUser, false},
		{genai.RoleUser, responses.EasyInputMessageRoleUser, false},
		{genai.RoleModel, responses.EasyInputMessageRoleAssistant, false},
		{"system", responses.EasyInputMessageRoleSystem, false},
		{"developer", responses.EasyInputMessageRoleDeveloper, false},
		{"invalid", "", true},
	}
	for _, tc := range tests {
		t.Run(string(tc.role), func(t *testing.T) {
			got, err := normalizeRole(tc.role)
			if (err != nil) != tc.wantErr {
				t.Fatalf("normalizeRole() error = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Fatalf("normalizeRole() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNewJSONSchemaFormat(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *genai.GenerateContentConfig
		want    *responses.ResponseFormatTextJSONSchemaConfigParam
		wantErr bool
	}{
		{
			name:    "no schema",
			cfg:     &genai.GenerateContentConfig{},
			wantErr: true,
		},
		{
			name: "with response schema",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{Title: "CustomTitle", Type: genai.TypeObject},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "CustomTitle",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title": "CustomTitle",
					"type":  "object",
				},
			},
		},

		{
			name: "with json schema",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{"type": "object"},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "adk_response",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"type": "object",
				},
			},
		},
		{
			name: "with nested response schema",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema: &genai.Schema{
					Title: "NestedTitle",
					Type:  genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"b_string": {Type: genai.TypeString},
						"a_object": {
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"d_int":  {Type: genai.TypeInteger},
								"c_bool": {Type: genai.TypeBoolean},
							},
						},
					},
				},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "NestedTitle",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title":                "NestedTitle",
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"a_object", "b_string"},
					"properties": map[string]any{
						"b_string": map[string]any{
							"type": "string",
						},
						"a_object": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"c_bool", "d_int"},
							"properties": map[string]any{
								"d_int": map[string]any{
									"type": "integer",
								},
								"c_bool": map[string]any{
									"type": "boolean",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "with complex json schema for strict output",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{
					"title": "NestedTitle",
					"type":  "object",
					"properties": map[string]any{
						"b_string": map[string]any{"type": "string"},
						"a_object": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"d_int":  map[string]any{"type": "integer"},
								"c_bool": map[string]any{"type": "boolean"},
							},
						},
						"c_array": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"e_float": map[string]any{"type": "number"},
								},
							},
						},
						"d_ref": map[string]any{
							"$ref":        "#/$defs/my_def",
							"description": "this should be deleted",
						},
					},
					"$defs": map[string]any{
						"my_def": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"f_string": map[string]any{"type": "string"},
							},
						},
					},
					"anyOf": []any{
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"g_string": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
			want: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Name:   "adk_response",
				Strict: param.NewOpt(true),
				Type:   constant.JSONSchema("json_schema"),
				Schema: map[string]any{
					"title":                "NestedTitle",
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"a_object", "b_string", "c_array", "d_ref"},
					"properties": map[string]any{
						"b_string": map[string]any{
							"type": "string",
						},
						"a_object": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"c_bool", "d_int"},
							"properties": map[string]any{
								"d_int": map[string]any{
									"type": "integer",
								},
								"c_bool": map[string]any{
									"type": "boolean",
								},
							},
						},
						"c_array": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type":                 "object",
								"additionalProperties": false,
								"required":             []string{"e_float"},
								"properties": map[string]any{
									"e_float": map[string]any{"type": "number"},
								},
							},
						},
						"d_ref": map[string]any{
							"$ref": "#/$defs/my_def",
						},
					},
					"$defs": map[string]any{
						"my_def": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"f_string"},
							"properties": map[string]any{
								"f_string": map[string]any{"type": "string"},
							},
						},
					},
					"anyOf": []any{
						map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"g_string"},
							"properties": map[string]any{
								"g_string": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
		},
		{
			name: "with invalid json schema",
			cfg: &genai.GenerateContentConfig{
				ResponseJsonSchema: func() {}, // unmarshalable
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newJSONSchemaFormat(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Fatalf("newJSONSchemaFormat() error = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("newJSONSchemaFormat() got = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestNewJSONSchemaFormatDoesNotMutateResponseJSONSchema(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"nested": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"value": map[string]any{"type": "string"},
				},
			},
			"reference": map[string]any{
				"$ref":        "#/$defs/item",
				"description": "caller-owned metadata",
			},
		},
		"$defs": map[string]any{
			"item": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "integer"},
				},
			},
		},
	}
	want, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("json.Marshal(schema) error = %v", err)
	}

	format, err := newJSONSchemaFormat(&genai.GenerateContentConfig{ResponseJsonSchema: schema})
	if err != nil {
		t.Fatalf("newJSONSchemaFormat() error = %v", err)
	}
	if got := format.Schema["additionalProperties"]; got != false {
		t.Fatalf("newJSONSchemaFormat() additionalProperties = %v, want false", got)
	}

	got, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("json.Marshal(schema) after conversion error = %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("newJSONSchemaFormat() mutated ResponseJsonSchema: got %s, want %s", got, want)
	}
}

func TestBuildOpenAIParamsPreservesLargeJSONSchemaIntegers(t *testing.T) {
	const minimum = int64(9007199254740993)
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("return a count", genai.RoleUser),
		},
		Config: &genai.GenerateContentConfig{
			ResponseJsonSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"count": map[string]any{
						"type":    "integer",
						"minimum": minimum,
					},
				},
			},
		},
	}

	params, err := buildOpenAIParams("gpt-4o-mini", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() error = %v", err)
	}
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal(params) error = %v", err)
	}
	if got, want := string(data), `"minimum":9007199254740993`; !strings.Contains(got, want) {
		t.Fatalf("json.Marshal(params) = %s, want exact integer constraint %s", got, want)
	}
}
