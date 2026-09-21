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
	"math"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// chatWire marshals the params the way the SDK sends them, so assertions read
// the bytes rather than the Go structs. omitzero and union arms are where this
// package's defects live.
func chatWire(t *testing.T, req *model.LLMRequest) map[string]any {
	t.Helper()
	params, err := buildChatParams("gpt-4o-mini", req)
	if err != nil {
		t.Fatalf("buildChatParams() err = %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return out
}

func chatMessages(t *testing.T, wire map[string]any) []map[string]any {
	t.Helper()
	raw, ok := wire["messages"].([]any)
	if !ok {
		t.Fatalf("messages missing or not an array: %#v", wire["messages"])
	}
	msgs := make([]map[string]any, 0, len(raw))
	for _, m := range raw {
		msg, ok := m.(map[string]any)
		if !ok {
			t.Fatalf("message is not an object: %#v", m)
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

func userReq(parts ...*genai.Part) *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: parts}}}
}

func TestBuildChatParams_Roles(t *testing.T) {
	tests := []struct {
		name     string
		role     string
		wantRole string
	}{
		{name: "empty defaults to user", role: "", wantRole: "user"},
		{name: "user", role: "user", wantRole: "user"},
		{name: "model becomes assistant", role: "model", wantRole: "assistant"},
		{name: "system survives", role: "system", wantRole: "system"},
		{name: "developer survives", role: "developer", wantRole: "developer"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
				{Role: tt.role, Parts: []*genai.Part{{Text: "hi"}}},
			}})
			msgs := chatMessages(t, wire)
			if len(msgs) != 1 {
				t.Fatalf("messages = %d, want 1", len(msgs))
			}
			if got := msgs[0]["role"]; got != tt.wantRole {
				t.Errorf("role = %v, want %v", got, tt.wantRole)
			}
			if got := msgs[0]["content"]; got != "hi" {
				t.Errorf("content = %v, want %q", got, "hi")
			}
		})
	}
}

func TestBuildChatParams_UnsupportedRole(t *testing.T) {
	_, err := buildChatParams("m", &model.LLMRequest{Contents: []*genai.Content{
		{Role: "wizard", Parts: []*genai.Part{{Text: "hi"}}},
	}})
	if err == nil || !strings.Contains(err.Error(), "wizard") {
		t.Fatalf("err = %v, want one naming the role", err)
	}
}

func TestBuildChatParams_SystemInstructionLeadsMessages(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("be terse", genai.RoleUser),
		},
	})
	msgs := chatMessages(t, wire)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	if msgs[0]["role"] != "system" || msgs[0]["content"] != "be terse" {
		t.Errorf("first message = %#v, want the system instruction", msgs[0])
	}
	if msgs[1]["role"] != "user" {
		t.Errorf("second message role = %v, want user", msgs[1]["role"])
	}
	if _, ok := wire["instructions"]; ok {
		t.Error("instructions field sent; Chat Completions has no such field")
	}
}

func TestBuildChatParams_MultiTurnHistory(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "my locker is 8123"}}},
		{Role: "model", Parts: []*genai.Part{{Text: "noted"}}},
		{Role: "user", Parts: []*genai.Part{{Text: "which locker?"}}},
	}})
	msgs := chatMessages(t, wire)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}
	wantRoles := []string{"user", "assistant", "user"}
	for i, want := range wantRoles {
		if got := msgs[i]["role"]; got != want {
			t.Errorf("message %d role = %v, want %v", i, got, want)
		}
	}
	// The assistant turn goes back as plain content. Nothing here resembles the
	// Responses API's output_text wrapper.
	if got := msgs[1]["content"]; got != "noted" {
		t.Errorf("assistant content = %v, want %q", got, "noted")
	}
}

func TestBuildChatParams_TextPartsJoin(t *testing.T) {
	wire := chatWire(t, userReq(&genai.Part{Text: "one"}, &genai.Part{Text: "  "}, &genai.Part{Text: "two"}))
	msgs := chatMessages(t, wire)
	if got := msgs[0]["content"]; got != "one\ntwo" {
		t.Errorf("content = %q, want %q; whitespace-only parts are dropped", got, "one\ntwo")
	}
}

func TestBuildChatParams_ToolCallAndResultPairing(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "weather?"}}},
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call_1", Name: "get_weather", Args: map[string]any{"city": "Lisbon"},
		}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "call_1", Name: "get_weather", Response: map[string]any{"c": -7},
		}}}},
	}})
	msgs := chatMessages(t, wire)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}

	calls, ok := msgs[1]["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("assistant tool_calls = %#v, want one", msgs[1]["tool_calls"])
	}
	call := calls[0].(map[string]any)
	if call["id"] != "call_1" || call["type"] != "function" {
		t.Errorf("tool call = %#v, want id call_1 and type function", call)
	}
	fn := call["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v, want get_weather", fn["name"])
	}
	if fn["arguments"] != `{"city":"Lisbon"}` {
		t.Errorf("arguments = %v, want the JSON-encoded args", fn["arguments"])
	}

	// The result is its own tool-role message keyed to the call, not a part of
	// the user turn that carried it.
	if msgs[2]["role"] != "tool" {
		t.Errorf("result role = %v, want tool", msgs[2]["role"])
	}
	if msgs[2]["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v, want call_1", msgs[2]["tool_call_id"])
	}
}

func TestBuildChatParams_ToolResultWithoutCallID(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "f"}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "f"}}}},
	}})
	msgs := chatMessages(t, wire)
	calls := msgs[0]["tool_calls"].([]any)
	minted := calls[0].(map[string]any)["id"]
	if minted == "" {
		t.Fatal("no call id minted")
	}
	// A response with no ID pairs with the oldest outstanding call, so the two
	// still line up on the wire.
	if got := msgs[1]["tool_call_id"]; got != minted {
		t.Errorf("tool_call_id = %v, want the minted %v", got, minted)
	}
}

func TestBuildChatParams_UnknownToolResultIDRejected(t *testing.T) {
	_, err := buildChatParams("m", &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "nope", Name: "f"}}}},
	}})
	if err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("err = %v, want one naming the unknown call id", err)
	}
}

// TestBuildChatParams_ThoughtsNotReplayed pins the deliberate divergence from
// the Responses path, which replays prior-turn reasoning as assistant text.
func TestBuildChatParams_ThoughtsNotReplayed(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{
			{Text: "the user wants a joke", Thought: true},
			{Text: "here it is"},
		}},
	}})
	msgs := chatMessages(t, wire)
	if len(msgs) != 1 {
		t.Fatalf("messages = %d, want 1", len(msgs))
	}
	if got := msgs[0]["content"]; got != "here it is" {
		t.Errorf("content = %q, want only the answer", got)
	}
}

func TestBuildChatParams_ThoughtOnlyTurnProducesNoMessage(t *testing.T) {
	_, err := buildChatParams("m", &model.LLMRequest{Contents: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{{Text: "thinking", Thought: true}}},
	}})
	if !errors.Is(err, ErrNoContents) {
		t.Fatalf("err = %v, want %v", err, ErrNoContents)
	}
}

func TestBuildChatParams_NoContents(t *testing.T) {
	_, err := buildChatParams("m", &model.LLMRequest{})
	if !errors.Is(err, ErrNoContents) {
		t.Fatalf("err = %v, want %v", err, ErrNoContents)
	}
}

func TestBuildChatParams_NilRequest(t *testing.T) {
	_, err := buildChatParams("m", nil)
	if !errors.Is(err, ErrRequestNil) {
		t.Fatalf("err = %v, want %v", err, ErrRequestNil)
	}
}

func TestBuildChatParams_UnsupportedPart(t *testing.T) {
	_, err := buildChatParams("m", userReq(&genai.Part{InlineData: &genai.Blob{MIMEType: "image/png"}}))
	if err == nil || !strings.Contains(err.Error(), "unsupported content part") {
		t.Fatalf("err = %v, want an unsupported-part error", err)
	}
}

// TestApplyChatGenerationConfig_EndpointOnlyFields covers the three settings
// Chat Completions honours that the Responses path rejects outright.
func TestApplyChatGenerationConfig_EndpointOnlyFields(t *testing.T) {
	seed := int32(42)
	freq := float32(0.5)
	pres := float32(-0.25)
	wire := chatWire(t, &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config: &genai.GenerateContentConfig{
			StopSequences:    []string{"STOP", "END"},
			Seed:             &seed,
			FrequencyPenalty: &freq,
			PresencePenalty:  &pres,
		},
	})
	stop, ok := wire["stop"].([]any)
	if !ok || len(stop) != 2 || stop[0] != "STOP" {
		t.Errorf("stop = %#v, want the two sequences", wire["stop"])
	}
	if wire["seed"] != float64(42) {
		t.Errorf("seed = %v, want 42", wire["seed"])
	}
	if got := wire["frequency_penalty"].(float64); math.Abs(got-0.5) > 1e-6 {
		t.Errorf("frequency_penalty = %v, want 0.5", got)
	}
	if got := wire["presence_penalty"].(float64); math.Abs(got+0.25) > 1e-6 {
		t.Errorf("presence_penalty = %v, want -0.25", got)
	}
}

func TestApplyChatGenerationConfig_TranslatedFields(t *testing.T) {
	temp := float32(0.3)
	topP := float32(0.9)
	logprobs := int32(3)
	wire := chatWire(t, &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config: &genai.GenerateContentConfig{
			Temperature:      &temp,
			TopP:             &topP,
			MaxOutputTokens:  256,
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
			ServiceTier:      genai.ServiceTierFlex,
		},
	})
	// float32 config widened to float64 on the wire, so compare with tolerance.
	if got := wire["temperature"].(float64); math.Abs(got-0.3) > 1e-6 {
		t.Errorf("temperature = %v, want 0.3", got)
	}
	if got := wire["top_p"].(float64); math.Abs(got-0.9) > 1e-6 {
		t.Errorf("top_p = %v, want 0.9", got)
	}
	// The cap is max_completion_tokens here, not the Responses name.
	if wire["max_completion_tokens"] != float64(256) {
		t.Errorf("max_completion_tokens = %v, want 256", wire["max_completion_tokens"])
	}
	if _, ok := wire["max_output_tokens"]; ok {
		t.Error("max_output_tokens sent; that is the Responses field")
	}
	if wire["logprobs"] != true || wire["top_logprobs"] != float64(3) {
		t.Errorf("logprobs = %v, top_logprobs = %v", wire["logprobs"], wire["top_logprobs"])
	}
	if wire["service_tier"] != "flex" {
		t.Errorf("service_tier = %v, want flex", wire["service_tier"])
	}
}

func TestApplyChatGenerationConfig_RejectedFields(t *testing.T) {
	topK := float32(5)
	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
		want error
	}{
		{name: "topK", cfg: &genai.GenerateContentConfig{TopK: &topK}, want: ErrTopKNotSupported},
		{name: "candidate count", cfg: &genai.GenerateContentConfig{CandidateCount: 2}, want: ErrMultipleCandidatesNotSupported},
		{name: "labels", cfg: &genai.GenerateContentConfig{Labels: map[string]string{"a": "b"}}, want: ErrLabelsNotSupported},
		{name: "safety settings", cfg: &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{}}}, want: ErrSafetySettingsNotSupported},
		{name: "mime type", cfg: &genai.GenerateContentConfig{ResponseMIMEType: "text/csv"}, want: ErrUnsupportedMIMEType},
		{name: "cached content", cfg: &genai.GenerateContentConfig{CachedContent: "c"}, want: ErrUnsupportedConfigField},
		{name: "speech config", cfg: &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}, want: ErrUnsupportedConfigField},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := buildChatParams("m", &model.LLMRequest{
				Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
				Config:   tt.cfg,
			})
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestApplyChatGenerationConfig_SeedAccepted is the counterpart of the rejected
// list: Seed is refused on Responses and must not be refused here.
func TestApplyChatGenerationConfig_SeedAccepted(t *testing.T) {
	seed := int32(7)
	_, err := buildChatParams("m", &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config:   &genai.GenerateContentConfig{Seed: &seed},
	})
	if err != nil {
		t.Fatalf("buildChatParams() err = %v, want nil", err)
	}
}

func TestApplyChatGenerationConfig_ResponseFormat(t *testing.T) {
	t.Run("json object without a schema", func(t *testing.T) {
		wire := chatWire(t, &model.LLMRequest{
			Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
			Config:   &genai.GenerateContentConfig{ResponseMIMEType: "application/json"},
		})
		format, ok := wire["response_format"].(map[string]any)
		if !ok || format["type"] != "json_object" {
			t.Fatalf("response_format = %#v, want json_object", wire["response_format"])
		}
		if _, ok := wire["text"]; ok {
			t.Error("text field sent; that is the Responses slot")
		}
	})

	t.Run("json schema", func(t *testing.T) {
		wire := chatWire(t, &model.LLMRequest{
			Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
			Config: &genai.GenerateContentConfig{ResponseSchema: &genai.Schema{
				Title: "answer",
				Type:  genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"city": {Type: genai.TypeString},
				},
			}},
		})
		format := wire["response_format"].(map[string]any)
		if format["type"] != "json_schema" {
			t.Fatalf("type = %v, want json_schema", format["type"])
		}
		schema := format["json_schema"].(map[string]any)
		if schema["name"] != "answer" || schema["strict"] != true {
			t.Errorf("json_schema = %#v, want name answer and strict true", schema)
		}
		body := schema["schema"].(map[string]any)
		if body["additionalProperties"] != false {
			t.Error("strict rewriting did not run on the schema")
		}
	})
}

func TestApplyChatThinkingConfig(t *testing.T) {
	budget := func(n int32) *int32 { return &n }
	tests := []struct {
		name   string
		cfg    *genai.ThinkingConfig
		want   string
		absent bool
	}{
		{name: "level low", cfg: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, want: "low"},
		{name: "budget zero means none", cfg: &genai.ThinkingConfig{ThinkingBudget: budget(0)}, want: "none"},
		{name: "positive budget means medium", cfg: &genai.ThinkingConfig{ThinkingBudget: budget(1024)}, want: "medium"},
		{name: "dynamic budget defers to the model", cfg: &genai.ThinkingConfig{ThinkingBudget: budget(-1)}, absent: true},
		// No reasoning summary exists on this endpoint, so asking for thoughts
		// is accepted and changes nothing rather than failing the call.
		{name: "include thoughts alone is ignored", cfg: &genai.ThinkingConfig{IncludeThoughts: true}, absent: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := chatWire(t, &model.LLMRequest{
				Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
				Config:   &genai.GenerateContentConfig{ThinkingConfig: tt.cfg},
			})
			got, present := wire["reasoning_effort"]
			if tt.absent {
				if present {
					t.Fatalf("reasoning_effort = %v, want it absent", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("reasoning_effort = %v, want %v", got, tt.want)
			}
			if _, ok := wire["reasoning"]; ok {
				t.Error("reasoning object sent; this endpoint takes a scalar")
			}
		})
	}
}

func TestApplyChatThinkingConfig_RejectsNonsenseBudget(t *testing.T) {
	budget := int32(-5)
	_, err := buildChatParams("m", &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &budget}},
	})
	if !errors.Is(err, ErrUnsupportedConfigField) {
		t.Fatalf("err = %v, want %v", err, ErrUnsupportedConfigField)
	}
}

// TestBuildChatParams_InterleavedTextAndCall covers the shape a streamed turn
// actually leaves in the session. The aggregator starts a new text part
// whenever the content kind changes, so speech either side of a tool call
// arrives as two parts in one content
// (internal/llminternal/stream_aggregator.go:109, :140).
//
// A Chat Completions assistant message has one content string and one
// tool_calls array, so the interleaving has to flatten. What must not happen is
// either half of the speech going missing.
func TestBuildChatParams_InterleavedTextAndCall(t *testing.T) {
	wire := chatWire(t, &model.LLMRequest{Contents: []*genai.Content{
		{Role: "model", Parts: []*genai.Part{
			{Text: "thinking about it", Thought: true},
			{Text: "let me check"},
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "get_weather"}},
			{Text: "it is hailing at -7"},
		}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: "call_1", Name: "get_weather", Response: map[string]any{"c": -7},
		}}}},
	}})
	msgs := chatMessages(t, wire)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want the assistant turn and its tool result", len(msgs))
	}

	// Both text parts survive, joined, with the thought left out.
	got, _ := msgs[0]["content"].(string)
	if got != "let me check\nit is hailing at -7" {
		t.Errorf("content = %q, want both text parts joined and the thought dropped", got)
	}
	if calls, ok := msgs[0]["tool_calls"].([]any); !ok || len(calls) != 1 {
		t.Errorf("tool_calls = %#v, want the one call alongside the text", msgs[0]["tool_calls"])
	}
	if msgs[1]["role"] != "tool" || msgs[1]["tool_call_id"] != "call_1" {
		t.Errorf("second message = %#v, want the paired tool result", msgs[1])
	}
}
