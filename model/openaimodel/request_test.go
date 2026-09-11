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
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
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
	// The leading turn is what makes this test bite: on its own the
	// unsupported part leaves the request empty, so a build that skipped it
	// silently would still fail with ErrNoContents and look like a rejection.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("q", genai.RoleUser),
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{Data: []byte{0x1}}},
				},
			},
		},
	}
	_, err := buildOpenAIParams("fallback", req)
	if err == nil {
		t.Fatalf("expected error for inline data part")
	}
	if errors.Is(err, ErrNoContents) || !strings.Contains(err.Error(), "unsupported content part") {
		t.Errorf("buildOpenAIParams() err = %v, want an unsupported-content-part error", err)
	}
}

// describeInput renders converted input items compactly, so a table case can
// state the entire request body: "in/user:hi", "out/assistant:A|B" for an
// assistant output message carrying two text items, "call:name/id",
// "output:id".
//
// Input and output messages are rendered distinctly on purpose: they differ in
// the content type they carry — "input_text" against "output_text" — and the
// Responses API rejects the former on the assistant role, so a replayed
// assistant turn emitted as an input message would be a real defect that a
// shared "assistant:" rendering would hide.
func describeInput(items responses.ResponseInputParam) []string {
	got := make([]string, 0, len(items))
	for _, item := range items {
		switch {
		case item.OfMessage != nil:
			texts := make([]string, 0, len(item.OfMessage.Content.OfInputItemContentList))
			for _, c := range item.OfMessage.Content.OfInputItemContentList {
				if c.OfInputText == nil {
					texts = append(texts, "<non-text>")
					continue
				}
				texts = append(texts, c.OfInputText.Text)
			}
			got = append(got, "in/"+string(item.OfMessage.Role)+":"+strings.Join(texts, "|"))
		case item.OfOutputMessage != nil:
			texts := make([]string, 0, len(item.OfOutputMessage.Content))
			for _, c := range item.OfOutputMessage.Content {
				if c.OfOutputText == nil {
					texts = append(texts, "<non-text>")
					continue
				}
				texts = append(texts, c.OfOutputText.Text)
			}
			got = append(got, "out/assistant:"+strings.Join(texts, "|"))
		case item.OfFunctionCall != nil:
			got = append(got, "call:"+item.OfFunctionCall.Name+"/"+item.OfFunctionCall.CallID)
		case item.OfFunctionCallOutput != nil:
			got = append(got, "output:"+item.OfFunctionCallOutput.CallID.Or(""))
		default:
			got = append(got, "<unrecognized item>")
		}
	}
	return got
}

func TestBuildOpenAIParams_DropsReplayedThoughts(t *testing.T) {
	thought := func(text string) *genai.Part { return &genai.Part{Text: text, Thought: true} }
	modelTurn := func(parts ...*genai.Part) *genai.Content {
		return &genai.Content{Role: string(genai.RoleModel), Parts: parts}
	}
	userTurn := func(parts ...*genai.Part) *genai.Content {
		return &genai.Content{Role: string(genai.RoleUser), Parts: parts}
	}

	tests := []struct {
		name        string
		contents    []*genai.Content
		want        []string
		wantErr     error
		wantErrText string
	}{
		{
			name: "thought_before_answer",
			contents: []*genai.Content{
				genai.NewContentFromText("what is 2+2?", genai.RoleUser),
				modelTurn(thought("do not reveal the scratchpad"), &genai.Part{Text: "4"}),
				genai.NewContentFromText("and 3+3?", genai.RoleUser),
			},
			want: []string{"in/user:what is 2+2?", "out/assistant:4", "in/user:and 3+3?"},
		},
		{
			// A turn that produced only reasoning contributes nothing, rather
			// than an assistant message the model never said.
			name: "thought_only_turn",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				modelTurn(thought("still thinking")),
				genai.NewContentFromText("q2", genai.RoleUser),
			},
			want: []string{"in/user:q", "in/user:q2"},
		},
		{
			// The blank-skip is per text item, not per message: it is what
			// makes dropping blank reasoning a no-op, so it is pinned here.
			name:     "blank_text_is_skipped_beside_real_text",
			contents: []*genai.Content{modelTurn(&genai.Part{Text: "   "}, &genai.Part{Text: "real"})},
			want:     []string{"out/assistant:real"},
		},
		{
			// The same on the user path, which builds an input message through
			// newMessage rather than newOutputMessage.
			name:     "blank_text_is_skipped_beside_real_text_in_user_turn",
			contents: []*genai.Content{userTurn(&genai.Part{Text: "   "}, &genai.Part{Text: "real"})},
			want:     []string{"in/user:real"},
		},
		{
			name:     "thought_between_answers",
			contents: []*genai.Content{modelTurn(&genai.Part{Text: "A"}, thought("T"), &genai.Part{Text: "B"})},
			want:     []string{"out/assistant:A|B"},
		},
		{
			// Sub-agents and A2A peers can hand back a thought on a user turn.
			name:     "thought_in_user_turn",
			contents: []*genai.Content{userTurn(thought("leaked"), &genai.Part{Text: "real"})},
			want:     []string{"in/user:real"},
		},
		{
			// A thought carrying only a signature has no text to leak, but it
			// used to reach the default arm and fail the whole request.
			name: "signature_only_thought",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				modelTurn(&genai.Part{Thought: true, ThoughtSignature: []byte("sig")}),
			},
			want: []string{"in/user:q"},
		},
		{
			// Dropping a thought-marked call would strand its response and
			// fail the request in callTracker.
			name: "thought_marked_call_and_response_survive",
			contents: []*genai.Content{
				modelTurn(&genai.Part{
					Thought:      true,
					FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "c1"},
				}),
				userTurn(&genai.Part{
					Thought:          true,
					FunctionResponse: &genai.FunctionResponse{Name: "lookup", ID: "c1", Response: map[string]any{"ok": true}},
				}),
			},
			want: []string{"call:lookup/c1", "output:c1"},
		},
		{
			// One part carrying both: the reasoning must not reach the wire and
			// the call must still be emitted.
			name: "thought_text_riding_on_a_call",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				modelTurn(&genai.Part{
					Thought:      true,
					Text:         "scratchpad",
					FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "c1"},
				}),
				userTurn(&genai.Part{
					FunctionResponse: &genai.FunctionResponse{Name: "lookup", ID: "c1", Response: map[string]any{"ok": true}},
				}),
			},
			want: []string{"in/user:q", "call:lookup/c1", "output:c1"},
		},
		{
			// Same shape on the response side: the reasoning is dropped and the
			// tool output survives, rather than the reverse.
			name: "thought_text_riding_on_a_response",
			contents: []*genai.Content{
				modelTurn(&genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "c1"}}),
				userTurn(&genai.Part{
					Thought:          true,
					Text:             "scratchpad",
					FunctionResponse: &genai.FunctionResponse{Name: "lookup", ID: "c1", Response: map[string]any{"ok": true}},
				}),
			},
			want: []string{"call:lookup/c1", "output:c1"},
		},
		{
			// Ordinary text riding on a call is not reasoning, so nothing is
			// dropped: the text keeps its place ahead of the call.
			name: "plain_text_riding_on_a_call_keeps_both",
			contents: []*genai.Content{
				modelTurn(&genai.Part{
					Text:         "on it",
					FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "c1"},
				}),
			},
			want: []string{"out/assistant:on it", "call:lookup/c1"},
		},
		{
			// A bare thought signature has nowhere to go in a Responses
			// request, but it is not a reason to fail the conversation.
			name: "signature_without_thought_marker_is_dropped",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				modelTurn(&genai.Part{ThoughtSignature: []byte("sig")}),
			},
			want: []string{"in/user:q"},
		},
		{
			// Marking media as a thought must not smuggle it past the
			// unsupported-part check and out of the request unannounced.
			name: "thought_marked_media_is_still_rejected",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				userTurn(&genai.Part{
					Thought:    true,
					InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				}),
			},
			wantErrText: "unsupported content part",
		},
		{
			// The same part with reasoning text riding on it. Suppressing the
			// text must not also suppress the rejection, or the image leaves
			// the request unannounced — the arm keys on what the part
			// contributed, not on whether it had text.
			name: "thought_text_riding_on_media_is_still_rejected",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				userTurn(&genai.Part{
					Thought:    true,
					Text:       "scratch",
					InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				}),
			},
			wantErrText: "unsupported content part",
		},
		{
			name: "thought_text_riding_on_code_is_still_rejected",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				modelTurn(&genai.Part{
					Thought:        true,
					Text:           "scratch",
					ExecutableCode: &genai.ExecutableCode{Code: "print(1)"},
				}),
			},
			wantErrText: "unsupported content part",
		},
		{
			// Dropping the reasoning must not swallow the role error the same
			// turn would have raised had its text been an answer: nothing
			// buffers, so flushText returns before normalizeRole runs.
			name: "unsupported_role_still_reported_on_a_thought_only_turn",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				{Role: "assistant", Parts: []*genai.Part{thought("scratch")}},
			},
			wantErrText: `unsupported role "assistant"`,
		},
		{
			// The same turn with the marker off, showing the error is reported
			// identically either way.
			name: "unsupported_role_reported_on_an_ordinary_turn",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				{Role: "assistant", Parts: []*genai.Part{{Text: "answer"}}},
			},
			wantErrText: `unsupported role "assistant"`,
		},
		{
			// And on a thought with no text to drop: the part still leaves the
			// request, so the turn is still one the package cannot send.
			name: "unsupported_role_still_reported_on_a_textless_thought",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				{Role: "assistant", Parts: []*genai.Part{{Thought: true}}},
			},
			wantErrText: `unsupported role "assistant"`,
		},
		{
			// Same for a bare signature, which reaches the drop by the other
			// door — the marker unset, the signature alone.
			name: "unsupported_role_still_reported_on_a_bare_signature",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				{Role: "assistant", Parts: []*genai.Part{{ThoughtSignature: []byte("sig")}}},
			},
			wantErrText: `unsupported role "assistant"`,
		},
		{
			// Not a thought at all: an image riding on ordinary text used to
			// leave the request silently, because the text matched an arm and
			// the rejection never ran. The check is independent of what the
			// part contributed, so it is reported here too.
			name: "media_riding_on_plain_text_is_rejected",
			contents: []*genai.Content{
				userTurn(&genai.Part{
					Text:       "describe this",
					InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				}),
			},
			wantErrText: "unsupported content part: InlineData",
		},
		{
			// Same hole on the call arm.
			name: "media_riding_on_a_call_is_rejected",
			contents: []*genai.Content{
				modelTurn(&genai.Part{
					FunctionCall: &genai.FunctionCall{Name: "lookup", ID: "c1"},
					InlineData:   &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				}),
			},
			wantErrText: "unsupported content part: InlineData",
		},
		{
			// A part that carries only bookkeeping reaches nothing: it is not
			// reasoning, so it is reported rather than dropped.
			name: "metadata_only_part_is_reported",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				userTurn(&genai.Part{PartMetadata: map[string]any{"src": "a"}}),
			},
			wantErrText: "unsupported content part: carries nothing to send",
		},
		{
			// buildContentsDefault filters these out upstream, so this is the
			// contract for a caller reaching convertContents directly: an
			// empty part is reported, not quietly skipped.
			name: "empty_part_is_reported",
			contents: []*genai.Content{
				genai.NewContentFromText("q", genai.RoleUser),
				userTurn(&genai.Part{}),
			},
			wantErrText: "unsupported content part: carries nothing to send",
		},
		{
			// Text riding on a response that pairs with nothing. The response
			// used to be discarded because the text matched an arm first, so
			// the pairing the API requires went unchecked; it is now enforced
			// wherever the response appears.
			name: "orphan_response_riding_on_text_is_reported",
			contents: []*genai.Content{
				userTurn(&genai.Part{
					Text:             "here you go",
					FunctionResponse: &genai.FunctionResponse{Name: "lookup"},
				}),
			},
			wantErrText: "missing call id",
		},
		{
			// Same shape on the call arm: a call with no name is unsendable,
			// and riding on text no longer hides it.
			name: "nameless_call_riding_on_text_is_reported",
			contents: []*genai.Content{
				modelTurn(&genai.Part{
					Text:         "on it",
					FunctionCall: &genai.FunctionCall{},
				}),
			},
			wantErr: ErrFunctionCallMissingName,
		},
		{
			// A request left empty by the drop is reported rather than sent,
			// and says the drop emptied it rather than that nothing was sent.
			name:        "only_thoughts",
			contents:    []*genai.Content{modelTurn(thought("scratch"))},
			wantErr:     ErrNoContents,
			wantErrText: "every part was dropped as replayed reasoning",
		},
		{
			// The wrap is keyed on a part having been dropped, not on the
			// request being empty for any reason, so a turn that mixes the two
			// still reports the drop.
			name:        "thought_and_blank_answer",
			contents:    []*genai.Content{modelTurn(thought("scratch"), &genai.Part{Text: "   "})},
			wantErr:     ErrNoContents,
			wantErrText: "every part was dropped as replayed reasoning",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := buildOpenAIParams("fallback", &model.LLMRequest{Contents: tt.contents})
			if tt.wantErr != nil || tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("buildOpenAIParams() err = nil, want an error")
				}
				if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
					t.Fatalf("buildOpenAIParams() err = %v, want %v", err, tt.wantErr)
				}
				if tt.wantErrText != "" && !strings.Contains(err.Error(), tt.wantErrText) {
					t.Errorf("buildOpenAIParams() err = %q, want it to mention %q", err, tt.wantErrText)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildOpenAIParams() err = %v", err)
			}
			if got := describeInput(params.Input.OfInputItemList); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("input items = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestBuildOpenAIParams_NoContentsSentinelIdentity pins which requests get the
// bare ErrNoContents and which get it wrapped, because a caller comparing with
// == rather than errors.Is sees only the bare one. Only a drop that suppressed
// text the model would otherwise have seen earns the wrap.
func TestBuildOpenAIParams_NoContentsSentinelIdentity(t *testing.T) {
	tests := []struct {
		name     string
		contents []*genai.Content
		wantBare bool
	}{
		{"nil_contents", nil, true},
		{"empty_contents", []*genai.Content{}, true},
		{"content_with_no_parts", []*genai.Content{{Role: string(genai.RoleUser)}}, true},
		{"content_with_only_nil_parts", []*genai.Content{
			{Role: string(genai.RoleUser), Parts: []*genai.Part{nil}},
		}, true},
		{
			// Nothing was dropped as reasoning here: the text was accepted and
			// then skipped as blank, exactly as it is without this change.
			"only_blank_text",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "   "}}}},
			true,
		},
		{
			// The two together, which is where the rule earns its keep: the
			// part is reasoning, so it is dropped, but its text was blank and
			// would have been skipped anyway, so the drop is not what emptied
			// the request and the bare sentinel stands.
			"only_blank_reasoning",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "   ", Thought: true}}}},
			true,
		},
		{
			// A thought with nothing to suppress: it leaves the request, but
			// no text of it would ever have reached the model.
			"only_bare_thought",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{{Thought: true}}}},
			true,
		},
		{
			"only_signature",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{{ThoughtSignature: []byte("sig")}}}},
			true,
		},
		{
			"only_reasoning",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "scratch", Thought: true}}}},
			false,
		},
		{
			// The drop still earns the wrap when it suppressed real text, even
			// though a blank sibling is what the request was left with.
			"reasoning_and_blank_answer",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{
				{Text: "scratch", Thought: true}, {Text: "   "},
			}}},
			false,
		},
		{
			// The flag accumulates rather than tracking the last part, so a
			// blank thought after a real one cannot talk it back down.
			"real_reasoning_then_blank_reasoning",
			[]*genai.Content{{Role: string(genai.RoleModel), Parts: []*genai.Part{
				{Text: "scratch", Thought: true}, {Text: "   ", Thought: true},
			}}},
			false,
		},
		{
			// The real turn comes first, so a flag reset per content block
			// would show up here where the reverse order would hide it.
			"real_reasoning_turn_then_blank_one",
			[]*genai.Content{
				{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "scratch", Thought: true}}},
				{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: "   ", Thought: true}}},
			},
			false,
		},
		{
			// The user path builds an input message rather than an output one,
			// and skips blank text through a different function.
			"only_blank_text_user_turn",
			[]*genai.Content{{Role: string(genai.RoleUser), Parts: []*genai.Part{{Text: "   "}}}},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := buildOpenAIParams("fallback", &model.LLMRequest{Contents: tt.contents})
			if !errors.Is(err, ErrNoContents) {
				t.Fatalf("buildOpenAIParams() err = %v, want it to wrap %v", err, ErrNoContents)
			}
			//nolint:errorlint // the point of the test is the identity, not the chain.
			if gotBare := err == ErrNoContents; gotBare != tt.wantBare {
				t.Errorf("err == ErrNoContents is %v, want %v (err = %q)", gotBare, tt.wantBare, err)
			}
		})
	}
}

func TestReplayedReasoning(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{"nil_part", nil, false},
		// Nothing marks these as reasoning, so there is no reason to drop
		// them: convertContents reports them instead.
		{"empty_part", &genai.Part{}, false},
		{"metadata_only", &genai.Part{PartMetadata: map[string]any{"src": "a"}}, false},
		{"thought_text", &genai.Part{Text: "scratch", Thought: true}, true},
		{"thought_signature_only", &genai.Part{Thought: true, ThoughtSignature: []byte("sig")}, true},
		{"bare_thought", &genai.Part{Thought: true}, true},
		{"signature_without_thought_marker", &genai.Part{ThoughtSignature: []byte("sig")}, true},
		{"plain_text", &genai.Part{Text: "answer"}, false},
		{"signature_on_answer", &genai.Part{Text: "answer", ThoughtSignature: []byte("sig")}, false},
		// Marking content as a thought must not make it vanish.
		{
			"thought_marked_inline_data",
			&genai.Part{Thought: true, InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}},
			false,
		},
		{
			"thought_marked_file_data",
			&genai.Part{Thought: true, FileData: &genai.FileData{FileURI: "gs://b/o"}},
			false,
		},
		{
			"thought_marked_executable_code",
			&genai.Part{Thought: true, ExecutableCode: &genai.ExecutableCode{Code: "print(1)"}},
			false,
		},
		{
			"thought_marked_code_result",
			&genai.Part{Thought: true, CodeExecutionResult: &genai.CodeExecutionResult{Output: "1"}},
			false,
		},
		{
			"thought_marked_call",
			&genai.Part{Thought: true, FunctionCall: &genai.FunctionCall{Name: "f"}},
			false,
		},
		{
			"thought_marked_response",
			&genai.Part{Thought: true, FunctionResponse: &genai.FunctionResponse{Name: "f"}},
			false,
		},
		// Server-side tool traffic and transcription are payload too, even
		// though nothing in the repo populates them today.
		{
			"thought_marked_tool_call",
			&genai.Part{Thought: true, ToolCall: &genai.ToolCall{ID: "t1"}},
			false,
		},
		{
			"thought_marked_tool_response",
			&genai.Part{Thought: true, ToolResponse: &genai.ToolResponse{ID: "t1"}},
			false,
		},
		{
			"thought_marked_audio_transcription",
			&genai.Part{Thought: true, AudioTranscription: &genai.Transcription{Text: "hello"}},
			false,
		},
		// These three only qualify media carried in another field, so on a
		// thought they hold nothing back.
		{
			"thought_with_video_metadata",
			&genai.Part{Thought: true, Text: "scratch", VideoMetadata: &genai.VideoMetadata{FPS: genai.Ptr(2.0)}},
			true,
		},
		{
			"thought_with_media_resolution",
			&genai.Part{Thought: true, Text: "scratch", MediaResolution: &genai.PartMediaResolution{
				Level: genai.PartMediaResolutionLevelMediaResolutionLow,
			}},
			true,
		},
		{
			"thought_with_part_metadata",
			&genai.Part{Thought: true, Text: "scratch", PartMetadata: map[string]any{"src": "a"}},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := replayedReasoning(tt.part); got != tt.want {
				t.Errorf("replayedReasoning() = %v, want %v", got, tt.want)
			}
		})
	}
}

// accountedForFields is the specification unsupportedPayload implements: the
// genai.Part fields convertContents knows what to do with, and why.
//
// The list is deliberately restated here instead of derived from the
// production code, so a field genai adds later is absent from it by
// construction and the walk below demands it be reported instead of silently
// accepted — the failure a denylist would not produce.
var accountedForFields = map[string]string{
	"Text":             "sent, or dropped when it is reasoning",
	"Thought":          "the marker deciding which",
	"ThoughtSignature": "no Responses input item can carry one",
	"FunctionCall":     "sent as a function_call item",
	"FunctionResponse": "sent as a function_call_output item",
	"VideoMetadata":    "qualifies media carried in another field",
	"MediaResolution":  "qualifies media carried in another field",
	"PartMetadata":     "caller bookkeeping, never content",
}

func TestUnsupportedPayload_WalksEveryPartField(t *testing.T) {
	partType := reflect.TypeOf(genai.Part{})
	for i := range partType.NumField() {
		field := partType.Field(i)
		t.Run(field.Name, func(t *testing.T) {
			if !field.IsExported() {
				t.Fatalf("genai.Part gained unexported field %s, which this walk cannot set; "+
					"check by hand whether unsupportedPayload should report it", field.Name)
			}
			part := &genai.Part{}
			reflect.ValueOf(part).Elem().Field(i).Set(nonZero(t, field.Type))

			got := unsupportedPayload(part)
			if why, accounted := accountedForFields[field.Name]; accounted {
				if got != "" {
					t.Errorf("unsupportedPayload() = %q for a part carrying only %s, want %q: %s",
						got, field.Name, "", why)
				}
				return
			}
			if got != field.Name {
				t.Fatalf("unsupportedPayload() = %q for a part carrying only %s, want %q: "+
					"a field the package cannot send must be reported, not dropped",
					got, field.Name, field.Name)
			}
			// The predicate agreeing is not enough: the caller has to see the
			// error, and it has to see it for the shape this change is about —
			// the field riding on something sendable, which is what used to
			// carry it out of the request unnoticed.
			want := "unsupported content part: " + field.Name
			for _, ride := range ridingShapes(part) {
				req := &model.LLMRequest{Contents: []*genai.Content{
					{Role: string(genai.RoleModel), Parts: []*genai.Part{ride.part}},
				}}
				_, err := buildOpenAIParams("fallback", req)
				// HasSuffix rather than Contains: one field name can prefix
				// another, and reporting the shorter one must not pass.
				if err == nil || !strings.HasSuffix(err.Error(), want) {
					t.Errorf("buildOpenAIParams() err = %v for %s carried %s, want it to end with %q",
						err, field.Name, ride.name, want)
				}
			}
		})
	}
}

// ridingShapes returns part as it arrives alone and alongside each thing that
// would otherwise be emitted for it, so a field cannot be reported on its own
// and slip through on a part that also had something to send.
func ridingShapes(part *genai.Part) []struct {
	name string
	part *genai.Part
} {
	withText := *part
	withText.Text = "here you go"

	withThought := *part
	withThought.Text = "scratchpad"
	withThought.Thought = true

	withCall := *part
	withCall.FunctionCall = &genai.FunctionCall{Name: "lookup", ID: "c1"}

	return []struct {
		name string
		part *genai.Part
	}{
		{"alone", part},
		{"on text", &withText},
		{"on reasoning text", &withThought},
		{"on a function call", &withCall},
	}
}

// TestReplayedReasoning_EveryUnaccountedFieldDisqualifies is the same walk seen
// from the drop: reasoning is droppable only when the part holds nothing else,
// so a field the package cannot send has to keep the part out of the drop and
// on to the rejection.
func TestReplayedReasoning_EveryUnaccountedFieldDisqualifies(t *testing.T) {
	partType := reflect.TypeOf(genai.Part{})
	for i := range partType.NumField() {
		field := partType.Field(i)
		if _, accounted := accountedForFields[field.Name]; accounted || !field.IsExported() {
			continue
		}
		t.Run(field.Name, func(t *testing.T) {
			part := &genai.Part{Thought: true, Text: "scratch"}
			reflect.ValueOf(part).Elem().Field(i).Set(nonZero(t, field.Type))
			if replayedReasoning(part) {
				t.Errorf("replayedReasoning() = true for a thought carrying %s; "+
					"it would be dropped instead of rejected as unsupported", field.Name)
			}
		})
	}
}

// nonZero builds a non-zero value of typ, for the field walk above.
func nonZero(t *testing.T, typ reflect.Type) reflect.Value {
	t.Helper()
	switch typ.Kind() {
	case reflect.Pointer:
		return reflect.New(typ.Elem())
	case reflect.Map:
		return reflect.MakeMap(typ)
	case reflect.Slice:
		return reflect.MakeSlice(typ, 1, 1)
	case reflect.String:
		return reflect.ValueOf("x").Convert(typ)
	case reflect.Bool:
		return reflect.ValueOf(true).Convert(typ)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return reflect.ValueOf(int64(1)).Convert(typ)
	case reflect.Float32, reflect.Float64:
		return reflect.ValueOf(1.0).Convert(typ)
	default:
		t.Fatalf("genai.Part gained a %s field; teach nonZero how to fill it", typ.Kind())
		return reflect.Value{}
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
// valid one gets reasoning rather than an error.
//
// Go cannot enumerate a string enum's constants, so a level genai adds later is
// caught at runtime by name instead, and the length check below only catches an
// entry added to the map that this list forgot.
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
		// genai calls this one "Default service tier, which is standard", so it
		// lands where standard does rather than on auto, which would hand the
		// caller whichever tier their project happens to have configured.
		{genai.ServiceTierUnspecified, responses.ResponseNewParamsServiceTierDefault},
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
func TestRequestTimeoutHonorsAPositiveTimeout(t *testing.T) {
	timeout := 45 * time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
	if err := applyGenerationConfig(&responses.ResponseNewParams{}, cfg); err != nil {
		t.Fatalf("applyGenerationConfig() error = %v, want nil: a timeout is honored", err)
	}
	if got := requestTimeout(cfg); got != timeout {
		t.Errorf("requestTimeout() = %v, want %v", got, timeout)
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

// Pins against the SDK rather than asserting in prose: openai-go records any
// case-insensitive Authorization as an override and then skips attaching the
// configured API key, so forwarding a caller's header would replace the real
// credential rather than accompany it.
//
// It drives openai-go directly, so a future version that stops overriding fails
// this test and Headers can be reconsidered on evidence.
func TestCallerAuthorizationHeaderDisplacesTheAPIKey(t *testing.T) {
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
		t.Fatalf("baseline Authorization = %s, want the configured key", redact(got))
	}
	_, _ = client.Responses.New(context.Background(), params,
		option.WithHeaderAdd("Authorization", "Bearer caller"))
	if got == "Bearer real-key" {
		t.Fatal("openai-go now keeps the configured key alongside a caller Authorization header; " +
			"the reason HTTPOptions.Headers is rejected no longer holds, so revisit it")
	}
	if got != "Bearer caller" {
		t.Errorf("Authorization = %s, want the caller header to have displaced the key", redact(got))
	}
}

// Headers are ignored rather than refused, so a config that carried them
// against main keeps working here. What must never happen is forwarding them:
// TestCallerAuthorizationHeaderDisplacesTheAPIKey shows what an Authorization header
// would do, and a Gemini credential would simply reach the wrong provider.
func TestHeadersAffectNeitherValidationNorTimeout(t *testing.T) {
	headers := []http.Header{
		{"Authorization": []string{"Bearer caller"}},
		{"authorization": []string{"Bearer lowercase"}},
		{"X-Goog-Api-Key": []string{"gemini-key"}},
		{"X-Trace-Id": []string{"harmless"}},
	}
	// Names, never values: a header value is exactly the thing not to log.
	for _, h := range headers {
		names := slices.Sorted(maps.Keys(h))
		cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Headers: h}}
		if err := applyGenerationConfig(&responses.ResponseNewParams{}, cfg); err != nil {
			t.Fatalf("headers %v: error = %v, want nil: headers are ignored, not refused", names, err)
		}
		if got := requestTimeout(cfg); got != 0 {
			t.Errorf("headers %v: produced a %v timeout, want none", names, got)
		}
	}
}

// The end of that promise, checked where it can actually be observed: no header
// a caller put in HTTPOptions may appear on the outgoing request.
func TestHTTPOptionsHeadersNeverReachTheWire(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "blocking"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) { assertNoHeaderReachesTheWire(t, stream) })
	}
}

func assertNoHeaderReachesTheWire(t *testing.T, stream bool) {
	t.Helper()
	wireTimeout := 30 * time.Second
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","object":"response","status":"completed",` +
			`"output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
			`"content":[{"type":"output_text","text":"hi","annotations":[]}]}]}`))
	}))
	defer srv.Close()

	m, err := NewModel(context.Background(), "gpt-4o-mini", &ClientConfig{APIKey: "real-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		// A timeout is set deliberately. Without one the translation returns
		// early, so header forwarding reintroduced after that point would
		// never run here and the test would pass vacuously.
		Config: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{
			Timeout: &wireTimeout,
			Headers: http.Header{
				"Authorization":  []string{"Bearer caller"},
				"X-Goog-Api-Key": []string{"gemini-key"},
				"X-Trace-Id":     []string{"harmless"},
			},
		}},
	}
	for _, err := range m.GenerateContent(context.Background(), req, stream) {
		if err != nil && !stream {
			t.Fatalf("GenerateContent() error = %v", err)
		}
	}
	if v := got.Get("Authorization"); v != "Bearer real-key" {
		t.Errorf("Authorization = %s, want the configured key: a caller header displaced it", redact(v))
	}
	if v := got.Get("X-Goog-Api-Key"); v != "" {
		t.Errorf("X-Goog-Api-Key = %s, want absent: a Gemini credential reached the provider", redact(v))
	}
	if v := got.Get("X-Trace-Id"); v != "" {
		t.Errorf("X-Trace-Id = %s, want absent: headers are not forwarded", redact(v))
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
		// The budget has to be one applyThinkingConfig actually rejects, or
		// there is no competing error for the named one to win against and the
		// case passes whether precedence works or not.
		{"StopSequences over ThinkingConfig", &genai.GenerateContentConfig{
			StopSequences:  []string{"STOP"},
			ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr(int32(-2))},
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
		"ResponseMIMEType": true,
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

// The timeout has to reach the request, not merely be computed: asserting that
// requestTimeout returns the right duration says nothing about the
// context.WithTimeout wiring in generate and generateStream, which is the only
// thing HTTPOptions still does.
//
// The server answers slowly but successfully, so a wired timeout fails the call
// quickly while an unwired one waits and succeeds — outcomes differing in kind
// rather than in timing, and so not a function of machine load.
func TestHTTPOptionsTimeoutReachesTheRequest(t *testing.T) {
	const serverDelay = 3 * time.Second

	for _, stream := range []bool{false, true} {
		name := "blocking"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			// Closed before the server is, so a handler still sleeping when the
			// caller has already given up does not hold up srv.Close.
			release := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				select {
				case <-time.After(serverDelay):
				case <-r.Context().Done():
					return
				case <-release:
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"r","object":"response","status":"completed",` +
					`"output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
					`"content":[{"type":"output_text","text":"hi","annotations":[]}]}]}`))
			}))
			defer srv.Close()
			defer close(release)

			timeout := 100 * time.Millisecond
			m, err := NewModel(context.Background(), "gpt-4o-mini",
				&ClientConfig{APIKey: "test", BaseURL: srv.URL})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}
			req := &model.LLMRequest{
				Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
				Config:   &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}},
			}

			start := time.Now()
			var got error
			for _, err := range m.GenerateContent(context.Background(), req, stream) {
				if err != nil {
					got = err
					break
				}
			}
			elapsed := time.Since(start)

			if got == nil {
				t.Fatalf("call succeeded after %v with a %v timeout configured: "+
					"the timeout never reached the request", elapsed, timeout)
			}
			// Secondary: the mutation is already caught by the call succeeding
			// above, so this only has to be loose enough not to flake on a
			// loaded machine while still catching a wildly wrong bound.
			if elapsed >= serverDelay {
				t.Errorf("call took %v, i.e. it waited for the server rather than for its %v timeout", elapsed, timeout)
			}
		})
	}
}

// ignoredHTTPOptionFields is part of the contract, so it has to name a real
// field, and every field has to be accounted for in exactly one of the three
// categories the package documents.
func TestHTTPOptionFieldsAreAccountedFor(t *testing.T) {
	honored := map[string]bool{"Timeout": true}
	ignored := make(map[string]bool, len(ignoredHTTPOptionFields))
	for _, name := range ignoredHTTPOptionFields {
		ignored[name] = true
	}
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
		n := 0
		for _, in := range []bool{honored[field.Name], ignored[field.Name], rejected[field.Name]} {
			if in {
				n++
			}
		}
		if n != 1 {
			t.Errorf("genai.HTTPOptions.%s is in %d of {honored, ignored, rejected}, want exactly 1", field.Name, n)
		}
	}
	for name := range ignored {
		if _, ok := optType.FieldByName(name); !ok {
			t.Errorf("ignoredHTTPOptionFields lists %q, which genai.HTTPOptions no longer has", name)
		}
	}
	for name := range rejected {
		if _, ok := optType.FieldByName(name); !ok {
			t.Errorf("unsupportedHTTPOptionFields lists %q, which genai.HTTPOptions no longer has", name)
		}
	}
}

// redact describes a header value without reproducing it. These assertions run
// against stub credentials, but a maintainer debugging a live problem may well
// have swapped a real key in, and a failing test should not put it in the log.
func redact(v string) string {
	if v == "" {
		return "empty"
	}
	return fmt.Sprintf("<%d-byte value>", len(v))
}

// requestTimeout must be safe on its own terms. applyGenerationConfig rejects a
// non-positive timeout before any request is built, so this guard is defense in
// depth — and it is tested directly rather than assumed unreachable, because a
// guard that only holds while callers keep the right order is not a guard.
func TestRequestTimeoutGuardsNonPositiveItself(t *testing.T) {
	for _, d := range []time.Duration{0, -time.Second} {
		timeout := d
		cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}}
		if got := requestTimeout(cfg); got != 0 {
			t.Errorf("requestTimeout(%v) = %v, want 0: a non-positive bound must not become a deadline", d, got)
		}
	}
	// The positive case still comes through, so the guard is not simply off.
	positive := time.Second
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &positive}}
	if got := requestTimeout(cfg); got != positive {
		t.Errorf("requestTimeout(%v) = %v, want it unchanged", positive, got)
	}
}

// An iter.Seq2 may be ranged more than once, and each range is its own call.
// The timeout is applied inside the returned closure, so it must derive from
// the context the caller passed rather than overwrite it: assigning to the
// captured variable would leave the second range starting from the deadline the
// first one had already cancelled, failing instantly and for a reason the
// caller could not see.
func TestGenerateContentIsReRangeableWithATimeout(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "blocking"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) { assertReRangeable(t, stream) })
	}
}

func assertReRangeable(t *testing.T, stream bool) {
	t.Helper()
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","object":"response","status":"completed",` +
			`"output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
			`"content":[{"type":"output_text","text":"hi","annotations":[]}]}]}`))
	}))
	defer srv.Close()

	timeout := 30 * time.Second
	m, err := NewModel(context.Background(), "gpt-4o-mini", &ClientConfig{APIKey: "test", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
		Config:   &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &timeout}},
	}

	seq := m.GenerateContent(context.Background(), req, stream)
	for pass := 1; pass <= 2; pass++ {
		var got error
		for _, err := range seq {
			if err != nil {
				got = err
				break
			}
		}
		if got != nil {
			t.Fatalf("pass %d: error = %v, want nil: the sequence is not reusable, "+
				"which happens when the timeout overwrites the captured context", pass, got)
		}
	}
	if calls != 2 {
		t.Errorf("server saw %d calls, want 2: the second range did not reach it", calls)
	}
}

// geminiShapedHTTPOptions carries one live value per unsupportedHTTPOptionFields
// entry, shared so the pairing test below derives from the cases that actually
// run rather than from a second list agreeing with neither.
//
// Live values matter because a predicate is a closure: a reflection test over
// the names stays green however that closure is edited.
var geminiShapedHTTPOptions = []struct {
	field string
	opts  *genai.HTTPOptions
}{
	{"BaseURL", &genai.HTTPOptions{BaseURL: "https://example.test"}},
	{"BaseURLResourceScope", &genai.HTTPOptions{BaseURLResourceScope: genai.ResourceScope("global")}},
	{"APIVersion", &genai.HTTPOptions{APIVersion: "v1beta"}},
	{"ExtraBody", &genai.HTTPOptions{ExtraBody: map[string]any{"k": "v"}}},
	{"ExtrasRequestProvider", &genai.HTTPOptions{
		ExtrasRequestProvider: func(m map[string]any) map[string]any { return m },
	}},
	{"RetryOptions", &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr(int32(3))}}},
}

func TestApplyGenerationConfigRejectsGeminiShapedHTTPOptions(t *testing.T) {
	for _, tc := range geminiShapedHTTPOptions {
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

// Pairs the table with the cases above, reading both rather than restating
// either, so a predicate added without a case fails here instead of shipping
// untested. The sibling table is guarded the same way.
func TestEveryUnsupportedHTTPOptionFieldIsDriven(t *testing.T) {
	driven := make(map[string]bool, len(geminiShapedHTTPOptions))
	for _, tc := range geminiShapedHTTPOptions {
		driven[tc.field] = true
	}
	for _, field := range unsupportedHTTPOptionFields {
		if !driven[field.name] {
			t.Errorf("unsupportedHTTPOptionFields has %q with no case in "+
				"geminiShapedHTTPOptions, so its predicate never runs", field.name)
		}
	}
	if len(driven) != len(unsupportedHTTPOptionFields) {
		t.Errorf("%d cases for %d predicates", len(driven), len(unsupportedHTTPOptionFields))
	}
}

// Closes the accounting test's blind side: that one compares names against
// three lists, so writing a new field into translated greens it while the field
// is still dropped.
//
// Here every field the lists call translated has to reach the params, with the
// ones needing a whole subsystem to be meaningful left to their own tests.
func TestTranslatedFieldsAreNotRejected(t *testing.T) {
	tests := []struct {
		field string
		cfg   *genai.GenerateContentConfig
	}{
		{"Temperature", &genai.GenerateContentConfig{Temperature: genai.Ptr(float32(0.5))}},
		{"TopP", &genai.GenerateContentConfig{TopP: genai.Ptr(float32(0.9))}},
		{"MaxOutputTokens", &genai.GenerateContentConfig{MaxOutputTokens: 128}},
		{"SystemInstruction", &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("be terse", genai.RoleUser),
		}},
		{"ResponseMIMEType", &genai.GenerateContentConfig{ResponseMIMEType: "application/json"}},
		{"ResponseLogprobs", &genai.GenerateContentConfig{ResponseLogprobs: true}},
		{"Logprobs", &genai.GenerateContentConfig{ResponseLogprobs: true, Logprobs: genai.Ptr(int32(3))}},
		{"ThinkingConfig", &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
		}},
		{"ServiceTier", &genai.GenerateContentConfig{ServiceTier: genai.ServiceTierFlex}},
		{"HTTPOptions.Timeout", &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{Timeout: genai.Ptr(30 * time.Second)},
		}},
	}

	for _, tc := range tests {
		t.Run(tc.field, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, tc.cfg); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v, want nil: %s is listed as translated", err, tc.field)
			}
			// Listed as translated means it reaches the params, not merely that
			// it is tolerated; HTTPOptions.Timeout is the one that lands
			// elsewhere, on the context, so it is checked through requestTimeout.
			if tc.field == "HTTPOptions.Timeout" {
				if requestTimeout(tc.cfg) == 0 {
					t.Error("requestTimeout() = 0, want the configured bound")
				}
				return
			}
			if reflect.DeepEqual(*params, responses.ResponseNewParams{}) {
				t.Errorf("%s left the params untouched, so it is not translated", tc.field)
			}
		})
	}
}
