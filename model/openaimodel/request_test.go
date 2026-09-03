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
	"reflect"
	"strings"
	"testing"

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
// Input and output messages are rendered distinctly on purpose. They differ in
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
			wantErrText: "carries nothing to send",
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
			wantErrText: "carries nothing to send",
		},
		{
			// A request left empty by the drop is reported rather than sent,
			// and says the drop emptied it rather than that nothing was sent.
			name:        "only_thoughts",
			contents:    []*genai.Content{modelTurn(thought("scratch"))},
			wantErr:     ErrNoContents,
			wantErrText: "every part was dropped",
		},
		{
			// The same, reached the other way: a blank answer buffers text
			// that newOutputMessage then skips, so nothing is emitted.
			name:        "only_blank_text",
			contents:    []*genai.Content{modelTurn(&genai.Part{Text: "   "})},
			wantErr:     ErrNoContents,
			wantErrText: "every part was dropped",
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
// genai.Part fields convertContents knows what to do with, and why. Every other
// field must be reported rather than dropped, which is what the walk below
// asserts field by field.
//
// The list is deliberately restated here instead of derived from the production
// code, so that widening what the package silently accepts has to be written
// down twice. A field genai adds later is absent from it by construction, so
// the walk demands it be reported — the failure a denylist would not produce.
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
			// error. Building the request proves convertContents does not
			// swallow it on the way out.
			req := &model.LLMRequest{Contents: []*genai.Content{
				{Role: string(genai.RoleUser), Parts: []*genai.Part{part}},
			}}
			_, err := buildOpenAIParams("fallback", req)
			want := "unsupported content part: " + field.Name
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("buildOpenAIParams() err = %v, want it to mention %q", err, want)
			}
		})
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
