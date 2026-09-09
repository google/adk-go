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
	"encoding/base64"
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
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{ExecutableCode: &genai.ExecutableCode{Code: "1+1", Language: genai.LanguagePython}},
				},
			},
		},
	}
	if _, err := buildOpenAIParams("fallback", req); err == nil {
		t.Fatalf("expected error for executable code part")
	}
}

// TestBuildOpenAIParams_InlineDataImage guards that a genai.Part.InlineData
// image (base64 bytes + MIME type) is converted into an OpenAI
// responses.ResponseInputImageParam using a base64 data URL, matching
// adk-python's _inline_data_part_to_response_content.
func TestBuildOpenAIParams_InlineDataImage(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: pngBytes}},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("unexpected input items: %+v", items)
	}
	content := items[0].OfMessage.Content.OfInputItemContentList
	if len(content) != 1 || content[0].OfInputImage == nil {
		t.Fatalf("unexpected message content: %+v", content)
	}
	img := content[0].OfInputImage
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
	if !img.ImageURL.Valid() || img.ImageURL.Value != wantURL {
		t.Fatalf("image url = %+v, want %q", img.ImageURL, wantURL)
	}
	if got := img.Type; got != constant.InputImage("input_image") {
		t.Errorf("image type = %q, want input_image", got)
	}
}

// TestBuildOpenAIParams_InlineDataNonImageFile guards that non-image
// InlineData (e.g. a PDF) becomes an input_file item carrying an inline
// base64 data URL rather than an input_image item.
func TestBuildOpenAIParams_InlineDataNonImageFile(t *testing.T) {
	pdfBytes := []byte{0x25, 0x50, 0x44, 0x46}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: pdfBytes}},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("unexpected input items: %+v", items)
	}
	content := items[0].OfMessage.Content.OfInputItemContentList
	if len(content) != 1 || content[0].OfInputFile == nil {
		t.Fatalf("unexpected message content: %+v", content)
	}
	f := content[0].OfInputFile
	wantData := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfBytes)
	if !f.FileData.Valid() || f.FileData.Value != wantData {
		t.Fatalf("file data = %+v, want %q", f.FileData, wantData)
	}
	if got := f.Type; got != constant.InputFile("input_file") {
		t.Errorf("file type = %q, want input_file", got)
	}
}

// TestBuildOpenAIParams_FileDataImageURI guards that a genai.Part.FileData
// pointing at a remote image URI is converted into an input_image item that
// carries the URI directly (no re-encoding).
func TestBuildOpenAIParams_FileDataImageURI(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{FileData: &genai.FileData{FileURI: "https://example.com/cat.png", MIMEType: "image/png"}},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("unexpected input items: %+v", items)
	}
	content := items[0].OfMessage.Content.OfInputItemContentList
	if len(content) != 1 || content[0].OfInputImage == nil {
		t.Fatalf("unexpected message content: %+v", content)
	}
	img := content[0].OfInputImage
	if !img.ImageURL.Valid() || img.ImageURL.Value != "https://example.com/cat.png" {
		t.Fatalf("image url = %+v, want https://example.com/cat.png", img.ImageURL)
	}
}

// TestBuildOpenAIParams_FileDataNonImageURI guards that a genai.Part.FileData
// pointing at a non-image remote file (e.g. a PDF) becomes an input_file item
// carrying the URL, and that an OpenAI file-id-shaped URI ("file-...") is
// routed through FileID instead of FileURL.
func TestBuildOpenAIParams_FileDataNonImageURI(t *testing.T) {
	tests := []struct {
		name       string
		uri        string
		wantFileID string
		wantURL    string
	}{
		{name: "http url", uri: "https://example.com/doc.pdf", wantURL: "https://example.com/doc.pdf"},
		{name: "openai file id", uri: "file-abc123", wantFileID: "file-abc123"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role: string(genai.RoleUser),
						Parts: []*genai.Part{
							{FileData: &genai.FileData{FileURI: tc.uri, MIMEType: "application/pdf"}},
						},
					},
				},
			}
			params, err := buildOpenAIParams("fallback", req)
			if err != nil {
				t.Fatalf("buildOpenAIParams() err = %v", err)
			}
			items := params.Input.OfInputItemList
			if len(items) != 1 || items[0].OfMessage == nil {
				t.Fatalf("unexpected input items: %+v", items)
			}
			content := items[0].OfMessage.Content.OfInputItemContentList
			if len(content) != 1 || content[0].OfInputFile == nil {
				t.Fatalf("unexpected message content: %+v", content)
			}
			f := content[0].OfInputFile
			if tc.wantURL != "" {
				if !f.FileURL.Valid() || f.FileURL.Value != tc.wantURL {
					t.Fatalf("file url = %+v, want %q", f.FileURL, tc.wantURL)
				}
			}
			if tc.wantFileID != "" {
				if !f.FileID.Valid() || f.FileID.Value != tc.wantFileID {
					t.Fatalf("file id = %+v, want %q", f.FileID, tc.wantFileID)
				}
			}
		})
	}
}

// TestBuildOpenAIParams_MixedTextAndImagePreservesOrder guards that a single
// message mixing text and image parts is emitted as ONE message whose
// content list mixes input_text and input_image items IN PART ORDER, rather
// than being dropped, erroring, or reordered by the old text-only
// accumulator.
func TestBuildOpenAIParams_MixedTextAndImagePreservesOrder(t *testing.T) {
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: string(genai.RoleUser),
				Parts: []*genai.Part{
					{Text: "What is in this image?"},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: pngBytes}},
					{Text: "Answer briefly."},
				},
			},
		},
	}
	params, err := buildOpenAIParams("fallback", req)
	if err != nil {
		t.Fatalf("buildOpenAIParams() err = %v", err)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil {
		t.Fatalf("expected exactly one message, got: %+v", items)
	}
	content := items[0].OfMessage.Content.OfInputItemContentList
	if len(content) != 3 {
		t.Fatalf("expected 3 content parts, got %d: %+v", len(content), content)
	}
	if content[0].OfInputText == nil || content[0].OfInputText.Text != "What is in this image?" {
		t.Errorf("content[0] = %+v, want input_text %q", content[0], "What is in this image?")
	}
	if content[1].OfInputImage == nil {
		t.Errorf("content[1] = %+v, want input_image", content[1])
	}
	if content[2].OfInputText == nil || content[2].OfInputText.Text != "Answer briefly." {
		t.Errorf("content[2] = %+v, want input_text %q", content[2], "Answer briefly.")
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
