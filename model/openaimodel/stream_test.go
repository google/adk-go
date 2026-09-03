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
	"testing"

	"github.com/openai/openai-go/v3/responses"

	"google.golang.org/adk/v2/internal/llminternal"
)

func decodeEvent(t *testing.T, body string) responses.ResponseStreamEventUnion {
	t.Helper()
	var evt responses.ResponseStreamEventUnion
	if err := json.Unmarshal([]byte(body), &evt); err != nil {
		t.Fatalf("decodeEvent: %v", err)
	}
	return evt
}

func TestStreamTranslator_TextDelta(t *testing.T) {
	tr := newStreamTranslator()
	event := decodeEvent(t, `{"type":"response.output_text.delta","delta":"chunk"}`)
	resp, err := tr.process(event)
	if err != nil {
		t.Fatalf("process() err = %v", err)
	}
	if resp == nil || resp.Candidates[0].Content.Parts[0].Text != "chunk" {
		t.Fatalf("unexpected translation: %+v", resp)
	}
}

func TestStreamTranslator_FunctionCall(t *testing.T) {
	tr := newStreamTranslator()
	added := decodeEvent(t, `{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_real"}}`)
	if _, err := tr.process(added); err != nil {
		t.Fatalf("process(added) err = %v", err)
	}
	delta := decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":\""}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	delta = decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"Paris\"}"}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	done := decodeEvent(t, `{"type":"response.function_call_arguments.done","item_id":"fc_1","name":"lookup","arguments":""}`)
	resp, err := tr.process(done)
	if err != nil {
		t.Fatalf("process(done) err = %v", err)
	}
	part := resp.Candidates[0].Content.Parts[0]
	if part.FunctionCall == nil || part.FunctionCall.Name != "lookup" {
		t.Fatalf("function call not translated: %+v", part)
	}
	if part.FunctionCall.Args["city"] != "Paris" {
		t.Fatalf("args mismatch: %+v", part.FunctionCall.Args)
	}
	if part.FunctionCall.ID != "call_real" {
		t.Fatalf("call ID mismatch, got %q, want %q", part.FunctionCall.ID, "call_real")
	}
}

func TestStreamTranslator_WithAggregator(t *testing.T) {
	tr := newStreamTranslator()
	aggregator := llminternal.NewStreamingResponseAggregator()

	events := []responses.ResponseStreamEventUnion{
		decodeEvent(t, `{"type":"response.output_text.delta","delta":"hel"}`),
		decodeEvent(t, `{"type":"response.output_text.delta","delta":"lo"}`),
	}
	var finalText string
	for _, evt := range events {
		resp, err := tr.process(evt)
		if err != nil || resp == nil {
			t.Fatalf("unexpected translator result: resp=%v err=%v", resp, err)
		}
		for llmResp, err := range aggregator.ProcessResponse(context.Background(), resp) {
			if err != nil {
				t.Fatalf("aggregator err = %v", err)
			}
			if !llmResp.Partial && llmResp.Content != nil && len(llmResp.Content.Parts) > 0 {
				finalText += llmResp.Content.Parts[0].Text
			}
		}
	}
	if final := aggregator.Close(); final != nil && final.Content != nil && len(final.Content.Parts) > 0 {
		finalText += final.Content.Parts[0].Text
	}
	if finalText != "hello" {
		t.Fatalf("aggregated text mismatch got=%q", finalText)
	}
}

func TestStreamTranslator_FunctionCall_MissingDoneName(t *testing.T) {
	tr := newStreamTranslator()
	added := decodeEvent(t, `{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_real","name":"lookup"}}`)
	if _, err := tr.process(added); err != nil {
		t.Fatalf("process(added) err = %v", err)
	}
	delta := decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":\""}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	delta = decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"Paris\"}"}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	done := decodeEvent(t, `{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":""}`)
	resp, err := tr.process(done)
	if err != nil {
		t.Fatalf("process(done) err = %v", err)
	}
	part := resp.Candidates[0].Content.Parts[0]
	if part.FunctionCall == nil || part.FunctionCall.Name != "lookup" {
		t.Fatalf("function call not translated: %+v", part)
	}
	if part.FunctionCall.Args["city"] != "Paris" {
		t.Fatalf("args mismatch: %+v", part.FunctionCall.Args)
	}
	if part.FunctionCall.ID != "call_real" {
		t.Fatalf("call ID mismatch, got %q, want %q", part.FunctionCall.ID, "call_real")
	}
}

func TestStreamTranslator_ResponseFailed(t *testing.T) {
	tr := newStreamTranslator()
	event := decodeEvent(t, `{"type":"response.failed","response":{"id":"resp_123","status":"failed","error":{"code":"server_error","message":"the model failed to generate a response"}}}`)
	resp, err := tr.process(event)
	if resp != nil {
		t.Errorf("process() = %+v, want nil alongside the error", resp)
	}
	if !errors.Is(err, ErrResponseFailed) {
		t.Fatalf("process() err = %v, want errors.Is(err, ErrResponseFailed)", err)
	}
	if want := `openai: response failed (id "resp_123", code "server_error"): the model failed to generate a response`; err.Error() != want {
		t.Errorf("process() err = %q, want %q", err, want)
	}
}

// TestStreamTranslator_ResponseFailed_PathsAgree pins that one server failure
// reads the same on the blocking path and as a streamed "response.failed".
// Comparing the two texts to each other, not to literals, is what stops them
// drifting apart again.
func TestStreamTranslator_ResponseFailed_PathsAgree(t *testing.T) {
	tests := []struct {
		name    string
		errJSON string
	}{
		{name: "message and code", errJSON: `,"error":{"code":"server_error","message":"upstream exploded"}`},
		{name: "message only", errJSON: `,"error":{"message":"upstream exploded"}`},
		{name: "code only", errJSON: `,"error":{"code":"rate_limit_exceeded"}`},
		{name: "no error object", errJSON: ``},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			event := decodeEvent(t, `{"type":"response.failed","response":{"status":"failed"`+tc.errJSON+`}}`)
			failed := event.AsResponseFailed()

			_, streamErr := newStreamTranslator().process(event)
			_, blockErr := convertResponse(&failed.Response)

			for path, err := range map[string]error{"stream": streamErr, "blocking": blockErr} {
				if !errors.Is(err, ErrResponseFailed) {
					t.Fatalf("%s path err = %v, want errors.Is(err, ErrResponseFailed)", path, err)
				}
			}
			if streamErr.Error() != blockErr.Error() {
				t.Errorf("paths disagree:\n stream   = %q\n blocking = %q", streamErr, blockErr)
			}
		})
	}
}
