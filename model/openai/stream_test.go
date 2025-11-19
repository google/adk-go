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

package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/internal/llminternal"
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
	delta := decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"call-1","delta":"{\"city\":\""}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	delta = decodeEvent(t, `{"type":"response.function_call_arguments.delta","item_id":"call-1","delta":"Paris\"}"}`)
	if _, err := tr.process(delta); err != nil {
		t.Fatalf("process(delta) err = %v", err)
	}
	done := decodeEvent(t, `{"type":"response.function_call_arguments.done","item_id":"call-1","name":"lookup","arguments":""}`)
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
