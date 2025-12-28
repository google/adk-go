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

package llminternal_test

import (
	"bytes"
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	llminternal "google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
)

type streamAggregatorTest struct {
	name                 string
	initialResponses     []*genai.Content
	numberOfStreamCalls  int
	streamResponsesCount int
	want                 []*genai.Content
	wantPartial          []bool
}

func TestStreamAggregator(t *testing.T) {
	ctx := t.Context()
	testCases := []streamAggregatorTest{
		{
			name: "two streams of 2 responses each",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  2,
			streamResponsesCount: 2,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				// Results from second GenerateStream call
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false,
				// Results from second GenerateStream call
				true, true, false,
			},
		},
		{
			name: "two streams of 3 and 2 responses each",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response5", "model"),
			},
			numberOfStreamCalls:  2,
			streamResponsesCount: 3,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response1response2response3", "model"),
				// Results from second GenerateStream call
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response5", "model"),
				genai.NewContentFromText("response4response5", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, true, false,
				// Results from second GenerateStream call
				true, true, false,
			},
		},
		{
			name: "stream with intermediate response should reset",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				nil, // force reset with empty context
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 5,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				nil, // proxy still send the nil
				// Results from second GenerateStream call
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false, false,
				true, true, false,
			},
		},
		{
			name: "stream with audio should reset",
			initialResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 5,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromText("response1response2", "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromText("response3", "model"),
				genai.NewContentFromText("response4", "model"),
				genai.NewContentFromText("response3response4", "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				true, true, false, false,
				true, true, false,
			},
		},
		{
			name: "audio stream should not generate any aggregated",
			initialResponses: []*genai.Content{
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
			},
			numberOfStreamCalls:  1,
			streamResponsesCount: 3,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
				genai.NewContentFromParts([]*genai.Part{{VideoMetadata: &genai.VideoMetadata{}}}, "model"),
			},
			wantPartial: []bool{
				// Results from first GenerateStream call
				false, false, false,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			responsesCopy := make([]*genai.Content, len(tc.initialResponses))
			copy(responsesCopy, tc.initialResponses)

			mockModel := &testutil.MockModel{
				Responses:            responsesCopy,
				StreamResponsesCount: tc.streamResponsesCount,
			}

			count := 0
			callCount := 0
			for callCount < tc.numberOfStreamCalls {
				for got, err := range mockModel.GenerateStream(ctx, &model.LLMRequest{}) {
					if err != nil {
						t.Fatalf("found error while iterating stream")
					}
					if count >= len(tc.want) {
						t.Fatalf("stream generated more values than the expected %d", len(tc.want))
					}
					if diff := cmp.Diff(tc.want[count], got.Content); diff != "" {
						t.Errorf("Model.GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.Content, tc.want[count], diff)
					}
					if got.Partial != tc.wantPartial[count] {
						t.Errorf("Model.GenerateStream() = %v, want Partial value %v\n", got.Partial, tc.wantPartial[count])
					}
					count++
				}
				callCount++
			}
			if count != len(tc.want) {
				t.Errorf("unexpected stream length, expected %d got %d", len(tc.want), count)
			}
		})
	}
}

func TestStreamAggregatorStreamingFunctionCallArguments(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	thoughtSignature := []byte("signature-bytes")
	var finalResponse *model.LLMResponse

	process := func(resp *genai.GenerateContentResponse) {
		for llmResp, err := range aggregator.ProcessResponse(ctx, resp) {
			if err != nil {
				t.Fatalf("ProcessResponse returned error: %v", err)
			}
			if llmResp == nil || llmResp.Content == nil || len(llmResp.Content.Parts) == 0 {
				continue
			}
			part := llmResp.Content.Parts[0]
			if part.FunctionCall != nil && !llmResp.Partial {
				finalResponse = llmResp
			}
		}
	}

	process(newFunctionCallChunk("get_weather", "fc_001", thoughtSignature, true, []*genai.PartialArg{
		{JsonPath: "$.location", StringValue: "New "},
	}...))

	if finalResponse != nil {
		t.Fatalf("got final response before stream finished: %+v", finalResponse)
	}

	process(newFunctionCallChunk("", "fc_001", nil, true, []*genai.PartialArg{
		{JsonPath: "$.location", StringValue: "York"},
	}...))

	process(newFunctionCallChunk("", "fc_001", nil, false, []*genai.PartialArg{
		{JsonPath: "$.unit", StringValue: "celsius"},
	}...))

	if finalResponse == nil {
		t.Fatalf("expected final response after streaming function call")
	}

	if len(finalResponse.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(finalResponse.Content.Parts))
	}
	fcPart := finalResponse.Content.Parts[0]
	if fcPart.FunctionCall == nil {
		t.Fatalf("expected function call part in final response")
	}
	if got, want := fcPart.FunctionCall.Args["location"], "New York"; got != want {
		t.Fatalf("location arg mismatch: got %v want %v", got, want)
	}
	if got, want := fcPart.FunctionCall.Args["unit"], "celsius"; got != want {
		t.Fatalf("unit arg mismatch: got %v want %v", got, want)
	}
	if got := fcPart.FunctionCall.ID; got != "fc_001" {
		t.Fatalf("function call id mismatch: got %q want %q", got, "fc_001")
	}
	if !bytes.Equal(fcPart.ThoughtSignature, thoughtSignature) {
		t.Fatalf("thought signature mismatch: got %v want %v", fcPart.ThoughtSignature, thoughtSignature)
	}

	if closeResp := aggregator.Close(); closeResp != nil {
		t.Fatalf("expected no additional response from Close, got %+v", closeResp)
	}
}

func TestStreamAggregatorKeepsPendingFunctionCallWhenEmptyChunkArrives(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	thoughtSignature := []byte("signature-bytes")

	process := func(resp *genai.GenerateContentResponse) *model.LLMResponse {
		for llmResp, err := range aggregator.ProcessResponse(ctx, resp) {
			if err != nil {
				t.Fatalf("ProcessResponse returned error: %v", err)
			}
			if llmResp != nil {
				return llmResp
			}
		}
		return nil
	}

	// Start a streaming function call with partial args.
	process(newFunctionCallChunk("get_weather", "fc_001", thoughtSignature, true, []*genai.PartialArg{
		{JsonPath: "$.location", StringValue: "San"},
	}...))

	// Simulate an empty function call chunk (no PartialArgs).
	process(newFunctionCallChunk("", "fc_001", nil, true))

	// Finish stream without more args. Close should flush the pending function call.
	finalResp := aggregator.Close()
	if finalResp == nil || finalResp.Content == nil || len(finalResp.Content.Parts) == 0 {
		t.Fatalf("expected final response from Close, got nil/empty")
	}

	part := finalResp.Content.Parts[0]
	if part.FunctionCall == nil {
		t.Fatalf("expected function call part in final response")
	}
	if got, want := part.FunctionCall.Args["location"], "San"; got != want {
		t.Fatalf("location arg mismatch: got %v want %v", got, want)
	}
	if !bytes.Equal(part.ThoughtSignature, thoughtSignature) {
		t.Fatalf("thought signature mismatch: got %v want %v", part.ThoughtSignature, thoughtSignature)
	}
}

func TestStreamAggregatorParallelFunctionCallsPreserveOrderAndSignature(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	thoughtSignature := []byte("signature-bytes")

	part1 := genai.NewPartFromFunctionCall("fn_one", map[string]any{"x": "1"})
	part1.ThoughtSignature = thoughtSignature
	part2 := genai.NewPartFromFunctionCall("fn_two", map[string]any{"y": "2"})

	responses := collectResponses(t, aggregator, ctx, newPartsChunk([]*genai.Part{part1, part2}, genai.FinishReasonStop))
	finalResp := lastNonPartial(responses)
	if finalResp == nil || finalResp.Content == nil {
		t.Fatalf("expected final response with content")
	}

	if got := len(finalResp.Content.Parts); got != 2 {
		t.Fatalf("expected 2 parts, got %d", got)
	}
	if finalResp.Content.Parts[0].FunctionCall == nil || finalResp.Content.Parts[1].FunctionCall == nil {
		t.Fatalf("expected function call parts")
	}
	if !bytes.Equal(finalResp.Content.Parts[0].ThoughtSignature, thoughtSignature) {
		t.Fatalf("first function call signature mismatch: got %v want %v", finalResp.Content.Parts[0].ThoughtSignature, thoughtSignature)
	}
	if len(finalResp.Content.Parts[1].ThoughtSignature) != 0 {
		t.Fatalf("expected second function call to have no signature")
	}
	if got, want := finalResp.Content.Parts[0].FunctionCall.Name, "fn_one"; got != want {
		t.Fatalf("first function call name mismatch: got %q want %q", got, want)
	}
	if got, want := finalResp.Content.Parts[1].FunctionCall.Name, "fn_two"; got != want {
		t.Fatalf("second function call name mismatch: got %q want %q", got, want)
	}
}

func TestStreamAggregatorSequentialFunctionCallsPreserveSignatures(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	sig1 := []byte("sig-one")
	sig2 := []byte("sig-two")

	part1 := genai.NewPartFromFunctionCall("fn_first", map[string]any{"a": "1"})
	part1.ThoughtSignature = sig1
	part2 := genai.NewPartFromFunctionCall("fn_second", map[string]any{"b": "2"})
	part2.ThoughtSignature = sig2

	responses := collectResponses(t, aggregator, ctx,
		newPartsChunk([]*genai.Part{part1}, genai.FinishReasonStop),
		newPartsChunk([]*genai.Part{part2}, genai.FinishReasonStop),
	)

	var gotSigs [][]byte
	for _, resp := range responses {
		if resp == nil || resp.Partial || resp.Content == nil || len(resp.Content.Parts) == 0 {
			continue
		}
		part := resp.Content.Parts[0]
		if part.FunctionCall != nil {
			gotSigs = append(gotSigs, part.ThoughtSignature)
		}
	}
	if len(gotSigs) != 2 {
		t.Fatalf("expected 2 function call responses, got %d", len(gotSigs))
	}
	if !bytes.Equal(gotSigs[0], sig1) || !bytes.Equal(gotSigs[1], sig2) {
		t.Fatalf("function call signatures mismatch: got %v want [%v %v]", gotSigs, sig1, sig2)
	}
}

func TestStreamAggregatorPreservesSignatureOnEmptyFinalTextPart(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	signature := []byte("final-signature")

	responses := collectResponses(t, aggregator, ctx,
		newTextChunk("Hello", nil, false, genai.FinishReason("")),
		newTextChunk("", signature, false, genai.FinishReasonStop),
	)

	finalResp := lastNonPartial(responses)
	if finalResp == nil || finalResp.Content == nil {
		t.Fatalf("expected final response with content")
	}
	if got := len(finalResp.Content.Parts); got != 2 {
		t.Fatalf("expected 2 parts, got %d", got)
	}
	if finalResp.Content.Parts[0].Text != "Hello" {
		t.Fatalf("unexpected first part text: %q", finalResp.Content.Parts[0].Text)
	}
	if len(finalResp.Content.Parts[1].ThoughtSignature) == 0 {
		t.Fatalf("expected signature on final empty part")
	}
	if !bytes.Equal(finalResp.Content.Parts[1].ThoughtSignature, signature) {
		t.Fatalf("signature mismatch: got %v want %v", finalResp.Content.Parts[1].ThoughtSignature, signature)
	}
}

func TestStreamAggregatorDoesNotMergeSignedTextPart(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aggregator := llminternal.NewStreamingResponseAggregator()
	signature := []byte("text-signature")

	responses := collectResponses(t, aggregator, ctx,
		newTextChunk("A", nil, false, genai.FinishReason("")),
		newTextChunk("B", signature, false, genai.FinishReason("")),
		newTextChunk("C", nil, false, genai.FinishReasonStop),
	)

	finalResp := lastNonPartial(responses)
	if finalResp == nil || finalResp.Content == nil {
		t.Fatalf("expected final response with content")
	}
	if got := len(finalResp.Content.Parts); got != 3 {
		t.Fatalf("expected 3 parts, got %d", got)
	}
	if finalResp.Content.Parts[0].Text != "A" || finalResp.Content.Parts[1].Text != "B" || finalResp.Content.Parts[2].Text != "C" {
		t.Fatalf("unexpected text parts: %q %q %q", finalResp.Content.Parts[0].Text, finalResp.Content.Parts[1].Text, finalResp.Content.Parts[2].Text)
	}
	if !bytes.Equal(finalResp.Content.Parts[1].ThoughtSignature, signature) {
		t.Fatalf("signed part signature mismatch: got %v want %v", finalResp.Content.Parts[1].ThoughtSignature, signature)
	}
}

func newFunctionCallChunk(name, id string, sig []byte, willContinue bool, args ...*genai.PartialArg) *genai.GenerateContentResponse {
	part := &genai.Part{
		FunctionCall: &genai.FunctionCall{
			Name:         name,
			ID:           id,
			PartialArgs:  args,
			WillContinue: genai.Ptr(willContinue),
		},
	}
	if len(sig) > 0 {
		part.ThoughtSignature = sig
	}
	finishReason := genai.FinishReason("")
	if !willContinue {
		finishReason = genai.FinishReasonStop
	}
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{part},
				},
				FinishReason: finishReason,
			},
		},
	}
}

func newTextChunk(text string, sig []byte, thought bool, finishReason genai.FinishReason) *genai.GenerateContentResponse {
	part := &genai.Part{Text: text, Thought: thought}
	if len(sig) > 0 {
		part.ThoughtSignature = sig
	}
	return newPartsChunk([]*genai.Part{part}, finishReason)
}

func newPartsChunk(parts []*genai.Part, finishReason genai.FinishReason) *genai.GenerateContentResponse {
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content:      &genai.Content{Role: "model", Parts: parts},
				FinishReason: finishReason,
			},
		},
	}
}

type responseAggregator interface {
	ProcessResponse(ctx context.Context, genResp *genai.GenerateContentResponse) iter.Seq2[*model.LLMResponse, error]
	Close() *model.LLMResponse
}

func collectResponses(t *testing.T, aggregator responseAggregator, ctx context.Context, responses ...*genai.GenerateContentResponse) []*model.LLMResponse {
	t.Helper()
	var out []*model.LLMResponse
	for _, resp := range responses {
		for llmResp, err := range aggregator.ProcessResponse(ctx, resp) {
			if err != nil {
				t.Fatalf("ProcessResponse returned error: %v", err)
			}
			if llmResp != nil {
				out = append(out, llmResp)
			}
		}
	}
	if final := aggregator.Close(); final != nil {
		out = append(out, final)
	}
	return out
}

func lastNonPartial(responses []*model.LLMResponse) *model.LLMResponse {
	for i := len(responses) - 1; i >= 0; i-- {
		if responses[i] != nil && !responses[i].Partial {
			return responses[i]
		}
	}
	return nil
}
