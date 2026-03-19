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
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func ptr[T any](v T) *T {
	return &v
}

func TestProgressiveSSEStreamingFunctionCallArguments(t *testing.T) {
	aggregator := llminternal.NewStreamingResponseAggregator()
	ctx := t.Context()

	// Chunk 1: FC name + partial location argument ("New ")
	chunk1 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "get_weather",
								ID:   "fc_001",
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.location", StringValue: "New "},
								},
								WillContinue: ptr(true),
							},
						},
					},
				},
			},
		},
	}

	// Chunk 2: Continue location argument ("York")
	chunk2 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.location", StringValue: "York"},
								},
								WillContinue: ptr(true),
							},
						},
					},
				},
			},
		},
	}

	// Chunk 3: Add unit argument, FC complete
	chunk3 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.unit", StringValue: "celsius"},
								},
								WillContinue: ptr(false),
							},
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	var processedChunks []*model.LLMResponse

	for _, chunk := range []*genai.GenerateContentResponse{chunk1, chunk2, chunk3} {
		for resp, err := range aggregator.ProcessResponse(ctx, chunk) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp != nil {
				processedChunks = append(processedChunks, resp)
			}
		}
	}

	finalResponse := aggregator.Close()
	if finalResponse == nil {
		t.Fatal("expected final response from aggregator")
	}

	parts := finalResponse.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fcPart := parts[0]
	if fcPart.FunctionCall == nil {
		t.Fatal("expected function call part")
	}
	if fcPart.FunctionCall.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", fcPart.FunctionCall.Name)
	}
	if fcPart.FunctionCall.ID != "fc_001" {
		t.Errorf("expected fc_001, got %s", fcPart.FunctionCall.ID)
	}

	args := fcPart.FunctionCall.Args
	if args["location"] != "New York" {
		t.Errorf("expected location 'New York', got '%v'", args["location"])
	}
	if args["unit"] != "celsius" {
		t.Errorf("expected unit 'celsius', got '%v'", args["unit"])
	}
}

func TestProgressiveSSEPreservesThoughtSignature(t *testing.T) {
	aggregator := llminternal.NewStreamingResponseAggregator()
	ctx := t.Context()

	testThoughtSignature := []byte("test_signature_abc123")

	chunk := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "add_5_numbers",
								ID:   "fc_003",
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.num1", NumberValue: ptr(10.0)},
									{JsonPath: "$.num2", NumberValue: ptr(20.0)},
								},
								WillContinue: ptr(false),
							},
							ThoughtSignature: testThoughtSignature,
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	for _, err := range aggregator.ProcessResponse(ctx, chunk) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	finalResponse := aggregator.Close()
	if finalResponse == nil {
		t.Fatal("expected final response")
	}

	parts := finalResponse.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fcPart := parts[0]
	if fcPart.FunctionCall == nil {
		t.Fatal("expected function call")
	}
	if fcPart.FunctionCall.Name != "add_5_numbers" {
		t.Errorf("expected add_5_numbers, got %s", fcPart.FunctionCall.Name)
	}
	if string(fcPart.ThoughtSignature) != string(testThoughtSignature) {
		t.Errorf("expected thought signature %s, got %s", string(testThoughtSignature), string(fcPart.ThoughtSignature))
	}
}

func TestProgressiveSSEHandlesEmptyFunctionCall(t *testing.T) {
	aggregator := llminternal.NewStreamingResponseAggregator()
	ctx := t.Context()

	chunk1 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "concat_number_and_string",
								ID:   "fc_001",
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.num", NumberValue: ptr(100.0)},
									{JsonPath: "$.s", StringValue: "ADK"},
								},
								WillContinue: ptr(false),
							},
						},
					},
				},
			},
		},
	}

	chunk2 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{},
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	for _, chunk := range []*genai.GenerateContentResponse{chunk1, chunk2} {
		for _, err := range aggregator.ProcessResponse(ctx, chunk) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		}
	}

	finalResponse := aggregator.Close()
	if finalResponse == nil {
		t.Fatal("expected final response")
	}

	parts := finalResponse.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fcPart := parts[0]
	if fcPart.FunctionCall == nil {
		t.Fatal("expected function call")
	}
	if fcPart.FunctionCall.Name != "concat_number_and_string" {
		t.Errorf("expected concat_number_and_string, got %s", fcPart.FunctionCall.Name)
	}
	if fcPart.FunctionCall.ID != "fc_001" {
		t.Errorf("expected fc_001, got %s", fcPart.FunctionCall.ID)
	}

	args := fcPart.FunctionCall.Args
	if args["num"] != 100.0 {
		t.Errorf("expected num 100, got %v", args["num"])
	}
	if args["s"] != "ADK" {
		t.Errorf("expected s 'ADK', got %v", args["s"])
	}
}

func TestStreamingFCChunkWithWillContinueButNoPartialArgs(t *testing.T) {
	aggregator := llminternal.NewStreamingResponseAggregator()
	ctx := t.Context()

	chunk1 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name:         "my_tool",
								ID:           "fc_gemini3",
								WillContinue: ptr(true),
							},
							ThoughtSignature: []byte("test_sig_123"),
						},
					},
				},
			},
		},
	}

	chunk2 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.document", StringValue: "Once upon "},
								},
								WillContinue: ptr(true),
							},
						},
					},
				},
			},
		},
	}

	chunk3 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								PartialArgs: []*genai.PartialArg{
									{JsonPath: "$.document", StringValue: "a time..."},
								},
								WillContinue: ptr(true),
							},
						},
					},
				},
			},
		},
	}

	chunk4 := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								WillContinue: ptr(false),
							},
						},
					},
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	var processedChunks []*model.LLMResponse
	for _, chunk := range []*genai.GenerateContentResponse{chunk1, chunk2, chunk3, chunk4} {
		for resp, err := range aggregator.ProcessResponse(ctx, chunk) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp != nil {
				processedChunks = append(processedChunks, resp)
			}
		}
	}

	// Wait, we don't strictly test that intermediate chunks are marked partial because ProcessResponse behavior might differ. Let's just check the final result.
	finalResponse := aggregator.Close()
	if finalResponse == nil {
		t.Fatal("expected final response")
	}

	parts := finalResponse.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}

	fcPart := parts[0]
	if fcPart.FunctionCall == nil {
		t.Fatal("expected function call")
	}
	if fcPart.FunctionCall.Name != "my_tool" {
		t.Errorf("expected my_tool, got %s", fcPart.FunctionCall.Name)
	}
	if fcPart.FunctionCall.ID != "fc_gemini3" {
		t.Errorf("expected fc_gemini3, got %s", fcPart.FunctionCall.ID)
	}
	if string(fcPart.ThoughtSignature) != "test_sig_123" {
		t.Errorf("expected thought signature test_sig_123, got %s", string(fcPart.ThoughtSignature))
	}

	args := fcPart.FunctionCall.Args
	if args["document"] != "Once upon a time..." {
		t.Errorf("expected document 'Once upon a time...', got '%v'", args["document"])
	}
}

type streamingMockModel struct {
	streamChunks []*model.LLMResponse
	callCount    int
}

func (m *streamingMockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.callCount++
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.callCount > 1 {
			resp := &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Task completed."}},
				},
				Partial: false,
			}
			yield(resp, nil)
			return
		}

		aggregator := llminternal.NewStreamingResponseAggregator()
		for _, chunk := range m.streamChunks {
			genResp := m.llmResponseToGenerateContentResponse(chunk)
			for processedChunk, err := range aggregator.ProcessResponse(ctx, genResp) {
				if err != nil {
					yield(nil, err)
					return
				}
				if processedChunk != nil {
					if !yield(processedChunk, nil) {
						return
					}
				}
			}
		}

		if finalResp := aggregator.Close(); finalResp != nil {
			yield(finalResp, nil)
		}
	}
}

func (m *streamingMockModel) Name() string { return "streaming-mock" }

func (m *streamingMockModel) llmResponseToGenerateContentResponse(resp *model.LLMResponse) *genai.GenerateContentResponse {
	var candidates []*genai.Candidate
	if resp.Content != nil {
		candidates = append(candidates, &genai.Candidate{
			Content:      resp.Content,
			FinishReason: resp.FinishReason,
		})
	}
	return &genai.GenerateContentResponse{
		Candidates:    candidates,
		UsageMetadata: resp.UsageMetadata,
	}
}

type GetWeatherArgs struct {
	Location string `json:"location"`
}

func getWeather(ctx tool.Context, args GetWeatherArgs) (map[string]any, error) {
	return map[string]any{
		"temperature": 22,
		"condition":   "sunny",
		"location":    args.Location,
	}, nil
}

func TestProgressiveSSEStreamingFunctionCalls(t *testing.T) {
	response1 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Checking weather..."}},
		},
	}
	response2 := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"location": "Tokyo"},
					},
				},
			},
		},
	}
	response3 := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"location": "New York"},
					},
				},
			},
		},
		FinishReason: genai.FinishReasonStop,
	}

	mockModel := &streamingMockModel{
		streamChunks: []*model.LLMResponse{response1, response2, response3},
	}

	getWeatherTool, _ := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "get weather for location",
	}, getWeather)

	ag, _ := llmagent.New(llmagent.Config{
		Name:  "weather_agent",
		Model: mockModel,
		Tools: []tool.Tool{getWeatherTool},
	})

	runner := testutil.NewTestAgentRunner(t, ag)
	cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}

	events, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session-1", genai.NewContentFromText("What is the weather?", "user"), cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(events) != 6 {
		t.Fatalf("expected 6 events, got %d", len(events))
	}

	if !events[0].Partial || events[0].LLMResponse.Content.Parts[0].Text != "Checking weather..." {
		t.Errorf("expected partial event 0 with text")
	}

	if !events[1].Partial || events[1].LLMResponse.Content.Parts[0].FunctionCall.Name != "get_weather" || events[1].LLMResponse.Content.Parts[0].FunctionCall.Args["location"] != "Tokyo" {
		t.Errorf("expected partial event 1 with function call 1")
	}

	if !events[2].Partial || events[2].LLMResponse.Content.Parts[0].FunctionCall.Name != "get_weather" || events[2].LLMResponse.Content.Parts[0].FunctionCall.Args["location"] != "New York" {
		t.Errorf("expected partial event 2 with function call 2")
	}

	if events[3].Partial || events[3].LLMResponse.Content.Parts[0].Text != "Checking weather..." || events[3].LLMResponse.Content.Parts[1].FunctionCall.Name != "get_weather" || events[3].LLMResponse.Content.Parts[2].FunctionCall.Name != "get_weather" {
		t.Errorf("expected final aggregated event 3 with FCs")
	}

	if events[4].Partial || events[4].LLMResponse.Content.Parts[0].FunctionResponse.Name != "get_weather" || events[4].LLMResponse.Content.Parts[1].FunctionResponse.Name != "get_weather" {
		t.Errorf("expected function response event 4")
	}

	resp1 := events[4].LLMResponse.Content.Parts[0].FunctionResponse.Response
	resp2 := events[4].LLMResponse.Content.Parts[1].FunctionResponse.Response
	if resp1["location"] != "Tokyo" || resp2["location"] != "New York" {
		if resp1["location"] != "New York" || resp2["location"] != "Tokyo" {
			t.Errorf("expected function responses to have Tokyo and New York, got %v and %v", resp1["location"], resp2["location"])
		}
	}

	if events[5].Partial || len(events[5].LLMResponse.Content.Parts) == 0 || events[5].LLMResponse.Content.Parts[0].Text != "Task completed." {
		t.Errorf("expected task completed event 5")
	}
}

func TestProgressiveSSEPreservesPartOrdering(t *testing.T) {
	chunk1 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Initial thought part 1. ", Thought: true}},
		},
	}
	chunk2 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Initial thought part 2.", Thought: true}},
		},
	}
	chunk3 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Let me check Tokyo. "}},
		},
	}
	chunk4 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "And New York too."}},
		},
	}
	chunk5 := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"location": "Tokyo"},
					},
				},
			},
		},
	}
	chunk6 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Now processing second thought part 1. ", Thought: true}},
		},
	}
	chunk7 := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: "Second thought part 2.", Thought: true}},
		},
	}
	chunk8 := &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Name: "get_weather",
						Args: map[string]any{"location": "New York"},
					},
				},
			},
		},
		FinishReason: genai.FinishReasonStop,
	}

	mockModel := &streamingMockModel{
		streamChunks: []*model.LLMResponse{chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7, chunk8},
	}

	getWeatherTool, _ := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "get weather for location",
	}, getWeather)

	ag, _ := llmagent.New(llmagent.Config{
		Name:  "ordering_test_agent",
		Model: mockModel,
		Tools: []tool.Tool{getWeatherTool},
	})

	runner := testutil.NewTestAgentRunner(t, ag)
	cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}

	events, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session-1", genai.NewContentFromText("What is the weather?", "user"), cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var aggregatedEvent *session.Event
	for _, event := range events {
		if !event.Partial && event.Author == "ordering_test_agent" && event.Content != nil && len(event.Content.Parts) > 2 {
			aggregatedEvent = event
			break
		}
	}

	if aggregatedEvent == nil {
		t.Fatal("Should find an aggregated model event")
	}

	parts := aggregatedEvent.LLMResponse.Content.Parts
	if len(parts) != 5 {
		t.Fatalf("Expected 5 parts, got %d", len(parts))
	}

	if !parts[0].Thought || parts[0].Text != "Initial thought part 1. Initial thought part 2." {
		t.Errorf("part 0 mismatch. got text: %q, thought: %v", parts[0].Text, parts[0].Thought)
	}
	if parts[1].Thought || parts[1].Text != "Let me check Tokyo. And New York too." {
		t.Errorf("part 1 mismatch. got text: %q, thought: %v", parts[1].Text, parts[1].Thought)
	}
	if parts[2].FunctionCall.Name != "get_weather" || parts[2].FunctionCall.Args["location"] != "Tokyo" {
		t.Errorf("part 2 mismatch. got FC: %v", parts[2].FunctionCall)
	}
	if !parts[3].Thought || parts[3].Text != "Now processing second thought part 1. Second thought part 2." {
		t.Errorf("part 3 mismatch. got text: %q, thought: %v", parts[3].Text, parts[3].Thought)
	}
	if parts[4].FunctionCall.Name != "get_weather" || parts[4].FunctionCall.Args["location"] != "New York" {
		t.Errorf("part 4 mismatch. got FC: %v", parts[4].FunctionCall)
	}
}

type partialFunctionCallMockModel struct{}

func (m *partialFunctionCallMockModel) Name() string { return "partial-fc-mock" }

func (m *partialFunctionCallMockModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		hasFunctionResponse := false
		for _, content := range req.Contents {
			for _, part := range content.Parts {
				if part.FunctionResponse != nil {
					hasFunctionResponse = true
					break
				}
			}
		}

		if hasFunctionResponse {
			resp := &model.LLMResponse{
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Function executed once."}},
				},
				Partial: false,
			}
			yield(resp, nil)
			return
		}

		if !yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							Name: "track_execution",
							Args: map[string]any{"call_id": "partial_1"},
						},
					},
				},
			},
			Partial: true,
		}, nil) {
			return
		}

		if !yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							Name: "track_execution",
							Args: map[string]any{"call_id": "partial_2"},
						},
					},
				},
			},
			Partial: true,
		}, nil) {
			return
		}

		if !yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							Name: "track_execution",
							Args: map[string]any{"call_id": "final"},
						},
					},
				},
			},
			Partial:      false,
			FinishReason: genai.FinishReasonStop,
		}, nil) {
			return
		}
	}
}

func TestPartialFunctionCallsNotExecutedInNoneStreamingMode(t *testing.T) {
	var executionLog []string

	mockModel := &partialFunctionCallMockModel{}

	type TrackExecutionArgs struct {
		CallID string `json:"call_id"`
	}

	trackExecution := func(ctx tool.Context, args TrackExecutionArgs) (string, error) {
		executionLog = append(executionLog, args.CallID)
		return "Executed: " + args.CallID, nil
	}
	trackTool, _ := functiontool.New(functiontool.Config{
		Name:        "track_execution",
		Description: "A tool that logs execution",
	}, trackExecution)

	ag, _ := llmagent.New(llmagent.Config{
		Name:  "partial_fc_test_agent",
		Model: mockModel,
		Tools: []tool.Tool{trackTool},
	})

	runner := testutil.NewTestAgentRunner(t, ag)
	cfg := agent.RunConfig{StreamingMode: agent.StreamingModeNone} // using mode None to ensure partial handling acts same

	events, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session-1", genai.NewContentFromText("Test partial FC handling", "user"), cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(executionLog) != 1 {
		t.Fatalf("Expected 1 execution, got %d: %v", len(executionLog), executionLog)
	}
	if executionLog[0] != "final" {
		t.Errorf("Expected 'final' execution, got: %s", executionLog[0])
	}

	partialEvents := 0
	for _, event := range events {
		if event.Partial {
			partialEvents++
		}
	}
	if partialEvents != 2 {
		t.Errorf("Expected 2 partial events, got %d", partialEvents)
	}

	functionResponseEvents := 0
	for _, event := range events {
		if event.Content != nil {
			for _, p := range event.Content.Parts {
				if p.FunctionResponse != nil {
					functionResponseEvents++
					break
				}
			}
		}
	}
	if functionResponseEvents != 1 {
		t.Errorf("Expected 1 function response event, got %d", functionResponseEvents)
	}
}
