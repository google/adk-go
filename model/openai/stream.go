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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// streamTranslator helps us process OpenAI streaming events by buffering
// function arguments until a complete function call is received.
type streamTranslator struct {
	functionArgs map[string]*strings.Builder
}

func newStreamTranslator() *streamTranslator {
	return &streamTranslator{
		functionArgs: make(map[string]*strings.Builder),
	}
}

func (t *streamTranslator) process(evt responses.ResponseStreamEventUnion) (*genai.GenerateContentResponse, error) {
	// We process each incoming OpenAI streaming event and convert it into a
	// generic genai.GenerateContentResponse.
	switch evt.Type {
	case responseOutputTextDelta:
		delta := evt.AsResponseOutputTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// For text deltas, we create a response with a single text part.
		return singlePartResponse(&genai.Part{Text: delta.Delta}), nil
	case responseReasoningTextDelta:
		delta := evt.AsResponseReasoningTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// Reasoning text deltas are treated as thought parts.
		return singlePartResponse(&genai.Part{Text: delta.Delta, Thought: true}), nil
	case responseReasoningSummaryTextDelta:
		delta := evt.AsResponseReasoningSummaryTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// Reasoning summary deltas are also treated as thought parts.
		return singlePartResponse(&genai.Part{Text: delta.Delta, Thought: true}), nil
	case responseFunctionCallArgumentsDelta:
		delta := evt.AsResponseFunctionCallArgumentsDelta()
		if delta.Delta != "" {
			// We buffer function call arguments as they stream in, identified by ItemID.
			buf := t.buffer(delta.ItemID)
			buf.WriteString(delta.Delta)
		}
		return nil, nil
	case responseFunctionCallArgumentsDone:
		done := evt.AsResponseFunctionCallArgumentsDone()
		// When function call arguments are complete, we emit the full function call.
		part, err := t.emitFunctionCall(done)
		if err != nil {
			return nil, err
		}
		return singlePartResponse(part), nil
	case "response.failed":
		failed := evt.AsResponseFailed()
		// If the response failed, we return an error with the message.
		return nil, fmt.Errorf("openai response failed: %s", failed.Response.Error.Message)
	case "error":
		// Generic stream errors are also returned.
		if evt.Message != "" {
			return nil, fmt.Errorf("openai stream error: %s", evt.Message)
		}
		return nil, fmt.Errorf("openai stream error")
	case responseOutputTextDone,
		responseReasoningTextDone,
		responseReasoningSummaryTextDone,
		responseCompleted,
		responseInProgress,
		responseOutputItemAdded,
		responseOutputItemDone:
		// These are informational events that don't directly translate to a new part.
		return nil, nil
	default:
		// We ignore any other unknown event types.
		return nil, nil
	}
}

// buffer is a helper that provides a strings.Builder for a given ItemID.
// We use this to accumulate partial function call arguments as they stream in.
func (t *streamTranslator) buffer(id string) *strings.Builder {
	if id == "" {
		id = "default"
	}
	if b, ok := t.functionArgs[id]; ok {
		return b
	}
	b := &strings.Builder{}
	t.functionArgs[id] = b
	return b
}

// emitFunctionCall is called when we receive a "response.function_call_arguments.done" event.
// We construct a genai.Part with a genai.FunctionCall by retrieving the complete,
// buffered function arguments (either from the done event or our functionArgs map)
// and unmarshaling them from JSON. Finally, we clean up the buffered arguments.
func (t *streamTranslator) emitFunctionCall(done responses.ResponseFunctionCallArgumentsDoneEvent) (*genai.Part, error) {
	payload := done.Arguments
	if payload == "" {
		if b, ok := t.functionArgs[done.ItemID]; ok {
			payload = b.String()
		}
	}
	delete(t.functionArgs, done.ItemID)
	if payload == "" {
		payload = "{}"
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(payload), &args); err != nil {
		return nil, fmt.Errorf("openai: parse streamed function args: %w", err)
	}
	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			Name: done.Name,
			ID:   done.ItemID,
			Args: args,
		},
	}, nil
}

func singlePartResponse(part *genai.Part) *genai.GenerateContentResponse {
	if part == nil {
		return nil
	}
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: &genai.Content{
					Role:  string(genai.RoleModel),
					Parts: []*genai.Part{part},
				},
				FinishReason: genai.FinishReasonUnspecified,
			},
		},
	}
}
