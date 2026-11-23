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

package responses

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	oairesponses "github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// errFunctionCallMissingItemID reports a streamed function call that could not
// be tied to its arguments, because an argument event named its item neither
// by item_id nor by output_index, in a turn whose terminal event did not state
// every call either.
var errFunctionCallMissingItemID = errors.New("openai: streamed function call event missing item_id")

// streamTranslator helps us process OpenAI streaming events by buffering
// function arguments until a complete function call is received.
type streamTranslator struct {
	functionArgs map[string]*strings.Builder
	itemToCallID map[string]string
	itemToName   map[string]string
	// indexToItem maps an output_index to the item ID that
	// "response.output_item.added" announced there, so an argument event
	// carrying only output_index still reaches its call. adk-python keys
	// stream items by output_index ahead of item_id
	// (integrations/openai/_openai_responses_llm.py, _stream_output_key) and
	// never fails a call it cannot place. Here item_id stays the primary key,
	// as the protocol defines it, and a call that cannot be placed fails the
	// turn rather than reaching the tool without its arguments.
	indexToItem map[int64]string
	// announcedCalls counts function_call items "response.output_item.added"
	// announced, and doneCalls the "response.function_call_arguments.done"
	// events, placed or not. See calls.
	announcedCalls int
	doneCalls      int
	// orphanArgs is set when argument deltas arrived that no item could
	// claim. They are dropped, so a call stating no arguments of its own is
	// held back rather than emitted without them.
	orphanArgs bool
	// missingItemID is set when a call could not be emitted: its done event
	// named no item, or it was held back for orphanArgs. generateStream fails
	// the turn unless the terminal event states every call itself.
	missingItemID bool
}

func newStreamTranslator() *streamTranslator {
	return &streamTranslator{
		functionArgs: make(map[string]*strings.Builder),
		itemToCallID: make(map[string]string),
		itemToName:   make(map[string]string),
		indexToItem:  make(map[int64]string),
	}
}

func (t *streamTranslator) process(evt oairesponses.ResponseStreamEventUnion) (*genai.GenerateContentResponse, error) {
	// We process each incoming OpenAI streaming event and convert it into a
	// generic genai.GenerateContentResponse.
	switch evt.Type {
	case responseOutputTextDelta:
		delta := evt.AsResponseOutputTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// For text deltas, we create a response with a single text part.
		return shared.SinglePartResponse(&genai.Part{Text: delta.Delta}), nil
	case responseRefusalDelta:
		delta := evt.AsResponseRefusalDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// Blocking responses expose refusals as text, so streaming does the same.
		return shared.SinglePartResponse(&genai.Part{Text: delta.Delta}), nil
	case responseReasoningTextDelta:
		delta := evt.AsResponseReasoningTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// Reasoning text deltas are treated as thought parts.
		return shared.SinglePartResponse(&genai.Part{Text: delta.Delta, Thought: true}), nil
	case responseReasoningSummaryTextDelta:
		delta := evt.AsResponseReasoningSummaryTextDelta()
		if delta.Delta == "" {
			return nil, nil
		}
		// Reasoning summary deltas are also treated as thought parts.
		return shared.SinglePartResponse(&genai.Part{Text: delta.Delta, Thought: true}), nil
	case responseFunctionCallArgumentsDelta:
		delta := evt.AsResponseFunctionCallArgumentsDelta()
		id := t.itemID(delta.ItemID, delta.OutputIndex, delta.JSON.OutputIndex.Valid())
		if id == "" {
			// Arguments no call can claim. They are dropped, and a call that
			// later states no arguments of its own is held back rather than
			// emitted without these (see emitFunctionCall).
			if delta.Delta != "" {
				t.orphanArgs = true
			}
			return nil, nil
		}
		if delta.Delta != "" {
			// We buffer function call arguments as they stream in, identified by ItemID.
			buf := t.buffer(id)
			buf.WriteString(delta.Delta)
		}
		return nil, nil
	case responseFunctionCallArgumentsDone:
		done := evt.AsResponseFunctionCallArgumentsDone()
		t.doneCalls++
		id := t.itemID(done.ItemID, done.OutputIndex, done.JSON.OutputIndex.Valid())
		if id == "" {
			// Nothing names this call, so it cannot be emitted.
			t.missingItemID = true
			return nil, nil
		}
		// When function call arguments are complete, we emit the full function call.
		part, err := t.emitFunctionCall(id, done.Arguments)
		if err != nil || part == nil {
			return nil, err
		}
		return shared.SinglePartResponse(part), nil
	case responseFailed:
		failed := evt.AsResponseFailed()
		// Built by the same renderer the blocking path uses, so one server
		// failure reads the same however it arrived.
		return nil, failedResponseError(&failed.Response)
	case errorEvent:
		// Generic stream errors are also returned.
		// Same treatment as a failed response body: the text is the server's,
		// so it is capped, and quoted rather than interpolated bare.
		if msg := shared.ClipServerText(evt.Message); msg != "" {
			return nil, fmt.Errorf("openai stream error: %q", msg)
		}
		return nil, fmt.Errorf("openai stream error")
	case responseOutputItemAdded:
		added := evt.AsResponseOutputItemAdded()
		if added.Item.Type == "function_call" {
			t.announcedCalls++
			if added.JSON.OutputIndex.Valid() {
				t.indexToItem[added.OutputIndex] = added.Item.ID
			}
		}
		if added.Item.ID != "" && added.Item.CallID != "" {
			t.itemToCallID[added.Item.ID] = added.Item.CallID
		}
		if added.Item.ID != "" && added.Item.Name != "" {
			t.itemToName[added.Item.ID] = added.Item.Name
		}
		return nil, nil
	case responseOutputTextDone,
		responseRefusalDone,
		responseReasoningTextDone,
		responseReasoningSummaryTextDone,
		responseCompleted,
		responseIncomplete,
		responseInProgress,
		responseOutputItemDone:
		// Informational events with no part of their own. generateStream reads
		// the finish reason off the terminal ones directly.
		return nil, nil
	default:
		// We ignore any other unknown event types.
		return nil, nil
	}
}

// calls is how many function calls the stream showed evidence of: one per
// announcement, or one per done event when a call arrived without being
// announced. generateStream compares it with the calls the terminal event
// states.
func (t *streamTranslator) calls() int {
	return max(t.announcedCalls, t.doneCalls)
}

// lostCall reports whether the stream ended with a call it could not emit with
// its arguments: one held back or never placed (missingItemID), or unclaimed
// argument deltas that no done event followed for some call, so they belonged
// to a call the stream never finished.
func (t *streamTranslator) lostCall() bool {
	return t.missingItemID || t.orphanArgs && (t.doneCalls == 0 || t.doneCalls < t.announcedCalls)
}

// itemID names the item an argument event belongs to: its item_id, or failing
// that the item "response.output_item.added" announced at its output_index.
// It returns "" when neither identifies one. indexValid reports whether the
// event carried output_index at all, which its zero value cannot.
func (t *streamTranslator) itemID(itemID string, outputIndex int64, indexValid bool) string {
	if itemID != "" || !indexValid {
		return itemID
	}
	return t.indexToItem[outputIndex]
}

// buffer is a helper that provides a strings.Builder for a given ItemID.
// We use this to accumulate partial function call arguments as they stream in.
func (t *streamTranslator) buffer(id string) *strings.Builder {
	if b, ok := t.functionArgs[id]; ok {
		return b
	}
	b := &strings.Builder{}
	t.functionArgs[id] = b
	return b
}

// emitFunctionCall is called when we receive a "response.function_call_arguments.done" event
// for item id. We construct a genai.Part with a genai.FunctionCall by retrieving the complete,
// buffered function arguments (either from the done event or our functionArgs map)
// and unmarshaling them from JSON. Finally, we clean up the buffered arguments.
//
// It returns a nil part when the call states no arguments while argument
// deltas no item could claim have streamed: those may have been this call's,
// and emitting it with none would run the tool on input the model never gave.
func (t *streamTranslator) emitFunctionCall(id, arguments string) (*genai.Part, error) {
	payload := arguments
	if payload == "" {
		if b, ok := t.functionArgs[id]; ok {
			payload = b.String()
		}
	}
	delete(t.functionArgs, id)

	callID := id
	if mapped, ok := t.itemToCallID[id]; ok {
		callID = mapped
	}
	delete(t.itemToCallID, id)

	// The done event carries no name of its own: it arrived on the
	// function_call item of the earlier response.output_item.added event.
	name := t.itemToName[id]
	delete(t.itemToName, id)

	if payload == "" && t.orphanArgs {
		t.missingItemID = true
		return nil, nil
	}
	if payload == "" {
		payload = "{}"
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(payload), &args); err != nil {
		return nil, fmt.Errorf("openai: parse streamed function args: %w", err)
	}
	if args == nil {
		// Match the blocking path: JSON null means the call takes no arguments.
		args = map[string]any{}
	}
	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			Name: name,
			ID:   callID,
			Args: args,
		},
	}, nil
}
