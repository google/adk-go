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

package llminternal

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// ContentRequestProcessor populates the LLMRequest's Contents based on
// the InvocationContext that includes the previous events.
func ContentsRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/contents.py) - extract function call results, etc.
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		// Do nothing.
		return nil // In python, no error is yielded.
	}
	fn := buildContentsDefault // "" or "default".
	if llmAgent.internal().IncludeContents == "none" {
		// Include current turn context only (no conversation history)
		fn = buildContentsCurrentTurnContextOnly
	}
	var events []*session.Event
	if ctx.Session() != nil {
		for e := range ctx.Session().Events().All() {
			events = append(events, e)
		}
	}
	contents, err := fn(ctx.Agent().Name(), ctx.Branch(), events)
	if err != nil {
		return err
	}
	req.Contents = append(req.Contents, contents...)
	return nil
}

// buildContentsDefault returns the contents for the LLM request by applying
// filtering, rearrangement, and content processing to the given events.
func buildContentsDefault(agentName, invocationBranch string, events []*session.Event) ([]*genai.Content, error) {
	// parse the events, leaving the contents and the function calls and responses from the current agent.
	var filtered []*session.Event
	for _, ev := range events {
		content := utils.Content(ev)
		// Skip events without content or generated neither by user nor
		// by model.
		// e.g. events purely for mutating session states.
		if content == nil || content.Role == "" || len(content.Parts) == 0 {
			// TODO: log a bad event with content but no Role is skipped
			// Note: python checks here if content.Parts[0] is an empty string and skip if so.
			// But unlike python that distinguishes None vs empty string, two cases are indistinguishable in Go.
			continue
		}
		// Skip events that do not belong to the current branch.
		// TODO: can we use a richer type for branch (e.g. []string) instead of using string prefix test?
		if !eventBelongsToBranch(invocationBranch, ev) {
			continue
		}
		if isAuthEvent(ev) {
			continue
		}
		if isOtherAgentReply(agentName, ev) {
			filtered = append(filtered, ConvertForeignEvent(ev))
		} else {
			filtered = append(filtered, ev)
		}
	}

	//  src/google/adk/flows/llm_flows/contents.py
	// 	 - _rearrange_events_for_async_function_response
	filtered, err := rearrangeEventsForLatestFunctionResponse(filtered)
	if err != nil {
		return nil, err
	}
	//   - _rearrange_events_for_async_function_responses_in_history
	filtered, err = rearrangeEventsForFunctionResponsesInHistory(filtered)
	if err != nil {
		return nil, err
	}

	var contents []*genai.Content
	for _, ev := range filtered {
		content := clone(utils.Content(ev))
		if content == nil {
			continue
		}
		utils.RemoveClientFunctionCallID(content)
		contents = append(contents, content)
	}
	return contents, nil
}

func eventBelongsToBranch(invocationBranch string, event *session.Event) bool {
	if invocationBranch == "" {
		return true
	}
	if event.Branch == invocationBranch {
		return true
	}
	// We use dot to delimit branch nodes. To avoid simple prefix match
	// (e.g. agent_0 unexpectedly matching agent_00), require either perfect branch
	// match, or match prefix with an additional explicit '.'
	return strings.HasPrefix(invocationBranch, event.Branch+".")
}

// rearrangeEventsForLatestFunctionResponse
// This function only acts if the very last event is a function response.
// It searches backward for the matching call, deletes all intervening events,
// and appends a single (merged) response.
// If the latest function_response is for an async function_call, all events
// between the initial function_call and the latest function_response will be removed.
func rearrangeEventsForLatestFunctionResponse(events []*session.Event) ([]*session.Event, error) {
	if len(events) < 2 {
		return events, nil
	}

	lastEvent := events[len(events)-1]
	lastResponses := listFunctionResponsesFromEvent(lastEvent)
	// No need to process, since the latest event is not function_response.
	if len(lastResponses) == 0 {
		return events, nil
	}

	// Create response id set
	responseIDs := make(map[string]struct{})
	for _, res := range lastResponses {
		responseIDs[res.ID] = struct{}{}
	}

	// Check if its already in the correct position
	prevEvent := events[len(events)-2]
	prevCalls := listFunctionCallsFromEvent(prevEvent)
	if len(prevCalls) > 0 {
		for _, call := range prevCalls {
			if _, found := responseIDs[call.ID]; found {
				// The latest response is already matched with the immediately
				// preceding call event. The history is clean. Nothing to do.
				return events, nil
			}
		}
	}

	functionCallEventIdx := -1
	var allCallIDsFromMatchingEvent map[string]struct{}

SearchLoop: // A label to allow breaking out of the nested loop
	for idx := len(events) - 2; idx >= 0; idx-- {
		event := events[idx]
		calls := listFunctionCallsFromEvent(event)

		if len(calls) > 0 {
			for _, call := range calls {
				if _, found := responseIDs[call.ID]; found {
					// Match found. This is the event we're looking for.
					functionCallEventIdx = idx

					// Create a new set of all call IDs from this specific event
					allCallIDsFromMatchingEvent = make(map[string]struct{})
					for _, c := range calls {
						allCallIDsFromMatchingEvent[c.ID] = struct{}{}
					}

					// Validation check
					// last response event should only contain the responses for the
					// function calls in the same function call event
					for respID := range responseIDs {
						if _, exists := allCallIDsFromMatchingEvent[respID]; !exists {
							return nil, fmt.Errorf(
								"validation failed: last response event has IDs not in the matching call event. Call IDs: %v, Response IDs: %v",
								allCallIDsFromMatchingEvent, responseIDs,
							)
						}
					}

					// Update the tracked IDs to be ALL IDs from the call event
					responseIDs = allCallIDsFromMatchingEvent

					// Exit the search loop
					break SearchLoop
				}
			}
		}
	}

	if functionCallEventIdx == -1 {
		return nil, fmt.Errorf(
			"no function call event found for function responses ids: %v",
			responseIDs,
		)
	}

	// Collect all function response events *between* the call and the last response.
	var responseEventsToMerge []*session.Event
	for i := functionCallEventIdx + 1; i < len(events)-1; i++ {
		event := events[i]
		responses := listFunctionResponsesFromEvent(event)
		if len(responses) == 0 {
			continue
		}

		// Check if this event contains any response relevant to our call.
		isRelated := false
		for _, res := range responses {
			if _, exists := responseIDs[res.ID]; exists {
				isRelated = true
				break
			}
		}

		if isRelated {
			responseEventsToMerge = append(responseEventsToMerge, event)
		}
	}

	// Add the final response event itself to the list to be merged.
	responseEventsToMerge = append(responseEventsToMerge, events[len(events)-1])

	resultEvents := events[:functionCallEventIdx+1]
	mergedEvent, err := mergeFunctionResponseEvents(responseEventsToMerge)
	if err != nil {
		return nil, err
	}
	resultEvents = append(resultEvents, mergedEvent)
	return resultEvents, nil
}

// rearrangeEventsForFunctionResponsesInHistory reorganizes an entire event history to ensure
// every function call event is immediately followed by a single, consolidated
// function response event.
//
// This function processes the whole slice of events to clean up and correctly
// pair function calls with their corresponding responses, which is especially
// useful for histories involving long running tool calls where
// responses may not have originally been consecutive. It preserves all
// non-tool-call events (like user messages) in their original order.
//
// It returns a new, correctly ordered slice of events or an error if the
// history is malformed (e.g., a response is found without a corresponding call).
func rearrangeEventsForFunctionResponsesInHistory(events []*session.Event) ([]*session.Event, error) {
	if len(events) < 2 {
		return events, nil
	}

	// Create a map to store the index of the event containing each function response.
	callIDToResponseEventIndex := make(map[string]int)
	for i, event := range events {
		responses := listFunctionResponsesFromEvent(event)

		if len(responses) > 0 {
			for _, res := range responses {
				callIDToResponseEventIndex[res.ID] = i
			}
		}
	}

	// Rebuild the event list
	var resultEvents []*session.Event

	for _, event := range events {
		// If the event contains responses, skip it. It will be handled
		// when we process its corresponding call event.
		if len(listFunctionResponsesFromEvent(event)) > 0 {
			continue
		}

		calls := listFunctionCallsFromEvent(event)
		if len(calls) == 0 {
			// This is a regular event (e.g., user message). Just append it.
			resultEvents = append(resultEvents, event)
		} else {
			// This is a function call event, append it and search for responses
			resultEvents = append(resultEvents, event)

			// Find the unique indices of all corresponding response events.
			// Using a map[int]struct{} as a set.
			responseEventIndicesSet := make(map[int]struct{})
			for _, call := range calls {
				if index, found := callIDToResponseEventIndex[call.ID]; found {
					responseEventIndicesSet[index] = struct{}{}
				}
			}

			// If no responses were found for any calls in this event, continue.
			if len(responseEventIndicesSet) == 0 {
				continue
			}

			// If there's only one unique response event, append it directly.
			if len(responseEventIndicesSet) == 1 {
				for index := range responseEventIndicesSet { // A trick to get the single key
					resultEvents = append(resultEvents, events[index])
				}
			} else {
				// Multiple response events exist for that function call so we merge them.
				// Collect and sort the indices to process events in order.
				var sortedIndices []int
				for index := range responseEventIndicesSet {
					sortedIndices = append(sortedIndices, index)
				}
				sort.Ints(sortedIndices)

				// Collect the actual event objects to be merged.
				eventsToMerge := make([]*session.Event, len(sortedIndices))
				for i, index := range sortedIndices {
					eventsToMerge[i] = events[index]
				}

				// Merge the events and append the single result.
				mergedEvent, err := mergeFunctionResponseEvents(eventsToMerge)
				if err != nil {
					return nil, fmt.Errorf("failed to merge response events: %w", err)
				}
				resultEvents = append(resultEvents, mergedEvent)
			}
		}
	}

	return resultEvents, nil
}

// mergeFunctionResponseEvents merges a list of function response events into one.
//
// Its key goal is to ensure that a function call event is followed by a
// single, consolidated response event containing all relevant parts.
//
// The input `functionResponseEvents` must meet several requirements:
//  1. The list must be sorted in increasing order of timestamp.
//  2. The first event is treated as the initial "base" response event.
//  3. All later events must contain at least one response part related
//     to the original function call.
//
// The function returns a single merged event with the following properties:
//  1. Function response parts from later events will replace any matching
//     (by function call ID) response parts from the initial event.
//  2. All non-function-response parts (e.g., text) are appended to the
//     end of the part list.
//
// Caveat: This implementation doesn't support a parallel function call
// event that contains async function calls of the same name.
func mergeFunctionResponseEvents(functionResponseEvents []*session.Event) (*session.Event, error) {
	if len(functionResponseEvents) == 0 {
		return nil, fmt.Errorf("at least one function_response event is required")
	}

	// 1. Use the first event as the base
	mergedEvent := cloneEvent(functionResponseEvents[0])
	if mergedEvent == nil {
		return nil, fmt.Errorf("mergedEvent based on the first event should not be nil")
	}
	if mergedEvent.Content == nil {
		return nil, fmt.Errorf("content for the first event should not be nil")
	}
	partsInMergedEvent := mergedEvent.LLMResponse.Content.Parts

	if len(partsInMergedEvent) == 0 {
		return nil, fmt.Errorf("there should be at least one function_response part")
	}

	// 2. Create an index (map) of function_response parts by their ID
	partIndicesInMergedEvent := make(map[string]int)
	for idx, part := range partsInMergedEvent {
		if part.FunctionResponse != nil {
			functionCallID := part.FunctionResponse.ID
			partIndicesInMergedEvent[functionCallID] = idx
		}
	}

	// 3. Merge subsequent events
	for _, event := range functionResponseEvents[1:] {
		if len(event.LLMResponse.Content.Parts) == 0 {
			return nil, fmt.Errorf("event should contain at least one part")
		}

		// 4. Update or Append parts
		for _, part := range event.LLMResponse.Content.Parts {
			if part.FunctionResponse != nil {
				functionCallID := part.FunctionResponse.ID
				// If we've seen this response ID before, replace it
				if idx, found := partIndicesInMergedEvent[functionCallID]; found {
					partsInMergedEvent[idx] = part
				} else {
					// Otherwise, append it and update the index
					partsInMergedEvent = append(partsInMergedEvent, part)
					partIndicesInMergedEvent[functionCallID] = len(partsInMergedEvent) - 1
				}
			} else {
				// If it's not a function response, just append it
				partsInMergedEvent = append(partsInMergedEvent, part)
			}
		}
	}

	// Update the parts slice in the merged event in case it was reallocated
	mergedEvent.LLMResponse.Content.Parts = partsInMergedEvent

	return mergedEvent, nil
}

// buildContentsCurrentTurnContextOnly returns contents for the current turn only (no conversation history).
//
// When include_contents='none', we want to include:
//   - The current user input
//   - Tool calls and responses from the current turn
//
// But exclude conversation history from previous turns.
//
//	In multi-agent scenarios, the "current turn" for an agent starts from an
//	actual user or from another agent.
func buildContentsCurrentTurnContextOnly(agentName, branch string, events []*session.Event) ([]*genai.Content, error) {
	// Find the latest event that starts the current turn and process from there
	for i := len(events) - 1; i >= 0; i-- {
		event := events[i]
		if event.Author == "user" || isOtherAgentReply(agentName, event) {
			return buildContentsDefault(agentName, branch, events[i:])
		}
	}
	// NOTE: in Python, it returns [] if there is no event authored by a user or another agent,
	// but that may be a bug.
	return buildContentsDefault(agentName, branch, events)
}

func isOtherAgentReply(currentAgentName string, ev *session.Event) bool {
	return ev.Author != currentAgentName && ev.Author != "user"
}

// ConvertForeignEvent converts an event authored by another agent as
// a user-content event.
// This is to provide another aget's output as context to the current agent,
// so that the current agent can continue to respond, such as summarizing
// the previous agent's reply, etc.
func ConvertForeignEvent(ev *session.Event) *session.Event {
	content := utils.Content(ev)
	if content == nil || len(content.Parts) == 0 {
		return ev
	}

	converted := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "For context:"}},
	}
	for _, p := range content.Parts {
		switch {
		case p.Text != "":
			converted.Parts = append(converted.Parts, &genai.Part{
				Text: fmt.Sprintf("[%s] said: %s", ev.Author, p.Text),
			})
		case p.FunctionCall != nil:
			converted.Parts = append(converted.Parts, &genai.Part{
				Text: fmt.Sprintf("[%s] called tool %q with parameters: %s", ev.Author, p.FunctionCall.Name, stringify(p.FunctionCall.Args)),
			})
		case p.FunctionResponse != nil:
			converted.Parts = append(converted.Parts, &genai.Part{
				Text: fmt.Sprintf("[%s] %q tool returned result: %v", ev.Author, p.FunctionResponse.Name, stringify(p.FunctionResponse.Response)),
			})
		default: // fallback to the original part for non-text and non-functionCall parts.
			converted.Parts = append(converted.Parts, p)
		}
	}

	return &session.Event{ // made-up event. Don't go through types.NewEvent.
		Timestamp:   ev.Timestamp,
		Author:      "user",
		LLMResponse: model.LLMResponse{Content: converted},
		Branch:      ev.Branch,
	}
}

func stringify(v any) string {
	s, _ := json.Marshal(v)
	return string(s)
}

// requestEUCFunctionCallName is a special function to handle credential
// request.
const requestEUCFunctionCallName = "adk_request_credential"

func isAuthEvent(ev *session.Event) bool {
	c := utils.Content(ev)
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p.FunctionCall != nil && p.FunctionCall.Name == requestEUCFunctionCallName {
			return true
		}
		if p.FunctionResponse != nil && p.FunctionResponse.Name == requestEUCFunctionCallName {
			return true
		}
	}
	return false
}

func listFunctionCallsFromEvent(e *session.Event) []*genai.FunctionCall {
	funcCalls := make([]*genai.FunctionCall, 0)
	if e.LLMResponse.Content != nil && e.LLMResponse.Content.Parts != nil {
		for _, part := range e.LLMResponse.Content.Parts {
			if part.FunctionCall != nil {
				funcCalls = append(funcCalls, part.FunctionCall)
			}
		}
	}
	return funcCalls
}

func listFunctionResponsesFromEvent(e *session.Event) []*genai.FunctionResponse {
	funcResponses := make([]*genai.FunctionResponse, 0)
	if e.LLMResponse.Content != nil && e.LLMResponse.Content.Parts != nil {
		for _, part := range e.LLMResponse.Content.Parts {
			if part.FunctionResponse != nil {
				funcResponses = append(funcResponses, part.FunctionResponse)
			}
		}
	}
	return funcResponses
}

func cloneEvent(e *session.Event) *session.Event {
	if e == nil {
		return nil
	}

	// 1. Create a new Event instance
	newEvent := &session.Event{
		ID:           e.ID,
		Timestamp:    e.Timestamp,
		InvocationID: e.InvocationID,
		Branch:       e.Branch,
		Author:       e.Author,
		Actions:      e.Actions,
	}

	// 2. Deep copy the LongRunningToolIDs slice
	if e.LongRunningToolIDs != nil {
		newEvent.LongRunningToolIDs = make([]string, len(e.LongRunningToolIDs))
		copy(newEvent.LongRunningToolIDs, e.LongRunningToolIDs)
	}

	// TODO check if copy parts is needed
	// 3. Deep copy the LLMResponse pointer struct and content
	if e.LLMResponse.Content != nil {
		newEvent.LLMResponse.Content = &genai.Content{
			Parts: make([]*genai.Part, len(e.LLMResponse.Content.Parts)),
			Role:  e.LLMResponse.Content.Role,
		}
		copy(newEvent.LLMResponse.Content.Parts, e.LLMResponse.Content.Parts)
	}

	return newEvent
}
