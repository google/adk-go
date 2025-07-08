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

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

// contentRequestProcessor populates the LLMRequest's Contents based on
// the InvocationContext that includes the previous events.
func contentsRequestProcessor(ctx context.Context, parentCtx *adk.InvocationContext, req *adk.LLMRequest) error {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/contents.py) - extract function call results, etc.
	llmAgent := asLLMAgent(parentCtx.Agent)
	if llmAgent == nil {
		// Do nothing.
		return nil // In python, no error is yielded.
	}
	fn := buildContentsDefault // "" or "default".
	if llmAgent.IncludeContents == "none" {
		// Include current turn context only (no conversation history)
		fn = buildContentsCurrentTurnContextOnly
	}
	var events []*adk.Event
	if parentCtx.Session != nil {
		events = parentCtx.Session.Events
	}
	contents, err := fn(llmAgent.Name(), parentCtx.Branch, events)
	if err != nil {
		return err
	}
	req.Contents = append(req.Contents, contents...)
	return nil
}

// buildContentsDefault returns the contents for the LLM request by applying
// filtering, rearrangement, and content processing to the given events.
func buildContentsDefault(agentName, branch string, events []*adk.Event) ([]*genai.Content, error) {
	branchPrefix := branch + "."

	// parse the events, leaving the contents and the function calls and responses from the current agent.
	var filtered []*adk.Event
	for _, ev := range events {
		content := content(ev)
		// Skip events without content or generated neither by user nor
		// by model.
		// e.g. events purely for mutating session states.
		if content == nil || content.Role == "" || len(content.Parts) == 0 {
			// Note: python checks here if content.Parts[0] is an empty string and skip if so.
			// But unlike python that distinguishes None vs empty string, two cases are indistinguishable in Go.
			continue
		}
		// Skip events that do not belong to the current branch.
		// TODO: can we use a richier type for branch (e.g. []string) instead of using string prefix test?
		if branch != "" && (ev.Branch != branch && !strings.HasPrefix(ev.Branch, branchPrefix)) {
			continue
		}
		if isAuthEvent(ev) {
			continue
		}
		if isOtherAgentReply(agentName, ev) {
			filtered = append(filtered, convertForeignEvent(ev))
		} else {
			filtered = append(filtered, ev)
		}
	}

	// TODO: Rearrange filtered events for proper function call/response pairing.
	//  src/google/adk/flows/llm_flows/contents.py
	// 	 - _rearrange_events_for_async_function_response
	//   - _rearrange_events_for_async_function_responses_in_history

	var contents []*genai.Content
	for _, ev := range filtered {
		content := clone(content(ev))
		if content == nil {
			continue
		}
		removeClientFunctionCallID(content)
		contents = append(contents, content)
	}
	return contents, nil
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
func buildContentsCurrentTurnContextOnly(agentName, branch string, events []*adk.Event) ([]*genai.Content, error) {
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

func isOtherAgentReply(currentAgentName string, ev *adk.Event) bool {
	return ev.Author != currentAgentName && ev.Author != "user"
}

// convertForeignEvent converts an event authored by another agent as
// a user-content event.
// This is to provide another aget's output as context to the current agent,
// so that the current agent can continue to respond, such as summarizing
// previous agent's reply, etc.
func convertForeignEvent(ev *adk.Event) *adk.Event {
	content := content(ev)
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
				Text: fmt.Sprintf("[%s] said: %s", ev.Author, p.Text)})
		case p.FunctionCall != nil:
			converted.Parts = append(converted.Parts, &genai.Part{
				Text: fmt.Sprintf("[%s] called tool %q with parameters: %s", ev.Author, p.FunctionCall.Name, stringify(p.FunctionCall.Args))})
		case p.FunctionResponse != nil:
			converted.Parts = append(converted.Parts, &genai.Part{
				Text: fmt.Sprintf("[%s] %q tool returned result: %v", ev.Author, p.FunctionResponse.Name, stringify(p.FunctionResponse.Response))})
		default: // fallback to the original part for non-text and non-functionCall parts.
			converted.Parts = append(converted.Parts, p)
		}
	}

	return &adk.Event{ // made-up event. Don't go through adk.NewEvent.
		Time:        ev.Time,
		Author:      "user",
		LLMResponse: &adk.LLMResponse{Content: converted},
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

func isAuthEvent(ev *adk.Event) bool {
	c := content(ev)
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
