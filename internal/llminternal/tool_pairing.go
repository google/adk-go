// Copyright 2026 Google LLC
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
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
)

// missingFunctionResult stands in for a result that was never recorded. It
// names no cause, because there is none to name: the turn may have been
// interrupted, or the process may have died between the two events.
const missingFunctionResult = "No response available for this function call."

// pendingFunctionResult stands in for a call ADK is deliberately holding open:
// a long-running tool, a human approval, or a request for user input. No
// function response exists for these until the answer arrives, so the call is
// pending rather than lost. The distinction changes what the model does next:
// told a tool returned nothing, it reissues the call or proceeds without it;
// told the call is still awaiting a response, it can wait.
const pendingFunctionResult = "This call is awaiting a response and has not completed yet."

// pendingCallIDs returns the ids of the calls ADK is holding open.
// LongRunningToolIDs lives on the event and does not survive the conversion to
// contents, so it is read here.
func pendingCallIDs(events []*session.Event) map[string]bool {
	pending := make(map[string]bool)
	for _, event := range events {
		if event == nil {
			continue
		}
		for _, id := range event.LongRunningToolIDs {
			pending[id] = true
		}
	}
	return pending
}

// pairUnansweredFunctionCalls gives every function call a function response in
// the immediately following content.
//
// A call and its result are two separate session events. When a turn ends
// between them (a restart, an OOM kill, a disconnect, a cancellation), the
// session keeps a function call that no function response answers. That
// history is replayed on every later turn, and a provider that requires strict
// pairing then rejects the whole conversation, so the session stays unusable
// until it is deleted. The OpenAI Responses API rejects a function call item
// that no function call output answers, and Anthropic answers "tool_use ids
// were found without tool_result blocks immediately after".
//
// The repair runs on the request rather than the stored events, so recorded
// history stays intact and a session that is already broken heals on its next
// turn without a migration. It also sits above the session service, so it
// applies to every store.
//
// Pairing is checked positionally, against the immediately following content,
// because that is the invariant the provider enforces. A conversation whose
// calls are all answered is returned unchanged, and any conversation this does
// change is one the provider would have rejected.
func pairUnansweredFunctionCalls(contents []*genai.Content, pending map[string]bool) []*genai.Content {
	paired := make([]*genai.Content, 0, len(contents))
	for index, content := range contents {
		paired = append(paired, content)

		calls := utils.FunctionCalls(content)
		if len(calls) == 0 {
			continue
		}

		var following *genai.Content
		if index+1 < len(contents) {
			following = contents[index+1]
		}
		answered := responseIDs(following)
		unanswered := unansweredCalls(calls, answered)
		if len(unanswered) == 0 {
			continue
		}

		parts := make([]*genai.Part, 0, len(unanswered))
		for _, call := range unanswered {
			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       call.ID,
					Name:     call.Name,
					Response: map[string]any{"result": placeholderFor(call, pending)},
				},
			})
		}

		// Join a turn that already answers this call event, so the results stay
		// in one message; otherwise the results need a turn of their own,
		// before whatever currently follows.
		if len(answered) > 0 {
			following.Parts = withResultsInserted(following.Parts, parts)
			continue
		}
		paired = append(paired, &genai.Content{Role: genai.RoleUser, Parts: parts})
	}
	return paired
}

func placeholderFor(call *genai.FunctionCall, pending map[string]bool) string {
	if call.ID != "" && pending[call.ID] {
		return pendingFunctionResult
	}
	return missingFunctionResult
}

// unansweredCalls returns the calls that answered does not account for. Ids are
// consumed one at a time rather than matched through a set, so a turn that
// carries several calls without an id (Gemini omits them, and ADK strips its
// own client ids before the request is built) does not have a single response
// silently answer all of them.
func unansweredCalls(calls []*genai.FunctionCall, answered []string) []*genai.FunctionCall {
	remaining := make([]string, len(answered))
	copy(remaining, answered)

	var unanswered []*genai.FunctionCall
	for _, call := range calls {
		matched := false
		for i, id := range remaining {
			if id == call.ID {
				remaining = append(remaining[:i], remaining[i+1:]...)
				matched = true
				break
			}
		}
		if !matched {
			unanswered = append(unanswered, call)
		}
	}
	return unanswered
}

func responseIDs(content *genai.Content) []string {
	var ids []string
	for _, response := range utils.FunctionResponses(content) {
		ids = append(ids, response.ID)
	}
	return ids
}

// withResultsInserted returns parts with results added to its leading run of
// responses. Anthropic requires every tool result to precede any other block in
// the message that carries it, so a placeholder appended after a trailing text
// part would be rejected for the same reason the missing result was.
func withResultsInserted(parts, results []*genai.Part) []*genai.Part {
	insertIndex := len(parts)
	for i, part := range parts {
		if part == nil || part.FunctionResponse == nil {
			insertIndex = i
			break
		}
	}
	joined := make([]*genai.Part, 0, len(parts)+len(results))
	joined = append(joined, parts[:insertIndex]...)
	joined = append(joined, results...)
	return append(joined, parts[insertIndex:]...)
}
