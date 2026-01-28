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
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
)

type confirmedCall struct {
	confirmation *toolconfirmation.ToolConfirmation
	call         genai.FunctionCall
}

func RequestConfirmationRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return // In python, no error is yielded.
		}

		toolsmap := make(map[string]tool.Tool)
		for _, tool := range f.Tools {
			toolsmap[tool.Name()] = tool
		}

		var events []*session.Event
		if ctx.Session() != nil {
			for e := range ctx.Session().Events().All() {
				events = append(events, e)
			}
		}
		confirmationResponses := make(map[string]toolconfirmation.ToolConfirmation)
		confirmationEventIndex := -1
		for k := len(events) - 1; k >= 0; k-- {
			event := events[k]
			// Find the first event authored by user
			if event.Author != "user" {
				continue
			}
			responses := utils.FunctionResponses(event.Content)
			if len(responses) == 0 {
				return
			}
			for _, funcResp := range responses {
				if funcResp.Name != toolconfirmation.FunctionCallName {
					continue
				}
				var tc toolconfirmation.ToolConfirmation
				if funcResp.Response != nil {
					resp, hasResponseKey := funcResp.Response["response"]
					// ADK web client will send a request that is always encapsulated in a 'response' key.
					if hasResponseKey && len(funcResp.Response) == 1 {
						if jsonString, ok := resp.(string); ok {
							err := json.Unmarshal([]byte(jsonString), &tc)
							if err != nil {
								yield(nil, fmt.Errorf("error 'response' key found but failed unmarshalling confirmation function response for event id %q: %w", event.ID, err))
								return
							}
						} else {
							yield(nil, fmt.Errorf("error 'response' key found but value is not a string for confirmation function response for event id %q", event.ID))
							return
						}
					} else {
						tempJSON, err := json.Marshal(funcResp.Response)
						if err != nil {
							yield(nil, fmt.Errorf("error failed marshalling confirmation function response for event id %q: %w", event.ID, err))
							return
						}
						err = json.Unmarshal(tempJSON, &tc)
						if err != nil {
							yield(nil, fmt.Errorf("error failed unmarshalling confirmation function response for event id %q: %w", event.ID, err))
							return
						}
					}
				}
				confirmationResponses[funcResp.ID] = tc
			}
			confirmationEventIndex = k
			break
		}

		if len(confirmationResponses) == 0 {
			return
		}

		// TODO could we skip events for >= confirmationEventIndex
		for k := len(events) - 2; k >= 0; k-- {
			event := events[k]
			// Find the system generated FunctionCall event requesting the tool confirmation
			calls := utils.FunctionCalls(event.Content)
			if len(calls) == 0 {
				continue
			}
			toolsToResumeByFunctionCallID := map[string]*confirmedCall{}
			for _, functionCall := range calls {
				confirmation, ok := confirmationResponses[functionCall.ID]
				if !ok {
					continue
				}
				originalFunctionCall, err := toolconfirmation.OriginalCallFrom(functionCall)
				if err != nil {
					continue
				}

				toolsToResumeByFunctionCallID[originalFunctionCall.ID] = &confirmedCall{
					confirmation: &confirmation,
					call:         *originalFunctionCall,
				}
			}

			if len(toolsToResumeByFunctionCallID) == 0 {
				continue
			}

			// TODO consider forward or backward pass instead of nested loops
			// Remove the tools that have already been confirmed.
			for j := len(events) - 1; j > confirmationEventIndex; j-- {
				event = events[j]
				responses := utils.FunctionResponses(event.Content)
				if len(responses) == 0 {
					continue
				}
				for _, resp := range responses {
					delete(toolsToResumeByFunctionCallID, resp.ID)
				}
				if len(toolsToResumeByFunctionCallID) == 0 {
					break
				}
			}
			if len(toolsToResumeByFunctionCallID) == 0 {
				continue
			}

			parts := make([]*genai.Part, 0)
			toolsToResumeConfirmation := make(map[string]*toolconfirmation.ToolConfirmation, len(toolsToResumeByFunctionCallID))
			for callID, cc := range toolsToResumeByFunctionCallID {
				parts = append(parts, &genai.Part{FunctionCall: &cc.call})
				toolsToResumeConfirmation[callID] = cc.confirmation
			}

			ev, err := f.handleFunctionCalls(ctx, toolsmap, &model.LLMResponse{
				Content: &genai.Content{Parts: parts, Role: genai.RoleUser},
			}, toolsToResumeConfirmation)
			if !yield(ev, err) {
				return
			}
		}
	}
}
