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
	"errors"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

func RequestConfirmationRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return // In python, no error is yielded.
		}

		var events []*session.Event
		if ctx.Session() != nil {
			for e := range ctx.Session().Events().All() {
				events = append(events, e)
			}
		}
		requestConfirmationFR := make(map[string]toolconfirmation.ToolConfirmation)
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
				if funcResp.Name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME {
					continue
				}
				var tc toolconfirmation.ToolConfirmation
				if funcResp.Response != nil {
					resp, hasResponseKey := funcResp.Response["response"]
					// ADK web client will send a request that is always encapsulated in a  'response' key.
					if hasResponseKey && len(funcResp.Response) == 1 {
						if jsonString, ok := resp.(string); ok {
							err := json.Unmarshal([]byte(jsonString), &tc)
							if err != nil {
								yield(nil, fmt.Errorf("error 'response' key found but failed unmarshalling confirmation function response %w", err))
								return
							}
						} else {
							yield(nil, errors.New("error 'response' key found but value is not a string for confirmation function response"))
							return
						}
					} else {
						tempJSON, _ := json.Marshal(funcResp.Response)
						err := json.Unmarshal(tempJSON, &tc)
						if err != nil {
							yield(nil, fmt.Errorf("error failed unmarshalling confirmation function response %w", err))
							return
						}
					}
				}
				requestConfirmationFR[funcResp.ID] = tc
			}
			confirmationEventIndex = k
			break
		}

		if len(requestConfirmationFR) == 0 {
			return
		}

		for k := len(events) - 2; k >= 0; k-- {
			event := events[k]
			// Find the system generated FunctionCall event requesting the tool confirmation
			calls := utils.FunctionCalls(event.Content)
			if len(calls) == 0 {
				return
			}
			toolsToResumeConfirmation := map[string]toolconfirmation.ToolConfirmation{}
			toolsToResumeWithArgs := map[string]genai.FunctionCall{}
			for _, functionCall := range calls {
				confirmation, ok := requestConfirmationFR[functionCall.ID]
				if !ok {
					continue
				}

				originalCallRaw, ok := functionCall.Args["originalFunctionCall"]
				if !ok {
					continue
				}

				// Use JSON round-tripping to simulate Python's **kwargs unpacking
				var originalFunctionCall genai.FunctionCall
				jsonBytes, err := json.Marshal(originalCallRaw)
				if err != nil {
					continue
				}

				if err := json.Unmarshal(jsonBytes, &originalFunctionCall); err != nil {
					continue
				}

				toolsToResumeConfirmation[originalFunctionCall.ID] = confirmation
				toolsToResumeWithArgs[originalFunctionCall.ID] = originalFunctionCall
			}

			if len(toolsToResumeConfirmation) == 0 {
				continue
			}

			// Remove the tools that have already been confirmed.
			for j := len(events) - 1; j >= confirmationEventIndex; j-- {
				event = events[j]
				responses := utils.FunctionResponses(event.Content)
				if len(responses) == 0 {
					continue
				}
				for _, resp := range responses {
					if _, ok := toolsToResumeConfirmation[resp.ID]; ok {
						delete(toolsToResumeConfirmation, resp.ID)
						delete(toolsToResumeWithArgs, resp.ID)
					}
				}
				if len(toolsToResumeConfirmation) == 0 {
					break
				}
			}
			if len(toolsToResumeConfirmation) == 0 {
				continue
			}
		}
	}
}
