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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type confirmedCall struct {
	confirmation   *toolconfirmation.ToolConfirmation
	confirmationID string
	call           genai.FunctionCall
}

type invocationSessionService interface {
	SessionService() session.Service
}

type confirmationClaimLock struct {
	mu   sync.Mutex
	refs int
}

var confirmationClaimLocks = struct {
	sync.Mutex
	values map[string]*confirmationClaimLock
}{values: make(map[string]*confirmationClaimLock)}

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

		agentName := ctx.Agent().Name()

		// Detect tampered confirmations before resuming any tool.
		//
		// The resume path below trusts the tool name and arguments embedded in
		// the agent-authored confirmation-request event (via OriginalCallFrom),
		// gated only by the author check. The user's approval carries only the
		// confirmation-call ID and a {"confirmed": true} decision; it does not
		// restate or bind the arguments. So if two events present the same
		// confirmation-call ID but embed different original calls, a single user
		// approval is ambiguous: we cannot tell which call the user actually saw
		// and approved. An attacker who can append an event stamped with this
		// agent's name (for example by tampering with the session store) can
		// exploit this to swap in different arguments, or a different tool,
		// behind a legitimate approval. When a confirmation-call ID resolves to
		// conflicting original calls across events, refuse to resume it (fail
		// closed): the legitimate action is blocked rather than the attacker's
		// action executed.
		conflictingConfirmations := make(map[string]bool)
		originalCallByConfirmationID := make(map[string]string)
		for _, event := range events {
			if event.Author != agentName {
				continue
			}
			for _, functionCall := range utils.FunctionCalls(event.Content) {
				if _, ok := confirmationResponses[functionCall.ID]; !ok {
					continue
				}
				originalFunctionCall, err := toolconfirmation.OriginalCallFrom(functionCall)
				if err != nil {
					continue
				}
				canonical, err := json.Marshal(originalFunctionCall)
				if err != nil {
					continue
				}
				if prev, seen := originalCallByConfirmationID[functionCall.ID]; seen {
					if prev != string(canonical) {
						conflictingConfirmations[functionCall.ID] = true
					}
					continue
				}
				originalCallByConfirmationID[functionCall.ID] = string(canonical)
			}
		}

		// TODO could we skip events for >= confirmationEventIndex
		for k := len(events) - 2; k >= 0; k-- {
			event := events[k]
			// Find the system generated FunctionCall event requesting the tool confirmation.
			//
			// Only this agent can ask this agent's user for confirmation. Function call
			// parts also reach the session from other places, notably an A2A peer
			// response converted into a model-role event: honouring a confirmation
			// request from such an event would let the peer choose which local tool
			// runs, and with what arguments, bypassing the human-in-the-loop gate
			// entirely. Skip any event not authored by this agent.
			if event.Author != agentName {
				continue
			}
			calls := utils.FunctionCalls(event.Content)
			if len(calls) == 0 {
				continue
			}
			toolsToResumeByFunctionCallID := map[string]*confirmedCall{}
			// Record the order the confirmation requests appear in the event so we
			// can re-dispatch the resumed calls in that same order below. Ranging
			// over the map directly would use Go's randomized map iteration order,
			// which makes the re-dispatched function calls, the assembled response,
			// and the resulting StateDelta last-writer-wins merge all
			// non-deterministic across runs.
			var resumeOrder []string
			for _, functionCall := range calls {
				confirmation, ok := confirmationResponses[functionCall.ID]
				if !ok {
					continue
				}
				if conflictingConfirmations[functionCall.ID] {
					// Ambiguous/tampered confirmation (see above): this
					// confirmation-call ID resolves to conflicting original calls
					// across events, so we cannot trust which one the user
					// approved. Refuse to resume it.
					continue
				}
				originalFunctionCall, err := toolconfirmation.OriginalCallFrom(functionCall)
				if err != nil {
					continue
				}

				// Record each original call ID in resumeOrder only once: two
				// confirmation requests in the same event can resolve to the same
				// originalFunctionCall.ID, and the resumed tool must be dispatched
				// once, not once per confirmation.
				_, seen := toolsToResumeByFunctionCallID[originalFunctionCall.ID]
				if !seen {
					resumeOrder = append(resumeOrder, originalFunctionCall.ID)
				}
				toolsToResumeByFunctionCallID[originalFunctionCall.ID] = &confirmedCall{
					confirmation:   &confirmation,
					confirmationID: functionCall.ID,
					call:           *originalFunctionCall,
				}
			}

			if len(toolsToResumeByFunctionCallID) == 0 {
				continue
			}

			removeAlreadyCompletedTools(events, k, confirmationEventIndex, agentName, toolsToResumeByFunctionCallID)
			if len(toolsToResumeByFunctionCallID) == 0 {
				continue
			}

			removeAlreadyCompletedTools(events, confirmationEventIndex, len(events), agentName, toolsToResumeByFunctionCallID)
			if len(toolsToResumeByFunctionCallID) == 0 {
				continue
			}

			// Re-dispatch in request order (resumeOrder), so the parts slice and the
			// downstream response/StateDelta merge are deterministic.
			parts := make([]*genai.Part, 0, len(toolsToResumeByFunctionCallID))
			toolsToResumeConfirmation := make(map[string]*toolconfirmation.ToolConfirmation, len(toolsToResumeByFunctionCallID))
			for _, callID := range resumeOrder {
				cc, ok := toolsToResumeByFunctionCallID[callID]
				if !ok {
					// Skip calls whose response is already present in this event
					// snapshot (removed by the delete pass above); re-dispatching such
					// a call would run its tool a second time within this resume.
					continue
				}
				if cc.confirmation.Confirmed {
					claimed, claimEvent, err := claimConfirmation(ctx, cc, agentName)
					if err != nil {
						if !yield(nil, err) {
							return
						}
						return
					}
					if !claimed {
						delete(toolsToResumeByFunctionCallID, callID)
						continue
					}
					if claimEvent != nil {
						events = append(events, claimEvent)
					}
				}
				parts = append(parts, &genai.Part{FunctionCall: &cc.call})
				toolsToResumeConfirmation[callID] = cc.confirmation
			}
			if len(parts) == 0 {
				continue
			}

			ev, err := f.handleFunctionCalls(ctx, toolsmap, &model.LLMResponse{
				Content: &genai.Content{Parts: parts, Role: genai.RoleUser},
			}, toolsToResumeConfirmation, nil)
			if !yield(ev, err) {
				return
			}
		}
	}
}

func claimConfirmation(ctx agent.InvocationContext, call *confirmedCall, agentName string) (bool, *session.Event, error) {
	service := session.Service(nil)
	if provider, ok := ctx.(invocationSessionService); ok {
		service = provider.SessionService()
	}
	// Direct processor users may provide an invocation context without a
	// runner-owned service. Preserve that supported testing/embedding path;
	// runner-created contexts always carry the service.
	if service == nil || ctx.Session() == nil {
		return true, nil, nil
	}

	claim := session.NewEvent(ctx, ctx.InvocationID())
	claim.ID = confirmationClaimEventID(ctx.Session(), agentName, call.call.ID)
	claim.Author = agentName
	claim.Actions.ConfirmationClaim = &session.ConfirmationClaim{
		ConfirmationID: call.confirmationID,
		FunctionCallID: call.call.ID,
	}

	if appender, ok := service.(session.EventAppender); ok {
		inserted, err := appender.AppendEventIfAbsent(ctx, ctx.Session(), claim)
		return inserted, claim, err
	}

	// A custom service may not expose a cross-process atomic primitive. The
	// keyed lock prevents duplicate claims among runners in this process; the
	// latest read closes the stale-session window before the append.
	release := lockConfirmationClaim(ctx.Session(), agentName, call.call.ID)
	defer release()
	latest, err := service.Get(ctx, &session.GetRequest{
		AppName:   ctx.Session().AppName(),
		UserID:    ctx.Session().UserID(),
		SessionID: ctx.Session().ID(),
	})
	if err != nil {
		return false, nil, fmt.Errorf("failed to verify confirmation claim: %w", err)
	}
	if hasConfirmationClaim(latest.Session, agentName, call.call.ID) {
		return false, nil, nil
	}
	if err := service.AppendEvent(ctx, ctx.Session(), claim); err != nil {
		return false, nil, fmt.Errorf("failed to persist confirmation claim: %w", err)
	}
	return true, claim, nil
}

func confirmationClaimEventID(sess session.Session, agentName, functionCallID string) string {
	sum := sha256.Sum256([]byte(sess.AppName() + "\x00" + sess.UserID() + "\x00" + sess.ID() + "\x00" + agentName + "\x00" + functionCallID))
	return fmt.Sprintf("adk-confirmation-claim-%x", sum[:])
}

func hasConfirmationClaim(sess session.Session, agentName, functionCallID string) bool {
	if sess == nil {
		return false
	}
	for event := range sess.Events().All() {
		if event.Author != agentName || event.Actions.ConfirmationClaim == nil {
			continue
		}
		if event.Actions.ConfirmationClaim.FunctionCallID == functionCallID {
			return true
		}
	}
	return false
}

func lockConfirmationClaim(sess session.Session, agentName, functionCallID string) func() {
	key := sess.AppName() + "\x00" + sess.UserID() + "\x00" + sess.ID() + "\x00" + agentName + "\x00" + functionCallID
	confirmationClaimLocks.Lock()
	lock := confirmationClaimLocks.values[key]
	if lock == nil {
		lock = &confirmationClaimLock{}
		confirmationClaimLocks.values[key] = lock
	}
	lock.refs++
	confirmationClaimLocks.Unlock()

	lock.mu.Lock()
	return func() {
		lock.mu.Unlock()
		confirmationClaimLocks.Lock()
		lock.refs--
		if lock.refs == 0 {
			delete(confirmationClaimLocks.values, key)
		}
		confirmationClaimLocks.Unlock()
	}
}

// removeAlreadyCompletedTools removes only agent-authored tool responses from
// the bounded event window. The lower bound is the confirmation request: the
// earlier request-time error must not suppress the first approval, while later
// completed responses must suppress replays. A rejection is not a completion;
// it leaves the confirmation eligible for a later answer.
func removeAlreadyCompletedTools(events []*session.Event, startIndex, endIndex int, agentName string, toolsToResume map[string]*confirmedCall) {
	for j := startIndex + 1; j < endIndex; j++ {
		event := events[j]
		if event.Author != agentName {
			continue
		}
		if claim := event.Actions.ConfirmationClaim; claim != nil {
			delete(toolsToResume, claim.FunctionCallID)
			if len(toolsToResume) == 0 {
				return
			}
		}
		responses := utils.FunctionResponses(event.Content)
		if len(responses) == 0 {
			continue
		}
		for _, resp := range responses {
			if isConfirmationRejection(resp) {
				continue
			}
			delete(toolsToResume, resp.ID)
		}
		if len(toolsToResume) == 0 {
			return
		}
	}
}

func isConfirmationRejection(resp *genai.FunctionResponse) bool {
	if resp == nil || resp.Response == nil {
		return false
	}
	errText, ok := resp.Response["error"].(string)
	return ok && strings.HasSuffix(errText, tool.ErrConfirmationRejected.Error())
}
