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

package remoteagent

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/log"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

type userFunctionCall struct {
	response  *session.Event
	taskID    a2a.TaskID
	contextID string
}

// toUserFunctionCall returns a non-nil struct when the last event in the session has a FunctionResponse
// with user-provided data. The struct contains both call and response events.
func getUserFunctionCallAt(events session.Events, index int) *userFunctionCall {
	if index < 0 || index >= events.Len() {
		return nil
	}
	candidate := events.At(index)
	if candidate.Author != "user" {
		return nil
	}
	fnCallID, ok := getFunctionResponseCallID(candidate)
	if !ok {
		return nil
	}
	for i := index - 1; i >= 0; i-- {
		request := events.At(i)
		if !isFunctionCallEvent(request, fnCallID) {
			continue
		}
		result := &userFunctionCall{response: candidate}
		tid, ctxID := adka2a.GetA2ATaskInfo(request)
		result.taskID = tid
		result.contextID = ctxID
		return result
	}
	return nil
}

func isFunctionCallEvent(event *session.Event, callID string) bool {
	if event == nil || event.Content == nil {
		return false
	}
	return slices.ContainsFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part.FunctionCall != nil && part.FunctionCall.ID == callID
	})
}

// getFunctionResponseCallID finds the first part with non-nil FunctionResponse and returns the call ID.
func getFunctionResponseCallID(event *session.Event) (string, bool) {
	if event.Content == nil {
		return "", false
	}
	responsePartIndex := slices.IndexFunc(event.Content.Parts, func(part *genai.Part) bool {
		return part.FunctionResponse != nil
	})
	if responsePartIndex < 0 {
		return "", false
	}
	return event.Content.Parts[responsePartIndex].FunctionResponse.ID, true
}

// toMissingRemoteSessionParts returns content parts for all events we think are not present in the remote session
// and a2a contextID if it was found in a remote agent event metadata.
// We iterate session events backward until all events are processed or an event authored by a remote agent is found.
// Parts from all events we processed are returned as a single list.
// The returned contextID might be an empty string. This means the current remote agent invocation is not associates with
// any of the previous one. In this case a new contextID will be generated on the remote server.
func toMissingRemoteSessionParts(ctx agent.InvocationContext, events session.Events, cfg A2AConfig) ([]*a2a.Part, string) {
	partCount, contextID := 0, ""
	// only events after this index are not in the remote session
	lastRemoteResponseIndex := -1
	for i := events.Len() - 1; i >= 0; i-- {
		event := events.At(i)
		if event.Author == ctx.Agent().Name() {
			lastRemoteResponseIndex = i
			_, contextID = adka2a.GetA2ATaskInfo(event)
			break
		}
		if event.LLMResponse.Content != nil {
			partCount += len(event.Content.Parts)
		}
	}

	result := make([]*a2a.Part, 0, partCount)
	for i := lastRemoteResponseIndex + 1; i < events.Len(); i++ {
		event := events.At(i)
		// Only wrap foreign agent events as user messages when the current agent has an explicit name.
		// If Agent().Name() is empty (e.g., in anonymous wrappers or conformance harnesses), event.Author != ""
		// would falsely match and attribute events as foreign turns.
		if ctx.Agent().Name() != "" && event.Author != "user" && event.Author != ctx.Agent().Name() {
			event = presentAsUserMessage(ctx, event)
		}
		if event.Content == nil || len(event.Content.Parts) == 0 {
			continue
		}
		parts, err := convertParts(ctx, cfg, event)
		if err != nil {
			log.Warn(ctx, "failed to convert parts for session event", "index", i, "error", err)
			continue
		}
		result = append(result, parts...)
	}
	return result, contextID
}

func presentAsUserMessage(ctx agent.InvocationContext, agentEvent *session.Event) *session.Event {
	event := session.NewEvent(ctx, ctx.InvocationID())
	event.Author = "user"

	if agentEvent.Content == nil {
		return event
	}

	payloadParts := make([]*genai.Part, 0, len(agentEvent.Content.Parts))
	for _, part := range agentEvent.Content.Parts {
		if part.Thought {
			continue
		}
		if part.Text != "" {
			text := fmt.Sprintf("[%s] said:\n%s", agentEvent.Author, llminternal.QuoteUntrusted(part.Text))
			payloadParts = append(payloadParts, genai.NewPartFromText(text))
		} else if part.FunctionCall != nil {
			call := part.FunctionCall
			// The tool name is elided but left unfenced -- eliding markers
			// is a complete answer to marker forgery, not to a name that
			// carries a paragraph of unfenced text. Longer version of this
			// comment, and the same tradeoff, in contents_processor.go's
			// ConvertForeignEvent, which this function mirrors for the A2A
			// path.
			text := fmt.Sprintf("[%s] called tool `%s` with parameters:\n%s",
				agentEvent.Author, llminternal.ElideQuoteMarkers(call.Name), llminternal.QuoteUntrusted(fmt.Sprintf("%v", call.Args)))
			payloadParts = append(payloadParts, genai.NewPartFromText(text))
		} else if part.FunctionResponse != nil {
			resp := part.FunctionResponse
			text := fmt.Sprintf("[%s] `%s` tool returned result:\n%s",
				agentEvent.Author, llminternal.ElideQuoteMarkers(resp.Name), llminternal.QuoteUntrusted(fmt.Sprintf("%v", resp.Response)))
			payloadParts = append(payloadParts, genai.NewPartFromText(text))
		} else {
			// FileData, InlineData, ExecutableCode, and CodeExecutionResult
			// parts all land here and are relayed verbatim, with no fence
			// and no elision -- same tradeoff as ConvertForeignEvent's
			// default case, for the same reason, including the same
			// caveat that this is not neutral ground: see that function's
			// comment on its own default case for the full reasoning,
			// which applies here unchanged.
			//
			// A part that is itself the zero value (e.g. an empty
			// streaming-tail part with Thought false) is skipped rather
			// than relayed, for the same reason ConvertForeignEvent skips
			// one: appending it here would make len(payloadParts) > 0
			// look satisfied by a part carrying nothing at all, stranding
			// the preamble below in front of an empty payload once such a
			// part is later dropped by any downstream emptiness filter.
			if part == nil || reflect.ValueOf(*part).IsZero() {
				continue
			}
			payloadParts = append(payloadParts, part)
		}
	}
	if len(payloadParts) > 0 {
		parts := make([]*genai.Part, 0, len(payloadParts)+1)
		parts = append(parts, &genai.Part{Text: llminternal.OtherAgentContextPreamble})
		parts = append(parts, payloadParts...)
		event.Content = genai.NewContentFromParts(parts, genai.RoleUser)
	}
	return event
}

func convertParts(ctx agent.InvocationContext, cfg A2AConfig, event *session.Event) ([]*a2a.Part, error) {
	parts := make([]*a2a.Part, 0, len(event.Content.Parts))
	if cfg.GenAIPartConverter != nil {
		for _, part := range event.Content.Parts {
			cp, err := cfg.GenAIPartConverter(ctx, event, part)
			if err != nil {
				return nil, err
			}
			if cp != nil {
				parts = append(parts, cp)
			}
		}
	} else {
		var err error
		parts, err = adka2a.ToA2AParts(event.Content.Parts, event.LongRunningToolIDs)
		if err != nil {
			return nil, fmt.Errorf("event part conversion failed: %w", err)
		}
	}
	return parts, nil
}
