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

package telemetry

import (
	"context"
	"iter"

	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// StartInvocationSpan starts the runner's span before its plugin callbacks.
// The caller must end it after callbacks and post-invocation work finish.
// The legacy schema uses invocation; the default schema uses invoke_workflow.
// Nested invocations in the default schema reuse their enclosing workflow
// context and return a no-op span, so ending it cannot end the enclosing span.
func StartInvocationSpan(ctx context.Context, root AgentLike, sessionID string) (context.Context, trace.Span) {
	if useLegacySchema() {
		return StartTrace(ctx, "invocation")
	}
	// Unlike ADK Python, nested invocations add no span, preserving the
	// existing Go convention of omitting implementation details.
	if ctx.Value(workflowScopeKey{}) != nil {
		// A runner called by a callback may have the same root name, but it
		// does not own the enclosing runner's workflow span.
		return context.WithValue(ctx, invocationWorkflowKey{}, nil), noop.Span{}
	}
	ctx, span := startInvokeWorkflowSpan(ctx, root.Name(), sessionID)
	if isWorkflowAgent(root) {
		ctx = context.WithValue(ctx, invocationWorkflowKey{}, root.Name())
	}
	return ctx, span
}

// InstrumentInvocation keeps the runner's span open through event iteration,
// including callbacks and deferred compaction.
func InstrumentInvocation(ctx context.Context, root AgentLike, sessionID string, run func(context.Context) iter.Seq2[*session.Event, error]) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		ctx, span := StartInvocationSpan(ctx, root, sessionID)
		var firstErr error
		defer func() {
			RecordError(span, firstErr)
			span.End()
		}()
		for ev, err := range run(ctx) {
			if err != nil && firstErr == nil {
				firstErr = err
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

// InstrumentGenerateContent wraps a model call in its span and log records.
// The span ends before the response that ends the call is yielded: an error,
// a response reporting a finish reason, or the one response of a call that is
// not streamed. The legacy schema ends it at the first response not marked
// partial instead.
func InstrumentGenerateContent[T any](ctx context.Context, params GenerateContentParams, response func(T) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[T, error]) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		ctx, span := startGenerateContentSpan(ctx, params)
		logRequest(ctx, params.Request, params.Backend)
		details := operationDetails{span: span}
		details.noteLLMRequest(params.Request)

		var result generateContentResult
		ended := false
		end := func() {
			if ended {
				return
			}
			ended = true
			traceGenerateContentResult(span, result)
			details.noteLLMRequest(params.Request)
			details.emit(ctx, params, result)
			span.End()
		}
		defer end()
		legacy := useLegacySchema()
		for v, err := range call(ctx) {
			resp, eventID := response(v)
			switch {
			case legacy:
				result = generateContentResult{Response: resp, EventID: eventID, Error: err}
				if err != nil {
					end()
				} else if !resp.Partial {
					logResponse(ctx, resp, params.Backend)
					end()
				}
			case ended || (resp == nil && err == nil):
				// After the end, what an aggregator assembles from the chunks
				// already recorded.
			default:
				result.Error = err
				if resp != nil {
					result.Response, result.EventID = resp, eventID
					result.Responses = append(result.Responses, resp)
				}
				// Partial is set by ADK's aggregator rather than the model, and a
				// model.LLM may leave it unset; the finish reason comes from the
				// model, as in adk-python's _instrumentation.py.
				if err != nil || reportedFinishReason(resp, nil) != "" || !params.Stream {
					end()
				}
			}
			if !yield(v, err) {
				return
			}
		}
	}
}
