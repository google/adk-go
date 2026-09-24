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

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// InstrumentInvocation wraps run, one runner invocation, in its entrypoint
// invoke_workflow span, named for root, matching adk-python's
// record_invocation. The span records the first error run yields.
//
// run gets the context carrying the span. A workflow root opens that span
// itself, and the legacy schema has none, so for both run gets ctx unchanged.
func InstrumentInvocation(ctx context.Context, root AgentLike, sessionID string, run func(context.Context) iter.Seq2[*session.Event, error]) iter.Seq2[*session.Event, error] {
	if useLegacySchema() || isWorkflowAgent(root) {
		return run(ctx)
	}
	return func(yield func(*session.Event, error) bool) {
		ctx, span := startInvokeWorkflowSpan(ctx, root.Name(), sessionID)
		var firstErr error
		defer func() {
			recordErrorAndStatus(span, firstErr)
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

// InstrumentGenerateContent wraps call, one model call, in its telemetry: the
// generate_content span, and the log records describing the call. It is the
// only way to trace a model call, so the span and the log records always
// describe the same call.
//
// call gets the context carrying the span. response reads the model response,
// and the id of the event it becomes, out of each value call yields. The span
// ends at the first error or final response, before that value is passed on,
// so it does not time the caller's handling of it; otherwise it ends when
// iteration stops.
func InstrumentGenerateContent[T any](ctx context.Context, params GenerateContentParams, response func(T) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[T, error]) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		// The event is emitted after the call, and a model may append to
		// Contents while handling it: keep the request as the model got it.
		params := params
		req := *params.Request
		params.Request = &req
		ctx, span := startGenerateContentSpan(ctx, params)
		logRequest(ctx, params.Request, params.Backend)

		var result generateContentResult
		ended := false
		end := func() {
			if ended {
				return
			}
			ended = true
			traceGenerateContentResult(span, result)
			logInferenceOperationDetails(ctx, params, result)
			span.End()
		}
		defer end()
		for v, err := range call(ctx) {
			resp, eventID := response(v)
			result = generateContentResult{Response: resp, EventID: eventID, Error: err}
			if err != nil {
				end()
			} else if !resp.Partial {
				logResponse(ctx, resp, params.Backend)
				end()
			}
			if !yield(v, err) {
				return
			}
		}
	}
}
