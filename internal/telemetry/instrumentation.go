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
	"errors"
	"iter"
	"time"

	"go.opentelemetry.io/otel/trace"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// InstrumentInvocation wraps a runner invocation in an invoke_workflow span.
// Under the legacy schema, for a workflow root (which opens that span itself),
// and inside an existing invoke_workflow, run gets ctx unchanged.
func InstrumentInvocation(ctx context.Context, root AgentLike, sessionID string, run func(context.Context) iter.Seq2[*session.Event, error]) iter.Seq2[*session.Event, error] {
	// Unlike ADK Python, a nested invocation (e.g. agent as tool) gets no span: the spec omits implementation details.
	if useLegacySchema() || isWorkflowAgent(root) || ctx.Value(workflowScopeKey{}) != nil {
		return run(ctx)
	}
	return func(yield func(*session.Event, error) bool) {
		ctx, span := startInvokeWorkflowSpan(ctx, root.Name(), sessionID)
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

// InstrumentGenerateContent wraps a model call in its span, log records and
// client inference metrics.
// The span ends before the response that ends the call is yielded: an error,
// a response reporting a finish reason, or the one response of a call that is
// not streamed. The legacy schema ends it at the first response not marked
// partial instead.
func InstrumentGenerateContent[T any](ctx context.Context, params GenerateContentParams, response func(T) (*model.LLMResponse, string), call func(context.Context) iter.Seq2[T, error]) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		start := time.Now()
		ctx, span := startGenerateContentSpan(ctx, params, trace.WithTimestamp(start))
		logRequest(ctx, params.Request, params.Backend)
		details := operationDetails{span: span}
		details.noteLLMRequest(params.Request)
		// Both span formats record the same metrics. adk-python's older schema
		// records no inference token metrics; Go has no older metrics to keep.
		metrics := newClientInference(params, start)

		var result generateContentResult
		var spanEnd time.Time
		ended := false
		end := func() {
			if ended {
				return
			}
			ended = true
			traceGenerateContentResult(span, result)
			details.noteLLMRequest(params.Request)
			details.emit(ctx, params, result)
			spanEnd = time.Now()
			span.End(trace.WithTimestamp(spanEnd))
		}
		// Set on every path that returns normally, so the deferred function
		// can tell a panic apart.
		returned := false
		// The metrics count responses that arrive after the span ended, such
		// as an aggregate carrying usage, so they wait for the whole call.
		defer func() {
			if !returned && !ended {
				result.Error = errInferencePanicked
				metrics.observe(ctx, nil, errInferencePanicked, true)
			}
			end()
			metrics.record(ctx, spanEnd)
		}()
		legacy := useLegacySchema()
		for v, err := range call(ctx) {
			resp, eventID := response(v)
			metrics.observe(ctx, resp, err, !ended)
			switch {
			case legacy:
				result = generateContentResult{Response: resp, EventID: eventID, Error: inferenceError(err)}
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
				result.Error = inferenceError(err)
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
				returned = true
				return
			}
		}
		returned = true
	}
}

// errInferencePanicked stands for a panic that unwound a model call while its
// span was open. Its error type is the conventions' fallback value, _OTHER,
// since a panic value has no stable, low-cardinality name.
var errInferencePanicked = errors.New("model call panicked")

// inferenceError is the error the inference span, its event and its metrics
// record for a model error. A canceled call did not fail, so it records none:
// the span keeps an unset status and nothing gets an error type.
func inferenceError(err error) error {
	if errors.Is(err, context.Canceled) {
		return nil
	}
	return err
}
