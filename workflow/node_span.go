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

package workflow

import (
	"context"
	"errors"

	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry"
)

// nodeSpan is the single invoke_node span shared by both schedulers and
// the parallel worker.
type nodeSpan struct {
	span trace.Span
	// panicCtx is set for nodes whose own span is a no-op (Start and
	// EmitsOwnSpan nodes). Recording a recovered panic on the no-op span
	// would discard it, and the span already on this context is not a
	// safe target either: trace.SpanFromContext never returns nil, so it
	// can itself be a no-op (a direct Run with a plain context, or a
	// root the legacy schema opens no invoke_workflow span for), and it
	// outlives the attempt, so an Error status there would mark a run
	// failed that a retry went on to complete. recordPanic opens a short
	// dedicated span from this context instead.
	panicCtx context.Context
}

// startNodeSpan begins an invoke_node span for n and returns a context
// carrying it. It is a noop span for Start and for nodes that emit their
// own span (EmitsOwnSpan, e.g. an agent node's invoke_agent), so callers
// can unconditionally defer end / call recordError.
func startNodeSpan(ctx agent.Context, n Node) (nodeSpan, agent.Context) {
	if n == Start || n.Config().EmitsOwnSpan {
		return nodeSpan{span: noop.Span{}, panicCtx: ctx}, ctx
	}
	spanCtx, span := telemetry.StartNodeSpan(ctx, ctx, telemetry.OperationNode{Node: n})
	return nodeSpan{span: span}, ctx.WithAgentContext(spanCtx)
}

// recordPanic records err — a recovered panic's value and stack — on the
// node's own span, or on a short dedicated node_panic span when the
// node's own span is a no-op. The run error for a panic is generic, so
// this record is the only place an operator can read the panic detail;
// it must land on a span that is exported under every supported
// telemetry configuration, without flipping the status of a span the
// attempt does not own.
func (s nodeSpan) recordPanic(err error) {
	if s.panicCtx != nil {
		telemetry.RecordPanicSpan(s.panicCtx, err)
		return
	}
	telemetry.RecordError(s.span, err)
}

// recordError marks the span failed with recordErr, unless the outcome is
// expected control flow (see isNodeControlFlow). classifyErr picks the
// error to classify — pass the unwrapped cause when recordErr hides the
// sentinel (the dynamic scheduler wraps with "%w: %v", dropping
// context.Canceled); nil classifies on recordErr. Does not end the span.
func (s nodeSpan) recordError(recordErr, classifyErr error) {
	if recordErr == nil {
		return
	}
	if classifyErr == nil {
		classifyErr = recordErr
	}
	if isNodeControlFlow(classifyErr) {
		return
	}
	telemetry.RecordError(s.span, recordErr)
}

func (s nodeSpan) end() { s.span.End() }

// isNodeControlFlow reports whether err is expected control flow
// (cancellation or a HITL / wait-for-output pause), not a node failure.
// ErrNodeWaitingForOutput wraps ErrNodeInterrupted, so one check covers
// both pauses.
func isNodeControlFlow(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, ErrNodeInterrupted)
}
