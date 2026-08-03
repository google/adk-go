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

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/telemetry"
)

// nodeSpan is the single invoke_node span shared by both schedulers and
// the parallel worker.
type nodeSpan struct {
	span trace.Span
}

// startNodeSpan begins an invoke_node span for n and returns a context
// carrying it. It is a noop span for Start and for nodes that emit their
// own span (EmitsOwnSpan, e.g. an agent node's invoke_agent), so callers
// can unconditionally defer end / call recordError.
func startNodeSpan(ctx agent.Context, n Node) (nodeSpan, agent.Context) {
	if n == Start || n.Config().EmitsOwnSpan {
		return nodeSpan{span: noop.Span{}}, ctx
	}
	spanCtx, span := telemetry.StartNodeSpan(ctx, ctx, telemetry.OperationNode{Node: n})
	return nodeSpan{span: span}, ctx.WithAgentContext(spanCtx)
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
	s.span.RecordError(recordErr)
	s.span.SetStatus(codes.Error, recordErr.Error())
}

func (s nodeSpan) end() { s.span.End() }

// isNodeControlFlow reports whether err is expected control flow
// (cancellation or a HITL / wait-for-output pause), not a node failure.
// ErrNodeWaitingForOutput wraps ErrNodeInterrupted, so one check covers
// both pauses.
func isNodeControlFlow(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, ErrNodeInterrupted)
}
