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
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/session"
)

// gen_ai.operation.name attribute values for the agent / workflow /
// node spans. These mirror the values set by adk-python's
// node_tracing module.
const (
	invokeWorkflowOperationName = "invoke_workflow"
	invokeNodeOperationName     = "invoke_node"
)

// Attribute keys not yet in the Go semconv package; the string
// literals match the keys used by adk-python.
var (
	genAIWorkflowName = attribute.Key("gen_ai.workflow.name")
	genAINodeName     = attribute.Key("gen_ai.node.name")
)

// Operation is the internal sum-type which describes what can be used to start a span.
type Operation interface {
	isOperation()
}

// AgentLike is the minimal contract telemetry needs to describe a BaseAgent.
type AgentLike interface {
	Name() string
	Description() string
}

// WorkflowNodeLike is the minimal contract telemetry needs to describe a workflow BaseNode.
type WorkflowNodeLike interface {
	Name() string
}

// OperationAgent is the [Operation] variant for an agent invocation.
// concrete span name (invoke_agent vs invoke_workflow) is decided
// inside [StartNodeSpan] by peeking at the agent's internal
// State.AgentType.
type OperationAgent struct {
	Agent AgentLike
}

func (OperationAgent) isOperation() {}

// OperationNode is the [Operation] variant for a workflow BaseNode activation.
type OperationNode struct {
	Node WorkflowNodeLike
}

func (OperationNode) isOperation() {}

// InvocationContext is the minimal copy of agent.InvocationContext necessary for telemetry.
type InvocationContext interface {
	Session() session.Session
	InvocationID() string
}

// StartNodeSpan emits an OpenTelemetry span and returns a derived context with the new Span.
// Emits:
// - invoke_workflow <workflow_name> if provided with [OperationNode] or workflow agent.
// - invoke_agent <agent_name> if provided with non-workflow agent.
// - invoke_node <node_name> if provided with a generic node.
// - otherwise doesn't start any new span and returns the input context unchanged.
func StartNodeSpan(ctx context.Context, ictx InvocationContext, op Operation) (context.Context, trace.Span) {
	switch o := op.(type) {
	case OperationAgent:
		if isWorkflowAgent(o.Agent) {
			return startInvokeWorkflowSpan(ctx, ictx, o.Agent)
		}
		return startInvokeAgentSpan(ctx, ictx, o.Agent)
	case OperationNode:
		return startInvokeNodeSpan(ctx, ictx, o.Node)
	default:
		return ctx, noop.Span{}
	}
}

func isWorkflowAgent(a AgentLike) bool {
	internal, ok := a.(agentinternal.Agent)
	if !ok {
		return false
	}
	return agentinternal.Reveal(internal).AgentType == agentinternal.TypeWorkflowAgent
}

// sessionID returns ictx.Session().ID() or "" if the
// InvocationContext has no session attached. Tolerates nil
// sessions because workflow tests construct mock invocation
// contexts without one.
func sessionID(ictx InvocationContext) string {
	s := ictx.Session()
	if s == nil {
		return ""
	}
	return s.ID()
}

func startInvokeAgentSpan(ctx context.Context, ictx InvocationContext, a AgentLike) (context.Context, trace.Span) {
	return tracer.Start(ctx, fmt.Sprintf("invoke_agent %s", a.Name()), trace.WithAttributes(
		gcpVertexAgentInvocationID.String(ictx.InvocationID()), // used by adk-web
		semconv.GenAIOperationNameInvokeAgent,
		semconv.GenAIAgentDescription(a.Description()),
		semconv.GenAIAgentName(a.Name()),
		semconv.GenAIConversationID(sessionID(ictx)),
	))
}

func startInvokeWorkflowSpan(ctx context.Context, ictx InvocationContext, w AgentLike) (context.Context, trace.Span) {
	return tracer.Start(ctx, fmt.Sprintf("invoke_workflow %s", w.Name()), trace.WithAttributes(
		semconv.GenAIOperationNameKey.String(invokeWorkflowOperationName),
		genAIWorkflowName.String(w.Name()),
		semconv.GenAIConversationID(sessionID(ictx)),
	))
}

func startInvokeNodeSpan(ctx context.Context, ictx InvocationContext, n WorkflowNodeLike) (context.Context, trace.Span) {
	return tracer.Start(ctx, fmt.Sprintf("invoke_node %s", n.Name()), trace.WithAttributes(
		semconv.GenAIOperationNameKey.String(invokeNodeOperationName),
		genAINodeName.String(n.Name()),
		semconv.GenAIConversationID(sessionID(ictx)),
	))
}
