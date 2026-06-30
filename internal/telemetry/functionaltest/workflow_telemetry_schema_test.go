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

package functionaltest_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/internal/telemetry/telemetrytest"
	"google.golang.org/adk/v2/internal/telemetry/telemetrytestcase"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// errorStage selects which component of the canonical workflow
// scenario fails, so a single topology exercises both the happy
// path and every error-propagation path. Go has no enums; a typed
// int with a stringer is the idiomatic substitute.
type errorStage int

const (
	// errorNone is the happy path: every stage succeeds.
	errorNone errorStage = iota
	// errorStaticNode fails the static FunctionNode.
	errorStaticNode
	// errorDynamicNode fails the dynamic orchestrator body itself,
	// before it delegates to any child.
	errorDynamicNode
	// errorFirstAgent fails the first collaborative agent (the
	// ModeTask agent invoked via RunNode).
	errorFirstAgent
	// errorSecondAgent fails the second collaborative agent (the
	// ModeSingleTurn agent invoked via RunNode).
	errorSecondAgent
)

func (s errorStage) String() string {
	switch s {
	case errorNone:
		return "none"
	case errorStaticNode:
		return "static_node"
	case errorDynamicNode:
		return "dynamic_node"
	case errorFirstAgent:
		return "first_agent"
	case errorSecondAgent:
		return "second_agent"
	default:
		return fmt.Sprintf("errorStage(%d)", int(s))
	}
}

// TestTelemetrySchema_Workflow runs the canonical workflow scenario
// end-to-end and asserts the emitted span+log tree matches the
// expected shape exactly, across the happy path and every
// error-propagation stage.
func TestTelemetrySchema_Workflow(t *testing.T) {
	tests := []struct {
		name    string
		stage   errorStage
		wantErr bool
		want    *telemetrytest.SpanDigest
	}{
		{name: "happy_path", stage: errorNone, want: telemetrytestcase.WorkflowCase},
		{name: "error_static_node", stage: errorStaticNode, wantErr: true, want: telemetrytestcase.WorkflowErrorStaticNodeCase},
		{name: "error_dynamic_node", stage: errorDynamicNode, wantErr: true, want: telemetrytestcase.WorkflowErrorDynamicNodeCase},
		{name: "error_first_agent", stage: errorFirstAgent, wantErr: true, want: telemetrytestcase.WorkflowErrorFirstAgentCase},
		{name: "error_second_agent", stage: errorSecondAgent, wantErr: true, want: telemetrytestcase.WorkflowErrorSecondAgentCase},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Pin content elision to the production default so this
			// test is independent of whichever capture-content state
			// a previously-run test left in the global config.
			t.Setenv(captureMessageContentEnvVar, "")
			telemetry.ApplyEnv()

			spanExp := tracetest.NewInMemoryExporter()
			telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))

			logExp := telemetrytest.NewInMemoryLogExporter()
			telemetry.OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(
				sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExp)),
			))

			wfAgent := newWorkflowScenario(t, tc.stage)
			err := telemetrytest.RunScenario(t, wfAgent, "hello")
			if tc.wantErr && err == nil {
				t.Fatalf("stage %s: expected run error, got nil", tc.stage)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("stage %s: unexpected run error: %v", tc.stage, err)
			}

			got := telemetrytest.BuildDigests(t, spanExp.GetSpans(), logExp.Records())
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("telemetry mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// newWorkflowScenario builds the canonical workflow agent, injecting
// a failure at stage if stage != errorNone.
func newWorkflowScenario(t *testing.T, stage errorStage) agent.Agent {
	t.Helper()

	// No RetryConfig: a failing node fails on its first attempt, so the
	// error scenarios emit exactly one span per node rather than one
	// per retry — keeping the expected telemetry deterministic.
	nodeCfg := workflow.NodeConfig{}

	// Static node: uppercases the workflow input. Fails when the
	// scenario targets the static stage.
	staticNode := workflow.NewFunctionNode("static_node",
		func(_ agent.Context, in string) (string, error) {
			if stage == errorStaticNode {
				return "", fmt.Errorf("boom: static node failed")
			}
			return strings.ToUpper(in), nil
		}, nodeCfg)

	// Collaborative agent #1: a ModeTask agent that finishes in a
	// single turn (one text reply, no further tool calls).
	taskAgent := newCollaborativeAgent(t, "task_agent", llmagent.ModeTask, stage != errorFirstAgent, "task complete")
	taskNode, err := workflow.NewAgentNode(taskAgent, nodeCfg)
	if err != nil {
		t.Fatalf("NewAgentNode(task_agent): %v", err)
	}

	// Collaborative agent #2: a ModeSingleTurn agent.
	singleTurnAgent := newCollaborativeAgent(t, "single_turn_agent", llmagent.ModeSingleTurn, stage != errorSecondAgent, "single turn complete")
	singleTurnNode, err := workflow.NewAgentNode(singleTurnAgent, nodeCfg)
	if err != nil {
		t.Fatalf("NewAgentNode(single_turn_agent): %v", err)
	}

	// echoNode is a plain (non-agent) FunctionNode delegated to via
	// RunNode. Unlike the agent nodes — which emit their own
	// invoke_agent span and so are NOT wrapped in invoke_node — a
	// FunctionNode does get an invoke_node span, including on a
	// cache hit. It's the regression guard for cached-delegation
	// observability.
	echoNode := workflow.NewFunctionNode("echo_node",
		func(_ agent.Context, in string) (string, error) { return in, nil }, nodeCfg)

	routerNode := workflow.NewDynamicNode("router_node",
		func(ctx agent.Context, in string, _ func(*session.Event) error) (string, error) {
			if stage == errorDynamicNode {
				return "", fmt.Errorf("boom: dynamic node failed")
			}
			// Collaborative agents: each emits invoke_agent, no
			// invoke_node wrapper.
			if _, err := workflow.RunNode[any](ctx, taskNode, in); err != nil {
				return "", err
			}
			if _, err := workflow.RunNode[any](ctx, singleTurnNode, in); err != nil {
				return "", err
			}
			// Plain node twice with the SAME run id: the first call
			// runs it, the second is served from the RunNode result
			// cache. Both must emit an invoke_node span.
			if _, err := workflow.RunNode[any](ctx, echoNode, in, workflow.WithRunID("echo")); err != nil {
				return "", err
			}
			if _, err := workflow.RunNode[any](ctx, echoNode, in, workflow.WithRunID("echo")); err != nil {
				return "", err
			}
			return "done", nil
		}, nodeCfg)

	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "my_workflow",
		Description: "static node then a dynamic node delegating to two collaborative agents and a cached function node",
		Edges:       workflow.Chain(workflow.Start, staticNode, routerNode),
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}
	return wfAgent
}

// newCollaborativeAgent builds a hermetic llmagent in the given
// collaborative mode. When ok is false the backing model has no
// canned response and returns an error, so the agent fails — used
// to exercise error propagation from a delegated collaborative
// agent.
func newCollaborativeAgent(t *testing.T, name string, mode llmagent.Mode, ok bool, reply string) agent.Agent {
	t.Helper()
	var responses []*genai.Content
	if ok {
		responses = []*genai.Content{{
			Role:  "model",
			Parts: []*genai.Part{{Text: reply}},
		}}
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: "collaborative " + name,
		Model:       &testutil.MockModel{Responses: responses},
		Instruction: "you are helpful",
		Mode:        mode,
	})
	if err != nil {
		t.Fatalf("llmagent.New(%s): %v", name, err)
	}
	return a
}
