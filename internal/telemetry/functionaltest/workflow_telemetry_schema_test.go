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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/internal/telemetry/telemetrytest"
	"google.golang.org/adk/internal/telemetry/telemetrytestcase"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// TestTelemetrySchema_WorkflowDynamic runs the
// [telemetrytestcase.WorkflowDynamicCase] scenario end-to-end and
// asserts the emitted span tree matches the expected shape.
// No LLM is involved so no log records are emitted; the test still
// installs an in-memory log exporter so a regression that suddenly
// starts emitting events would be caught by the cmp.Diff against the
// empty-Logs expectation.
//
// Hermetic: no network calls, no LLMs, no GCP.
//
// Mirrors test_telemetry_schema in
// adk-python/tests/unittests/telemetry/test_node_functional.py.
func TestTelemetrySchema_Workflow(t *testing.T) {
	spanExp := tracetest.NewInMemoryExporter()
	telemetry.OverrideTracerForTesting(t, sdktrace.NewTracerProvider(sdktrace.WithSyncer(spanExp)))

	logExp := telemetrytest.NewInMemoryLogExporter()
	telemetry.OverrideLoggerForTesting(t, sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewSimpleProcessor(logExp)),
	))

	upperFn := func(_ agent.InvocationContext, in string) (string, error) {
		return strings.ToUpper(in), nil
	}
	suffixFn := func(_ agent.InvocationContext, in string) (string, error) {
		return in + " IS AWESOME!", nil
	}

	nodeCfg := workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()}
	upperNode := workflow.NewFunctionNode("upper_node", upperFn, nodeCfg)
	suffixNode := workflow.NewFunctionNode("suffix_node", suffixFn, nodeCfg)

	// routerNode is a dynamic orchestrator: it expresses its execution
	// order as Go code, delegating to the two function nodes via
	// workflow.RunNode rather than wiring them as static graph edges.
	routerNode := workflow.NewDynamicNode("router_node",
		func(ctx workflow.NodeContext, in string, _ func(*session.Event) error) (string, error) {
			upper, err := workflow.RunNode[string](ctx, upperNode, in)
			if err != nil {
				return "", err
			}
			return workflow.RunNode[string](ctx, suffixNode, upper)
		},
		nodeCfg,
	)

	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "my_workflow",
		Description: "dynamic orchestrator delegating to two function nodes",
		Edges:       workflow.Chain(workflow.Start, routerNode),
	})
	if err != nil {
		t.Fatalf("workflowagent.New: %v", err)
	}

	telemetrytest.RunScenario(t, wfAgent, "hello")

	got := telemetrytest.BuildDigests(t, spanExp.GetSpans(), logExp.Records())
	if diff := cmp.Diff(telemetrytestcase.WorkflowDynamicCase, got); diff != "" {
		t.Errorf("telemetry mismatch (-want +got):\n%s", diff)
	}
}
