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

package runner

import (
	"context"
	"errors"
	"iter"
	"os"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/telemetry"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

func forEachTraceSchema(t *testing.T, test func(*testing.T)) {
	t.Helper()
	for _, schema := range []string{"otel_semconv_1_36", "otel_semconv_1_44"} {
		t.Run(schema, func(t *testing.T) {
			t.Cleanup(telemetry.ApplyEnv)
			t.Setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema)
			telemetry.ApplyEnv()
			test(t)
		})
	}
}

func isInvocationSpan(sp tracetest.SpanStub) bool {
	if os.Getenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN") == "otel_semconv_1_36" {
		return sp.Name == "invocation"
	}
	if !strings.HasPrefix(sp.Name, "invoke_workflow ") {
		return false
	}
	for _, attr := range sp.Attributes {
		if string(attr.Key) == "gen_ai.workflow.nested" && attr.Value.AsBool() {
			return false
		}
	}
	return true
}

func mustTraceValue[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

func newInvocationTraceExporter(t *testing.T) (*sdktrace.TracerProvider, *tracetest.InMemoryExporter) {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(context.WithoutCancel(t.Context())) })
	telemetry.OverrideTracerForTesting(t, tp)
	return tp, exp
}

func invocationSpan(t *testing.T, exp *tracetest.InMemoryExporter) tracetest.SpanStub {
	t.Helper()
	var found []tracetest.SpanStub
	for _, sp := range exp.GetSpans() {
		if isInvocationSpan(sp) {
			found = append(found, sp)
		}
	}
	if len(found) != 1 {
		t.Fatalf("ended invocation spans = %d, want 1", len(found))
	}
	return found[0]
}

func TestRunInvocationTrace(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		for _, kind := range []string{"agent", "node", "workflow"} {
			for _, parent := range []bool{false, true} {
				name := kind
				if parent {
					name += "/parent"
				} else {
					name += "/root"
				}
				t.Run(name, func(t *testing.T) {
					tp, exp := newInvocationTraceExporter(t)
					ctx := t.Context()
					var parentSC trace.SpanContext
					if parent {
						var span trace.Span
						ctx, span = tp.Tracer("test").Start(ctx, "caller")
						defer span.End()
						parentSC = span.SpanContext()
					}
					seen := map[string]trace.SpanContext{}
					observe := func(name string, ctx context.Context) {
						t.Helper()
						sp := trace.SpanFromContext(ctx)
						if !sp.IsRecording() {
							t.Errorf("%s has no active span", name)
						}
						seen[name] = sp.SpanContext()
					}
					p := mustTraceValue(plugin.New(plugin.Config{
						Name: "tracing",
						OnUserMessageCallback: func(ctx agent.InvocationContext, _ *genai.Content) (*genai.Content, error) {
							observe("user", ctx)
							return nil, nil
						},
						BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) { observe("before_run", ctx); return nil, nil },
						AfterRunCallback:  func(ctx agent.InvocationContext) { observe("after_run", ctx) },
						OnEventCallback: func(ctx agent.InvocationContext, _ *session.Event) (*session.Event, error) {
							observe("event", ctx)
							return nil, nil
						},
						BeforeAgentCallback: func(ctx agent.Context) (*genai.Content, error) { observe("before_agent", ctx); return nil, nil },
						AfterAgentCallback:  func(ctx agent.Context) (*genai.Content, error) { observe("after_agent", ctx); return nil, nil },
						BeforeModelCallback: func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
							observe("before_model", ctx)
							return nil, nil
						},
						AfterModelCallback: func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
							observe("after_model", ctx)
							return nil, nil
						},
						BeforeToolCallback: func(ctx agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
							observe("before_tool", ctx)
							return nil, nil
						},
						AfterToolCallback: func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
							observe("after_tool", ctx)
							return nil, nil
						},
					}))
					sampleTool := mustTraceValue(functiontool.New(functiontool.Config{Name: "sample"}, func(agent.Context, struct{}) (struct{}, error) { return struct{}{}, nil }))
					root := mustTraceValue(llmagent.New(llmagent.Config{
						Name:  "root",
						Model: &toolCallingModel{toolName: "sample"},
						Tools: []tool.Tool{sampleTool},
					}))
					switch kind {
					case "agent":
						root = mustTraceValue(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "workflow", SubAgents: []agent.Agent{root}}}))
					case "workflow":
						node := mustTraceValue(workflow.NewAgentNode(root, workflow.NodeConfig{}))
						root = mustTraceValue(workflowagent.New(workflowagent.Config{Name: "workflow", Edges: workflow.Chain(workflow.Start, node)}))
					}
					r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
					drain(t, r.Run(ctx, "u", "s", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}))
					invocation := invocationSpan(t, exp)
					if !invocation.Parent.Equal(parentSC) {
						t.Error("invocation did not inherit the caller's parent")
					}
					for _, name := range []string{"user", "before_run", "after_run", "event", "before_agent", "after_agent", "before_model", "after_model", "before_tool", "after_tool"} {
						sc, ok := seen[name]
						if !ok {
							t.Errorf("%s callback was not called", name)
							continue
						}
						if sc.TraceID() != invocation.SpanContext.TraceID() {
							t.Errorf("%s is outside the invocation trace", name)
						}
					}
					for _, name := range []string{"user", "before_run", "after_run", "event"} {
						if seen[name].SpanID() != invocation.SpanContext.SpanID() {
							t.Errorf("%s did not receive the invocation span", name)
						}
					}
					for _, sp := range exp.GetSpans() {
						if sp.SpanContext.TraceID() != invocation.SpanContext.TraceID() {
							t.Errorf("span %q is outside the invocation trace", sp.Name)
						}
						if !sp.SpanContext.Equal(invocation.SpanContext) && (sp.StartTime.Before(invocation.StartTime) || sp.EndTime.After(invocation.EndTime)) {
							t.Errorf("span %q outlived the invocation", sp.Name)
						}
					}
				})
			}
		}
	})
}

func TestRunNestedWorkflowFromCallback(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		for _, callback := range []string{"before_runner", "after_runner", "before_agent", "after_agent"} {
			t.Run(callback, func(t *testing.T) {
				_, exp := newInvocationTraceExporter(t)
				newRoot := func() agent.Agent {
					node := workflow.NewFunctionNode("reply", func(_ agent.Context, in string) (string, error) {
						return in, nil
					}, workflow.NodeConfig{})
					return mustTraceValue(workflowagent.New(workflowagent.Config{Name: "workflow", Edges: workflow.Chain(workflow.Start, node)}))
				}
				inner := mustTraceValue(New(Config{AppName: "inner", Agent: newRoot(), SessionService: session.InMemoryService(), AutoCreateSession: true}))
				runInner := func(ctx agent.InvocationContext) {
					if strings.HasSuffix(callback, "agent") {
						drain(t, inner.rootAgent.Run(ctx))
					} else {
						drain(t, inner.Run(ctx, "u", "inner", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}))
					}
					if !trace.SpanFromContext(ctx).IsRecording() {
						t.Error("inner runner ended the outer invocation")
					}
				}
				p := mustTraceValue(plugin.New(plugin.Config{
					Name: "nested",
					BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
						if strings.HasPrefix(callback, "before") {
							runInner(ctx)
						}
						return nil, nil
					},
					AfterRunCallback: func(ctx agent.InvocationContext) {
						if strings.HasPrefix(callback, "after") {
							runInner(ctx)
						}
					},
				}))
				outer := mustTraceValue(New(Config{AppName: "outer", Agent: newRoot(), SessionService: session.InMemoryService(), AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
				drain(t, outer.Run(t.Context(), "u", "outer", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}))
				var workflows []tracetest.SpanStub
				for _, span := range exp.GetSpans() {
					if span.Name == "invoke_workflow workflow" {
						workflows = append(workflows, span)
					}
				}
				if len(workflows) != 2 {
					t.Fatalf("workflow spans = %d, want 2", len(workflows))
				}
				if workflows[0].SpanContext.TraceID() != workflows[1].SpanContext.TraceID() {
					t.Error("nested runner escaped the outer trace")
				}
				if strings.HasSuffix(callback, "runner") {
					ids := map[string]bool{}
					for _, sp := range workflows {
						for _, attr := range sp.Attributes {
							if attr.Key == "gen_ai.conversation.id" {
								ids[attr.Value.AsString()] = true
							}
						}
					}
					if !ids["inner"] || !ids["outer"] {
						t.Error("nested runner overwrote the outer conversation ID")
					}
				}
			})
		}
	})
}

func TestRunInvocationSpanEnds(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		for _, node := range []bool{false, true} {
			for _, outcome := range []string{"complete", "consumer_stop", "user_error", "before_error", "short_circuit", "agent_error", "cancel"} {
				name := "agent/" + outcome
				if node {
					name = "node/" + outcome
				}
				t.Run(name, func(t *testing.T) {
					_, exp := newInvocationTraceExporter(t)
					ctx, cancel := context.WithCancel(t.Context())
					defer cancel()
					fail := errors.New("test failure")
					p := mustTraceValue(plugin.New(plugin.Config{
						Name: "tracing",
						OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
							if outcome == "user_error" {
								return nil, fail
							}
							return nil, nil
						},
						BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
							switch outcome {
							case "before_error":
								return nil, fail
							case "short_circuit":
								return genai.NewContentFromText("done", genai.RoleModel), nil
							case "cancel":
								cancel()
							}
							return nil, nil
						},
						AfterRunCallback: func(ctx agent.InvocationContext) {
							if !trace.SpanFromContext(ctx).IsRecording() {
								t.Error("span ended before AfterRunCallback")
							}
						},
					}))
					root := mustTraceValue(agent.New(agent.Config{Name: "root", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							if outcome == "agent_error" {
								yield(nil, fail)
								return
							}
							if ctx.Err() != nil {
								yield(nil, ctx.Err())
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "root"
							yield(ev, nil)
						}
					}}))
					if node {
						var m model.LLM = &scriptedModel{replyFmt: "done %d"}
						if outcome == "agent_error" {
							m = &erroringModel{}
						}
						root = mustTraceValue(llmagent.New(llmagent.Config{Name: "root", Model: m}))
					}
					r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
					var gotErr bool
					for _, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}) {
						gotErr = gotErr || err != nil
						if outcome == "consumer_stop" {
							break
						}
					}
					wantErr := outcome == "user_error" || outcome == "before_error" || outcome == "agent_error" || outcome == "cancel"
					if gotErr != wantErr {
						t.Errorf("run error = %t, want %t", gotErr, wantErr)
					}
					invocationSpan(t, exp)
				})
			}
		}
	})
}

func TestRunInvocationSpanOnRepeatedIteration(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		_, exp := newInvocationTraceExporter(t)
		root := mustTraceValue(agent.New(agent.Config{Name: "root", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		}}))
		r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true}))
		events := r.Run(t.Context(), "u", "s", nil, agent.RunConfig{})
		for range 2 {
			drain(t, events)
		}
		var traces []trace.TraceID
		for _, sp := range exp.GetSpans() {
			if isInvocationSpan(sp) {
				if sp.Parent.IsValid() {
					t.Error("repeated iteration inherited an earlier span")
				}
				traces = append(traces, sp.SpanContext.TraceID())
			}
		}
		if len(traces) != 2 {
			t.Fatalf("invocation traces = %d, want 2", len(traces))
		}
		if traces[0] == traces[1] {
			t.Error("independent invocations share a trace")
		}
	})
}

func TestRunLiveInvocationTrace(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		for _, parentTrace := range []bool{false, true} {
			for _, outcome := range []string{"complete", "consumer_stop", "stream_error", "multiple_errors", "event_error", "append_error", "before_error", "short_circuit", "short_circuit_stop", "setup_error", "close_before_iteration", "cancel"} {
				name := outcome + "/root"
				if parentTrace {
					name = outcome + "/parent"
				}
				t.Run(name, func(t *testing.T) {
					tp, exp := newInvocationTraceExporter(t)
					ctx := t.Context()
					var parentSC trace.SpanContext
					if parentTrace {
						var parent trace.Span
						ctx, parent = tp.Tracer("test").Start(ctx, "caller")
						defer parent.End()
						parentSC = parent.SpanContext()
					}
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()
					var invocation trace.Span
					firstError := genai.APIError{Code: 429, Message: "test failure"}
					var afterRun bool
					p := mustTraceValue(plugin.New(plugin.Config{
						Name: "tracing",
						BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
							invocation = trace.SpanFromContext(ctx)
							if !invocation.IsRecording() {
								t.Error("BeforeRunCallback has no active span")
							}
							if outcome == "before_error" {
								return nil, firstError
							}
							if outcome == "short_circuit" || outcome == "short_circuit_stop" {
								return genai.NewContentFromText("done", genai.RoleModel), nil
							}
							return nil, nil
						},
						OnEventCallback: func(ctx agent.InvocationContext, _ *session.Event) (*session.Event, error) {
							if !trace.SpanContextFromContext(ctx).Equal(invocation.SpanContext()) {
								t.Error("OnEventCallback changed invocation span")
							}
							if outcome == "event_error" {
								return nil, firstError
							}
							return nil, nil
						},
						AfterRunCallback: func(ctx agent.InvocationContext) {
							afterRun = true
							if !trace.SpanFromContext(ctx).IsRecording() {
								t.Error("span ended before AfterRunCallback")
							}
						},
					}))
					root := &mockLiveAgent{Agent: mustTraceValue(agent.New(agent.Config{Name: "live"})), runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
						if !trace.SpanContextFromContext(ctx).Equal(invocation.SpanContext()) {
							t.Error("live agent changed invocation span")
						}
						if outcome == "setup_error" {
							return nil, nil, firstError
						}
						return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
							childCtx, child := tp.Tracer("test").Start(ctx, "live_agent")
							defer child.End()
							if outcome == "stream_error" || outcome == "multiple_errors" {
								if !yield(nil, firstError) || outcome == "stream_error" {
									return
								}
								if !yield(nil, errors.New("later failure")) {
									return
								}
							}
							if outcome == "cancel" {
								<-childCtx.Done()
								yield(nil, childCtx.Err())
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "live"
							ev.ID = "trace-event"
							yield(ev, nil)
						}, nil
					}}
					service := session.InMemoryService()
					if outcome == "append_error" {
						service = &failingLiveAppendService{Service: service, failID: "trace-event", err: firstError}
					}
					r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: service, AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
					live, events, err := r.RunLive(ctx, "u", "s", agent.LiveRunConfig{})
					observedError := err
					wantErr := outcome == "before_error" || outcome == "setup_error"
					if (err != nil) != wantErr {
						t.Fatalf("RunLive error = %t, want %t", err != nil, wantErr)
					}
					if err == nil {
						if !invocation.IsRecording() {
							t.Error("span ended when RunLive returned")
						}
						if outcome == "cancel" {
							cancel()
						}
						if outcome == "close_before_iteration" {
							if err := live.Close(); err != nil {
								t.Fatal(err)
							}
							if !invocation.IsRecording() {
								t.Error("Close ended invocation before iteration")
							}
						}
						var streamErr bool
						for _, err := range events {
							streamErr = streamErr || err != nil
							if observedError == nil && err != nil {
								observedError = err
							}
							if outcome == "consumer_stop" || outcome == "short_circuit_stop" {
								break
							}
						}
						if streamErr != (outcome == "stream_error" || outcome == "multiple_errors" || outcome == "event_error" || outcome == "append_error" || outcome == "cancel") {
							t.Error("unexpected stream error outcome")
						}
						if !afterRun {
							t.Error("AfterRunCallback was not called")
						}
						if err := live.Close(); err != nil {
							t.Fatal(err)
						}
					}
					sp := invocationSpan(t, exp)
					failed := wantErr || outcome == "stream_error" || outcome == "multiple_errors" || outcome == "event_error" || outcome == "append_error" || outcome == "cancel"
					wantStatus := codes.Unset
					if failed {
						wantStatus = codes.Error
					}
					if sp.Status.Code != wantStatus {
						t.Errorf("span status = %v, want %v", sp.Status.Code, wantStatus)
					}
					if failed {
						if len(sp.Events) != 1 {
							t.Errorf("span error events = %d, want 1", len(sp.Events))
						}
						if observedError == nil || sp.Status.Description != observedError.Error() {
							t.Error("span did not record the first failure")
						}
					}
					var errorType string
					for _, attr := range sp.Attributes {
						if attr.Key == "error.type" {
							errorType = attr.Value.AsString()
						}
					}
					wantType := ""
					if failed && os.Getenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN") == "otel_semconv_1_44" {
						wantType = "429"
						if outcome == "cancel" {
							wantType = "*errors.errorString"
						}
					}
					if errorType != wantType {
						t.Error("span error.type does not match the failure and schema")
					}
					if !sp.Parent.Equal(parentSC) {
						t.Error("live invocation did not inherit the caller's span")
					}
					for _, child := range exp.GetSpans() {
						if child.SpanContext.TraceID() != sp.SpanContext.TraceID() {
							t.Error("live span escaped the invocation trace")
						}
						if child.EndTime.After(sp.EndTime) {
							t.Error("live child outlived invocation span")
						}
					}
				})
			}
		}
	})
}
