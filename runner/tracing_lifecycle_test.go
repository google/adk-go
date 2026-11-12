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
	"iter"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

func TestRunInvocationSessionID(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		for _, kind := range []string{"workflow", "node", "live"} {
			for _, sessionID := range []string{"", "provided"} {
				t.Run(kind+"/"+sessionID, func(t *testing.T) {
					_, exp := newInvocationTraceExporter(t)
					ctx := t.Context()
					var actualID string
					p := mustTraceValue(plugin.New(plugin.Config{Name: "session", BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
						actualID = ctx.Session().ID()
						return nil, nil
					}}))
					var root agent.Agent
					switch kind {
					case "workflow":
						node := workflow.NewFunctionNode("reply", func(_ agent.Context, in string) (string, error) { return in, nil }, workflow.NodeConfig{})
						root = mustTraceValue(workflowagent.New(workflowagent.Config{Name: "root", Edges: workflow.Chain(workflow.Start, node)}))
					case "node":
						root = mustTraceValue(llmagent.New(llmagent.Config{Name: "root", Model: &scriptedModel{replyFmt: "done %d"}}))
					case "live":
						root = &mockLiveAgent{Agent: mustTraceValue(agent.New(agent.Config{Name: "root"})), runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
							return &dummyLiveSession{}, func(func(*session.Event, error) bool) {}, nil
						}}
					}
					r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
					if kind == "live" {
						live, events, err := r.RunLive(ctx, "u", sessionID, agent.LiveRunConfig{})
						if err != nil {
							t.Fatal(err)
						}
						t.Cleanup(func() { _ = live.Close() })
						drain(t, events)
					} else {
						drain(t, r.Run(ctx, "u", sessionID, genai.NewContentFromText("hello", genai.RoleUser), agent.RunConfig{}))
					}
					if actualID == "" {
						t.Fatal("session service returned an empty ID")
					}
					var checkedRoot bool
					for _, sp := range exp.GetSpans() {
						var id string
						for _, a := range sp.Attributes {
							if a.Key == "gen_ai.conversation.id" {
								id = a.Value.AsString()
							}
						}
						switch {
						case sp.Name == "invocation":
							if id != "" {
								t.Error("legacy invocation gained a conversation ID attribute")
							}
						case strings.HasPrefix(sp.Name, "invoke_"):
							if id != actualID {
								t.Error("span does not carry the stored session ID")
							}
							if sp.Name == "invoke_workflow root" {
								checkedRoot = true
							}
						}
					}
					if os.Getenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN") == "otel_semconv_1_44" && !checkedRoot {
						t.Error("missing invocation workflow span")
					}
				})
			}
		}
	})
}

type closingTraceLiveSession struct {
	closed chan struct{}
	once   sync.Once
}

func (*closingTraceLiveSession) Send(agent.LiveRequest) error { return nil }

func (s *closingTraceLiveSession) Close() error {
	s.once.Do(func() { close(s.closed) })
	return nil
}

func TestRunLiveTraceEndsAfterCloseAndCallbacks(t *testing.T) {
	forEachTraceSchema(t, func(t *testing.T) {
		_, exp := newInvocationTraceExporter(t)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()
		started, afterStarted, releaseAfter := make(chan struct{}), make(chan struct{}), make(chan struct{})
		liveSession := &closingTraceLiveSession{closed: make(chan struct{})}
		var invocation trace.Span
		p := mustTraceValue(plugin.New(plugin.Config{Name: "trace", BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			invocation = trace.SpanFromContext(ctx)
			return nil, nil
		}, AfterRunCallback: func(agent.InvocationContext) {
			close(afterStarted)
			select {
			case <-releaseAfter:
			case <-ctx.Done():
			}
		}}))
		root := &mockLiveAgent{Agent: mustTraceValue(agent.New(agent.Config{Name: "live"})), runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return liveSession, func(yield func(*session.Event, error) bool) {
				close(started)
				select {
				case <-liveSession.closed:
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "live"
					yield(ev, nil)
				case <-ctx.Done():
					yield(nil, ctx.Err())
				}
			}, nil
		}}
		r := mustTraceValue(New(Config{AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true, PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}}}))
		live, events, err := r.RunLive(ctx, "u", "s", agent.LiveRunConfig{})
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := live.Close(); err != nil {
				t.Error(err)
			}
		}()
		done := make(chan error, 1)
		go func() {
			var streamErr error
			for _, err := range events {
				if err != nil {
					streamErr = err
				}
			}
			done <- streamErr
		}()
		select {
		case <-started:
		case <-ctx.Done():
			t.Fatal("stream did not start")
		}
		if err := live.Close(); err != nil {
			t.Fatal(err)
		}
		select {
		case <-afterStarted:
		case <-ctx.Done():
			t.Fatal("Close did not end the stream")
		}
		if !invocation.IsRecording() {
			t.Error("span ended before AfterRunCallback returned")
		}
		close(releaseAfter)
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			t.Fatal("event iterator did not finish")
		}
		invocationSpan(t, exp)
	})
}
