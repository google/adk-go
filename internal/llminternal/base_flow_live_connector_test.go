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

package llminternal

import (
	"context"
	"errors"
	"io"
	"iter"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

type flowLiveConnector struct {
	fakeLiveModel
	connect func(context.Context, *model.LLMRequest) (model.LiveConnection, error)
}

func (m *flowLiveConnector) Connect(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	return m.connect(ctx, req)
}

type flowLiveConnection struct {
	historyErr error
	send       func(context.Context, *genai.Content) error
	receive    func(context.Context, func(*model.LLMResponse, error) bool)
}

func (c *flowLiveConnection) SendHistory(context.Context, []*genai.Content) error {
	return c.historyErr
}

func (c *flowLiveConnection) SendContent(ctx context.Context, content *genai.Content) error {
	return c.send(ctx, content)
}

func (*flowLiveConnection) SendRealtime(context.Context, any) error { return nil }

func (c *flowLiveConnection) Receive(ctx context.Context) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) { c.receive(ctx, yield) }
}

func (*flowLiveConnection) Close() error { return nil }

func TestRunLiveRetainsHandleBehindPendingToolResponse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const toolName, oldHandle, newHandle = "lookup", "old", "new"
		allowHandle, failSend := make(chan struct{}), make(chan struct{})
		first := &flowLiveConnection{
			send: func(ctx context.Context, content *genai.Content) error {
				select {
				case <-failSend:
					return io.EOF
				case <-ctx.Done():
					return ctx.Err()
				}
			},
			receive: func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
				call := genai.NewContentFromParts([]*genai.Part{{FunctionCall: &genai.FunctionCall{
					Name: toolName, ID: "call-1",
				}}}, genai.RoleModel)
				if !yield(&model.LLMResponse{Content: call}, nil) {
					return
				}
				select {
				case <-allowHandle:
					if yield(&model.LLMResponse{SessionResumptionHandle: newHandle}, nil) {
						<-ctx.Done()
					}
				case <-ctx.Done():
				}
			},
		}
		dials := 0
		f := &Flow{
			Model: &flowLiveConnector{connect: func(_ context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
				dials++
				cfg := req.LiveConnectConfig.SessionResumption
				switch dials {
				case 1:
					if cfg == nil || cfg.Handle != oldHandle {
						t.Error("initial dial did not use the original handle")
					}
					return first, nil
				case 2:
					if cfg == nil || cfg.Handle != newHandle {
						t.Error("redial did not use the handle queued behind the tool response")
					}
					return &flowLiveConnection{receive: func(context.Context, func(*model.LLMResponse, error) bool) {}}, nil
				default:
					return nil, errors.New("unexpected live reconnect")
				}
			}},
			Tools: []tool.Tool{&mockFunctionTool{name: toolName, runFunc: func(agent.Context, map[string]any) (map[string]any, error) {
				return map[string]any{"ok": true}, nil
			}}},
			reconnect: testLiveReconnectPolicy(2, 2),
		}
		ctx, cancel := newLiveInvocationContext(t)
		defer cancel()
		ctx.(*icontext.InvocationContext).SetLiveSessionResumptionHandle(oldHandle)
		sess, seq, err := f.RunLive(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = sess.Close() }()

		// Keep the only sender busy until the flow has queued a tool response.
		if err := sess.Send(agent.LiveRequest{Content: genai.NewContentFromText("input", genai.RoleUser)}); err != nil {
			t.Fatalf("sending input: %v", err)
		}
		next, stop := iter.Pull2(seq)
		defer stop()
		for {
			ev, err, ok := next()
			if !ok || err != nil {
				t.Fatalf("live stream ended before the tool response: %v", err)
			}
			if len(utils.FunctionResponses(ev.Content)) > 0 {
				break
			}
		}
		synctest.Wait()
		close(allowHandle)
		// The reader is now parked trying to forward the handle. The pending
		// tool response prevents it from reaching the event loop before the
		// sender exits, so the connection cannot recover the handle there.
		synctest.Wait()
		close(failSend)
		for ev, err, ok := next(); ok; ev, err, ok = next() {
			if err != nil || ev != nil {
				t.Fatalf("expected clean EOF from the resumed connection, got event=%t, error=%v", ev != nil, err)
			}
		}
		if dials != 2 {
			t.Errorf("got %d dials, want 2", dials)
		}
	})
}

func TestRunLiveSendBeforeEventsOnTerminalError(t *testing.T) {
	wantErr := errors.New("live test failure")
	for _, tc := range []struct {
		name      string
		wantText  string
		sendFirst bool
	}{
		{name: "preprocess"},
		{name: "connect"},
		{name: "nil connection", wantText: "connector returned a nil live connection"},
		{name: "history"},
		{name: "transport send", sendFirst: true},
		{name: "consecutive retry limit", wantText: "consecutive attempts", sendFirst: true},
		{name: "short-lived retry limit", wantText: "short-lived connections", sendFirst: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				conn := &flowLiveConnection{
					historyErr: wantErr,
					send:       func(context.Context, *genai.Content) error { return wantErr },
					receive:    func(ctx context.Context, _ func(*model.LLMResponse, error) bool) { <-ctx.Done() },
				}
				var dials int
				f := &Flow{
					Model: &flowLiveConnector{connect: func(context.Context, *model.LLMRequest) (model.LiveConnection, error) {
						dials++
						switch tc.name {
						case "nil connection":
							return nil, nil
						case "connect":
							return nil, wantErr
						case "consecutive retry limit", "short-lived retry limit":
							if dials > 1 {
								return nil, wantErr
							}
							return &flowLiveConnection{
								send:    func(context.Context, *genai.Content) error { return io.EOF },
								receive: conn.receive,
							}, nil
						default:
							return conn, nil
						}
					}},
					RequestProcessors: []func(agent.InvocationContext, *model.LLMRequest, *Flow) iter.Seq2[*session.Event, error]{
						func(_ agent.InvocationContext, req *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								switch tc.name {
								case "preprocess":
									yield(nil, wantErr)
								case "history":
									req.Contents = []*genai.Content{genai.NewContentFromText("history", genai.RoleUser)}
								}
							}
						},
					},
					reconnect: testLiveReconnectPolicy(1, 2),
				}
				if tc.name == "short-lived retry limit" {
					f.reconnect = testLiveReconnectPolicy(2, 1)
				}
				ctx, cancel := newLiveInvocationContext(t)
				defer cancel()
				sess, seq, err := f.RunLive(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = sess.Close() }()

				req := agent.LiveRequest{Content: genai.NewContentFromText("input", genai.RoleUser)}
				if tc.sendFirst {
					// Send enqueues input. This write stops the writer, so
					// the next Send must be released by teardown.
					if err := sess.Send(req); err != nil {
						t.Fatalf("enqueueing first input: %v", err)
					}
				}
				sent := make(chan error, 1)
				go func() { sent <- sess.Send(req) }()
				select {
				case err := <-sent:
					if !errors.Is(err, io.EOF) {
						t.Errorf("Send before iteration: got %v, want EOF", err)
					}
				case <-time.After(time.Second):
					t.Fatal("Send is waiting for the Events iterator to start")
				}

				n := 0
				for ev, err := range seq {
					n++
					if ev != nil || err == nil ||
						(tc.name != "nil connection" && !errors.Is(err, wantErr)) ||
						(tc.wantText != "" && !strings.Contains(err.Error(), tc.wantText)) {
						t.Errorf("expected terminal error for %s, got event=%t, error=%v", tc.name, ev != nil, err)
					}
				}
				if n != 1 {
					t.Errorf("iterator yielded %d results, want 1", n)
				}
			})
		})
	}
}
