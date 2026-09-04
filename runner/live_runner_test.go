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
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

type mockLiveAgent struct {
	agent.Agent
	runLiveFn func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func (m *mockLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return m.runLiveFn(ctx)
}

type dummyLiveSession struct{}

func (d *dummyLiveSession) Send(req agent.LiveRequest) error { return nil }
func (d *dummyLiveSession) Close() error                     { return nil }

func newRunLiveTestRunner(
	t *testing.T,
	sessionID string,
	sequence func(agent.InvocationContext) iter.Seq2[*session.Event, error],
) (*Runner, session.Service) {
	t.Helper()

	const appName, userID = "testApp", "testUser"
	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "test_agent"}))
	r, err := New(Config{
		AppName: appName,
		Agent: &mockLiveAgent{
			Agent: testAgent,
			runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
				return &dummyLiveSession{}, sequence(ctx), nil
			},
		},
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}
	return r, sessionService
}

func getRunLiveTestEvents(t *testing.T, sessionService session.Service, sessionID string) session.Events {
	t.Helper()
	getResp, err := sessionService.Get(t.Context(), &session.GetRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}
	return getResp.Session.Events()
}

func TestRunner_RunLive_NilEventYieldedTerminatesRun(t *testing.T) {
	ctx := context.Background()
	appName, userID, sessionID := "testApp", "testUser", "testSessionNilFlood"

	sessionService := session.InMemoryService()
	_, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "nil_flood"}))
	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				for {
					if !yield(nil, nil) {
						return
					}
				}
			}, nil
		},
	}

	r, err := New(Config{
		AppName:        appName,
		Agent:          mockLive,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, iter, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	type result struct {
		err        error
		iterations int
	}
	done := make(chan result, 1)
	go func() {
		got := result{}
		for _, err := range iter {
			got.err = err
			got.iterations++
			break
		}
		done <- got
	}()

	select {
	case got := <-done:
		if got.iterations != 1 {
			t.Fatalf("RunLive() iterations = %d, want 1", got.iterations)
		}
		if got.err == nil {
			t.Fatal("RunLive() yielded no error for a nil event")
		}
		if !strings.Contains(got.err.Error(), "nil_flood") {
			t.Fatalf("RunLive() error = %v, want the agent name", got.err)
		}
	case <-time.After(time.Second):
		t.Fatal("RunLive() did not terminate after a nil event")
	}
}

func TestRunner_RunLive_Callbacks(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession"

	sessionService := session.InMemoryService()
	_, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	var beforeRunCalled, afterRunCalled bool

	p, err := plugin.New(plugin.Config{
		Name: "test_plugin",
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			beforeRunCalled = true
			return nil, nil
		},
		AfterRunCallback: func(ctx agent.InvocationContext) {
			afterRunCalled = true
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "test_agent"}))
	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				yield(session.NewEvent(ctx, ctx.InvocationID()), nil)
			}, nil
		},
	}

	r, err := New(Config{
		AppName:        appName,
		Agent:          mockLive,
		SessionService: sessionService,
		PluginConfig: PluginConfig{
			Plugins: []*plugin.Plugin{p},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sess, iter, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	if sess == nil {
		t.Fatal("expected LiveSession to be returned")
	}

	if !beforeRunCalled {
		t.Error("BeforeRunCallback was not called before starting RunLive")
	}

	if afterRunCalled {
		t.Error("AfterRunCallback should not be called until iterator is consumed")
	}

	for _, err := range iter {
		if err != nil {
			t.Fatal(err)
		}
	}

	if !afterRunCalled {
		t.Error("AfterRunCallback was not called after iterator was consumed")
	}
}

func TestRunner_RunLive_EarlyExit(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession2"

	sessionService := session.InMemoryService()
	_, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	expectedContent := genai.NewContentFromText("early exit content", "")

	p, err := plugin.New(plugin.Config{
		Name: "test_plugin",
		BeforeRunCallback: func(ctx agent.InvocationContext) (*genai.Content, error) {
			return expectedContent, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "test_agent"}))
	var runLiveCalled bool
	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			runLiveCalled = true
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {}, nil
		},
	}

	r, err := New(Config{
		AppName:        appName,
		Agent:          mockLive,
		SessionService: sessionService,
		PluginConfig: PluginConfig{
			Plugins: []*plugin.Plugin{p},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	sess, iter, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	if runLiveCalled {
		t.Error("RunLive should not have been called on the agent due to early exit")
	}

	var events []*session.Event
	for ev, err := range iter {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}

	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}

	if events[0].LLMResponse.Content != expectedContent {
		t.Errorf("expected content %v, got %v", expectedContent, events[0].LLMResponse.Content)
	}

	err = sess.Send(agent.LiveRequest{})
	if err == nil || !strings.Contains(err.Error(), "session is closed") {
		t.Errorf("expected error 'session is closed' when sending to early exited session, got %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestRunner_RunLive_ChronologicalBuffering(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession3"

	sessionService := session.InMemoryService()
	_, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "test_agent"}))
	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				// 1. Partial Transcription
				ev1 := session.NewEvent(ctx, ctx.InvocationID())
				ev1.LLMResponse.Partial = true
				ev1.LLMResponse.OutputTranscription = &genai.Transcription{Text: "Hello"}
				if !yield(ev1, nil) {
					return
				}

				// 2. Function Call (happening during transcription)
				ev2 := session.NewEvent(ctx, ctx.InvocationID())
				ev2.LLMResponse.Content = &genai.Content{
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "test_func"}}},
				}
				if !yield(ev2, nil) {
					return
				}

				// 3. Final Transcription
				ev3 := session.NewEvent(ctx, ctx.InvocationID())
				ev3.LLMResponse.OutputTranscription = &genai.Transcription{Text: "Hello there."}
				if !yield(ev3, nil) {
					return
				}
			}, nil
		},
	}

	r, err := New(Config{
		AppName:        appName,
		Agent:          mockLive,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}

	_, iter, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	// Consume iterator to execute everything
	for _, err := range iter {
		if err != nil {
			t.Fatal(err)
		}
	}

	// Verify Session History
	getResp, err := sessionService.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatal(err)
	}

	events := getResp.Session.Events()
	// We expect 2 saved events: Final Transcription first, Function Call second.
	// (Partial Transcription is not saved).
	if events.Len() != 2 {
		t.Fatalf("expected 2 saved events in session, got %d", events.Len())
	}

	// First saved event should be the final transcription
	if events.At(0).LLMResponse.OutputTranscription == nil {
		t.Errorf("expected first saved event to be transcription, but got %v", events.At(0))
	}

	if events.At(0).LLMResponse.OutputTranscription.Text != "Hello there." {
		t.Errorf("expected first saved event to be transcription with text: %q, got: %q", "Hello there.", events.At(0).LLMResponse.OutputTranscription.Text)
	}

	// Second saved event should be the function call
	if events.At(1).LLMResponse.Content == nil || events.At(1).LLMResponse.Content.Parts[0].FunctionCall == nil {
		t.Errorf("expected second saved event to be function call, but got %v", events.At(1))
	}
}

func TestRunner_RunLive_FlushesBufferedEventsAtEOF(t *testing.T) {
	const sessionID = "testSessionBufferedEOF"
	var bufferedCall *session.Event
	r, sessionService := newRunLiveTestRunner(t, sessionID, func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			partial := session.NewEvent(ctx, ctx.InvocationID())
			partial.LLMResponse.Partial = true
			partial.LLMResponse.InputTranscription = &genai.Transcription{Text: "book me a "}
			if !yield(partial, nil) {
				return
			}

			bufferedCall = session.NewEvent(ctx, ctx.InvocationID())
			bufferedCall.LLMResponse.Content = &genai.Content{
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "book_flight"}}},
			}
			yield(bufferedCall, nil)
		}
	})

	_, stream, err := r.RunLive(t.Context(), "testUser", sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	var received []*session.Event
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event == bufferedCall {
			if got := getRunLiveTestEvents(t, sessionService, sessionID).Len(); got != 1 {
				t.Fatalf("persisted events while yielding buffered call = %d, want 1", got)
			}
		}
		received = append(received, event)
	}

	if len(received) != 2 {
		t.Fatalf("RunLive() yielded %d events, want partial transcription and buffered call", len(received))
	}
	if received[1] != bufferedCall {
		t.Fatalf("RunLive() second event = %v, want buffered function call", received[1])
	}

	events := getRunLiveTestEvents(t, sessionService, sessionID)
	if got := events.Len(); got != 1 {
		t.Fatalf("persisted events after EOF = %d, want 1", got)
	}
	content := events.At(0).LLMResponse.Content
	if content == nil || len(content.Parts) != 1 || content.Parts[0].FunctionCall == nil || content.Parts[0].FunctionCall.Name != "book_flight" {
		t.Fatalf("persisted event = %v, want book_flight function call", events.At(0))
	}
}

func TestRunner_RunLive_DoesNotFlushBufferedEventsAfterConsumerStops(t *testing.T) {
	const sessionID = "testSessionBufferedStop"
	var upstreamSawStop bool
	r, sessionService := newRunLiveTestRunner(t, sessionID, func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			partial := session.NewEvent(ctx, ctx.InvocationID())
			partial.LLMResponse.Partial = true
			partial.LLMResponse.InputTranscription = &genai.Transcription{Text: "book me a "}
			if !yield(partial, nil) {
				return
			}

			call := session.NewEvent(ctx, ctx.InvocationID())
			call.LLMResponse.Content = &genai.Content{
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "book_flight"}}},
			}
			if !yield(call, nil) {
				return
			}

			continuedPartial := session.NewEvent(ctx, ctx.InvocationID())
			continuedPartial.LLMResponse.Partial = true
			continuedPartial.LLMResponse.InputTranscription = &genai.Transcription{Text: "book me a flight"}
			if !yield(continuedPartial, nil) {
				upstreamSawStop = true
			}
		}
	})

	_, stream, err := r.RunLive(t.Context(), "testUser", sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	yieldCalls := 0
	stream(func(_ *session.Event, err error) bool {
		if err != nil {
			t.Fatal(err)
		}
		yieldCalls++
		return yieldCalls < 2
	})

	if yieldCalls != 2 {
		t.Fatalf("downstream yield calls = %d, want 2", yieldCalls)
	}
	if !upstreamSawStop {
		t.Fatal("upstream iterator did not observe the downstream stop")
	}
	if got := getRunLiveTestEvents(t, sessionService, sessionID).Len(); got != 0 {
		t.Fatalf("persisted events after consumer stop = %d, want 0", got)
	}
}
