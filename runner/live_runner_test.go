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
	"iter"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
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

// TestRunner_RunLive_SkipsNilEvents covers the RunLive half of the nil guard,
// which the Run-side tests never reach.
func TestRunner_RunLive_SkipsNilEvents(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession"

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatal(err)
	}

	testAgent, err := agent.New(agent.Config{
		Name: "live_nil_yielder",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				if !yield(nil, nil) {
					return
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "agent"
				ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{{Text: "after nil"}}}
				if !yield(ev, nil) {
					return
				}
				_ = yield(nil, nil)
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

	_, events, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}

	var gotTexts []string
	for ev, err := range events {
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if ev == nil {
			t.Error("RunLive yielded a nil event")
			continue
		}
		gotTexts = append(gotTexts, ev.LLMResponse.Content.Parts[0].Text)
	}

	if diff := cmp.Diff([]string{"after nil"}, gotTexts); diff != "" {
		t.Errorf("delivered events mismatch (-want +got):\n%s", diff)
	}
}

// TestRunner_RunLive_NilEventRunIsBounded is the RunLive counterpart of
// TestRunner_NilEventRunIsBounded.
func TestRunner_RunLive_NilEventRunIsBounded(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession"

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatal(err)
	}

	testAgent := must(agent.New(agent.Config{Name: "endless_live_nil_yielder"}))
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

	_, events, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}

	var gotErr error
	for ev, err := range events {
		if ev != nil {
			t.Errorf("RunLive yielded an event, want none")
		}
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("RunLive() error = nil, want an error naming the agent")
	}
	if !strings.Contains(gotErr.Error(), "endless_live_nil_yielder") {
		t.Errorf("RunLive() error = %q, want it to name the offending agent", gotErr)
	}
}

// TestRunner_RunLive_BufferedEventDoesNotResetNilBound pins that an event the
// live path consumes without yielding (transcription buffering) does not reset
// the nil bound. Resetting there would let an interleaved stream of nils and
// buffered events ride the bound forever without ever reaching the consumer.
func TestRunner_RunLive_BufferedEventDoesNotResetNilBound(t *testing.T) {
	ctx := t.Context()
	appName, userID, sessionID := "testApp", "testUser", "testSession"

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		t.Fatal(err)
	}

	// A partial transcription event puts the loop into buffering mode, after
	// which tool-call events are consumed into bufferedEvents without a yield.
	transcription := func() *session.Event {
		ev := session.NewEvent(ctx, "inv")
		ev.Author = "agent"
		ev.LLMResponse.Partial = true
		ev.LLMResponse.OutputTranscription = &genai.Transcription{Text: "hi"}
		return ev
	}
	toolCall := func() *session.Event {
		ev := session.NewEvent(ctx, "inv")
		ev.Author = "agent"
		ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "fn", ID: "1"}},
		}}
		return ev
	}

	testAgent := must(agent.New(agent.Config{Name: "interleaving_agent"}))
	mockLive := &mockLiveAgent{
		Agent: testAgent,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
				if !yield(transcription(), nil) {
					return
				}
				// nils and buffered tool calls, interleaved forever
				for {
					for range maxNilEventsBetweenDeliveries {
						if !yield(nil, nil) {
							return
						}
					}
					if !yield(toolCall(), nil) {
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

	_, events, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}

	var gotErr error
	for _, err := range events {
		gotErr = err
	}

	if gotErr == nil {
		t.Fatal("RunLive() error = nil, want the nil bound to terminate the run")
	}
	if !strings.Contains(gotErr.Error(), "interleaving_agent") {
		t.Errorf("RunLive() error = %q, want it to name the offending agent", gotErr)
	}
}
