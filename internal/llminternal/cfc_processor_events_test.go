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
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

const cfcHelloTurn = `{"serverContent":{"modelTurn":{"parts":[{"text":"hello"}],"role":"model"},"turnComplete":true}}`

// scrubCallback returns an after-model callback that replaces every response
// it sees with the text "scrubbed" and counts its calls in calls.
func scrubCallback(calls *atomic.Int32) AfterModelCallback {
	return func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
		calls.Add(1)
		return &model.LLMResponse{Content: genai.NewContentFromText("scrubbed", genai.RoleModel)}, nil
	}
}

// Under CFC the after-model callbacks see the model's responses only, as
// without CFC: the content-less event that the built-in code executor's
// response step yields before the model's event reaches the run as it is.
// adk-python runs them on every event of the live call
// (flows/llm_flows/core/_model_call.py); Go does not (ledger: Fixed bug,
// ruling R18).
func TestCFCAfterModelCallbacksSkipResponseProcessorEvents(t *testing.T) {
	tests := []struct {
		name    string
		replace bool
		want    []string
	}{
		{
			name:    "replacing callback gives one replacement",
			replace: true,
			want:    []string{"test_agent empty", "test_agent text scrubbed (turn complete)"},
		},
		{
			name:    "observing callback sees the model response once",
			replace: false,
			want:    []string{"test_agent empty", "test_agent text hello (turn complete)"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			defer assertNoRunLiveLeak(t, baseline)
			client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
				if !writeFrames(t, conn, cfcHelloTurn) {
					return
				}
				blockUntilClientCloses(conn)
			})
			var calls atomic.Int32
			callback := func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
				calls.Add(1)
				return nil, nil
			}
			if tc.replace {
				callback = scrubCallback(&calls)
			}
			f := &Flow{
				Model:               &cfcModel{name: "gemini-2.0-flash", client: client},
				RequestProcessors:   liveOnlyProcessors,
				ResponseProcessors:  DefaultResponseProcessors,
				AfterModelCallbacks: []AfterModelCallback{callback},
			}
			ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "test_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

			events, errs := collectEvents(t, f.Run(ctx))

			if len(errs) != 0 {
				t.Fatalf("Run() errors = %v, want none", errs)
			}
			if diff := cmp.Diff(tc.want, describeCFCEvents(events)); diff != "" {
				t.Errorf("events mismatch (-want +got):\n%s", diff)
			}
			if len(events) > 0 && events[0].Content != nil {
				t.Errorf("first event Content = %v, want nil", events[0].Content)
			}
			if got := calls.Load(); got != 1 {
				t.Errorf("after-model callbacks ran %d times, want 1", got)
			}
		})
	}
}

// Under CFC the after-model callbacks do not see an event that a request
// processor yields when the live call builds its request: it reaches the run
// as the processor made it, as without CFC.
func TestCFCAfterModelCallbacksSkipRequestProcessorEvents(t *testing.T) {
	baseline, _ := runLiveStacks()
	defer assertNoRunLiveLeak(t, baseline)
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		if !writeFrames(t, conn, cfcHelloTurn) {
			return
		}
		blockUntilClientCloses(conn)
	})
	noteProcessor := func(ctx agent.InvocationContext, _ *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Author = ctx.Agent().Name()
			ev.Content = genai.NewContentFromText("note", genai.RoleModel)
			yield(ev, nil)
		}
	}
	var calls atomic.Int32
	f := &Flow{
		Model: &cfcModel{name: "gemini-2.0-flash", client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{
			liveConfigProcessor, noteProcessor,
		},
		AfterModelCallbacks: []AfterModelCallback{scrubCallback(&calls)},
	}
	ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "test_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

	events, errs := collectEvents(t, f.Run(ctx))

	if len(errs) != 0 {
		t.Fatalf("Run() errors = %v, want none", errs)
	}
	// The request processors run once for the step and once more for the
	// live call's own request (ledger: Ported flaw, ruling R11).
	want := []string{"test_agent text note", "test_agent text note", "test_agent text scrubbed (turn complete)"}
	if diff := cmp.Diff(want, describeCFCEvents(events)); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("after-model callbacks ran %d times, want 1", got)
	}
}

// A run that stops during the CFC call's request processors stops the
// processor that is yielding: its next yield returns false and it is not
// asked for more.
func TestCFCRunStoppedDuringRequestProcessorStopsProcessor(t *testing.T) {
	baseline, _ := runLiveStacks()
	client, _ := startFakeLiveServer(t, func(_ int, conn *websocket.Conn) {
		blockUntilClientCloses(conn)
	})
	var runs atomic.Int32
	var mu sync.Mutex
	var liveYields []bool
	notesProcessor := func(ctx agent.InvocationContext, _ *model.LLMRequest, _ *Flow) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			live := runs.Add(1) == 2
			for _, text := range []string{"note 1", "note 2", "note 3"} {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = ctx.Agent().Name()
				ev.Content = genai.NewContentFromText(text, genai.RoleModel)
				ok := yield(ev, nil)
				if live {
					mu.Lock()
					liveYields = append(liveYields, ok)
					mu.Unlock()
				}
				if !ok {
					return
				}
			}
		}
	}
	f := &Flow{
		Model: &cfcModel{name: "gemini-2.0-flash", client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{
			liveConfigProcessor, notesProcessor,
		},
	}
	ctx := newCFCContext(t, cfcRun{agent: newCFCAgent(t, "test_agent", &State{}), runConfig: agent.RunConfig{SupportCFC: true}})

	done := make(chan []string, 1)
	go func() {
		var got []string
		for ev, err := range f.Run(ctx) {
			if err != nil {
				got = append(got, "error "+err.Error())
				break
			}
			got = append(got, describeCFCEvent(ev))
			// The step's three events, then the live call's first.
			if len(got) == 4 {
				break
			}
		}
		done <- got
	}()
	var got []string
	select {
	case got = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the run did not stop within 10s")
	}
	assertNoRunLiveLeak(t, baseline)

	want := []string{"test_agent text note 1", "test_agent text note 2", "test_agent text note 3", "test_agent text note 1"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("events mismatch (-want +got):\n%s", diff)
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]bool{true, false}, liveYields); diff != "" {
		t.Errorf("live call's processor yield results mismatch (-want +got):\n%s", diff)
	}
}
