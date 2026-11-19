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

package llminternal_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// The tests in this file pin how the history of function calls and
// responses is rearranged before it reaches the model, as adk-python's
// flows/llm_flows/tools/_rearranger.py does, through the public entry point
// ContentsRequestProcessor.

const rcAgentName = "test_agent"

func rcCall(id, name string) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: name}}
}

func rcResp(id, name string, resp map[string]any) *genai.Part {
	return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: name, Response: resp}}
}

func rcText(s string) *genai.Part { return &genai.Part{Text: s} }

func rcEvent(author, inv, role string, parts ...*genai.Part) *session.Event {
	return &session.Event{Author: author, InvocationID: inv, LLMResponse: model.LLMResponse{
		Content: &genai.Content{Role: role, Parts: parts},
	}}
}

func rcContent(role string, parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: role, Parts: parts}
}

// rcBuild runs ContentsRequestProcessor for an LLM agent named test_agent
// over a session holding events, and returns the request contents or the
// processor's error. It fails the test when the processor changes the stored
// events.
func rcBuild(t *testing.T, events []*session.Event) ([]*genai.Content, error) {
	t.Helper()
	before, err := json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		after, err := json.Marshal(events)
		if err != nil {
			t.Fatal(err)
		}
		if string(before) != string(after) {
			t.Errorf("ContentsRequestProcessor mutated session events (-before +after):\n%s", cmp.Diff(string(before), string(after)))
		}
	}()
	a := utils.Must(llmagent.New(llmagent.Config{Name: rcAgentName, Model: &testModel{}}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   a,
		Session: &fakeSession{events: events},
	})
	req := &model.LLMRequest{}
	for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("ContentsRequestProcessor generated an unexpected event")
		}
		if err != nil {
			return nil, err
		}
	}
	return req.Contents, nil
}

// rcCheck fails the test unless rcBuild returns want and no error.
func rcCheck(t *testing.T, events []*session.Event, want []*genai.Content, opts ...cmp.Option) {
	t.Helper()
	got, err := rcBuild(t, events)
	if err != nil {
		t.Fatalf("ContentsRequestProcessor failed: %v", err)
	}
	if diff := cmp.Diff(want, got, opts...); diff != "" {
		t.Errorf("LLMRequest.Contents mismatch (-want +got):\n%s", diff)
	}
}

// ignoreRoleAt leaves the role of the content at index i of the compared
// contents out of the comparison.
func ignoreRoleAt(i int) cmp.Option {
	return cmp.FilterPath(func(p cmp.Path) bool {
		if len(p) != 4 {
			return false
		}
		idx, ok := p[1].(cmp.SliceIndex)
		if !ok || idx.Key() != i {
			return false
		}
		f, ok := p[3].(cmp.StructField)
		return ok && f.Name() == "Role"
	}, cmp.Ignore())
}

func inScope(scope string, ev *session.Event) *session.Event {
	ev.IsolationScope = scope
	return ev
}

// TestContentsRequestProcessor_ScopeRules pins where adk-go keeps a response
// that adk-python would drop, and where it does not.
func TestContentsRequestProcessor_ScopeRules(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			// The call c2 is in another isolation scope, so the agent does
			// not see it. Fails if resp c2 stays beside the owned resp c1, or
			// if any unowned trailing part turns into an error.
			name: "S2: unowned response beside an owned one is removed",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				inScope("task", rcEvent(rcAgentName, "", "model", rcCall("c2", "t2"))),
				rcEvent(rcAgentName, "", "model", rcCall("c1", "t1")),
				rcEvent("user", "", "user",
					rcResp("c1", "t1", map[string]any{"r": 1}),
					rcResp("c2", "t2", map[string]any{"r": 2})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "t1")),
				rcContent("user", rcResp("c1", "t1", map[string]any{"r": 1})),
			},
		},
		{
			// Fails if every split across two call events raises the
			// validation error, not only one with a plain event after the
			// newest owner.
			name: "S3: no plain event after the newest owner splits",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("A", "ta")),
				rcEvent(rcAgentName, "", "model", rcCall("B", "tb")),
				rcEvent("user", "", "user",
					rcResp("A", "ta", map[string]any{"r": "a"}),
					rcResp("B", "tb", map[string]any{"r": "b"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("A", "ta")),
				rcContent("user", rcResp("A", "ta", map[string]any{"r": "a"})),
				rcContent("model", rcCall("B", "tb")),
				rcContent("user", rcResp("B", "tb", map[string]any{"r": "b"})),
			},
		},
		{
			// The text left by the dropped orphaned response zz sits between
			// the newest owner and the last event. Fails if such a remnant
			// counts as a plain event and raises the validation error.
			name: "S3: orphan remnant after the newest owner does not block the split",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("A", "ta")),
				rcEvent(rcAgentName, "", "model", rcCall("B", "tb")),
				rcEvent("user", "", "user", rcText("note"), rcResp("zz", "t", map[string]any{"r": "orphan"})),
				rcEvent("user", "", "user",
					rcResp("A", "ta", map[string]any{"r": "a"}),
					rcResp("B", "tb", map[string]any{"r": "b"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("A", "ta")),
				rcContent("user", rcResp("A", "ta", map[string]any{"r": "a"})),
				rcContent("user", rcText("note")),
				rcContent("model", rcCall("B", "tb")),
				rcContent("user", rcResp("B", "tb", map[string]any{"r": "b"})),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rcCheck(t, tc.events, tc.want)
		})
	}
}

// TestContentsRequestProcessor_OrphanedResponseRemoval pins which events and
// calls remain when the contents build drops a function response that no
// earlier call answers.
func TestContentsRequestProcessor_OrphanedResponseRemoval(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			// Fails if the calls of an event that lost an orphaned response
			// are not recorded, so the later response c2 is orphaned too.
			name: "calls of an event that lost an orphaned response still answer",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcResp("zz", "t", map[string]any{"r": "orphan"}), rcCall("c2", "tool_b")),
				rcEvent("user", "", "user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c2", "tool_b")),
				rcContent("user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
			},
		},
		{
			// Fails if an empty name counts as a name to match, so a
			// response with neither an ID nor a name answers a call with
			// neither.
			name: "response with neither an ID nor a name answers no call",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("", "")),
				rcEvent("user", "", "user", rcResp("", "", map[string]any{"r": "x"})),
				rcEvent("user", "", "user", rcText("next")),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("", "")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// The response c1 becomes the last event and moves next to its
			// call, which leaves "mid" out. Fails if the emptied event stays
			// last: the "mid" text then follows the response.
			name: "event holding only an orphaned response leaves the history",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("c1", "tool")),
				rcEvent(rcAgentName, "", "model", rcText("mid")),
				rcEvent("user", "", "user", rcResp("c1", "tool", map[string]any{"r": "late"})),
				rcEvent("user", "", "user", rcResp("zz", "other", map[string]any{"r": "orphan"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "tool")),
				rcContent("user", rcResp("c1", "tool", map[string]any{"r": "late"})),
			},
		},
		{
			// "note" goes into c1's merged response with the progress
			// response beside it. Fails if the event that lost resp zz is
			// also kept after c1 for its text, so "note" is sent twice.
			name: "text of a remnant whose response was merged is sent once",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("c1", "ta")),
				rcEvent("user", "", "user",
					rcText("note"),
					rcResp("zz", "t", map[string]any{"r": "orphan"}),
					rcResp("c1", "ta", map[string]any{"r": "progress"})),
				rcEvent("user", "", "user", rcResp("c1", "ta", map[string]any{"r": "final"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "ta")),
				rcContent("user", rcText("note"), rcResp("c1", "ta", map[string]any{"r": "final"})),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rcCheck(t, tc.events, tc.want)
		})
	}
}

// TestContentsRequestProcessor_UnownedTrailingResponsesFail pins the error
// for a last event whose responses all answer calls the agent cannot see,
// with the number of responses. It fails if the error is raised only for a
// last event with one response.
func TestContentsRequestProcessor_UnownedTrailingResponsesFail(t *testing.T) {
	_, err := rcBuild(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		inScope("task", rcEvent(rcAgentName, "", "model", rcCall("c1", "t1"))),
		inScope("task", rcEvent(rcAgentName, "", "model", rcCall("c2", "t2"))),
		rcEvent("user", "", "user",
			rcResp("c1", "t1", map[string]any{"r": 1}),
			rcResp("c2", "t2", map[string]any{"r": 2})),
	})
	const want = "no function call event found for the 2 function responses of the last event"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("ContentsRequestProcessor error = %v, want one containing %q", err, want)
	}
}

// TestContentsRequestProcessor_ReusedCallIDLateResponseAnswersNewestCall pins
// that a response goes to the newest earlier call with its ID, so two calls
// with one ID keep their order and the response follows the second. It fails
// if the oldest call owns the response.
func TestContentsRequestProcessor_ReusedCallIDLateResponseAnswersNewestCall(t *testing.T) {
	callWithArg := func(n int) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "t", Args: map[string]any{"n": n}}}
	}
	rcCheck(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		rcEvent(rcAgentName, "", "model", callWithArg(1)),
		rcEvent(rcAgentName, "", "model", callWithArg(2)),
		rcEvent("user", "", "user", rcResp("c1", "t", map[string]any{"r": "late"})),
	}, []*genai.Content{
		rcContent("user", rcText("go")),
		rcContent("model", callWithArg(1)),
		rcContent("model", callWithArg(2)),
		rcContent("user", rcResp("c1", "t", map[string]any{"r": "late"})),
	})
}

// TestContentsRequestProcessor_SplitValidationErrorHasNoModelData pins that
// the error for a split blocked by a plain event after the newest owner names
// none of the function call and response IDs, which come from the model. It
// fails if the error lists the IDs.
func TestContentsRequestProcessor_SplitValidationErrorHasNoModelData(t *testing.T) {
	_, err := rcBuild(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		rcEvent(rcAgentName, "", "model", rcCall("call_7f3a", "ta")),
		rcEvent(rcAgentName, "", "model", rcCall("call_9b21", "tb")),
		rcEvent(rcAgentName, "", "model", rcText("working")),
		rcEvent("user", "", "user",
			rcResp("call_7f3a", "ta", map[string]any{"r": "a"}),
			rcResp("call_9b21", "tb", map[string]any{"r": "b"})),
	})
	const prefix = "validation failed: last response event has IDs not in the matching call event"
	if err == nil || !strings.HasPrefix(err.Error(), prefix) {
		t.Fatalf("ContentsRequestProcessor error = %v, want one starting with %q", err, prefix)
	}
	for _, id := range []string{"call_7f3a", "call_9b21"} {
		if strings.Contains(err.Error(), id) {
			t.Errorf("ContentsRequestProcessor error contains the model's ID %q", id)
		}
	}
}

// TestContentsRequestProcessor_KeepUnrelatedExchanges pins that a late
// response moved next to its call keeps the tool exchanges that happened
// in between, where adk-python drops every event after the newest owner. It
// fails if those exchanges are dropped.
func TestContentsRequestProcessor_KeepUnrelatedExchanges(t *testing.T) {
	rcCheck(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		rcEvent(rcAgentName, "", "model", rcCall("L", "l")),
		rcEvent("user", "", "user", rcResp("L", "l", map[string]any{"r": "prog"})),
		rcEvent(rcAgentName, "", "model", rcCall("S", "s")),
		rcEvent("user", "", "user", rcResp("S", "s", map[string]any{"r": "s"})),
		rcEvent("user", "", "user", rcResp("L", "l", map[string]any{"r": "final"})),
	}, []*genai.Content{
		rcContent("user", rcText("go")),
		rcContent("model", rcCall("S", "s")),
		rcContent("user", rcResp("S", "s", map[string]any{"r": "s"})),
		rcContent("model", rcCall("L", "l")),
		rcContent("user", rcResp("L", "l", map[string]any{"r": "final"})),
	})
}

// TestContentsRequestProcessor_PythonQuirks pins adk-python's split of the
// last event, which merges an intermediate event holding responses of two
// calls into each call's response, its text included. It fails if the text
// goes to the newest owner only.
func TestContentsRequestProcessor_PythonQuirks(t *testing.T) {
	rcCheck(t, []*session.Event{
		rcEvent(rcAgentName, "", "model", rcCall("a", "ta")),
		rcEvent(rcAgentName, "", "model", rcCall("b", "tb")),
		rcEvent("user", "", "user",
			rcResp("a", "ta", map[string]any{"r": "a1"}),
			rcResp("b", "tb", map[string]any{"r": "b1"}),
			rcText("note")),
		rcEvent("user", "", "user",
			rcResp("a", "ta", map[string]any{"r": "a2"}),
			rcResp("b", "tb", map[string]any{"r": "b2"})),
	}, []*genai.Content{
		rcContent("model", rcCall("a", "ta")),
		rcContent("user", rcResp("a", "ta", map[string]any{"r": "a2"}), rcText("note")),
		rcContent("model", rcCall("b", "tb")),
		rcContent("user", rcResp("b", "tb", map[string]any{"r": "b2"}), rcText("note")),
	})
}

// TestContentsRequestProcessor_StaleReply pins when a model text between two
// updates of one call is dropped as superseded, and the conditions that keep
// it.
func TestContentsRequestProcessor_StaleReply(t *testing.T) {
	start := func() *session.Event { return rcEvent("user", "i0", "user", rcText("go")) }
	next := func() *session.Event { return rcEvent("user", "i2", "user", rcText("next")) }
	callWatch := func() *session.Event { return rcEvent(rcAgentName, "inv1", "model", rcCall("c1", "watch")) }
	reply := func(s string) *session.Event { return rcEvent(rcAgentName, "inv1", "model", rcText(s)) }
	update := func(author, result string) *session.Event {
		return rcEvent(author, "inv1", "user", rcResp("c1", "watch", map[string]any{"r": result}))
	}
	withConfirmation := func(ev *session.Event) *session.Event {
		ev.Actions.RequestedToolConfirmations = map[string]toolconfirmation.ToolConfirmation{"c1": {Hint: "Confirm?"}}
		return ev
	}
	withAuth := func(ev *session.Event) *session.Event {
		ev.Actions.RequestedAuthConfigs = map[string]toolauth.Config{"c1": {}}
		return ev
	}
	image := func() *genai.Part {
		return &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}}
	}

	tests := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			// adk-python's rule is positional: the reply is about c2 but sits
			// between two updates of c1. Fails if the drop looks at which
			// call the text is about.
			name: "reply about another parallel call between two updates is dropped",
			events: []*session.Event{
				start(),
				rcEvent(rcAgentName, "inv1", "model", rcCall("c1", "tool_a")),
				rcEvent(rcAgentName, "inv1", "model", rcCall("c2", "tool_b")),
				rcEvent("user", "inv1", "user", rcResp("c1", "tool_a", map[string]any{"r": "progress_a"})),
				reply("tool_b finished: here you go"),
				rcEvent("user", "inv1", "user", rcResp("c1", "tool_a", map[string]any{"r": "done_a"})),
				rcEvent("user", "inv1", "user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				reply("All finished."),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "tool_a")),
				rcContent("user", rcResp("c1", "tool_a", map[string]any{"r": "done_a"})),
				rcContent("model", rcCall("c2", "tool_b")),
				rcContent("user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				rcContent("model", rcText("All finished.")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if a tool confirmation request does not keep the reply.
			name: "reply kept after an update that requested a tool confirmation",
			events: []*session.Event{
				start(),
				callWatch(),
				withConfirmation(update(rcAgentName, "needs confirmation")),
				reply("Are you sure?"),
				update(rcAgentName, "deleted"),
				reply("Done."),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "deleted"})),
				rcContent("model", rcText("Are you sure?")),
				rcContent("model", rcText("Done.")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if a credential request does not keep the reply.
			name: "reply kept after an update that requested credentials",
			events: []*session.Event{
				start(),
				callWatch(),
				withAuth(update(rcAgentName, "needs credentials")),
				reply("Please sign in."),
				update(rcAgentName, "done"),
				reply("Done."),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "done"})),
				rcContent("model", rcText("Please sign in.")),
				rcContent("model", rcText("Done.")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if the authors of the two updates are not compared.
			name: "reply kept when the two updates have different authors",
			events: []*session.Event{
				start(),
				callWatch(),
				update(rcAgentName, "pending"),
				reply("Pending approval."),
				update("user", "approved"),
				reply("Done."),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "approved"})),
				rcContent("model", rcText("Pending approval.")),
				rcContent("model", rcText("Done.")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if the reply's own invocation is not compared with the
			// updates' invocation.
			name: "reply of another invocation between two updates of one invocation is kept",
			events: []*session.Event{
				start(),
				callWatch(),
				update(rcAgentName, "progress"),
				rcEvent(rcAgentName, "i3", "model", rcText("From another invocation")),
				update(rcAgentName, "done"),
				reply("Done."),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "done"})),
				rcContent("model", rcText("From another invocation")),
				rcContent("model", rcText("Done.")),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if a model event that holds a call counts as a stale
			// reply.
			name: "model event with text and a call between two updates is kept",
			events: []*session.Event{
				start(),
				callWatch(),
				update(rcAgentName, "progress"),
				rcEvent(rcAgentName, "inv1", "model", rcText("Starting b"), rcCall("c2", "tool_b")),
				update(rcAgentName, "done"),
				rcEvent("user", "inv1", "user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "done"})),
				rcContent("model", rcText("Starting b"), rcCall("c2", "tool_b")),
				rcContent("user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				rcContent("user", rcText("next")),
			},
		},
		{
			// A model text event that also carries the response of c2 is a
			// stale text reply, as in adk-python, which does not look at
			// responses; so the run is pruned and "Checking" is dropped. The
			// c2 event still follows its call. Fails if an event holding a
			// function response stops counting as a stale reply.
			name: "model text event carrying another call's response does not stop the prune",
			events: []*session.Event{
				start(),
				rcEvent(rcAgentName, "inv1", "model", rcCall("c1", "tool_a")),
				rcEvent(rcAgentName, "inv1", "model", rcCall("c2", "tool_b")),
				rcEvent("user", "inv1", "user", rcResp("c1", "tool_a", map[string]any{"r": "progress_a"})),
				reply("Checking"),
				rcEvent(rcAgentName, "inv1", "model", rcText("b is done"), rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				rcEvent("user", "inv1", "user", rcResp("c1", "tool_a", map[string]any{"r": "done_a"})),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "tool_a")),
				rcContent("user", rcResp("c1", "tool_a", map[string]any{"r": "done_a"})),
				rcContent("model", rcCall("c2", "tool_b")),
				rcContent("model", rcText("b is done"), rcResp("c2", "tool_b", map[string]any{"r": "b"})),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if a model event with no text counts as a stale reply.
			name: "model event without text between two updates is kept",
			events: []*session.Event{
				start(),
				callWatch(),
				update(rcAgentName, "progress"),
				rcEvent(rcAgentName, "inv1", "model", image()),
				update(rcAgentName, "done"),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "done"})),
				rcContent("model", image()),
				rcContent("user", rcText("next")),
			},
		},
		{
			// Fails if the role of the event is not checked.
			name: "agent-authored text with the user role between two updates is kept",
			events: []*session.Event{
				start(),
				callWatch(),
				update(rcAgentName, "progress"),
				rcEvent(rcAgentName, "inv1", "user", rcText("note")),
				update(rcAgentName, "done"),
				next(),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "watch")),
				rcContent("user", rcResp("c1", "watch", map[string]any{"r": "done"})),
				rcContent("user", rcText("note")),
				rcContent("user", rcText("next")),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rcCheck(t, tc.events, tc.want)
		})
	}
}

// TestContentsRequestProcessor_ParallelResponsesWithOneKeyInOneEvent pins
// that one event holding two id-less responses to one call event counts as
// one update of that call. It fails, with a panic, if the event is counted
// once per response part.
func TestContentsRequestProcessor_ParallelResponsesWithOneKeyInOneEvent(t *testing.T) {
	rcCheck(t, []*session.Event{
		rcEvent("user", "inv1", "user", rcText("go")),
		rcEvent(rcAgentName, "inv1", "model", rcCall("", "tool_a"), rcCall("", "tool_b")),
		rcEvent("user", "inv1", "user",
			rcResp("", "tool_a", map[string]any{"r": "a"}),
			rcResp("", "tool_b", map[string]any{"r": "b"})),
		rcEvent("user", "inv1", "user", rcText("next")),
	}, []*genai.Content{
		rcContent("user", rcText("go")),
		rcContent("model", rcCall("", "tool_a"), rcCall("", "tool_b")),
		rcContent("user",
			rcResp("", "tool_a", map[string]any{"r": "a"}),
			rcResp("", "tool_b", map[string]any{"r": "b"})),
		rcContent("user", rcText("next")),
	})
}

// TestContentsRequestProcessor_ParallelCallsAnsweredByOneEventWithText pins
// that one response event answering two calls of one call event is sent once,
// with its text once. It fails if the event is merged with itself.
func TestContentsRequestProcessor_ParallelCallsAnsweredByOneEventWithText(t *testing.T) {
	rcCheck(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		rcEvent(rcAgentName, "", "model", rcCall("c1", "tool_a"), rcCall("c2", "tool_b")),
		rcEvent("user", "", "user",
			rcResp("c1", "tool_a", map[string]any{"r": "a"}),
			rcResp("c2", "tool_b", map[string]any{"r": "b"}),
			rcText("note")),
		rcEvent("user", "", "user", rcText("next")),
	}, []*genai.Content{
		rcContent("user", rcText("go")),
		rcContent("model", rcCall("c1", "tool_a"), rcCall("c2", "tool_b")),
		rcContent("user",
			rcResp("c1", "tool_a", map[string]any{"r": "a"}),
			rcResp("c2", "tool_b", map[string]any{"r": "b"}),
			rcText("note")),
		rcContent("user", rcText("next")),
	})
}

// TestContentsRequestProcessor_TrailingResponseSplit pins how the last event
// is split when its responses answer calls in two call events.
func TestContentsRequestProcessor_TrailingResponseSplit(t *testing.T) {
	tests := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
		opts   []cmp.Option
	}{
		{
			// Fails if the text is dropped or goes with the oldest owner.
			name: "non-response part goes with the newest owner",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("c1", "tool_a")),
				rcEvent(rcAgentName, "", "model", rcCall("c2", "tool_b")),
				rcEvent("user", "", "user",
					rcResp("c1", "tool_a", map[string]any{"r": "a"}),
					rcResp("c2", "tool_b", map[string]any{"r": "b"}),
					rcText("note")),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "tool_a")),
				rcContent("user", rcResp("c1", "tool_a", map[string]any{"r": "a"})),
				rcContent("model", rcCall("c2", "tool_b")),
				rcContent("user", rcResp("c2", "tool_b", map[string]any{"r": "b"}), rcText("note")),
			},
		},
		{
			// Fails if the consumed response stays in the earlier event, or
			// if that event goes whole, call included. The merged c1 response
			// takes its role from the first intermediate event, as in
			// adk-python, so its role is not compared.
			name: "event with a consumed response and a call keeps only the call",
			events: []*session.Event{
				rcEvent("user", "", "user", rcText("go")),
				rcEvent(rcAgentName, "", "model", rcCall("c1", "tool_a")),
				rcEvent(rcAgentName, "", "model",
					rcResp("c1", "tool_a", map[string]any{"p": 50}),
					rcCall("c2", "tool_b")),
				rcEvent("user", "", "user",
					rcResp("c1", "tool_a", map[string]any{"p": 100}),
					rcResp("c2", "tool_b", map[string]any{"r": "b"})),
			},
			want: []*genai.Content{
				rcContent("user", rcText("go")),
				rcContent("model", rcCall("c1", "tool_a")),
				rcContent("", rcResp("c1", "tool_a", map[string]any{"p": 100})),
				rcContent("model", rcCall("c2", "tool_b")),
				rcContent("user", rcResp("c2", "tool_b", map[string]any{"r": "b"})),
			},
			opts: []cmp.Option{ignoreRoleAt(2)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rcCheck(t, tc.events, tc.want, tc.opts...)
		})
	}
}

// TestContentsRequestProcessor_TrailingEventAnsweringItsOwnCall pins that a
// call in the same event as its response does not answer it, so the
// response is dropped with no error, as in adk-python. It fails if the
// response counts as answered or as retained for a call outside the history.
func TestContentsRequestProcessor_TrailingEventAnsweringItsOwnCall(t *testing.T) {
	rcCheck(t, []*session.Event{
		rcEvent("user", "", "user", rcText("go")),
		rcEvent(rcAgentName, "", "model", rcCall("c1", "tool"), rcResp("c1", "tool", map[string]any{"r": "same"})),
	}, []*genai.Content{
		rcContent("user", rcText("go")),
	})
}
