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

package compactioninternal

import (
	"context"
	"iter"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// epoch anchors the synthetic timestamps used across these tests. Tests express
// times as small integers via at(); only their relative order matters.
var epoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// at returns a deterministic timestamp n seconds after the test epoch.
func at(n int) time.Time { return epoch.Add(time.Duration(n) * time.Second) }

// ids extracts the event IDs of events, for readable diffs in table tests.
// An empty result is normalized to nil so "no events" reads the same whether
// the caller returned nil or an empty slice.
func ids(events []*session.Event) []string {
	if len(events) == 0 {
		return nil
	}
	out := make([]string, len(events))
	for i, ev := range events {
		out[i] = ev.ID
	}
	return out
}

func newEvent(id, invocationID string, ts int, author string, parts ...*genai.Part) *session.Event {
	ev := &session.Event{
		ID:           id,
		InvocationID: invocationID,
		Timestamp:    at(ts),
		Author:       author,
	}
	if len(parts) > 0 {
		ev.LLMResponse.Content = &genai.Content{Role: author, Parts: parts}
	}
	return ev
}

func textEvent(id, invocationID string, ts int, text string) *session.Event {
	return newEvent(id, invocationID, ts, "user", &genai.Part{Text: text})
}

func modelTextEvent(id, invocationID string, ts int, text string) *session.Event {
	return newEvent(id, invocationID, ts, "model", &genai.Part{Text: text})
}

func callEvent(id, invocationID string, ts int, callID string) *session.Event {
	return newEvent(id, invocationID, ts, "model", &genai.Part{
		FunctionCall: &genai.FunctionCall{ID: callID, Name: "tool_" + callID},
	})
}

func multiCallEvent(id, invocationID string, ts int, callIDs ...string) *session.Event {
	parts := make([]*genai.Part, 0, len(callIDs))
	for _, c := range callIDs {
		parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: c, Name: "tool_" + c}})
	}
	return newEvent(id, invocationID, ts, "model", parts...)
}

func responseEvent(id, invocationID string, ts int, callID string) *session.Event {
	return newEvent(id, invocationID, ts, "user", &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID: callID, Name: "tool_" + callID, Response: map[string]any{"result": "ok"},
		},
	})
}

func callAndResponseEvent(id, invocationID string, ts int, callID string) *session.Event {
	return newEvent(id, invocationID, ts, "model",
		&genai.Part{FunctionResponse: &genai.FunctionResponse{ID: callID, Name: "tool_" + callID}},
		&genai.Part{FunctionCall: &genai.FunctionCall{ID: callID, Name: "tool_" + callID}},
	)
}

func confirmationEvent(id, invocationID string, ts int, callID string) *session.Event {
	ev := newEvent(id, invocationID, ts, "model")
	ev.Actions.RequestedToolConfirmations = map[string]toolconfirmation.ToolConfirmation{
		callID: {Hint: "approve?"},
	}
	return ev
}

// compactionEvent builds a stored compaction event: it sits at timestamp ts in
// the stream and covers the inclusive range [start, end].
// compactionEvent builds a stored record covering [start, end] except for
// excludedIDs, which is how a real one records the holes window selection left.
func compactionEvent(id string, ts, start, end int, summary string, excluded ...session.EventRef) *session.Event {
	return &session.Event{
		ID:           id,
		InvocationID: "compaction-" + id,
		Timestamp:    at(ts),
		Author:       "user",
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   at(start),
				EndTimestamp:     at(end),
				CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: summary}}},
				ExcludedEvents:   excluded,
			},
		},
	}
}

// fakeSummarizer records the windows it is handed and returns a canned summary,
// so window-selection behaviour can be tested without a model.
type fakeSummarizer struct {
	// summary is the text of the returned summary. Empty means "decline",
	// which makes SummarizeEvents return no content.
	summary string
	// err, when set, is returned instead of a summary.
	err error

	// windows records the event IDs of every window passed in, in call order.
	windows [][]string
	calls   int
}

func (f *fakeSummarizer) SummarizeEvents(_ context.Context, events []*session.Event) (*genai.Content, *genai.GenerateContentResponseUsageMetadata, error) {
	f.calls++
	f.windows = append(f.windows, ids(events))
	if f.err != nil {
		return nil, nil, f.err
	}
	if f.summary == "" || len(events) == 0 {
		return nil, nil, nil
	}
	return &genai.Content{Parts: []*genai.Part{{Text: f.summary}}}, nil, nil
}

// fakeModel returns canned responses and records the requests it received.
type fakeModel struct {
	responses []*model.LLMResponse
	requests  []*model.LLMRequest
	err       error
}

func (m *fakeModel) Name() string { return "fake-model" }

func (m *fakeModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.requests = append(m.requests, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.err != nil {
			yield(nil, m.err)
			return
		}
		for _, r := range m.responses {
			if !yield(r, nil) {
				return
			}
		}
	}
}

var _ model.LLM = (*fakeModel)(nil)

// slidingWindowStored runs SlidingWindow and closes the span the way a caller
// that stored the summary does, which is what most tests mean.
func slidingWindowStored(ctx context.Context, cfg *compaction.Config, sess session.Session) (*session.Event, error) {
	ev, finish, err := SlidingWindow(ctx, cfg, sess, "")
	finish(err, "")
	return ev, err
}

// tailRetentionStored is slidingWindowStored for the tail-retention strategy.
func excl(invocationID string, ts int) session.EventRef {
	return session.EventRef{InvocationID: invocationID, Timestamp: at(ts)}
}
