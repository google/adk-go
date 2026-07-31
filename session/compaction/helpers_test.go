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

package compaction

import (
	"context"
	"iter"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// epoch anchors the synthetic timestamps used across these tests. Tests express
// times as small integers via at(); only their relative order matters.
var epoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// at returns a deterministic timestamp n seconds after the test epoch.
func at(n int) time.Time { return epoch.Add(time.Duration(n) * time.Second) }

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

// compactionEvent builds a stored compaction event covering [start, end].
func compactionEvent(id string, ts, start, end int, summary string) *session.Event {
	return &session.Event{
		ID:        id,
		Timestamp: at(ts),
		Author:    "user",
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   at(start),
				EndTimestamp:     at(end),
				CompactedContent: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: summary}}},
			},
		},
	}
}
