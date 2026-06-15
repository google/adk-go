// Copyright 2025 Google LLC
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

// FINDING H2 — mergeParallelFunctionResponseEvents nil-dereferences on nil entries.
//
// Bug: parallel function-call handling can leave nil entries in the events
// slice (e.g. for long-running/deferred tools that early-return). The merge
// loop skips nil entries when collecting parts, but then unconditionally does
// `ev := events[0]; ev.LLMResponse = ...; ev.Actions = *actions`. When
// events[0] is nil this panics with a nil pointer dereference; when every
// entry is nil, `actions` is also nil so `*actions` panics as well.
//
// Expected: the merge tolerates nil entries (skipping them) and never panics.
//
// This test currently FAILS, demonstrating the bug.

package llminternal

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func vbugH2ValidEvent() *session.Event {
	return &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: "hi"}},
			},
		},
	}
}

func TestVbugH2_MergeParallelFunctionResponseEvents_NilEntries(t *testing.T) {
	cases := []struct {
		name   string
		events []*session.Event
	}{
		{name: "leading nil then valid", events: []*session.Event{nil, vbugH2ValidEvent()}},
		{name: "all nil", events: []*session.Event{nil, nil}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("mergeParallelFunctionResponseEvents panicked on nil entries: %v", r)
				}
			}()
			_, _ = mergeParallelFunctionResponseEvents(tc.events)
		})
	}
}
