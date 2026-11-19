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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestMergeParallelFunctionResponseEvents_TransferReason(t *testing.T) {
	// transfer is one function response event's transfer target and reason;
	// an empty target means the call made no transfer.
	type transfer struct{ target, reason string }
	tests := []struct {
		name string
		in   []transfer
		want transfer
	}{
		{name: "later reason wins", in: []transfer{{"A", "r1"}, {"B", "r2"}}, want: transfer{"B", "r2"}},
		{name: "reason of the later transfer only", in: []transfer{{"A", ""}, {"B", "r"}}, want: transfer{"B", "r"}},
		{name: "a call without a transfer keeps the reason", in: []transfer{{"A", "r"}, {"", ""}}, want: transfer{"A", "r"}},
		{name: "a reason without a transfer is merged", in: []transfer{{"A", ""}, {"", "r"}}, want: transfer{"A", "r"}},
		{name: "single event", in: []transfer{{"A", "r"}}, want: transfer{"A", "r"}},
		// adk-python merges each field's last non-None value on its own
		// (flows/llm_flows/tools/_batch_executor.py), so a reason can end up
		// beside another call's target.
		{name: "earlier reason stays beside a later target without one", in: []transfer{{"A", "r"}, {"B", ""}}, want: transfer{"B", "r"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var events []*session.Event
			for _, tr := range tc.in {
				events = append(events, &session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{Text: "response"}}, "user")},
					Actions:     session.EventActions{TransferToAgent: tr.target, TransferReason: tr.reason},
				})
			}
			got, err := mergeParallelFunctionResponseEvents(events)
			if err != nil {
				t.Fatalf("mergeParallelFunctionResponseEvents() error = %v", err)
			}
			if gotTransfer := (transfer{got.Actions.TransferToAgent, got.Actions.TransferReason}); gotTransfer != tc.want {
				t.Errorf("merged (TransferToAgent, TransferReason) = %+v, want %+v", gotTransfer, tc.want)
			}
		})
	}
}
