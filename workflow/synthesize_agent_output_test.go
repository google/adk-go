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

package workflow

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// LRT-bearing events are "final" by IsFinalResponse() but represent
// a pause, not a completion: promoting their text would cache a
// transient string and short-circuit the re-run on resume.
func TestSynthesizeAgentOutput_SkipsLongRunningToolEvent(t *testing.T) {
	t.Parallel()

	// Non-empty text so the assertion catches a regression where the
	// LRT guard is removed (the text would otherwise be promoted).
	ev := &session.Event{
		LongRunningToolIDs: []string{"fc-pending"},
	}
	ev.LLMResponse.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "Awaiting your approval…"}},
	}

	synthesizeAgentOutput(ev)

	if ev.Output != nil {
		t.Errorf("Output = %v, want nil (LRT-bearing events must not be promoted)", ev.Output)
	}
	if ev.NodeInfo != nil && ev.NodeInfo.MessageAsOutput {
		t.Errorf("NodeInfo.MessageAsOutput = true, want false/unset for LRT pauses")
	}
}

// Positive control: plain model text without LRT gets promoted.
// Pins the LRT guard as a targeted skip, not a global behaviour change.
func TestSynthesizeAgentOutput_PromotesPlainModelText(t *testing.T) {
	t.Parallel()

	ev := &session.Event{}
	ev.LLMResponse.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: "hello"}},
	}

	synthesizeAgentOutput(ev)

	if got, want := ev.Output, "hello"; got != want {
		t.Errorf("Output = %v, want %q", ev.Output, want)
	}
	if ev.NodeInfo == nil || !ev.NodeInfo.MessageAsOutput {
		t.Errorf("NodeInfo.MessageAsOutput = %v, want true", ev.NodeInfo)
	}
}
