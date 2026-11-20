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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// partialOnlyModel streams one partial chunk and then ends without a final
// aggregated response, which the model.LLM interface permits.
type partialOnlyModel struct {
	model.LLM
}

func (partialOnlyModel) Name() string { return "partial-only-mock" }

func (partialOnlyModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText("hel", genai.RoleModel),
			Partial: true,
		}, nil)
	}
}

// TestRun_StreamEndingOnPartialDoesNotRace checks that the runner does not
// write to a partial event the producing node still reads. Partial events are
// handed over without waiting for the consumer, so any write to them races the
// flow's own read of the last event. Run with -race.
func TestRun_StreamEndingOnPartialDoesNotRace(t *testing.T) {
	a, err := llmagent.New(llmagent.Config{Name: "streamer", Model: partialOnlyModel{}})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	r, err := New(Config{
		AppName:           "testApp",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() failed: %v", err)
	}
	msg := genai.NewContentFromText("hi", genai.RoleUser)
	cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}
	for _, err := range r.Run(t.Context(), "u", "s", msg, cfg) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}
}
