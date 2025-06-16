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

package agent_test

import (
	"flag"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/adk-go/agent"
	"github.com/google/adk-go/model"
)

var manual = flag.Bool("manual", false, "Run manual tests that require a valid GenAI API key")

func TestLLMAgent(t *testing.T) {
	if !*manual {
		// TODO(hakim): remove this after making this test deterministic.
		t.Skip("Skipping manual test. Set -manual flag to run it.")
	}
	ctx := t.Context()
	m, err := model.NewGeminiModel(ctx, "gemini-2.0-flash", nil)
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	a := &agent.LLMAgent{
		AgentName:         "hello_world_agent",
		AgentDescription:  "hello world agent",
		Model:             m,
		Instruction:       "Roll the dice and report only the result.",
		GlobalInstruction: "Answer as precisely as possible.",

		// TODO: set tools, planner.

		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
	}
	stream, err := a.Run(ctx, &adk.InvocationContext{
		InvocationID: "12345",
		Agent:        a,
	})
	if err != nil {
		t.Fatalf("failed to run agent: %v", err)
	}
	n := 0
	for ev, err := range stream {
		n++
		if err != nil {
			t.Errorf("unexpectd error = (%v, %v)", ev.LLMResponse, err)
		}
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil || len(ev.LLMResponse.Content.Parts) == 0 || ev.LLMResponse.Content.Parts[0].Text == "" {
			t.Errorf("unexpected response = %v", ev.LLMResponse)
		}
	}
	if n == 0 {
		t.Error("no events received")
	}
}
