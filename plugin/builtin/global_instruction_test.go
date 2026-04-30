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

package builtin_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin/builtin"
)

func TestGlobalInstruction_ConstructorSucceedsForStaticText(t *testing.T) {
	p, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{Instruction: "Be terse."})
	if err != nil {
		t.Fatalf("GlobalInstruction: %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil plugin")
	}
}

func TestGlobalInstruction_ConstructorSucceedsForFunc(t *testing.T) {
	p, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{
		InstructionFunc: func(agent.CallbackContext) (string, error) { return "dynamic", nil },
	})
	if err != nil {
		t.Fatalf("GlobalInstruction: %v", err)
	}
	if p == nil {
		t.Fatal("expected non-nil plugin")
	}
}

func TestGlobalInstruction_DefaultName(t *testing.T) {
	p, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{Instruction: "x"})
	if err != nil {
		t.Fatalf("GlobalInstruction: %v", err)
	}
	if p.Name() != "global_instruction" {
		t.Errorf("Name = %q, want global_instruction", p.Name())
	}
}

func TestGlobalInstruction_OverrideName(t *testing.T) {
	p, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{
		Name:        "custom",
		Instruction: "x",
	})
	if err != nil {
		t.Fatalf("GlobalInstruction: %v", err)
	}
	if p.Name() != "custom" {
		t.Errorf("Name = %q, want custom", p.Name())
	}
}

func TestGlobalInstruction_AppliesToEmptyRequest(t *testing.T) {
	p, _ := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{Instruction: "Be terse."})
	req := &model.LLMRequest{}
	resp, err := p.BeforeModelCallback()(nil, req)
	if err != nil || resp != nil {
		t.Fatalf("BeforeModel = (%v, %v), want (nil, nil)", resp, err)
	}
	if req.Config == nil || req.Config.SystemInstruction == nil {
		t.Fatal("SystemInstruction should be set")
	}
	if got := req.Config.SystemInstruction.Parts[0].Text; got != "Be terse." {
		t.Errorf("SystemInstruction[0] = %q, want Be terse.", got)
	}
}

func TestGlobalInstruction_PrependsToExisting(t *testing.T) {
	p, _ := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{Instruction: "Be terse."})
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "Use English."}}},
		},
	}
	if _, err := p.BeforeModelCallback()(nil, req); err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	parts := req.Config.SystemInstruction.Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(parts))
	}
	if parts[0].Text != "Be terse." || parts[1].Text != "Use English." {
		t.Errorf("parts = %q,%q", parts[0].Text, parts[1].Text)
	}
}

func TestGlobalInstruction_EmptyInstructionIsNoop(t *testing.T) {
	p, _ := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{Instruction: ""})
	req := &model.LLMRequest{}
	if _, err := p.BeforeModelCallback()(nil, req); err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if req.Config != nil && req.Config.SystemInstruction != nil {
		t.Error("expected no SystemInstruction injection for empty instruction")
	}
}

func TestGlobalInstruction_DynamicProvider(t *testing.T) {
	p, _ := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{
		InstructionFunc: func(agent.CallbackContext) (string, error) {
			return "dynamic text", nil
		},
	})
	req := &model.LLMRequest{}
	if _, err := p.BeforeModelCallback()(nil, req); err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if got := req.Config.SystemInstruction.Parts[0].Text; got != "dynamic text" {
		t.Errorf("got %q, want dynamic text", got)
	}
}
