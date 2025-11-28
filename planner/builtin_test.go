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

package planner

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestBuiltInPlanner_New(t *testing.T) {
	p := NewBuiltInPlanner(nil)
	if p == nil {
		t.Fatal("NewBuiltInPlanner returned nil")
	}
	if p.ThinkingConfig != nil {
		t.Errorf("Expected ThinkingConfig to be nil, got %+v", p.ThinkingConfig)
	}

	config := &genai.ThinkingConfig{IncludeThoughts: true}
	p = NewBuiltInPlanner(config)
	if p == nil {
		t.Fatal("NewBuiltInPlanner returned nil")
	}
	if p.ThinkingConfig == nil {
		t.Fatal("Expected ThinkingConfig to be set, got nil")
	}
	if p.ThinkingConfig.IncludeThoughts != true {
		t.Errorf("Expected IncludeThoughts true, got %v", p.ThinkingConfig.IncludeThoughts)
	}
}

func TestBuiltInPlanner_BuildPlanningInstruction(t *testing.T) {
	p := NewBuiltInPlanner(&genai.ThinkingConfig{
		IncludeThoughts: true,
	})

	instruction := p.BuildPlanningInstruction(nil, nil)
	if instruction != "" {
		t.Errorf("BuiltInPlanner should return empty instruction, got: %q", instruction)
	}
}

func TestBuiltInPlanner_ProcessPlanningResponse(t *testing.T) {
	p := NewBuiltInPlanner(&genai.ThinkingConfig{
		IncludeThoughts: true,
	})

	parts := []*genai.Part{
		{Text: "Hello world"},
		{Text: "Another part"},
	}

	result := p.ProcessPlanningResponse(nil, parts)
	if result != nil {
		t.Errorf("BuiltInPlanner should return nil for response processing, got: %+v", result)
	}
}

func TestBuiltInPlanner_ApplyThinkingConfig(t *testing.T) {
	thinkingConfig := &genai.ThinkingConfig{IncludeThoughts: true}
	p := NewBuiltInPlanner(thinkingConfig)

	t.Run("creates config when missing", func(t *testing.T) {
		req := &model.LLMRequest{Config: nil}

		p.ApplyThinkingConfig(req)

		if req.Config == nil {
			t.Fatal("req.Config should not be nil after ApplyThinkingConfig")
		}
		if req.Config.ThinkingConfig == nil {
			t.Fatal("req.Config.ThinkingConfig should not be nil")
		}
		if !req.Config.ThinkingConfig.IncludeThoughts {
			t.Errorf("Expected IncludeThoughts to be true, got false")
		}
	})

	t.Run("updates existing config", func(t *testing.T) {
		req := &model.LLMRequest{Config: &genai.GenerateContentConfig{}}

		p.ApplyThinkingConfig(req)

		if req.Config.ThinkingConfig == nil {
			t.Fatal("req.Config.ThinkingConfig should not be nil")
		}
		if !req.Config.ThinkingConfig.IncludeThoughts {
			t.Errorf("Expected IncludeThoughts to be true, got false")
		}
	})

	t.Run("no-op when planner config is nil", func(t *testing.T) {
		req := &model.LLMRequest{Config: nil}
		NewBuiltInPlanner(nil).ApplyThinkingConfig(req)
		if req.Config != nil {
			t.Error("req.Config should be nil when planner has nil config")
		}
	})
}
