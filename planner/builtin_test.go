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
	req := &struct {
		Config *genai.GenerateContentConfig
	}{
		Config: nil,
	}

	if req.Config != nil && req.Config.ThinkingConfig != nil {
		t.Error("Expected ThinkingConfig to not be set")
	}

	thinkingConfig := &genai.ThinkingConfig{IncludeThoughts: true}
	req = &struct {
		Config *genai.GenerateContentConfig
	}{
		Config: nil,
	}

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.ThinkingConfig = thinkingConfig

	if req.Config == nil {
		t.Error("Expected Config to be set")
	} else if req.Config.ThinkingConfig == nil {
		t.Error("Expected ThinkingConfig to be set")
	} else if req.Config.ThinkingConfig.IncludeThoughts != true {
		t.Errorf("Expected IncludeThoughts true, got %v", req.Config.ThinkingConfig.IncludeThoughts)
	}

	req = &struct {
		Config *genai.GenerateContentConfig
	}{
		Config: &genai.GenerateContentConfig{},
	}
	req.Config.ThinkingConfig = thinkingConfig

	if req.Config.ThinkingConfig == nil {
		t.Error("Expected ThinkingConfig to be set")
	} else if req.Config.ThinkingConfig.IncludeThoughts != true {
		t.Errorf("Expected IncludeThoughts true, got %v", req.Config.ThinkingConfig.IncludeThoughts)
	}
}
