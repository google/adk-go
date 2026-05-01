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

package skillregistry

import (
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/skill"
)

// SkillsInstructionPluginConfig configures NewSkillsInstructionPlugin.
type SkillsInstructionPluginConfig struct {
	// Name overrides the default plugin name "skills_instruction".
	Name string

	// Registry is the source of skills. Required.
	Registry *Registry

	// Header is the optional preamble appended before each loaded
	// skill's instructions. Mirrors Python's "<skill name=...>" wrapping
	// when adk-python injects skill content.
	Header func(s *skill.Skill) string
}

// NewSkillsInstructionPlugin returns a plugin that, on every model call,
// appends the instructions of every currently-loaded skill to the LLM
// request's system instruction. Pair with the list_skills / load_skill
// builtin tools so the agent can dynamically pull only the guidance it
// needs without paying the context cost of every skill upfront.
//
// The plugin is idempotent: skill instructions are appended fresh each
// turn from the current state, so unloading a skill (or removing it
// from state) takes effect on the next model call.
func NewSkillsInstructionPlugin(cfg SkillsInstructionPluginConfig) (*plugin.Plugin, error) {
	if cfg.Registry == nil {
		return nil, fmt.Errorf("skillregistry: SkillsInstructionPlugin: Registry is required")
	}
	name := cfg.Name
	if name == "" {
		name = "skills_instruction"
	}
	header := cfg.Header
	if header == nil {
		header = func(s *skill.Skill) string {
			return fmt.Sprintf("\n\n# Skill: %s\n%s\n",
				s.Frontmatter.Name, s.Frontmatter.Description)
		}
	}

	before := func(cctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
		state := cctx.ReadonlyState()
		if state == nil {
			return nil, nil
		}
		skills := LoadedSkills(cfg.Registry, state)
		if len(skills) == 0 {
			return nil, nil
		}
		var b strings.Builder
		for _, s := range skills {
			b.WriteString(header(s))
			b.WriteString(s.Instructions)
		}
		injected := b.String()
		if injected == "" {
			return nil, nil
		}
		if req.Config == nil {
			req.Config = &genai.GenerateContentConfig{}
		}
		if req.Config.SystemInstruction == nil {
			req.Config.SystemInstruction = &genai.Content{
				Parts: []*genai.Part{{Text: injected}},
			}
		} else {
			req.Config.SystemInstruction.Parts = append(
				req.Config.SystemInstruction.Parts,
				&genai.Part{Text: injected},
			)
		}
		return nil, nil
	}

	return plugin.New(plugin.Config{
		Name:                name,
		BeforeModelCallback: before,
	})
}
