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
	"errors"
	"fmt"

	"google.golang.org/adk/skill"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// SkillInfo is the lightweight metadata returned by list_skills.
type SkillInfo struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// ListSkillsArgs is the input shape for list_skills.
type ListSkillsArgs struct {
	Query string   `json:"query,omitempty"`
	Tags  []string `json:"tags,omitempty"`
}

// ListSkillsResult is what list_skills returns to the LLM.
type ListSkillsResult struct {
	Skills []SkillInfo `json:"skills"`
}

// NewListSkillsTool builds a tool the LLM uses to discover skills
// without paying the instruction-injection cost up front.
func NewListSkillsTool(reg *Registry) (tool.Tool, error) {
	if reg == nil {
		return nil, errors.New("skillregistry: NewListSkillsTool: registry must not be nil")
	}
	return functiontool.New[ListSkillsArgs, ListSkillsResult](
		functiontool.Config{
			Name: "list_skills",
			Description: "List the skills available in the dynamic skill registry. " +
				"Filter by an optional substring query and optional tags. " +
				"Use this BEFORE calling load_skill to discover what is available.",
		},
		func(_ tool.Context, args ListSkillsArgs) (ListSkillsResult, error) {
			fms := reg.List(Filter{Query: args.Query, Tags: args.Tags})
			out := make([]SkillInfo, 0, len(fms))
			for _, fm := range fms {
				out = append(out, SkillInfo{Name: fm.Name, Description: fm.Description})
			}
			return ListSkillsResult{Skills: out}, nil
		},
	)
}

// LoadSkillArgs is the input shape for load_skill.
type LoadSkillArgs struct {
	Name string `json:"name"`
}

// LoadSkillResult is what load_skill returns to the LLM.
type LoadSkillResult struct {
	Loaded       []string `json:"loaded"`
	Message      string   `json:"message"`
	Instructions string   `json:"instructions,omitempty"`
}

// NewLoadSkillTool builds the tool the LLM uses to activate a specific
// skill by name. Activation persists in session state (under
// StateKeyLoadedSkills) so the skill's instructions are injected into
// the next LLM request via the SkillsInstructionPlugin.
//
// The tool also returns the skill's instructions in its response so the
// model has immediate access without waiting for the next turn — this
// matches Python's behavior of treating "load" as both an activation
// signal and a content delivery.
func NewLoadSkillTool(reg *Registry) (tool.Tool, error) {
	if reg == nil {
		return nil, errors.New("skillregistry: NewLoadSkillTool: registry must not be nil")
	}
	return functiontool.New[LoadSkillArgs, LoadSkillResult](
		functiontool.Config{
			Name: "load_skill",
			Description: "Activate a skill by name from the dynamic skill registry. " +
				"After calling this, the skill's instructions become part of your context. " +
				"Discover available skills by calling list_skills first.",
		},
		func(ctx tool.Context, args LoadSkillArgs) (LoadSkillResult, error) {
			if args.Name == "" {
				return LoadSkillResult{}, errors.New("load_skill: name must not be empty")
			}
			s, err := reg.Get(args.Name)
			if err != nil {
				return LoadSkillResult{}, err
			}
			state := ctx.State()
			if state == nil {
				return LoadSkillResult{}, errors.New("load_skill: tool context has nil State")
			}
			current := LoadedNames(state)
			for _, n := range current {
				if n == args.Name {
					return LoadSkillResult{
						Loaded:       current,
						Message:      fmt.Sprintf("%q already loaded", args.Name),
						Instructions: s.Instructions,
					}, nil
				}
			}
			current = append(current, args.Name)
			if err := state.Set(StateKeyLoadedSkills, current); err != nil {
				return LoadSkillResult{}, fmt.Errorf("load_skill: persist state: %w", err)
			}
			return LoadSkillResult{
				Loaded:       current,
				Message:      fmt.Sprintf("loaded skill %q", args.Name),
				Instructions: s.Instructions,
			}, nil
		},
	)
}

// LoadedSkills returns the skill objects currently active in the given
// state. Skills missing from the registry are silently skipped.
func LoadedSkills(reg *Registry, state interface {
	Get(key string) (any, error)
}) []*skill.Skill {
	out := make([]*skill.Skill, 0)
	for _, name := range LoadedNames(state) {
		s, err := reg.Get(name)
		if err != nil {
			continue
		}
		out = append(out, s)
	}
	return out
}
