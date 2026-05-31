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

package skilltool

import (
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/skilltoolset/skill"
)

// SearchSkillsArgs represents the input to search for a skill in the registry.
type SearchSkillArgs struct {
	Query string `json:"query" jsonschema:"Semantic or keyword search query."`
}

// SearchSkillsResult represents the output for SearchSkills tool.
type SearchSkillsResult struct {
	Frontmatters []FrontmatterJSON `json:"skills"`
}

// SearchSkills creates a tool.Tool to search for skills.
func SearchSkills(source skill.Source) (tool.Tool, error) {
	return functiontool.New(
		functiontool.Config{
			Name:        "search_skills",
			Description: "Searches for relevant skills in the registry based on a semantic or keyword query.",
		},
		func(ctx tool.Context, args SearchSkillArgs) (*SearchSkillsResult, error) {
			return searchSkills(ctx, args, source)
		},
	)
}

func searchSkills(ctx tool.Context, args SearchSkillArgs, source skill.Source) (*SearchSkillsResult, error) {
	skills, err := source.Search(ctx, args.Query)
	if err != nil {
		return nil, err
	}
	var frontmatters []FrontmatterJSON
	for _, skill := range skills {
		frontmatters = append(frontmatters, FrontmatterJSON{
			Name:        skill.Name,
			Description: skill.Description,
		})
	}
	return &SearchSkillsResult{Frontmatters: frontmatters}, nil
}
