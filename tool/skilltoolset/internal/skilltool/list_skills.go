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
	"html"
	"strings"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/skilltoolset/skill"
)

// ListSkillsArgs represents the input for ListSkills tool.
type ListSkillsArgs struct{}

// ListSkillsResult represents the output for ListSkills tool.
type ListSkillsResult struct {
	SkillsXML string `json:"skills"`
}

// ListSkills creates a tool.Tool to list available skills.
func ListSkills(source skill.Source) (tool.Tool, error) {
	return functiontool.New(
		functiontool.Config{
			Name:        "list_skills",
			Description: "Lists all available skills with their names and descriptions.",
		},
		func(ctx tool.Context, args ListSkillsArgs) (*ListSkillsResult, error) {
			return listSkills(ctx, args, source)
		},
	)
}

func listSkills(ctx tool.Context, _ ListSkillsArgs, source skill.Source) (*ListSkillsResult, error) {
	skills, err := source.ListFrontmatters(ctx)
	if err != nil {
		return nil, err
	}
	return &ListSkillsResult{SkillsXML: SkillsToXML(skills)}, nil
}

func SkillsToXML(frontmatters []*skill.Frontmatter) string {
	var sb strings.Builder
	sb.WriteString("<available_skills>\n")
	for _, fm := range frontmatters {
		sb.WriteString("<skill>\n")
		sb.WriteString("<name>\n")
		sb.WriteString(html.EscapeString(fm.Name))
		sb.WriteString("\n</name>\n")
		sb.WriteString("<description>\n")
		sb.WriteString(html.EscapeString(fm.Description))
		sb.WriteString("\n</description>\n")
		sb.WriteString("</skill>\n")
	}
	sb.WriteString("</available_skills>")
	return sb.String()
}
