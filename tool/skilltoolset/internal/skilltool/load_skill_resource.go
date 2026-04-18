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
	"fmt"
	"io"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/skilltoolset/skill"
)

const maxResourceSize = 10 * 1024 * 1024 // 10 MiB

// LoadSkillResourceArgs represents the input for retrieving a resource out of a skill's resources.
type LoadSkillResourceArgs struct {
	SkillName    string `json:"skill_name" jsonschema:"The name of the skill."`
	ResourcePath string `json:"resource_path" jsonschema:"The relative path to the resource (e.g., 'references/my_doc.md', 'assets/template.txt', or 'scripts/setup.sh')."`
}

// LoadSkillResourceResult encapsulates the resource content.
type LoadSkillResourceResult struct {
	SkillName string `json:"skill_name,omitempty"`
	Path      string `json:"path,omitempty"`
	Content   string `json:"content,omitempty"`
}

// LoadSkillResource creates a tool.Tool to load a resource file for a skill.
func LoadSkillResource(source skill.Source) (tool.Tool, error) {
	return functiontool.New(
		functiontool.Config{
			Name:        "load_skill_resource",
			Description: "Loads a resource file (e.g., from references/ or assets/) associated with the specified skill.",
		},
		func(ctx tool.Context, args LoadSkillResourceArgs) (*LoadSkillResourceResult, error) {
			return loadSkillResource(ctx, args, source)
		},
	)
}

func loadSkillResource(ctx tool.Context, args LoadSkillResourceArgs, source skill.Source) (*LoadSkillResourceResult, error) {
	if args.SkillName == "" {
		return nil, fmt.Errorf("skill name is required to load a resource")
	}
	if args.ResourcePath == "" {
		return nil, fmt.Errorf("resource path is required to load a resource for skill %q", args.SkillName)
	}
	reader, err := source.LoadResource(ctx, args.SkillName, args.ResourcePath)
	if err != nil {
		return nil, fmt.Errorf("load resource '%s' from skill '%s': %w", args.ResourcePath, args.SkillName, err)
	}
	defer func() {
		_ = reader.Close()
	}()
	content, err := io.ReadAll(io.LimitReader(reader, maxResourceSize+1))
	if err != nil {
		return nil, fmt.Errorf("read resource '%s' from skill '%s': %w", args.ResourcePath, args.SkillName, err)
	}
	if int64(len(content)) > maxResourceSize {
		return nil, fmt.Errorf("resource '%s' from skill '%s' is too large (limit: %d bytes)", args.ResourcePath, args.SkillName, maxResourceSize)
	}
	return &LoadSkillResourceResult{
		SkillName: args.SkillName,
		Path:      args.ResourcePath,
		Content:   string(content),
	}, nil
}
