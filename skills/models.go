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

package skills

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

// Skill from https://github.com/agentskills/agentskills
// https://agentskills.io/specification

var validNameRegex = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// Frontmatter L1 skill content: metadata parsed from SKILL.md frontmatter for skill discovery.
// Attributes:
//
//	name: Skill name in kebab-case (required).
//	description: What the skill does and when the model should use it (required).
//	license: License for the skill (optional).
//	compatibility: Compatibility information for the skill (optional).
//	allowed_tools: Tool patterns the skill requires (optional, experimental).
//		Accepts both ``allowed_tools`` and the YAML-friendly ``allowed-tools`` key.
//	metadata: Key-value pairs for client-specific properties (defaults to empty dict).
type Frontmatter struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license,omitempty"`
	Compatibility string            `yaml:"compatibility,omitempty"`
	AllowedTools  string            `yaml:"allowed_tools,omitempty"`
	Metadata      map[string]string `yaml:"metadata,omitempty"`
}

func (f *Frontmatter) UnmarshalYAML(node *yaml.Node) error {
	type tempFrontmatter struct {
		Name          string            `yaml:"name"`
		Description   string            `yaml:"description"`
		License       string            `yaml:"license,omitempty"`
		Compatibility string            `yaml:"compatibility,omitempty"`
		AllowedTools  string            `yaml:"allowed_tools,omitempty"`
		AllowedTools2 string            `yaml:"allowed-tools,omitempty"`
		Metadata      map[string]string `yaml:"metadata,omitempty"`
	}

	var temp tempFrontmatter
	if err := node.Decode(&temp); err != nil {
		return err
	}

	if temp.AllowedTools != "" {
		f.AllowedTools = temp.AllowedTools
	} else {
		f.AllowedTools = temp.AllowedTools2
	}

	f.Name = temp.Name
	f.Description = temp.Description
	f.License = temp.License
	f.Compatibility = temp.Compatibility
	f.Metadata = temp.Metadata

	return nil
}

func (f *Frontmatter) Validate() error {
	if len(f.Name) < 1 || len(f.Name) > 64 {
		return fmt.Errorf("name must be 1-64 characters")
	}
	if !validNameRegex.MatchString(f.Name) {
		return fmt.Errorf("name must be lowercase kebab-case (a-z, 0-9, hyphens), with no leading, trailing, or consecutive hyphens")
	}

	// Note: The rule "Must match the parent directory name" cannot be validated here
	// as the Skill struct does not contain information about its file path.
	// This validation should be performed by the caller when loading the skill.

	if strings.TrimSpace(f.Description) == "" {
		return fmt.Errorf("description must not be empty")
	}
	if len(f.Description) > 1024 {
		return fmt.Errorf("description must be at most 1024 characters")
	}

	if f.Compatibility != "" && len(f.Compatibility) > 500 {
		return fmt.Errorf("compatibility must be 1-500 characters")
	}

	return nil
}

func (f *Frontmatter) SkillPromptEntry() string {
	return fmt.Sprintf("- name: %s, description: %s", f.Name, f.Description)
}

func (f *Frontmatter) ToJsonString() string {
	bytes, _ := json.Marshal(f)
	return string(bytes)
}

type Script struct {
	Src string
}

func (s *Script) String() string {
	return s.Src
}

// Resources L3 skill content: additional instructions, assets, and scripts, loaded as needed.
// Attributes:
//
//	references: Additional markdown files with instructions, workflows, or guidance.
//	assets: Resource materials like database schemas, API documentation, templates, or examples.
//	scripts: Executable scripts that can be run via bash.
type Resources struct {
	References map[string]string
	Assets     map[string]string
	Scripts    map[string]*Script
}

func (r *Resources) GetReference(referenceId string) (string, bool) {
	if r.References == nil {
		return "", false
	}
	v, ok := r.References[referenceId]
	return v, ok
}

func (r *Resources) GetAsset(assetId string) (string, bool) {
	if r.Assets == nil {
		return "", false
	}
	v, ok := r.Assets[assetId]
	return v, ok
}

func (r *Resources) GetScript(scriptId string) (*Script, bool) {
	if r.Scripts == nil {
		return nil, false
	}
	v, ok := r.Scripts[scriptId]
	return v, ok
}

func (r *Resources) ListReferences() []string {
	out := make([]string, 0, len(r.References))
	for k := range r.References {
		out = append(out, k)
	}
	return out
}

func (r *Resources) ListAssets() []string {
	out := make([]string, 0, len(r.Assets))
	for k := range r.Assets {
		out = append(out, k)
	}
	return out
}

func (r *Resources) ListScripts() []string {
	out := make([]string, 0, len(r.Scripts))
	for k := range r.Scripts {
		out = append(out, k)
	}
	return out
}

type Skill struct {
	Frontmatter  *Frontmatter
	Instructions string
	Resources    *Resources
	SkillMDPath  string
}

func (s *Skill) Name() string {
	return s.Frontmatter.Name
}

func (s *Skill) Description() string {
	return s.Frontmatter.Description
}

func (s *Skill) GetSkillPath() string {
	return filepath.Dir(s.SkillMDPath)
}

func (s *Skill) WriteSkill(path string) error {
	if path == "" {
		return fmt.Errorf("path must not be empty")
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("path invalid: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("path must be a directory")
	}

	if s.Frontmatter == nil || s.Frontmatter.Name == "" {
		return fmt.Errorf("skill name is missing")
	}

	skillDir := filepath.Join(path, s.Frontmatter.Name)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		return fmt.Errorf("failed to create skill directory: %w", err)
	}

	// Write SKILL.md
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(s.Instructions), 0o644); err != nil {
		return fmt.Errorf("failed to write SKILL.md: %w", err)
	}

	if s.Resources == nil {
		return nil
	}

	// References
	if len(s.Resources.References) > 0 {
		refDir := filepath.Join(skillDir, "references")
		if err := os.MkdirAll(refDir, 0o755); err != nil {
			return fmt.Errorf("failed to create references directory: %w", err)
		}
		for name, content := range s.Resources.References {
			targetPath := filepath.Join(refDir, name)
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return fmt.Errorf("failed to create directory for reference %s: %w", name, err)
			}
			if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
				return fmt.Errorf("failed to write reference %s: %w", name, err)
			}
		}
	}

	// Assets
	if len(s.Resources.Assets) > 0 {
		assetsDir := filepath.Join(skillDir, "assets")
		if err := os.MkdirAll(assetsDir, 0o755); err != nil {
			return fmt.Errorf("failed to create assets directory: %w", err)
		}
		for name, content := range s.Resources.Assets {
			targetPath := filepath.Join(assetsDir, name)
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return fmt.Errorf("failed to create directory for asset %s: %w", name, err)
			}
			if err := os.WriteFile(targetPath, []byte(content), 0o644); err != nil {
				return fmt.Errorf("failed to write asset %s: %w", name, err)
			}
		}
	}

	// Scripts
	if len(s.Resources.Scripts) > 0 {
		scriptsDir := filepath.Join(skillDir, "scripts")
		if err := os.MkdirAll(scriptsDir, 0o755); err != nil {
			return fmt.Errorf("failed to create scripts directory: %w", err)
		}
		for name, script := range s.Resources.Scripts {
			if script == nil {
				continue
			}
			targetPath := filepath.Join(scriptsDir, name)
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return fmt.Errorf("failed to create directory for script %s: %w", name, err)
			}
			if err := os.WriteFile(targetPath, []byte(script.Src), 0o755); err != nil {
				return fmt.Errorf("failed to write script %s: %w", name, err)
			}
		}
	}

	s.SkillMDPath = filepath.Join(skillDir, "SKILL.md")

	return nil
}
