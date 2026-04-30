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

// Package skill is the data model for Agent Skills. Mirrors a subset of
// adk-python's google.adk.skills.
//
// A Skill carries three layers of content:
//
//   - L1 Frontmatter: metadata used for skill discovery (name +
//     description). Loaded eagerly so the model can list available
//     skills.
//   - L2 Instructions: markdown body executed when the model triggers
//     the skill. Loaded on-demand.
//   - L3 Resources: additional reference docs, assets, and scripts.
//     Loaded as the instructions reference them.
//
// FormatAsXML renders a slice of skills (or frontmatters) as the
// <available_skills>…</available_skills> block adk-python emits into
// system instructions.
package skill

import (
	"fmt"
	"html"
	"regexp"
	"strings"
	"unicode/utf8"
)

// kebabPattern matches lowercase kebab-case names (a-z, 0-9, single
// hyphens between segments).
var kebabPattern = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// snakeOrKebabPattern matches kebab-case OR snake_case names but not
// mixed.
var snakeOrKebabPattern = regexp.MustCompile(`^([a-z0-9]+(-[a-z0-9]+)*|[a-z0-9]+(_[a-z0-9]+)*)$`)

// Frontmatter is the metadata for a skill. Mirrors models.Frontmatter.
type Frontmatter struct {
	Name          string
	Description   string
	License       string
	Compatibility string
	// AllowedTools is a space-delimited list of tool names pre-approved
	// to run when this skill is active.
	AllowedTools string
	Metadata     map[string]any
}

// SnakeCaseAllowed is a per-process toggle equivalent to
// FeatureName.SNAKE_CASE_SKILL_NAME in adk-python. When true, names may
// be snake_case in addition to kebab-case.
var SnakeCaseAllowed = false

// Validate checks that the frontmatter satisfies the skill spec
// (name format, length caps, non-empty description). Mirrors
// adk-python's _validate_name / _validate_description / _validate_compatibility.
func (f *Frontmatter) Validate() error {
	if utf8.RuneCountInString(f.Name) > 64 {
		return fmt.Errorf("skill: name must be at most 64 characters")
	}
	pattern := kebabPattern
	msg := "skill: name must be lowercase kebab-case (a-z, 0-9, hyphens), no leading/trailing/consecutive delimiters"
	if SnakeCaseAllowed {
		pattern = snakeOrKebabPattern
		msg = "skill: name must be lowercase kebab-case or snake_case (a-z, 0-9, hyphens or underscores), no mixed delimiters"
	}
	if !pattern.MatchString(f.Name) {
		return fmt.Errorf("%s; got %q", msg, f.Name)
	}
	if f.Description == "" {
		return fmt.Errorf("skill: description must not be empty")
	}
	if utf8.RuneCountInString(f.Description) > 1024 {
		return fmt.Errorf("skill: description must be at most 1024 characters")
	}
	if utf8.RuneCountInString(f.Compatibility) > 500 {
		return fmt.Errorf("skill: compatibility must be at most 500 characters")
	}
	return nil
}

// Resources holds the L3 content of a skill: additional reference
// documents, asset files, and scripts. Maps are keyed by virtual path or
// name. Mirrors models.Resources.
type Resources struct {
	References map[string][]byte
	Assets     map[string][]byte
	Scripts    map[string]string
}

// GetReference returns the content of references[id], if any.
func (r *Resources) GetReference(id string) ([]byte, bool) {
	if r.References == nil {
		return nil, false
	}
	v, ok := r.References[id]
	return v, ok
}

// GetAsset returns the content of assets[id], if any.
func (r *Resources) GetAsset(id string) ([]byte, bool) {
	if r.Assets == nil {
		return nil, false
	}
	v, ok := r.Assets[id]
	return v, ok
}

// GetScript returns the content of scripts[id], if any.
func (r *Resources) GetScript(id string) (string, bool) {
	if r.Scripts == nil {
		return "", false
	}
	v, ok := r.Scripts[id]
	return v, ok
}

// Skill bundles frontmatter (L1), instructions (L2), and resources (L3).
type Skill struct {
	Frontmatter  Frontmatter
	Instructions string
	Resources    Resources
}

// Name returns the skill's name (convenience accessor).
func (s *Skill) Name() string { return s.Frontmatter.Name }

// Description returns the skill's description.
func (s *Skill) Description() string { return s.Frontmatter.Description }

// FormatAsXML renders skills as the <available_skills>…</available_skills>
// block. The function accepts both *Skill and *Frontmatter values.
//
// Mirrors prompt.format_skills_as_xml.
func FormatAsXML(skills []*Skill) string {
	if len(skills) == 0 {
		return "<available_skills>\n</available_skills>"
	}
	var b strings.Builder
	b.WriteString("<available_skills>")
	for _, s := range skills {
		b.WriteString("\n<skill>\n<name>\n")
		b.WriteString(html.EscapeString(s.Frontmatter.Name))
		b.WriteString("\n</name>\n<description>\n")
		b.WriteString(html.EscapeString(s.Frontmatter.Description))
		b.WriteString("\n</description>\n</skill>")
	}
	b.WriteString("\n</available_skills>")
	return b.String()
}
