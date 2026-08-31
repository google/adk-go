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

// Package skill provides structs and functions for working with agent skills.
package skill

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"unicode"

	"gopkg.in/yaml.v3"
)

var (
	frontmatterSeparator    = []byte("---\n")
	frontmatterSeparatorWin = []byte("---\r\n")
)

// Frontmatter represents the YAML metadata at the top of a SKILL.md file.
// For more details, see https://agentskills.io/specification#frontmatter.
type Frontmatter struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license,omitempty"`
	Compatibility string            `yaml:"compatibility,omitempty"`
	Metadata      map[string]string `yaml:"metadata,omitempty"`
	AllowedTools  []string          `yaml:"allowed-tools,omitempty"`
}

// frontmatterYAML keeps the public Frontmatter representation stable while
// accepting both the specification's scalar allowed-tools value and the YAML
// sequence accepted by earlier ADK Go releases.
type frontmatterYAML struct {
	Name          string            `yaml:"name"`
	Description   string            `yaml:"description"`
	License       string            `yaml:"license,omitempty"`
	Compatibility string            `yaml:"compatibility,omitempty"`
	Metadata      map[string]string `yaml:"metadata,omitempty"`
	AllowedTools  allowedToolsYAML  `yaml:"allowed-tools,omitempty"`
}

type allowedToolsYAML []string

const allowedToolsFormat = "allowed-tools must be a whitespace- or comma-separated list of Tool or Tool(well-balanced expression)"

func (a *allowedToolsYAML) UnmarshalYAML(node *yaml.Node) error {
	switch node.Kind {
	case yaml.SequenceNode:
		var tools []string
		if err := node.Decode(&tools); err != nil {
			return err
		}
		*a = tools
		return nil
	case yaml.ScalarNode:
		if node.Tag == "!!null" {
			*a = nil
			return nil
		}
		if node.Tag != "!!str" {
			return fmt.Errorf("allowed-tools must be a string or a sequence of strings")
		}
		tools, err := splitAllowedTools(node.Value)
		if err != nil {
			return err
		}
		*a = tools
		return nil
	default:
		return fmt.Errorf("allowed-tools must be a string or a sequence of strings")
	}
}

// splitAllowedTools splits tool names or tool expressions separated by
// whitespace or commas. Separators inside balanced parentheses are preserved.
func splitAllowedTools(value string) ([]string, error) {
	var tools []string
	var current strings.Builder
	depth := 0
	flush := func() {
		if current.Len() == 0 {
			return
		}
		tools = append(tools, current.String())
		current.Reset()
	}

	for _, r := range value {
		switch {
		case r == '(':
			depth++
			current.WriteRune(r)
		case r == ')':
			if depth == 0 {
				return nil, fmt.Errorf("%s: unexpected ')'", allowedToolsFormat)
			}
			depth--
			current.WriteRune(r)
		case depth == 0 && (unicode.IsSpace(r) || r == ','):
			flush()
		default:
			current.WriteRune(r)
		}
	}
	if depth != 0 {
		return nil, fmt.Errorf("%s: unclosed '('", allowedToolsFormat)
	}
	flush()
	return tools, nil
}

func (f *frontmatterYAML) toFrontmatter() *Frontmatter {
	return &Frontmatter{
		Name:          f.Name,
		Description:   f.Description,
		License:       f.License,
		Compatibility: f.Compatibility,
		Metadata:      f.Metadata,
		AllowedTools:  []string(f.AllowedTools),
	}
}

// Parse reads and validates YAML frontmatter from a SKILL.md reader.
// On success, the reader will contain the remaining Markdown instruction.
// Use when reading the entire file into memory is expensive.
func Parse(r *bufio.Reader) (*Frontmatter, error) {
	line, err := r.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("reading first line: %w", err)
	}
	if !bytes.Equal(line, frontmatterSeparator) && !bytes.Equal(line, frontmatterSeparatorWin) {
		return nil, fmt.Errorf("invalid frontmatter separator line: must be '---' followed by a new line")
	}

	yamlBytes := bytes.Buffer{}
	for {
		line, err := r.ReadBytes('\n')
		if errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("missing closing frontmatter separator line: must be '---' followed by a new line")
		}
		if err != nil {
			return nil, fmt.Errorf("read a line of frontmatter: %w", err)
		}
		if bytes.Equal(line, frontmatterSeparator) || bytes.Equal(line, frontmatterSeparatorWin) {
			break
		}
		yamlBytes.Write(line)
	}

	fmYAML := frontmatterYAML{}
	decoder := yaml.NewDecoder(bytes.NewReader(yamlBytes.Bytes()))
	decoder.KnownFields(true)
	if err := decoder.Decode(&fmYAML); err != nil {
		return nil, fmt.Errorf("parse frontmatter: %w", err)
	}
	fm := fmYAML.toFrontmatter()
	if err := Validate(fm); err != nil {
		return nil, fmt.Errorf("invalid frontmatter: %w", err)
	}
	return fm, nil
}

// ParseBytes splits and validates YAML frontmatter from SKILL.md content bytes.
// It returns the Frontmatter and the Markdown instruction string.
// Use when the entire file content is in memory.
func ParseBytes(content []byte) (*Frontmatter, string, error) {
	reader := bufio.NewReader(bytes.NewReader(content))
	frontmatter, err := Parse(reader)
	if err != nil {
		return nil, "", fmt.Errorf("parse frontmatter: %w", err)
	}
	markdown, err := io.ReadAll(reader)
	if err != nil {
		return nil, "", fmt.Errorf("read markdown: %w", err)
	}
	return frontmatter, string(markdown), nil
}

// Validate checks if the Frontmatter struct is valid according to the
// specification. It enforces field lengths, formats, required fields, etc.
//
// Note: "skill name must match the parent directory name" check is omitted from
// this function as the Frontmatter struct lacks the file path context.
func Validate(fm *Frontmatter) error {
	if fm == nil {
		return fmt.Errorf("frontmatter cannot be nil")
	}

	if len(fm.Name) < 1 || len(fm.Name) > 64 {
		return fmt.Errorf("name must be between 1 and 64 characters long")
	}
	if strings.HasPrefix(fm.Name, "-") || strings.HasSuffix(fm.Name, "-") {
		return fmt.Errorf("name must not start or end with a hyphen")
	}
	if strings.Contains(fm.Name, "--") {
		return fmt.Errorf("name must not contain consecutive hyphens")
	}
	for _, ch := range fm.Name {
		isLowerAlpha := ch >= 'a' && ch <= 'z'
		isNumeric := ch >= '0' && ch <= '9'
		isHyphen := ch == '-'
		if !isLowerAlpha && !isNumeric && !isHyphen {
			return fmt.Errorf("name may only contain lowercase alphanumeric characters (a-z, 0-9) and hyphens")
		}
	}

	if len(fm.Description) < 1 || len(fm.Description) > 1024 {
		return fmt.Errorf("description must be between 1 and 1024 characters long")
	}

	if len(fm.Compatibility) > 500 {
		return fmt.Errorf("compatibility must not exceed 500 characters")
	}

	return nil
}

// Build creates the full SKILL.md file content from a Frontmatter struct and a
// Markdown instruction body. It validates the Frontmatter before building.
func Build(fm *Frontmatter, markdown string) ([]byte, error) {
	if err := Validate(fm); err != nil {
		return nil, fmt.Errorf("invalid frontmatter: %w", err)
	}
	marshalled, err := yaml.Marshal(fm)
	if err != nil {
		return nil, fmt.Errorf("marshal frontmatter: %w", err)
	}
	return slices.Concat(frontmatterSeparator, marshalled, frontmatterSeparator, []byte(markdown)), nil
}
