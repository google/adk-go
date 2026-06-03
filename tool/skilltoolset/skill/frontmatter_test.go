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

package skill

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"testing/iotest"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		frontmatter *Frontmatter
		wantErr     bool
	}{
		{
			name:        "valid frontmatter",
			frontmatter: &Frontmatter{Name: "valid-name", Description: "A valid description."},
		},
		{
			name:        "nil frontmatter",
			frontmatter: nil,
			wantErr:     true,
		},
		{
			name:        "name too short",
			frontmatter: &Frontmatter{Name: "", Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "name too long",
			frontmatter: &Frontmatter{Name: strings.Repeat("a", 65), Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "name starts with hyphen",
			frontmatter: &Frontmatter{Name: "-name", Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "name ends with hyphen",
			frontmatter: &Frontmatter{Name: "name-", Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "name has consecutive hyphens",
			frontmatter: &Frontmatter{Name: "na--me", Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "name contains uppercase letters",
			frontmatter: &Frontmatter{Name: "Invalid-Name", Description: "Valid description."},
			wantErr:     true,
		},
		{
			name:        "description too short",
			frontmatter: &Frontmatter{Name: "valid-name", Description: ""},
			wantErr:     true,
		},
		{
			name:        "description too long",
			frontmatter: &Frontmatter{Name: "valid-name", Description: strings.Repeat("a", 1025)},
			wantErr:     true,
		},
		{
			name:        "compatibility too long",
			frontmatter: &Frontmatter{Name: "valid-name", Description: "Valid.", Compatibility: strings.Repeat("a", 501)},
			wantErr:     true,
		},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			err := Validate(testcase.frontmatter)
			if got, want := (err != nil), testcase.wantErr; got != want {
				t.Fatalf("expected error %v, got %v", want, got)
			}
		})
	}
}

func TestParse_Valid(t *testing.T) {
	tests := []struct {
		name            string
		input           string
		want            *Frontmatter
		wantInstruction string
	}{
		{
			name:  "valid minimal frontmatter",
			input: "---\nname: skill\ndescription: skill\n---\nMarkdown Body",
			want: &Frontmatter{
				Name:        "skill",
				Description: "skill",
			},
			wantInstruction: "Markdown Body",
		},
		{
			name:  "valid minimal frontmatter with windows line endings",
			input: "---\r\nname: skill\r\ndescription: skill\r\n---\r\nMarkdown Body",
			want: &Frontmatter{
				Name:        "skill",
				Description: "skill",
			},
			wantInstruction: "Markdown Body",
		},
		{
			name: "valid full frontmatter",
			input: `---
name: my-cool-skill
description: A cool skill.
metadata:
  author: "Cool Author"
  version: "1.0.0"
allowed-tools:
  - tool1
  - tool2
compatibility: "compatible indeed"
license: "yes"
---

# Long

Multi-line
Markdown
Body

`,
			want: &Frontmatter{
				Name:        "my-cool-skill",
				Description: "A cool skill.",
				Metadata: map[string]string{
					"author":  "Cool Author",
					"version": "1.0.0",
				},
				AllowedTools:  []string{"tool1", "tool2"},
				Compatibility: "compatible indeed",
				License:       "yes",
			},
			wantInstruction: "\n# Long\n\nMulti-line\nMarkdown\nBody\n\n",
		},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			t.Parallel()
			reader := bufio.NewReader(strings.NewReader(testcase.input))
			got, err := Parse(reader)
			if err != nil {
				t.Fatalf("Parse: expected no error, got: %v", err)
			}

			if !reflect.DeepEqual(got, testcase.want) {
				t.Errorf("Parse: got %v, want %v", got, testcase.want)
			}
			bytes, err := io.ReadAll(reader)
			if err != nil {
				t.Fatalf("io.ReadAll: expected no error, got: %v", err)
			}
			if string(bytes) != testcase.wantInstruction {
				t.Errorf("expected unread instruction %q, got %q", testcase.wantInstruction, string(bytes))
			}
		})
	}
}

func TestParse_InvalidFormat(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "missing opening separator",
			input: "name: skill\ndescription: skill\n---\n# Markdown\nBody",
		},
		{
			name:  "missing newline after opening separator",
			input: "---name: skill\ndescription: skill\n---\n# Markdown\nBody",
		},
		{
			name:  "whitespace after opening separator",
			input: "--- \nname: skill\ndescription: skill\n---\n# Markdown\nBody",
		},
		{
			name:  "missing closing separator",
			input: "---\nname: skill\ndescription: skill\n",
		},
		{
			name:  "missing newline after closing separator",
			input: "---\nname: skill\ndescription: skill\n---# Markdown\nBody",
		},
		{
			name:  "whitespace after closing separator",
			input: "---\nname: skill\ndescription: skill\n--- \n# Markdown\nBody",
		},
		{
			name:  "invalid yaml",
			input: "---\n_ : invalid : _\n---\n# Markdown\nBody",
		},
		{
			name:  "fails frontmatter validation",
			input: "---\nname: INVALID_NAME\ndescription: test\n---\n# Markdown\nBody",
		},
		{
			name:  "unknown fields",
			input: "---\nname: skill\ndescription: skill\nunknown-field: field\n---\n# Markdown\nBody",
		},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			t.Parallel()
			reader := bufio.NewReader(strings.NewReader(testcase.input))

			_, err := Parse(reader)

			if err == nil {
				t.Errorf("expected error, got nil")
			}
		})
	}
}

func TestParse_FailingReader(t *testing.T) {
	errImmediate := fmt.Errorf("immediate error")
	tests := []struct {
		name    string
		reader  io.Reader
		wantErr error
	}{
		{
			name:    "immediate error",
			reader:  iotest.ErrReader(errImmediate),
			wantErr: errImmediate,
		},
		{
			name:    "timeout error",
			reader:  iotest.TimeoutReader(strings.NewReader("---")),
			wantErr: iotest.ErrTimeout,
		},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			t.Parallel()
			_, err := Parse(bufio.NewReader(testcase.reader))
			if !errors.Is(err, testcase.wantErr) {
				t.Errorf("Parse: expected error %v, got %v", testcase.wantErr, err)
			}
		})
	}
}

func TestParseBytes(t *testing.T) {
	input := []byte("---\nname: my-skill\ndescription: A cool skill\n---\n# Markdown Content\nHere.")
	wantFrontmatter := &Frontmatter{
		Name:        "my-skill",
		Description: "A cool skill",
	}
	wantInstruction := "# Markdown Content\nHere."

	gotFrontmatter, gotInstruction, err := ParseBytes(input)
	if err != nil {
		t.Fatalf("ParseBytes failed: %v", err)
	}

	if !reflect.DeepEqual(gotFrontmatter, wantFrontmatter) {
		t.Errorf("ParseBytes: got frontmatter %v, want %v", gotFrontmatter, wantFrontmatter)
	}
	if gotInstruction != wantInstruction {
		t.Errorf("ParseBytes: got instruction %q, want %q", gotInstruction, wantInstruction)
	}
}

func TestParseBytes_Error(t *testing.T) {
	input := []byte("---\ninvalid-frontmatter without closing")
	_, _, err := ParseBytes(input)

	if err == nil {
		t.Fatalf("ParseBytes: expected error for invalid frontmatter bytes, got nil")
	}
}

func TestBuild(t *testing.T) {
	frontmatter := &Frontmatter{
		Name:        "my-skill",
		Description: "A test skill.",
		Metadata: map[string]string{
			"author": "Cool Author",
		},
		AllowedTools: []string{"tool1", "tool2"},
	}
	instruction := "# Instruction\nDo something cool."

	outBytes, err := Build(frontmatter, instruction)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	outStr := string(outBytes)
	// We don't rely on strict exact string matching for the entire block.
	if !strings.HasPrefix(outStr, "---\n") {
		t.Errorf("Build: output must start with frontmatter separator")
	}
	if !strings.Contains(outStr, "\n---\n") {
		t.Errorf("Build: output missing proper closing separator")
	}
	for _, field := range []string{"name", "description", "metadata", "allowed-tools"} {
		if !strings.Contains(outStr, field) {
			t.Errorf("Build: output missing %s field", field)
		}
	}
	for _, field := range []string{"compatibility", "license"} {
		if strings.Contains(outStr, field) {
			t.Errorf("Build: output should not contain %s field", field)
		}
	}
	// Parse the output to verify it's correct.
	parsedFrontmatter, parsedInstruction, err := ParseBytes([]byte(outStr))
	if err != nil {
		t.Fatalf("ParseBytes: failed to parse frontmatter after Build: %v", err)
	}
	if !reflect.DeepEqual(parsedFrontmatter, frontmatter) {
		t.Errorf("ParseBytes: parsed frontmatter after Build: %v, want %v", parsedFrontmatter, frontmatter)
	}
	if parsedInstruction != instruction {
		t.Errorf("ParseBytes: parsed instruction after Build: %q, want %q", parsedInstruction, instruction)
	}
}

func TestBuild_ValidationError(t *testing.T) {
	fm := &Frontmatter{
		Name:        "INVALID_NAME!!!",
		Description: "A test skill.",
	}

	_, err := Build(fm, "Some markdown")

	if err == nil {
		t.Fatalf("Build: expected validation error, got nil")
	}
}
