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

package skill_test

import (
	"strings"
	"testing"

	"google.golang.org/adk/skill"
)

func TestFrontmatter_Validate_OK(t *testing.T) {
	f := &skill.Frontmatter{Name: "my-skill", Description: "does a thing"}
	if err := f.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

func TestFrontmatter_Validate_RejectsBadNames(t *testing.T) {
	cases := []string{"My-Skill", "my_skill", "my--skill", "-foo", "foo-", "foo_bar"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			f := &skill.Frontmatter{Name: name, Description: "x"}
			if err := f.Validate(); err == nil {
				t.Errorf("expected error for %q", name)
			}
		})
	}
}

func TestFrontmatter_Validate_AllowsSnakeCaseWhenEnabled(t *testing.T) {
	skill.SnakeCaseAllowed = true
	defer func() { skill.SnakeCaseAllowed = false }()
	f := &skill.Frontmatter{Name: "my_skill", Description: "x"}
	if err := f.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
	// Mixed delimiters still rejected.
	f.Name = "my_kebab-skill"
	if err := f.Validate(); err == nil {
		t.Error("mixed delimiters should be rejected")
	}
}

func TestFrontmatter_Validate_RejectsLong(t *testing.T) {
	f := &skill.Frontmatter{Name: strings.Repeat("a", 70), Description: "x"}
	if err := f.Validate(); err == nil {
		t.Error("expected error for long name")
	}
	f = &skill.Frontmatter{Name: "x", Description: strings.Repeat("d", 1100)}
	if err := f.Validate(); err == nil {
		t.Error("expected error for long description")
	}
}

func TestFrontmatter_Validate_RejectsEmptyDescription(t *testing.T) {
	f := &skill.Frontmatter{Name: "ok", Description: ""}
	if err := f.Validate(); err == nil {
		t.Error("expected error for empty description")
	}
}

func TestFormatAsXML(t *testing.T) {
	skills := []*skill.Skill{
		{Frontmatter: skill.Frontmatter{Name: "alpha", Description: "first"}},
		{Frontmatter: skill.Frontmatter{Name: "beta", Description: "second"}},
	}
	out := skill.FormatAsXML(skills)
	for _, want := range []string{"<available_skills>", "</available_skills>", "<skill>", "alpha", "beta"} {
		if !strings.Contains(out, want) {
			t.Errorf("FormatAsXML output missing %q:\n%s", want, out)
		}
	}
}

func TestFormatAsXML_Empty(t *testing.T) {
	out := skill.FormatAsXML(nil)
	if !strings.Contains(out, "<available_skills>") || !strings.Contains(out, "</available_skills>") {
		t.Errorf("empty XML missing tags: %s", out)
	}
}

func TestFormatAsXML_EscapesHTML(t *testing.T) {
	skills := []*skill.Skill{
		{Frontmatter: skill.Frontmatter{Name: "pkg", Description: "uses <script> tag"}},
	}
	out := skill.FormatAsXML(skills)
	if strings.Contains(out, "<script>") {
		t.Errorf("description should be HTML-escaped: %s", out)
	}
	if !strings.Contains(out, "&lt;script&gt;") {
		t.Error("expected escaped <script> in output")
	}
}

func TestResources_Accessors(t *testing.T) {
	r := &skill.Resources{
		References: map[string][]byte{"r1": []byte("ref")},
		Assets:     map[string][]byte{"a1": []byte("ast")},
		Scripts:    map[string]string{"s1": "echo hi"},
	}
	if v, ok := r.GetReference("r1"); !ok || string(v) != "ref" {
		t.Errorf("GetReference = %q,%v", v, ok)
	}
	if v, ok := r.GetAsset("a1"); !ok || string(v) != "ast" {
		t.Errorf("GetAsset = %q,%v", v, ok)
	}
	if v, ok := r.GetScript("s1"); !ok || v != "echo hi" {
		t.Errorf("GetScript = %q,%v", v, ok)
	}
}

func TestSkill_NameAndDescriptionAccessors(t *testing.T) {
	s := &skill.Skill{Frontmatter: skill.Frontmatter{Name: "n", Description: "d"}}
	if s.Name() != "n" || s.Description() != "d" {
		t.Error("accessor mismatch")
	}
}
