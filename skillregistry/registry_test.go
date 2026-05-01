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

package skillregistry_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/skill"
	"google.golang.org/adk/skillregistry"
)

func mkSkill(name, desc, body string) *skill.Skill {
	return &skill.Skill{
		Frontmatter:  skill.Frontmatter{Name: name, Description: desc},
		Instructions: body,
	}
}

func TestRegister_RejectsInvalidFrontmatter(t *testing.T) {
	r := skillregistry.New()
	err := r.Register(skill.Frontmatter{Name: "Bad-Name", Description: "x"}, func() (*skill.Skill, error) { return nil, nil })
	if err == nil {
		t.Error("expected validation error")
	}
}

func TestRegister_RejectsNilBuilder(t *testing.T) {
	r := skillregistry.New()
	err := r.Register(skill.Frontmatter{Name: "ok", Description: "x"}, nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestRegisterSkill_RejectsNil(t *testing.T) {
	r := skillregistry.New()
	if err := r.RegisterSkill(nil); err == nil {
		t.Error("expected error")
	}
}

func TestGet_LazilyBuildsAndCaches(t *testing.T) {
	r := skillregistry.New()
	calls := 0
	r.Register(skill.Frontmatter{Name: "lazy", Description: "x"}, func() (*skill.Skill, error) {
		calls++
		return mkSkill("lazy", "x", "body"), nil
	})
	for i := 0; i < 3; i++ {
		s, err := r.Get("lazy")
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if s.Name() != "lazy" {
			t.Errorf("Name = %q", s.Name())
		}
	}
	if calls != 1 {
		t.Errorf("builder calls = %d, want 1", calls)
	}
}

func TestGet_PropagatesBuilderError(t *testing.T) {
	r := skillregistry.New()
	want := errors.New("build broke")
	r.Register(skill.Frontmatter{Name: "broken", Description: "x"}, func() (*skill.Skill, error) { return nil, want })
	if _, err := r.Get("broken"); !errors.Is(err, want) {
		t.Errorf("err = %v", err)
	}
}

func TestList_FiltersByQuery(t *testing.T) {
	r := skillregistry.New()
	r.RegisterSkill(mkSkill("math-helper", "math help", ""))
	r.RegisterSkill(mkSkill("search-helper", "web search help", ""))
	got := r.List(skillregistry.Filter{Query: "search"})
	if len(got) != 1 || got[0].Name != "search-helper" {
		t.Errorf("got %v", got)
	}
}

func TestList_FiltersByTags(t *testing.T) {
	r := skillregistry.New()
	r.Register(skill.Frontmatter{
		Name: "a", Description: "x",
		Metadata: map[string]any{"tags": []string{"math", "tools"}},
	}, func() (*skill.Skill, error) { return mkSkill("a", "x", ""), nil })
	r.Register(skill.Frontmatter{
		Name: "b", Description: "x",
		Metadata: map[string]any{"tags": []string{"web"}},
	}, func() (*skill.Skill, error) { return mkSkill("b", "x", ""), nil })

	got := r.List(skillregistry.Filter{Tags: []string{"math"}})
	if len(got) != 1 || got[0].Name != "a" {
		t.Errorf("got %v", got)
	}
}

func TestNames(t *testing.T) {
	r := skillregistry.New()
	r.RegisterSkill(mkSkill("z", "x", ""))
	r.RegisterSkill(mkSkill("a", "x", ""))
	names := r.Names()
	if len(names) != 2 || names[0] != "a" {
		t.Errorf("Names = %v", names)
	}
}

type fakeStateGetter struct{ m map[string]any }

func (f *fakeStateGetter) Get(key string) (any, error) {
	if v, ok := f.m[key]; ok {
		return v, nil
	}
	return nil, errors.New("not found")
}

func TestLoadedNames_AcceptsBothShapes(t *testing.T) {
	got := skillregistry.LoadedNames(&fakeStateGetter{m: map[string]any{
		skillregistry.StateKeyLoadedSkills: []string{"a", "b"},
	}})
	if len(got) != 2 || got[0] != "a" {
		t.Errorf("[]string: %v", got)
	}
	got = skillregistry.LoadedNames(&fakeStateGetter{m: map[string]any{
		skillregistry.StateKeyLoadedSkills: []any{"a", "b"},
	}})
	if len(got) != 2 || got[1] != "b" {
		t.Errorf("[]any: %v", got)
	}
	got = skillregistry.LoadedNames(&fakeStateGetter{m: map[string]any{}})
	if got != nil {
		t.Errorf("missing key should return nil, got %v", got)
	}
}

func TestLoadedSkills_SkipsUnknown(t *testing.T) {
	r := skillregistry.New()
	r.RegisterSkill(mkSkill("a", "x", ""))
	skills := skillregistry.LoadedSkills(r, &fakeStateGetter{m: map[string]any{
		skillregistry.StateKeyLoadedSkills: []string{"a", "missing"},
	}})
	if len(skills) != 1 || skills[0].Name() != "a" {
		t.Errorf("got %v", skills)
	}
}

func TestNewListSkillsTool_Name(t *testing.T) {
	r := skillregistry.New()
	tt, err := skillregistry.NewListSkillsTool(r)
	if err != nil {
		t.Fatalf("NewListSkillsTool: %v", err)
	}
	if tt.Name() != "list_skills" {
		t.Errorf("Name = %q", tt.Name())
	}
}

func TestNewLoadSkillTool_Name(t *testing.T) {
	r := skillregistry.New()
	tt, err := skillregistry.NewLoadSkillTool(r)
	if err != nil {
		t.Fatalf("NewLoadSkillTool: %v", err)
	}
	if tt.Name() != "load_skill" {
		t.Errorf("Name = %q", tt.Name())
	}
}

func TestNewListSkillsTool_RejectsNilRegistry(t *testing.T) {
	if _, err := skillregistry.NewListSkillsTool(nil); err == nil {
		t.Error("expected error")
	}
}

func TestNewLoadSkillTool_RejectsNilRegistry(t *testing.T) {
	if _, err := skillregistry.NewLoadSkillTool(nil); err == nil {
		t.Error("expected error")
	}
}

func TestNewSkillsInstructionPlugin_RejectsNilRegistry(t *testing.T) {
	if _, err := skillregistry.NewSkillsInstructionPlugin(skillregistry.SkillsInstructionPluginConfig{}); err == nil {
		t.Error("expected error")
	}
}

func TestNewSkillsInstructionPlugin_BuildsOK(t *testing.T) {
	r := skillregistry.New()
	r.RegisterSkill(mkSkill("a", "desc", "instructions body"))
	p, err := skillregistry.NewSkillsInstructionPlugin(skillregistry.SkillsInstructionPluginConfig{
		Registry: r,
	})
	if err != nil {
		t.Fatalf("NewSkillsInstructionPlugin: %v", err)
	}
	if p.Name() != "skills_instruction" {
		t.Errorf("Name = %q", p.Name())
	}
}
