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

package skilltool_test

import (
	"context"
	"io"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/skilltoolset/internal/skilltool"
	"google.golang.org/adk/tool/skilltoolset/skill"
)

type mockSource struct {
	frontmatters []*skill.Frontmatter
	resources    map[string]map[string]string
	instructions map[string]string
}

func (m *mockSource) ListFrontmatters(ctx context.Context) ([]*skill.Frontmatter, error) {
	return m.frontmatters, nil
}

func (m *mockSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	var res []string
	for p := range m.resources[name] {
		if strings.HasPrefix(p, subpath) {
			res = append(res, p)
		}
	}
	return res, nil
}

func (m *mockSource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	for _, fm := range m.frontmatters {
		if fm.Name == name {
			return fm, nil
		}
	}
	return nil, skill.ErrSkillNotFound
}

func (m *mockSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	inst, ok := m.instructions[name]
	if !ok {
		return "", skill.ErrSkillNotFound
	}
	return inst, nil
}

func (m *mockSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	res, ok := m.resources[name][resourcePath]
	if !ok {
		return nil, skill.ErrResourceNotFound
	}
	return io.NopCloser(strings.NewReader(res)), nil
}

func createToolContext(t *testing.T) tool.Context {
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	return toolinternal.NewToolContext(invCtx, "", nil, nil)
}

func TestListSkills(t *testing.T) {
	source := &mockSource{
		frontmatters: []*skill.Frontmatter{
			{Name: "skill1", Description: "description1"},
			{Name: "skill2", Description: "description2"},
		},
	}

	tTool, err := skilltool.ListSkills(source)
	if err != nil {
		t.Fatalf("ListSkills failed: %v", err)
	}

	funcTool, ok := tTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("tool does not implement toolinternal.FunctionTool")
	}

	result, err := funcTool.Run(createToolContext(t), map[string]any{})
	if err != nil {
		t.Fatalf("tool.Run failed: %v", err)
	}

	want := map[string]any{
		"skills": "<available_skills>\n<skill>\n<name>\nskill1\n</name>\n<description>\ndescription1\n</description>\n</skill>\n<skill>\n<name>\nskill2\n</name>\n<description>\ndescription2\n</description>\n</skill>\n</available_skills>",
	}

	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}

func TestLoadSkill(t *testing.T) {
	source := &mockSource{
		frontmatters: []*skill.Frontmatter{
			{
				Name:        "skill1",
				Description: "description1",
				Metadata: map[string]string{
					"key1": "value1",
				},
			},
		},
		instructions: map[string]string{
			"skill1": "instructions1",
		},
	}
	tool, err := skilltool.LoadSkill(source)
	if err != nil {
		t.Fatalf("LoadSkill failed: %v", err)
	}
	functionTool, ok := tool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("LoadSkill tool does not implement toolinternal.FunctionTool")
	}

	got, err := functionTool.Run(createToolContext(t), map[string]any{"name": "skill1"})
	if err != nil {
		t.Fatalf("LoadSkill tool.Run failed: %v", err)
	}
	want := map[string]any{
		"skill_name":   "skill1",
		"instructions": "instructions1",
		"frontmatter": map[string]any{
			"name":        "skill1",
			"description": "description1",
			"metadata": map[string]any{
				"key1": "value1",
			},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LoadSkill result mismatch (-want +got):\n%s", diff)
	}
}

func TestLoadSkillResource(t *testing.T) {
	source := &mockSource{
		resources: map[string]map[string]string{
			"skill1": {"assets/data.txt": "content1"},
		},
	}
	tool, err := skilltool.LoadSkillResource(source)
	if err != nil {
		t.Fatalf("LoadSkillResource: %v", err)
	}
	functionTool, ok := tool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("LoadSkillResource: tool does not implement toolinternal.FunctionTool")
	}
	result, err := functionTool.Run(createToolContext(t), map[string]any{
		"skill_name":    "skill1",
		"resource_path": "assets/data.txt",
	})
	if err != nil {
		t.Fatalf("tool.Run failed: %v", err)
	}
	want := map[string]any{
		"skill_name": "skill1",
		"path":       "assets/data.txt",
		"content":    "content1",
	}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("result mismatch (-want +got):\n%s", diff)
	}
}
