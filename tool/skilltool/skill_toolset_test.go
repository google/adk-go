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

package skilltool

import (
	"encoding/json"
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"google.golang.org/adk/code_executors"
	"google.golang.org/adk/skills"
	"google.golang.org/adk/tool"
)

var mockSkills = map[string]*skills.Skill{
	"multiplication-calculator": {
		Frontmatter: &skills.Frontmatter{
			Name:        "multiplication-calculator",
			Description: "提供乘法数值计算功能。当需要执行乘法运算任务时使用此技能。",
		},
		Instructions: "---\nname: multiplication-calculator\ndescription: 提供乘法数值计算功能。当需要执行乘法运算任务时使用此技能。\n---\n\n# 乘法数值计算器\n\n## 概述\n\n乘法数值计算器技能提供简单的数字相乘能力。该技能包含Python脚本，可用于执行乘法计算任务。\n\n## 快速开始\n\n要使用乘法计算功能，可以直接调用提供的Python脚本：\n\n```bash\n# 运行演示脚本\npython scripts/multiply.py <num1> <num2> ... <numn>\n```",
		Resources: &skills.Resources{
			Scripts: map[string]*skills.Script{
				"multiply.py": {
					Src: `#!/usr/bin/env python3
import sys

"""
乘法数值计算脚本
提供多种乘法运算功能
"""


def multiply_list(numbers):
    """
    列表中的所有数字相乘

    Args:
        numbers (list): 数字列表

    Returns:
        float: 所有数字的乘积

    Raises:
        ValueError: 如果列表为空
    """
    if not numbers:
        raise ValueError("数字列表不能为空")

    result = 1
    for num in numbers:
        result *= num
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multiply.py <num1> <num2> ...<numn>")
        sys.exit(1)
    nums = []
    for n in sys.argv[1:]:
        nums.append(float(n))
    out = multiply_list(nums)
    print(out)
`,
				},
			},
		},
	},
}

func createMockSkill(t *testing.T) []*skills.Skill {
	tmpDir := t.TempDir()
	log.Print("Created temp dir: " + tmpDir)
	var skillList []*skills.Skill
	for _, sk := range mockSkills {
		err := sk.WriteSkill(tmpDir)
		if err != nil {
			t.Fatalf("write skill %s to %s error:%s", sk.Name(), tmpDir, err)
		}
		skillList = append(skillList, sk)
	}
	return skillList
}

func TestListSkillsTool(t *testing.T) {
	skillList := createMockSkill(t)
	toolset, err := NewSkillToolset(skillList, code_executors.NewUnsafeLocalCodeExecutor(300*time.Second))
	assert.NoError(t, err)

	listTool := toolset.listSkillsTool()
	assert.Equal(t, "list_skills", listTool.Name())

	result, err := toolset.listSkillsToolHandler(nil, listSkillsArgs{})
	assert.NoError(t, err)

	outputMap := result
	xmlResult, ok := outputMap["result"].(string)
	assert.True(t, ok)
	assert.Contains(t, xmlResult, "multiplication-calculator")
	assert.Contains(t, xmlResult, "提供乘法数值计算功能。当需要执行乘法运算任务时使用此技能。")
}

func TestLoadSkillTool(t *testing.T) {
	skillList := createMockSkill(t)
	toolset, err := NewSkillToolset(skillList, nil)
	assert.NoError(t, err)

	loadTool := toolset.loadSkillTool()
	assert.Equal(t, "load_skill", loadTool.Name())

	// Test missing name
	result, err := toolset.loadSkillToolHandler(nil, loadSkillArgs{})
	assert.NoError(t, err)
	outputMap := result
	assert.Equal(t, "MISSING_SKILL_NAME", outputMap["error_code"])

	// Test skill not found
	result, err = toolset.loadSkillToolHandler(nil, loadSkillArgs{Name: "unknown-skill"})
	assert.NoError(t, err)
	outputMap = result
	assert.Equal(t, "SKILL_NOT_FOUND", outputMap["error_code"])

	// Test success
	result, err = toolset.loadSkillToolHandler(nil, loadSkillArgs{Name: "multiplication-calculator"})
	assert.NoError(t, err)
	outputMap = result
	assert.Equal(t, "multiplication-calculator", outputMap["skill_name"])
	assert.Equal(t, mockSkills["multiplication-calculator"].Instructions, outputMap["instructions"])
	assert.NotEmpty(t, outputMap["frontmatter"])
}

func TestLoadSkillResourceTool(t *testing.T) {
	skillList := createMockSkill(t)
	toolset, err := NewSkillToolset(skillList, nil)
	assert.NoError(t, err)

	resourceTool := toolset.loadSkillResourceTool()
	assert.Equal(t, "load_skill_resource", resourceTool.Name())

	// Test missing params
	result, err := toolset.loadSkillResourceToolHandler(nil, loadSkillResourceArgs{})
	assert.NoError(t, err)
	outputMap := result
	assert.Equal(t, "MISSING_SKILL_NAME", outputMap["error_code"])

	result, err = toolset.loadSkillResourceToolHandler(nil, loadSkillResourceArgs{SkillName: "multiplication-calculator"})
	assert.NoError(t, err)
	outputMap = result
	assert.Equal(t, "MISSING_RESOURCE_PATH", outputMap["error_code"])

	// Test success
	result, err = toolset.loadSkillResourceToolHandler(nil, loadSkillResourceArgs{
		SkillName: "multiplication-calculator",
		Path:      "scripts/multiply.py",
	})
	assert.NoError(t, err)
	outputMap = result
	assert.Equal(t, "multiplication-calculator", outputMap["skill_name"])
	assert.Equal(t, "scripts/multiply.py", outputMap["path"])
	assert.Equal(t, mockSkills["multiplication-calculator"].Resources.Scripts["multiply.py"].String(), outputMap["content"])

	// Test not found
	result, err = toolset.loadSkillResourceToolHandler(nil, loadSkillResourceArgs{
		SkillName: "multiplication-calculator",
		Path:      "scripts/unknown.py",
	})
	assert.NoError(t, err)
	outputMap = result
	assert.Equal(t, "RESOURCE_NOT_FOUND", outputMap["error_code"])
}

type mockToolContext struct {
	tool.Context
}

func (m *mockToolContext) InvocationID() string {
	return "test-invocation-id"
}

func TestRunSkillScriptTool(t *testing.T) {
	skillList := createMockSkill(t)

	mockExecutor := code_executors.NewUnsafeLocalCodeExecutor(300 * time.Second)

	toolset, err := NewSkillToolset(skillList, mockExecutor)
	assert.NoError(t, err)

	runTool := toolset.runSkillScriptTool()
	assert.Equal(t, "run_skill_script", runTool.Name())

	// Test missing params
	result, err := toolset.runSkillScriptToolHandler(&mockToolContext{}, runSkillScriptArgs{})
	assert.NoError(t, err)
	outputMap := result
	assert.Equal(t, "MISSING_SKILL_NAME", outputMap["error_code"])

	// Test success
	result, err = toolset.runSkillScriptToolHandler(&mockToolContext{}, runSkillScriptArgs{
		SkillName:  "multiplication-calculator",
		ScriptPath: "scripts/multiply.py",
		Args:       []string{"2", "3", "4"},
	})
	assert.NoError(t, err)
	outputMap = result
	str, _ := json.Marshal(outputMap)
	println(string(str))
	assert.Equal(t, "success", outputMap["status"])
	assert.Equal(t, "24.0\n", outputMap["stdout"])
}
