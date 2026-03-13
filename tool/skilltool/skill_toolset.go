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
	"fmt"
	"log"
	"path/filepath"
	"sort"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/code_executors"
	"google.golang.org/adk/model"
	"google.golang.org/adk/skills"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

const (
	DEFAULT_SKILL_SYSTEM_INSTRUCTION = "" +
		"You can use specialized 'skills' to help you with complex tasks. You MUST use the skill tools to interact with these skills.\n\n" +
		"Skills are folders of instructions and resources that extend your capabilities for specialized tasks. Each skill folder contains:\n" +
		"- **SKILL.md** (required): The main instruction file with skill metadata and detailed markdown instructions.\n" +
		"- **references/** (Optional): Additional documentation or examples for skill usage.\n" +
		"- **assets/** (Optional): Templates, scripts or other resources used by the skill.\n" +
		"- **scripts/** (Optional): Executable scripts that can be run via bash.\n\n" +
		"This is very important:\n\n" +
		"1. If a skill seems relevant to the current user query, you MUST use the `load_skill` tool with `name=\"<SKILL_NAME>\"` to read its full instructions before proceeding.\n" +
		"2. Once you have read the instructions, follow them exactly as documented before replying to the user. For example, If the instruction lists multiple steps, please make sure you complete all of them in order.\n" +
		"3. The `load_skill_resource` tool is for viewing files within a skill's directory (e.g., `references/*`, `assets/*`, `scripts/*`). Do NOT use other tools to access these files.\n" +
		"4. Use `run_skill_script` to run scripts from a skill's `scripts/` directory. Use `load_skill_resource` to view script content first if needed.\n"
)

// SkillToolset A toolset for managing and interacting with agent skills.
type SkillToolset struct {
	skills       map[string]*skills.Skill
	tools        []tool.Tool
	codeExecutor code_executors.CodeExecutor
}

func NewSkillToolset(skillList []*skills.Skill, codeExecutor code_executors.CodeExecutor) (*SkillToolset, error) {
	m := make(map[string]*skills.Skill, len(skillList))
	for _, s := range skillList {
		if _, dup := m[s.Name()]; dup {
			return nil, fmt.Errorf("duplicate skill name '%s'", s.Name())
		}
		m[s.Name()] = s
	}
	st := &SkillToolset{
		skills:       m,
		codeExecutor: codeExecutor,
	}
	st.tools = []tool.Tool{
		st.listSkillsTool(),
		st.loadSkillTool(),
		st.loadSkillResourceTool(),
		st.runSkillScriptTool(),
	}
	return st, nil
}

func (s *SkillToolset) Name() string {
	return "SkillToolset"
}

func (s *SkillToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return s.tools, nil
}

//func (s *SkillToolset) GetTools() []tool.Tool {
//	return s.tools
//}

func (s *SkillToolset) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	skillList := s.listSkills()
	skillXML := skills.FormatSkillsAsXML(skillList)
	instruction := []string{DEFAULT_SKILL_SYSTEM_INSTRUCTION, skillXML}
	if req.Config.SystemInstruction == nil {
		req.Config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{
					Text: strings.Join(instruction, "\n\n"),
				},
			},
			Role: "user",
		}
	} else {
		req.Config.SystemInstruction.Parts = append(req.Config.SystemInstruction.Parts,
			&genai.Part{
				Text: strings.Join(instruction, "\n\n"),
			},
		)
	}
	systemInstructionStr, _ := json.Marshal(req.Config.SystemInstruction)
	log.Printf("SkillToolset After ProcessRequest SystemInstruction is %s", systemInstructionStr)
	return nil
}

func (s *SkillToolset) getSkill(name string) (*skills.Skill, bool) {
	sk, ok := s.skills[name]
	return sk, ok
}

func (s *SkillToolset) listSkills() []*skills.Skill {
	out := make([]*skills.Skill, 0, len(s.skills))
	for _, v := range s.skills {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name() < out[j].Name() })
	return out
}

type listSkillsArgs struct{}

func (s *SkillToolset) listSkillsToolHandler(ctx tool.Context, args listSkillsArgs) (map[string]any, error) {
	xml := skills.FormatSkillsAsXML(s.listSkills())
	return map[string]any{"result": xml}, nil
}

// listSkillsTool Tool to list all available skills.
func (s *SkillToolset) listSkillsTool() tool.Tool {
	t, _ := functiontool.New(functiontool.Config{
		Name:        "list_skills",
		Description: "Lists all available skills with their names and descriptions.",
	}, s.listSkillsToolHandler)
	return t
}

type loadSkillArgs struct {
	Name string `json:"name" jsonschema:"The name of the skill to load."`
}

func (s *SkillToolset) loadSkillToolHandler(ctx tool.Context, args loadSkillArgs) (map[string]any, error) {
	if strings.TrimSpace(args.Name) == "" {
		return map[string]any{
			"error":      "Skill name is required.",
			"error_code": "MISSING_SKILL_NAME",
		}, nil
	}
	sk, ok := s.getSkill(args.Name)
	if !ok {
		return map[string]any{
			"error":      fmt.Sprintf("Skill '%s' not found.", args.Name),
			"error_code": "SKILL_NOT_FOUND",
		}, nil
	}

	return map[string]any{
		"skill_name":   sk.Name(),
		"instructions": sk.Instructions,
		"frontmatter":  sk.Frontmatter.ToJsonString(),
	}, nil
}

// loadSkillTool Tool to load a skill's instructions."""
func (s *SkillToolset) loadSkillTool() tool.Tool {
	t, _ := functiontool.New(functiontool.Config{
		Name:        "load_skill",
		Description: "Loads the SKILL.md instructions for a given skill.",
	}, s.loadSkillToolHandler)
	return t
}

type loadSkillResourceArgs struct {
	SkillName string `json:"skill_name" jsonschema:"The name of the skill."`
	Path      string `json:"path" jsonschema:"The relative path to the resource (e.g., 'references/x.md', 'assets/template.txt', 'scripts/setup.sh')."`
}

func (s *SkillToolset) loadSkillResourceToolHandler(ctx tool.Context, args loadSkillResourceArgs) (map[string]any, error) {
	if strings.TrimSpace(args.SkillName) == "" {
		return map[string]any{"error": "Skill name is required.", "error_code": "MISSING_SKILL_NAME"}, nil
	}
	if strings.TrimSpace(args.Path) == "" {
		return map[string]any{"error": "Resource path is required.", "error_code": "MISSING_RESOURCE_PATH"}, nil
	}
	sk, ok := s.getSkill(args.SkillName)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("Skill '%s' not found.", args.SkillName), "error_code": "SKILL_NOT_FOUND"}, nil
	}
	var content string
	var found bool
	if strings.HasPrefix(args.Path, "references/") {
		name := strings.TrimPrefix(args.Path, "references/")
		content, found = sk.Resources.GetReference(name)
	} else if strings.HasPrefix(args.Path, "assets/") {
		name := strings.TrimPrefix(args.Path, "assets/")
		content, found = sk.Resources.GetAsset(name)
	} else if strings.HasPrefix(args.Path, "scripts/") {
		name := strings.TrimPrefix(args.Path, "scripts/")
		scr, ok2 := sk.Resources.GetScript(name)
		if ok2 && scr != nil {
			content, found = scr.Src, true
		}
	} else {
		return map[string]any{
			"error":      "Path must start with 'references/', 'assets/', or 'scripts/'.",
			"error_code": "INVALID_RESOURCE_PATH",
		}, nil
	}
	if !found {
		return map[string]any{
			"error":      fmt.Sprintf("Resource '%s' not found in skill '%s'.", args.Path, args.SkillName),
			"error_code": "RESOURCE_NOT_FOUND",
		}, nil
	}
	return map[string]any{
		"skill_name": sk.Name(),
		"path":       args.Path,
		"content":    content,
	}, nil
}

// loadSkillResourceTool Tool to load resources (references, assets, or scripts) from a skill."""
func (s *SkillToolset) loadSkillResourceTool() tool.Tool {
	t, _ := functiontool.New(functiontool.Config{
		Name:        "load_skill_resource",
		Description: "Loads a resource file (from references/, assets/, or scripts/) from within a skill.",
	}, s.loadSkillResourceToolHandler)
	return t
}

type runSkillScriptArgs struct {
	SkillName  string   `json:"skill_name" jsonschema:"The name of the skill."`
	ScriptPath string   `json:"script_path" jsonschema:"The relative path to the script (e.g., 'scripts/setup.py')."`
	Args       []string `json:"args_list" jsonschema:"Optional arguments to pass to the script as list."`
}

func (s *SkillToolset) runSkillScriptToolHandler(ctx tool.Context, args runSkillScriptArgs) (map[string]any, error) {
	if strings.TrimSpace(args.SkillName) == "" {
		return map[string]any{"error": "Skill name is required.", "error_code": "MISSING_SKILL_NAME"}, nil
	}
	if strings.TrimSpace(args.ScriptPath) == "" {
		return map[string]any{"error": "Script path is required.", "error_code": "MISSING_SCRIPT_PATH"}, nil
	}
	sk, ok := s.getSkill(args.SkillName)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("Skill '%s' not found.", args.SkillName), "error_code": "SKILL_NOT_FOUND"}, nil
	}
	name := args.ScriptPath
	if strings.HasPrefix(args.ScriptPath, "scripts/") {
		name = strings.TrimPrefix(args.ScriptPath, "scripts/")
	}

	if scr, ok := sk.Resources.GetScript(name); !ok || scr == nil {
		return map[string]any{"error": fmt.Sprintf("Script '%s' not found in skill '%s'.", args.ScriptPath, args.SkillName), "error_code": "SCRIPT_NOT_FOUND"}, nil
	}
	if s.codeExecutor == nil {
		return map[string]any{
			"error":      "No code executor configured. A code executor is required to run scripts.",
			"error_code": "NO_CODE_EXECUTOR",
		}, nil
	}
	argsStr, _ := json.Marshal(args)
	log.Printf("runSkillScriptToolHandler args is %s", string(argsStr))
	codeExecutorResult, err := s.codeExecutor.ExecuteCode(nil, code_executors.CodeExecutionInput{
		Args:        args.Args,
		ScriptPath:  filepath.Join(sk.GetSkillPath(), "scripts", name),
		InputFiles:  nil,
		ExecutionID: ctx.InvocationID(),
	})
	resultStr, _ := json.Marshal(codeExecutorResult)
	log.Printf("codeExecutor result is %s", string(resultStr))
	if err != nil {
		return map[string]any{
			"error":      fmt.Sprintf("Failed to execute script '%s':\n%s", args.ScriptPath, err.Error()),
			"error_code": "EXECUTION_ERROR",
		}, nil
	}
	status := "success"

	return map[string]any{
		"skill_name":  sk.Name(),
		"script_path": args.ScriptPath,
		"stdout":      codeExecutorResult.StdOut,
		"stderr":      codeExecutorResult.StdErr,
		"status":      status,
	}, nil
}

// runSkillScriptTool Tool to execute scripts from a skill's scripts/ directory."""
func (s *SkillToolset) runSkillScriptTool() tool.Tool {
	t, _ := functiontool.New(functiontool.Config{
		Name:        "run_skill_script",
		Description: "Executes a script from a skill's scripts/ directory.",
	}, s.runSkillScriptToolHandler)
	return t
}
