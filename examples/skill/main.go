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

package main

import (
	"context"
	"log"
	"os"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/code_executors"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/skills"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/skilltool"
)

func main() {
	ctx := context.Background()

	skillPathList := []string{
		"You Skill Path",
	}
	var skillList []*skills.Skill
	for _, path := range skillPathList {
		skill, err := skills.LoadSkillFromDir(path)
		if err != nil {
			panic(err)
		}
		skillList = append(skillList, skill)
	}

	skillToolset, err := skilltool.NewSkillToolset(skillList, code_executors.NewUnsafeLocalCodeExecutor(300*time.Second))
	if err != nil {
		panic(err)
	}

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "skill_agent",
		Model:       model,
		Description: "Agent to answer questions.",
		Instruction: "Your SOLE purpose is to answer questions. You can use the skill tool when necessary.",
		Toolsets: []tool.Toolset{
			skillToolset,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
