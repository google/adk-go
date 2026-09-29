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

// Package provides an example of using skills via skill toolset.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	err := Process(ctx)
	if err != nil {
		panic(err)
	}

	// private-kdroste-test-skill
}

func Process(ctx context.Context) error {
	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: "kdroste-adk-2025-12", Location: "global"}
	source, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
	}

	model, err := gemini.NewModel(ctx, "gemini-flash-latest", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	skillToolset, err := skilltoolset.New(ctx, skilltoolset.Config{Source: source})
	if err != nil {
		log.Fatalf("Failed to create skill toolset: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "skills_agent",
		Model:       model,
		Description: "Agent to demonstrate using skills.",
		Instruction: "You are a helpful assistant.",
		Toolsets:    []tool.Toolset{skillToolset},
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
	return nil
}

func Process4(ctx context.Context) error {

	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: "kdroste-adk-2025-12", Location: "global"})
	if err != nil {
		return fmt.Errorf("cannot skillregistry.NewClient: %w", err)
	}
	s, err := c.GetSkill("projects/kdroste-adk-2025-12/locations/global/skills/private-kdroste-dice-thrower-03")
	if err != nil {
		return fmt.Errorf("cannot GetSkill: %w", err)
	}
	log.Printf("s: %+v", s)
	log.Printf("s.Frontmatter: %+v", s.Frontmatter)

	rev := s.DefaultRevision
	zip, err := c.GetZip(rev)
	if err != nil {
		return fmt.Errorf("cannot GetZip: %w", err)
	}
	for _, f := range zip.File {
		log.Printf("File in zip: %v", f.Name)
	}

	// r, err := c.GetRevision(rev)
	// if err != nil {
	// 	return fmt.Errorf("cannot GetRevision: %w", err)
	// }
	// log.Printf("r: %+v", r)

	return nil
}

func Process3(ctx context.Context) error {
	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: "kdroste-adk-2025-12", Location: "us"}
	src, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
	}

	f, err := src.FindFrontmatters(ctx, "aa")
	if err != nil {
		return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	}
	log.Printf("F: %+v", f)
	return nil
}

func Process2(ctx context.Context) error {
	cfg := skillregistry.SkillRegistrySourceConfig{ProjectID: "kdroste-adk-2025-12", Location: "global"}
	src, err := skillregistry.NewSkillRegistrySource(ctx, cfg)
	if err != nil {
		return fmt.Errorf("cannot NewSkillRegistrySource: %w", err)
	}

	// skills, err := src.ListSkills()
	// if err != nil {
	// 	panic(err)
	// }

	// for _, s := range skills {
	// 	log.Printf("State: %v", s.State)
	// }

	frs, err := src.ListFrontmatters(ctx)
	if err != nil {
		return fmt.Errorf("cannot ListFrontmatters: %w", err)
	}

	for _, f := range frs {
		log.Printf("f: %+v", f)
		break
	}

	// s, err := src.Load(ctx, "discoveryengine.googleapis.com-report-writing")
	// if err == nil {
	// 	return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	// }
	// log.Printf("s: %+v", s)

	// s, err := src.LoadFrontmatter(ctx, "discoveryengine.googleapis.com-report-writing")
	// if err == nil {
	// 	return fmt.Errorf("cannot LoadFrontmatter: %w", err)
	// }
	// log.Printf("s: %+v", s)

	return nil
}

// func main2() {
// 	ctx := context.Background()
// 	c, err := skillregistry.NewClient(ctx, skillregistry.Config{ProjectID: "kdroste-adk-2025-12", Location: "us"})
// 	if err != nil {
// 		panic(err)
// 	}
// 	skills, err := c.ListSkills()
// 	if err != nil {
// 		panic(err)
// 	}

// 	for _, s := range skills {
// 		log.Printf("State: %v", s.State)
// 	}
// 	_ = skills
// }

// func main2() {
// 	ctx := context.Background()
// 	cfg := skillregistry.SkillRegistryClientConfig{
// 		ProjectID: "kdroste-adk-2025-12",
// 		Location:  "us-central1",
// 	}
// 	c, err := skillregistry.NewSkillRegistryClient(ctx, cfg)
// 	if err != nil {
// 		panic(err)
// 	}
// 	skills := c.AllSkills(ctx)
// 	for sk, err := range skills {
// 		if err != nil {
// 			panic(err)
// 		}
// 		log.Printf("Skill: %+v", sk)
// 		// log.Printf("Zip: %+v", sk.ZippedFilesystem)

// 		s, err := c.GetSkill(ctx, sk.Name)
// 		if err != nil {
// 			panic(err)
// 		}
// 		_ = s
// 		// log.Printf("Skill: %+v", s)

// 		// log.Printf("Zip: %+v", s.ZippedFilesystem)

// 		// rootPath := "/usr/local/google/home/kdroste/tmp/skills"
// 		// p := path.Join(rootPath, sk.DisplayName+".zip")

// 		// log.Printf("Will write to %s", p)
// 		// err = os.WriteFile(p, []byte(s.ZippedFilesystem), 0644)
// 		// if err != nil {
// 		// 	panic(err)
// 		// }

// 	}

// }
