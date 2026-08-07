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
	"log"
	"os"
	"path"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

func main() {
	ctx := context.Background()
	cfg := skill.SkillRegistryClientConfig{
		ProjectID: "kdroste-cloud-sandbox-68115",
		Location:  "us-central1",
	}
	c, err := skill.NewSkillRegistryClient(ctx, cfg)
	if err != nil {
		panic(err)
	}
	skills := c.AllSkills(ctx)
	for sk, err := range skills {
		if err != nil {
			panic(err)
		}
		log.Printf("Skill: %+v", sk)
		log.Printf("Zip: %+v", sk.ZippedFilesystem)

		s, err := c.GetSkill(ctx, sk.Name)
		if err != nil {
			panic(err)
		}

		log.Printf("Zip: %+v", s.ZippedFilesystem)

		rootPath := "/usr/local/google/home/kdroste/tmp/skills"
		p := path.Join(rootPath, sk.DisplayName+".zip")

		log.Printf("Will write to %s", p)
		err = os.WriteFile(p, []byte(s.ZippedFilesystem), 0644)
		if err != nil {
			panic(err)
		}

	}

}
