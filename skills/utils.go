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

package skills

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// loadDir Recursively load files from a directory into a dictionary.
// Args:
//
//	directory: Path to the directory to load.
//
// Returns:
//
//	Dictionary mapping relative file paths to their string content.
func loadDir(directory string) (map[string]string, error) {
	files := make(map[string]string)
	info, err := os.Stat(directory)
	if err != nil {
		return files, nil
	}
	if !info.IsDir() {
		return files, nil
	}
	err = filepath.WalkDir(directory, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		for _, part := range strings.Split(path, string(filepath.Separator)) {
			if part == "__pycache__" {
				if d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
		}
		if d.IsDir() {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read file %s: %w", path, err)
		}
		if !utf8.Valid(b) {
			return errors.New("invalid UTF-8")
		}
		rel, err := filepath.Rel(directory, path)
		if err != nil {
			return fmt.Errorf("get relative path: %w", err)
		}
		rel = filepath.ToSlash(rel)
		files[rel] = string(b)
		return nil
	})
	if err != nil {
		return files, err
	}
	return files, nil
}

func parseFrontmatter(text []byte) (*Frontmatter, error) {
	var skillMeta Frontmatter

	if err := yaml.Unmarshal(text, &skillMeta); err != nil {
		return nil, fmt.Errorf("failed to parse frontmatter {%s} error: %v", text, err)
	}

	if skillMeta.Name == "" || skillMeta.Description == "" {
		return nil, fmt.Errorf("skill %s frontmatter is missing name or description. Please check the SKILL.md file", text)
	}

	log.Printf("Successfully loaded skill frontmatter ,name = %s,description=%s", skillMeta.Name, skillMeta.Description)
	return &skillMeta, nil
}

func parseSkillMD(skillDir string) (*Skill, error) {
	skill := &Skill{}
	info, err := os.Stat(skillDir)
	if err != nil {
		return skill, fmt.Errorf("skill directory '%s' stat error:%w", skillDir, err)
	}
	if !info.IsDir() {
		return skill, fmt.Errorf("skill directory '%s' is not a directory", skillDir)
	}

	var skillMD string
	for _, name := range []string{"SKILL.md", "skill.md"} {
		p := filepath.Join(skillDir, name)
		if _, err := os.Stat(p); err == nil {
			skillMD = p
			break
		}
	}
	if skillMD == "" {
		return skill, fmt.Errorf("SKILL.md not found in '%s'", skillDir)
	}

	file, err := os.Open(skillMD)
	if err != nil {
		return skill, fmt.Errorf("open skill file '%s' error:%w", skillMD, err)
	}
	defer func() {
		_ = file.Close()
	}()
	sc := bufio.NewScanner(file)
	if !sc.Scan() {
		return skill, err
	}

	if strings.TrimSpace(sc.Text()) != "---" {
		return skill, fmt.Errorf("failed to parse %s, invalid frontmatter", skillMD)
	}

	var frontmatterLines []string
	var contextLines []string
	isFrontmatterLines := true

	for sc.Scan() {
		line := sc.Text()
		if isFrontmatterLines && strings.TrimSpace(line) == "---" {
			if len(frontmatterLines) == 0 {
				return nil, fmt.Errorf("failed to parse %s, empty frontmatter", skillMD)
			}
			isFrontmatterLines = false
			frontmatterStr := strings.Join(frontmatterLines, "\n")
			skill.Frontmatter, err = parseFrontmatter([]byte(frontmatterStr))
			if err != nil {
				return skill, fmt.Errorf("failed to parse frontmatter '%s', %w", skillMD, err)
			}
			if err := skill.Frontmatter.Validate(); err != nil {
				return skill, fmt.Errorf("skill %s frontmatter invalid :%w", skillDir, err)
			}
			continue
		}
		if isFrontmatterLines {
			frontmatterLines = append(frontmatterLines, line)
		} else {
			contextLines = append(contextLines, line)
		}
	}

	if isFrontmatterLines {
		return skill, fmt.Errorf("failed to parse %s, missing closing frontmatter separator", skillMD)
	}

	skill.Instructions = strings.Join(contextLines, "\n")
	skill.SkillMDPath = skillMD

	log.Printf("Successfully loaded skill %s locally from %s", skill.Frontmatter.Name, skillDir)
	return skill, nil
}

func LoadSkillFromDir(skillDir string) (*Skill, error) {
	var skill *Skill
	var err error
	abs, err := filepath.Abs(skillDir)
	if err != nil {
		return skill, err
	}
	skill, err = parseSkillMD(abs)
	if err != nil {
		return skill, fmt.Errorf("failed to parse skill directory '%s' error:%w", skillDir, err)
	}

	if base := filepath.Base(abs); base != skill.Name() {
		return skill, fmt.Errorf("skill name '%s' does not match directory name '%s'", skill.Name(), base)
	}
	refs, err := loadDir(filepath.Join(abs, "references"))
	if err != nil {
		return skill, fmt.Errorf("failed to load references dir '%s' error:%w", abs, err)
	}
	assets, err := loadDir(filepath.Join(abs, "assets"))
	if err != nil {
		return skill, fmt.Errorf("failed to load assets dir '%s' error:%w", abs, err)
	}
	rawScripts, err := loadDir(filepath.Join(abs, "scripts"))
	if err != nil {
		return skill, fmt.Errorf("failed to load scripts dir '%s' error:%w", abs, err)
	}
	scripts := make(map[string]*Script, len(rawScripts))
	for k, v := range rawScripts {
		scripts[k] = &Script{Src: v}
	}
	skill.Resources = &Resources{
		References: refs,
		Assets:     assets,
		Scripts:    scripts,
	}

	return skill, nil
}

func ReadSkillProperties(skillDir string) (*Frontmatter, error) {
	abs, err := filepath.Abs(skillDir)
	if err != nil {
		return nil, err
	}
	skill, err := parseSkillMD(abs)
	if err != nil {
		return nil, err
	}
	return skill.Frontmatter, nil
}
