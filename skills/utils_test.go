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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var mockSkill = &Skill{
	Frontmatter: &Frontmatter{
		Name:        "test-skill",
		Description: "A test skill for unit testing",
		Metadata: map[string]string{
			"version": "1.0.0",
		},
	},
	Instructions: "---\nname: test-skill\ndescription: A test skill for unit testing\n---\n\n# Test Skill\n\nThis is a test skill.",
	Resources: &Resources{
		Scripts: map[string]*Script{
			"test_script.py": {
				Src: `print("Hello World")`,
			},
		},
		References: map[string]string{
			"ref.md": "# Reference\nThis is a reference file.",
		},
		Assets: map[string]string{
			"data.json":       `{"key": "value"}`,
			"subdir/file.txt": "content in subdir",
		},
	},
}

func TestLoadSkillFromDir(t *testing.T) {
	// 1. Create a temporary directory
	tmpDir := t.TempDir()

	// 2. Write the mock skill to the temporary directory
	err := mockSkill.WriteSkill(tmpDir)
	require.NoError(t, err)

	// The skill is written to tmpDir/test-skill
	skillDir := filepath.Join(tmpDir, mockSkill.Name())

	// 3. Test LoadSkillFromDir
	loadedSkill, err := LoadSkillFromDir(skillDir)
	require.NoError(t, err)
	require.NotNil(t, loadedSkill)

	// Verify Frontmatter
	assert.Equal(t, mockSkill.Frontmatter.Name, loadedSkill.Frontmatter.Name)
	assert.Equal(t, mockSkill.Frontmatter.Description, loadedSkill.Frontmatter.Description)

	// Verify Instructions
	// Note: parseSkillMD separates frontmatter and content.
	// The original Instructions string includes frontmatter, but loadedSkill.Instructions should only have the content.
	expectedInstructions := "\n# Test Skill\n\nThis is a test skill."
	assert.Equal(t, expectedInstructions, loadedSkill.Instructions)

	// Verify Resources
	require.NotNil(t, loadedSkill.Resources)

	// Verify Scripts
	script, ok := loadedSkill.Resources.GetScript("test_script.py")
	assert.True(t, ok)
	assert.Equal(t, mockSkill.Resources.Scripts["test_script.py"].Src, script.Src)

	// Verify References
	ref, ok := loadedSkill.Resources.GetReference("ref.md")
	assert.True(t, ok)
	assert.Equal(t, mockSkill.Resources.References["ref.md"], ref)

	// Verify Assets
	asset, ok := loadedSkill.Resources.GetAsset("data.json")
	assert.True(t, ok)
	assert.Equal(t, mockSkill.Resources.Assets["data.json"], asset)

	// Verify Assets in subdir (loadDir is recursive)
	assetSub, ok := loadedSkill.Resources.GetAsset("subdir/file.txt")
	assert.True(t, ok)
	assert.Equal(t, mockSkill.Resources.Assets["subdir/file.txt"], assetSub)
}

func TestReadSkillProperties(t *testing.T) {
	tmpDir := t.TempDir()
	err := mockSkill.WriteSkill(tmpDir)
	require.NoError(t, err)

	skillDir := filepath.Join(tmpDir, mockSkill.Name())

	frontmatter, err := ReadSkillProperties(skillDir)
	require.NoError(t, err)
	require.NotNil(t, frontmatter)

	assert.Equal(t, mockSkill.Frontmatter.Name, frontmatter.Name)
	assert.Equal(t, mockSkill.Frontmatter.Description, frontmatter.Description)
}

func TestLoadSkillFromDir_Errors(t *testing.T) {
	tmpDir := t.TempDir()

	// Test non-existent directory
	_, err := LoadSkillFromDir(filepath.Join(tmpDir, "non-existent"))
	assert.Error(t, err)

	// Test empty directory (no SKILL.md)
	emptyDir := filepath.Join(tmpDir, "empty-skill")
	_ = mockSkill.WriteSkill(tmpDir) // write normal skill first
	// overwrite with empty dir
	// actually let's just make a new dir
	err = os.Mkdir(emptyDir, 0o755)
	require.NoError(t, err)

	_, err = LoadSkillFromDir(emptyDir)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "SKILL.md not found")
}
