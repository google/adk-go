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
	"strings"
	"testing"
)

func TestSkill_Valid(t *testing.T) {
	tests := []struct {
		name    string
		skill   Frontmatter
		wantErr bool
	}{
		{
			name: "valid skill",
			skill: Frontmatter{
				Name:        "pdf-processing",
				Description: "Extracts text and tables from PDF files.",
			},
			wantErr: false,
		},
		{
			name: "valid skill with compatibility",
			skill: Frontmatter{
				Name:          "data-analysis",
				Description:   "Analyzes data.",
				Compatibility: "Requires python 3.9",
			},
			wantErr: false,
		},
		{
			name: "invalid name - empty",
			skill: Frontmatter{
				Name:        "",
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid name - too long",
			skill: Frontmatter{
				Name:        strings.Repeat("a", 65),
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid name - uppercase",
			skill: Frontmatter{
				Name:        "PDF-Processing",
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid name - starts with hyphen",
			skill: Frontmatter{
				Name:        "-pdf",
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid name - ends with hyphen",
			skill: Frontmatter{
				Name:        "pdf-",
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid name - consecutive hyphens",
			skill: Frontmatter{
				Name:        "pdf--processing",
				Description: "Valid description",
			},
			wantErr: true,
		},
		{
			name: "invalid description - empty",
			skill: Frontmatter{
				Name:        "valid-name",
				Description: "",
			},
			wantErr: true,
		},
		{
			name: "invalid description - too long",
			skill: Frontmatter{
				Name:        "valid-name",
				Description: strings.Repeat("a", 1025),
			},
			wantErr: true,
		},
		{
			name: "invalid compatibility - too long",
			skill: Frontmatter{
				Name:          "valid-name",
				Description:   "Valid description",
				Compatibility: strings.Repeat("a", 501),
			},
			wantErr: true,
		},
		{
			name: "valid skill with license",
			skill: Frontmatter{
				Name:        "valid-name",
				Description: "Valid description",
				License:     "MIT",
			},
			wantErr: false,
		},
		{
			name: "valid skill with metadata",
			skill: Frontmatter{
				Name:        "valid-name",
				Description: "Valid description",
				Metadata: map[string]string{
					"author":  "example-org",
					"version": "1.0",
				},
			},
			wantErr: false,
		},
		{
			name: "valid skill with allowed tools",
			skill: Frontmatter{
				Name:         "valid-name",
				Description:  "Valid description",
				AllowedTools: "Bash(git:*) Bash(jq:*) Read",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.skill.Validate(); (err != nil) != tt.wantErr {
				t.Errorf("Skill.Valid() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
