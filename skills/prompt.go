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
	"html"
	"strings"
)

//type SkillLike interface {
//	Name() string
//	Description() string
//}

func FormatSkillsAsXML(skills []*Skill) string {
	if len(skills) == 0 {
		return "<available_skills>\n</available_skills>"
	}

	var b strings.Builder
	b.WriteString("<available_skills>\n")

	for _, item := range skills {
		b.WriteString("<skill>\n")
		b.WriteString("<name>\n")
		b.WriteString(html.EscapeString(item.Name()))
		b.WriteString("\n</name>\n")
		b.WriteString("<description>\n")
		b.WriteString(html.EscapeString(item.Description()))
		b.WriteString("\n</description>\n")
		b.WriteString("</skill>\n")
	}

	b.WriteString("</available_skills>")
	return b.String()
}
