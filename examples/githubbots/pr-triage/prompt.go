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

package main

import (
	_ "embed"
	"strings"
)

//go:embed prompt_instruction.txt
var promptTemplate string

// contextSectionEnabled is the instruction block used when the context-request
// tool is registered. It is a constant so that every word the model is told
// about that tool ships with the binary.
const contextSectionEnabled = `## Second: does the description carry enough context?

A reviewer arriving cold should be able to tell what the change does and why. If
the description clearly does not give them that, call ` + "`request_more_context`" + ` once
with the pieces that are missing, chosen from this fixed list:

%ITEMS%

You choose only WHICH pieces to ask for. The wording of the comment is fixed and
written by the repository, so do not attempt to phrase anything yourself.

Ask only when it genuinely helps. A short description on an obvious one-line fix
is fine. A typo correction, a dependency bump, or a self-explanatory rename does
not need a paragraph. Do not ask for something the description or the changed
file paths already answer, and never ask merely because the author is new or the
prose is not polished. When in doubt, ask for nothing.`

// contextSectionDisabled is used when REQUEST_CONTEXT is off: the tool is not
// registered, so the prompt must not offer it.
const contextSectionDisabled = `## Second: nothing

Asking the author for more context is disabled for this repository. Assignment
is your only action.`

// renderPrompt substitutes the configuration placeholders into the embedded
// prompt and returns a finished instruction string.
//
// IMPORTANT: llmagent.Config.Instruction treats {placeholder} tokens as
// session-state references and errors on unknown keys. renderPrompt must
// therefore leave zero stray braces; this is enforced by a test. Config-derived
// values are already charset-validated at load time, and stripped again here so
// a future config path cannot reintroduce a brace.
func renderPrompt(cfg *Config) string {
	section := contextSectionDisabled
	if cfg.contextRequestsEnabled() {
		section = strings.ReplaceAll(contextSectionEnabled, "%ITEMS%", bulletList(contextItemKeys()))
	}
	r := strings.NewReplacer(
		"{OWNER}", stripBraces(cfg.Owner),
		"{REPO}", stripBraces(cfg.Repo),
		"{COMPONENTS}", stripBraces(bulletList(cfg.components())),
		"{CONTEXT_SECTION}", stripBraces(section),
	)
	return r.Replace(promptTemplate)
}

// bulletList renders values as a markdown list, one per line.
func bulletList(values []string) string {
	var b strings.Builder
	for i, v := range values {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString("- " + v)
	}
	return b.String()
}

var braceStripper = strings.NewReplacer("{", "", "}", "")

// stripBraces removes brace characters so a substituted value cannot be parsed
// as an llmagent session-state placeholder.
func stripBraces(s string) string { return braceStripper.Replace(s) }
