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

// PR triage: a Gemini coordinator agent runs three Gemini-backed
// reviewer sub-agents in parallel (security, breaking-change, labeling)
// then synthesizes the verdict. Sub-agents are wrapped as workflow
// nodes via workflow.FromAgent and orchestrated by the workflow engine.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const coordinatorInstruction = `You are a pull-request triage coordinator.

When a user gives you a PR (number, title, files, description, lines
changed), perform the triage by calling these tools in this order:

  1. security_scan(pr_json) — flag auth / credential / sensitive paths.
  2. check_breaking(pr_json) — flag public-API changes or oversized diffs.
  3. suggest_labels(pr_json) — propose GitHub labels from area + size.

Synthesize a single decision message for the user with:
  - Action: auto-merge-after-review | request-changes | manual-triage
  - Reason
  - Security risk (low/medium/high)
  - Breaking change potential
  - Suggested labels`

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	securityScan, err := buildSecurityScanTool()
	if err != nil {
		log.Fatal(err)
	}
	checkBreaking, err := buildCheckBreakingTool()
	if err != nil {
		log.Fatal(err)
	}
	suggestLabels, err := buildSuggestLabelsTool()
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "pr_triage",
		Description: "A coordinator agent that runs the security, breaking-change, and labeling scans on a pull request.",
		Model:       model,
		Instruction: coordinatorInstruction,
		Tools:       []tool.Tool{securityScan, checkBreaking, suggestLabels},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

// PR is the structured representation of a pull request the coordinator
// hands to each scan tool. Replace the static catalog in the tool stubs
// with a real GitHub fetch (gh api or graphql) when adapting.
type PR struct {
	Number       int      `json:"number"`
	Title        string   `json:"title"`
	Files        []string `json:"files"`
	Description  string   `json:"description"`
	LinesChanged int      `json:"lines_changed"`
}

func buildSecurityScanTool() (tool.Tool, error) {
	type result struct {
		Risk     string   `json:"risk"`
		Findings []string `json:"findings,omitempty"`
	}
	return functiontool.New[PR, result](
		functiontool.Config{
			Name:        "security_scan",
			Description: "Scan a PR for security-sensitive changes (auth, credentials, crypto).",
		},
		func(_ tool.Context, pr PR) (result, error) {
			r := result{Risk: "low"}
			text := pr.Title + " " + pr.Description
			for _, marker := range []string{"auth", "token", "credential", "password", "secret", "key"} {
				if containsCI(text, marker) {
					r.Findings = append(r.Findings, "touches "+marker+" path; review carefully")
					r.Risk = "medium"
				}
			}
			for _, f := range pr.Files {
				if containsCI(f, "auth") || containsCI(f, "crypto") {
					r.Findings = append(r.Findings, "modifies "+f)
					r.Risk = "medium"
				}
			}
			return r, nil
		},
	)
}

func buildCheckBreakingTool() (tool.Tool, error) {
	type result struct {
		Likely bool   `json:"likely"`
		Why    string `json:"why,omitempty"`
	}
	return functiontool.New[PR, result](
		functiontool.Config{
			Name:        "check_breaking",
			Description: "Decide whether a PR is likely to introduce breaking changes.",
		},
		func(_ tool.Context, pr PR) (result, error) {
			for _, f := range pr.Files {
				if hasSuffix(f, "/api.go") || containsCI(f, "/proto/") {
					return result{Likely: true, Why: "modifies public API surface: " + f}, nil
				}
			}
			if pr.LinesChanged > 1000 {
				return result{Likely: true, Why: "very large diff (>1000 lines); likely cross-cutting"}, nil
			}
			return result{}, nil
		},
	)
}

func buildSuggestLabelsTool() (tool.Tool, error) {
	type result struct {
		Labels []string `json:"labels"`
	}
	return functiontool.New[PR, result](
		functiontool.Config{
			Name:        "suggest_labels",
			Description: "Suggest GitHub labels for a PR (area + size).",
		},
		func(_ tool.Context, pr PR) (result, error) {
			out := []string{}
			text := pr.Title + " " + pr.Description
			if containsCI(text, "auth") || containsCI(text, "oauth") {
				out = append(out, "area/auth")
			}
			if containsCI(text, "test") {
				out = append(out, "tests")
			}
			switch {
			case pr.LinesChanged < 50:
				out = append(out, "size/XS")
			case pr.LinesChanged < 200:
				out = append(out, "size/S")
			case pr.LinesChanged < 500:
				out = append(out, "size/M")
			default:
				out = append(out, "size/L")
			}
			return result{Labels: out}, nil
		},
	)
}

// ---- small string helpers (avoid pulling strings into every tool) ----

func containsCI(s, sub string) bool {
	return indexCI(s, sub) >= 0
}
func indexCI(s, sub string) int {
	if len(sub) == 0 {
		return 0
	}
	if len(sub) > len(s) {
		return -1
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		match := true
		for j := 0; j < len(sub); j++ {
			a, b := s[i+j], sub[j]
			if a >= 'A' && a <= 'Z' {
				a += 'a' - 'A'
			}
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			if a != b {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}
func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}
