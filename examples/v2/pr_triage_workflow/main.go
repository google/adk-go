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

// Real-world demo: an automated triage pipeline for a GitHub PR.
//
//	START → fetch_pr ─┬─ security_scan ─┐
//	                  ├─ check_breaking ─┼─ join → decide
//	                  └─ suggest_labels ─┘
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// PR is the structured representation of a pull request that flows
// through the triage pipeline.
type PR struct {
	Number       int
	Title        string
	Author       string
	Files        []string
	Description  string
	LinesChanged int
}

// fetchAttempts simulates a flaky upstream — the first call fails 503,
// retry succeeds. Lets us exercise WithRetry on a real-feeling failure.
var fetchAttempts atomic.Int32

// fakeGitHubAPI returns canned PRs by number so the demo runs offline.
// Replace with a real `gh` call or the GitHub API in production.
func fakeGitHubAPI(num int) (PR, error) {
	prs := map[int]PR{
		1234: {
			Number:       1234,
			Title:        "Add OAuth2 token refresh to auth middleware",
			Author:       "alice",
			Files:        []string{"auth/middleware.go", "auth/oauth.go", "auth/oauth_test.go"},
			Description:  "Implements automatic token refresh when the upstream returns 401. Adds a wrapper around oauth2.Config.TokenSource.",
			LinesChanged: 240,
		},
	}
	if pr, ok := prs[num]; ok {
		return pr, nil
	}
	return PR{}, errors.New("PR not found")
}

func main() {
	fetchPR := workflow.Func("fetch_pr",
		func(_ *workflow.NodeContext, _ any) (PR, error) {
			n := fetchAttempts.Add(1)
			if n == 1 {
				return PR{}, errors.New("503 Service Unavailable (transient)")
			}
			return fakeGitHubAPI(1234)
		},
		workflow.WithRetry(&workflow.RetryConfig{
			MaxAttempts:  3,
			InitialDelay: 5 * time.Millisecond,
			MaxDelay:     50 * time.Millisecond,
			Jitter:       -1,
		}),
	)

	type SecurityReport struct {
		Findings []string
		Risk     string // "low" | "medium" | "high"
	}
	securityScan := workflow.Func("security_scan",
		func(_ *workflow.NodeContext, pr PR) (SecurityReport, error) {
			rep := SecurityReport{Risk: "low"}
			text := strings.ToLower(pr.Title + " " + pr.Description)
			if strings.Contains(text, "auth") || strings.Contains(text, "token") || strings.Contains(text, "credential") {
				rep.Findings = append(rep.Findings, "touches authentication path; review token-storage assumptions")
				rep.Risk = "medium"
			}
			return rep, nil
		},
		// Cap the scan so a misbehaving SAST integration doesn't hang the pipeline.
		workflow.WithTimeout(5*time.Second),
	)

	type BreakingReport struct {
		Likely bool
		Why    string
	}
	checkBreaking := workflow.Func("check_breaking",
		func(_ *workflow.NodeContext, pr PR) (BreakingReport, error) {
			for _, f := range pr.Files {
				if strings.HasSuffix(f, "/api.go") || strings.Contains(f, "/proto/") {
					return BreakingReport{Likely: true, Why: "modifies public API surface: " + f}, nil
				}
			}
			if pr.LinesChanged > 1000 {
				return BreakingReport{Likely: true, Why: "very large diff (>1000 lines) — likely cross-cutting"}, nil
			}
			return BreakingReport{}, nil
		})

	suggestLabels := workflow.Func("suggest_labels",
		func(_ *workflow.NodeContext, pr PR) ([]string, error) {
			out := []string{}
			text := strings.ToLower(pr.Title + " " + pr.Description)
			if strings.Contains(text, "auth") || strings.Contains(text, "oauth") {
				out = append(out, "area/auth")
			}
			if strings.Contains(text, "test") {
				out = append(out, "tests")
			}
			if pr.LinesChanged < 50 {
				out = append(out, "size/XS")
			} else if pr.LinesChanged < 200 {
				out = append(out, "size/S")
			} else if pr.LinesChanged < 500 {
				out = append(out, "size/M")
			} else {
				out = append(out, "size/L")
			}
			return out, nil
		})

	join := workflow.Join("triage_join")

	type Decision struct {
		PRNumber          int
		Action            string // "auto-merge-after-review" | "request-changes" | "manual-triage"
		Labels            []string
		SecurityRisk      string
		BreakingPotential bool
		Reason            string
	}
	decide := workflow.Func("decide",
		func(_ *workflow.NodeContext, in any) (Decision, error) {
			m := in.(map[string]any)
			sec := m["security_scan"].(SecurityReport)
			brk := m["check_breaking"].(BreakingReport)
			labels := m["suggest_labels"].([]string)

			d := Decision{
				PRNumber:          1234,
				Labels:            labels,
				SecurityRisk:      sec.Risk,
				BreakingPotential: brk.Likely,
			}
			switch {
			case sec.Risk == "high" || brk.Likely:
				d.Action = "manual-triage"
				if brk.Likely {
					d.Reason = "potential breaking change: " + brk.Why
				} else {
					d.Reason = "high security risk: " + strings.Join(sec.Findings, "; ")
				}
			case sec.Risk == "medium":
				d.Action = "request-changes"
				d.Reason = "security review recommended: " + strings.Join(sec.Findings, "; ")
			default:
				d.Action = "auto-merge-after-review"
				d.Reason = "no risk signals; routine review"
			}
			return d, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "pr_triage",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, fetchPR),
			workflow.Connect(fetchPR, securityScan),
			workflow.Connect(fetchPR, checkBreaking),
			workflow.Connect(fetchPR, suggestLabels),
			workflow.Connect(securityScan, join),
			workflow.Connect(checkBreaking, join),
			workflow.Connect(suggestLabels, join),
			workflow.Connect(join, decide),
		},
	})
	if err != nil {
		log.Fatalf("workflow.New: %v", err)
	}

	wfAgent, err := wf.AsAgent()
	if err != nil {
		log.Fatalf("AsAgent: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "pr_triage_demo",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("runner.New: %v", err)
	}

	fmt.Println("triaging PR #1234...")
	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "triage 1234"}}}
	for ev, err := range r.Run(context.Background(), "ops-bot", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run err: %v", err)
		}
		if ev.Author == "decide" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			d := ev.Actions.NodeInfo.Output.(Decision)
			fmt.Println()
			fmt.Printf("PR #%d decision: %s\n", d.PRNumber, d.Action)
			fmt.Printf("  reason          : %s\n", d.Reason)
			fmt.Printf("  security risk   : %s\n", d.SecurityRisk)
			fmt.Printf("  breaking change : %v\n", d.BreakingPotential)
			fmt.Printf("  suggested labels: %s\n", strings.Join(d.Labels, ", "))
		}
	}
	fmt.Printf("\nfetch_pr attempts: %d (one transient 503, then success)\n", fetchAttempts.Load())
}
