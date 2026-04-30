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

// Package eval is the evaluation harness for ADK agents. Mirrors a
// subset of adk-python's google.adk.evaluation.
//
// Phase 9 ships the data model and core scorers (exact match, contains,
// prefix); LLM-as-judge and BLEU/ROUGE scorers ship in dedicated
// subpackages so their dependencies stay lazy.
//
// Run an eval set against an agent via Runner.Run; the harness records
// per-case Outcomes and rolls them up into a Report.
package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
)

// Case is one input/expected-output pair to test against an agent.
type Case struct {
	// ID is a stable identifier used for reporting and selective re-runs.
	ID string `json:"id"`

	// Description is human-readable context for the case.
	Description string `json:"description,omitempty"`

	// Input is the user's prompt for this case.
	Input string `json:"input"`

	// ExpectedOutput is the reference answer used by Scorer.Score.
	ExpectedOutput string `json:"expected_output"`

	// Tags optionally categorize the case (e.g. "regression", "edge-case").
	Tags []string `json:"tags,omitempty"`

	// Metadata is arbitrary per-case data (e.g. seed, sub-domain).
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Set is a collection of evaluation cases.
type Set struct {
	Name  string `json:"name"`
	Cases []Case `json:"cases"`
}

// LoadSet reads an eval set from a JSON file at path.
func LoadSet(path string) (*Set, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return DecodeSet(f)
}

// DecodeSet reads a Set from r as JSON.
func DecodeSet(r io.Reader) (*Set, error) {
	var set Set
	if err := json.NewDecoder(r).Decode(&set); err != nil {
		return nil, fmt.Errorf("eval: decode set: %w", err)
	}
	return &set, nil
}

// Outcome records the result of running one Case.
type Outcome struct {
	Case     Case          `json:"case"`
	Output   string        `json:"output"`
	Score    float64       `json:"score"`
	Passed   bool          `json:"passed"`
	Reason   string        `json:"reason,omitempty"`
	Duration time.Duration `json:"duration_ms"`
	Err      string        `json:"error,omitempty"`
}

// Report aggregates per-case outcomes.
type Report struct {
	SetName        string    `json:"set_name"`
	Outcomes       []Outcome `json:"outcomes"`
	PassCount      int       `json:"pass_count"`
	FailCount      int       `json:"fail_count"`
	ErrorCount     int       `json:"error_count"`
	AverageScore   float64   `json:"average_score"`
	StartedAt      time.Time `json:"started_at"`
	CompletedAt    time.Time `json:"completed_at"`
}

// PassRate returns the fraction of cases that passed.
func (r *Report) PassRate() float64 {
	if total := r.PassCount + r.FailCount + r.ErrorCount; total > 0 {
		return float64(r.PassCount) / float64(total)
	}
	return 0
}

// Scorer computes a similarity score in [0, 1] between the agent's
// output and the case's expected output. Returning a score >= the
// configured threshold marks the case passed; otherwise it fails.
type Scorer interface {
	Name() string
	Score(ctx context.Context, c Case, output string) (score float64, reason string, err error)
}

// AgentRunner abstracts how the harness runs a case against an agent.
// Production code uses the runner package; tests use a stub.
type AgentRunner interface {
	Run(ctx context.Context, userID, sessionID, input string) (string, error)
}

// Runner drives an evaluation: for each Case in the Set, invoke the
// AgentRunner and Score the output. Concurrency = 1 in this Phase 9
// baseline; parallel evaluation is a future enhancement.
type Runner struct {
	Agent     AgentRunner
	Scorer    Scorer
	Threshold float64 // Score >= Threshold means passed; default 0.5.
}

// Run runs the eval set sequentially and returns a Report.
func (r *Runner) Run(ctx context.Context, set *Set) (*Report, error) {
	if r.Agent == nil {
		return nil, fmt.Errorf("eval: Runner.Agent is nil")
	}
	if r.Scorer == nil {
		return nil, fmt.Errorf("eval: Runner.Scorer is nil")
	}
	threshold := r.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	rep := &Report{SetName: set.Name, StartedAt: time.Now()}
	var totalScore float64
	for _, c := range set.Cases {
		start := time.Now()
		out, runErr := r.Agent.Run(ctx, "eval", c.ID, c.Input)
		elapsed := time.Since(start)
		oc := Outcome{Case: c, Output: out, Duration: elapsed}

		if runErr != nil {
			oc.Err = runErr.Error()
			rep.ErrorCount++
		} else {
			score, reason, err := r.Scorer.Score(ctx, c, out)
			if err != nil {
				oc.Err = err.Error()
				rep.ErrorCount++
			} else {
				oc.Score = score
				oc.Reason = reason
				oc.Passed = score >= threshold
				if oc.Passed {
					rep.PassCount++
				} else {
					rep.FailCount++
				}
				totalScore += score
			}
		}
		rep.Outcomes = append(rep.Outcomes, oc)
	}
	rep.CompletedAt = time.Now()
	if total := len(rep.Outcomes) - rep.ErrorCount; total > 0 {
		rep.AverageScore = totalScore / float64(total)
	}
	return rep, nil
}

// ExactMatchScorer scores 1.0 when output equals expected (after
// trimming whitespace) and 0.0 otherwise.
type ExactMatchScorer struct{}

// Name implements Scorer.
func (ExactMatchScorer) Name() string { return "exact_match" }

// Score implements Scorer.
func (ExactMatchScorer) Score(_ context.Context, c Case, output string) (float64, string, error) {
	if strings.TrimSpace(output) == strings.TrimSpace(c.ExpectedOutput) {
		return 1.0, "exact match", nil
	}
	return 0.0, "no match", nil
}

// ContainsScorer scores 1.0 when output contains every substring listed
// in Case.ExpectedOutput (split on newlines), 0.0 otherwise.
type ContainsScorer struct{}

// Name implements Scorer.
func (ContainsScorer) Name() string { return "contains" }

// Score implements Scorer.
func (ContainsScorer) Score(_ context.Context, c Case, output string) (float64, string, error) {
	for _, want := range strings.Split(c.ExpectedOutput, "\n") {
		w := strings.TrimSpace(want)
		if w == "" {
			continue
		}
		if !strings.Contains(output, w) {
			return 0.0, fmt.Sprintf("missing substring: %q", w), nil
		}
	}
	return 1.0, "all substrings present", nil
}

// silence unused-import warnings during incremental builds; placeholder
// for future agent.Agent / genai.Content integration as the runner
// adapter lands.
var _ = agent.RunConfig{}
var _ = genai.RoleUser
