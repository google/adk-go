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

package eval_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/adk/eval"
)

// stubAgent returns a deterministic mapping from input -> output.
type stubAgent struct {
	answers map[string]string
	err     error
}

func (s *stubAgent) Run(_ context.Context, _, _, input string) (string, error) {
	if s.err != nil {
		return "", s.err
	}
	return s.answers[input], nil
}

func TestExactMatchScorer(t *testing.T) {
	s := eval.ExactMatchScorer{}
	score, _, err := s.Score(context.Background(), eval.Case{ExpectedOutput: "hello"}, "hello")
	if err != nil || score != 1.0 {
		t.Errorf("match: score=%v err=%v", score, err)
	}
	score, _, _ = s.Score(context.Background(), eval.Case{ExpectedOutput: "hello"}, "world")
	if score != 0.0 {
		t.Errorf("no-match score = %v, want 0", score)
	}
}

func TestContainsScorer(t *testing.T) {
	s := eval.ContainsScorer{}
	c := eval.Case{ExpectedOutput: "alpha\nbeta"}
	score, _, _ := s.Score(context.Background(), c, "the alpha and beta words")
	if score != 1.0 {
		t.Errorf("score = %v, want 1.0", score)
	}
	score, reason, _ := s.Score(context.Background(), c, "alpha only")
	if score != 0.0 {
		t.Errorf("score = %v", score)
	}
	if !strings.Contains(reason, "missing") {
		t.Errorf("reason = %q", reason)
	}
}

func TestRunner_Run_AggregatesPassFail(t *testing.T) {
	set := &eval.Set{
		Name: "test_set",
		Cases: []eval.Case{
			{ID: "c1", Input: "q1", ExpectedOutput: "a1"},
			{ID: "c2", Input: "q2", ExpectedOutput: "a2"},
			{ID: "c3", Input: "q3", ExpectedOutput: "a3"},
		},
	}
	agent := &stubAgent{answers: map[string]string{"q1": "a1", "q2": "wrong", "q3": "a3"}}
	r := &eval.Runner{Agent: agent, Scorer: eval.ExactMatchScorer{}, Threshold: 1.0}
	rep, err := r.Run(context.Background(), set)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.PassCount != 2 || rep.FailCount != 1 {
		t.Errorf("pass=%d fail=%d, want 2/1", rep.PassCount, rep.FailCount)
	}
	if rep.PassRate() < 0.65 || rep.PassRate() > 0.67 {
		t.Errorf("pass rate = %v, want ~0.667", rep.PassRate())
	}
}

func TestRunner_Run_AgentErrorIncrementsErrorCount(t *testing.T) {
	set := &eval.Set{Name: "errs", Cases: []eval.Case{{ID: "c1", ExpectedOutput: "x"}}}
	r := &eval.Runner{
		Agent:  &stubAgent{err: errors.New("boom")},
		Scorer: eval.ExactMatchScorer{},
	}
	rep, _ := r.Run(context.Background(), set)
	if rep.ErrorCount != 1 {
		t.Errorf("ErrorCount = %d, want 1", rep.ErrorCount)
	}
	if rep.Outcomes[0].Err == "" {
		t.Error("expected Err to be populated")
	}
}

func TestRunner_RejectsNilAgent(t *testing.T) {
	r := &eval.Runner{Scorer: eval.ExactMatchScorer{}}
	if _, err := r.Run(context.Background(), &eval.Set{}); err == nil {
		t.Error("expected error for nil agent")
	}
}

func TestDecodeSet(t *testing.T) {
	json := `{"name":"s","cases":[{"id":"c1","input":"q","expected_output":"a"}]}`
	set, err := eval.DecodeSet(strings.NewReader(json))
	if err != nil {
		t.Fatalf("DecodeSet: %v", err)
	}
	if set.Name != "s" || len(set.Cases) != 1 || set.Cases[0].ID != "c1" {
		t.Errorf("set = %+v", set)
	}
}
