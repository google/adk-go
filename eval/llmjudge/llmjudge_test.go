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

package llmjudge_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/eval"
	"google.golang.org/adk/eval/llmjudge"
	"google.golang.org/adk/model"
)

type stubLLM struct {
	name     string
	response string
	err      error
}

func (s *stubLLM) Name() string { return s.name }
func (s *stubLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if s.err != nil {
			yield(nil, s.err)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{{Text: s.response}}},
		}, nil)
	}
}

func TestNew_RequiresLLM(t *testing.T) {
	if _, err := llmjudge.New(llmjudge.Config{}); err == nil {
		t.Error("expected error for nil LLM")
	}
}

func TestScore_StrictJSON(t *testing.T) {
	llm := &stubLLM{name: "stub", response: `{"score": 0.9, "reason": "matches well"}`}
	s, err := llmjudge.New(llmjudge.Config{LLM: llm})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	score, reason, err := s.Score(context.Background(), eval.Case{Input: "q", ExpectedOutput: "x"}, "y")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 0.9 || reason != "matches well" {
		t.Errorf("score=%v reason=%q", score, reason)
	}
}

func TestScore_StripsCodeFences(t *testing.T) {
	llm := &stubLLM{name: "stub", response: "```json\n{\"score\": 0.5, \"reason\": \"meh\"}\n```"}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	score, reason, err := s.Score(context.Background(), eval.Case{}, "")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 0.5 || reason != "meh" {
		t.Errorf("score=%v reason=%q", score, reason)
	}
}

func TestScore_RegexFallback(t *testing.T) {
	// Some models prepend prose before the JSON object. The regex
	// fallback handles "score: 0.7" / "reason: ..." patterns.
	llm := &stubLLM{name: "stub", response: `My evaluation: score: 0.7, reason: "partial overlap"`}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	score, reason, err := s.Score(context.Background(), eval.Case{}, "")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 0.7 {
		t.Errorf("score = %v", score)
	}
	if reason == "" {
		t.Error("expected reason from regex fallback")
	}
}

func TestScore_ClampsToRange(t *testing.T) {
	llm := &stubLLM{name: "stub", response: `{"score": 1.5, "reason": "x"}`}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	score, _, _ := s.Score(context.Background(), eval.Case{}, "")
	if score != 1.0 {
		t.Errorf("score = %v, want 1.0 (clamped)", score)
	}
}

func TestScore_PropagatesLLMError(t *testing.T) {
	want := errors.New("upstream broken")
	llm := &stubLLM{name: "stub", err: want}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	_, _, err := s.Score(context.Background(), eval.Case{}, "")
	if !errors.Is(err, want) {
		t.Errorf("err = %v, want wraps %v", err, want)
	}
}

func TestScore_ErrorOnEmptyResponse(t *testing.T) {
	llm := &stubLLM{name: "stub", response: ""}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	_, _, err := s.Score(context.Background(), eval.Case{}, "")
	if err == nil {
		t.Error("expected error for empty response")
	}
}

func TestScore_ErrorOnUnparseable(t *testing.T) {
	llm := &stubLLM{name: "stub", response: "I cannot parse this"}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	_, _, err := s.Score(context.Background(), eval.Case{}, "")
	if err == nil {
		t.Error("expected error for unparseable response")
	}
}

func TestName_Default(t *testing.T) {
	llm := &stubLLM{name: "stub"}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm})
	if s.Name() != "llm_judge" {
		t.Errorf("Name = %q", s.Name())
	}
}

func TestName_Override(t *testing.T) {
	llm := &stubLLM{name: "stub"}
	s, _ := llmjudge.New(llmjudge.Config{LLM: llm, Name: "my_judge"})
	if s.Name() != "my_judge" {
		t.Errorf("Name = %q", s.Name())
	}
}
