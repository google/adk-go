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

package classifier_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/classifier"
)

type constantClassifier struct {
	model string
}

func (c constantClassifier) Name() string { return c.model }

func (c constantClassifier) Classify(
	_ context.Context,
	_ any,
	questions map[string]classifier.Question,
) (*classifier.Response, error) {
	answers := make(map[string]classifier.Answer, len(questions))
	for name := range questions {
		answers[name] = classifier.BinaryAnswer{Probability: 0.5}
	}
	return &classifier.Response{
		ModelVersion: c.model,
		Answers:      answers,
	}, nil
}

func TestCustomClassifierImplementsInterface(t *testing.T) {
	var clf classifier.Classifier = constantClassifier{model: "constant"}

	resp, err := clf.Classify(t.Context(), "x", map[string]classifier.Question{
		"risky": classifier.Binary{},
	})
	if err != nil {
		t.Fatalf("Classify() unexpected error: %v", err)
	}
	want := &classifier.Response{
		ModelVersion: "constant",
		Answers: map[string]classifier.Answer{
			"risky": classifier.BinaryAnswer{Probability: 0.5},
		},
	}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("Classify() mismatch (-want +got):\n%s", diff)
	}
	if got := clf.Name(); got != "constant" {
		t.Errorf("Name() = %q, want %q", got, "constant")
	}
}

func TestValidateQuestion(t *testing.T) {
	type customDoc struct {
		Task    string   `json:"task"`
		Context []string `json:"context"`
	}

	tests := []struct {
		name    string
		q       classifier.Question
		wantErr bool
	}{
		{
			name: "choice_valid_with_nil_and_string_criteria",
			q: classifier.Choice{
				Instructions: "Which department?",
				Criteria:     map[string]any{"billing": "Invoices", "technical": nil},
			},
		},
		{
			name: "choice_pointer_valid",
			q: &classifier.Choice{
				Criteria: map[string]any{"a": nil, "b": nil},
			},
		},
		{
			name: "choice_json_instructions_and_criteria",
			q: classifier.Choice{
				Instructions: customDoc{Task: "route", Context: []string{"ticket"}},
				Criteria: map[string]any{
					"a": map[string]any{"examples": []string{"x"}},
					"b": nil,
				},
			},
		},
		{
			name:    "nil_question",
			q:       nil,
			wantErr: true,
		},
		{
			name:    "typed_nil_choice_pointer",
			q:       (*classifier.Choice)(nil),
			wantErr: true,
		},
		{
			name:    "typed_nil_binary_pointer",
			q:       (*classifier.Binary)(nil),
			wantErr: true,
		},
		{
			name:    "typed_nil_score_pointer",
			q:       (*classifier.Score)(nil),
			wantErr: true,
		},
		{
			name:    "choice_fewer_than_two_labels",
			q:       classifier.Choice{Criteria: map[string]any{"only": nil}},
			wantErr: true,
		},
		{
			name:    "choice_empty_criteria",
			q:       classifier.Choice{},
			wantErr: true,
		},
		{
			name: "choice_rejects_bare_number_instructions",
			q: classifier.Choice{
				Instructions: 42,
				Criteria:     map[string]any{"a": nil, "b": nil},
			},
			wantErr: true,
		},
		{
			name: "choice_rejects_non_serializable_criteria",
			q: classifier.Choice{
				Criteria: map[string]any{"a": make(chan int), "b": nil},
			},
			wantErr: true,
		},
		{
			name: "choice_rejects_typed_nil_criteria_value",
			q: classifier.Choice{
				Criteria: map[string]any{"a": (*customDoc)(nil), "b": "ok"},
			},
			wantErr: true,
		},
		{
			name: "choice_rejects_raw_byte_slice_instructions",
			q: classifier.Choice{
				Instructions: []byte("abc"),
				Criteria:     map[string]any{"a": nil, "b": nil},
			},
			wantErr: true,
		},
		{
			name: "choice_accepts_json_raw_message",
			q: classifier.Choice{
				Instructions: json.RawMessage(`{"task":"route"}`),
				Criteria:     map[string]any{"a": nil, "b": nil},
			},
		},
		{
			name: "binary_empty_valid",
			q:    classifier.Binary{},
		},
		{
			name: "binary_with_criteria_valid",
			q:    classifier.Binary{Instructions: "Deletes data?", Criteria: "Deletes user data."},
		},
		{
			name: "binary_pointer_valid",
			q:    &classifier.Binary{Criteria: []string{"delete", "drop"}},
		},
		{
			name:    "binary_rejects_bool_criteria",
			q:       classifier.Binary{Criteria: true},
			wantErr: true,
		},
		{
			name:    "binary_rejects_typed_nil_instructions",
			q:       classifier.Binary{Instructions: (*customDoc)(nil)},
			wantErr: true,
		},
		{
			name: "score_valid",
			q: classifier.Score{
				Instructions: "How urgent?",
				Criteria:     []any{"low", map[string]any{"level": "high"}},
			},
		},
		{
			name: "score_pointer_valid",
			q:    &classifier.Score{Criteria: []any{"low", "high"}},
		},
		{
			name:    "score_fewer_than_two_levels",
			q:       classifier.Score{Criteria: []any{"low"}},
			wantErr: true,
		},
		{
			name:    "score_rejects_nil_level",
			q:       classifier.Score{Criteria: []any{"low", nil}},
			wantErr: true,
		},
		{
			name:    "score_rejects_bare_number_level",
			q:       classifier.Score{Criteria: []any{"low", 2}},
			wantErr: true,
		},
		{
			name: "score_rejects_bare_bool_instructions",
			q: classifier.Score{
				Instructions: false,
				Criteria:     []any{"low", "high"},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := classifier.ValidateQuestion(tc.q)
			if tc.wantErr {
				if !errors.Is(err, classifier.ErrInvalidQuestion) {
					t.Fatalf("ValidateQuestion() error = %v, want errors.Is(ErrInvalidQuestion)", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateQuestion() unexpected error: %v", err)
			}
		})
	}
}

func TestQuestionJSONRoundTripAndValidation(t *testing.T) {
	questions := map[string]classifier.Question{
		"route": classifier.Choice{
			Instructions: "Which team?",
			Criteria:     map[string]any{"a": "first", "b": nil},
		},
		"risky": classifier.Binary{
			Criteria: "Deletes data.",
		},
		"urgency": classifier.Score{
			Instructions: "How urgent?",
			Criteria:     []any{"low", "high"},
		},
	}

	for name, q := range questions {
		t.Run("roundtrip_"+name, func(t *testing.T) {
			b, err := json.Marshal(q)
			if err != nil {
				t.Fatalf("json.Marshal() unexpected error: %v", err)
			}
			got, err := classifier.UnmarshalQuestion(b)
			if err != nil {
				t.Fatalf("UnmarshalQuestion() unexpected error: %v", err)
			}
			if diff := cmp.Diff(q, got); diff != "" {
				t.Errorf("Question round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}

	invalidJSON := []struct {
		name string
		raw  string
	}{
		{name: "misspelled_field_on_choice", raw: `{"type":"choice","instruction":"typo","criteria":{"a":null,"b":null}}`},
		{name: "single_label_choice", raw: `{"type":"choice","criteria":{"only":null}}`},
		{name: "single_level_score", raw: `{"type":"score","criteria":["low"]}`},
		{name: "invalid_binary_criteria", raw: `{"type":"binary","criteria":true}`},
		{name: "extra_field_on_binary", raw: `{"type":"binary","extra":1}`},
		{name: "unknown_question_type", raw: `{"type":"other"}`},
		{name: "malformed_json", raw: `{`},
	}
	for _, tc := range invalidJSON {
		t.Run("invalid_"+tc.name, func(t *testing.T) {
			if _, err := classifier.UnmarshalQuestion([]byte(tc.raw)); !errors.Is(err, classifier.ErrInvalidQuestion) {
				t.Fatalf("UnmarshalQuestion() error = %v, want errors.Is(ErrInvalidQuestion)", err)
			}
		})
	}

	// Direct MarshalJSON validation on invalid concrete question structs.
	for _, badQ := range []classifier.Question{
		classifier.Choice{Criteria: map[string]any{"only": nil}},
		classifier.Binary{Criteria: 123},
		classifier.Score{Criteria: []any{"only"}},
	} {
		if _, err := json.Marshal(badQ); !errors.Is(err, classifier.ErrInvalidQuestion) {
			t.Errorf("json.Marshal(%T) error = %v, want ErrInvalidQuestion", badQ, err)
		}
	}

	// Direct UnmarshalJSON type-mismatch and null checks on concrete question structs.
	var cq classifier.Choice
	if err := json.Unmarshal([]byte(`{"type":"binary"}`), &cq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Choice.UnmarshalJSON(binary) error = %v, want ErrInvalidQuestion", err)
	}
	if err := json.Unmarshal([]byte(`{"type":null,"criteria":{"a":null,"b":null}}`), &cq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Choice.UnmarshalJSON(type:null) error = %v, want ErrInvalidQuestion", err)
	}
	var bq classifier.Binary
	if err := json.Unmarshal([]byte(`{"type":"score","criteria":["a","b"]}`), &bq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Binary.UnmarshalJSON(score) error = %v, want ErrInvalidQuestion", err)
	}
	if err := bq.UnmarshalJSON([]byte(`null`)); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Binary.UnmarshalJSON(null) error = %v, want ErrInvalidQuestion", err)
	}
	if err := json.Unmarshal([]byte(`{"type":null}`), &bq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Binary.UnmarshalJSON(type:null) error = %v, want ErrInvalidQuestion", err)
	}
	var sq classifier.Score
	if err := json.Unmarshal([]byte(`{"type":"binary"}`), &sq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Score.UnmarshalJSON(binary) error = %v, want ErrInvalidQuestion", err)
	}
	if err := json.Unmarshal([]byte(`{"type":null,"criteria":["a","b"]}`), &sq); !errors.Is(err, classifier.ErrInvalidQuestion) {
		t.Errorf("Score.UnmarshalJSON(type:null) error = %v, want ErrInvalidQuestion", err)
	}
}

func TestValidateInput(t *testing.T) {
	tests := []struct {
		name    string
		input   any
		wantErr bool
	}{
		{name: "string", input: "hello"},
		{name: "empty_string", input: ""},
		{name: "map", input: map[string]any{"tool": "sql", "args": map[string]any{"q": "SELECT 1"}}},
		{name: "slice", input: []any{"turn 1", "turn 2"}},
		{name: "struct", input: struct{ Text string }{Text: "ticket"}},
		{name: "genai_content_ptr", input: genai.NewContentFromText("hi", genai.RoleUser)},
		{name: "genai_content_val", input: genai.Content{Parts: []*genai.Part{{Text: "hi"}}}},
		{name: "nil", input: nil, wantErr: true},
		{name: "typed_nil_content", input: (*genai.Content)(nil), wantErr: true},
		{name: "typed_nil_slice", input: ([]string)(nil), wantErr: true},
		{name: "bare_int", input: 42, wantErr: true},
		{name: "bare_bool", input: true, wantErr: true},
		{name: "byte_slice", input: []byte("raw"), wantErr: true},
		{name: "nan_in_map", input: map[string]float64{"x": math.NaN()}, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := classifier.ValidateInput(tc.input)
			if tc.wantErr {
				if !errors.Is(err, classifier.ErrInvalidInput) {
					t.Fatalf("ValidateInput() error = %v, want errors.Is(ErrInvalidInput)", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateInput() unexpected error: %v", err)
			}
		})
	}
}

func TestValidateAnswer(t *testing.T) {
	neg := -0.1
	over := 1.5
	nan := math.NaN()
	inf := math.Inf(1)
	validConf := 0.8

	tests := []struct {
		name    string
		ans     classifier.Answer
		wantErr bool
	}{
		{
			name: "binary_valid",
			ans:  classifier.BinaryAnswer{Probability: 0.25, Confidence: &validConf},
		},
		{
			name: "binary_pointer_valid",
			ans:  &classifier.BinaryAnswer{Probability: 0.0},
		},
		{
			name: "choice_valid",
			ans: classifier.ChoiceAnswer{
				Choice:        "a",
				Probabilities: map[string]float64{"a": 1.0, "b": 0.0},
				Confidence:    &validConf,
			},
		},
		{
			name: "choice_pointer_valid",
			ans: &classifier.ChoiceAnswer{
				Choice:        "a",
				Probabilities: map[string]float64{"a": 1.0},
			},
		},
		{
			name: "score_valid",
			ans: classifier.ScoreAnswer{
				Score:         1.5,
				Probabilities: map[int]float64{0: 0.0, 1: 0.5, 2: 0.5},
				Confidence:    &validConf,
			},
		},
		{
			name: "score_pointer_valid",
			ans: &classifier.ScoreAnswer{
				Score:         0.0,
				Probabilities: map[int]float64{0: 1.0},
			},
		},
		{name: "nil_answer", ans: nil, wantErr: true},
		{name: "typed_nil_binary", ans: (*classifier.BinaryAnswer)(nil), wantErr: true},
		{name: "typed_nil_choice", ans: (*classifier.ChoiceAnswer)(nil), wantErr: true},
		{name: "typed_nil_score", ans: (*classifier.ScoreAnswer)(nil), wantErr: true},
		{name: "binary_prob_negative", ans: classifier.BinaryAnswer{Probability: -0.1}, wantErr: true},
		{name: "binary_prob_above_one", ans: classifier.BinaryAnswer{Probability: 1.5}, wantErr: true},
		{name: "binary_prob_nan", ans: classifier.BinaryAnswer{Probability: nan}, wantErr: true},
		{name: "binary_prob_inf", ans: classifier.BinaryAnswer{Probability: inf}, wantErr: true},
		{name: "binary_conf_negative", ans: classifier.BinaryAnswer{Probability: 0.5, Confidence: &neg}, wantErr: true},
		{name: "binary_conf_above_one", ans: classifier.BinaryAnswer{Probability: 0.5, Confidence: &over}, wantErr: true},
		{name: "choice_empty_probs", ans: classifier.ChoiceAnswer{Choice: "a"}, wantErr: true},
		{name: "choice_prob_negative", ans: classifier.ChoiceAnswer{Choice: "a", Probabilities: map[string]float64{"a": -0.1}}, wantErr: true},
		{name: "choice_prob_above_one", ans: classifier.ChoiceAnswer{Choice: "a", Probabilities: map[string]float64{"a": 1.5}}, wantErr: true},
		{name: "choice_conf_above_one", ans: classifier.ChoiceAnswer{Choice: "a", Probabilities: map[string]float64{"a": 1.0}, Confidence: &over}, wantErr: true},
		{name: "score_negative", ans: classifier.ScoreAnswer{Score: -0.1, Probabilities: map[int]float64{0: 1.0}}, wantErr: true},
		{name: "score_nan", ans: classifier.ScoreAnswer{Score: nan, Probabilities: map[int]float64{0: 1.0}}, wantErr: true},
		{name: "score_inf", ans: classifier.ScoreAnswer{Score: inf, Probabilities: map[int]float64{0: 1.0}}, wantErr: true},
		{name: "score_empty_probs", ans: classifier.ScoreAnswer{Score: 0}, wantErr: true},
		{name: "score_negative_level", ans: classifier.ScoreAnswer{Score: 0, Probabilities: map[int]float64{-1: 1.0}}, wantErr: true},
		{name: "score_prob_above_one", ans: classifier.ScoreAnswer{Score: 0, Probabilities: map[int]float64{0: 1.2}}, wantErr: true},
		{name: "score_conf_negative", ans: classifier.ScoreAnswer{Score: 0, Probabilities: map[int]float64{0: 1.0}, Confidence: &neg}, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := classifier.ValidateAnswer(tc.ans)
			if tc.wantErr {
				if !errors.Is(err, classifier.ErrInvalidAnswer) {
					t.Fatalf("ValidateAnswer() error = %v, want errors.Is(ErrInvalidAnswer)", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateAnswer() unexpected error: %v", err)
			}
		})
	}
}

func TestResponseAndAnswerJSONRoundTrip(t *testing.T) {
	conf9 := 0.9
	conf5 := 0.5
	orig := classifier.Response{
		ModelVersion: "m",
		Answers: map[string]classifier.Answer{
			"route": classifier.ChoiceAnswer{
				Choice:        "a",
				Probabilities: map[string]float64{"a": 0.9, "b": 0.1},
			},
			"risky": classifier.BinaryAnswer{
				Probability: 0.2,
				Confidence:  &conf9,
			},
			"urgency": classifier.ScoreAnswer{
				Score:         0.75,
				Probabilities: map[int]float64{0: 0.25, 1: 0.75},
				Confidence:    &conf5,
			},
		},
	}

	encoded, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("json.Marshal() unexpected error: %v", err)
	}

	var restored classifier.Response
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatalf("json.Unmarshal() unexpected error: %v", err)
	}

	if diff := cmp.Diff(orig, restored); diff != "" {
		t.Errorf("JSON round-trip mismatch (-want +got):\n%s", diff)
	}

	// Standalone Answer round-trip and type-mismatch checks.
	choiceBytes, err := json.Marshal(orig.Answers["route"])
	if err != nil {
		t.Fatalf("json.Marshal(ChoiceAnswer) error: %v", err)
	}
	var gotChoice classifier.ChoiceAnswer
	if err := json.Unmarshal(choiceBytes, &gotChoice); err != nil {
		t.Fatalf("json.Unmarshal(ChoiceAnswer) error: %v", err)
	}
	if diff := cmp.Diff(orig.Answers["route"], gotChoice); diff != "" {
		t.Errorf("ChoiceAnswer round-trip mismatch (-want +got):\n%s", diff)
	}

	var wrongBinary classifier.BinaryAnswer
	if err := json.Unmarshal(choiceBytes, &wrongBinary); !errors.Is(err, classifier.ErrInvalidAnswer) {
		t.Errorf("BinaryAnswer.UnmarshalJSON(choice) error = %v, want ErrInvalidAnswer", err)
	}
	var wrongScore classifier.ScoreAnswer
	if err := json.Unmarshal(choiceBytes, &wrongScore); !errors.Is(err, classifier.ErrInvalidAnswer) {
		t.Errorf("ScoreAnswer.UnmarshalJSON(choice) error = %v, want ErrInvalidAnswer", err)
	}
	var wrongChoice classifier.ChoiceAnswer
	if err := json.Unmarshal([]byte(`{"type":"binary","probability":0.5}`), &wrongChoice); !errors.Is(err, classifier.ErrInvalidAnswer) {
		t.Errorf("ChoiceAnswer.UnmarshalJSON(binary) error = %v, want ErrInvalidAnswer", err)
	}
}

func TestResponseUnmarshalRejectsInvalidJSON(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{name: "extra_field_on_response", raw: `{"answers": {}, "extra": 1}`},
		{name: "missing_answers", raw: `{"model_version": "m"}`},
		{name: "extra_field_on_binary", raw: `{"answers": {"q": {"type": "binary", "probability": 0.5, "extra": 1}}}`},
		{name: "extra_field_on_choice", raw: `{"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "extra": 1}}}`},
		{name: "extra_field_on_score", raw: `{"answers": {"q": {"type": "score", "score": 0, "probabilities": {"0": 1.0}, "extra": 1}}}`},
		{name: "missing_binary_probability", raw: `{"answers": {"q": {"type": "binary"}}}`},
		{name: "missing_choice_field", raw: `{"answers": {"q": {"type": "choice", "probabilities": {"a": 1.0}}}}`},
		{name: "missing_score_field", raw: `{"answers": {"q": {"type": "score", "probabilities": {"0": 1.0}}}}`},
		{name: "null_choice_probability", raw: `{"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": null}}}}`},
		{name: "null_score_probability", raw: `{"answers": {"q": {"type": "score", "score": 0, "probabilities": {"0": null}}}}`},
		{name: "out_of_range_binary_probability", raw: `{"answers": {"q": {"type": "binary", "probability": 1.2}}}`},
		{name: "out_of_range_choice_probability", raw: `{"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.2}}}}`},
		{name: "out_of_range_score_probability", raw: `{"answers": {"q": {"type": "score", "score": 0, "probabilities": {"0": 1.2}}}}`},
		{name: "unknown_answer_type", raw: `{"answers": {"q": {"type": "other"}}}`},
		{name: "non_object_answer_entry", raw: `{"answers": {"q": 123}}`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var resp classifier.Response
			err := json.Unmarshal([]byte(tc.raw), &resp)
			if !errors.Is(err, classifier.ErrInvalidAnswer) {
				t.Fatalf("json.Unmarshal() error = %v, want errors.Is(ErrInvalidAnswer)", err)
			}
		})
	}

	t.Run("direct_unmarshal_rejects_trailing_tokens_and_null_type", func(t *testing.T) {
		var resp classifier.Response
		err := resp.UnmarshalJSON([]byte(`{"answers": {}}{"extra": 1}`))
		if !errors.Is(err, classifier.ErrInvalidAnswer) {
			t.Fatalf("UnmarshalJSON() error = %v, want errors.Is(ErrInvalidAnswer)", err)
		}
		var ba classifier.BinaryAnswer
		if err := json.Unmarshal([]byte(`{"type":null,"probability":0.5}`), &ba); !errors.Is(err, classifier.ErrInvalidAnswer) {
			t.Errorf("BinaryAnswer.UnmarshalJSON(type:null) error = %v, want ErrInvalidAnswer", err)
		}
		if err := ba.UnmarshalJSON([]byte(`null`)); !errors.Is(err, classifier.ErrInvalidAnswer) {
			t.Errorf("BinaryAnswer.UnmarshalJSON(null) error = %v, want ErrInvalidAnswer", err)
		}
	})
}

func TestResponseMarshalRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name string
		resp classifier.Response
	}{
		{name: "nil_answers_map", resp: classifier.Response{}},
		{name: "nil_answer_entry", resp: classifier.Response{Answers: map[string]classifier.Answer{"q": nil}}},
		{name: "typed_nil_choice_ptr", resp: classifier.Response{Answers: map[string]classifier.Answer{"q": (*classifier.ChoiceAnswer)(nil)}}},
		{name: "invalid_binary_prob", resp: classifier.Response{Answers: map[string]classifier.Answer{"q": classifier.BinaryAnswer{Probability: 1.5}}}},
		{name: "invalid_choice_prob", resp: classifier.Response{Answers: map[string]classifier.Answer{"q": &classifier.ChoiceAnswer{Choice: "a", Probabilities: map[string]float64{"a": -0.1}}}}},
		{name: "invalid_score_prob", resp: classifier.Response{Answers: map[string]classifier.Answer{"q": &classifier.ScoreAnswer{Score: 0, Probabilities: map[int]float64{0: 1.5}}}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := json.Marshal(tc.resp)
			if !errors.Is(err, classifier.ErrInvalidAnswer) {
				t.Fatalf("json.Marshal() error = %v, want errors.Is(ErrInvalidAnswer)", err)
			}
		})
	}
}
