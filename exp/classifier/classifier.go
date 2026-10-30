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

package classifier

import (
	"bytes"
	"context"
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"

	"google.golang.org/genai"
)

var (
	// ErrInvalidQuestion is returned when a [Question] or question map fails validation.
	ErrInvalidQuestion = errors.New("classifier: invalid question")
	// ErrInvalidAnswer is returned when an [Answer] or [Response] fails validation.
	ErrInvalidAnswer = errors.New("classifier: invalid answer")
	// ErrInvalidInput is returned when an input passed to [Classifier.Classify]
	// is nil, is not JSON-serializable as a string, object, or array, contains
	// google.golang.org/genai struct values other than a single top-level
	// [genai.Content] or [*genai.Content], or is unsupported by the classifier
	// implementation.
	ErrInvalidInput = errors.New("classifier: invalid input")
)

// Classifier answers typed questions about an input with probability distributions.
//
// Classify evaluates every question in questions against input and returns a
// [Response] whose Answers map uses the same question keys. Every
// implementation accepts strings and JSON-serializable maps, structs, slices,
// and arrays; [genai.Content] and [*genai.Content] carry multimodal parts for
// implementations whose underlying model supports them, and fail with
// [ErrInvalidInput] otherwise. Other google.golang.org/genai struct values,
// such as a []*genai.Content history or a bare [genai.Part], are rejected
// wherever they appear in JSON input (see [ValidateInput]). Invalid questions
// fail with [ErrInvalidQuestion].
//
// Implementations must be safe for concurrent use. Name mirrors adk-python's
// BaseClassifier.model attribute using ADK Go's Name() method convention
// (as on [google.golang.org/adk/v2/model.LLM]).
type Classifier interface {
	Name() string
	Classify(ctx context.Context, input any, questions map[string]Question) (*Response, error)
}

// Question is a typed question answered by a [Classifier]: [Choice], [Binary],
// or [Score].
type Question interface {
	isQuestion()
}

// Choice asks a [Classifier] to pick one of two or more named labels.
//
// Instructions and non-nil Criteria values must be JSON content: a string, or a
// JSON-serializable map, struct, slice, or array. Criteria must contain at
// least two labels; a label's description may be nil.
type Choice struct {
	Instructions any
	Criteria     map[string]any
}

func (Choice) isQuestion() {}

// MarshalJSON validates q and encodes it with a "type": "choice" discriminator,
// matching adk-python's Choice JSON format.
func (q Choice) MarshalJSON() ([]byte, error) {
	if err := validateChoice(q); err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type         string         `json:"type"`
		Instructions any            `json:"instructions,omitempty"`
		Criteria     map[string]any `json:"criteria"`
	}{
		Type:         "choice",
		Instructions: q.Instructions,
		Criteria:     q.Criteria,
	})
}

// UnmarshalJSON decodes a JSON Choice question, rejecting unknown fields and
// validating Criteria and Instructions.
func (q *Choice) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type         json.RawMessage `json:"type"`
		Instructions any             `json:"instructions"`
		Criteria     map[string]any  `json:"criteria"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidQuestion, err)
	}
	if err := checkDiscriminator(wire.Type, "choice", ErrInvalidQuestion); err != nil {
		return err
	}
	parsed := Choice{
		Instructions: wire.Instructions,
		Criteria:     wire.Criteria,
	}
	if err := validateChoice(parsed); err != nil {
		return err
	}
	*q = parsed
	return nil
}

// Binary asks a [Classifier] for the probability that a statement is true.
//
// Instructions and Criteria, when non-nil, must be JSON content: a string, or a
// JSON-serializable map, struct, slice, or array. Criteria describes what
// counts as true; anything else counts as false.
type Binary struct {
	Instructions any
	Criteria     any
}

func (Binary) isQuestion() {}

// MarshalJSON validates q and encodes it with a "type": "binary" discriminator,
// matching adk-python's Binary JSON format.
func (q Binary) MarshalJSON() ([]byte, error) {
	if err := validateBinary(q); err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type         string `json:"type"`
		Instructions any    `json:"instructions,omitempty"`
		Criteria     any    `json:"criteria,omitempty"`
	}{
		Type:         "binary",
		Instructions: q.Instructions,
		Criteria:     q.Criteria,
	})
}

// UnmarshalJSON decodes a JSON Binary question, rejecting unknown fields and
// validating Criteria and Instructions.
func (q *Binary) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type         json.RawMessage `json:"type"`
		Instructions any             `json:"instructions"`
		Criteria     any             `json:"criteria"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidQuestion, err)
	}
	if err := checkDiscriminator(wire.Type, "binary", ErrInvalidQuestion); err != nil {
		return err
	}
	parsed := Binary{
		Instructions: wire.Instructions,
		Criteria:     wire.Criteria,
	}
	if err := validateBinary(parsed); err != nil {
		return err
	}
	*q = parsed
	return nil
}

// Score asks a [Classifier] to place an input on an ordered rubric starting at
// level 0.
//
// Instructions, when non-nil, and every element of Criteria must be JSON
// content: a string, or a JSON-serializable map, struct, slice, or array.
// Criteria must contain at least two levels, and no element may be nil.
type Score struct {
	Instructions any
	Criteria     []any
}

func (Score) isQuestion() {}

// MarshalJSON validates q and encodes it with a "type": "score" discriminator,
// matching adk-python's Score JSON format.
func (q Score) MarshalJSON() ([]byte, error) {
	if err := validateScore(q); err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type         string `json:"type"`
		Instructions any    `json:"instructions,omitempty"`
		Criteria     []any  `json:"criteria"`
	}{
		Type:         "score",
		Instructions: q.Instructions,
		Criteria:     q.Criteria,
	})
}

// UnmarshalJSON decodes a JSON Score question, rejecting unknown fields and
// validating Criteria and Instructions.
func (q *Score) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type         json.RawMessage `json:"type"`
		Instructions any             `json:"instructions"`
		Criteria     []any           `json:"criteria"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidQuestion, err)
	}
	if err := checkDiscriminator(wire.Type, "score", ErrInvalidQuestion); err != nil {
		return err
	}
	parsed := Score{
		Instructions: wire.Instructions,
		Criteria:     wire.Criteria,
	}
	if err := validateScore(parsed); err != nil {
		return err
	}
	*q = parsed
	return nil
}

// UnmarshalQuestion decodes a JSON question by its "type" discriminator
// ("choice", "binary", or "score"), matching adk-python's discriminated
// Question union.
func UnmarshalQuestion(data []byte) (Question, error) {
	var disc struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &disc); err != nil {
		return nil, fmt.Errorf("%w: invalid question JSON", ErrInvalidQuestion)
	}
	switch disc.Type {
	case "choice":
		var q Choice
		if err := q.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return q, nil
	case "binary":
		var q Binary
		if err := q.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return q, nil
	case "score":
		var q Score
		if err := q.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return q, nil
	default:
		return nil, fmt.Errorf("%w: unsupported question type", ErrInvalidQuestion)
	}
}

// Answer is a [Classifier]'s answer to a [Question]: [ChoiceAnswer],
// [BinaryAnswer], or [ScoreAnswer].
type Answer interface {
	isAnswer()
}

// ChoiceAnswer is the answer to a [Choice] question.
type ChoiceAnswer struct {
	// Choice is the most likely label.
	Choice string
	// Probabilities maps each label in the question to its probability in [0, 1].
	Probabilities map[string]float64
	// Confidence is the provider's confidence in [0, 1], or nil when unreported.
	Confidence *float64
}

func (ChoiceAnswer) isAnswer() {}

// MarshalJSON validates a and encodes it with a "type": "choice" discriminator,
// matching adk-python's ChoiceAnswer wire format.
func (a ChoiceAnswer) MarshalJSON() ([]byte, error) {
	if err := validateChoiceAnswer(a); err != nil {
		return nil, err
	}
	return json.Marshal(choiceAnswerJSON{
		Type:          "choice",
		Choice:        a.Choice,
		Probabilities: a.Probabilities,
		Confidence:    a.Confidence,
	})
}

// UnmarshalJSON decodes a JSON ChoiceAnswer, rejecting unknown fields, null
// probability entries, and probabilities outside [0, 1].
func (a *ChoiceAnswer) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type          json.RawMessage     `json:"type"`
		Choice        *string             `json:"choice"`
		Probabilities map[string]*float64 `json:"probabilities"`
		Confidence    *float64            `json:"confidence"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidAnswer, err)
	}
	if err := checkDiscriminator(wire.Type, "choice", ErrInvalidAnswer); err != nil {
		return err
	}
	if wire.Choice == nil {
		return fmt.Errorf("%w: choice answer requires choice", ErrInvalidAnswer)
	}
	probs := make(map[string]float64, len(wire.Probabilities))
	for label, p := range wire.Probabilities {
		if p == nil {
			// The label comes from the decoded payload, so it stays out of the error.
			return fmt.Errorf("%w: choice probability must not be null", ErrInvalidAnswer)
		}
		probs[label] = *p
	}
	parsed := ChoiceAnswer{
		Choice:        *wire.Choice,
		Probabilities: probs,
		Confidence:    wire.Confidence,
	}
	if err := validateChoiceAnswer(parsed); err != nil {
		return err
	}
	*a = parsed
	return nil
}

// BinaryAnswer is the answer to a [Binary] question.
type BinaryAnswer struct {
	// Probability is the probability in [0, 1] that the statement is true.
	Probability float64
	// Confidence is the provider's confidence in [0, 1], or nil when unreported.
	Confidence *float64
}

func (BinaryAnswer) isAnswer() {}

// MarshalJSON validates a and encodes it with a "type": "binary" discriminator,
// matching adk-python's BinaryAnswer wire format.
func (a BinaryAnswer) MarshalJSON() ([]byte, error) {
	if err := validateBinaryAnswer(a); err != nil {
		return nil, err
	}
	return json.Marshal(binaryAnswerJSON{
		Type:        "binary",
		Probability: a.Probability,
		Confidence:  a.Confidence,
	})
}

// UnmarshalJSON decodes a JSON BinaryAnswer, rejecting unknown fields and
// probabilities outside [0, 1].
func (a *BinaryAnswer) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type        json.RawMessage `json:"type"`
		Probability *float64        `json:"probability"`
		Confidence  *float64        `json:"confidence"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidAnswer, err)
	}
	if err := checkDiscriminator(wire.Type, "binary", ErrInvalidAnswer); err != nil {
		return err
	}
	if wire.Probability == nil {
		return fmt.Errorf("%w: binary answer requires probability", ErrInvalidAnswer)
	}
	parsed := BinaryAnswer{
		Probability: *wire.Probability,
		Confidence:  wire.Confidence,
	}
	if err := validateBinaryAnswer(parsed); err != nil {
		return err
	}
	*a = parsed
	return nil
}

// ScoreAnswer is the answer to a [Score] question.
type ScoreAnswer struct {
	// Score is the expected rubric level, weighted by Probabilities.
	Score float64
	// Probabilities maps each 0-based rubric level to its probability in [0, 1].
	Probabilities map[int]float64
	// Confidence is the provider's confidence in [0, 1], or nil when unreported.
	Confidence *float64
}

func (ScoreAnswer) isAnswer() {}

// MarshalJSON validates a and encodes it with a "type": "score" discriminator,
// matching adk-python's ScoreAnswer wire format.
func (a ScoreAnswer) MarshalJSON() ([]byte, error) {
	if err := validateScoreAnswer(a); err != nil {
		return nil, err
	}
	return json.Marshal(scoreAnswerJSON{
		Type:          "score",
		Score:         a.Score,
		Probabilities: a.Probabilities,
		Confidence:    a.Confidence,
	})
}

// UnmarshalJSON decodes a JSON ScoreAnswer, rejecting unknown fields, null
// probability entries, and out-of-range values.
func (a *ScoreAnswer) UnmarshalJSON(data []byte) error {
	var wire struct {
		Type          json.RawMessage  `json:"type"`
		Score         *float64         `json:"score"`
		Probabilities map[int]*float64 `json:"probabilities"`
		Confidence    *float64         `json:"confidence"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidAnswer, err)
	}
	if err := checkDiscriminator(wire.Type, "score", ErrInvalidAnswer); err != nil {
		return err
	}
	if wire.Score == nil {
		return fmt.Errorf("%w: score answer requires score", ErrInvalidAnswer)
	}
	probs := make(map[int]float64, len(wire.Probabilities))
	for level, p := range wire.Probabilities {
		if p == nil {
			return fmt.Errorf("%w: probabilities[%d] must not be null", ErrInvalidAnswer, level)
		}
		probs[level] = *p
	}
	parsed := ScoreAnswer{
		Score:         *wire.Score,
		Probabilities: probs,
		Confidence:    wire.Confidence,
	}
	if err := validateScoreAnswer(parsed); err != nil {
		return err
	}
	*a = parsed
	return nil
}

// Response holds the answers produced by a single [Classifier.Classify] call.
type Response struct {
	// Answers maps each caller-supplied question name to its [Answer].
	Answers map[string]Answer
	// ModelVersion is the concrete model version reported by the provider, or
	// empty when unreported.
	ModelVersion string
}

type responseJSON struct {
	Answers      map[string]json.RawMessage `json:"answers"`
	ModelVersion string                     `json:"model_version,omitempty"`
}

// MarshalJSON validates r and encodes it in the JSON format of adk-python's
// ClassifierResponse.
func (r Response) MarshalJSON() ([]byte, error) {
	if r.Answers == nil {
		return nil, fmt.Errorf("%w: answers map is nil", ErrInvalidAnswer)
	}
	raw := make(map[string]json.RawMessage, len(r.Answers))
	for name, ans := range r.Answers {
		b, err := marshalAnswer(ans)
		if err != nil {
			// Names may come from a decoded Response, so they stay out of the error.
			return nil, err
		}
		raw[name] = b
	}
	return json.Marshal(responseJSON{
		Answers:      raw,
		ModelVersion: r.ModelVersion,
	})
}

// UnmarshalJSON decodes a JSON ClassifierResponse, dispatching each entry in
// answers by its "type" field ("choice", "binary", or "score") and rejecting
// unknown fields or out-of-range probabilities.
func (r *Response) UnmarshalJSON(data []byte) error {
	var wire struct {
		Answers      map[string]json.RawMessage `json:"answers"`
		ModelVersion string                     `json:"model_version"`
	}
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidAnswer, err)
	}
	if wire.Answers == nil {
		return fmt.Errorf("%w: missing answers object", ErrInvalidAnswer)
	}
	answers := make(map[string]Answer, len(wire.Answers))
	for name, raw := range wire.Answers {
		ans, err := unmarshalAnswer(raw)
		if err != nil {
			// The name comes from the decoded payload, so it stays out of the error.
			return err
		}
		answers[name] = ans
	}
	r.Answers = answers
	r.ModelVersion = wire.ModelVersion
	return nil
}

// ValidateQuestion checks that q is a non-nil [Choice], [Binary], or [Score]
// whose Criteria and Instructions satisfy the classifier contract.
func ValidateQuestion(q Question) error {
	switch v := q.(type) {
	case nil:
		return fmt.Errorf("%w: question is nil", ErrInvalidQuestion)
	case Choice:
		return validateChoice(v)
	case *Choice:
		if v == nil {
			return fmt.Errorf("%w: question is nil", ErrInvalidQuestion)
		}
		return validateChoice(*v)
	case Binary:
		return validateBinary(v)
	case *Binary:
		if v == nil {
			return fmt.Errorf("%w: question is nil", ErrInvalidQuestion)
		}
		return validateBinary(*v)
	case Score:
		return validateScore(v)
	case *Score:
		if v == nil {
			return fmt.Errorf("%w: question is nil", ErrInvalidQuestion)
		}
		return validateScore(*v)
	default:
		return fmt.Errorf("%w: unsupported question type %T", ErrInvalidQuestion, q)
	}
}

// ValidateAnswer checks that a is a non-nil [ChoiceAnswer], [BinaryAnswer], or
// [ScoreAnswer] with finite probabilities and confidence in [0, 1].
//
// Unlike adk-python's Pydantic answer models, ValidateAnswer requires non-empty
// Probabilities on ChoiceAnswer and ScoreAnswer, and non-negative
// ScoreAnswer.Score and rubric level keys (matching 0-based Score rubrics).
func ValidateAnswer(a Answer) error {
	switch v := a.(type) {
	case nil:
		return fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
	case ChoiceAnswer:
		return validateChoiceAnswer(v)
	case *ChoiceAnswer:
		if v == nil {
			return fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
		return validateChoiceAnswer(*v)
	case BinaryAnswer:
		return validateBinaryAnswer(v)
	case *BinaryAnswer:
		if v == nil {
			return fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
		return validateBinaryAnswer(*v)
	case ScoreAnswer:
		return validateScoreAnswer(v)
	case *ScoreAnswer:
		if v == nil {
			return fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
		return validateScoreAnswer(*v)
	default:
		return fmt.Errorf("%w: unsupported answer type %T", ErrInvalidAnswer, a)
	}
}

// ValidateInput checks that input is a valid classifier input: a
// [*genai.Content] or [genai.Content], a string, or a JSON-serializable map,
// struct, slice, or array.
//
// google.golang.org/genai struct values are accepted only as a single
// top-level Content. Anywhere inside JSON input they fail with
// [ErrInvalidInput], whether as a []*genai.Content history (including the
// result of [genai.Text]), a bare [genai.Part], or a field or element of a map,
// slice, or struct. Their JSON form carries inline data and thought parts that
// implementations do not inspect, and adk-python rejects them because
// json.dumps cannot serialize them. String constants such as [genai.RoleUser]
// are plain JSON and remain accepted. Pass earlier turns as text or plain Go
// values instead.
func ValidateInput(input any) error {
	switch v := input.(type) {
	case nil:
		return fmt.Errorf("%w: input is nil", ErrInvalidInput)
	case *genai.Content:
		if v == nil {
			return fmt.Errorf("%w: input content is nil", ErrInvalidInput)
		}
		return nil
	case genai.Content:
		return nil
	default:
		if err := validateJSONContent(input); err != nil {
			return fmt.Errorf("%w: %w", ErrInvalidInput, err)
		}
		return nil
	}
}

func validateChoice(q Choice) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return fmt.Errorf("%w: instructions: %w", ErrInvalidQuestion, err)
		}
	}
	if len(q.Criteria) < 2 {
		return fmt.Errorf("%w: choice requires at least 2 criteria labels, got %d", ErrInvalidQuestion, len(q.Criteria))
	}
	for label, desc := range q.Criteria {
		if desc == nil {
			continue
		}
		if err := validateJSONContent(desc); err != nil {
			return fmt.Errorf("%w: criteria[%q]: %w", ErrInvalidQuestion, label, err)
		}
	}
	return nil
}

func validateBinary(q Binary) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return fmt.Errorf("%w: instructions: %w", ErrInvalidQuestion, err)
		}
	}
	if q.Criteria != nil {
		if err := validateJSONContent(q.Criteria); err != nil {
			return fmt.Errorf("%w: criteria: %w", ErrInvalidQuestion, err)
		}
	}
	return nil
}

func validateScore(q Score) error {
	if q.Instructions != nil {
		if err := validateJSONContent(q.Instructions); err != nil {
			return fmt.Errorf("%w: instructions: %w", ErrInvalidQuestion, err)
		}
	}
	if len(q.Criteria) < 2 {
		return fmt.Errorf("%w: score requires at least 2 criteria levels, got %d", ErrInvalidQuestion, len(q.Criteria))
	}
	for i, desc := range q.Criteria {
		if desc == nil {
			return fmt.Errorf("%w: criteria[%d] must not be nil", ErrInvalidQuestion, i)
		}
		if err := validateJSONContent(desc); err != nil {
			return fmt.Errorf("%w: criteria[%d]: %w", ErrInvalidQuestion, i, err)
		}
	}
	return nil
}

// validateJSONContent enforces adk-python's JSONContent contract: a JSON
// string, object, or array (rejecting nil, bare booleans, bare numbers, raw
// []byte slices, values that encoding/json cannot marshal, and
// google.golang.org/genai struct values at any depth).
func validateJSONContent(v any) error {
	if _, isBytes := v.([]byte); isBytes {
		return errors.New("[]byte is not valid JSON content; use string or json.RawMessage")
	}
	b, err := json.Marshal(v)
	if err != nil {
		return errors.New("value is not JSON-serializable")
	}
	switch b[0] {
	case '"', '{', '[':
	default:
		return fmt.Errorf("expected string, object, or array JSON content, got %T", v)
	}
	// json.Marshal has succeeded, so the value is acyclic and the walk below
	// terminates.
	return rejectGenaiValues(reflect.ValueOf(v), 0)
}

var (
	genaiPkgPath      = reflect.TypeFor[genai.Content]().PkgPath()
	jsonMarshalerType = reflect.TypeFor[json.Marshaler]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
)

// maxJSONContentDepth bounds rejectGenaiValues as a safeguard; it is far deeper
// than any JSON input a classifier is expected to receive.
const maxJSONContentDepth = 1000

// rejectGenaiValues walks the values encoding/json would serialize by
// reflection and fails on any struct type from google.golang.org/genai, whether
// held by value, by pointer (nil or not), or as the static element type of an
// empty map or slice. In adk-python these are Pydantic models that json.dumps
// rejects, so a []*genai.Content history or a bare genai.Part never reaches a
// provider. The Go SDK's structs marshal cleanly, so without this check their
// JSON form (including inline data and thought parts) would be sent as
// classifier input. Types with their own MarshalJSON or MarshalText are not
// entered, matching encoding/json, unless the method is promoted from an
// embedded genai struct, which it would serialize.
func rejectGenaiValues(v reflect.Value, depth int) error {
	if depth > maxJSONContentDepth {
		return fmt.Errorf("JSON content is nested more than %d levels deep", maxJSONContentDepth)
	}
	if !v.IsValid() {
		return nil
	}
	// Look through interfaces first: encoding/json serializes the dynamic
	// value, and a static interface type such as json.Marshaler says nothing
	// about what that value is.
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil
		}
		return rejectGenaiValues(v.Elem(), depth)
	}
	t := v.Type()
	if err := rejectGenaiType(t); err != nil {
		return err
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil
		}
		return rejectGenaiValues(v.Elem(), depth)
	}
	if usesCustomMarshaler(v) {
		if v.Kind() == reflect.Struct {
			return rejectEmbeddedGenaiTypes(t, nil)
		}
		return nil
	}
	switch v.Kind() {
	case reflect.Slice, reflect.Array:
		if err := rejectGenaiType(t.Elem()); err != nil {
			return err
		}
		if isScalarKind(t.Elem().Kind()) {
			return nil
		}
		for i := range v.Len() {
			if err := rejectGenaiValues(v.Index(i), depth+1); err != nil {
				return err
			}
		}
	case reflect.Map:
		if err := rejectGenaiType(t.Elem()); err != nil {
			return err
		}
		if isScalarKind(t.Elem().Kind()) {
			return nil
		}
		for iter := v.MapRange(); iter.Next(); {
			if err := rejectGenaiValues(iter.Value(), depth+1); err != nil {
				return err
			}
		}
	case reflect.Struct:
		for i := range t.NumField() {
			if !jsonVisibleField(t.Field(i)) {
				continue
			}
			if err := rejectGenaiValues(v.Field(i), depth+1); err != nil {
				return err
			}
		}
	}
	return nil
}

// rejectGenaiType fails when t, after dereferencing any pointers, is a struct
// type declared in google.golang.org/genai.
func rejectGenaiType(t reflect.Type) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() == reflect.Struct && t.PkgPath() == genaiPkgPath {
		return fmt.Errorf(
			"%s is not JSON content; pass google.golang.org/genai values only as a single top-level genai.Content, or render them as text or plain Go values",
			t,
		)
	}
	return nil
}

// rejectEmbeddedGenaiTypes fails when struct type t embeds a genai struct,
// directly or through further embedding. A MarshalJSON or MarshalText promoted
// from such a field serializes the genai value, so the field must be checked
// before the marshaler is trusted. The check is type-only; seen guards against
// recursive embedding such as type T struct{ *T }.
func rejectEmbeddedGenaiTypes(t reflect.Type, seen map[reflect.Type]bool) error {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct || seen[t] {
		return nil
	}
	for i := range t.NumField() {
		sf := t.Field(i)
		if !sf.Anonymous || !jsonVisibleField(sf) {
			continue
		}
		if err := rejectGenaiType(sf.Type); err != nil {
			return err
		}
		if seen == nil {
			seen = map[reflect.Type]bool{}
		}
		seen[t] = true
		if err := rejectEmbeddedGenaiTypes(sf.Type, seen); err != nil {
			return err
		}
	}
	return nil
}

// usesCustomMarshaler reports whether encoding/json would serialize v through
// its own MarshalJSON or MarshalText rather than by reflection.
// Pointer-receiver methods count only when v is addressable, as in
// encoding/json.
func usesCustomMarshaler(v reflect.Value) bool {
	t := v.Type()
	if t.Implements(jsonMarshalerType) || t.Implements(textMarshalerType) {
		return true
	}
	if !v.CanAddr() {
		return false
	}
	pt := reflect.PointerTo(t)
	return pt.Implements(jsonMarshalerType) || pt.Implements(textMarshalerType)
}

// jsonVisibleField mirrors encoding/json's field selection: exported fields and
// embedded structs are serialized; other unexported fields and fields tagged
// `json:"-"` are not.
func jsonVisibleField(sf reflect.StructField) bool {
	if sf.Tag.Get("json") == "-" {
		return false
	}
	if sf.IsExported() {
		return true
	}
	if !sf.Anonymous {
		return false
	}
	ft := sf.Type
	if ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	return ft.Kind() == reflect.Struct
}

// isScalarKind reports whether values of kind k cannot contain a struct, so
// large []byte, []string, or map[string]float64 values are not walked.
func isScalarKind(k reflect.Kind) bool {
	switch k {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128,
		reflect.String:
		return true
	default:
		return false
	}
}

func validateProbability(field string, p float64) error {
	if math.IsNaN(p) || p < 0 || p > 1 {
		return fmt.Errorf("%w: %s must be a finite number in [0, 1]", ErrInvalidAnswer, field)
	}
	return nil
}

func validateConfidence(c *float64) error {
	if c == nil {
		return nil
	}
	return validateProbability("confidence", *c)
}

func validateChoiceAnswer(a ChoiceAnswer) error {
	if len(a.Probabilities) == 0 {
		return fmt.Errorf("%w: choice probabilities must not be empty", ErrInvalidAnswer)
	}
	// Labels may come from a decoded payload, so they stay out of the error.
	for _, p := range a.Probabilities {
		if err := validateProbability("choice probability", p); err != nil {
			return err
		}
	}
	return validateConfidence(a.Confidence)
}

func validateBinaryAnswer(a BinaryAnswer) error {
	if err := validateProbability("probability", a.Probability); err != nil {
		return err
	}
	return validateConfidence(a.Confidence)
}

func validateScoreAnswer(a ScoreAnswer) error {
	if math.IsNaN(a.Score) || math.IsInf(a.Score, 0) || a.Score < 0 {
		return fmt.Errorf("%w: score must be a finite non-negative number", ErrInvalidAnswer)
	}
	if len(a.Probabilities) == 0 {
		return fmt.Errorf("%w: score probabilities must not be empty", ErrInvalidAnswer)
	}
	for level, p := range a.Probabilities {
		if level < 0 {
			return fmt.Errorf("%w: score level %d must be non-negative", ErrInvalidAnswer, level)
		}
		if err := validateProbability(fmt.Sprintf("probabilities[%d]", level), p); err != nil {
			return err
		}
	}
	return validateConfidence(a.Confidence)
}

type choiceAnswerJSON struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

type binaryAnswerJSON struct {
	Type        string   `json:"type"`
	Probability float64  `json:"probability"`
	Confidence  *float64 `json:"confidence,omitempty"`
}

type scoreAnswerJSON struct {
	Type          string          `json:"type"`
	Score         float64         `json:"score"`
	Probabilities map[int]float64 `json:"probabilities"`
	Confidence    *float64        `json:"confidence,omitempty"`
}

func marshalAnswer(a Answer) ([]byte, error) {
	switch v := a.(type) {
	case nil:
		return nil, fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
	case *ChoiceAnswer:
		if v == nil {
			return nil, fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
	case *BinaryAnswer:
		if v == nil {
			return nil, fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
	case *ScoreAnswer:
		if v == nil {
			return nil, fmt.Errorf("%w: answer is nil", ErrInvalidAnswer)
		}
	case ChoiceAnswer, BinaryAnswer, ScoreAnswer:
	default:
		return nil, fmt.Errorf("%w: unsupported answer type %T", ErrInvalidAnswer, a)
	}
	b, err := json.Marshal(a)
	if err != nil {
		var marshalerErr *json.MarshalerError
		if errors.As(err, &marshalerErr) && marshalerErr.Err != nil {
			return nil, marshalerErr.Err
		}
		return nil, err
	}
	return b, nil
}

func unmarshalAnswer(data []byte) (Answer, error) {
	var disc struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &disc); err != nil {
		return nil, fmt.Errorf("%w: invalid answer JSON", ErrInvalidAnswer)
	}
	switch disc.Type {
	case "choice":
		var ans ChoiceAnswer
		if err := ans.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return ans, nil
	case "binary":
		var ans BinaryAnswer
		if err := ans.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return ans, nil
	case "score":
		var ans ScoreAnswer
		if err := ans.UnmarshalJSON(data); err != nil {
			return nil, err
		}
		return ans, nil
	default:
		return nil, fmt.Errorf("%w: unsupported answer type", ErrInvalidAnswer)
	}
}

func checkDiscriminator(raw json.RawMessage, want string, sentinel error) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil
	}
	var got string
	if trimmed[0] != '"' || json.Unmarshal(trimmed, &got) != nil || got != want {
		return fmt.Errorf("%w: expected %s type", sentinel, want)
	}
	return nil
}

func decodeStrict(data []byte, dst any) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return errors.New("expected JSON object")
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return errors.New("malformed JSON payload")
	}
	if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("trailing JSON data")
	}
	return nil
}
