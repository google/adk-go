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
	"errors"
	"fmt"
	"iter"
	"maps"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/workflowroute"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// DefaultThreshold is the probability cutoff used by [Node] for [Binary]
// questions when [NodeConfig.Threshold] is nil.
const DefaultThreshold = 0.5

const routeQuestionName = "route"

func init() {
	workflowroute.Register(func(node any, out workflowroute.OutgoingSummary) error {
		if n, ok := node.(*Node); ok {
			return n.validateOutgoingEdges(out)
		}
		return nil
	})
}

var _ workflow.Node = (*Node)(nil)

// NodeConfig configures a workflow routing [Node] constructed with [NewNode].
type NodeConfig struct {
	// Name identifies the node within the workflow graph. Required.
	Name string
	// Description is an optional human-readable description of the node.
	Description string
	// NodeConfig holds workflow-level node settings such as Timeout and RetryConfig.
	NodeConfig workflow.NodeConfig
	// Classifier answers Question on each node activation. Required.
	Classifier Classifier
	// Question is the [Choice] or [Binary] question whose answer selects the
	// route. [Score] questions are rejected because scores are continuous rather
	// than discrete route labels.
	Question Question
	// Threshold is the probability in [0, 1] at or above which a [Binary]
	// question routes to workflow.BoolRoute(true). Nil defaults to
	// [DefaultThreshold] (0.5). Setting Threshold for a [Choice] question fails
	// with [ErrInvalidQuestion].
	Threshold *float64
	// AnswerKey, when non-empty, stores the serialized answer map under this key
	// in session state via EventActions.StateDelta.
	AnswerKey string
	// InputSchema and OutputSchema optionally validate and coerce the node's
	// input and output, matching [workflow.BaseNode].
	InputSchema  *jsonschema.Schema
	OutputSchema *jsonschema.Schema
}

// Node is a [workflow.Node] that routes a workflow by asking a [Classifier] one
// [Choice] or [Binary] question about its input.
//
// For a [Choice] question, the node emits the chosen label (matched by
// [workflow.StringRoute] or [workflow.MultiRoute]). For a [Binary] question, it
// emits "true" when the probability is at or above Threshold and "false"
// otherwise (matched by [workflow.BoolRoute] or [workflow.MultiRoute]). The
// node forwards the input it received as Event.Output so downstream nodes
// receive what the classifier evaluated.
//
// When the node has routed outgoing edges, [workflow.New] validates that every
// edge route can be emitted by Question and that every possible answer has an
// outgoing edge unless [workflow.Default] or an unconditional edge covers the
// remaining answers. Routes are matched by kind as in adk-python: a [Binary]
// question needs bool routes and a [Choice] question string routes, so a
// [workflow.StringRoute] of "true" on a Binary node is rejected even though
// the engine compares emitted routes as strings at run time.
//
// The zero value is not usable; construct one with [NewNode]. A Node
// constructed with [NewNode] is safe for concurrent use across workflow runs.
type Node struct {
	workflow.BaseNode
	classifier Classifier
	question   Question // normalized non-pointer Choice (with cloned Criteria) or Binary
	threshold  float64
	answerKey  string
}

// NewNode validates cfg and returns a workflow routing [Node], mirroring
// adk-python's ClassifierNode.
//
// Unlike adk-python, which raises ValidationError, GraphValidationError, or
// WorkflowDataError, Go returns [ErrInvalidNodeConfig] (wrapping
// [ErrInvalidQuestion] when the question is invalid), [ErrInvalidRoute] during
// [workflow.New], and [ErrInvalidInput] or [ErrInvalidAnswer] at runtime.
func NewNode(cfg NodeConfig) (*Node, error) {
	if strings.TrimSpace(cfg.Name) == "" {
		return nil, fmt.Errorf("%w: Name must not be empty", ErrInvalidNodeConfig)
	}
	if cfg.Classifier == nil {
		return nil, fmt.Errorf("%w: Classifier must not be nil", ErrInvalidNodeConfig)
	}
	// Reject Score questions before per-field criteria checks so Score{} reports
	// an unsupported routing question type rather than a level-count error.
	if isScoreQuestion(cfg.Question) {
		return nil, fmt.Errorf(
			"%w: %w: Score questions cannot select a workflow route; use Choice or Binary",
			ErrInvalidNodeConfig, ErrInvalidQuestion,
		)
	}
	if err := ValidateQuestion(cfg.Question); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidNodeConfig, err)
	}

	var normalized Question
	threshold := DefaultThreshold
	switch q := derefQuestion(cfg.Question).(type) {
	case Choice:
		if cfg.Threshold != nil {
			return nil, fmt.Errorf(
				"%w: %w: Threshold only applies to Binary questions; a Choice always routes to the chosen label",
				ErrInvalidNodeConfig, ErrInvalidQuestion,
			)
		}
		normalized = Choice{
			Instructions: q.Instructions,
			Criteria:     maps.Clone(q.Criteria),
		}
	case Binary:
		if cfg.Threshold != nil {
			t := *cfg.Threshold
			if math.IsNaN(t) || t < 0 || t > 1 {
				return nil, fmt.Errorf("%w: Threshold must be a finite number in [0, 1]", ErrInvalidNodeConfig)
			}
			threshold = t
		}
		normalized = q
	}

	if cfg.AnswerKey != "" && strings.TrimSpace(cfg.AnswerKey) == "" {
		return nil, fmt.Errorf("%w: AnswerKey must not be whitespace-only", ErrInvalidNodeConfig)
	}

	var inputResolved *jsonschema.Resolved
	if cfg.InputSchema != nil {
		r, err := cfg.InputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("%w: resolving InputSchema: %w", ErrInvalidNodeConfig, err)
		}
		inputResolved = r
	}
	var outputResolved *jsonschema.Resolved
	if cfg.OutputSchema != nil {
		r, err := cfg.OutputSchema.Resolve(nil)
		if err != nil {
			return nil, fmt.Errorf("%w: resolving OutputSchema: %w", ErrInvalidNodeConfig, err)
		}
		outputResolved = r
	}

	return &Node{
		BaseNode:   workflow.NewBaseNodeWithSchemas(cfg.Name, cfg.Description, cfg.NodeConfig, inputResolved, outputResolved),
		classifier: cfg.Classifier,
		question:   normalized,
		threshold:  threshold,
		answerKey:  cfg.AnswerKey,
	}, nil
}

// ValidateInput checks that input is a valid non-nil classifier input (see
// [ValidateInput]) and, when InputSchema is set, validates and coerces it
// against the schema. Under InputSchema, a [genai.Content] or [*genai.Content]
// input is validated using its non-thought text, parsed as JSON when the
// schema expects a non-string shape and the text parses, as adk-python's
// validate_node_data does for typed schemas; the classifier then receives that
// text or JSON rather than the Content. A Content with a nil or non-text part
// fails with [ErrInvalidInput] under InputSchema, whereas adk-python drops
// such parts. Without InputSchema, the Content is passed through unchanged.
//
// Like [workflow.BaseNode], ValidateInput returns the underlying error; the
// workflow engine wraps it in [workflow.ErrInputValidation] so
// [workflow.DefaultRetryConfig] does not retry it.
func (n *Node) ValidateInput(input any) (any, error) {
	if n == nil || n.classifier == nil {
		return nil, fmt.Errorf("%w: Node must be constructed with NewNode", ErrInvalidNodeConfig)
	}
	if input == nil {
		return nil, fmt.Errorf("%w: no input to classify", ErrInvalidInput)
	}
	// Check the raw input before schema coercion: coercion JSON-round-trips
	// values into plain maps and slices, which would carry a []*genai.Content
	// history or the thought parts of a Content past ValidateInput.
	if err := ValidateInput(input); err != nil {
		return nil, err
	}
	if n.InputSchema() == nil {
		return input, nil
	}
	toValidate := input
	if content := contentInput(input); content != nil {
		payload, err := contentSchemaPayload(content)
		if err != nil {
			return nil, err
		}
		toValidate = payload
	}
	validated, err := n.BaseNode.ValidateInput(toValidate)
	if err != nil {
		return nil, err
	}
	// Coercion can produce nil, a bare number, or a boolean, none of which is
	// classifier input.
	if err := ValidateInput(validated); err != nil {
		return nil, err
	}
	return validated, nil
}

// contentInput returns input as a *genai.Content when it is one, by value or
// pointer, and nil otherwise.
func contentInput(input any) *genai.Content {
	switch v := input.(type) {
	case *genai.Content:
		return v
	case genai.Content:
		return &v
	default:
		return nil
	}
}

// contentSchemaPayload returns the text a Content contributes to InputSchema
// validation: its non-thought text parts joined, as adk-python's
// extract_text_from_content does. Unlike adk-python, which drops non-text
// parts silently, contentSchemaPayload fails with [ErrInvalidInput] if any
// part (thought or not) carries a non-text field, so an InputSchema never
// discards multimodal input unnoticed. The field check matches isNonTextPart
// in exp/classifier/typesafe.
func contentSchemaPayload(content *genai.Content) (string, error) {
	var b strings.Builder
	for _, part := range content.Parts {
		if part == nil {
			return "", fmt.Errorf("%w: InputSchema applies to text and JSON input; the Content has a nil part", ErrInvalidInput)
		}
		rest := *part
		rest.Text = ""
		rest.Thought = false
		rest.PartMetadata = nil
		rest.ThoughtSignature = nil
		if !reflect.ValueOf(rest).IsZero() {
			return "", fmt.Errorf("%w: InputSchema applies to text and JSON input; the Content has a non-text part", ErrInvalidInput)
		}
		if part.Thought {
			continue
		}
		b.WriteString(part.Text)
	}
	return b.String(), nil
}

// Run classifies input against the node's question and yields a single routing
// event whose Output is input. Run expects input that the workflow scheduler
// has already passed through [Node.ValidateInput] and does not apply
// InputSchema itself, since coercing a string through the schema a second time
// can change it; a caller that invokes Run directly therefore gets no
// InputSchema validation. Run still checks that input is a non-nil classifier
// input (see [ValidateInput]) and wraps a failure in
// [workflow.ErrInputValidation].
//
// A classifier error that satisfies [ErrInvalidInput] or [ErrInvalidQuestion]
// is wrapped in [workflow.ErrInputValidation] as well, so
// [workflow.DefaultRetryConfig] does not retry a failure that repeats on every
// attempt; adk-python retries these.
func (n *Node) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if n == nil || n.classifier == nil {
			yield(nil, fmt.Errorf("%w: Node must be constructed with NewNode", ErrInvalidNodeConfig))
			return
		}
		if ctx == nil {
			yield(nil, fmt.Errorf("classifier.Node.Run: nil context for node %q", n.Name()))
			return
		}
		if err := ValidateInput(input); err != nil {
			yield(nil, fmt.Errorf("%w for node %q: %w", workflow.ErrInputValidation, n.Name(), err))
			return
		}

		// Unlike adk-python's _run_impl (which converts BaseModel inputs to
		// dicts), Go's Classifier.Classify accepts JSON-serializable structs
		// directly.
		resp, err := n.classifier.Classify(ctx, input, map[string]Question{
			routeQuestionName: cloneNodeQuestion(n.question),
		})
		if err != nil {
			// The classifier rejecting the input or the node's question fails
			// the same way on every attempt, and ErrInputValidation is the only
			// error workflow.DefaultRetryConfig does not retry.
			if errors.Is(err, ErrInvalidInput) || errors.Is(err, ErrInvalidQuestion) {
				yield(nil, fmt.Errorf("%w for node %q: %w", workflow.ErrInputValidation, n.Name(), err))
				return
			}
			yield(nil, fmt.Errorf("node %q: %w", n.Name(), err))
			return
		}
		if resp == nil {
			yield(nil, fmt.Errorf("%w: node %q: classifier returned nil response", ErrInvalidAnswer, n.Name()))
			return
		}
		ans := resp.Answers[routeQuestionName]
		if ans == nil {
			yield(nil, fmt.Errorf("%w: node %q: classifier returned no answer", ErrInvalidAnswer, n.Name()))
			return
		}
		// Unlike adk-python, validate probability and confidence bounds at
		// runtime so AnswerKey state maps stay valid JSON across all custom
		// Classifier implementations.
		if err := ValidateAnswer(ans); err != nil {
			yield(nil, fmt.Errorf("%w: node %q", err, n.Name()))
			return
		}

		route, err := n.routeForAnswer(ans)
		if err != nil {
			yield(nil, err)
			return
		}

		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = input
		ev.Routes = []string{route}
		if n.answerKey != "" {
			ev.Actions.StateDelta = map[string]any{n.answerKey: answerToStateMap(ans)}
		}
		yield(ev, nil)
	}
}

func (n *Node) routeForAnswer(ans Answer) (string, error) {
	deref := derefNodeAnswer(ans)
	switch q := n.question.(type) {
	case Choice:
		ca, ok := deref.(ChoiceAnswer)
		if !ok {
			return "", fmt.Errorf(
				"%w: node %q: classifier returned a %s answer to a choice question",
				ErrInvalidAnswer, n.Name(), answerTypeName(deref),
			)
		}
		if _, valid := q.Criteria[ca.Choice]; !valid {
			// Unlike adk-python's _route_for, do not echo ca.Choice in the
			// error message because it is model output.
			return "", fmt.Errorf(
				"%w: node %q: classifier chose a label outside declared criteria",
				ErrInvalidAnswer, n.Name(),
			)
		}
		return ca.Choice, nil

	default:
		ba, ok := deref.(BinaryAnswer)
		if !ok {
			return "", fmt.Errorf(
				"%w: node %q: classifier returned a %s answer to a binary question",
				ErrInvalidAnswer, n.Name(), answerTypeName(deref),
			)
		}
		return strconv.FormatBool(ba.Probability >= n.threshold), nil
	}
}

func (n *Node) validRoutes() (workflowroute.Kind, []string) {
	if q, ok := n.question.(Choice); ok {
		keys := slices.Sorted(maps.Keys(q.Criteria))
		labels := make([]string, len(keys))
		for i, k := range keys {
			labels[i] = strconv.Quote(k)
		}
		return workflowroute.KindString, labels
	}
	return workflowroute.KindBool, []string{"false", "true"}
}

func (n *Node) validateOutgoingEdges(out workflowroute.OutgoingSummary) error {
	if n.classifier == nil {
		return fmt.Errorf("%w: Node must be constructed with NewNode", ErrInvalidNodeConfig)
	}
	if len(out.Routes) == 0 {
		return nil
	}

	wantKind, validLabels := n.validRoutes()
	seen := make(map[string]bool, len(validLabels))
	for _, label := range validLabels {
		seen[label] = false
	}
	unknownSet := make(map[string]struct{})
	for _, r := range out.Routes {
		if r.Kind != wantKind {
			unknownSet[r.Label] = struct{}{}
			continue
		}
		if _, ok := seen[r.Label]; !ok {
			unknownSet[r.Label] = struct{}{}
			continue
		}
		seen[r.Label] = true
	}
	if len(unknownSet) > 0 {
		unknown := slices.Sorted(maps.Keys(unknownSet))
		return fmt.Errorf(
			"%w: graph validation failed: classifier node %q has edges for routes it never emits: [%s] (valid routes: [%s])",
			ErrInvalidRoute, n.Name(), strings.Join(unknown, ", "), strings.Join(validLabels, ", "),
		)
	}
	if !out.HasDefault && !out.HasUnrouted {
		var missing []string
		for _, label := range validLabels {
			if !seen[label] {
				missing = append(missing, label)
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf(
				"%w: graph validation failed: classifier node %q has no edge for routes [%s]; add an edge for each, workflow.Default, or an unconditional edge",
				ErrInvalidRoute, n.Name(), strings.Join(missing, ", "),
			)
		}
	}
	return nil
}

func isScoreQuestion(q Question) bool {
	switch v := q.(type) {
	case Score:
		return true
	case *Score:
		return v != nil
	default:
		return false
	}
}

// derefQuestion returns the value form of a pointer question so type switches
// need only handle Choice, Binary, and Score.
func derefQuestion(q Question) Question {
	switch v := q.(type) {
	case *Choice:
		return *v
	case *Binary:
		return *v
	case *Score:
		return *v
	default:
		return q
	}
}

func cloneNodeQuestion(q Question) Question {
	if cq, ok := q.(Choice); ok {
		return Choice{
			Instructions: cq.Instructions,
			Criteria:     maps.Clone(cq.Criteria),
		}
	}
	return q
}

func derefNodeAnswer(a Answer) Answer {
	switch v := a.(type) {
	case *ChoiceAnswer:
		return *v
	case *BinaryAnswer:
		return *v
	case *ScoreAnswer:
		return *v
	default:
		return a
	}
}

func answerTypeName(a Answer) string {
	switch a.(type) {
	case ChoiceAnswer:
		return "choice"
	case BinaryAnswer:
		return "binary"
	default:
		return "score"
	}
}

// answerToStateMap converts a ChoiceAnswer or BinaryAnswer to a fresh
// JSON-compatible map[string]any matching adk-python's answer.model_dump() and
// the map[string]any layout produced by encoding/json in persistent session
// services.
func answerToStateMap(ans Answer) map[string]any {
	switch a := derefNodeAnswer(ans).(type) {
	case ChoiceAnswer:
		probs := make(map[string]any, len(a.Probabilities))
		for k, p := range a.Probabilities {
			probs[k] = p
		}
		var conf any
		if a.Confidence != nil {
			conf = *a.Confidence
		}
		return map[string]any{
			"type":          "choice",
			"choice":        a.Choice,
			"probabilities": probs,
			"confidence":    conf,
		}
	default:
		ba := a.(BinaryAnswer)
		var conf any
		if ba.Confidence != nil {
			conf = *ba.Confidence
		}
		return map[string]any{
			"type":        "binary",
			"probability": ba.Probability,
			"confidence":  conf,
		}
	}
}
