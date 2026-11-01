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
	"encoding/base64"
	"errors"
	"fmt"
	"maps"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/exp/classifier"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

var triageQuestion = classifier.Choice{
	Instructions: "Route this ticket.",
	Criteria: map[string]any{
		"billing":   "Invoices or refunds",
		"technical": "Bugs",
	},
}

type recordedCall struct {
	Input     any
	Questions map[string]classifier.Question
}

type fixedClassifier struct {
	mu      sync.Mutex
	model   string
	answer  classifier.Answer
	respNil bool
	err     error
	mutateQ bool
	calls   []recordedCall
}

func (f *fixedClassifier) Name() string {
	if f.model == "" {
		return "fixed"
	}
	return f.model
}

func (f *fixedClassifier) Classify(
	_ context.Context,
	input any,
	questions map[string]classifier.Question,
) (*classifier.Response, error) {
	f.mu.Lock()
	snapshot := make(map[string]classifier.Question, len(questions))
	for k, q := range questions {
		if cq, ok := q.(classifier.Choice); ok {
			q = classifier.Choice{Instructions: cq.Instructions, Criteria: maps.Clone(cq.Criteria)}
		}
		snapshot[k] = q
	}
	f.calls = append(f.calls, recordedCall{Input: input, Questions: snapshot})
	f.mu.Unlock()

	if f.mutateQ {
		if cq, ok := questions["route"].(classifier.Choice); ok {
			cq.Criteria["mutated"] = "evil"
		}
	}
	if f.err != nil {
		return nil, f.err
	}
	if f.respNil {
		return nil, nil
	}
	answers := make(map[string]classifier.Answer, len(questions))
	if f.answer != nil {
		for name := range questions {
			answers[name] = f.answer
		}
	}
	return &classifier.Response{
		ModelVersion: f.Name(),
		Answers:      answers,
	}, nil
}

func choiceAnswer(label string) classifier.ChoiceAnswer {
	probs := map[string]float64{"billing": 0.1, "technical": 0.9}
	if label == "billing" {
		probs = map[string]float64{"billing": 0.9, "technical": 0.1}
	}
	return classifier.ChoiceAnswer{
		Choice:        label,
		Probabilities: probs,
	}
}

type capturingSink struct {
	mu     sync.Mutex
	inputs []any
	states []any
}

func newCapturingNode(name, stateKey string) (*workflow.FunctionNode, *capturingSink) {
	sink := &capturingSink{}
	node := workflow.NewFunctionNode(name, func(ctx agent.Context, in any) (any, error) {
		var st any
		if stateKey != "" {
			val, err := ctx.State().Get(stateKey)
			if err != nil {
				return nil, err
			}
			st = val
		}
		sink.mu.Lock()
		sink.inputs = append(sink.inputs, in)
		sink.states = append(sink.states, st)
		sink.mu.Unlock()
		return in, nil
	}, workflow.NodeConfig{})
	return node, sink
}

func runWorkflowViaRunner(
	t *testing.T,
	edges []workflow.Edge,
	msg *genai.Content,
) (session.Service, session.Session, error) {
	t.Helper()
	wfAgent, err := workflowagent.New(workflowagent.Config{
		Name:  "wf",
		Edges: edges,
	})
	if err != nil {
		return nil, nil, err
	}
	ss := session.InMemoryService()
	createResp, err := ss.Create(t.Context(), &session.CreateRequest{
		AppName: "test_app",
		UserID:  "user_1",
	})
	if err != nil {
		t.Fatalf("ss.Create() unexpected error: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          wfAgent,
		SessionService: ss,
	})
	if err != nil {
		t.Fatalf("runner.New() unexpected error: %v", err)
	}
	for _, runErr := range r.Run(t.Context(), "user_1", createResp.Session.ID(), msg, agent.RunConfig{}) {
		if runErr != nil {
			return ss, createResp.Session, runErr
		}
	}
	stored, err := ss.Get(t.Context(), &session.GetRequest{
		AppName:   "test_app",
		UserID:    "user_1",
		SessionID: createResp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("ss.Get() unexpected error: %v", err)
	}
	return ss, stored.Session, nil
}

func TestChoiceRoutesToChosenLabelAndPassesInputThrough(t *testing.T) {
	fc := &fixedClassifier{answer: choiceAnswer("technical"), mutateQ: true}
	node, err := classifier.NewNode(classifier.NodeConfig{
		Name:       "triage",
		Classifier: fc,
		Question:   triageQuestion,
	})
	if err != nil {
		t.Fatalf("NewNode() unexpected error: %v", err)
	}
	billingNode, billingSink := newCapturingNode("billing", "")
	techNode, techSink := newCapturingNode("technical", "")

	edges := []workflow.Edge{
		{From: workflow.Start, To: node},
		{From: node, To: billingNode, Route: workflow.StringRoute("billing")},
		{From: node, To: techNode, Route: workflow.StringRoute("technical")},
	}

	// Run twice to verify in-place mutation of questions["route"].Criteria by
	// the classifier on the first call does not leak into the second call.
	for range 2 {
		_, _, err = runWorkflowViaRunner(
			t,
			edges,
			genai.NewContentFromText("The API returns 500", genai.RoleUser),
		)
		if err != nil {
			t.Fatalf("runWorkflowViaRunner() unexpected error: %v", err)
		}
	}

	if len(billingSink.inputs) != 0 {
		t.Errorf("billing received %d inputs, want 0", len(billingSink.inputs))
	}
	if diff := cmp.Diff([]any{"The API returns 500", "The API returns 500"}, techSink.inputs); diff != "" {
		t.Errorf("technical inputs mismatch (-want +got):\n%s", diff)
	}
	if len(fc.calls) != 2 {
		t.Fatalf("classifier call count = %d, want 2", len(fc.calls))
	}
	for i, call := range fc.calls {
		if diff := cmp.Diff(map[string]classifier.Question{"route": triageQuestion}, call.Questions); diff != "" {
			t.Errorf("classifier call[%d] questions mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestBinaryRoutesOnThreshold(t *testing.T) {
	customThreshold := 0.6
	cases := []struct {
		name      string
		threshold *float64
		prob      float64
		wantRoute bool
	}{
		{name: "custom_above", threshold: &customThreshold, prob: 0.7, wantRoute: true},
		{name: "custom_equal", threshold: &customThreshold, prob: 0.6, wantRoute: true},
		{name: "custom_below", threshold: &customThreshold, prob: 0.59, wantRoute: false},
		{name: "default_equal_0.5", threshold: nil, prob: 0.5, wantRoute: true},
		{name: "default_below_0.49", threshold: nil, prob: 0.49, wantRoute: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fc := &fixedClassifier{answer: classifier.BinaryAnswer{Probability: tc.prob}}
			node, err := classifier.NewNode(classifier.NodeConfig{
				Name:       "gate",
				Classifier: fc,
				Question:   classifier.Binary{Instructions: "Is this risky?"},
				Threshold:  tc.threshold,
			})
			if err != nil {
				t.Fatalf("NewNode() unexpected error: %v", err)
			}
			yesNode, yesSink := newCapturingNode("yes", "")
			noNode, noSink := newCapturingNode("no", "")

			edges := []workflow.Edge{
				{From: workflow.Start, To: node},
				{From: node, To: yesNode, Route: workflow.BoolRoute(true)},
				{From: node, To: noNode, Route: workflow.BoolRoute(false)},
			}
			if _, _, err := runWorkflowViaRunner(
				t,
				edges,
				genai.NewContentFromText("check", genai.RoleUser),
			); err != nil {
				t.Fatalf("runWorkflowViaRunner() unexpected error: %v", err)
			}

			if tc.wantRoute {
				if len(yesSink.inputs) != 1 || len(noSink.inputs) != 0 {
					t.Errorf("got yes=%d no=%d, want yes=1 no=0", len(yesSink.inputs), len(noSink.inputs))
				}
			} else {
				if len(yesSink.inputs) != 0 || len(noSink.inputs) != 1 {
					t.Errorf("got yes=%d no=%d, want yes=0 no=1", len(yesSink.inputs), len(noSink.inputs))
				}
			}
		})
	}
}

func TestAnswerKeyStoresAnswerInStateAndLeavesStateUntouchedWhenUnset(t *testing.T) {
	t.Run("stores_choice_and_binary_answers_in_session_state", func(t *testing.T) {
		conf := 0.85
		triageClf := &fixedClassifier{
			answer: &classifier.ChoiceAnswer{
				Choice:        "technical",
				Probabilities: map[string]float64{"billing": 0.1, "technical": 0.9},
				Confidence:    &conf,
			},
		}
		triage, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: triageClf,
			Question:   &triageQuestion,
			AnswerKey:  "triage_answer",
		})
		if err != nil {
			t.Fatalf("NewNode(triage) unexpected error: %v", err)
		}

		urgentClf := &fixedClassifier{
			answer: &classifier.BinaryAnswer{Probability: 0.75},
		}
		urgentQ := classifier.Binary{Instructions: "Is it urgent?"}
		isUrgent, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "is_urgent",
			Classifier: urgentClf,
			Question:   &urgentQ,
			AnswerKey:  "urgency_answer",
		})
		if err != nil {
			t.Fatalf("NewNode(isUrgent) unexpected error: %v", err)
		}

		billingNode, _ := newCapturingNode("billing", "")
		pageNode, pageSink := newCapturingNode("page_oncall", "urgency_answer")
		queueNode, _ := newCapturingNode("eng_queue", "")

		edges := []workflow.Edge{
			{From: workflow.Start, To: triage},
			{From: triage, To: billingNode, Route: workflow.StringRoute("billing")},
			{From: triage, To: isUrgent, Route: workflow.StringRoute("technical")},
			{From: isUrgent, To: pageNode, Route: workflow.BoolRoute(true)},
			{From: isUrgent, To: queueNode, Route: workflow.BoolRoute(false)},
		}

		_, stored, err := runWorkflowViaRunner(
			t,
			edges,
			genai.NewContentFromText("Prod is down", genai.RoleUser),
		)
		if err != nil {
			t.Fatalf("runWorkflowViaRunner() unexpected error: %v", err)
		}

		wantTriageState := map[string]any{
			"type":          "choice",
			"choice":        "technical",
			"probabilities": map[string]any{"billing": 0.1, "technical": 0.9},
			"confidence":    0.85,
		}
		gotTriageState, err := stored.State().Get("triage_answer")
		if err != nil {
			t.Fatalf("State().Get(triage_answer) error: %v", err)
		}
		if diff := cmp.Diff(wantTriageState, gotTriageState); diff != "" {
			t.Errorf("triage_answer mismatch (-want +got):\n%s", diff)
		}

		wantUrgentState := map[string]any{
			"type":        "binary",
			"probability": 0.75,
			"confidence":  nil,
		}
		gotUrgentState, err := stored.State().Get("urgency_answer")
		if err != nil {
			t.Fatalf("State().Get(urgency_answer) error: %v", err)
		}
		if diff := cmp.Diff(wantUrgentState, gotUrgentState); diff != "" {
			t.Errorf("urgency_answer mismatch (-want +got):\n%s", diff)
		}
		// Downstream node observed urgency_answer synchronously via ctx.State().Get.
		if diff := cmp.Diff([]any{wantUrgentState}, pageSink.states); diff != "" {
			t.Errorf("downstream observed state mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("no_answer_key_leaves_state_empty", func(t *testing.T) {
		node, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: &fixedClassifier{answer: choiceAnswer("billing")},
			Question:   triageQuestion,
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		billingNode, _ := newCapturingNode("billing", "")
		otherNode, _ := newCapturingNode("other", "")

		_, stored, err := runWorkflowViaRunner(
			t,
			[]workflow.Edge{
				{From: workflow.Start, To: node},
				{From: node, To: billingNode, Route: workflow.StringRoute("billing")},
				{From: node, To: otherNode, Route: workflow.Default},
			},
			genai.NewContentFromText("Invoice question", genai.RoleUser),
		)
		if err != nil {
			t.Fatalf("runWorkflowViaRunner() unexpected error: %v", err)
		}
		for k := range stored.State().All() {
			t.Errorf("unexpected key %q written to session state", k)
		}
	})
}

func TestStructInputPassedToClassifierAndForwardedUnchanged(t *testing.T) {
	type ticket struct {
		Subject string    `json:"subject"`
		Opened  time.Time `json:"opened"`
	}
	wantTicket := ticket{
		Subject: "double charge",
		Opened:  time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC),
	}

	source := workflow.NewFunctionNode("src", func(_ agent.Context, _ any) (ticket, error) {
		return wantTicket, nil
	}, workflow.NodeConfig{})

	fc := &fixedClassifier{answer: choiceAnswer("billing")}
	node, err := classifier.NewNode(classifier.NodeConfig{
		Name:       "triage",
		Classifier: fc,
		Question:   triageQuestion,
	})
	if err != nil {
		t.Fatalf("NewNode() unexpected error: %v", err)
	}
	sinkNode, sink := newCapturingNode("sink", "")

	edges := []workflow.Edge{
		{From: workflow.Start, To: source},
		{From: source, To: node},
		{From: node, To: sinkNode},
	}
	if _, _, err := runWorkflowViaRunner(t, edges, genai.NewContentFromText("go", genai.RoleUser)); err != nil {
		t.Fatalf("runWorkflowViaRunner() unexpected error: %v", err)
	}

	if diff := cmp.Diff(wantTicket, fc.calls[0].Input); diff != "" {
		t.Errorf("classifier input mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]any{wantTicket}, sink.inputs); diff != "" {
		t.Errorf("forwarded output mismatch (-want +got):\n%s", diff)
	}
}

func TestRuntimeErrorsFailWorkflowRun(t *testing.T) {
	t.Run("missing_input_from_predecessor", func(t *testing.T) {
		noOutputSource := workflow.NewFunctionNode("src", func(ctx agent.Context, _ any) (*session.Event, error) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Routes = []string{"go"}
			return ev, nil
		}, workflow.NodeConfig{})

		node, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: &fixedClassifier{answer: choiceAnswer("billing")},
			Question:   triageQuestion,
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		sinkNode, _ := newCapturingNode("sink", "")

		_, _, err = runWorkflowViaRunner(
			t,
			[]workflow.Edge{
				{From: workflow.Start, To: noOutputSource},
				{From: noOutputSource, To: node, Route: workflow.StringRoute("go")},
				{From: node, To: sinkNode},
			},
			genai.NewContentFromText("hi", genai.RoleUser),
		)
		if !errors.Is(err, classifier.ErrInvalidInput) {
			t.Fatalf("run error = %v, want errors.Is(ErrInvalidInput)", err)
		}
	})

	t.Run("unusable_classifier_responses", func(t *testing.T) {
		boom := errors.New("upstream failed")
		cases := []struct {
			name       string
			clf        *fixedClassifier
			q          classifier.Question
			wantErr    error
			wantSubstr string
		}{
			{
				name:    "classifier_error",
				clf:     &fixedClassifier{err: boom},
				q:       triageQuestion,
				wantErr: boom,
			},
			{
				name:       "nil_response",
				clf:        &fixedClassifier{respNil: true},
				q:          triageQuestion,
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "nil response",
			},
			{
				name:       "missing_route_answer",
				clf:        &fixedClassifier{answer: nil},
				q:          triageQuestion,
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "returned no answer",
			},
			{
				name: "undeclared_choice_label_is_not_echoed",
				clf: &fixedClassifier{answer: classifier.ChoiceAnswer{
					Choice:        "secret-hallucinated-label",
					Probabilities: map[string]float64{"billing": 0.5, "technical": 0.5},
				}},
				q:          triageQuestion,
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "outside declared criteria",
			},
			{
				name:       "binary_answer_to_choice_question",
				clf:        &fixedClassifier{answer: classifier.BinaryAnswer{Probability: 0.5}},
				q:          triageQuestion,
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "binary answer to a choice question",
			},
			{
				name: "score_answer_to_choice_question",
				clf: &fixedClassifier{answer: classifier.ScoreAnswer{
					Score:         1.0,
					Probabilities: map[int]float64{0: 0.0, 1: 1.0},
				}},
				q:          triageQuestion,
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "score answer to a choice question",
			},
			{
				name:       "choice_answer_to_binary_question",
				clf:        &fixedClassifier{answer: choiceAnswer("billing")},
				q:          classifier.Binary{Instructions: "Is it urgent?"},
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "choice answer to a binary question",
			},
			{
				name: "score_answer_to_binary_question",
				clf: &fixedClassifier{answer: &classifier.ScoreAnswer{
					Score:         1.0,
					Probabilities: map[int]float64{0: 0.0, 1: 1.0},
				}},
				q:          classifier.Binary{Instructions: "Is it urgent?"},
				wantErr:    classifier.ErrInvalidAnswer,
				wantSubstr: "score answer to a binary question",
			},
			{
				name:    "out_of_range_probability_in_answer",
				clf:     &fixedClassifier{answer: classifier.BinaryAnswer{Probability: 1.5}},
				q:       classifier.Binary{Instructions: "Is it urgent?"},
				wantErr: classifier.ErrInvalidAnswer,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				node, err := classifier.NewNode(classifier.NodeConfig{
					Name:       "triage",
					Classifier: tc.clf,
					Question:   tc.q,
				})
				if err != nil {
					t.Fatalf("NewNode() unexpected error: %v", err)
				}
				sinkNode, _ := newCapturingNode("sink", "")
				_, _, err = runWorkflowViaRunner(
					t,
					[]workflow.Edge{
						{From: workflow.Start, To: node},
						{From: node, To: sinkNode},
					},
					genai.NewContentFromText("ticket", genai.RoleUser),
				)
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("run error = %v, want errors.Is(%v)", err, tc.wantErr)
				}
				if tc.wantSubstr != "" && !strings.Contains(err.Error(), tc.wantSubstr) {
					t.Errorf("run error = %q, want substring %q", err.Error(), tc.wantSubstr)
				}
				if strings.Contains(err.Error(), "secret-hallucinated-label") {
					t.Errorf("error echoed raw undeclared label: %v", err)
				}
			})
		}
	})

	t.Run("zero_and_nil_node_and_nil_context", func(t *testing.T) {
		var zero classifier.Node
		for _, err := range zero.Run(nil, "x") {
			if !errors.Is(err, classifier.ErrInvalidNodeConfig) {
				t.Fatalf("zero Node.Run() error = %v, want ErrInvalidNodeConfig", err)
			}
		}
		var nilNode *classifier.Node
		for _, err := range nilNode.Run(nil, "x") {
			if !errors.Is(err, classifier.ErrInvalidNodeConfig) {
				t.Fatalf("nil (*Node).Run() error = %v, want ErrInvalidNodeConfig", err)
			}
		}
		if _, err := nilNode.ValidateInput("x"); !errors.Is(err, classifier.ErrInvalidNodeConfig) {
			t.Fatalf("nil (*Node).ValidateInput() error = %v, want ErrInvalidNodeConfig", err)
		}
		if _, err := zero.ValidateInput("x"); !errors.Is(err, classifier.ErrInvalidNodeConfig) {
			t.Fatalf("zero Node.ValidateInput() error = %v, want ErrInvalidNodeConfig", err)
		}

		node, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: &fixedClassifier{answer: choiceAnswer("billing")},
			Question:   triageQuestion,
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		for _, runErr := range node.Run(nil, "x") {
			if runErr == nil || !strings.Contains(runErr.Error(), "nil context") {
				t.Fatalf("Node.Run(nil) error = %v, want nil context error", runErr)
			}
		}
		for _, runErr := range node.Run(agent.Promote(nil), 42) {
			if !errors.Is(runErr, classifier.ErrInvalidInput) || !errors.Is(runErr, workflow.ErrInputValidation) {
				t.Fatalf("Node.Run(42) error = %v, want ErrInvalidInput + ErrInputValidation", runErr)
			}
		}
	})

	t.Run("input_validation_errors_not_retried_under_default_retry_config", func(t *testing.T) {
		fc := &fixedClassifier{answer: choiceAnswer("billing")}
		badSource := workflow.NewFunctionNode("src", func(_ agent.Context, _ any) (int, error) {
			return 42, nil
		}, workflow.NodeConfig{})
		node, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: fc,
			Question:   triageQuestion,
			NodeConfig: workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()},
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		sinkNode, _ := newCapturingNode("sink", "")
		_, _, err = runWorkflowViaRunner(
			t,
			[]workflow.Edge{
				{From: workflow.Start, To: badSource},
				{From: badSource, To: node},
				{From: node, To: sinkNode},
			},
			genai.NewContentFromText("go", genai.RoleUser),
		)
		if !errors.Is(err, workflow.ErrInputValidation) || !errors.Is(err, classifier.ErrInvalidInput) {
			t.Fatalf("run error = %v, want ErrInputValidation + ErrInvalidInput", err)
		}
		if len(fc.calls) != 0 {
			t.Errorf("classifier calls = %d, want 0", len(fc.calls))
		}
	})

	// A classifier rejecting the input or the question fails identically on
	// every attempt, so Run marks those errors as input validation failures;
	// other classifier errors keep the retry behavior (adk-python retries all).
	t.Run("deterministic_classifier_errors_not_retried_under_default_retry_config", func(t *testing.T) {
		transient := errors.New("upstream failed")
		cases := []struct {
			name      string
			clfErr    error
			wantCalls int
			wantInput bool // errors.Is(err, workflow.ErrInputValidation)
		}{
			{name: "invalid_question", clfErr: fmt.Errorf("%w: needs instructions", classifier.ErrInvalidQuestion), wantCalls: 1, wantInput: true},
			{name: "invalid_input", clfErr: fmt.Errorf("%w: non-text part", classifier.ErrInvalidInput), wantCalls: 1, wantInput: true},
			{name: "other_error_still_retried", clfErr: transient, wantCalls: 2},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				retry := workflow.DefaultRetryConfig()
				retry.MaxAttempts = 2
				retry.InitialDelay = 0
				retry.MaxDelay = 0
				fc := &fixedClassifier{err: tc.clfErr}
				node, err := classifier.NewNode(classifier.NodeConfig{
					Name:       "triage",
					Classifier: fc,
					Question:   triageQuestion,
					NodeConfig: workflow.NodeConfig{RetryConfig: retry},
				})
				if err != nil {
					t.Fatalf("NewNode() unexpected error: %v", err)
				}
				sinkNode, _ := newCapturingNode("sink", "")
				_, _, err = runWorkflowViaRunner(
					t,
					[]workflow.Edge{
						{From: workflow.Start, To: node},
						{From: node, To: sinkNode},
					},
					genai.NewContentFromText("ticket", genai.RoleUser),
				)
				if !errors.Is(err, tc.clfErr) {
					t.Fatalf("run error = %v, want errors.Is(%v)", err, tc.clfErr)
				}
				if got := errors.Is(err, workflow.ErrInputValidation); got != tc.wantInput {
					t.Errorf("errors.Is(err, ErrInputValidation) = %t, want %t: %v", got, tc.wantInput, err)
				}
				if tc.wantInput {
					if got := strings.Count(err.Error(), workflow.ErrInputValidation.Error()); got != 1 {
						t.Errorf("run error mentions ErrInputValidation %d times, want 1: %v", got, err)
					}
				}
				if len(fc.calls) != tc.wantCalls {
					t.Errorf("classifier calls = %d, want %d", len(fc.calls), tc.wantCalls)
				}
			})
		}
	})
}

func TestGraphOutgoingEdgeValidation(t *testing.T) {
	makeChoiceNode := func(t *testing.T) *classifier.Node {
		t.Helper()
		criteria := map[string]any{"billing": "Charges", "technical": "Bugs"}
		n, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "triage",
			Classifier: &fixedClassifier{answer: choiceAnswer("billing")},
			Question:   classifier.Choice{Instructions: "Route", Criteria: criteria},
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		// Mutating the caller's map after NewNode must not affect validation.
		criteria["injected"] = "should be ignored"
		return n
	}
	makeBinaryNode := func(t *testing.T) *classifier.Node {
		t.Helper()
		n, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "gate",
			Classifier: &fixedClassifier{answer: classifier.BinaryAnswer{Probability: 1.0}},
			Question:   classifier.Binary{Instructions: "Urgent?"},
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		return n
	}

	a, _ := newCapturingNode("a", "")
	b, _ := newCapturingNode("b", "")

	invalidGraphs := []struct {
		name       string
		edges      func(t *testing.T) []workflow.Edge
		wantSubstr string
	}{
		{
			name: "choice_unknown_string_route",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("biling")},
					{From: n, To: b, Route: workflow.StringRoute("technical")},
				}
			},
			wantSubstr: `never emits: ["biling"]`,
		},
		{
			name: "choice_caller_map_mutation_does_not_leak_route",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("injected")},
					{From: n, To: b, Route: workflow.Default},
				}
			},
			wantSubstr: `never emits: ["injected"]`,
		},
		{
			name: "choice_bool_route_rejected",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.BoolRoute(true)},
					{From: n, To: b, Route: workflow.Default},
				}
			},
			wantSubstr: "never emits: [true]",
		},
		{
			name: "choice_missing_label_without_fallback",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("billing")},
				}
			},
			wantSubstr: `no edge for routes ["technical"]`,
		},
		{
			name: "binary_string_routes_rejected",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeBinaryNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("true")},
					{From: n, To: b, Route: workflow.StringRoute("false")},
				}
			},
			wantSubstr: `never emits: ["false", "true"]`,
		},
		{
			name: "binary_missing_false_route_without_fallback",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeBinaryNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.BoolRoute(true)},
				}
			},
			wantSubstr: "no edge for routes [false]",
		},
		{
			name: "binary_missing_true_route_without_fallback",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeBinaryNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.BoolRoute(false)},
				}
			},
			wantSubstr: "no edge for routes [true]",
		},
		{
			name: "empty_multiroute_rejected",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.MultiRoute[string]{}},
					{From: n, To: b, Route: workflow.Default},
				}
			},
			wantSubstr: "never emits: [[]]",
		},
	}

	for _, tc := range invalidGraphs {
		t.Run("invalid_"+tc.name, func(t *testing.T) {
			_, err := workflow.New("wf", tc.edges(t))
			if !errors.Is(err, classifier.ErrInvalidRoute) {
				t.Fatalf("workflow.New() error = %v, want errors.Is(ErrInvalidRoute)", err)
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("workflow.New() error = %q, want substring %q", err.Error(), tc.wantSubstr)
			}
		})
	}

	validGraphs := []struct {
		name  string
		edges func(t *testing.T) []workflow.Edge
	}{
		{
			name: "default_route_covers_missing_choice_labels",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("billing")},
					{From: n, To: b, Route: workflow.Default},
				}
			},
		},
		{
			name: "only_default_route_covers_all_labels",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.Default},
				}
			},
		},
		{
			name: "unrouted_edge_covers_missing_choice_labels",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.StringRoute("billing")},
					{From: n, To: b},
				}
			},
		},
		{
			name: "only_unrouted_edge_skips_route_validation",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a},
				}
			},
		},
		{
			name: "terminal_classifier_node_with_no_outgoing_edges",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
				}
			},
		},
		{
			name: "multiroute_string_covers_all_choice_labels",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeChoiceNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.MultiRoute[string]{"billing", "technical"}},
				}
			},
		},
		{
			name: "multiroute_bool_covers_all_binary_routes",
			edges: func(t *testing.T) []workflow.Edge {
				n := makeBinaryNode(t)
				return []workflow.Edge{
					{From: workflow.Start, To: n},
					{From: n, To: a, Route: workflow.MultiRoute[bool]{true, false}},
				}
			},
		},
	}

	for _, tc := range validGraphs {
		t.Run("valid_"+tc.name, func(t *testing.T) {
			if _, err := workflow.New("wf", tc.edges(t)); err != nil {
				t.Fatalf("workflow.New() unexpected error: %v", err)
			}
		})
	}

	t.Run("zero_value_node_in_workflow_rejected", func(t *testing.T) {
		zero := &classifier.Node{}
		_, err := workflow.New("wf", []workflow.Edge{
			{From: workflow.Start, To: zero},
			{From: zero, To: a, Route: workflow.StringRoute("billing")},
		})
		if !errors.Is(err, classifier.ErrInvalidNodeConfig) {
			t.Fatalf("workflow.New(zero Node) error = %v, want ErrInvalidNodeConfig", err)
		}
	})
}

func TestNewNodeValidationAndSchemas(t *testing.T) {
	clf := &fixedClassifier{answer: choiceAnswer("billing")}
	thZero := 0.0
	thOne := 1.0
	thValid := 0.8
	thNeg := -0.1
	thOver := 1.1
	thNaN := math.NaN()

	// Boundary Threshold values 0.0 and 1.0 are valid on Binary questions.
	for _, th := range []*float64{&thZero, &thOne} {
		if _, err := classifier.NewNode(classifier.NodeConfig{
			Name:       "gate",
			Classifier: clf,
			Question:   classifier.Binary{},
			Threshold:  th,
		}); err != nil {
			t.Fatalf("NewNode(Threshold=%g) unexpected error: %v", *th, err)
		}
	}

	badConfigs := []struct {
		name       string
		cfg        classifier.NodeConfig
		wantQErr   bool
		wantSubstr string
	}{
		{
			name: "empty_name",
			cfg:  classifier.NodeConfig{Classifier: clf, Question: triageQuestion},
		},
		{
			name: "whitespace_name",
			cfg:  classifier.NodeConfig{Name: "   ", Classifier: clf, Question: triageQuestion},
		},
		{
			name: "nil_classifier",
			cfg:  classifier.NodeConfig{Name: "triage", Question: triageQuestion},
		},
		{
			name:     "nil_question",
			cfg:      classifier.NodeConfig{Name: "triage", Classifier: clf},
			wantQErr: true,
		},
		{
			name: "valid_score_question_rejected_as_score",
			cfg: classifier.NodeConfig{
				Name:       "rate",
				Classifier: clf,
				Question:   classifier.Score{Criteria: []any{"low", "high"}},
			},
			wantQErr:   true,
			wantSubstr: "Score questions cannot select",
		},
		{
			name: "empty_score_question_rejected_as_score",
			cfg: classifier.NodeConfig{
				Name:       "rate",
				Classifier: clf,
				Question:   classifier.Score{},
			},
			wantQErr:   true,
			wantSubstr: "Score questions cannot select",
		},
		{
			name: "score_pointer_question_rejected",
			cfg: classifier.NodeConfig{
				Name:       "rate",
				Classifier: clf,
				Question:   &classifier.Score{Criteria: []any{"low", "high"}},
			},
			wantQErr:   true,
			wantSubstr: "Score questions cannot select",
		},
		{
			name: "threshold_on_choice_rejected",
			cfg: classifier.NodeConfig{
				Name:       "triage",
				Classifier: clf,
				Question:   triageQuestion,
				Threshold:  &thValid,
			},
			wantQErr: true,
		},
		{
			name: "negative_threshold",
			cfg: classifier.NodeConfig{
				Name:       "gate",
				Classifier: clf,
				Question:   classifier.Binary{},
				Threshold:  &thNeg,
			},
		},
		{
			name: "threshold_above_one",
			cfg: classifier.NodeConfig{
				Name:       "gate",
				Classifier: clf,
				Question:   classifier.Binary{},
				Threshold:  &thOver,
			},
		},
		{
			name: "nan_threshold",
			cfg: classifier.NodeConfig{
				Name:       "gate",
				Classifier: clf,
				Question:   classifier.Binary{},
				Threshold:  &thNaN,
			},
		},
		{
			name: "whitespace_answer_key",
			cfg: classifier.NodeConfig{
				Name:       "gate",
				Classifier: clf,
				Question:   classifier.Binary{},
				AnswerKey:  "   ",
			},
		},
		{
			name: "unresolvable_input_schema",
			cfg: classifier.NodeConfig{
				Name:        "gate",
				Classifier:  clf,
				Question:    classifier.Binary{},
				InputSchema: &jsonschema.Schema{Ref: "#/$defs/missing"},
			},
		},
		{
			name: "unresolvable_output_schema",
			cfg: classifier.NodeConfig{
				Name:         "gate",
				Classifier:   clf,
				Question:     classifier.Binary{},
				OutputSchema: &jsonschema.Schema{Ref: "#/$defs/missing"},
			},
		},
	}

	for _, tc := range badConfigs {
		t.Run(tc.name, func(t *testing.T) {
			_, err := classifier.NewNode(tc.cfg)
			if !errors.Is(err, classifier.ErrInvalidNodeConfig) {
				t.Fatalf("NewNode() error = %v, want errors.Is(ErrInvalidNodeConfig)", err)
			}
			if tc.wantQErr && !errors.Is(err, classifier.ErrInvalidQuestion) {
				t.Errorf("NewNode() error = %v, want errors.Is(ErrInvalidQuestion)", err)
			}
			if tc.wantSubstr != "" && !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("NewNode() error = %q, want substring %q", err.Error(), tc.wantSubstr)
			}
		})
	}

	// Execute InputSchema coercion/validation and genai.Content passthrough.
	objSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"ticket": {Type: "string"},
		},
		Required: []string{"ticket"},
	}
	typedClf := &fixedClassifier{answer: choiceAnswer("billing")}
	typedNode, err := classifier.NewNode(classifier.NodeConfig{
		Name:         "typed",
		Description:  "routes objects",
		Classifier:   typedClf,
		Question:     triageQuestion,
		InputSchema:  objSchema,
		OutputSchema: objSchema,
	})
	if err != nil {
		t.Fatalf("NewNode(schemas) unexpected error: %v", err)
	}
	sinkNode, sink := newCapturingNode("sink", "")
	if _, _, err := runWorkflowViaRunner(
		t,
		[]workflow.Edge{
			{From: workflow.Start, To: typedNode},
			{From: typedNode, To: sinkNode},
		},
		genai.NewContentFromText(`{"ticket":"inv-1"}`, genai.RoleUser),
	); err != nil {
		t.Fatalf("runWorkflowViaRunner(schema coercion) error: %v", err)
	}
	if diff := cmp.Diff(map[string]any{"ticket": "inv-1"}, typedClf.calls[0].Input); diff != "" {
		t.Errorf("coerced classifier input mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]any{map[string]any{"ticket": "inv-1"}}, sink.inputs); diff != "" {
		t.Errorf("coerced sink output mismatch (-want +got):\n%s", diff)
	}

	// Direct Run does not apply InputSchema (the engine does, once) but still
	// rejects values that are not classifier input.
	for _, runErr := range typedNode.Run(agent.Promote(nil), 42) {
		if !errors.Is(runErr, workflow.ErrInputValidation) || !errors.Is(runErr, classifier.ErrInvalidInput) {
			t.Fatalf("typedNode.Run(42) error = %v, want ErrInputValidation + ErrInvalidInput", runErr)
		}
	}

	// Predecessor emitting *genai.Content passes *genai.Content to Classifier and forwards it unchanged.
	wantContent := genai.NewContentFromText("multimodal content", genai.RoleUser)
	contentSource := workflow.NewFunctionNode("content_src", func(ctx agent.Context, _ any) (*session.Event, error) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = wantContent
		return ev, nil
	}, workflow.NodeConfig{})
	contentClf := &fixedClassifier{answer: choiceAnswer("billing")}
	contentNode, err := classifier.NewNode(classifier.NodeConfig{
		Name:       "content_triage",
		Classifier: contentClf,
		Question:   triageQuestion,
	})
	if err != nil {
		t.Fatalf("NewNode() error: %v", err)
	}
	contentSinkNode, contentSink := newCapturingNode("content_sink", "")
	if _, _, err := runWorkflowViaRunner(
		t,
		[]workflow.Edge{
			{From: workflow.Start, To: contentSource},
			{From: contentSource, To: contentNode},
			{From: contentNode, To: contentSinkNode},
		},
		genai.NewContentFromText("seed", genai.RoleUser),
	); err != nil {
		t.Fatalf("runWorkflowViaRunner(content) error: %v", err)
	}
	if diff := cmp.Diff(wantContent, contentClf.calls[0].Input); diff != "" {
		t.Errorf("genai.Content classifier input mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]any{wantContent}, contentSink.inputs); diff != "" {
		t.Errorf("genai.Content sink output mismatch (-want +got):\n%s", diff)
	}
}

// InputSchema coercion JSON-round-trips values into plain maps and slices, so
// the genai check must run on the raw input first and a Content must be
// unwrapped to its text rather than coerced into a Content-shaped map.
func TestInputSchemaDoesNotLaunderGenaiValues(t *testing.T) {
	objSchema := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"ticket": {Type: "string"}},
		Required:   []string{"ticket"},
	}
	arraySchema := &jsonschema.Schema{Type: "array"}
	stringSchema := &jsonschema.Schema{Type: "string"}
	imagePart := genai.NewPartFromBytes([]byte("png-bytes"), "image/png")
	thoughtPart := &genai.Part{Text: "secret reasoning", Thought: true}
	thoughtImagePart := &genai.Part{Thought: true, InlineData: &genai.Blob{Data: []byte("png-bytes"), MIMEType: "image/png"}}
	jsonTextContent := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		thoughtPart,
		{Text: `{"ticket":`, PartMetadata: map[string]any{"source": "ocr"}},
		{Text: `"inv-1"}`, ThoughtSignature: []byte("sig")},
	}}
	leaked := func(err error) bool {
		msg := err.Error()
		return strings.Contains(msg, "secret reasoning") ||
			strings.Contains(msg, "png-bytes") ||
			strings.Contains(msg, base64.StdEncoding.EncodeToString([]byte("png-bytes")))
	}

	newNode := func(t *testing.T, schema *jsonschema.Schema) (*classifier.Node, *fixedClassifier) {
		t.Helper()
		fc := &fixedClassifier{answer: choiceAnswer("billing")}
		node, err := classifier.NewNode(classifier.NodeConfig{
			Name:        "triage",
			Classifier:  fc,
			Question:    triageQuestion,
			InputSchema: schema,
		})
		if err != nil {
			t.Fatalf("NewNode() unexpected error: %v", err)
		}
		return node, fc
	}

	rejected := []struct {
		name   string
		schema *jsonschema.Schema
		input  any
		// schemaErr marks inputs rejected by the schema itself rather than by
		// the classifier input rules, which carry ErrInvalidInput.
		schemaErr bool
		// directRun marks inputs that Run rejects on its own: it re-checks only
		// classifier input rules, not InputSchema, which the engine applies.
		directRun bool
	}{
		{name: "nil_input", schema: objSchema, input: nil, directRun: true},
		{name: "content_history_under_array_schema", schema: arraySchema, input: []*genai.Content{jsonTextContent, {Parts: []*genai.Part{imagePart}}}, directRun: true},
		// A plain object schema would accept the Part's own JSON form.
		{name: "part_under_object_schema", schema: &jsonschema.Schema{Type: "object"}, input: &genai.Part{Text: "secret reasoning", Thought: true}, directRun: true},
		{name: "history_in_map_under_object_schema", schema: objSchema, input: map[string]any{"ticket": "inv-1", "history": []*genai.Content{jsonTextContent}}, directRun: true},
		{name: "content_with_image_under_object_schema", schema: objSchema, input: &genai.Content{Parts: []*genai.Part{{Text: `{"ticket":"inv-1"}`}, imagePart}}},
		{name: "content_with_thought_image_under_object_schema", schema: objSchema, input: &genai.Content{Parts: []*genai.Part{{Text: `{"ticket":"inv-1"}`}, thoughtImagePart}}},
		{name: "content_with_nil_part_under_string_schema", schema: stringSchema, input: genai.Content{Parts: []*genai.Part{nil}}},
		{name: "text_coerced_to_bare_number", schema: &jsonschema.Schema{Type: "integer"}, input: "42"},
		// The thought part is stripped before the schema error can echo the text.
		{name: "content_text_not_matching_schema", schema: objSchema, input: &genai.Content{Parts: []*genai.Part{thoughtPart, {Text: "not json"}}}, schemaErr: true},
	}
	for _, tc := range rejected {
		t.Run("rejects_"+tc.name, func(t *testing.T) {
			node, fc := newNode(t, tc.schema)
			_, err := node.ValidateInput(tc.input)
			if err == nil {
				t.Fatalf("ValidateInput(%T) succeeded, want error", tc.input)
			}
			if !tc.schemaErr && !errors.Is(err, classifier.ErrInvalidInput) {
				t.Errorf("ValidateInput(%T) error = %v, want ErrInvalidInput", tc.input, err)
			}
			if leaked(err) {
				t.Errorf("ValidateInput(%T) error leaked input content: %v", tc.input, err)
			}
			if !tc.directRun {
				return
			}
			yields := 0
			for _, runErr := range node.Run(agent.Promote(nil), tc.input) {
				yields++
				if !errors.Is(runErr, workflow.ErrInputValidation) || !errors.Is(runErr, classifier.ErrInvalidInput) {
					t.Fatalf("Run(%T) error = %v, want ErrInputValidation + ErrInvalidInput", tc.input, runErr)
				}
				if leaked(runErr) {
					t.Errorf("Run(%T) error leaked input content: %v", tc.input, runErr)
				}
				if got := strings.Count(runErr.Error(), workflow.ErrInputValidation.Error()); got != 1 {
					t.Errorf("Run(%T) error mentions ErrInputValidation %d times, want 1: %v", tc.input, got, runErr)
				}
			}
			if yields != 1 {
				t.Errorf("Run(%T) yielded %d times, want 1 error", tc.input, yields)
			}
			if len(fc.calls) != 0 {
				t.Errorf("classifier calls = %d, want 0", len(fc.calls))
			}
		})
	}
	// runWithPredecessor feeds output from a predecessor node into a Node with
	// schema and returns what the classifier and the downstream sink received.
	runWithPredecessor := func(t *testing.T, schema *jsonschema.Schema, output any) (*fixedClassifier, *capturingSink, error) {
		t.Helper()
		source := workflow.NewFunctionNode("src", func(ctx agent.Context, _ any) (*session.Event, error) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Output = output
			return ev, nil
		}, workflow.NodeConfig{})
		node, fc := newNode(t, schema)
		sinkNode, sink := newCapturingNode("sink", "")
		_, _, err := runWorkflowViaRunner(
			t,
			[]workflow.Edge{
				{From: workflow.Start, To: source},
				{From: source, To: node},
				{From: node, To: sinkNode},
			},
			genai.NewContentFromText("seed", genai.RoleUser),
		)
		return fc, sink, err
	}

	t.Run("content_text_is_unwrapped_and_thoughts_skipped", func(t *testing.T) {
		for _, input := range []any{jsonTextContent, *jsonTextContent} {
			node, _ := newNode(t, objSchema)
			got, err := node.ValidateInput(input)
			if err != nil {
				t.Fatalf("ValidateInput(%T) unexpected error: %v", input, err)
			}
			if diff := cmp.Diff(map[string]any{"ticket": "inv-1"}, got); diff != "" {
				t.Errorf("ValidateInput(%T) mismatch (-want +got):\n%s", input, diff)
			}
			fc, sink, err := runWithPredecessor(t, objSchema, input)
			if err != nil {
				t.Fatalf("run(%T) unexpected error: %v", input, err)
			}
			if len(fc.calls) != 1 {
				t.Fatalf("classifier calls = %d, want 1", len(fc.calls))
			}
			if diff := cmp.Diff(map[string]any{"ticket": "inv-1"}, fc.calls[0].Input); diff != "" {
				t.Errorf("classifier input mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]any{map[string]any{"ticket": "inv-1"}}, sink.inputs); diff != "" {
				t.Errorf("sink input mismatch (-want +got):\n%s", diff)
			}
		}
	})

	t.Run("content_text_under_string_schema_stays_text", func(t *testing.T) {
		input := &genai.Content{Parts: []*genai.Part{thoughtPart, {Text: "plain "}, {Text: "ticket"}}}
		fc, _, err := runWithPredecessor(t, stringSchema, input)
		if err != nil {
			t.Fatalf("run unexpected error: %v", err)
		}
		if len(fc.calls) != 1 || fc.calls[0].Input != "plain ticket" {
			t.Errorf("classifier calls = %+v, want one call with input %q", fc.calls, "plain ticket")
		}
		// A thought-only Content unwraps to "", as adk-python's
		// extract_text_from_content returns ''.
		node, _ := newNode(t, stringSchema)
		if got, err := node.ValidateInput(&genai.Content{Parts: []*genai.Part{thoughtPart}}); err != nil || got != "" {
			t.Errorf("ValidateInput(thought-only Content) = %#v, %v; want \"\", nil", got, err)
		}
	})

	t.Run("content_without_schema_is_passed_unchanged", func(t *testing.T) {
		input := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "ticket"}, imagePart}}
		fc, _, err := runWithPredecessor(t, nil, input)
		if err != nil {
			t.Fatalf("run unexpected error: %v", err)
		}
		if len(fc.calls) != 1 || fc.calls[0].Input != input {
			t.Errorf("classifier calls = %+v, want one call with the original *genai.Content", fc.calls)
		}
	})

	t.Run("predecessor_history_under_array_schema_fails_the_run", func(t *testing.T) {
		history := []*genai.Content{jsonTextContent, {Parts: []*genai.Part{imagePart}}}
		fc, sink, err := runWithPredecessor(t, arraySchema, history)
		if !errors.Is(err, workflow.ErrInputValidation) || !errors.Is(err, classifier.ErrInvalidInput) {
			t.Fatalf("run error = %v, want ErrInputValidation + ErrInvalidInput", err)
		}
		if got := strings.Count(err.Error(), workflow.ErrInputValidation.Error()); got != 1 {
			t.Errorf("run error mentions ErrInputValidation %d times, want 1: %v", got, err)
		}
		if len(fc.calls) != 0 || len(sink.inputs) != 0 {
			t.Errorf("classifier calls = %d, sink inputs = %d, want 0 and 0", len(fc.calls), len(sink.inputs))
		}
	})

	// InputSchema is applied once, by the engine. A second pass in Run would
	// JSON-decode the string "42" again and fail it as a bare number.
	t.Run("schema_is_applied_once", func(t *testing.T) {
		anyOf := &jsonschema.Schema{AnyOf: []*jsonschema.Schema{{Type: "string"}, {Type: "number"}}}
		fc, sink, err := runWithPredecessor(t, anyOf, genai.NewContentFromText(`"42"`, genai.RoleUser))
		if err != nil {
			t.Fatalf("run unexpected error: %v", err)
		}
		if len(fc.calls) != 1 || fc.calls[0].Input != "42" {
			t.Errorf("classifier calls = %+v, want one call with the string \"42\"", fc.calls)
		}
		if diff := cmp.Diff([]any{"42"}, sink.inputs); diff != "" {
			t.Errorf("sink input mismatch (-want +got):\n%s", diff)
		}
	})
}
