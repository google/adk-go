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

package generatortool_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/generatortool"
	"google.golang.org/adk/v2/workflow"
)

func ExampleNew() {
	type salesArgs struct {
		Region string `json:"region"`
	}
	type salesReport struct {
		Rows int `json:"rows"`
	}
	analyze, err := generatortool.New(functiontool.Config{Name: "analyze", Description: "Counts the sales rows of a region."},
		func(ctx agent.Context, args salesArgs, emit func(*session.Event) error) (salesReport, error) {
			// The user sees this event while the tool runs; the model does not.
			progress := &session.Event{LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("Counting rows in "+args.Region+"...", genai.RoleModel),
			}}
			if err := emit(progress); err != nil {
				return salesReport{}, err
			}
			// The result is the function response the model sees.
			return salesReport{Rows: 42}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(analyze,
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "analyze", Args: map[string]any{"region": "EMEA"}}}}},
		genai.NewContentFromText("EMEA has 42 rows.", genai.RoleModel),
	)
	printEvents(r.Run(context.Background(), "user", "s1", genai.NewContentFromText("How many rows in EMEA?", genai.RoleUser), agent.RunConfig{}))

	// Output:
	// root_agent calls analyze map[region:EMEA]
	// analyze: Counting rows in EMEA...
	// response from analyze: map[rows:42]
	// root_agent: EMEA has 42 rows.
}

func ExampleNew_pause() {
	ask, err := generatortool.New(functiontool.Config{Name: "ask", Description: "Analyzes the sales of a region the user picks."},
		func(ctx agent.Context, _ struct{}, emit func(*session.Event) error) (map[string]any, error) {
			// The first run asks the user and pauses: return the error.
			// When the user answers, the call runs again from the top and
			// ResumeOrRequestInput returns the answer.
			region, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "clarify_region", Message: "Which region?"})
			if err != nil {
				return nil, err
			}
			return map[string]any{"region": region}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(ask,
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "ask", Args: map[string]any{}}}}},
		genai.NewContentFromText("Analyzing APAC.", genai.RoleModel),
	)
	ctx := context.Background()
	fmt.Println("-- turn 1")
	printEvents(r.Run(ctx, "user", "s1", genai.NewContentFromText("Analyze sales.", genai.RoleUser), agent.RunConfig{}))
	fmt.Println("-- turn 2")
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: "clarify_region", Name: workflow.WorkflowInputFunctionCallName, Response: map[string]any{"result": "APAC"},
	}}}}
	printEvents(r.Run(ctx, "user", "s1", answer, agent.RunConfig{}))

	// Output:
	// -- turn 1
	// root_agent calls ask map[]
	// ask calls adk_request_input map[interruptId:clarify_region message:Which region? payload:<nil>]
	// -- turn 2
	// response from ask: map[region:APAC]
	// root_agent: Analyzing APAC.
}

// A generator tool whose calls read state and write a value computed from it
// sets Sequential, as a function tool does, so that the calls of one model
// turn run one after the other and each sees the earlier calls' writes.
func ExampleNew_sequential() {
	type countResult struct {
		Count int `json:"count"`
	}
	count, err := generatortool.New(functiontool.Config{Name: "count", Description: "Counts the calls.", Sequential: true},
		func(ctx agent.Context, _ struct{}, emit func(*session.Event) error) (countResult, error) {
			n := 0
			if v, err := ctx.State().Get("calls"); err == nil {
				n = v.(int)
			} else if !errors.Is(err, session.ErrStateKeyNotExist) {
				return countResult{}, err
			}
			n++
			if err := ctx.State().Set("calls", n); err != nil {
				return countResult{}, err
			}
			return countResult{Count: n}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(count,
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "count", Args: map[string]any{}}},
			{FunctionCall: &genai.FunctionCall{ID: "call_2", Name: "count", Args: map[string]any{}}},
		}},
		genai.NewContentFromText("Counted twice.", genai.RoleModel),
	)
	printEvents(r.Run(context.Background(), "user", "s1", genai.NewContentFromText("Count twice.", genai.RoleUser), agent.RunConfig{}))

	// Output:
	// root_agent calls count map[]
	// root_agent calls count map[]
	// response from count: map[count:1]
	// response from count: map[count:2]
	// root_agent: Counted twice.
}

// exampleRunner returns a runner for an LLM agent named root_agent with one
// tool, whose model answers with turns, one per request.
func exampleRunner(t tool.Tool, turns ...*genai.Content) *runner.Runner {
	a, err := llmagent.New(llmagent.Config{Name: "root_agent", Model: &exampleModel{turns: turns}, Tools: []tool.Tool{t}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "sales", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}
	return r
}

// exampleModel is a model that answers each request with the next of its
// scripted turns.
type exampleModel struct{ turns []*genai.Content }

func (m *exampleModel) Name() string { return "scripted" }

func (m *exampleModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(m.turns) == 0 {
			yield(nil, errors.New("script exhausted"))
			return
		}
		turn := m.turns[0]
		m.turns = m.turns[1:]
		yield(&model.LLMResponse{Content: turn, FinishReason: genai.FinishReasonStop}, nil)
	}
}

// printEvents prints the function calls, function responses and texts of a run.
func printEvents(events iter.Seq2[*session.Event, error]) {
	for ev, err := range events {
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Printf("%s calls %s %v\n", ev.Author, p.FunctionCall.Name, p.FunctionCall.Args)
			case p.FunctionResponse != nil:
				fmt.Printf("response from %s: %v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
			case p.Text != "":
				fmt.Printf("%s: %s\n", ev.Author, p.Text)
			}
		}
	}
}
