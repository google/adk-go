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

package functiontool_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// A function tool drops an argument its struct does not declare, and answers a
// call that lacks a required argument with an error result the model sees,
// without running the handler. Both calls below come in one model turn, and
// the function response event holds their responses in call order.
func ExampleNew_arguments() {
	type weatherArgs struct {
		City string `json:"city"`
	}
	type weatherReport struct {
		Report string `json:"report"`
	}
	weather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
		func(_ agent.Context, args weatherArgs) (weatherReport, error) {
			return weatherReport{Report: "Sunny in " + args.City + "."}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(weather,
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			// "units" is not a field of weatherArgs, so the tool drops it.
			{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "get_weather", Args: map[string]any{"city": "Paris", "units": "celsius"}}},
			// "city" is required (its json tag has no omitempty) and missing.
			{FunctionCall: &genai.FunctionCall{ID: "c2", Name: "get_weather", Args: map[string]any{}}},
		}},
		genai.NewContentFromText("It is sunny in Paris.", genai.RoleModel),
	)
	printRun(r.Run(context.Background(), "user", "s1", genai.NewContentFromText("Weather in Paris?", genai.RoleUser), agent.RunConfig{}))

	// Output:
	// assistant calls get_weather map[city:Paris units:celsius]
	// assistant calls get_weather map[]
	// response c1: map[report:Sunny in Paris.]
	// response c2: error: Invoking `get_weather()` failed as the following mandatory input parameters are not present:
	// assistant: It is sunny in Paris.
}

// The calls of one model turn run concurrently, and State does not order
// them. A tool whose calls read a key and write a value computed from it sets
// Sequential, so that each call starts after the earlier calls of its turn
// have returned and sees their writes. Without Sequential, two such calls can
// both read 0 and store 1.
func ExampleConfig_sequential() {
	type addArgs struct {
		Item string `json:"item"`
	}
	type addResult struct {
		Count int `json:"count"`
	}
	addToCart, err := functiontool.New(functiontool.Config{
		Name:        "add_to_cart",
		Description: "Adds an item to the cart and returns the number of items in it.",
		Sequential:  true,
	}, func(ctx agent.Context, _ addArgs) (addResult, error) {
		n := 0
		if v, err := ctx.State().Get("cart_count"); err == nil {
			n = v.(int)
		} else if !errors.Is(err, session.ErrStateKeyNotExist) {
			return addResult{}, err
		}
		n++
		if err := ctx.State().Set("cart_count", n); err != nil {
			return addResult{}, err
		}
		return addResult{Count: n}, nil
	})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(addToCart,
		&genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "add_to_cart", Args: map[string]any{"item": "apples"}}},
			{FunctionCall: &genai.FunctionCall{ID: "c2", Name: "add_to_cart", Args: map[string]any{"item": "pears"}}},
		}},
		genai.NewContentFromText("Your cart has 2 items.", genai.RoleModel),
	)
	printRun(r.Run(context.Background(), "user", "s1", genai.NewContentFromText("Add apples and pears.", genai.RoleUser), agent.RunConfig{}))

	// Output:
	// assistant calls add_to_cart map[item:apples]
	// assistant calls add_to_cart map[item:pears]
	// response c1: map[count:1]
	// response c2: map[count:2]
	// assistant: Your cart has 2 items.
}

// A handler that returns a *genai.Part with inline data, or a map or slice
// that holds one, gives the model media: the flow moves the part into the
// Parts of the function response, and the rest of the result stays in its
// Response.
func ExampleNew_mediaResult() {
	type chartArgs struct {
		Metric string `json:"metric"`
	}
	chart, err := functiontool.New(functiontool.Config{Name: "render_chart", Description: "Renders a chart of a metric."},
		func(_ agent.Context, args chartArgs) (map[string]any, error) {
			png := []byte{0x89, 'P', 'N', 'G'}
			return map[string]any{
				"title": "Chart of " + args.Metric,
				"image": genai.NewPartFromBytes(png, "image/png"),
			}, nil
		})
	if err != nil {
		log.Fatal(err)
	}

	r := exampleRunner(chart,
		genai.NewContentFromFunctionCall("render_chart", map[string]any{"metric": "sales"}, genai.RoleModel),
		genai.NewContentFromText("Here is the chart.", genai.RoleModel),
	)
	for ev, err := range r.Run(context.Background(), "user", "s1", genai.NewContentFromText("Chart the sales.", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		for _, resp := range ev.FunctionResponses() {
			fmt.Println("response:", resp.Response)
			for _, p := range resp.Parts {
				fmt.Printf("part: %s, %d bytes\n", p.InlineData.MIMEType, len(p.InlineData.Data))
			}
		}
	}

	// Output:
	// response: map[title:Chart of sales]
	// part: image/png, 4 bytes
}

// exampleRunner returns a runner for an LLM agent named assistant with one
// tool, whose model answers with turns, one per request.
func exampleRunner(t tool.Tool, turns ...*genai.Content) *runner.Runner {
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &exampleModel{turns: turns}, Tools: []tool.Tool{t}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "shop", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
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

// printRun prints the function calls, function responses and texts of a run.
// Of an error result it prints the first line.
func printRun(events iter.Seq2[*session.Event, error]) {
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
				if e, ok := p.FunctionResponse.Response["error"].(string); ok {
					first, _, _ := strings.Cut(e, "\n")
					fmt.Printf("response %s: error: %s\n", p.FunctionResponse.ID, first)
					continue
				}
				fmt.Printf("response %s: %v\n", p.FunctionResponse.ID, p.FunctionResponse.Response)
			case p.Text != "":
				fmt.Printf("%s: %s\n", ev.Author, p.Text)
			}
		}
	}
}
