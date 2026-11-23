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

package session_test

import (
	"context"
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
)

// weatherTurnModel calls get_weather on its first request and answers with
// text on every later one.
type weatherTurnModel struct{ calls int }

func (*weatherTurnModel) Name() string { return "weather-turn" }

func (m *weatherTurnModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.calls++
	content := genai.NewContentFromText("It is sunny in Paris.", genai.RoleModel)
	if m.calls == 1 {
		content = genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Paris"}, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

func describeNodeInfo(ev *session.Event) string {
	if ev.NodeInfo == nil {
		return fmt.Sprintf("NodeInfo nil, name %q", ev.NodeInfo.Name())
	}
	return fmt.Sprintf("path %q, name %q", ev.NodeInfo.Path, ev.NodeInfo.Name())
}

// The runner runs an LLM agent as a workflow node even when the caller wrote
// no workflow, so every event the agent emits carries a NodeInfo whose path is
// the agent's name. The user's message event and an event built by hand carry
// none, and the accessors of a nil NodeInfo return "".
func ExampleNodeInfo_Name() {
	type weatherArgs struct {
		City string `json:"city"`
	}
	getWeather, err := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Returns the forecast for a city."},
		func(_ agent.Context, args weatherArgs) (map[string]any, error) {
			return map[string]any{"forecast": "sunny"}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: &weatherTurnModel{},
		Tools: []tool.Tool{getWeather},
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("Weather in Paris?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}, runner.WithYieldUserMessage()) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[%s] %s\n", ev.Author, describeNodeInfo(ev))
	}

	fmt.Printf("[built by hand] %s\n", describeNodeInfo(&session.Event{}))
	// Output:
	// [user] NodeInfo nil, name ""
	// [assistant] path "assistant", name "assistant"
	// [assistant] path "assistant", name "assistant"
	// [assistant] path "assistant", name "assistant"
	// [built by hand] NodeInfo nil, name ""
}
