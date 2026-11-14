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

package multimodaltoolresults_test

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/multimodaltoolresults"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// printingModel stands in for a real model: it prints the parts of the last
// content of each request it receives. It calls the tool "get_report" when the
// user's message starts with "report", and otherwise replies "ok".
type printingModel struct{}

func (printingModel) Name() string { return "printing" }

func (printingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	last := req.Contents[len(req.Contents)-1]
	var parts []string
	for _, p := range last.Parts {
		switch {
		case p.FunctionResponse != nil:
			parts = append(parts, "response:"+p.FunctionResponse.Name)
		default:
			parts = append(parts, p.Text)
		}
	}
	fmt.Println(parts)

	reply := genai.NewContentFromText("ok", genai.RoleModel)
	if last.Role == genai.RoleUser && len(last.Parts) > 0 && strings.HasPrefix(last.Parts[0].Text, "report") {
		reply = genai.NewContentFromFunctionCall("get_report", map[string]any{}, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: reply}, nil)
	}
}

// converse runs one session with a runner that has plugin p and a tool that
// returns a text part, and sends each message in turn.
func converse(p *plugin.Plugin, messages ...string) {
	ctx := context.Background()
	report, err := functiontool.New(functiontool.Config{Name: "get_report", Description: "Returns the report."},
		func(agent.Context, struct{}) (*genai.Part, error) {
			return genai.NewPartFromText("sales rose 4%"), nil
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: printingModel{}, Tools: []tool.Tool{report}})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: sessions,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, text := range messages {
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
		}
	}
}

// With the default retention, the text part the tool returned is attached
// after the function response in the next model request only.
func ExampleNew() {
	p, err := multimodaltoolresults.New(multimodaltoolresults.Config{})
	if err != nil {
		log.Fatal(err)
	}

	converse(p, "report please", "thanks")
	// Output:
	// [report please]
	// [response:get_report sales rose 4%]
	// [thanks]
}

// With session retention, the text part the tool returned is attached to
// every later model request of the session, in later turns too.
func ExampleNew_session() {
	p, err := multimodaltoolresults.New(multimodaltoolresults.Config{Retention: multimodaltoolresults.RetentionSession})
	if err != nil {
		log.Fatal(err)
	}

	converse(p, "report please", "thanks")
	// Output:
	// [report please]
	// [response:get_report sales rose 4%]
	// [thanks sales rose 4%]
}
