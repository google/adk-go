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

package authtool_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// A tool that calls an API with an API key. The function receives the key as
// a credential; the model sees only the city parameter.
func ExampleNew() {
	type weatherArgs struct {
		City string `json:"city"`
	}
	weather, err := authtool.New(authtool.Config{
		Config: functiontool.Config{Name: "get_weather", Description: "Returns the weather in a city."},
		AuthConfig: &toolauth.Config{
			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
			RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "example-key"},
		},
	}, func(ctx agent.Context, a weatherArgs, cred *toolauth.Credential, authErr error) (map[string]any, error) {
		// A real tool sends its request with the header X-Api-Key: cred.APIKey.
		return map[string]any{"city": a.City, "forecast": "sunny", "key_set": cred.APIKey != ""}, nil
	})
	if err != nil {
		log.Fatal(err)
	}

	// A scripted model stands in for Gemini: it calls the tool, then answers.
	llm := &scriptedModel{replies: []*genai.Content{
		genai.NewContentFromFunctionCall("get_weather", map[string]any{"city": "Paris"}, genai.RoleModel),
		genai.NewContentFromText("It is sunny in Paris.", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "weather_agent", Model: llm, Tools: []tool.Tool{weather}})
	if err != nil {
		log.Fatal(err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "weather", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	s, err := sessions.Create(ctx, &session.CreateRequest{AppName: "weather", UserID: "user-1"})
	if err != nil {
		log.Fatal(err)
	}

	msg := genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user-1", s.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionResponse != nil:
				fmt.Println("tool:", p.FunctionResponse.Response)
			case p.Text != "":
				fmt.Println("model:", p.Text)
			}
		}
	}
	// Output:
	// tool: map[city:Paris forecast:sunny key_set:true]
	// model: It is sunny in Paris.
}

// scriptedModel answers each request with the next of its replies.
type scriptedModel struct {
	replies []*genai.Content
}

func (m *scriptedModel) Name() string { return "scripted" }

func (m *scriptedModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(m.replies) == 0 {
			yield(nil, fmt.Errorf("scripted model has no reply left"))
			return
		}
		reply := m.replies[0]
		m.replies = m.replies[1:]
		yield(&model.LLMResponse{Content: reply, FinishReason: genai.FinishReasonStop}, nil)
	}
}
