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

package tool_test

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
	"google.golang.org/adk/v2/tool/toolauth"
)

// weatherToolset is a toolset whose service lists its tools only to a client
// that sends an API key.
type weatherToolset struct{ cfg *toolauth.Config }

func (weatherToolset) Name() string { return "weather" }

// AuthConfig tells ADK which credential the toolset needs.
func (ts weatherToolset) AuthConfig() *toolauth.Config { return ts.cfg }

// Tools reads the credential ADK resolved for the config, under its key.
func (ts weatherToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	cred := agent.ResolvedCredential(ctx, ts.cfg.CredentialKey)
	if cred == nil {
		return nil, errors.New("weather: no credential")
	}
	fmt.Printf("weather lists its tools with API key %q\n", cred.APIKey)
	forecast, err := functiontool.New(functiontool.Config{Name: "get_forecast", Description: "Returns the forecast for a city."},
		func(agent.Context, struct{ City string }) (map[string]any, error) {
			return map[string]any{"forecast": "sunny"}, nil
		})
	if err != nil {
		return nil, err
	}
	return []tool.Tool{forecast}, nil
}

// replyModel answers every request with the same text.
type replyModel struct{ text string }

func (replyModel) Name() string { return "reply-model" }

func (m replyModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText(m.text, genai.RoleModel)}, nil)
	}
}

// A toolset that needs a credential before it can list its tools implements
// AuthenticatedToolset. Before each model call ADK resolves the credential its
// config describes, here an API key the config carries, and the toolset reads
// it with agent.ResolvedCredential under the config's key. A config without a
// ready credential, such as an OAuth2 one, makes ADK ask the client for it
// with an adk_request_credential call instead.
func ExampleAuthenticatedToolset() {
	ts := weatherToolset{cfg: &toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "demo-key"},
		CredentialKey:     "weather-api",
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:     "assistant",
		Model:    replyModel{text: "It is sunny."},
		Toolsets: []tool.Toolset{ts},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("credential key:", ts.cfg.CredentialKey)
	for ev, err := range r.Run(context.Background(), "user", "s1", genai.NewContentFromText("Weather?", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s: %s\n", ev.Author, ev.Content.Parts[0].Text)
	}

	// Output:
	// credential key: weather-api
	// weather lists its tools with API key "demo-key"
	// assistant: It is sunny.
}
