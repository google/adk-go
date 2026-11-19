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

package agent_test

import (
	"context"
	"fmt"
	"iter"
	"log"
	"maps"
	"net/http"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// requestRecorder stands in for a real model: it keeps the request it
// receives and answers it with one text response.
type requestRecorder struct {
	lastReq *model.LLMRequest
}

func (*requestRecorder) Name() string { return "recorder" }

func (m *requestRecorder) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText("done", genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// runOnce runs a in a new in-memory session with one user message and cfg.
func runOnce(a agent.Agent, cfg agent.RunConfig) {
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
	msg := genai.NewContentFromText("hello", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, cfg) {
		if err != nil {
			log.Fatal(err)
		}
	}
}

// RunConfig.Labels are merged over the agent's own labels on every model
// request of the run, winning on a shared key. ADK also adds the
// "adk_agent_name" label, set to the name of the agent that makes the call,
// unless a label with that key is already present.
func ExampleRunConfig_labels() {
	llm := &requestRecorder{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "billing_agent",
		Model: llm,
		GenerateContentConfig: &genai.GenerateContentConfig{
			Labels: map[string]string{"team": "search", "env": "prod"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	runOnce(a, agent.RunConfig{
		Labels: &map[string]string{"team": "billing", "customer": "acme"},
	})

	labels := llm.lastReq.Config.Labels
	for _, k := range slices.Sorted(maps.Keys(labels)) {
		fmt.Printf("%s=%s\n", k, labels[k])
	}
	// Output:
	// adk_agent_name=billing_agent
	// customer=acme
	// env=prod
	// team=billing
}

// RunConfig.HTTPOptions is merged into the HTTP options of every model
// request of the run. When the agent sets its own HTTPOptions, headers are
// merged by canonical header name, with the run's value winning, and the
// agent's APIVersion and BaseURL are kept.
func ExampleRunConfig_httpOptions() {
	llm := &requestRecorder{}
	a, err := llmagent.New(llmagent.Config{
		Name:  "search_agent",
		Model: llm,
		GenerateContentConfig: &genai.GenerateContentConfig{
			HTTPOptions: &genai.HTTPOptions{
				APIVersion: "v1",
				Headers:    http.Header{"x-team": {"search"}},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	runOnce(a, agent.RunConfig{
		HTTPOptions: &genai.HTTPOptions{
			APIVersion: "v1beta1",
			Headers: http.Header{
				"X-Team":           {"billing"},
				"X-Request-Source": {"batch"},
			},
		},
	})

	opts := llm.lastReq.Config.HTTPOptions
	fmt.Println("APIVersion:", opts.APIVersion)
	for _, k := range slices.Sorted(maps.Keys(opts.Headers)) {
		fmt.Printf("%s: %v\n", k, opts.Headers[k])
	}
	// Output:
	// APIVersion: v1
	// X-Request-Source: [batch]
	// X-Team: [billing]
}
