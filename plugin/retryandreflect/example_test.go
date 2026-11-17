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

package retryandreflect_test

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/retryandreflect"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// scriptedModel stands in for a real model: it calls the tool "check_quota"
// when the last content of the request is the user's message, and otherwise
// replies "ok".
type scriptedModel struct{}

func (scriptedModel) Name() string { return "scripted" }

func (scriptedModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	reply := genai.NewContentFromText("ok", genai.RoleModel)
	if last := req.Contents[len(req.Contents)-1]; last.Role == genai.RoleUser && last.Parts[0].FunctionResponse == nil {
		reply = genai.NewContentFromFunctionCall("check_quota", map[string]any{}, genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: reply}, nil)
	}
}

// A tool that reports a failure in its result rather than as an error still
// gets the plugin's reflection response when an extractor recognizes the
// failure.
func ExampleWithResultErrorExtractor() {
	ctx := context.Background()
	p, err := retryandreflect.New(retryandreflect.WithResultErrorExtractor(
		func(_ agent.Context, _ tool.Tool, _, result map[string]any) error {
			if result["status"] == "error" {
				return fmt.Errorf("%v", result["detail"])
			}
			return nil
		}))
	if err != nil {
		log.Fatal(err)
	}

	type quota struct {
		Status string `json:"status"`
		Detail string `json:"detail"`
	}
	checkQuota, err := functiontool.New(functiontool.Config{Name: "check_quota", Description: "Checks the quota."},
		func(agent.Context, struct{}) (quota, error) {
			return quota{Status: "error", Detail: "quota exceeded"}, nil
		})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: scriptedModel{}, Tools: []tool.Tool{checkQuota}})
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

	msg := genai.NewContentFromText("check my quota", genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if fr := part.FunctionResponse; fr != nil {
				fmt.Println(fr.Response["error_type"], fr.Response["retry_count"])
			}
		}
	}
	// Output: ToolError 1
}
