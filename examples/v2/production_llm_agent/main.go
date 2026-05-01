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

// production_llm_agent shows the production-shape pattern: retried
// model + structured logging + global safety instruction.
package main

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"os"
	"sync/atomic"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/retry"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/plugin/builtin"
)

// flakyLLM fails the first 2 calls with HTTP 503 then succeeds.
type flakyLLM struct {
	calls atomic.Int32
}

func (f *flakyLLM) Name() string { return "flaky-stub" }
func (f *flakyLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		n := f.calls.Add(1)
		if n < 3 {
			yield(nil, genai.APIError{Code: 503, Status: "Service Unavailable"})
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: []*genai.Part{{Text: "Sure, the warranty on this item is two years."}},
		}}, nil)
	}
}

func main() {
	// Production wrapping: retry on 429/5xx with exponential backoff.
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	flaky := &flakyLLM{}
	wrapped := retry.Wrap(flaky, retry.Config{
		MaxAttempts:  4,
		InitialDelay: 50 * time.Millisecond,
		MaxDelay:     500 * time.Millisecond,
		Jitter:       -1, // deterministic for the demo
		OnRetry: func(attempt int, err error, delay time.Duration) {
			logger.Warn("model_retry",
				slog.Int("attempt", attempt),
				slog.String("err", err.Error()),
				slog.String("delay", delay.String()))
		},
	})

	// Two builtin plugins ready to compose into an App.
	logging, err := builtin.NewLogging(builtin.LoggingConfig{Logger: logger})
	if err != nil {
		fmt.Println(err)
		return
	}
	globalInst, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{
		Instruction: "You are a customer support assistant. Answer concisely. " +
			"Never make up policies; if you don't know, say so.",
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	// In production:
	//
	//   agent, _ := llmagent.New(llmagent.Config{
	//       Name: "support", Model: wrapped, Tools: []tool.Tool{...},
	//   })
	//   appCfg, _ := app.New(app.App{
	//       Name: "support_app", RootAgent: agent,
	//       Plugins: []*plugin.Plugin{logging, globalInst},
	//   })
	//   r, _ := runner.New(runner.Config{App: appCfg, SessionService: ...})
	//
	// Here we exercise the retry wrapper directly so the demo is offline.

	fmt.Println("=== single LLM call: 2 transient 503s, then success ===")
	req := &model.LLMRequest{Model: wrapped.Name()}
	for resp, err := range wrapped.GenerateContent(context.Background(), req, false) {
		if err != nil {
			fmt.Printf("FINAL ERROR: %v\n", err)
			return
		}
		if resp.Content != nil {
			fmt.Printf("RESPONSE: %s\n", resp.Content.Parts[0].Text)
		}
	}
	fmt.Printf("\ntotal model calls: %d (2 retries + 1 success)\n", flaky.calls.Load())
	_ = logging
	_ = globalInst
	_ = plugin.Plugin{}
}
