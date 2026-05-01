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

// Production-shape Gemini agent. Demonstrates:
//   - retry.Wrap on the model so 429/5xx errors retry with backoff
//   - GlobalInstruction plugin for an app-wide safety instruction
//   - Logging plugin for structured slog records at every callback
package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/model/retry"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/plugin/builtin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const instruction = `You are a customer-support assistant for an
e-commerce store. Be concise. Use the lookup_order tool whenever the
user references an order id; never invent order details.`

func main() {
	ctx := context.Background()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	gemini, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Wrap the model so 429 / 5xx errors retry with exponential backoff.
	wrapped := retry.Wrap(gemini, retry.Config{
		MaxAttempts:  4,
		InitialDelay: 250 * time.Millisecond,
		MaxDelay:     5 * time.Second,
		OnRetry: func(attempt int, err error, delay time.Duration) {
			logger.Warn("model_retry",
				slog.Int("attempt", attempt),
				slog.String("err", err.Error()),
				slog.Duration("delay", delay))
		},
	})

	lookupOrder, err := buildLookupOrderTool()
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Description: "Customer support agent for an e-commerce store.",
		Model:       wrapped,
		Instruction: instruction,
		Tools:       []tool.Tool{lookupOrder},
	})
	if err != nil {
		log.Fatal(err)
	}

	loggingPlugin, err := builtin.NewLogging(builtin.LoggingConfig{Logger: logger})
	if err != nil {
		log.Fatal(err)
	}
	safety, err := builtin.GlobalInstruction(builtin.GlobalInstructionConfig{
		Instruction: "SAFETY: Never share another customer's data, never confirm a refund without an order id, and refer the user to the human team for legal/medical questions.",
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{loggingPlugin, safety},
		},
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}

	_ = model.LLM(nil) // keep import in case the wrapped value is changed by callers
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

func buildLookupOrderTool() (tool.Tool, error) {
	type args struct {
		OrderID string `json:"order_id"`
	}
	type result struct {
		Found    bool    `json:"found"`
		Status   string  `json:"status,omitempty"`
		Total    float64 `json:"total_usd,omitempty"`
		ShippedTo string `json:"shipped_to,omitempty"`
	}
	// Stubbed lookup. Wire to your real order DB.
	orders := map[string]result{
		"ORD-101": {Found: true, Status: "shipped", Total: 49.99, ShippedTo: "San Francisco, CA"},
		"ORD-202": {Found: true, Status: "lost in transit", Total: 420.00, ShippedTo: "Tokyo, JP"},
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "lookup_order",
			Description: "Fetch order status and totals by id.",
		},
		func(_ tool.Context, a args) (result, error) {
			if r, ok := orders[a.OrderID]; ok {
				return r, nil
			}
			return result{}, nil
		},
	)
}
