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

// Package main demonstrates the ErrGuardrailBlocked sentinel and
// OnGuardrailBlockedCallback for policy-aware tool enforcement.
//
// Run:
//
//	export GOOGLE_API_KEY=<your-api-key>
//	go run ./examples/compliance -- --console
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/guardrail"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// transferFunds simulates a funds transfer tool.
func transferFunds(ctx agent.Context, args map[string]any) (map[string]any, error) {
	recipient, _ := args["recipient"].(string)
	amount, _ := args["amount_minor_units"].(float64)
	ref := "TXN-XXXX"
	if len(recipient) >= 4 {
		ref = "TXN-" + strings.ToUpper(recipient[:4])
	}
	return map[string]any{
		"reference": ref,
		"status":    "submitted",
		"amount":    amount,
	}, nil
}

// policyGuardrail is a BeforeToolCallback that enforces two simple rules:
//   - Narrations containing flagged keywords are denied
//   - Transfers over 1,000,000 minor units are denied
func policyGuardrail(_ agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if t.Name() != "transfer_funds" {
		return nil, nil // only guard the transfer tool
	}
	narration := strings.ToLower(fmt.Sprintf("%v", args["narration"]))
	for _, kw := range []string{"launder", "evade", "fake"} {
		if strings.Contains(narration, kw) {
			return nil, &guardrail.ErrGuardrailBlocked{
				Policy: "aml-narration",
				Reason: "narration flagged by AML keyword screening",
			}
		}
	}
	amount, _ := args["amount_minor_units"].(float64)
	if amount > 1_000_000 {
		return nil, &guardrail.ErrGuardrailBlocked{
			Policy: "high-value-transfer",
			Reason: fmt.Sprintf("amount %.0f exceeds the single-transfer limit of 1,000,000", amount),
		}
	}
	return nil, nil
}

// blockedHandler is an OnGuardrailBlockedCallback that converts policy denials
// into structured tool responses the LLM can reason about.
func blockedHandler(_ agent.Context, t tool.Tool, _ map[string]any, blocked *guardrail.ErrGuardrailBlocked) (map[string]any, error) {
	log.Printf("[guardrail] tool=%q policy=%q reason=%q", t.Name(), blocked.Policy, blocked.Reason)
	return map[string]any{
		"status":  "blocked",
		"policy":  blocked.Policy,
		"message": blocked.Reason,
	}, nil
}

func main() {
	ctx := context.Background()

	mdl, err := gemini.NewModel(ctx, "gemini-2.0-flash-exp", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	transferTool, err := functiontool.New(functiontool.Config{
		Name:        "transfer_funds",
		Description: "Transfer funds. Args: recipient (string), amount_minor_units (integer), narration (string).",
	}, transferFunds)
	if err != nil {
		log.Fatalf("failed to create transfer tool: %v", err)
	}

	compliancePlugin, err := plugin.New(plugin.Config{
		Name:                       "compliance-guardrail",
		BeforeToolCallback:         policyGuardrail,
		OnGuardrailBlockedCallback: blockedHandler,
	})
	if err != nil {
		log.Fatalf("failed to create plugin: %v", err)
	}

	complianceAgent, err := llmagent.New(llmagent.Config{
		Name:  "compliance_agent",
		Model: mdl,
		Tools: []tool.Tool{transferTool},
		Instruction: `You are a payment assistant.
When a transfer is blocked by policy, explain clearly why it cannot proceed and what the user can do instead.`,
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(complianceAgent),
		PluginConfig: runner.PluginConfig{
			Plugins: []*plugin.Plugin{compliancePlugin},
		},
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
