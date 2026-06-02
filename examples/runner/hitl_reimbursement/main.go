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

// Command hitl_reimbursement is an interactive human-in-the-loop (HITL)
// example you can drive from the console launcher. It is the adk-go port
// of adk-python's contributing/samples/human_in_loop.
//
// It is a plain LlmAgent — nothing opts into the node runtime. The Runner
// detects the LlmAgent and routes it through the ADK 2.0 node path, which
// bridges the agent's long-running tool into a workflow pause/resume.
//
// The reimbursement agent auto-approves amounts under $100. For larger
// amounts it calls the long-running tool ask_for_approval, which returns
// "pending" and pauses the run. The console renders the pending tool call
// as a prompt; your reply is sent back as the tool's FunctionResponse,
// resuming the agent (which then calls reimburse and confirms).
//
// Run it (needs Gemini credentials; see examples/bidi/README.md):
//
//	export GOOGLE_API_KEY="your_api_key_here"
//	go run ./examples/runner/hitl_reimbursement/ console
//
// Example session:
//
//	User -> Reimburse $200 for my conference ticket.
//	Agent -> ... (calls ask_for_approval; console shows the pending prompt)
//	User -> {"status":"approved","ticketId":"reimbursement-ticket-001"}
//	Agent -> Your $200 reimbursement has been approved and processed.
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// reimburseArgs / approvalArgs are the tool argument schemas. Field names
// map to the JSON the model produces for each tool call.
type reimburseArgs struct {
	Purpose string  `json:"purpose"`
	Amount  float64 `json:"amount"`
}

type approvalArgs struct {
	Purpose string  `json:"purpose"`
	Amount  float64 `json:"amount"`
}

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("gemini.NewModel: %v", err)
	}

	// reimburse: a normal tool that completes immediately.
	reimburse, err := functiontool.New(functiontool.Config{
		Name:        "reimburse",
		Description: "Reimburse the amount of money to the employee.",
	}, func(_ agent.ToolContext, _ reimburseArgs) (map[string]string, error) {
		return map[string]string{"status": "ok"}, nil
	})
	if err != nil {
		log.Fatalf("functiontool.New(reimburse): %v", err)
	}

	// ask_for_approval: a LONG-RUNNING tool. It returns "pending"
	// immediately; the real decision arrives later as the human's reply,
	// which the Runner's node path routes back to this call on resume.
	askForApproval, err := functiontool.New(functiontool.Config{
		Name:          "ask_for_approval",
		Description:   "Ask for approval for the reimbursement.",
		IsLongRunning: true,
	}, func(_ agent.ToolContext, args approvalArgs) (map[string]any, error) {
		return map[string]any{
			"status":   "pending",
			"amount":   args.Amount,
			"ticketId": "reimbursement-ticket-001",
		}, nil
	})
	if err != nil {
		log.Fatalf("functiontool.New(ask_for_approval): %v", err)
	}

	// A plain LlmAgent. Same behavior as adk-python's human_in_loop sample.
	reimbursementAgent, err := llmagent.New(llmagent.Config{
		Name:  "reimbursement_agent",
		Model: model,
		Description: "Handles employee reimbursements, asking a manager for " +
			"approval when the amount exceeds $100.",
		Instruction: `You are an agent whose job is to handle the reimbursement process for
the employees. If the amount is less than $100, you will automatically
approve the reimbursement and call reimburse().

If the amount is greater than $100, you will ask for approval from the
manager by calling ask_for_approval(). If the manager approves, you will
call reimburse() to reimburse the amount to the employee. If the manager
rejects, you will inform the employee of the rejection.`,
		Tools: []tool.Tool{reimburse, askForApproval},
	})
	if err != nil {
		log.Fatalf("llmagent.New: %v", err)
	}

	log.Printf("hitl_reimbursement ready — try: \"Reimburse $200 for my conference ticket.\"")

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(reimbursementAgent),
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
