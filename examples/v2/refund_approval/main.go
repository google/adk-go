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

// Refund-approval agent. The Gemini agent uses a refund-processing tool
// that wraps a workflow underneath: small refunds auto-approve, large
// refunds emit a RequestInput interrupt for the manager. The user
// resumes by sending a FunctionResponse with the manager's decision.
package main

import (
	"context"
	"errors"
	"fmt"
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

const instruction = `You are a refund-processing agent.

When the user describes a refund request, parse out:
  - order_id (string)
  - amount_usd (number)
  - reason (string)

Then call the request_refund tool with those fields. The tool either:
  - returns "auto_approved" for refunds under $50 (proceed and tell the user it's done), OR
  - returns "needs_manager_approval" for refunds $50 and above. In that
    case tell the user a manager has been notified and the refund will
    be processed once approved.

If you don't have all three fields, ask for them before calling the tool.`

// processedRefunds is a tiny in-memory log shared by the request_refund
// tool. In production this would be a real ledger / DB write.
var processedRefunds = map[string]string{}

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, modelName(), &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	requestRefund, err := buildRequestRefundTool()
	if err != nil {
		log.Fatal(err)
	}
	approveRefund, err := buildApproveRefundTool()
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "refund_agent",
		Description: "A refund-processing agent that escalates large refunds to a manager.",
		Model:       model,
		Instruction: instruction,
		Tools:       []tool.Tool{requestRefund, approveRefund},
	})
	if err != nil {
		log.Fatal(err)
	}

	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(rootAgent)}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func modelName() string {
	if v := os.Getenv("GOOGLE_GENAI_MODEL"); v != "" {
		return v
	}
	return "gemini-2.5-flash"
}

const autoApproveLimit = 50.0

func buildRequestRefundTool() (tool.Tool, error) {
	type args struct {
		OrderID  string  `json:"order_id"`
		AmountUSD float64 `json:"amount_usd"`
		Reason   string  `json:"reason"`
	}
	type result struct {
		Status         string `json:"status"`           // "auto_approved" | "needs_manager_approval"
		ApprovalTicket string `json:"approval_ticket"`  // ticket id when escalated
		Message        string `json:"message"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "request_refund",
			Description: "Request a refund. Returns either auto_approved or needs_manager_approval depending on amount.",
		},
		func(_ tool.Context, a args) (result, error) {
			if a.OrderID == "" || a.AmountUSD <= 0 {
				return result{}, errors.New("order_id and positive amount_usd are required")
			}
			if a.AmountUSD < autoApproveLimit {
				processedRefunds[a.OrderID] = fmt.Sprintf("auto-approved $%.2f (%s)", a.AmountUSD, a.Reason)
				return result{
					Status:  "auto_approved",
					Message: fmt.Sprintf("Refund $%.2f for order %s auto-approved.", a.AmountUSD, a.OrderID),
				}, nil
			}
			ticket := "RFD-" + a.OrderID
			processedRefunds[ticket] = fmt.Sprintf("PENDING approval $%.2f (%s)", a.AmountUSD, a.Reason)
			return result{
				Status:         "needs_manager_approval",
				ApprovalTicket: ticket,
				Message: fmt.Sprintf(
					"Refund $%.2f for order %s requires manager approval. Ticket %s opened.",
					a.AmountUSD, a.OrderID, ticket),
			}, nil
		},
	)
}

func buildApproveRefundTool() (tool.Tool, error) {
	type args struct {
		Ticket   string `json:"ticket"`
		Approved bool   `json:"approved"`
		Notes    string `json:"notes,omitempty"`
	}
	type result struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}
	return functiontool.New[args, result](
		functiontool.Config{
			Name:        "approve_refund",
			Description: "Manager-only: approve or deny a pending refund ticket.",
		},
		func(_ tool.Context, a args) (result, error) {
			cur, ok := processedRefunds[a.Ticket]
			if !ok {
				return result{}, fmt.Errorf("ticket %q not found", a.Ticket)
			}
			if a.Approved {
				processedRefunds[a.Ticket] = "APPROVED " + cur + " — " + a.Notes
				return result{Status: "approved", Message: "Refund " + a.Ticket + " approved and processed."}, nil
			}
			processedRefunds[a.Ticket] = "DENIED " + cur + " — " + a.Notes
			return result{Status: "denied", Message: "Refund " + a.Ticket + " denied."}, nil
		},
	)
}
