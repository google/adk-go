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

// refund_approval is a HITL refund workflow. Small refunds auto-approve;
// large refunds pause for a manager's decision via RequestInput, then
// resume on the next Runner.Run with the manager's FunctionResponse.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

type RefundReq struct {
	OrderID  string
	Amount   float64
	Reason   string
	Customer string
}

type Decision struct {
	Approved bool
	Notes    string
}

const (
	managerInterruptID = "manager_approval"
	autoApproveLimit   = 50.0
)

// approvalNode handles both auto-approval (small refund) and HITL
// (large refund). On the large-refund path it emits a RequestInput on
// the first invocation and returns the manager's verdict on resume.
type approvalNode struct {
	workflow.Base
}

func (a *approvalNode) RunImpl(ctx *workflow.NodeContext, in any, em workflow.EventEmitter) error {
	req, ok := in.(RefundReq)
	if !ok {
		return fmt.Errorf("approval: input type %T, want RefundReq", in)
	}
	if req.Amount <= autoApproveLimit {
		return em.Output(Decision{
			Approved: true,
			Notes:    fmt.Sprintf("auto-approved (under $%.0f threshold)", autoApproveLimit),
		})
	}
	if v, ok := ctx.ResumeInput(managerInterruptID); ok {
		// Resume path: the manager's response arrived via FunctionResponse.
		if mp, isMap := v.(map[string]any); isMap {
			approved, _ := mp["approved"].(bool)
			notes, _ := mp["notes"].(string)
			return em.Output(Decision{Approved: approved, Notes: notes})
		}
		return errors.New("manager response shape unexpected")
	}
	return em.RequestInput(workflow.RequestInput{
		Prompt: fmt.Sprintf(
			"Refund of $%.2f for order %s (%s, customer=%s). Approve and add a one-line note.",
			req.Amount, req.OrderID, req.Reason, req.Customer),
		InterruptID: managerInterruptID,
	})
}

func main() {
	loadRequest := workflow.Func("load_request",
		func(ctx *workflow.NodeContext, _ any) (RefundReq, error) {
			v, err := ctx.Session().State().Get("refund_req")
			if err != nil {
				return RefundReq{}, fmt.Errorf("session state missing refund_req: %w", err)
			}
			req, ok := v.(RefundReq)
			if !ok {
				return RefundReq{}, fmt.Errorf("refund_req has type %T, want RefundReq", v)
			}
			return req, nil
		})

	validate := workflow.Func("validate",
		func(_ *workflow.NodeContext, req RefundReq) (RefundReq, error) {
			if req.Amount <= 0 {
				return RefundReq{}, errors.New("refund amount must be positive")
			}
			if req.OrderID == "" {
				return RefundReq{}, errors.New("missing order id")
			}
			return req, nil
		})

	approval := &approvalNode{}
	if err := approval.SetMetadata("approval", "Auto-approve small or pause for manager", workflow.NodeSpec{}); err != nil {
		log.Fatal(err)
	}

	processRefund := workflow.Func("process_refund",
		func(_ *workflow.NodeContext, d Decision) (string, error) {
			if !d.Approved {
				return "REFUND DENIED: " + d.Notes, nil
			}
			return "REFUND PROCESSED: " + d.Notes, nil
		})

	wf, err := workflow.New(workflow.Config{
		Name: "refund",
		Edges: []workflow.Edge{
			workflow.Connect(workflow.START, loadRequest),
			workflow.Connect(loadRequest, validate),
			workflow.Connect(validate, approval),
			workflow.Connect(approval, processRefund),
		},
	})
	if err != nil {
		log.Fatalf("workflow.New: %v", err)
	}

	wfAgent, _ := wf.AsAgent()
	r, _ := runner.New(runner.Config{
		AppName:           "refund_demo",
		Agent:             wfAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})

	fmt.Println("=== scenario 1: small refund ($23.50) — auto-approves ===")
	runRefund(r, "session-small", RefundReq{
		OrderID: "ORD-101", Amount: 23.50, Reason: "duplicate charge", Customer: "alice",
	}, nil)

	fmt.Println("\n=== scenario 2: large refund ($420.00) — pauses ===")
	runRefund(r, "session-large", RefundReq{
		OrderID: "ORD-202", Amount: 420.00, Reason: "lost shipment", Customer: "bob",
	}, nil)

	fmt.Println("\n--- pretend the manager reviewed the prompt and approved ---")

	resumeMsg := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       managerInterruptID,
				Name:     "adk_request_input",
				Response: map[string]any{"approved": true, "notes": "carrier confirmed lost; reissue."},
			},
		}},
	}
	fmt.Println("\n=== scenario 2 resume: manager approved ===")
	runRefund(r, "session-large", RefundReq{}, resumeMsg)
}

// runRefund drives one Run invocation against the runner. The typed
// RefundReq is plumbed via session state (WithStateDelta) so the
// load_request node can read it on the first turn. On a resume call
// msg carries the manager's FunctionResponse.
func runRefund(r *runner.Runner, sessID string, req RefundReq, msg *genai.Content) {
	ctx := context.Background()
	opts := []runner.RunOption{}
	if msg == nil {
		msg = &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
			Text: fmt.Sprintf("refund order=%s amount=%.2f reason=%q customer=%s",
				req.OrderID, req.Amount, req.Reason, req.Customer),
		}}}
		opts = append(opts, runner.WithStateDelta(map[string]any{"refund_req": req}))
	}
	for ev, err := range r.Run(ctx, "ops", sessID, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			fmt.Printf("  error: %v\n", err)
			return
		}
		if len(ev.LongRunningToolIDs) > 0 {
			fmt.Printf("  PAUSED awaiting manager input (interrupt id=%s)\n", ev.LongRunningToolIDs[0])
		}
		if ev.Author == "process_refund" && ev.Actions.NodeInfo != nil && ev.Actions.NodeInfo.Output != nil {
			fmt.Printf("  %v\n", ev.Actions.NodeInfo.Output)
		}
	}
}
