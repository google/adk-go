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

// Binary hitl is a runnable workflow agent that exercises the
// human-in-the-loop patterns the workflow engine supports today:
//
//   - conditional pause (auto-approve when amount <= threshold,
//     ask a human when amount > threshold);
//   - workflow input request with a typed payload (the operator
//     sees the expense report inline);
//   - re-entry resume on a single node (the asker re-runs after
//     the operator answers, reads the answer via ResumedInput,
//     and emits a routing event);
//   - handoff resume (the auto-path output flows straight into
//     the next node without a pause).
//
// Tool confirmation (functiontool's RequireConfirmation) is a
// related but distinct mechanism that lives inside the LLMAgent
// flow today; it is not yet wired through ToolNode in the
// workflow engine, so this sample does not use it. The console
// launcher's tool-confirmation render path is still exercised by
// any LLMAgent-driven sample.
//
// Run it with the console launcher and try a few inputs:
//
//	go run ./examples/workflow/hitl/ console
//
//	User -> 50 lunch with team
//	# → auto-approved, output flows downstream (handoff resume)
//
//	User -> 250 client dinner
//	# → '[HITL input] Manager approval required ...'
//	[user]: yes
//	# → '✓ Approved' (re-entry routes to the approved branch)
//
//	User -> 250 birthday cake
//	[user]: revise: cake is too expensive
//	# → '↺ Needs revision: cake is too expensive'
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"os"
	"strconv"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/workflow"
)

// expenseReport is the typed value flowing between nodes. Parsed
// from the user's free-form text on turn 1 by the parse node.
type expenseReport struct {
	AmountUSD   int    `json:"amountUsd"`
	Description string `json:"description"`
}

// expenseReportFromAny accepts both a typed expenseReport (the
// in-process happy path) and a map[string]any (the post-JSON-
// round-trip shape that any state-persisted any-typed value takes
// after Workflow.Resume reloads it from session.State). Returns
// the canonical typed value.
//
// Re-entry resume rehydrates a node's input from session.State,
// where every any field arrives as map[string]any after
// json.Unmarshal. Without this normaliser the re-entry code path
// would see a map and silently zero-out struct fields on type
// assertion.
func expenseReportFromAny(v any) expenseReport {
	switch t := v.(type) {
	case expenseReport:
		return t
	case map[string]any:
		var report expenseReport
		bytes, err := json.Marshal(t)
		if err != nil {
			return report
		}
		_ = json.Unmarshal(bytes, &report)
		return report
	default:
		return expenseReport{}
	}
}

// autoApprovalThreshold is the cutoff above which the workflow
// pauses for manager approval; at or below it the request is
// auto-approved without a prompt. Tunable via the
// HITL_AUTO_APPROVAL_THRESHOLD env var; defaults to 100.
func autoApprovalThreshold() int {
	if v := os.Getenv("HITL_AUTO_APPROVAL_THRESHOLD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return 100
}

// parseExpense turns the user's free-form text into a typed
// expenseReport. Format: "<amount> <description...>", e.g.
// "250 client dinner". Garbage in → 0/raw text, which then trips
// the auto-approval branch (amount 0 ≤ threshold).
func parseExpense(_ agent.InvocationContext, raw string) (expenseReport, error) {
	raw = strings.TrimSpace(raw)
	parts := strings.SplitN(raw, " ", 2)
	if len(parts) == 0 {
		return expenseReport{}, nil
	}
	amount, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		// First token wasn't a number; treat the whole input
		// as the description with amount 0 (auto-approve path).
		return expenseReport{Description: raw}, nil
	}
	description := ""
	if len(parts) == 2 {
		description = strings.TrimSpace(parts[1])
	}
	return expenseReport{AmountUSD: amount, Description: description}, nil
}

// evaluateRequestNode emits a routing event that splits the
// pipeline into the auto-approve and needs-review branches. No
// pause here — the conditional-pause behaviour is achieved by
// routing to the re-entry asker only when amount > threshold.
//
// Direct translation of the auto-approve guard from
// adk-python/contributing/workflow_samples/request_input_advanced/
// (without the pause), wired with workflow.Routes for explicit
// branching.
type evaluateRequestNode struct {
	workflow.BaseNode
	threshold int
}

func newEvaluateRequestNode(threshold int) *evaluateRequestNode {
	return &evaluateRequestNode{
		BaseNode:  workflow.NewBaseNode("evaluate_request", "routes the report to auto-approve or human-review", workflow.NodeConfig{}),
		threshold: threshold,
	}
}

func (n *evaluateRequestNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		report := expenseReportFromAny(input)

		ev := session.NewEvent(ctx.InvocationID())
		ev.Actions.StateDelta["output"] = report
		if report.AmountUSD <= n.threshold {
			ev.Routes = []string{"auto"}
		} else {
			ev.Routes = []string{"needs_review"}
		}
		yield(ev, nil)
	}
}

// reviewDecisionNode is the re-entry asker. On the first
// activation it issues a RequestInput keyed by "manager_approval"
// and exits without emitting an output, so the engine parks it
// in NodeWaiting. On the next turn the scheduler re-activates it
// with the operator's reply available via ctx.ResumedInput; the
// node then emits a routing event that dispatches to one of the
// three downstream branches.
//
// Direct translation of
// adk-python/contributing/workflow_samples/request_input_rerun/
// to the expense-approval domain. The original Python sample is
// the cleanest reference for the @node(rerun_on_resume=True)
// pattern: ask, exit, re-enter on resume, decide.
type reviewDecisionNode struct {
	workflow.BaseNode
}

func newReviewDecisionNode() *reviewDecisionNode {
	rerun := true
	return &reviewDecisionNode{
		BaseNode: workflow.NewBaseNode(
			"review_decision",
			"asks the manager for approval and routes by their reply",
			workflow.NodeConfig{RerunOnResume: &rerun},
		),
	}
}

func (n *reviewDecisionNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		report := expenseReportFromAny(input)

		response, isResume := ctx.ResumedInput("manager_approval")
		if !isResume {
			yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "manager_approval",
				Message:     fmt.Sprintf("Approve $%d for %q? Reply 'yes', 'no', or 'revise: <feedback>'.", report.AmountUSD, report.Description),
				Payload:     report,
			}), nil)
			return
		}

		reply := strings.TrimSpace(strings.ToLower(fmt.Sprint(response)))
		ev := session.NewEvent(ctx.InvocationID())
		switch {
		case reply == "yes" || reply == "y" || reply == "approve":
			ev.Routes = []string{"approved"}
			ev.Actions.StateDelta["output"] = report
		case reply == "no" || reply == "n" || reply == "reject":
			ev.Routes = []string{"rejected"}
			ev.Actions.StateDelta["output"] = report
		case strings.HasPrefix(reply, "revise:"):
			ev.Routes = []string{"revise"}
			ev.Actions.StateDelta["output"] = strings.TrimSpace(strings.TrimPrefix(reply, "revise:"))
		default:
			ev.Routes = []string{"revise"}
			ev.Actions.StateDelta["output"] = reply
		}
		yield(ev, nil)
	}
}

// fileDisbursement is a tool that pretends to push money out
// the door. Files the expense and emits a result string the
// approved-notification node renders to the operator.
type disburseArgs struct {
	AmountUSD   int    `json:"amountUsd"`
	Description string `json:"description"`
}

type disburseResult struct {
	Status string `json:"status"`
}

func newDisburseTool() (tool.Tool, error) {
	return functiontool.New[disburseArgs, disburseResult](
		functiontool.Config{
			Name:        "file_disbursement",
			Description: "files an expense disbursement.",
		},
		func(_ tool.Context, args disburseArgs) (disburseResult, error) {
			log.Printf("file_disbursement: disbursing $%d for %q", args.AmountUSD, args.Description)
			return disburseResult{Status: fmt.Sprintf("disbursed $%d for %q", args.AmountUSD, args.Description)}, nil
		},
	)
}

// renderResultText formats a node's value into the user-facing
// text that the console launcher prints as agent output.
func renderResultText(prefix string, value any) string {
	return fmt.Sprintf("%s: %v", prefix, value)
}

// notifyApproved, notifyRejected, notifyRevise are the three
// terminal nodes the route from review_decision lands on.
func notifyApproved(_ agent.InvocationContext, _ disburseResult) (string, error) {
	return "✓ Disbursement filed.", nil
}

func notifyRejected(_ agent.InvocationContext, in any) (string, error) {
	return renderResultText("✗ Rejected", expenseReportFromAny(in)), nil
}

func notifyRevise(_ agent.InvocationContext, feedback string) (string, error) {
	return renderResultText("↺ Needs revision", feedback), nil
}

func main() {
	ctx := context.Background()

	threshold := autoApprovalThreshold()

	parseNode := workflow.NewFunctionNode("parse_expense", parseExpense, workflow.NodeConfig{})
	evaluateNode := newEvaluateRequestNode(threshold)
	reviewNode := newReviewDecisionNode()

	disburseTool, err := newDisburseTool()
	if err != nil {
		log.Fatalf("failed to build disburse tool: %v", err)
	}
	disburseNode, err := workflow.NewToolNode(disburseTool, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("failed to build disburse tool node: %v", err)
	}
	approvedNode := workflow.NewFunctionNode("notify_approved", notifyApproved, workflow.NodeConfig{})
	rejectedNode := workflow.NewFunctionNode("notify_rejected", notifyRejected, workflow.NodeConfig{})
	reviseNode := workflow.NewFunctionNode("notify_revise", notifyRevise, workflow.NodeConfig{})

	// Graph:
	//
	//   START → parse_expense → evaluate_request
	//                              │ "auto"          │ "needs_review"
	//                              ↓                 ↓
	//                              │             review_decision (re-entry asker)
	//                              │                 ├─ "approved" ─┐
	//                              │                 ├─ "rejected" → notify_rejected
	//                              │                 └─ "revise"   → notify_revise
	//                              ↓                                │
	//                          file_disbursement ←───────────────────┘
	//                              ↓
	//                          notify_approved
	edges := workflow.Concat(
		workflow.Chain(workflow.Start, parseNode, evaluateNode),

		// evaluateNode's routing event splits the pipeline.
		// Both branches feed disburseNode directly: the
		// engine's typeutil.ConvertToWithJSONSchema in ToolNode
		// coerces expenseReport (typed) into disburseArgs (the
		// tool's input schema) using their matching JSON tags,
		// so no explicit adapter node is needed between them.
		[]workflow.Edge{
			{From: evaluateNode, To: disburseNode, Route: workflow.StringRoute("auto")},
			{From: evaluateNode, To: reviewNode, Route: workflow.StringRoute("needs_review")},
		},

		// reviewNode (re-entry asker) routes by manager reply.
		[]workflow.Edge{
			{From: reviewNode, To: disburseNode, Route: workflow.StringRoute("approved")},
			{From: reviewNode, To: rejectedNode, Route: workflow.StringRoute("rejected")},
			{From: reviewNode, To: reviseNode, Route: workflow.StringRoute("revise")},
		},

		// Common disbursement chain (auto path + manual approval).
		workflow.Chain(disburseNode, approvedNode),
	)

	hitlAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "expense_approval",
		Description: "approves and disburses small expenses, escalating large ones to a human reviewer",
		Edges:       edges,
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("expense approval workflow ready (auto-approve threshold: $%d)", threshold)
	log.Printf("try: '50 lunch'  (auto-approve)  /  '250 client dinner'  (will pause for approval)")

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(hitlAgent),
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
