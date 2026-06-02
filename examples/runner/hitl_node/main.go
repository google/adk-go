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

// Command hitl_node shows that a plain LlmAgent is automatically driven
// through the ADK 2.0 node runtime by the Runner, including the
// human-in-the-loop (HITL) pause/resume cycle. The user configures
// nothing special — just runner.Config{Agent: <llmAgent>}.
//
// The agent has a long-running tool ("request_approval"). When the model
// calls it, the LlmAgent emits LongRunningToolIDs; the Runner's node
// wrapper bridges that into a workflow pause and persists the run state.
// On the next turn we send the matching function response (the human's
// decision) and the agent resumes to its final answer.
//
// This uses a real Gemini model, like the other examples. Set credentials
// first (see examples/bidi/README.md):
//
//	export GOOGLE_API_KEY="your_api_key_here"
//	go run ./examples/runner/hitl_node/
package main

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const (
	appName   = "hitl_node_sample"
	userID    = "user-1"
	sessionID = "session-1"
)

// approvalArgs is the argument schema for the long-running tool.
type approvalArgs struct {
	Action string `json:"action"`
}

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-lite", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("gemini.NewModel: %v", err)
	}

	// A long-running tool: when the model calls it, the run pauses until a
	// human supplies the response on a later turn. The handler returns a
	// "pending" marker; the real decision arrives as the resume reply.
	requestApproval, err := functiontool.New(functiontool.Config{
		Name:          "request_approval",
		Description:   "Requests human approval for an action and waits for the decision.",
		IsLongRunning: true,
	}, func(_ agent.ToolContext, args approvalArgs) (map[string]string, error) {
		return map[string]string{"status": "pending", "action": args.Action}, nil
	})
	if err != nil {
		log.Fatalf("functiontool.New: %v", err)
	}

	// A plain LlmAgent with the long-running tool. Nothing here opts into
	// the node runtime — the Runner detects the LlmAgent and routes it
	// through the node path (with HITL) automatically.
	approver, err := llmagent.New(llmagent.Config{
		Name:        "approver",
		Model:       model,
		Description: "Performs actions only after human approval.",
		Instruction: "When the user asks to perform an action, call request_approval " +
			"with that action and wait. After approval, confirm in one short sentence.",
		Tools: []tool.Tool{requestApproval},
	})
	if err != nil {
		log.Fatalf("llmagent.New: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             approver, // <-- just an agent; no node/workflow config
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("runner.New: %v", err)
	}

	// --- Turn 1: the model calls the long-running tool and the run pauses.
	fmt.Println("=== Turn 1: ask the agent to perform an action ===")
	interruptID := ""
	for ev, err := range r.Run(ctx, userID, sessionID, userText("Please delete the production database."), agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("turn 1: %v", err)
		}
		if ev == nil {
			continue
		}
		printText(ev)
		if len(ev.LongRunningToolIDs) > 0 {
			interruptID = ev.LongRunningToolIDs[0]
		}
		if ev.RequestedInput != nil {
			fmt.Printf("[pause] awaiting human approval (interrupt %s)\n", ev.RequestedInput.InterruptID)
		}
	}
	if interruptID == "" {
		log.Fatal("the agent did not call the long-running tool; try rephrasing the prompt")
	}

	// --- Turn 2: the human approves; resume by sending the function response.
	fmt.Println("=== Turn 2: human approves; resume the agent ===")
	for ev, err := range r.Run(ctx, userID, sessionID, approvalReply(interruptID, "approved"), agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("turn 2: %v", err)
		}
		printText(ev)
	}
}

func userText(text string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
}

// approvalReply builds the resume message: a function response whose ID is
// the long-running tool call ID being answered.
func approvalReply(id, decision string) *genai.Content {
	return &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       id,
				Name:     "request_approval",
				Response: map[string]any{"decision": decision},
			},
		}},
	}
}

func printText(ev *session.Event) {
	if ev == nil || ev.LLMResponse.Content == nil {
		return
	}
	for _, p := range ev.LLMResponse.Content.Parts {
		if p.Text != "" {
			fmt.Printf("[%s] %s\n", ev.Author, p.Text)
		}
	}
}
