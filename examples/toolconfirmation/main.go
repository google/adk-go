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

// Package provides an example console app for tool confirmation.
// The example shows requestVacationDays func which requires user confirmation. It uses ctx.ToolConfirmation() and ctx.RequestConfirmation() to define the confimation process.
// This is the example for most advanced scenario which gives the full control how to define confirmation flow.
// For simpler setup use `RequireConfirmation` or `RequireConfirmationProvider` in [functiontool.Config].
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/toolconfirmation"

	"google.golang.org/genai"
)

// RequestVacationArgs defines the arguments for our long-running tool.
type RequestVacationArgs struct {
	Days   int    `json:"days"`
	UserID string `json:"user_id"`
}

type ConfirmationPayload struct {
	DaysApproved int `json:"days_approved"`
}

// RequestVacationResults defines the *initial* output of our long-running tool.
type RequestVacationResults struct {
	Status       string `json:"status"`
	DaysApproved int    `json:"days_approved"`
	RequestID    string `json:"request_id"`
}

type VacationRequest struct {
	ID           string
	UserID       string
	Days         int
	Status       string // PENDING, APPROVED, REJECTED
	CallID       string
	DaysApproved int
	Confirmation *toolconfirmation.ToolConfirmation
}

var (
	// Stores the requests with a reqID key
	requestsByReqID = make(map[string]*VacationRequest)
	// Stores the requests with a callID key
	requestsByCallID = make(map[string]*VacationRequest)
	requestCounter   = 0
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	vacationAgent, err := createRequestVacationDaysAgent(model)
	if err != nil {
		log.Fatalf("Failed to create vacation agent: %v", err)
	}

	sessionService := session.InMemoryService()
	session, err := sessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n--- Menu ---")
		fmt.Println("1: Chat with LLM")
		fmt.Println("2: Manage Vacation Requests")
		fmt.Println("3: Exit")
		fmt.Print("Choose an option: ")

		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			return
		}
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			runChatSession(ctx, vacationAgent, sessionService, reader, session.Session.ID())
		case "2":
			runVacationSession(ctx, vacationAgent, sessionService, reader, session.Session.ID())
		case "3":
			fmt.Println("Exiting.")
			return
		default:
			fmt.Println("Invalid option. Please try again.")
		}
	}
}

// requestVacationDays simulates the *initiation* of a long-running ticket creation task.
func requestVacationDays(ctx tool.Context, args RequestVacationArgs) (*RequestVacationResults, error) {
	log.Printf("TOOL_EXEC: 'requestVacationDays' called with days: %d for user %s (Call ID: %s)\n", args.Days, args.UserID, ctx.FunctionCallID())

	if args.Days <= 0 {
		return nil, fmt.Errorf("invalid days to request %d", args.Days)
	}

	confirmation := ctx.ToolConfirmation()
	if confirmation == nil {
		requestID := fmt.Sprintf("req-%d", requestCounter)
		requestCounter++

		req := &VacationRequest{
			ID:     requestID,
			UserID: args.UserID,
			Days:   args.Days,
			Status: "PENDING",
		}

		// Store the pending request
		requestsByReqID[requestID] = req
		requestsByCallID[ctx.FunctionCallID()] = req

		err := ctx.RequestConfirmation(
			"Please approve or reject the tool call request_time_off() by responding with a FunctionResponse with an expected ToolConfirmation payload.",
			ConfirmationPayload{
				DaysApproved: 0,
			})
		if err != nil {
			return nil, err
		}
		return &RequestVacationResults{
			Status:    "Manager approval is required.",
			RequestID: requestID,
		}, nil
	}

	// This part normally wouldn't be reached in the first call
	req, ok := requestsByCallID[ctx.FunctionCallID()]
	if !ok {
		return nil, fmt.Errorf("unable to get request using payload %s and function call id %s", confirmation.Payload, ctx.FunctionCallID())
	}
	req.Confirmation = confirmation
	if confirmation.Confirmed {
		jsonBytes, err := json.Marshal(confirmation.Payload)
		if err != nil {
			return nil, fmt.Errorf("error marshalling payload %s: %w", confirmation.Payload, err)
		}
		var payload ConfirmationPayload
		if err := json.Unmarshal(jsonBytes, &payload); err != nil {
			return nil, fmt.Errorf("error unmarshalling payload %s: %w", confirmation.Payload, err)
		}
		approvedDays := min(payload.DaysApproved, args.Days)
		req.Status = "APPROVED"
		req.DaysApproved = payload.DaysApproved
		requestsByReqID[req.ID] = req // Update status
		return &RequestVacationResults{
			Status:       "The time off request is accepted.",
			DaysApproved: approvedDays,
			RequestID:    req.ID,
		}, nil
	} else {
		req.Status = "REJECTED"
		requestsByReqID[req.ID] = req // Update status
		req.DaysApproved = 0
		return &RequestVacationResults{
			Status:       "The time off request is rejected.",
			DaysApproved: 0,
			RequestID:    req.ID,
		}, nil
	}
}

func createRequestVacationDaysAgent(model model.LLM) (agent.Agent, error) {
	vacationTool, err := functiontool.New(
		functiontool.Config{
			Name:        "request_vacation_days",
			Description: "Request vacation days for a user. Returns a request ID for tracking.",
		},
		requestVacationDays,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create vacation tool: %w", err)
	}

	return llmagent.New(llmagent.Config{
		Name:        "vacation_agent",
		Model:       model,
		Instruction: "You are a helpful assistant for requesting vacation days. When a user asks for time off, call the request_vacation_days tool, making sure to include the user's ID.",
		Tools:       []tool.Tool{vacationTool},
	})
}

const (
	userID  = "user" // Default user for interactions
	appName = "console_app"
)

// runTurn executes a single turn with the agent.
func runTurn(ctx context.Context, r *runner.Runner, sessionID string, content *genai.Content) {
	fmt.Printf("\n--- Sending to Agent ---\n")
	for event, err := range r.Run(ctx, userID, sessionID, content, agent.RunConfig{
		StreamingMode: agent.StreamingModeNone,
	}) {
		if err != nil {
			fmt.Printf("\nAGENT_ERROR: %v\n", err)
			continue
		}
		printEventSummary(event)
		if event.Content != nil {
			for _, part := range event.Content.Parts {
				fc := part.FunctionCall
				if fc != nil && fc.Name == toolconfirmation.RequestConfirmationFunctionCallName {
					originalFunctionCall, err := toolconfirmation.OriginalCallFrom(fc)
					if err != nil {
						continue
					}
					req, ok := requestsByCallID[originalFunctionCall.ID]
					if !ok {
						continue
					}
					fmt.Printf("Updating %s call id %s to %s\n", req.ID, req.CallID, fc.ID)
					req.CallID = fc.ID
				}
			}
		}
	}
}

func runChatSession(ctx context.Context, chatAgent agent.Agent, sessionService session.Service, reader *bufio.Reader, sessionID string) {
	fmt.Println("\n--- LLM Chat Mode ---")
	fmt.Println("Type 'back' to return to the main menu.")

	r, err := runner.New(runner.Config{AppName: appName, Agent: chatAgent, SessionService: sessionService})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	for {
		fmt.Print("You: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)

		if strings.ToLower(userInput) == "back" {
			break
		}

		if userInput != "" {
			userMessage := genai.NewContentFromText(userInput, genai.RoleUser)
			runTurn(ctx, r, sessionID, userMessage)
		}
	}
	fmt.Println("Exiting LLM Chat Mode.")
}

func runVacationSession(ctx context.Context, vacationAgent agent.Agent, sessionService session.Service, reader *bufio.Reader, sessionID string) {
	fmt.Println("\n--- Vacation Request Mode ---")
	fmt.Println("Type 'back' to return to the main menu.")
	fmt.Println("Commands: 'approve <ID>', 'reject <ID>'")
	displayVacationRequests()

	r, err := runner.New(runner.Config{AppName: appName, Agent: vacationAgent, SessionService: sessionService})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	for {
		fmt.Print("Vacation Command: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)
		inputLower := strings.ToLower(userInput)

		if inputLower == "back" {
			break
		}

		if strings.HasPrefix(inputLower, "approve ") {
			requestID := strings.TrimSpace(strings.TrimPrefix(inputLower, "approve "))
			processApproval(ctx, r, sessionID, requestID, true, reader)
		} else if strings.HasPrefix(inputLower, "reject ") {
			requestID := strings.TrimSpace(strings.TrimPrefix(inputLower, "reject "))
			processApproval(ctx, r, sessionID, requestID, false, reader)
		} else if userInput != "" {
			// Allow free text to interact with the vacation agent
			userMessage := genai.NewContentFromText(userInput, genai.RoleUser)
			runTurn(ctx, r, sessionID, userMessage)
		}
	}
	fmt.Println("Exiting Vacation Request Mode.")
}

func displayVacationRequests() {
	fmt.Println("\n--- Pending Vacation Requests ---")
	if len(requestsByReqID) == 0 {
		fmt.Println("No pending requests.")
		return
	}
	for _, req := range requestsByReqID {
		fmt.Printf("ID: %s, Call ID: %s, User: %s, Days: %d, Status: %s, Days Approved: %d\n", req.ID, req.CallID, req.UserID, req.Days, req.Status, req.DaysApproved)
	}
	fmt.Println("-------------------------------")
}

func processApproval(ctx context.Context, r *runner.Runner, sessionID, requestID string, approved bool, reader *bufio.Reader) {
	req, exists := requestsByReqID[requestID]
	if !exists || req.Status != "PENDING" {
		fmt.Printf("Request ID %s not found or not pending.\n", requestID)
		return
	}

	daysApproved := 0
	if approved {
		fmt.Printf("How many days to approve for %s (requested %d)? ", requestID, req.Days)
		daysInput, _ := reader.ReadString('\n')
		days, err := strconv.Atoi(strings.TrimSpace(daysInput))
		if err != nil || days < 0 || days > req.Days {
			fmt.Println("Invalid number of days. Approval cancelled.")
			return
		}
		daysApproved = days
		fmt.Printf("Approving %d days for request %s.\n", daysApproved, requestID)
	} else {
		fmt.Printf("Rejecting request %s.\n", requestID)
	}

	payload := ConfirmationPayload{DaysApproved: daysApproved}
	funcResponse := &genai.FunctionResponse{
		Name: toolconfirmation.RequestConfirmationFunctionCallName,
		ID:   req.CallID,
		Response: map[string]any{
			"confirmed": approved,
			"payload":   payload,
		},
	}

	appResponse := &genai.Content{
		Role:  string(genai.RoleUser), // Response comes from the app/user
		Parts: []*genai.Part{{FunctionResponse: funcResponse}},
	}
	runTurn(ctx, r, sessionID, appResponse)

	fmt.Println("Processing complete.")
	displayVacationRequests()
}

// printEventSummary provides a readable log of agent and LLM interactions.
func printEventSummary(event *session.Event) {
	if event.LLMResponse.Content != nil {
		for _, part := range event.LLMResponse.Content.Parts {
			author := event.Author
			if author == "" {
				author = "AGENT"
			}
			// Check for a text part.
			if part.Text != "" {
				fmt.Printf("[%s_TEXT]: %s\n", author, part.Text)
			}
			// Check for a function call part.
			if fc := part.FunctionCall; fc != nil {
				fmt.Printf("[%s_CALL]: %s(%v) Call ID: %s\n", author, fc.Name, fc.Args, fc.ID)
			}
			// Check for a function response part.
			if fr := part.FunctionResponse; fr != nil {
				fmt.Printf("[%s_RESPONSE]: %s(%v) Call ID: %s\n", author, fr.Name, fr.Response, fr.ID)
			}
		}
	}
}
