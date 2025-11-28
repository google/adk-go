// Copyright 2025 Google LLC
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

package llmagent_test

import (
	"context"
	"fmt"
	"iter"
	"maps"
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// FakeLLM is a mock implementation of model.LLM for testing.
type FakeLLM struct {
	GenerateContentFunc func(ctx context.Context, req *model.LLMRequest, stream bool) (model.LLMResponse, error)
}

func (f *FakeLLM) Name() string {
	return "fake-llm"
}

func (f *FakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if f.GenerateContentFunc != nil {
			resp, err := f.GenerateContentFunc(ctx, req, stream)
			yield(&resp, err)
		} else {
			// Default response
			yield(&model.LLMResponse{
				Content: genai.NewContentFromText("fake model response", genai.RoleModel),
			}, nil)
		}
	}
}

var testSessionService session.Service

type assertSessionParams struct {
	title                   string
	keysInCtxSession        []string
	keysInServiceSession    []string
	keysNotInServiceSession []string
}

func assertSessionValues(
	t *testing.T,
	cctx agent.CallbackContext,
	params *assertSessionParams,
) {
	t.Helper()

	getRequest := &session.GetRequest{
		AppName:   cctx.AppName(),
		UserID:    cctx.UserID(),
		SessionID: cctx.SessionID(),
	}
	getResponse, err := testSessionService.Get(cctx, getRequest)
	if err != nil {
		t.Fatalf("[%s] Failed to get session from service: %v", params.title, err)
	}
	sessionInService := getResponse.Session

	for _, key := range params.keysInCtxSession {
		if _, err := cctx.State().Get(key); err != nil {
			t.Errorf("[%s] Key %s not found in context session state: %v", params.title, key, err)
		}
	}

	for _, key := range params.keysInServiceSession {
		if _, err := sessionInService.State().Get(key); err != nil {
			t.Errorf("[%s] Key %s not found in service session state: %v", params.title, key, err)
		}
	}

	for _, key := range params.keysNotInServiceSession {
		if val, err := sessionInService.State().Get(key); err == nil {
			t.Errorf("[%s] Key %s unexpectedly found in service session state with value: %v", params.title, key, val)
		}
	}
}

// --- Callbacks (Modified to use *testing.T) ---
func beforeAgentCallback(t *testing.T) agent.BeforeAgentCallback {
	return func(cctx agent.CallbackContext) (*genai.Content, error) {
		if _, err := cctx.State().Get("before_agent_callback_state_key"); err == nil {
			return genai.NewContentFromText("Sorry, I can only reply once.", genai.RoleModel), nil
		}
		if err := cctx.State().Set("before_agent_callback_state_key", "before_agent_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, &assertSessionParams{
			title:                   "In before_agent_callback",
			keysInCtxSession:        []string{"before_agent_callback_state_key"},
			keysInServiceSession:    []string{},
			keysNotInServiceSession: []string{"before_agent_callback_state_key"},
		},
		)
		return nil, nil
	}
}

func beforeModelCallback(t *testing.T) func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
	return func(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
		if err := cctx.State().Set("before_model_callback_state_key", "before_model_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, &assertSessionParams{
			title:                   "In before_model_callback",
			keysInCtxSession:        []string{"before_agent_callback_state_key", "before_model_callback_state_key"},
			keysInServiceSession:    []string{"before_agent_callback_state_key"},
			keysNotInServiceSession: []string{"before_model_callback_state_key"},
		},
		)
		return nil, nil
	}
}

func afterModelCallback(t *testing.T) func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	return func(cctx agent.CallbackContext, llmResponse *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err := cctx.State().Set("after_model_callback_state_key", "after_model_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, &assertSessionParams{
			title:                   "In after_model_callback",
			keysInCtxSession:        []string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key"},
			keysInServiceSession:    []string{"before_agent_callback_state_key"},
			keysNotInServiceSession: []string{"before_model_callback_state_key", "after_model_callback_state_key"},
		},
		)
		return nil, nil
	}
}

func afterAgentCallback(t *testing.T) agent.AfterAgentCallback {
	return func(cctx agent.CallbackContext) (*genai.Content, error) {
		if err := cctx.State().Set("after_agent_callback_state_key", "after_agent_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, &assertSessionParams{
			title:                   "In after_agent_callback",
			keysInCtxSession:        []string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key", "after_agent_callback_state_key"},
			keysInServiceSession:    []string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key"},
			keysNotInServiceSession: []string{"after_agent_callback_state_key"},
		},
		)
		return nil, nil
	}
}

func TestAgentSessionLifecycle(t *testing.T) {
	ctx := context.Background()
	testSessionService = session.InMemoryService()

	// Setup Fake LLM
	fakeLLM := &FakeLLM{
		GenerateContentFunc: func(ctx context.Context, req *model.LLMRequest, stream bool) (model.LLMResponse, error) {
			return model.LLMResponse{
				Content: genai.NewContentFromText("test model response", genai.RoleModel),
			}, nil
		},
	}

	// Define Agent
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:                 "root_agent",
		Description:          "a verification agent.",
		Instruction:          "Test instruction",
		Model:                fakeLLM,
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{beforeAgentCallback(t)},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{beforeModelCallback(t)},
		AfterModelCallbacks:  []llmagent.AfterModelCallback{afterModelCallback(t)},
		AfterAgentCallbacks:  []agent.AfterAgentCallback{afterAgentCallback(t)},
	})
	if err != nil {
		t.Fatalf("Failed to create agent: %v", err)
	}

	// Setup Runner
	// Note: This Runner setup is a simplified guess. Actual implementation might need more services.
	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          rootAgent,
		SessionService: testSessionService,
	})
	if err != nil {
		t.Fatalf("Failed to create runner: %v", err)
	}

	// Create a session
	createReq := &session.CreateRequest{AppName: "test_app", UserID: "test_user"}
	createResp, err := testSessionService.Create(ctx, createReq)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	sessionID := createResp.Session.ID()

	// Run the agent
	userContent := genai.NewContentFromText("Hello agent", genai.RoleUser)

	eventStream := r.Run(ctx, "test_user", sessionID, userContent, agent.RunConfig{})

	// Iterate through events to trigger agent execution
	for _, err := range eventStream {
		if err != nil {
			t.Fatalf("Error during agent run: %v", err)
		}
	}

	// Final check of persisted state
	finalSession, _ := testSessionService.Get(ctx, &session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: sessionID})
	finalState := finalSession.Session.State()
	expectedKeys := []string{
		"before_agent_callback_state_key",
		"before_model_callback_state_key",
		"after_model_callback_state_key",
		"after_agent_callback_state_key",
	}
	for _, key := range expectedKeys {
		if _, err := finalState.Get(key); err != nil {
			t.Errorf("Key %s not found in final session state: %v", key, err)
		}
	}
}

// --- Tool Implementations ---

type WeatherArgs struct {
	Location string `json:"location"`
}

type WeatherResult struct {
	Location    string    `json:"location"`
	Temperature int       `json:"temperature"`
	Condition   string    `json:"condition"`
	Humidity    int       `json:"humidity"`
	Timestamp   time.Time `json:"timestamp"`
}

func GetWeather(ctx tool.Context, args WeatherArgs) (WeatherResult, error) {
	// Simulate weather data
	temperatures := []int{-10, -5, 0, 5, 10, 15, 20, 25, 30, 35}
	conditions := []string{"sunny", "cloudy", "rainy", "snowy", "windy"}

	return WeatherResult{
		Location:    args.Location,
		Temperature: temperatures[rand.Intn(len(temperatures))],
		Condition:   conditions[rand.Intn(len(conditions))],
		Humidity:    rand.Intn(61) + 30, // 30-90
		Timestamp:   time.Now(),
	}, nil
}

type CalculationArgs struct {
	Operation string  `json:"operation"`
	X         float64 `json:"x"`
	Y         float64 `json:"y"`
}

type CalculationResult struct {
	Operation string    `json:"operation"`
	X         float64   `json:"x"`
	Y         float64   `json:"y"`
	Result    any       `json:"result"`
	Timestamp time.Time `json:"timestamp"`
}

func Calculate(ctx tool.Context, args CalculationArgs) (CalculationResult, error) {
	operations := map[string]float64{
		"add":      args.X + args.Y,
		"subtract": args.X - args.Y,
		"multiply": args.X * args.Y,
	}
	if args.Operation == "divide" {
		if args.Y != 0 {
			operations["divide"] = args.X / args.Y
		} else {
			operations["divide"] = math.Inf(int(args.X))
		}
	}

	result, ok := operations[strings.ToLower(args.Operation)]
	if !ok {
		return CalculationResult{
			Operation: args.Operation,
			X:         args.X,
			Y:         args.Y,
			Result:    "Unknown operation",
			Timestamp: time.Now(),
		}, nil
	}

	return CalculationResult{
		Operation: args.Operation,
		X:         args.X,
		Y:         args.Y,
		Result:    result,
		Timestamp: time.Now(),
	}, nil
}

type LogActivityParams struct {
	Message string `json:"message"`
}

type LogEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message"`
}

type LogActivityResult struct {
	Status       string   `json:"status"`
	Entry        LogEntry `json:"entry"`
	TotalEntries int      `json:"total_entries"`
	err          error
}

func LogActivity(ctx tool.Context, params LogActivityParams) (LogActivityResult, error) {
	var activityLog []LogEntry
	val, err := ctx.State().Get("activity_log")
	if err == nil {
		activityLog, _ = val.([]LogEntry)
	}

	logEntry := LogEntry{Timestamp: time.Now(), Message: params.Message}
	activityLog = append(activityLog, logEntry)
	if err := ctx.State().Set("activity_log", activityLog); err != nil {
		return LogActivityResult{
			err: err,
		}, err
	}

	return LogActivityResult{
		Status:       "logged",
		Entry:        logEntry,
		TotalEntries: len(activityLog),
		err:          nil,
	}, nil
}

// --- Before Tool Callbacks ---

func beforeToolAuditCallback(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	fmt.Printf("🔍 AUDIT: About to call tool '%s' with args: %v\n", t.Name(), args)

	var auditLog []map[string]any
	val, err := ctx.State().Get("audit_log")
	if err == nil {
		auditLog, _ = val.([]map[string]any)
	}

	auditLog = append(auditLog, map[string]any{
		"type":      "before_call",
		"tool_name": t.Name(),
		"args":      args,
		"timestamp": time.Now(),
	})
	if err := ctx.State().Set("audit_log", auditLog); err != nil {
		return nil, err
	}
	return nil, nil // Continue execution
}

func beforeToolSecurityCallback(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if t.Name() == "get_weather" {
		location := ""
		if loc, ok := args["location"].(string); ok {
			location = loc
		}
		restricted := []string{"classified", "secret"}
		for _, r := range restricted {
			if strings.ToLower(location) == r {
				fmt.Printf("🚫 SECURITY: Blocked weather request for restricted location: %s\n", location)
				if err := ctx.State().Set("security_log", "example"); err != nil {
					return nil, err
				}
				return map[string]any{
					"error":              "Access denied",
					"reason":             "Location access is restricted",
					"requested_location": location,
				}, nil // Block execution
			}
		}
	}
	return nil, nil // Continue execution
}

func beforeToolValidationCallback(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if t.Name() == "calculate" {
		operation, _ := args["operation"].(string)
		y, yOK := args["y"].(float64)
		if strings.ToLower(operation) == "divide" && yOK && y == 0 {
			fmt.Println("🚫 VALIDATION: Prevented division by zero")
			return map[string]any{
				"error":     "Division by zero",
				"operation": operation,
				"x":         args["x"],
				"y":         args["y"],
			}, nil // Block execution
		}
	}
	return nil, nil // Continue execution
}

// --- After Tool Callbacks ---

func afterToolEnhancementCallback(ctx tool.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return result, err // Don't enhance if there was an error
	}
	fmt.Printf("✨ ENHANCE: Adding metadata to response from '%s'\n", t.Name())
	enhancedResponse := make(map[string]any)
	maps.Copy(enhancedResponse, result)
	enhancedResponse["enhanced"] = true
	enhancedResponse["enhancement_timestamp"] = time.Now()
	enhancedResponse["tool_name"] = t.Name()
	enhancedResponse["execution_context"] = "live_streaming"
	return enhancedResponse, nil
}

func afterToolAsyncCallback(ctx tool.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return result, err
	}
	fmt.Printf("🔄 ASYNC AFTER: Post-processing response from '%s'\n", t.Name())
	processedResponse := make(map[string]any)
	maps.Copy(processedResponse, result)
	processedResponse["async_processed"] = true
	processedResponse["processor"] = "async_after_callback"
	return processedResponse, nil
}

// --- Test Function ---

// --- Helper function to collect tool results ---
func collectToolResults(t *testing.T, stream iter.Seq2[*session.Event, error]) []map[string]any {
	t.Helper()
	var results []map[string]any
	for event, err := range stream {
		if err != nil {
			t.Fatalf("Error iterating through event stream: %v", err)
		}
		if event == nil || event.Content == nil {
			continue
		}

		for _, part := range event.Content.Parts {
			if part.FunctionResponse != nil {
				if part.FunctionResponse.Response != nil {
					results = append(results, part.FunctionResponse.Response)
				}
			}
		}
	}
	return results
}

func TestToolCallbacksAgent(t *testing.T) {
	// Fake LLM to control tool calls
	ctx := t.Context()
	service := session.InMemoryService()

	fakeLLM := &FakeLLM{
		GenerateContentFunc: func(ctx context.Context, req *model.LLMRequest, stream bool) (model.LLMResponse, error) {
			var userText string
			if len(req.Contents) == 1 && len(req.Contents[0].Parts) > 0 {
				userText = string(req.Contents[0].Parts[0].Text)
			} else if len(req.Contents) > 1 {
				userText = "after func"
			}

			var name string
			var args map[string]any
			switch userText {
			case "weather in London":
				name, args = "get_weather", map[string]any{"location": "London"}
			case "weather in secret":
				name, args = "get_weather", map[string]any{"location": "secret"}
			case "calculate 5 plus 3":
				name, args = "calculate", map[string]any{"operation": "add", "x": 5.0, "y": 3.0}
			case "calculate 5 divide by 0":
				name, args = "calculate", map[string]any{"operation": "divide", "x": 5.0, "y": 0.0}
			case "log this message":
				name, args = "log_activity", map[string]any{"message": "test log"}
			case "after func":
				return model.LLMResponse{
					Content: genai.NewContentFromText("Function Ended", genai.RoleModel),
				}, nil
			default:
				return model.LLMResponse{
					Content: genai.NewContentFromText("I'm not sure how to respond to that.", genai.RoleModel),
				}, nil
			}

			return model.LLMResponse{
				Content: genai.NewContentFromFunctionCall(name, args, genai.RoleModel),
			}, nil
		},
	}

	// Create tools
	getWeatherTool, _ := functiontool.New(functiontool.Config{Name: "get_weather", Description: "Get weather information"}, GetWeather)
	calculateTool, _ := functiontool.New(functiontool.Config{Name: "calculate", Description: "Perform mathematical calculations"}, Calculate)
	logActivityTool, _ := functiontool.New(functiontool.Config{Name: "log_activity", Description: "Log an activity message"}, LogActivity)

	agentConfig := llmagent.Config{
		Name:        "tool_callbacks_agent",
		Description: "Agent to test tool callbacks",
		Model:       fakeLLM,
		Instruction: "Follow user instructions to call tools.",
		Tools:       []tool.Tool{getWeatherTool, calculateTool, logActivityTool},
		BeforeToolCallbacks: []llmagent.BeforeToolCallback{
			beforeToolAuditCallback,
			beforeToolSecurityCallback,
			beforeToolValidationCallback,
		},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			afterToolEnhancementCallback,
			afterToolAsyncCallback,
		},
	}
	rootAgent, err := llmagent.New(agentConfig)
	if err != nil {
		t.Fatalf("Failed to create LLM Agent: %v", err)
	}

	// Setup Runner
	// Note: This Runner setup is a simplified guess. Actual implementation might need more services.
	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          rootAgent,
		SessionService: service,
	})
	if err != nil {
		t.Fatalf("Failed to create runner: %v", err)
	}

	tests := []struct {
		name            string
		query           string
		wantContent     []string // Substrings to check in the final response
		dontWantContent []string
		wantStateKeys   []string
	}{
		{
			name:            "Weather London - success",
			query:           "weather in London",
			wantContent:     []string{"London", "temperature", "enhanced:true"},
			dontWantContent: []string{"async_processed"},
		},
		{
			name:          "Weather Secret - blocked",
			query:         "weather in secret",
			wantContent:   []string{"Access denied", "Location access is restricted", "enhanced:true"}, // Callbacks still run on the error result
			wantStateKeys: []string{"security_log"},
		},
		{
			name:        "Calculate Add - success",
			query:       "calculate 5 plus 3",
			wantContent: []string{"operation:add", "result:8", "enhanced:true"},
		},
		{
			name:        "Calculate Divide by Zero - blocked",
			query:       "calculate 5 divide by 0",
			wantContent: []string{"Division by zero", "enhanced:true"}, // Callbacks still run
		},
		{
			name:          "Log Activity - success",
			query:         "log this message",
			wantContent:   []string{"status:logged", "total_entries:1", "enhanced:true"},
			wantStateKeys: []string{"activity_log"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Create a session
			createReq := &session.CreateRequest{AppName: "test_app", UserID: "test_user"}
			createResp, err := service.Create(ctx, createReq)
			if err != nil {
				t.Fatalf("Failed to create session: %v", err)
			}
			sessionID := createResp.Session.ID()

			userContent := genai.NewContentFromText(tc.query, genai.RoleUser) // Session ID based on test name
			eventStream := r.Run(ctx, "test_user", sessionID, userContent, agent.RunConfig{})

			toolResults := collectToolResults(t, eventStream)
			if err != nil {
				t.Fatalf("Agent run failed: %v", err)
			}

			if len(toolResults) == 0 {
				t.Fatalf("Expected tool results, got none")
			}
			lastResult := toolResults[len(toolResults)-1]

			// Check for expected content in the string representation of the result
			resultStr := fmt.Sprintf("%v", lastResult)
			for _, want := range tc.wantContent {
				if !strings.Contains(resultStr, want) {
					t.Errorf("Expected content %q not found in tool result: %s", want, resultStr)
				}
			}
			for _, dontWant := range tc.dontWantContent {
				if strings.Contains(resultStr, dontWant) {
					t.Errorf("Unexpected content %q found in tool result: %s", dontWant, resultStr)
				}
			}

			// Check state for log activity
			if len(tc.wantStateKeys) > 0 {
				currentSession, err := service.Get(context.Background(), &session.GetRequest{
					AppName:   "test_app",
					UserID:    "test_user",
					SessionID: sessionID,
				})
				if err != nil {
					t.Fatalf("Failed to get session: %v", err)
				}
				for _, key := range tc.wantStateKeys {
					if _, err := currentSession.Session.State().Get(key); err != nil {
						t.Errorf("Expected key %q not found in session state", key)
					}
				}
			}
		})
	}
}
