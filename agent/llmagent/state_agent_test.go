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
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
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

func assertSessionValues(
	t *testing.T,
	ctx context.Context,
	cctx agent.CallbackContext,
	title string,
	keysInCtxSession []string,
	keysInServiceSession []string,
	keysNotInServiceSession []string,
) {
	t.Helper()

	getRequest := &session.GetRequest{
		AppName:   cctx.AppName(),
		UserID:    cctx.UserID(),
		SessionID: cctx.SessionID(),
	}
	getResponse, err := testSessionService.Get(ctx, getRequest)
	if err != nil {
		t.Fatalf("[%s] Failed to get session from service: %v", title, err)
	}
	sessionInService := getResponse.Session

	for _, key := range keysInCtxSession {
		if _, err := cctx.State().Get(key); err != nil {
			t.Errorf("[%s] Key %s not found in context session state: %v", title, key, err)
		}
	}

	for _, key := range keysInServiceSession {
		if _, err := sessionInService.State().Get(key); err != nil {
			t.Errorf("[%s] Key %s not found in service session state: %v", title, key, err)
		}
	}

	for _, key := range keysNotInServiceSession {
		if val, err := sessionInService.State().Get(key); err == nil {
			t.Errorf("[%s] Key %s unexpectedly found in service session state with value: %v", title, key, val)
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
		assertSessionValues(t, cctx, cctx, "In before_agent_callback",
			[]string{"before_agent_callback_state_key"},
			[]string{},
			[]string{"before_agent_callback_state_key"})
		return nil, nil
	}
}

func beforeModelCallback(t *testing.T) func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
	return func(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error) {
		if err := cctx.State().Set("before_model_callback_state_key", "before_model_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, cctx, "In before_model_callback",
			[]string{"before_agent_callback_state_key", "before_model_callback_state_key"},
			[]string{"before_agent_callback_state_key"},
			[]string{"before_model_callback_state_key"})
		return nil, nil
	}
}

func afterModelCallback(t *testing.T) func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
	return func(cctx agent.CallbackContext, llmResponse *model.LLMResponse, err error) (*model.LLMResponse, error) {
		if err := cctx.State().Set("after_model_callback_state_key", "after_model_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, cctx, "In after_model_callback",
			[]string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key"},
			[]string{"before_agent_callback_state_key"},
			[]string{"before_model_callback_state_key", "after_model_callback_state_key"})
		return nil, nil
	}
}

func afterAgentCallback(t *testing.T) agent.AfterAgentCallback {
	return func(cctx agent.CallbackContext, event *session.Event, err error) (*genai.Content, error) {
		if err := cctx.State().Set("after_agent_callback_state_key", "after_agent_callback_state_value"); err != nil {
			return nil, fmt.Errorf("failed to set state: %w", err)
		}
		assertSessionValues(t, cctx, cctx, "In after_agent_callback",
			[]string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key", "after_agent_callback_state_key"},
			[]string{"before_agent_callback_state_key", "before_model_callback_state_key", "after_model_callback_state_key"},
			[]string{"after_agent_callback_state_key"})
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
