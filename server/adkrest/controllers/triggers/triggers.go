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

package triggers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
)

type RetriableRunner struct {
	sessionService  session.Service
	agentLoader     agent.Loader
	memoryService   memory.Service
	artifactService artifact.Service
	pluginConfig    runner.PluginConfig
	triggerConfig   TriggerConfig
}

func (r *RetriableRunner) RunAgent(ctx context.Context, appName, userID, messageContent string) ([]*session.Event, error) {
	// Each retry = new session
	sessReq := &session.CreateRequest{
		AppName: appName,
		UserID:  userID,
	}
	sessResp, err := r.sessionService.Create(ctx, sessReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	userMessage := genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: messageContent},
		},
	}

	curAgent, err := r.agentLoader.LoadAgent(appName)
	if err != nil {
		return nil, fmt.Errorf("failed to load agent: %v", err)
	}

	runR, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           curAgent,
		SessionService:  r.sessionService,
		MemoryService:   r.memoryService,
		ArtifactService: r.artifactService,
		PluginConfig:    r.pluginConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create runner: %v", err)
	}

	return r.runAgentWithRetry(ctx, runR, sessResp.Session.UserID(), sessResp.Session.ID(), &userMessage)
}

// runAgentWithRetry uses exponential backoff with jitter to handle 429 rate-limit errors.
// After MaxRetries is exhausted, raises an error to signal the upstream service (Pub/Sub, Eventarc) to retry at a higher level.
func (r *RetriableRunner) runAgentWithRetry(ctx context.Context, runR *runner.Runner, userID, sessionID string, userMessage *genai.Content) ([]*session.Event, error) {
	var runErr error
	events := []*session.Event{}
	for i := 0; i <= r.triggerConfig.MaxRetries; i++ {
		resp := runR.Run(ctx, userID, sessionID, userMessage, agent.RunConfig{StreamingMode: agent.StreamingModeNone})

		isThrottled := false
		for event, err := range resp {
			if err != nil {
				runErr = err
				if isResourceExhausted(err) {
					isThrottled = true
				}
				break
			}
			events = append(events, event)
		}

		if !isThrottled && runErr == nil {
			return events, nil // Success
		}

		if i < r.triggerConfig.MaxRetries && isThrottled {
			delay := calculateBackoff(i, r.triggerConfig.BaseDelay, r.triggerConfig.MaxDelay)
			time.Sleep(delay)
			runErr = nil // Clear error for next attempt
			continue
		}
		break // Not throttled (but error raised) or max retries reached
	}
	return nil, runErr
}

func respondError(w http.ResponseWriter, code int, msg string) {
	resp := models.TriggerResponse{Status: msg}
	controllers.EncodeJSONResponse(resp, code, w)
}

func respondSuccess(w http.ResponseWriter) {
	resp := models.TriggerResponse{Status: "success"}
	controllers.EncodeJSONResponse(resp, http.StatusOK, w)
}

// Check if an exception represents a transient rate-limit error.
func isResourceExhausted(err error) bool {
	return err != nil && (strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "ResourceExhausted"))
}

func calculateBackoff(attempt int, base, maxDelay time.Duration) time.Duration {
	backoff := float64(base) * math.Pow(2, float64(attempt))
	delay := min(time.Duration(backoff), maxDelay)
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.5)
	return delay + jitter
}

// Resolve the target app name from the request.
func appName(r *http.Request) (string, error) {
	vars := mux.Vars(r)
	appName := vars["app_name"]
	if appName == "" {
		return "", fmt.Errorf("no application name provided")
	}
	return appName, nil
}
