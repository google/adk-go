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
	"encoding/json"
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
)

const pubSubDefaultUserID = "pubsub-caller"

// PubSubController handles the PubSub trigger endpoints.
type PubSubController struct {
	runner    *RetriableRunner
	semaphore chan struct{}
}

// NewPubSubController creates a new PubSubController.
// Deprecated: use [NewPubSubControllerWithConfig], which does not have to grow a
// parameter every time the controller gains a setting, and which surfaces a
// configuration error instead of discarding it. This one is kept because it is
// released API and every existing call site still compiles; it is a candidate
// for removal at the next major version.
func NewPubSubController(sessionService session.Service, agentLoader agent.Loader, memoryService memory.Service, artifactService artifact.Service, pluginConfig runner.PluginConfig, triggerConfig TriggerConfig) *PubSubController {
	// No compaction, so nothing that can be rejected. The error return exists
	// for the config form, which can be handed a configuration that cannot
	// serve the apps behind this controller.
	c, _ := NewPubSubControllerWithConfig(ControllerConfig{
		SessionService:  sessionService,
		AgentLoader:     agentLoader,
		MemoryService:   memoryService,
		ArtifactService: artifactService,
		PluginConfig:    pluginConfig,
		TriggerConfig:   triggerConfig,
	})
	return c
}

// NewPubSubControllerWithConfig creates the controller from a config struct.
//
// A separate constructor rather than a variadic parameter on the one above:
// adding a parameter would change that function's type, which breaks any caller
// holding it as a value even though ordinary call sites still compile, and it
// is released API.
func NewPubSubControllerWithConfig(cfg ControllerConfig) (*PubSubController, error) {
	retriable := newRetriableRunner(cfg)
	if err := retriable.validateCompaction(); err != nil {
		return nil, err
	}
	return &PubSubController{
		runner: retriable,
	}, nil
}

// PubSubTriggerHandler handles the PubSub trigger endpoint.
func (c *PubSubController) PubSubTriggerHandler(w http.ResponseWriter, r *http.Request) {
	// Parse the request to the request model.
	var req models.PubSubTriggerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, fmt.Sprintf("failed to decode request: %v", err))
		return
	}

	agentMessage, err := messageContentFromPubSub(req)
	if err != nil {
		respondError(w, http.StatusBadRequest, fmt.Sprintf("failed to retrieve message content: %v", err))
		return
	}

	appName, err := appName(r)
	if err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to retrieve app name: %v", err))
		return
	}

	userID := req.Subscription
	if userID == "" {
		userID = pubSubDefaultUserID
	}

	// Semaphore limits concurrent agent calls based on the TriggerConfig.
	if c.semaphore != nil {
		c.semaphore <- struct{}{}
		defer func() { <-c.semaphore }()
	}

	if _, err := c.runner.RunAgent(r.Context(), appName, userID, agentMessage); err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to run agent: %v", err))
		return
	}

	respondSuccess(w)
}

func messageContentFromPubSub(req models.PubSubTriggerRequest) (string, error) {
	messageContent := make(map[string]any)
	if len(req.Message.Data) > 0 {
		messageContent["data"] = string(req.Message.Data)
	}
	// Add attributes to the messageContent if present
	if len(req.Message.Attributes) > 0 {
		messageContent["attributes"] = req.Message.Attributes
	}

	if len(messageContent) == 0 {
		return "", fmt.Errorf("empty message data and attributes")
	}

	agentMessage, err := json.Marshal(messageContent)
	if err != nil {
		return "", fmt.Errorf("failed to marshal agent message: %w", err)
	}
	return string(agentMessage), nil
}
