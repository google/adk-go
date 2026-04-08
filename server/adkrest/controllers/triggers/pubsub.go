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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
)

const defaultUserID = "pubsub-caller"

// PubSubController handles the PubSub trigger endpoints.
type PubSubController struct {
	runner    *RetriableRunner
	semaphore chan struct{}
}

// NewPubSubController creates a new PubSubController.
func NewPubSubController(sessionService session.Service, agentLoader agent.Loader, memoryService memory.Service, artifactService artifact.Service, pluginConfig runner.PluginConfig, triggerConfig TriggerConfig) *PubSubController {
	return &PubSubController{
		runner: &RetriableRunner{
			sessionService:  sessionService,
			agentLoader:     agentLoader,
			memoryService:   memoryService,
			artifactService: artifactService,
			pluginConfig:    pluginConfig,
			triggerConfig:   triggerConfig,
		},
		semaphore: make(chan struct{}, triggerConfig.MaxConcurrentRuns),
	}
}

// PubSubTriggerHandler handles the PubSub trigger endpoint.
func (c *PubSubController) PubSubTriggerHandler(w http.ResponseWriter, r *http.Request) {
	if c.semaphore != nil {
		c.semaphore <- struct{}{}
		defer func() { <-c.semaphore }()
	}

	// Parse the request to the request model.
	var req models.PubSubTriggerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, fmt.Sprintf("failed to decode request: %v", err))
		return
	}

	// Decode base64 message data.
	messageContent := make(map[string]any)
	if len(req.Message.Data) > 0 {
		// Avoids encoding the data twice later with json.Marshal.
		messageContent["data"] = string(req.Message.Data)
	}
	// Add attributes to the messageContent if present
	if len(req.Message.Attributes) > 0 {
		messageContent["attributes"] = req.Message.Attributes
	}

	if len(messageContent) == 0 {
		respondError(w, http.StatusBadRequest, "empty message data and attributes")
		return
	}

	agentMessage, err := json.Marshal(messageContent)
	if err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to marshal agent message: %v", err))
		return
	}

	appName, err := appName(r)
	if err != nil {
		respondError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if _, err := c.runner.RunAgent(r.Context(), appName, req.Subscription, string(agentMessage)); err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to run agent: %v", err))
		return
	}

	respondSuccess(w)
}
