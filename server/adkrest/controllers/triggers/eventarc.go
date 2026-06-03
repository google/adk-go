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
	"io"
	"net/http"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
)

const eventArcDefaultUserID = "eventarc-caller"

// EventarcController handles the Eventarc trigger endpoints.
type EventarcController struct {
	runner    *RetriableRunner
	semaphore chan struct{}
}

// NewEventarcController creates a new EventarcController.
func NewEventarcController(sessionService session.Service, agentLoader agent.Loader, memoryService memory.Service, artifactService artifact.Service, pluginConfig runner.PluginConfig, triggerConfig TriggerConfig) *EventarcController {
	return &EventarcController{
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

// EventarcTriggerHandler handles the Eventarc trigger endpoint.
func (c *EventarcController) EventarcTriggerHandler(w http.ResponseWriter, r *http.Request) {
	var event models.EventarcTriggerRequest
	contentType := r.Header.Get("Content-Type")
	// The HTTP Content-Type header MUST be set to the media type of an event format for structured mode.
	// https://github.com/cloudevents/spec/blob/main/cloudevents/bindings/http-protocol-binding.md#321-http-content-type
	if contentType == "application/cloudevents+json" {
		// --- STRUCTURED MODE ---
		// The entire event is in the body. Decode it.
		// The payload (Storage or Pub/Sub) gets safely trapped in event.Data as bytes.
		if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
			respondError(w, http.StatusBadRequest, fmt.Sprintf("failed to unmarshal eventarc request: %v", err))
			return
		}
	} else {
		// --- BINARY MODE ---
		// Metadata is in the headers.
		event.ID = r.Header.Get("ce-id")
		event.Type = r.Header.Get("ce-type")
		event.Source = r.Header.Get("ce-source")
		event.SpecVersion = r.Header.Get("ce-specversion")
		event.Time = r.Header.Get("ce-time")

		// The entire body is the payload.
		// We just read it as raw bytes into event.Data.
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to read body: %v", err))
			return
		}
		event.Data = bodyBytes
	}

	var messageContent string

	// Handle Pub/Sub Specifically ---
	if event.Type == "google.cloud.pubsub.topic.v1.messagePublished" {
		var pubsub models.PubSubTriggerRequest
		var err error
		// Unmarshal the raw bytes into our specific Pub/Sub struct
		if err := json.Unmarshal(event.Data, &pubsub); err != nil {
			respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to unmarshal pubsub data: %v", err))
			return
		}
		messageContent, err = messageContentFromPubSub(pubsub)
		if err != nil {
			respondError(w, http.StatusBadRequest, fmt.Sprintf("failed to retrieve message content: %v", err))
			return
		}
	} else {
		// Otherwise just marshal the whole event as an input data.
		// E.g. as https://googleapis.github.io/google-cloudevents/examples/binary/storage/StorageObjectData-simple.json
		messageBytes, err := json.Marshal(event)
		if err != nil {
			respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to marshal agent message: %v", err))
			return
		}
		messageContent = string(messageBytes)
	}

	appName, err := appName(r)
	if err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to retrieve app name: %v", err))
		return
	}

	userID := event.Source
	if userID == "" {
		userID = eventArcDefaultUserID
	}

	// Semaphore limits concurrent agent calls based on the TriggerConfig.
	if c.semaphore != nil {
		c.semaphore <- struct{}{}
		defer func() { <-c.semaphore }()
	}

	if _, err := c.runner.RunAgent(r.Context(), appName, userID, messageContent); err != nil {
		respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to run agent: %v", err))
		return
	}

	respondSuccess(w)
}
