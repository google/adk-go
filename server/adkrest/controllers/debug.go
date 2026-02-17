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

package controllers

import (
	"fmt"
	"net/http"
	"slices"

	"github.com/gorilla/mux"
	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/server/adkrest/internal/services"
	"google.golang.org/adk/session"
)

// DebugAPIController is the controller for the Debug API.
type DebugAPIController struct {
	sessionService session.Service
	agentloader    agent.Loader
	debugTelemetry *services.DebugTelemetry
}

// NewDebugAPIController creates the controller for the Debug API.
func NewDebugAPIController(sessionService session.Service, agentLoader agent.Loader, spansExporter *services.DebugTelemetry) *DebugAPIController {
	return &DebugAPIController{
		sessionService: sessionService,
		agentloader:    agentLoader,
		debugTelemetry: spansExporter,
	}
}

// EventSpanHandler returns the debug span for the event.
func (c *DebugAPIController) EventSpanHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	eventID := params["event_id"]
	if eventID == "" {
		http.Error(rw, "event_id parameter is required", http.StatusBadRequest)
		return
	}
	spans := c.debugTelemetry.GetSpansByEventID(eventID)
	key := string(semconv.GenAIOperationNameKey)
	// Return only generate content and execute tool spans.
	wantedOperations := []string{"execute_tool", "generate_content"}
	for _, span := range spans {
		opName := span.Attributes[key]
		if slices.Contains(wantedOperations, opName) {
			// Return the first span that matches the wanted operations - single event should contain only a single generate content or execute tool span.
			EncodeJSONResponse(span, http.StatusOK, rw)
			return
		}
	}
	http.Error(rw, fmt.Sprintf("event not found: %s", eventID), http.StatusNotFound)
}

// SessionSpansHandler returns the debug spans for the session.
func (c *DebugAPIController) SessionSpansHandler(rw http.ResponseWriter, req *http.Request) {
	params := mux.Vars(req)
	sessionID := params["session_id"]
	if sessionID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	spans := c.debugTelemetry.GetSpansBySessionID(sessionID)
	result := SessionTelemetry{
		SchemaVersion: 2,
		Spans:         spans,
	}
	EncodeJSONResponse(result, http.StatusOK, rw)
}

type SessionTelemetry struct {
	SchemaVersion int                  `json:"schema_version"`
	Spans         []services.DebugSpan `json:"spans"`
}

// EventGraphHandler returns the debug information for the session and session events in form of graph.
func (c *DebugAPIController) EventGraphHandler(rw http.ResponseWriter, req *http.Request) {
	vars := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(vars)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	resp, err := c.sessionService.Get(req.Context(), &session.GetRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
	})
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	eventID := vars["event_id"]
	if eventID == "" {
		http.Error(rw, "event_id parameter is required", http.StatusBadRequest)
		return
	}

	var event *session.Event
	for it := range resp.Session.Events().All() {
		if it.ID == eventID {
			event = it
			break
		}
	}

	if event == nil {
		http.Error(rw, "event not found", http.StatusNotFound)
		return
	}

	highlightedPairs := [][]string{}
	fc := functionalCalls(event)
	fr := functionalResponses(event)

	if len(fc) > 0 {
		for _, f := range fc {
			if f.Name != "" {
				highlightedPairs = append(highlightedPairs, []string{f.Name, event.Author})
			}
		}
	} else if len(fr) > 0 {
		for _, f := range fr {
			if f.Name != "" {
				highlightedPairs = append(highlightedPairs, []string{f.Name, event.Author})
			}
		}
	} else {
		highlightedPairs = append(highlightedPairs, []string{event.Author, event.Author})
	}

	agent, err := c.agentloader.LoadAgent(sessionID.AppName)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	graph, err := services.GetAgentGraph(req.Context(), agent, highlightedPairs)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(map[string]string{"dotSrc": graph}, http.StatusOK, rw)
}

func functionalCalls(event *session.Event) []*genai.FunctionCall {
	if event.LLMResponse.Content == nil || event.LLMResponse.Content.Parts == nil {
		return nil
	}
	fc := []*genai.FunctionCall{}
	for _, part := range event.LLMResponse.Content.Parts {
		if part.FunctionCall != nil {
			fc = append(fc, part.FunctionCall)
		}
	}
	return fc
}

func functionalResponses(event *session.Event) []*genai.FunctionResponse {
	if event.LLMResponse.Content == nil || event.LLMResponse.Content.Parts == nil {
		return nil
	}
	fr := []*genai.FunctionResponse{}
	for _, part := range event.LLMResponse.Content.Parts {
		if part.FunctionResponse != nil {
			fr = append(fr, part.FunctionResponse)
		}
	}
	return fr
}
