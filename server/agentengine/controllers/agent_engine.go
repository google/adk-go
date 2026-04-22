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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"google.golang.org/adk/server/agentengine/controllers/method"
	"google.golang.org/adk/server/agentengine/internal/models"
	"google.golang.org/adk/session"
)

// AgentEngineAPIController holds information about the supported methods
type AgentEngineAPIController struct {
	handlers map[string]method.MethodHandler
	service  session.Service
}

// NewAgentEngineAPIController creates a new AgentEngineAPIController. Verifies if registered methods are unique by name
func NewAgentEngineAPIController(service session.Service, handlers []method.MethodHandler) (*AgentEngineAPIController, error) {
	methodHandlers := map[string]method.MethodHandler{}
	for _, handler := range handlers {
		if _, ok := methodHandlers[handler.Name()]; ok {
			return nil, fmt.Errorf("duplicate method name: %v", handler.Name())
		}
		methodHandlers[handler.Name()] = handler
	}
	return &AgentEngineAPIController{service: service, handlers: methodHandlers}, nil
}

// Query provides a way to invoke all the methods
func (c *AgentEngineAPIController) Query(rw http.ResponseWriter, req *http.Request) {
	query := models.Query{}
	var payload []byte

	if req.ContentLength > 0 {
		var err error
		payload, err = io.ReadAll(req.Body)
		if err != nil {
			http.Error(rw, fmt.Errorf("ioutil.ReadAll failed: %v", err).Error(), http.StatusBadRequest)
			return
		}

		err = json.Unmarshal(payload, &query)
		if err != nil {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
	}

	err := c.handleQuery(req.Context(), rw, payload, query.ClassMethod)
	if err != nil {
		log.Printf("handleQuery failed: %v", err)
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
}

func (c *AgentEngineAPIController) handleQuery(context context.Context, rw http.ResponseWriter, payload []byte, classMethod string) error {
	handler, ok := c.handlers[classMethod]
	if !ok {
		return fmt.Errorf("unrecognized class method: %v", classMethod)
	}
	return handler.Handle(context, rw, payload)
}
