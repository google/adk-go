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

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
)

// AppsAPIController is the controller for the Apps API.
type AppsAPIController struct {
	agentLoader agent.Loader
}

// NewAppsAPIController creates a controller for Apps API.
func NewAppsAPIController(agentLoader agent.Loader) *AppsAPIController {
	return &AppsAPIController{agentLoader: agentLoader}
}

// ListAppsHandler handles listing all loaded agents.
func (c *AppsAPIController) ListAppsHandler(rw http.ResponseWriter, req *http.Request) {
	apps := c.agentLoader.ListAgents()
	EncodeJSONResponse(apps, http.StatusOK, rw)
}

// AppInfoHandler handles describing an app: its root agent and every agent
// reachable from it, with each agent's instruction, tools and children.
func (c *AppsAPIController) AppInfoHandler(rw http.ResponseWriter, req *http.Request) {
	appName := mux.Vars(req)["app_name"]
	if appName == "" {
		http.Error(rw, "app_name is required", http.StatusBadRequest)
		return
	}

	rootAgent, err := c.agentLoader.LoadAgent(appName)
	if err != nil {
		http.Error(rw, fmt.Sprintf("app not found: %s", appName), http.StatusNotFound)
		return
	}

	EncodeJSONResponse(services.GetAppInfo(req.Context(), appName, rootAgent), http.StatusOK, rw)
}
