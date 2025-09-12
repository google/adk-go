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

package handlers

import (
	"encoding/json"
	"net/http"

	"google.golang.org/adk/cmd/restapi/services"
)

type AppsApiController struct {
	agentLoader services.AgentLoader
}

func NewAppsApiController(agentLoader services.AgentLoader) *AppsApiController {
	return &AppsApiController{agentLoader: agentLoader}
}

func (c *AppsApiController) ListApps(rw http.ResponseWriter, req *http.Request) {
	apps := c.agentLoader.ListAgents()
	rw.WriteHeader(http.StatusOK)
	rw.Header().Set("Content-Type", "application/json")
	json.NewEncoder(rw).Encode(apps)
}
