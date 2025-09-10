package handlers

import (
	"encoding/json"
	"net/http"

	"google.golang.org/adk/web/services"
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
