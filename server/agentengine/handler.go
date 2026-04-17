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

// package agentengine brings functionality of serving commands for AgentEngine-deployed code
package agentengine

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/agentengine/controllers"
	"google.golang.org/adk/server/agentengine/controllers/method"
	"google.golang.org/adk/server/agentengine/internal/routers"
)

// NewHandler creates and returns an http.Handler for the AgentEngine API.
func NewHandler(config *launcher.Config, sseWriteTimeout time.Duration) (http.Handler, error) {
	router := mux.NewRouter().StrictSlash(true)

	reasonginEngineController, err := controllers.NewReasoningEngineAPIController(config.SessionService, []method.MethodHandler{
		method.NewCreateSessionHandler(config.SessionService),
		method.NewListSessionHandler(config.SessionService),
		method.NewGetSessionHandler(config.SessionService),
		method.NewDeleteSessionHandler(config.SessionService),
		method.NewStreamQueryHandler(config),
		method.NewAsyncStreamQueryHandler(config),
	})
	if err != nil {
		return nil, fmt.Errorf("controllers.NewReasoningEngineAPIController failed: %v", err)
	}

	setupRouter(router,
		routers.NewReasoningEngineAPIRouter(reasonginEngineController),
	)
	return router, nil
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
