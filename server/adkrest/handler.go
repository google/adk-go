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

package adkrest

import (
	"net/http"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/server/adkrest/internal/routers"
	"google.golang.org/adk/server/adkrest/internal/services"
	"google.golang.org/adk/telemetry"
)

// NewHandler creates and returns an http.Handler for the ADK REST API.
func NewHandler(config *launcher.Config, sseWriteTimeout time.Duration) http.Handler {
	debugTelemetry := services.NewDebugTelemetry()
	config.TelemetryOptions = append(config.TelemetryOptions, telemetry.WithSpanProcessors(debugTelemetry.SpanProcessor()))
	config.TelemetryOptions = append(config.TelemetryOptions, telemetry.WithLogRecordProcessors(debugTelemetry.LogProcessor()))

	router := mux.NewRouter().StrictSlash(true)
	// TODO: Allow taking a prefix to allow customizing the path
	// where the ADK REST API will be served.
	setupRouter(router,
		routers.NewSessionsAPIRouter(controllers.NewSessionsAPIController(config.SessionService)),
		routers.NewRuntimeAPIRouter(controllers.NewRuntimeAPIController(config.SessionService, config.MemoryService, config.AgentLoader, config.ArtifactService, sseWriteTimeout, config.PluginConfig)),
		routers.NewAppsAPIRouter(controllers.NewAppsAPIController(config.AgentLoader)),
		routers.NewDebugAPIRouter(controllers.NewDebugAPIController(config.SessionService, config.AgentLoader, debugTelemetry)),
		routers.NewArtifactsAPIRouter(controllers.NewArtifactsAPIController(config.ArtifactService)),
		&routers.EvalAPIRouter{},
	)
	return router
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
