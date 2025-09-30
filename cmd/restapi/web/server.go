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

// package web prepeares router dedicated to ADK REST API for http web server
package web

import (
	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
	"google.golang.org/adk/cmd/restapi/services"
	"google.golang.org/adk/internal/telemetry"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// SetupRouter initiates mux.Router with ADK REST API routers
func SetupRouter(router *mux.Router, routerConfig *config.ADKAPIRouterConfigs) *mux.Router {
	adkExporter := services.NewAPIServerSpanExporter()
	telemetry.AddSpanProcessor(sdktrace.NewSimpleSpanProcessor(adkExporter))
	return setupRouter(router,
		routers.NewSessionsAPIRouter(handlers.NewSessionsAPIController(routerConfig.SessionService)),
		routers.NewRuntimeAPIRouter(handlers.NewRuntimeAPIRouter(routerConfig.SessionService, routerConfig.AgentLoader, routerConfig.ArtifactService)),
		routers.NewAppsAPIRouter(handlers.NewAppsAPIController(routerConfig.AgentLoader)),
		routers.NewDebugAPIRouter(handlers.NewDebugAPIController(routerConfig.SessionService, routerConfig.AgentLoader, adkExporter)),
		routers.NewArtifactsAPIRouter(handlers.NewArtifactsAPIController(routerConfig.ArtifactService)),
	)
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
