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

// Package web prepares router dedicated to ADK REST API for http web server
package web

import (
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/server/restapi/handlers"
	"google.golang.org/adk/server/restapi/routers"
	"google.golang.org/adk/server/restapi/services"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// NewHandler creates and returns an http.Handler for the ADK REST API.
// This is the preferred way to integrate ADK REST API with any HTTP server.
// The returned handler can be registered with any standard Go HTTP server or router.
func NewHandler(config *adk.Config) http.Handler {
	adkExporter := services.NewAPIServerSpanExporter()
	telemetry.AddSpanProcessor(sdktrace.NewSimpleSpanProcessor(adkExporter))

	router := mux.NewRouter().StrictSlash(true)
	setupRouter(router,
		routers.NewSessionsAPIRouter(handlers.NewSessionsAPIController(config.SessionService)),
		routers.NewRuntimeAPIRouter(handlers.NewRuntimeAPIRouter(config.SessionService, config.AgentLoader, config.ArtifactService)),
		routers.NewAppsAPIRouter(handlers.NewAppsAPIController(config.AgentLoader)),
		routers.NewDebugAPIRouter(handlers.NewDebugAPIController(config.SessionService, config.AgentLoader, adkExporter)),
		routers.NewArtifactsAPIRouter(handlers.NewArtifactsAPIController(config.ArtifactService)),
		&routers.EvalAPIRouter{},
	)
	return router
}

// SetupRouter initiates mux.Router with ADK REST API routers.
//
// Deprecated: Use NewHandler instead. This function assumes the caller is using
// gorilla/mux and tightly couples the implementation to that router. NewHandler
// returns a standard http.Handler that can be used with any HTTP server or router.
func SetupRouter(router *mux.Router, routerConfig *adk.Config) *mux.Router {
	adkExporter := services.NewAPIServerSpanExporter()
	telemetry.AddSpanProcessor(sdktrace.NewSimpleSpanProcessor(adkExporter))
	return setupRouter(router,
		routers.NewSessionsAPIRouter(handlers.NewSessionsAPIController(routerConfig.SessionService)),
		routers.NewRuntimeAPIRouter(handlers.NewRuntimeAPIRouter(routerConfig.SessionService, routerConfig.AgentLoader, routerConfig.ArtifactService)),
		routers.NewAppsAPIRouter(handlers.NewAppsAPIController(routerConfig.AgentLoader)),
		routers.NewDebugAPIRouter(handlers.NewDebugAPIController(routerConfig.SessionService, routerConfig.AgentLoader, adkExporter)),
		routers.NewArtifactsAPIRouter(handlers.NewArtifactsAPIController(routerConfig.ArtifactService)),
		&routers.EvalAPIRouter{},
	)
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
