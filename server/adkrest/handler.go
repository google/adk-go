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
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/server/adkrest/internal/routers"
	"google.golang.org/adk/server/adkrest/internal/services"
	"google.golang.org/adk/session"
)

// NewServer creates a new ADK REST API server which implements [http.Handler] interface.
func NewServer(cfg ServerConfig) (*Server, error) {
	debugTelemetry := services.NewDebugTelemetry()

	router := mux.NewRouter().StrictSlash(true)
	// TODO: Allow taking a prefix to allow customizing the path
	// where the ADK REST API will be served.
	setupRouter(router,
		routers.NewSessionsAPIRouter(controllers.NewSessionsAPIController(cfg.SessionService)),
		routers.NewRuntimeAPIRouter(controllers.NewRuntimeAPIController(cfg.SessionService, cfg.MemoryService, cfg.AgentLoader, cfg.ArtifactService, cfg.SSEWriteTimeout, cfg.PluginConfig)),
		routers.NewAppsAPIRouter(controllers.NewAppsAPIController(cfg.AgentLoader)),
		routers.NewDebugAPIRouter(controllers.NewDebugAPIController(cfg.SessionService, cfg.AgentLoader, debugTelemetry)),
		routers.NewArtifactsAPIRouter(controllers.NewArtifactsAPIController(cfg.ArtifactService)),
		&routers.EvalAPIRouter{},
	)
	return &Server{
		router:         router,
		telemetryStore: debugTelemetry,
	}, nil
}

// ServerConfig contains parameters for the ADK REST API server.
type ServerConfig struct {
	SessionService  session.Service
	MemoryService   memory.Service
	AgentLoader     agent.Loader
	ArtifactService artifact.Service
	SSEWriteTimeout time.Duration
	PluginConfig    runner.PluginConfig
}

// Server is an HTTP server that serves the ADK REST API.
type Server struct {
	router         *mux.Router
	telemetryStore *services.DebugTelemetry
}

// ServeHTTP makes [Server] implement [http.Handler] interface.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}

// SpanProcessor returns a processor that captures spans used for /debug/trace endpoint of the ADK REST API server.
// You can register it in your application TracerProvider to populate it with these spans.
func (s *Server) SpanProcessor() trace.SpanProcessor {
	return s.telemetryStore.SpanProcessor()
}

// LogProcessor returns a processor that captures log records used for /debug/trace endpoint of the ADK REST API server.
// You can register it in your application LoggerProvider to populate it with these logs.
func (s *Server) LogProcessor() sdklog.Processor {
	return s.telemetryStore.LogProcessor()
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
