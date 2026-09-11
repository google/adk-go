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

// Package adkrest provides an HTTP server for the ADK REST API.
package adkrest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/compactionvalidate"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/controllers"
	"google.golang.org/adk/v2/server/adkrest/internal/routers"
	"google.golang.org/adk/v2/server/adkrest/internal/services"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/server/authz"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// validateCompactionAgainstAgents reports whether the compaction config can
// actually serve every app this server knows about.
func validateCompactionAgainstAgents(cfg ServerConfig) error {
	return compactionvalidate.AgainstAgents(cfg.Compaction, cfg.AgentLoader, runner.Config{
		SessionService:  cfg.SessionService,
		MemoryService:   cfg.MemoryService,
		ArtifactService: cfg.ArtifactService,
		PluginConfig:    cfg.PluginConfig,
	})
}

// NewServer creates a new ADK REST API server which implements [http.Handler] interface.
func NewServer(cfg ServerConfig) (*Server, error) {
	// Validated here rather than left to the first request. A compaction config
	// is rejected inside runner.New, which this server calls per request, so an
	// invalid one would otherwise start cleanly and then fail every request
	// with a 500 that names nothing the operator can act on.
	//
	// Against the agents, not just the shape. Validate() only checks the config
	// on its own, and the failure operators actually hit is a config with no
	// Summarizer over a root agent that is not an LLM agent, which is perfectly
	// well-shaped and 500s every request. Building a runner is the same code
	// path the request takes, so this cannot drift from it.
	if err := validateCompactionAgainstAgents(cfg); err != nil {
		return nil, err
	}

	debugTelemetry, err := services.NewDebugTelemetryWithConfig(&services.DebugTelemetryConfig{
		TraceCapacity: cfg.DebugConfig.TraceCapacity,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create debug telemetry service: %w", err)
	}

	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/health", healthHandler).Methods(http.MethodGet, http.MethodHead)
	// TODO: Allow taking a prefix to allow customizing the path
	// where the ADK REST API will be served.

	authorizer := cfg.Authorizer
	if authorizer == nil {
		authorizer = authz.NewNoop()
	}

	sessionsController := controllers.NewSessionsAPIController(cfg.SessionService)
	sessionsController.WithAuthorizer(authorizer)

	artifactsController := controllers.NewArtifactsAPIController(cfg.ArtifactService)
	artifactsController.WithAuthorizer(authorizer)

	subrouters := []routers.Router{
		routers.NewSessionsAPIRouter(sessionsController),
		routers.NewRuntimeAPIRouter(controllers.NewRuntimeAPIControllerWithConfig(controllers.RuntimeAPIControllerConfig{
			SessionService:  cfg.SessionService,
			MemoryService:   cfg.MemoryService,
			AgentLoader:     cfg.AgentLoader,
			ArtifactService: cfg.ArtifactService,
			SSETimeout:      cfg.SSEWriteTimeout,
			PluginConfig:    cfg.PluginConfig,
			Compaction:      cfg.Compaction,
			Authorizer:      authorizer,
		})),
		routers.NewAppsAPIRouter(controllers.NewAppsAPIController(cfg.AgentLoader)),
		routers.NewArtifactsAPIRouter(artifactsController),
		routers.NewVersionAPIRouter(controllers.NewVersionAPIController()),
		routers.NewAgentGraphAPIRouter(controllers.NewAgentGraphAPIController(cfg.AgentLoader)),
		&routers.TestsAPIRouter{},
		&routers.EvalAPIRouter{},
	}
	if cfg.DebugAPIConfig.IncludeDebugAPI {
		debugController := controllers.NewDebugAPIController(cfg.SessionService, cfg.AgentLoader, debugTelemetry)
		debugController.WithAuthorizer(authorizer)
		subrouters = append(subrouters, routers.NewDebugAPIRouter(debugController))
	}

	authenticator := cfg.Authenticator
	if authenticator == nil {
		authenticator = authn.NewNoop()
	}

	setupRouter(router, authenticator, subrouters...)
	srv := &Server{
		router:         router,
		telemetryStore: debugTelemetry,
	}

	return srv, nil
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// ServerConfig contains parameters for the ADK REST API server.
type ServerConfig struct {
	SessionService  session.Service
	MemoryService   memory.Service
	AgentLoader     agent.Loader
	ArtifactService artifact.Service
	SSEWriteTimeout time.Duration
	PluginConfig    runner.PluginConfig
	DebugConfig     DebugTelemetryConfig
	DebugAPIConfig  DebugAPIConfig

	// Authenticator authenticates inbound requests to every endpoint except the
	// public ones (/health and /version): a request without valid credentials
	// is answered 401, and an authenticated request carries the caller's
	// identity on its context. See the [authn] package for the built-in
	// providers.
	//
	// When nil it defaults to [authn.Noop], which authenticates every request as
	// an empty caller's identity — leaving every endpoint reachable without credentials.
	// Setting an Authenticator without also setting an Authorizer authenticates
	// callers but lets any authenticated caller act as any user; pair it with an
	// Authorizer to restrict that.
	Authenticator authn.Authenticator

	// Authorizer decides whether the authenticated caller's identity may act as the user
	// named in a request path (the {user_id} segment of the sessions, artifacts,
	// runtime and debug routes); a failure is answered 403.
	//
	// When nil it defaults to [authz.Noop], which permits any identity to act as
	// any user. Use [authz.Strict] to require the authenticated UserID to match
	// the path — but only alongside an Authenticator that sets a non-empty
	// UserID, since [authn.Noop]'s empty caller's identity would then be denied for every
	// user.
	Authorizer authz.Authorizer

	// Compaction enables context compaction for the sessions the
	// runners created here drive, replacing older events with summaries. Nil,
	// the default, disables compaction.
	//
	// The sliding window reduces prompt size by a constant factor rather than
	// bounding it. Only tail retention bounds growth, and it only fires when
	// more events accumulate between sliding-window compactions than
	// EventRetentionSize holds back, so a short interval with a large retention
	// size leaves it idle. See [compaction.Config].
	//
	// This setting is server-wide. One server can serve many applications
	// through its agent loader, and they all get this config or none of them
	// do, including the same Summarizer instance and so the same model. If
	// different applications need different compaction, or must not share a
	// summarizer, run them on separate servers.
	Compaction *compaction.Config
}

// DebugAPIConfig contains parameters for the debug API.
type DebugAPIConfig struct {
	// Controls if [routers.NewDebugAPIRouter] is included
	// WARNING: do not use debug api on PROD environment
	IncludeDebugAPI bool
}

// DebugTelemetryConfig contains parameters for the debug telemetry.
type DebugTelemetryConfig struct {
	// Maximum number of traces to keep in memory.
	// If <= 0, the default capacity 10_000 is used.
	TraceCapacity int
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

func setupRouter(router *mux.Router, authenticator authn.Authenticator, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, authenticator, subrouters...)
	return router
}
