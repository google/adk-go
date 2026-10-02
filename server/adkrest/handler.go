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
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/trace"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/originguard"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/server/adkrest/internal/routers"
	"google.golang.org/adk/server/adkrest/internal/services"
	"google.golang.org/adk/session"
)

// NewServer creates a new ADK REST API server which implements [http.Handler] interface.
func NewServer(cfg ServerConfig) (*Server, error) {
	debugTelemetry, err := services.NewDebugTelemetryWithConfig(&services.DebugTelemetryConfig{
		TraceCapacity: cfg.DebugConfig.TraceCapacity,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create debug telemetry service: %w", err)
	}

	policy := originguard.New(originguard.Config{
		BindHost:       cfg.BindHost,
		AllowedOrigins: cfg.AllowedOrigins,
	})

	router := mux.NewRouter().StrictSlash(true)

	// Apply request-body size limit to mitigate memory-exhaustion DoS before
	// any routes (including /health) are registered. A MaxPayloadSize of 0 or
	// less selects defaultMaxPayloadSize.
	router.Use(MaxBytesMiddleware(cfg.MaxPayloadSize))

	router.HandleFunc("/health", healthHandler).Methods(http.MethodGet, http.MethodHead)
	// TODO: Allow taking a prefix to allow customizing the path
	// where the ADK REST API will be served.
	setupRouter(router,
		routers.NewSessionsAPIRouter(controllers.NewSessionsAPIController(cfg.SessionService)),
		routers.NewRuntimeAPIRouter(controllers.NewRuntimeAPIController(cfg.SessionService, cfg.MemoryService, cfg.AgentLoader, cfg.ArtifactService, cfg.SSEWriteTimeout, cfg.PluginConfig, false)),
		routers.NewAppsAPIRouter(controllers.NewAppsAPIController(cfg.AgentLoader)),
		routers.NewDebugAPIRouter(controllers.NewDebugAPIController(cfg.SessionService, cfg.AgentLoader, debugTelemetry)),
		routers.NewArtifactsAPIRouter(controllers.NewArtifactsAPIController(cfg.ArtifactService)),
		routers.NewVersionAPIRouter(controllers.NewVersionAPIController()),
		routers.NewAgentGraphAPIRouter(controllers.NewAgentGraphAPIController(cfg.AgentLoader)),
		&routers.TestsAPIRouter{},
		&routers.EvalAPIRouter{},
	)
	return &Server{
		router:         router,
		handler:        policy.Middleware(router),
		telemetryStore: debugTelemetry,
	}, nil
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

	// AllowedOrigins lists the web origins allowed to call this server from a
	// browser, as scheme://host[:port]; a bare host or host:port is read as
	// http. A request whose Origin is not listed is served only when that
	// Origin is the request's own, so leaving this empty permits same-origin
	// browsers and nothing else.
	//
	// One same-origin case is still refused: an Origin that is not a loopback
	// address, on a server only this machine can reach. Such a server serves
	// loopback pages, so a page claiming to be somewhere else got here by
	// pointing its own DNS name at us. This is what stops a page reaching the
	// server, and the WebSocket in particular, by DNS rebinding.
	//
	// Only on such a server. Where the request arrives over a network the
	// reasoning does not hold, so neither this nor BindHost's check refuses a
	// rebound page. That includes inside a container, where the connection
	// arrives on a routable interface whatever address the port was published
	// on.
	//
	// Listing an origin also vouches for its host, which BindHost's check
	// consults. That check runs on requests with no Origin too, so on a server
	// with a loopback BindHost this field decides those as well.
	//
	// A server behind a reverse proxy on the same machine needs the proxy's
	// origin listed on both counts: it is not loopback, and the proxy puts its
	// own hostname in Host.
	//
	// A single "*" entry turns every check here off, and BindHost's with them.
	// It says the server is meant to be reachable from anywhere, so do not set
	// it on a server reachable from an untrusted network: this API is
	// unauthenticated, and its endpoints read and drive whole agent sessions.
	//
	// The /run_live WebSocket upgrade also applies gorilla/websocket's default
	// origin check after this one, so a cross-origin page gets 403 there even
	// when its origin is listed or the entry is "*".
	AllowedOrigins []string

	// BindHost is the address the caller will bind this server to.
	//
	// Naming a loopback address refuses any request whose Host is neither
	// loopback nor the host of an AllowedOrigins entry. That is the only way to
	// catch a rebound page's same-origin GET, on which a browser sends no
	// Origin at all.
	//
	// Naming one routable address says the server is exposed on purpose, and
	// turns off both that check and the loopback-Origin rule above: a server
	// reachable over the network is legitimately reachable under whatever name
	// resolves to it.
	//
	// A wildcard address ("", ":8080", "0.0.0.0", "[::]") names every
	// interface, so it says neither. There, and when this is left empty, the
	// address the connection was accepted on stands in for the loopback-Origin
	// rule, and the Host check stays off. Naming a loopback address is what
	// buys anything over saying nothing: the accepted address cannot tell a
	// browser on this machine from a sidecar proxy or an nginx proxy_pass to
	// 127.0.0.1.
	BindHost string

	// MaxPayloadSize limits request body size in bytes. If <= 0,
	// defaultMaxPayloadSize is used.
	MaxPayloadSize int64
}

// DebugTelemetryConfig contains parameters for the debug telemetry.
type DebugTelemetryConfig struct {
	// Maximum number of traces to keep in memory.
	// If <= 0, the default capacity 10_000 is used.
	TraceCapacity int
}

// Server is an HTTP server that serves the ADK REST API.
type Server struct {
	// router is the route table. It is what the server routes with; requests
	// reach it only through handler.
	router *mux.Router
	// handler is the route table behind the origin and Host checks built from
	// [ServerConfig.AllowedOrigins] and [ServerConfig.BindHost]. This is what
	// [Server.ServeHTTP] serves, so no request skips those checks.
	handler        http.Handler
	telemetryStore *services.DebugTelemetry
}

// ServeHTTP makes [Server] implement [http.Handler] interface.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
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
