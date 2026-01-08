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

package adka2a

import (
	"net/http"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
)

// HandlerConfig configures the A2A HTTP handlers.
type HandlerConfig struct {
	// ExecutorConfig is the configuration for the underlying A2A executor.
	ExecutorConfig ExecutorConfig

	// AgentCard is the optional agent card to serve at /.well-known/agent-card.json.
	// If nil, no agent card endpoint will be registered when using NewServeMux.
	AgentCard *a2a.AgentCard
}

// NewInvocationHandler creates an http.Handler that serves A2A requests using JSON-RPC transport.
// This is the recommended way to expose an ADK agent via A2A for most use cases.
//
// The returned handler can be registered with any HTTP router:
//
//	http.Handle("/a2a/invoke", adka2a.NewInvocationHandler(config))
//
// For custom transport layers or advanced configurations, use NewRequestHandler instead.
func NewInvocationHandler(config HandlerConfig, opts ...a2asrv.RequestHandlerOption) http.Handler {
	executor := NewExecutor(config.ExecutorConfig)
	requestHandler := a2asrv.NewHandler(executor, opts...)
	return a2asrv.NewJSONRPCHandler(requestHandler)
}

// NewServeMux creates an http.ServeMux with complete A2A endpoints configured.
// This includes:
//   - POST /a2a/invoke - JSON-RPC endpoint for agent invocation
//   - GET /.well-known/agent-card.json - Agent card endpoint (if AgentCard is provided in config)
//
// This is the simplest way to set up an A2A server:
//
//	config := adka2a.HandlerConfig{
//	    ExecutorConfig: adka2a.ExecutorConfig{...},
//	    AgentCard:      &a2a.AgentCard{...},
//	}
//	http.ListenAndServe(":8080", adka2a.NewServeMux(config))
func NewServeMux(config HandlerConfig, opts ...a2asrv.RequestHandlerOption) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/a2a/invoke", NewInvocationHandler(config, opts...))
	if config.AgentCard != nil {
		mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(config.AgentCard))
	}
	return mux
}

// NewRequestHandler creates a transport-agnostic A2A request handler.
// Use this when you need to wrap the handler with a custom transport layer
// or when integrating with gRPC via a2agrpc.Handler.
//
// For most HTTP use cases, prefer NewInvocationHandler which returns a ready-to-use http.Handler.
//
// Example with custom transport:
//
//	requestHandler := adka2a.NewRequestHandler(config)
//	grpcHandler := a2agrpc.NewHandler(requestHandler)
func NewRequestHandler(config HandlerConfig, opts ...a2asrv.RequestHandlerOption) a2asrv.RequestHandler {
	executor := NewExecutor(config.ExecutorConfig)
	return a2asrv.NewHandler(executor, opts...)
}
