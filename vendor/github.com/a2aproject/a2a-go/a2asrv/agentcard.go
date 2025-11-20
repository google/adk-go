// Copyright 2025 The A2A Authors
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

package a2asrv

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

const WellKnownAgentCardPath = "/.well-known/agent-card.json"

// AgentCardProducer creates an AgentCard instances used for agent discovery and capability negotiation.
type AgentCardProducer interface {
	// Card returns a self-describing manifest for an agent. It provides essential
	// metadata including the agent's identity, capabilities, skills, supported
	// communication methods, and security requirements.
	Card(ctx context.Context) (*a2a.AgentCard, error)
}

// AgentCardProducerFn is a function type which implements [AgentCardProducer].
type AgentCardProducerFn func(ctx context.Context) (*a2a.AgentCard, error)

func (fn AgentCardProducerFn) Card(ctx context.Context) (*a2a.AgentCard, error) {
	return fn(ctx)
}

// NewStaticAgentCardHandler creates an [http.Handler] implementation for serving a public [a2a.AgentCard]
// which is not expected to change while the program is running.
// The method panics if the argument json marhsaling fails.
func NewStaticAgentCardHandler(card *a2a.AgentCard) http.Handler {
	bytes, err := json.Marshal(card)
	if err != nil {
		panic(err.Error())
	}
	return http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		ctx := withRequestContext(req)
		if req.Method == "OPTIONS" {
			writePublicCardHTTPOptions(rw)
			rw.WriteHeader(http.StatusOK)
			return
		}
		if req.Method != "GET" {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		writeAgentCardBytes(ctx, rw, bytes)
	})
}

// NewAgentCardHandler creates an [http.Handler] implementation for serving a public [a2a.AgentCard].
func NewAgentCardHandler(producer AgentCardProducer) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		ctx := withRequestContext(req)
		if req.Method == "OPTIONS" {
			writePublicCardHTTPOptions(rw)
			rw.WriteHeader(http.StatusOK)
			return
		}
		if req.Method != "GET" {
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		card, err := producer.Card(ctx)
		if err != nil {
			log.Error(ctx, "agent card producer failed", err)
			rw.WriteHeader(http.StatusInternalServerError)
			return
		}
		cardBytes, err := json.Marshal(card)
		if err != nil {
			log.Error(ctx, "agent card marshaling failed", err)
			rw.WriteHeader(http.StatusInternalServerError)
			return
		}
		writeAgentCardBytes(ctx, rw, cardBytes)
	})
}

func withRequestContext(req *http.Request) context.Context {
	logger := log.LoggerFrom(req.Context())
	withAttrs := logger.With(
		"method", req.Method,
		"host", req.Host,
		"remote_addr", req.RemoteAddr,
	)
	return log.WithLogger(req.Context(), withAttrs)
}

func writePublicCardHTTPOptions(rw http.ResponseWriter) {
	rw.Header().Set("Access-Control-Allow-Origin", "*")
	rw.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	rw.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	rw.Header().Set("Access-Control-Max-Age", "86400")
}

func writeAgentCardBytes(ctx context.Context, rw http.ResponseWriter, bytes []byte) {
	rw.Header().Set("Access-Control-Allow-Origin", "*")
	rw.Header().Set("Content-Type", "application/json")
	if _, err := rw.Write(bytes); err != nil {
		log.Error(ctx, "failed to write agent card response", err)
	}
}
