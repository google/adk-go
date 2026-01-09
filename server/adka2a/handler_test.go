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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func TestNewInvocationHandler(t *testing.T) {
	agent, err := newEventReplayAgent(nil, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v", err)
	}

	config := HandlerConfig{
		ExecutorConfig: ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: session.InMemoryService(),
			},
		},
	}

	handler := NewInvocationHandler(config)
	if handler == nil {
		t.Fatal("NewInvocationHandler() returned nil")
	}

	// Verify the handler responds to JSON-RPC requests
	reqBody := `{"jsonrpc":"2.0","method":"message/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"hello"}]}},"id":"1"}`
	req := httptest.NewRequest(http.MethodPost, "/a2a/invoke", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("handler returned status %d, want %d", rec.Code, http.StatusOK)
	}
}

func TestNewServeMux(t *testing.T) {
	agent, err := newEventReplayAgent(nil, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v", err)
	}

	agentCard := &a2a.AgentCard{
		Name:               "test-agent",
		URL:                "http://localhost:8080/a2a/invoke",
		PreferredTransport: a2a.TransportProtocolJSONRPC,
	}

	config := HandlerConfig{
		ExecutorConfig: ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: session.InMemoryService(),
			},
		},
		AgentCard: agentCard,
	}

	mux := NewServeMux(config)
	if mux == nil {
		t.Fatal("NewServeMux() returned nil")
	}

	t.Run("agent card endpoint", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, a2asrv.WellKnownAgentCardPath, nil)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("agent card endpoint returned status %d, want %d", rec.Code, http.StatusOK)
		}

		body, _ := io.ReadAll(rec.Body)
		var card a2a.AgentCard
		if err := json.Unmarshal(body, &card); err != nil {
			t.Errorf("failed to unmarshal agent card: %v", err)
		}
		if card.Name != agentCard.Name {
			t.Errorf("agent card name = %q, want %q", card.Name, agentCard.Name)
		}
	})

	t.Run("invoke endpoint", func(t *testing.T) {
		reqBody := `{"jsonrpc":"2.0","method":"message/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"hello"}]}},"id":"1"}`
		req := httptest.NewRequest(http.MethodPost, "/a2a/invoke", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Errorf("invoke endpoint returned status %d, want %d", rec.Code, http.StatusOK)
		}
	})
}

func TestNewServeMux_WithoutAgentCard(t *testing.T) {
	agent, err := newEventReplayAgent(nil, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v", err)
	}

	config := HandlerConfig{
		ExecutorConfig: ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: session.InMemoryService(),
			},
		},
		// AgentCard is nil
	}

	mux := NewServeMux(config)
	if mux == nil {
		t.Fatal("NewServeMux() returned nil")
	}

	// Agent card endpoint should return 404 when not configured
	req := httptest.NewRequest(http.MethodGet, a2asrv.WellKnownAgentCardPath, nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Errorf("agent card endpoint returned status %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestNewRequestHandler(t *testing.T) {
	agent, err := newEventReplayAgent(nil, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v", err)
	}

	config := HandlerConfig{
		ExecutorConfig: ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: session.InMemoryService(),
			},
		},
	}

	handler := NewRequestHandler(config)
	if handler == nil {
		t.Fatal("NewRequestHandler() returned nil")
	}

	// The returned handler should be usable with different transports
	// Wrap it with JSON-RPC transport to verify
	jsonrpcHandler := a2asrv.NewJSONRPCHandler(handler)
	if jsonrpcHandler == nil {
		t.Fatal("a2asrv.NewJSONRPCHandler() returned nil")
	}
}
