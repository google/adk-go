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

package mcptoolset_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// answerTimeout bounds every wait of the raw server and of the assertions, so
// a broken sequence fails the test instead of hanging it.
const answerTimeout = 10 * time.Second

// TestElicitationCompleteHandler drives the full URL-mode sequence: the server
// raises an elicitation, the client handler blocks on it, the server reports
// the out-of-band completion, and only then does the tool call finish.
//
// The server answers the tools/call only after it reads the client's reply to
// the elicitation, and that reply only arrives once the completion handler has
// released the elicitation handler. The tool call therefore cannot succeed
// unless ElicitationCompleteHandler reached the client.
//
// The server is a raw JSON-RPC peer rather than an mcp.Server because the SDK
// exports no way for a server to send notifications/elicitation/complete.
func TestElicitationCompleteHandler(t *testing.T) {
	const (
		loginURL      = "https://idp.example.com/login"
		elicitationID = "elicitation-1"
	)

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	startRawServer(t, serverTransport, loginURL, elicitationID)

	completed := make(chan string, 1)
	elicitedURL := make(chan string, 1)
	ts, err := mcptoolset.New(mcptoolset.Config{
		Transport: clientTransport,
		ElicitationHandler: func(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
			elicitedURL <- req.Params.URL
			// URL mode: hold the call open until the server reports completion.
			select {
			case id := <-completed:
				if id != elicitationID {
					t.Errorf("Completion notification carried elicitation ID %q, want %q", id, elicitationID)
				}
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return &mcp.ElicitResult{Action: "accept"}, nil
		},
		ElicitationCompleteHandler: func(ctx context.Context, req *mcp.ElicitationCompleteNotificationRequest) {
			completed <- req.Params.ElicitationID
		},
	})
	if err != nil {
		t.Fatalf("Failed to create MCP tool set: %v", err)
	}

	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
	if err != nil {
		t.Fatalf("Tools call failed: %v", err)
	}
	toolCtx := agent.NewToolContext(invCtx, "", nil, nil)

	fnTool := tools[0].(toolinternal.FunctionTool)
	result, err := fnTool.Run(toolCtx, map[string]any{})
	if err != nil {
		t.Fatalf("Tool call failed: %v", err)
	}

	select {
	case gotURL := <-elicitedURL:
		if gotURL != loginURL {
			t.Errorf("Elicitation handler got URL %q, want %q", gotURL, loginURL)
		}
	case <-time.After(answerTimeout):
		t.Error("Elicitation handler was never called")
	}
	if got, want := result["output"], "logged in"; got != want {
		t.Errorf("Tool output = %v, want %v", got, want)
	}
}

// rawServer answers the MCP requests of a single client session over a
// JSON-RPC connection, so that a test can send messages the SDK's server does
// not expose.
type rawServer struct {
	conn mcp.Connection

	// replies carries the client's responses to the requests the server sent,
	// which the serve loop must not consume itself.
	replies chan *jsonrpc.Response
}

func startRawServer(t *testing.T, transport mcp.Transport, loginURL, elicitationID string) {
	t.Helper()

	conn, err := transport.Connect(t.Context())
	if err != nil {
		t.Fatalf("Failed to connect the server transport: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	s := &rawServer{conn: conn, replies: make(chan *jsonrpc.Response, 1)}
	go s.serve(context.WithoutCancel(t.Context()), loginURL, elicitationID)
}

func (s *rawServer) serve(ctx context.Context, loginURL, elicitationID string) {
	for {
		msg, err := s.conn.Read(ctx)
		if err != nil {
			return
		}
		if res, ok := msg.(*jsonrpc.Response); ok {
			select {
			case s.replies <- res:
			default:
			}
			continue
		}
		req, ok := msg.(*jsonrpc.Request)
		if !ok {
			continue
		}
		switch req.Method {
		case "server/discover":
			// Decline, so the client falls back to the initialize handshake and
			// to a protocol version that carries server-initiated elicitation.
			s.write(ctx, &jsonrpc.Response{
				ID:    req.ID,
				Error: &jsonrpc.Error{Code: jsonrpc.CodeMethodNotFound, Message: "server/discover not supported"},
			})
		case "initialize":
			s.respond(ctx, req.ID, map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "raw_server", "version": "v1.0.0"},
			})
		case "tools/list":
			s.respond(ctx, req.ID, map[string]any{
				"tools": []any{map[string]any{
					"name":        "gateway_tool",
					"description": "elicits a login before responding",
					"inputSchema": map[string]any{"type": "object"},
				}},
			})
		case "tools/call":
			go s.callTool(ctx, req.ID, loginURL, elicitationID)
		case "notifications/initialized", "notifications/cancelled":
		default:
			s.write(ctx, &jsonrpc.Response{
				ID:    req.ID,
				Error: &jsonrpc.Error{Code: jsonrpc.CodeMethodNotFound, Message: req.Method},
			})
		}
	}
}

// callTool answers a tools/call by raising a URL elicitation, reporting its
// out-of-band completion, and returning the result only once the client has
// answered the elicitation.
func (s *rawServer) callTool(ctx context.Context, id jsonrpc.ID, loginURL, elicitationID string) {
	elicitRequestID, err := jsonrpc.MakeID("server-elicit-1")
	if err != nil {
		s.fail(ctx, id, "failed to build the elicitation request ID")
		return
	}
	params, err := json.Marshal(map[string]any{
		"mode":          "url",
		"message":       "please log in",
		"url":           loginURL,
		"elicitationId": elicitationID,
	})
	if err != nil {
		s.fail(ctx, id, "failed to marshal the elicitation params")
		return
	}
	s.write(ctx, &jsonrpc.Request{ID: elicitRequestID, Method: "elicitation/create", Params: params})

	// The client handler blocks on this notification, so it must go out while
	// the elicitation request is still open.
	completeParams, err := json.Marshal(map[string]any{"elicitationId": elicitationID})
	if err != nil {
		s.fail(ctx, id, "failed to marshal the completion params")
		return
	}
	s.write(ctx, &jsonrpc.Request{Method: "notifications/elicitation/complete", Params: completeParams})

	select {
	case reply := <-s.replies:
		if reply.ID != elicitRequestID {
			s.fail(ctx, id, "the client answered an unexpected request")
			return
		}
		if reply.Error != nil {
			s.fail(ctx, id, "the client failed the elicitation: "+reply.Error.Error())
			return
		}
	case <-time.After(answerTimeout):
		s.fail(ctx, id, "the client never answered the elicitation")
		return
	case <-ctx.Done():
		return
	}

	s.respond(ctx, id, map[string]any{
		"content": []any{map[string]any{"type": "text", "text": "logged in"}},
	})
}

func (s *rawServer) respond(ctx context.Context, id jsonrpc.ID, result any) {
	raw, err := json.Marshal(result)
	if err != nil {
		return
	}
	s.write(ctx, &jsonrpc.Response{ID: id, Result: raw})
}

func (s *rawServer) fail(ctx context.Context, id jsonrpc.ID, message string) {
	s.write(ctx, &jsonrpc.Response{
		ID:    id,
		Error: &jsonrpc.Error{Code: jsonrpc.CodeInternalError, Message: message},
	})
}

func (s *rawServer) write(ctx context.Context, msg jsonrpc.Message) {
	_ = s.conn.Write(ctx, msg)
}
