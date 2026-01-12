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

// This example demonstrates forwarding A2A request metadata to MCP tool calls.
//
// This example shows how to:
//  1. Create an A2A server that exposes an ADK agent with MCP tools
//  2. Use BeforeExecuteCallback to attach A2A metadata to the context
//  3. Configure the MCP toolset to forward metadata to MCP tool calls
//  4. Access the forwarded metadata in MCP tool handlers
//

// ┌─────────────┐     A2A Request      ┌─────────────────┐     MCP CallTool   ┌────────────────┐
// │  A2A Client │ ──────────────────▶ │   ADK Agent     │ ─────────────────▶ │   MCP Server   │
// │             │   (with metadata)    │ (A2A + MCP)     │  (with metadata)   │   (echo tool)  │
// └─────────────┘                      └─────────────────┘                    └────────────────┘
//                                            │
//                                            │ BeforeExecuteCallback
//                                            │ attaches A2A metadata
//                                            │ to context
//                                            ▼
//                                     ┌─────────────────┐
//                                     │ A2AMetadata in  │
//                                     │ context.Context │
//                                     └─────────────────┘
//                                            │
//                                            │ MetadataProvider
//                                            │ extracts metadata
//                                            │ from context
//                                            ▼
//                                     ┌─────────────────┐
//                                     │ mcp.CallTool-   │
//                                     │ Params.Meta     │
//                                     └─────────────────┘

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/remoteagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
)

type a2aMetadataCtxKey struct{}

type A2AMetadata struct {
	// RequestMetadata contains arbitrary metadata from the A2A request.
	RequestMetadata map[string]any
	// MessageMetadata contains metadata from the A2A message.
	MessageMetadata map[string]any
}

// ContextWithA2AMetadata returns a new context with A2A metadata attached.
// This can be used in BeforeExecuteCallback to attach A2A request metadata
// to the context for downstream propagation to MCP tool calls.
func ContextWithA2AMetadata(ctx context.Context, meta *A2AMetadata) context.Context {
	return context.WithValue(ctx, a2aMetadataCtxKey{}, meta)
}

// A2AMetadataFromContext retrieves A2A metadata from the context.
// Returns nil if no metadata is present.
func A2AMetadataFromContext(ctx context.Context) *A2AMetadata {
	meta, ok := ctx.Value(a2aMetadataCtxKey{}).(*A2AMetadata)
	if !ok {
		return nil
	}
	return meta
}

// A2AMetadataProvider creates a function that extracts A2A request metadata
// from the context. The returned function can be used as mcptoolset.MetadataProvider
// to forward A2A metadata to MCP tool calls.
//
// The forwarded metadata includes:
//   - "a2a:task_id": The A2A task ID (if present)
//   - "a2a:context_id": The A2A context ID (if present)
//   - Any keys specified in forwardKeys from request/message metadata
//
// If forwardKeys is nil or empty, all request and message metadata keys are forwarded.
// If forwardKeys is non-empty, only the specified keys are forwarded.
func A2AMetadataProvider(forwardKeys []string) func(tool.Context) map[string]any {
	keySet := make(map[string]bool)
	for _, k := range forwardKeys {
		keySet[k] = true
	}

	return func(ctx tool.Context) map[string]any {
		a2aMeta := A2AMetadataFromContext(ctx)
		if a2aMeta == nil {
			return nil
		}

		result := make(map[string]any)

		// Forward selected or all metadata keys
		forwardMetadata := func(source map[string]any) {
			for k, v := range source {
				if len(keySet) == 0 || keySet[k] {
					result[k] = v
				}
			}
		}

		forwardMetadata(a2aMeta.RequestMetadata)
		forwardMetadata(a2aMeta.MessageMetadata)

		if len(result) == 0 {
			return nil
		}
		return result
	}
}

// EchoInput defines the input schema for the echo tool.
type EchoInput struct {
	Message string `json:"message" jsonschema:"The message to echo back"`
}

// EchoOutput defines the output schema for the echo tool.
type EchoOutput struct {
	Echo     string         `json:"echo" jsonschema:"The echoed message"`
	Metadata map[string]any `json:"metadata" jsonschema:"Metadata received from the request"`
}

// EchoWithMetadata is an MCP tool that echoes back the input message along with
// any metadata that was forwarded from the A2A request.
func EchoWithMetadata(ctx context.Context, req *mcp.CallToolRequest, input EchoInput) (*mcp.CallToolResult, EchoOutput, error) {
	// The metadata forwarded from A2A is available in req.Params.Meta
	metadata := make(map[string]any)
	if req.Params.Meta != nil {
		metadata = req.Params.Meta
	}

	// Log the received metadata for demonstration
	log.Printf("[MCP Tool] Received metadata: %v", metadata)

	return nil, EchoOutput{
		Echo:     fmt.Sprintf("You said: %s", input.Message),
		Metadata: metadata,
	}, nil
}

// createMCPTransport creates an in-memory MCP server with the echo tool.
func createMCPTransport(ctx context.Context) mcp.Transport {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "metadata_demo_server",
		Version: "v1.0.0",
	}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "echo_with_metadata",
		Description: "Echoes back the message along with any metadata from the request. Use this to verify metadata forwarding.",
	}, EchoWithMetadata)

	_, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		log.Fatalf("Failed to connect MCP server: %v", err)
	}

	return clientTransport
}

// createAgent creates an LLM agent with MCP tools configured to forward A2A metadata.
func createAgent(ctx context.Context) agent.Agent {
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create MCP toolset with metadata forwarding enabled
	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Transport: createMCPTransport(ctx),
		// MetadataProvider extracts A2A metadata from the context and forwards it to MCP tools.
		// A2AMetadataProvider(nil) forwards all metadata fields.
		// You can also specify specific keys to forward: A2AMetadataProvider([]string{"trace_id"})
		MetadataProvider: adka2a.A2AMetadataProvider(nil),
	})
	if err != nil {
		log.Fatalf("Failed to create MCP tool set: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "metadata_demo_agent",
		Model:       model,
		Description: "An agent that demonstrates A2A to MCP metadata forwarding.",
		Instruction: `You are a helpful assistant that can echo messages back to users.
When the user asks you to echo something or test metadata, use the echo_with_metadata tool.
The tool will show any metadata that was forwarded from the A2A request.`,
		Toolsets: []tool.Toolset{mcpToolSet},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	return a
}

// startA2AServer starts an HTTP server exposing the agent via A2A protocol.
// It uses BeforeExecuteCallback to attach A2A metadata to the context.
func startA2AServer(ctx context.Context) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("Failed to bind to a port: %v", err)
	}

	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}
	log.Printf("Starting A2A server on %s", baseURL.String())

	go func() {
		a := createAgent(ctx)
		agentPath := "/invoke"

		agentCard := &a2a.AgentCard{
			Name:               a.Name(),
			Skills:             adka2a.BuildAgentSkills(a),
			PreferredTransport: a2a.TransportProtocolJSONRPC,
			URL:                baseURL.JoinPath(agentPath).String(),
			Capabilities:       a2a.AgentCapabilities{Streaming: true},
		}

		executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        a.Name(),
				Agent:          a,
				SessionService: session.InMemoryService(),
			},
			// BeforeExecuteCallback is called before each agent execution.
			// Here we extract A2A request metadata and attach it to the context
			// so it can be forwarded to MCP tool calls.
			BeforeExecuteCallback: func(ctx context.Context, reqCtx *a2asrv.RequestContext) (context.Context, error) {
				log.Printf("[A2A Server] Received request with TaskID: %s, ContextID: %s", reqCtx.TaskID, reqCtx.ContextID)

				// Extract metadata from the A2A request
				meta := &adka2a.A2AMetadata{
					TaskID:    string(reqCtx.TaskID),
					ContextID: reqCtx.ContextID,
				}

				// Include reqest-level metadata
				if reqCtx.Metadata != nil {
					meta.RequestMetadata = reqCtx.Metadata
					log.Printf("[A2A Server] Request metadata: %v", reqCtx.Metadata)
				}

				// Include message-level metadata
				if reqCtx.Message != nil && reqCtx.Message.Metadata != nil {
					meta.MessageMetadata = reqCtx.Message.Metadata
					log.Printf("[A2A Server] Message metadata: %v", reqCtx.Message.Metadata)
				}

				return adka2a.ContextWithA2AMetadata(ctx, meta), nil
			},
		})

		mux := http.NewServeMux()
		mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))
		mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))

		if err := http.Serve(listener, mux); err != nil {
			log.Printf("A2A server stopped: %v", err)
		}
	}()

	return baseURL.String()
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Start A2A server with the agent
	a2aServerAddress := startA2AServer(ctx)

	// Create a remote agent that connects to the A2A server
	remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:            "Remote Metadata Demo Agent",
		AgentCardSource: a2aServerAddress,
	})
	if err != nil {
		log.Fatalf("Failed to create remote agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(remoteAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
