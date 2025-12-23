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

// Package main provides an example ADK agent that uses A2A.
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/remoteagent"
	"github.com/sjzsdu/adk-go/cmd/launcher"
	"github.com/sjzsdu/adk-go/cmd/launcher/full"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/server/adka2a"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/geminitool"
	"github.com/sjzsdu/adk-go/util/modelfactory"
)

// newWeatherAgent creates a simple LLM-agent as in the quickstart example.
func newWeatherAgent(ctx context.Context, modelConfig *modelfactory.Config) agent.Agent {
	// 从模型配置创建模型
	model := modelfactory.MustCreateModel(ctx, modelConfig)

	// 配置工具列表，根据模型类型决定使用哪种搜索工具
	tools := []tool.Tool{}

	// 根据模型类型选择合适的搜索工具
	tools = append(tools, geminitool.GoogleSearch{})

	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_time_agent",
		Model:       model,
		Description: "Agent to answer questions about the time and weather in a city.",
		Instruction: "I can answer your questions about the time and weather in a city. When you need information about the current weather or time in a specific city, use the google_search tool to find the latest information.",
		Tools:       tools,
	})
	if err != nil {
		log.Fatalf("Failed to create an agent: %v", err)
	}
	return agent
}

// startWeatherAgentServer starts an HTTP server which exposes a weather agent using A2A (Agent-To-Agent) protocol.
func startWeatherAgentServer(ctx context.Context, modelConfig *modelfactory.Config) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("Failed to bind to a port: %v", err)
	}

	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}

	log.Printf("Starting A2A server on %s", baseURL.String())

	go func() {
		agent := newWeatherAgent(ctx, modelConfig)

		agentPath := "/invoke"
		agentCard := &a2a.AgentCard{
			Name:               agent.Name(),
			Skills:             adka2a.BuildAgentSkills(agent),
			PreferredTransport: a2a.TransportProtocolJSONRPC,
			URL:                baseURL.JoinPath(agentPath).String(),
			Capabilities:       a2a.AgentCapabilities{Streaming: true},
		}

		mux := http.NewServeMux()
		mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

		executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: session.InMemoryService(),
			},
		})
		requestHandler := a2asrv.NewHandler(executor)
		mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(requestHandler))

		err := http.Serve(listener, mux)

		log.Printf("A2A server stopped: %v", err)
	}()

	return baseURL.String()
}

func main() {
	ctx := context.Background()

	// 解析命令行参数
	flag.Parse()

	// 从命令行参数创建模型配置
	modelConfig := modelfactory.NewFromFlags()

	a2aServerAddress := startWeatherAgentServer(ctx, modelConfig)

	remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:            "A2A Weather agent",
		AgentCardSource: a2aServerAddress,
	})
	if err != nil {
		log.Fatalf("Failed to create a remote agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(remoteAgent),
	}

	l := full.NewLauncher()
	// 过滤掉-model和-model-name参数，避免与launcher参数冲突
	launcherArgs := modelfactory.ExtractLauncherArgs(os.Args[1:])
	if err = l.Execute(ctx, config, launcherArgs); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
