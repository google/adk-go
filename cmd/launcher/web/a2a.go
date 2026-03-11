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

package web

import (
	"flag"
	"net/url"

	a2acore "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
)

// apiPath is a suffix used to build an A2A invocation URL
const apiPath = "/a2a/invoke"

// A2AConfig contains parameters for launching ADK A2A server
type A2AConfig struct {
	AgentURL string // user-provided url which will be used in the agent card to specify url for invoking A2A
}

func DefineA2AFlags(cfg *A2AConfig) {
	flag.StringVar(&cfg.AgentURL, "adk_weba2a_agent_url", "http://localhost:8080", "A2A host URL as advertised in the public agent card. It is used by A2A clients as a connection endpoint.")
}

type A2ASubLauncher struct {
	config *A2AConfig
}

// NewA2ASubLauncher creates new a2a launcher. It extends Web launcher
func NewA2ASubLauncher(cfg *A2AConfig) *A2ASubLauncher {
	return &A2ASubLauncher{
		config: cfg,
	}
}

// SetupSubrouters adds A2A paths to the main router.
func (a *A2ASubLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	publicURL, err := url.JoinPath(a.config.AgentURL, apiPath)
	if err != nil {
		return err
	}

	rootAgent := config.AgentLoader.RootAgent()
	agentCard := &a2acore.AgentCard{
		Name:                              rootAgent.Name(),
		Description:                       rootAgent.Description(),
		DefaultInputModes:                 []string{"text/plain"},
		DefaultOutputModes:                []string{"text/plain"},
		URL:                               publicURL,
		PreferredTransport:                a2acore.TransportProtocolJSONRPC,
		Skills:                            adka2a.BuildAgentSkills(rootAgent),
		Capabilities:                      a2acore.AgentCapabilities{Streaming: true},
		SupportsAuthenticatedExtendedCard: false,
	}
	router.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(agentCard))

	agent := config.AgentLoader.RootAgent()
	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:         agent.Name(),
			Agent:           agent,
			SessionService:  config.SessionService,
			ArtifactService: config.ArtifactService,
			PluginConfig:    config.PluginConfig,
		},
	})
	reqHandler := a2asrv.NewHandler(executor, config.A2AOptions...)
	router.Handle(apiPath, a2asrv.NewJSONRPCHandler(reqHandler))
	return nil
}
