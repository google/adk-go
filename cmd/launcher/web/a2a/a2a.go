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

// Package a2a provides a sublauncher that adds A2A capabilities to the web server
package a2a

import (
	"flag"
	"fmt"
	"net/url"

	a2acore "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
)

// apiPath is a suffix used to build an A2A invocation URL
const apiPath = "/a2a/invoke"

// a2aConfig contains parameters for launching ADK A2A server
type a2aConfig struct {
	agentURL string // user-provided url which will be used in the agent card to specify url for invoking A2A
}

type a2aLauncher struct {
	flags  *flag.FlagSet // flags are used to parse command-line arguments
	config *a2aConfig
}

// NewLauncher creates new a2a launcher. It extends Web launcher
func NewLauncher() web.Sublauncher {
	config := &a2aConfig{}

	fs := flag.NewFlagSet("a2a", flag.ContinueOnError)

	fs.StringVar(&config.agentURL, "a2a_agent_url", "http://localhost:8080", "A2A host URL as advertised in the public agent card. It is used by A2A clients as a connection endpoint.")

	return &a2aLauncher{
		config: config,
		flags:  fs,
	}
}

// CommandLineSyntax implements web.Sublauncher. Returns the command-line syntax for the A2A launcher.
func (a *a2aLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for A2A launcher.
func (a *a2aLauncher) Keyword() string {
	return "a2a"
}

func (a *a2aLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse a2a flags: %v", err)
	}
	restArgs := a.flags.Args()
	return restArgs, nil
}

// SetupSubrouters implements the web.Sublauncher interface. It adds A2A paths to the main router.
func (a *a2aLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	publicURL, err := url.JoinPath(a.config.agentURL, apiPath)
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
		},
	})
	reqHandler := a2asrv.NewHandler(executor, config.A2AOptions...)
	router.Handle(apiPath, a2asrv.NewJSONRPCHandler(reqHandler))
	return nil
}

// SimpleDescription implements web.Sublauncher
func (a *a2aLauncher) SimpleDescription() string {
	return fmt.Sprintf("starts A2A server which handles jsonrpc requests on %s path", apiPath)
}

// UserMessage implements web.Sublauncher.
func (a *a2aLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       a2a:  you can access A2A using jsonrpc protocol: %s", webUrl))
}
