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

// Package a2a allows to run A2A
package a2a

import (
	"flag"
	"fmt"
	"net/http"
	"strings"

	a2acore "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2agrpc"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/gorilla/mux"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/adk/adka2a"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/runner"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

type a2aConfig struct {
	agentURL string
}

type a2aLauncher struct {
	flags  *flag.FlagSet
	config *a2aConfig
}

// NewLauncher creates new a2a launcher. It extends Web launcher
func NewLauncher() web.WebSublauncher {
	config := &a2aConfig{}

	fs := flag.NewFlagSet("a2a", flag.ContinueOnError)

	fs.StringVar(&config.agentURL, "a2a_agent_url", "localhost:8080", "A2A gRPC host URL as advertised in the public agent card. It is used by A2A clients as a connection endpoint.")

	return &a2aLauncher{
		config: config,
		flags:  fs,
	}
}

func (a *a2aLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

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

// WrapHandlers implements web.WebSublauncher. Returns http handler which basing on content-type
// chooses between a2a grpc handler and provided handler
func (a *a2aLauncher) WrapHandlers(handler http.Handler, adkConfig *adk.Config) http.Handler {
	grpcSrv := grpc.NewServer()
	newA2AHandler(adkConfig).RegisterWith(grpcSrv)
	reflection.Register(grpcSrv)
	var result http.Handler
	result = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			grpcSrv.ServeHTTP(w, r)
		} else {
			handler.ServeHTTP(w, r)
		}
	})
	result = h2c.NewHandler(result, &http2.Server{})
	return result
}

// SimpleDescription implements web.WebSublauncher. For A2A no subrouter definition is needed
func (a *a2aLauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	rootAgent := adkConfig.AgentLoader.RootAgent()
	agentCard := &a2acore.AgentCard{
		Name:               rootAgent.Name(),
		Description:        rootAgent.Description(),
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		URL:                a.config.agentURL,
		PreferredTransport: a2acore.TransportProtocolGRPC,
		Skills:             adka2a.BuildAgentSkills(rootAgent),
		Capabilities:       a2acore.AgentCapabilities{Streaming: true},
		// gRPC GetAgentCard() method will be serving empty card.
		SupportsAuthenticatedExtendedCard: false,
	}
	router.Handle("/.well-known/agent-card.json", a2asrv.NewStaticAgentCardHandler(agentCard))
}

// SimpleDescription implements web.WebSublauncher
func (a *a2aLauncher) SimpleDescription() string {
	return "starts A2A server which handles grpc traffic"
}

// UserMessage implements web.WebSublauncher.
func (a *a2aLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       a2a:  you can access A2A using grpc protocol: %s", webUrl))
}

// newA2AHandler creates a new A2A handler from the provided ADK configuration.
func newA2AHandler(serveConfig *adk.Config) *a2agrpc.Handler {
	agent := serveConfig.AgentLoader.RootAgent()
	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:         agent.Name(),
			Agent:           agent,
			SessionService:  serveConfig.SessionService,
			ArtifactService: serveConfig.ArtifactService,
		},
	})
	reqHandler := a2asrv.NewHandler(executor, serveConfig.A2AOptions...)
	grpcHandler := a2agrpc.NewHandler(reqHandler)
	return grpcHandler
}
