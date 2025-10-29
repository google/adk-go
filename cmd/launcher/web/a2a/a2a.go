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

// package a2a allows to run A2A
package a2a

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/a2aproject/a2a-go/a2agrpc"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/gorilla/mux"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/adk/adka2a"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/runner"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

type a2aConfig struct {
	rootAgentName        string
	defaultRootAgentName string
}

type A2ALauncher struct {
	flags  *flag.FlagSet
	config *a2aConfig
}

func (a *A2ALauncher) FormatSyntax() string {
	return launcher.FormatFlagUsage(a.flags)
}

func (a *A2ALauncher) Keyword() string {
	return "a2a"
}

func (a *A2ALauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse a2a flags: %v", err)
	}
	// override missing rootAgentName with the default
	if a.config.rootAgentName == "" {
		a.config.rootAgentName = a.config.defaultRootAgentName
	}
	restArgs := a.flags.Args()
	return restArgs, nil
}

func (a *A2ALauncher) WrapHandlers(handler http.Handler, adkConfig *adk.Config) http.Handler {
	grpcSrv := grpc.NewServer()
	newA2AHandler(adkConfig, a.config.rootAgentName).RegisterWith(grpcSrv)
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

func (a *A2ALauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	// no need to setup subrouters, just return
}

// SimpleDescription implements web.WebSublauncher.
func (a *A2ALauncher) SimpleDescription() string {
	return "starts A2A server which handles grpc traffic"
}

// UserMessage implements web.WebSublauncher.
func (a *A2ALauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       a2a:  you can access A2A using grpc protocol: %s", webUrl))
}

func newA2AHandler(serveConfig *adk.Config, agentName string) *a2agrpc.Handler {
	agent, err := serveConfig.AgentLoader.LoadAgent(agentName)
	if err != nil {
		log.Fatalf("cannot load agent %s: %v", agentName, err)
	}
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

// NewLauncher creates new a2a launcher. It extends Web launcher
func NewLauncher(rootAgentName string) *A2ALauncher {
	config := &a2aConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.rootAgentName, "root_agent_name", "", "If you have multiple agents you should specify which one should be user for interactions. You can leave if empty if you have only one agent - it will be used by default")

	return &A2ALauncher{
		config: config,
		flags:  fs,
	}
}
