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

	"github.com/a2aproject/a2a-go/a2agrpc"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/gorilla/mux"
	"google.golang.org/adk/adka2a"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/runner"
	"google.golang.org/grpc"
)

type A2AConfig struct {
	rootAgentName        string
	defaultRootAgentName string
}

type A2ALauncher struct {
	flags  *flag.FlagSet
	config *A2AConfig
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

func (a *A2ALauncher) SetupRoutes(router *mux.Router, adkConfig *adk.Config) {
	grpcSrv := grpc.NewServer()
	newA2AHandler(adkConfig, a.config.rootAgentName).RegisterWith(grpcSrv)
	router.Headers("Content-Type", "application/grpc").Handler(grpcSrv)
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

// // ParseArgs returns a config from parsed arguments and the remaining un-parsed arguments
// func ParseArgs(args []string) (*A2AConfig, []string, error) {
// 	fs := flag.NewFlagSet("a2a", flag.ContinueOnError)

// 	rootAgentName := fs.String("a2a_root_agent_name", "", "If you have multiple agents you should specify which one should be user for interactions. You can leave if empty if you have only one agent - it will be used by default")
// 	localPortFlag := fs.Int("port", 8080, "Localhost port for the server")

// 	err := fs.Parse(args)
// 	if err != nil || !fs.Parsed() {
// 		return &A2AConfig{}, nil, fmt.Errorf("failed to parse flags: %v", err)
// 	}
// 	res := &A2AConfig{
// 		rootAgentName: *rootAgentName,
// 		port:          *localPortFlag,
// 	}
// 	return res, fs.Args(), nil
// }

// // BuildLauncher parses command line args and returns ready-to-run console launcher.
// func BuildLauncher(args []string) (launcher.Launcher, []string, error) {
// 	a2aConfig, argsLeft, err := ParseArgs(args)
// 	if err != nil {
// 		return nil, nil, fmt.Errorf("cannot parse arguments for a2a: %v: %w", args, err)
// 	}
// 	return &A2ALauncher{config: a2aConfig}, argsLeft, nil
// }

func newA2AHandler(serveConfig *adk.Config, agentName string) *a2agrpc.GRPCHandler {
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
	grpcHandler := a2agrpc.NewHandler(&adka2a.CardProducer{Agent: agent}, reqHandler)
	return grpcHandler
}

// func WrapHandler(router *mux.Router, config *adk.Config, agentName string) http.Handler {
// 	router.Headers()
// 	grpcSrv := grpc.NewServer()
// 	newA2AHandler(config, agentName).RegisterWith(grpcSrv)
// 	var handler http.Handler
// 	handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
// 		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
// 			grpcSrv.ServeHTTP(w, r)
// 		} else {
// 			router.ServeHTTP(w, r)
// 		}
// 	})
// 	handler = h2c.NewHandler(handler, &http2.Server{})
// 	return handler
// }

// func (l A2ALauncher) Run(ctx context.Context, config *adk.Config) error {
// 	// we need some session service, add one if missing
// 	if config.SessionService == nil {
// 		config.SessionService = session.InMemoryService()
// 	}

// 	grpcSrv := grpc.NewServer()
// 	newA2AHandler(config, l.config.rootAgentName).RegisterWith(grpcSrv)

// 	log.Printf("Starting the ADK REST API server: %+v", l.config)
// 	log.Println()
// 	log.Printf("You can call A2A using grpc protocol: %s", "http://localhost:"+strconv.Itoa(l.config.port))
// 	log.Println()
// 	return http.ListenAndServe(":"+strconv.Itoa(l.config.port), grpcSrv)
// }

// // // Run parses command line params, prepares api launcher and runs it
// // func Run(ctx context.Context, config *adk.Config) error {
// // 	// skip args[0] - executable file name
// // 	// skip unparsed arguments returned by BuildLauncher
// // 	launcherToRun, _, err := BuildLauncher(os.Args[1:])
// // 	if err != nil {
// // 		log.Fatalf("cannot build api launcher: %v", err)
// // 	}

// // 	err = launcherToRun.Run(ctx, config)
// // 	if err != nil {
// // 		log.Fatalf("run failed: %v", err)
// // 	}
// // 	return nil
// // }

// NewLauncher creates new a2a launcher. It extends Web launcher
func NewLauncher(rootAgentName string) *A2ALauncher {
	config := &A2AConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.rootAgentName, "a2a_root_agent_name", "", "If you have multiple agents you should specify which one should be user for interactions. You can leave if empty if you have only one agent - it will be used by default")

	return &A2ALauncher{
		config: config,
		flags:  fs,
	}
}
