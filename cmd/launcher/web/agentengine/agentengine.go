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

// Package agentengine provides a sublauncher that provides web interface as required by Agent Engine
package agentengine

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/agentengine"
)

// agentEngineConfig contains parameters for launching ADK Agent Engine server
type agentEngineConfig struct {
	pathPrefix      string
	agentEngineID   string
	maxPayloadSize  int64
	sseWriteTimeout time.Duration
}

type agentEngineLauncher struct {
	flags  *flag.FlagSet // flags are used to parse command-line arguments
	config *agentEngineConfig
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher(agentEngineId string) weblauncher.Sublauncher {
	config := &agentEngineConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "ADK Agent Engine API path prefix. Default is '/api'.")
	fs.Int64Var(&config.maxPayloadSize, "max_payload_size", 10*1024*1024, "The payload will be truncated after this amount of bytes")
	fs.DurationVar(&config.sseWriteTimeout, "sse-write-timeout", 120*time.Second, "SSE server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the SSE response after reading the headers & body")

	config.agentEngineID = agentEngineId

	return &agentEngineLauncher{
		config: config,
		flags:  fs,
	}
}

// CommandLineSyntax implements web.Sublauncher. Returns the command-line syntax for the agentEngine launcher.
func (a *agentEngineLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

// SimpleDescription implements web.Sublauncher
func (a *agentEngineLauncher) SimpleDescription() string {
	return "starts AgentEngine server which serves reasoning engine API while deployed to Agent Engine"
}

// UserMessage implements web.Sublauncher.
func (a *agentEngineLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	// TODO(kdroste) description
	printer(fmt.Sprintf("       agentEngine:  you can access this server locally by %s%s", webUrl, "/api/reasoning_engine"))
	printer(fmt.Sprintf("                                                           %s%s", webUrl, "/api/stream_reasoning_engine"))
	printer("                     to access it while deployed to Agent Engine, you should use")
	printer("                               https://${LOCATION_ID}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION_ID}/reasoningEngines/${RESOURCE_ID}:query")
	printer("                            or https://${LOCATION_ID}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION_ID}/reasoningEngines/${RESOURCE_ID}:streamQuery")
}

// SetupSubrouters adds the API router to the parent router.
func (a *agentEngineLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// Create the ADK AgentEngine API handler
	apiHandler, err := agentengine.NewHandler(config, a.config.sseWriteTimeout, a.config.maxPayloadSize, a.config.agentEngineID)
	if err != nil {
		return fmt.Errorf("agentengine.NewHandler failed: %v", err)
	}

	router.Methods("POST").
		PathPrefix(a.config.pathPrefix).
		Handler(http.StripPrefix(a.config.pathPrefix, apiHandler))

	return nil
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for A2A launcher.
func (a *agentEngineLauncher) Keyword() string {
	return "agentengine"
}

var _ weblauncher.Sublauncher = &agentEngineLauncher{}

// Parse parses the command-line arguments for the API launcher.
func (a *agentEngineLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse agent engine flags: %v", err)
	}
	p := a.config.pathPrefix
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	a.config.pathPrefix = strings.TrimSuffix(p, "/")

	restArgs := a.flags.Args()
	return restArgs, nil
}
