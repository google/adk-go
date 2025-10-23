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
	"context"
	"flag"
	"fmt"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
)

// WebConfig contains command-line params for web launcher
type WebConfig struct {
	LocalPort       int
	FrontendAddress string
	BackendAddress  string
	ServeA2A        bool
}

// WebLauncher allows to interact with an agent in browser (using ADK Web UI and ADK REST API)
type WebLauncher struct {
	Config *WebConfig
}

// Run starts web server, serving everything required for interaction via web browser
func (l WebLauncher) Run(ctx context.Context, config *adk.Config) error {
	Serve(l.Config, config)
	return nil
}

// BuildLauncher parses command line args and returns ready-to-run web launcher.
func BuildLauncher(args []string) (launcher.Launcher, []string, error) {
	webConfig, argsLeft, err := ParseArgs(args)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot parse arguments for web: %v: %w", args, err)
	}
	return &WebLauncher{Config: webConfig}, argsLeft, nil
}

func ParseArgs(args []string) (*WebConfig, []string, error) {
	fs := flag.NewFlagSet("web", flag.ContinueOnError)

	localPortFlag := fs.Int("port", 8080, "Localhost port for the server")
	frontendAddressFlag := fs.String("webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")
	backendAddressFlag := fs.String("api_server_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'. ")
	serveA2A := fs.Bool("serve_a2a", false, "Run a gRPC A2A (Agent-To-Agent) server on the provided address. Will use golang.org/x/net/http2 for HTTP/2 support. Protocol specification can be found at https://a2a-protocol.org.")

	err := fs.Parse(args)
	if err != nil || !fs.Parsed() {
		return &(WebConfig{}), nil, fmt.Errorf("failed to parse flags: %v", err)
	}
	res := WebConfig{
		LocalPort:       *localPortFlag,
		FrontendAddress: *frontendAddressFlag,
		BackendAddress:  *backendAddressFlag,
		ServeA2A:        *serveA2A,
	}
	return &res, fs.Args(), nil
}
