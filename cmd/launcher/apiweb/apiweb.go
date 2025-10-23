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

// package api allows to run ADK REST API and ADK Web UI
package apiweb

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/web/base"
	"google.golang.org/adk/cmd/launcher/webui"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
	"google.golang.org/adk/session"
)

type ApiWebConfig struct {
	port            int
	frontendAddress string
	backendAddress  string
}

type ApiWebLauncher struct {
	config *ApiWebConfig
}

func ParseArgs(args []string) (*ApiWebConfig, []string, error) {
	fs := flag.NewFlagSet("apiweb", flag.ContinueOnError)

	localPortFlag := fs.Int("port", 8080, "Localhost port for the server")
	frontendAddressFlag := fs.String("webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")
	backendAddressFlag := fs.String("api_server_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'.")

	err := fs.Parse(args)
	if err != nil || !fs.Parsed() {
		return &(ApiWebConfig{}), nil, fmt.Errorf("failed to parse flags: %v", err)
	}

	res := &ApiWebConfig{
		port:            *localPortFlag,
		frontendAddress: *frontendAddressFlag,
		backendAddress:  *backendAddressFlag,
	}
	return res, fs.Args(), nil
}

// BuildLauncher parses command line args and returns ready-to-run console launcher.
func BuildLauncher(args []string) (launcher.Launcher, []string, error) {
	apiConfig, argsLeft, err := ParseArgs(args)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot parse arguments for apiweb: %v: %w", args, err)
	}
	return &ApiWebLauncher{config: apiConfig}, argsLeft, nil
}

func addSubrouter(router *mux.Router, pathPrefix string, adkConfig *adk.Config) *mux.Router {
	rApi := router.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix(pathPrefix).Subrouter()
	restapiweb.SetupRouter(rApi, adkConfig)
	return rApi
}

func (l ApiWebLauncher) Run(ctx context.Context, config *adk.Config) error {
	// we need some session service, add one if missing
	if config.SessionService == nil {
		config.SessionService = session.InMemoryService()
	}

	router := base.BuildBaseRouter()
	rApi := addSubrouter(router, "/api/", config)
	_ = webui.AddSubrouter(router, "/ui/", config, l.config.backendAddress)

	// Setup serving of ADK REST API
	rApi.Use(base.CorsWithArgs(l.config.frontendAddress))

	log.Printf("Starting the ADK REST API server: %+v", l.config)
	log.Println()
	log.Printf("Open %s", "http://localhost:"+strconv.Itoa(l.config.port))
	log.Println()
	return http.ListenAndServe(":"+strconv.Itoa(l.config.port), router)
}

// Run parses command line params, prepares api launcher and runs it
func Run(ctx context.Context, config *adk.Config) error {
	// skip args[0] - executable file name
	// skip unparsed arguments returned by BuildLauncher
	launcherToRun, _, err := BuildLauncher(os.Args[1:])
	if err != nil {
		log.Fatalf("cannot build api launcher: %v", err)
	}

	err = launcherToRun.Run(ctx, config)
	if err != nil {
		log.Fatalf("run failed: %v", err)
	}
	return nil
}
