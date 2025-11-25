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

// Package webui provides a sublauncher that adds ADK Web UI to the web server (using url /ui/)
package webui

import (
	"embed"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/adkrest/controllers"
)

// webUIConfig contains parameters for launching ADK Web UI
type webUIConfig struct {
	backendAddress string
	pathPrefix     string
}

// webUILauncher can launch ADK Web UI
type webUILauncher struct {
	flags  *flag.FlagSet
	config *webUIConfig
}

// CommandLineSyntax implements web.Sublauncher. Returns the command-line syntax for the WebUI launcher.
func (w *webUILauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(w.flags)
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for WebUI launcher.
func (w *webUILauncher) Keyword() string {
	return "webui"
}

// Parse implements web.Sublauncher. After parsing webui-specific arguments returns remaining unparsed arguments
func (w *webUILauncher) Parse(args []string) ([]string, error) {
	err := w.flags.Parse(args)
	if err != nil || !w.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse webui flags: %v", err)
	}
	restArgs := w.flags.Args()
	return restArgs, nil
}

// SetupSubrouters implements the web.Sublauncher interface. It adds the
// WebUI subrouter to the main router.
func (w *webUILauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	w.AddSubrouter(router, w.config.pathPrefix, w.config.backendAddress)
	return nil
}

// SimpleDescription returns a simple description of the WebUI launcher.
func (w *webUILauncher) SimpleDescription() string {
	return "starts ADK Web UI server which provides UI for interacting with ADK REST API"
}

// UserMessage implements the web.Sublauncher interface. It prints a message
// to the user with the URL to access the WebUI.
func (w *webUILauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       webui:  you can access API using %s%s", webURL, w.config.pathPrefix))
}

// embed web UI files into the executable

//go:embed distr/*
var content embed.FS

// AddSubrouter adds a subrouter to serve the ADK Web UI.
func (w *webUILauncher) AddSubrouter(router *mux.Router, pathPrefix, backendAddress string) {
	// Setup serving of ADK Web UI
	rUI := router.Methods("GET").PathPrefix(pathPrefix).Subrouter()

	//   generate /assets/config/runtime-config.json in the runtime.
	//   It removes the need to prepare this file during deployment and update the distribution files.
	runtimeConfigResponse := struct {
		BackendUrl string `json:"backendUrl"`
	}{BackendUrl: backendAddress}
	rUI.Methods("GET").Path("/assets/config/runtime-config.json").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		controllers.EncodeJSONResponse(runtimeConfigResponse, http.StatusOK, w)
	})

	//   redirect the user from / to pathPrefix (/ui/)
	router.Methods("GET").Path("/").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, pathPrefix, http.StatusFound)
	})

	// serve web ui from the embedded resources
	ui, err := fs.Sub(content, "distr")
	if err != nil {
		log.Fatalf("cannot prepare ADK Web UI files as embedded content: %v", err)
	}
	rUI.Methods("GET").Handler(http.StripPrefix(pathPrefix, http.FileServer(http.FS(ui))))
}

// NewLauncher creates a new Sublauncher for the ADK Web UI.
func NewLauncher() weblauncher.Sublauncher {
	config := &webUIConfig{}

	fs := flag.NewFlagSet("webui", flag.ContinueOnError)
	fs.StringVar(&config.backendAddress, "api_server_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'.")
	config.pathPrefix = "/ui/"

	return &webUILauncher{
		config: config,
		flags:  fs,
	}
}
