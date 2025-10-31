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

package webui

import (
	"embed"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"

	weblauncher "google.golang.org/adk/cmd/launcher/web"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/internal/cli/util"
)

// webUIConfig contains parametres for lauching ADK Web UI
type webUIConfig struct {
	backendAddress string
	pathPrefix     string
}

// ApiLauncher can launch ADK Web UI
type WebUILauncher struct {
	flags  *flag.FlagSet
	config *webUIConfig
}

// CommandLineSyntax implements web.WebSublauncher.
func (w *WebUILauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(w.flags)
}

// Keyword implements web.WebSublauncher.
func (w *WebUILauncher) Keyword() string {
	return "webui"
}

// Parse implements web.WebSublauncher.
func (w *WebUILauncher) Parse(args []string) ([]string, error) {
	err := w.flags.Parse(args)
	if err != nil || !w.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse webui flags: %v", err)
	}
	restArgs := w.flags.Args()
	return restArgs, nil
}

// WrapHandlers implements the web.WebSublauncher interface. It does not change
// the top-level routes for the WebUI.
func (a *WebUILauncher) WrapHandlers(handler http.Handler, adkConfig *adk.Config) http.Handler {
	// webui doesn't change the top level routes
	return handler
}

// SetupSubrouters implements the web.WebSublauncher interface. It adds the
// WebUI subrouter to the main router.
func (w *WebUILauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	w.AddSubrouter(router, w.config.pathPrefix, adkConfig, w.config.backendAddress)
}

// SimpleDescription returns a simple description of the WebUI launcher.
func (w *WebUILauncher) SimpleDescription() string {
	return "starts ADK Web UI server which provides UI for interacting with ADK REST API"
}

// UserMessage implements the web.WebSublauncher interface. It prints a message
// to the user with the URL to access the WebUI.
func (w *WebUILauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       webui:  you can access API using %s%s", webUrl, w.config.pathPrefix))
}

// embed web UI files into the executable

//go:embed distr/*
var content embed.FS

// AddSubrouter adds a subrouter to serve the ADK Web UI.
func (w *WebUILauncher) AddSubrouter(router *mux.Router, pathPrefix string, adkConfig *adk.Config, backendAddress string) {
	// Setup serving of ADK Web UI
	rUi := router.Methods("GET").PathPrefix(pathPrefix).Subrouter()

	//   generate /assets/config/runtime-config.json in the runtime.
	//   It removes the need to prepare this file during deployment and update the distribution files.
	runtimeConfigResponse := struct {
		BackendUrl string `json:"backendUrl"`
	}{BackendUrl: backendAddress}
	rUi.Methods("GET").Path("/assets/config/runtime-config.json").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers.EncodeJSONResponse(runtimeConfigResponse, http.StatusOK, w)
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
	rUi.Methods("GET").Handler(http.StripPrefix(pathPrefix, http.FileServer(http.FS(ui))))
}

// NewLauncher creates a new WebSublauncher for the ADK Web UI.
func NewLauncher() weblauncher.WebSublauncher {
	config := &webUIConfig{}

	fs := flag.NewFlagSet("webui", flag.ContinueOnError)
	fs.StringVar(&config.backendAddress, "api_server_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'.")
	config.pathPrefix = "/ui/"

	return &WebUILauncher{
		config: config,
		flags:  fs,
	}
}
