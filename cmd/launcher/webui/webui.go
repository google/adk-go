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

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/restapi/handlers"
)

// WebUIConfig contains parametres for lauching ADK Web UI
type WebUIConfig struct {
	backendAddress string
	pathPrefix     string
}

// ApiLauncher can launch ADK Web UI
type WebUILauncher struct {
	flags  *flag.FlagSet
	config *WebUIConfig
}

// FormatSyntax implements web.WebSublauncher.
func (w *WebUILauncher) FormatSyntax() string {
	return launcher.FormatFlagUsage(w.flags)
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

// SetupRoutes implements web.WebSublauncher.
func (w *WebUILauncher) SetupRoutes(router *mux.Router, adkConfig *adk.Config) {
	// no need to modify top level routes
}

// SetupSubrouters implements web.WebSublauncher.
func (w *WebUILauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	w.AddSubrouter(router, w.config.pathPrefix, adkConfig, w.config.backendAddress)
}

// SimpleDescription implements web.WebSublauncher.
func (w *WebUILauncher) SimpleDescription() string {
	return "starts ADK Web UI server which provides UI for interacting with ADK REST API"
}

// UserMessage implements web.WebSublauncher.
func (w *WebUILauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       webui:  you can access API using %s%s", webUrl, w.config.pathPrefix))
}

// embed web UI files into the executable

//go:embed distr/*
var content embed.FS

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

func NewLauncher() *WebUILauncher {
	config := &WebUIConfig{}

	fs := flag.NewFlagSet("webui", flag.ContinueOnError)
	fs.StringVar(&config.backendAddress, "api_server_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'.")
	config.pathPrefix = "/ui/"

	return &WebUILauncher{
		config: config,
		flags:  fs,
	}
}
