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
	"embed"
	"flag"
	"io/fs"
	"log"
	"net/http"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/adkrest/controllers"
)

const WebUIPathPrefix = "/ui/"

// WebUIConfig contains parameters for launching ADK Web UI
type WebUIConfig struct {
	BackendAddress string
}

func DefineWebUIFlags(cfg *WebUIConfig) {
	flag.StringVar(&cfg.BackendAddress, "adk_webapi_address", "http://localhost:8080/api", "ADK REST API server address as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'.")
}

// WebUISubLauncher can launch ADK Web UI
type WebUISubLauncher struct {
	config *WebUIConfig
}

// NewLauncher creates a new Sublauncher for the ADK Web UI.
func NewWebUISubLauncher(cfg *WebUIConfig) *WebUISubLauncher {
	return &WebUISubLauncher{
		config: cfg,
	}
}

// SetupSubrouters adds the WebUI subrouter to the main router.
func (w *WebUISubLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	w.AddSubrouter(router, WebUIPathPrefix, w.config.BackendAddress)
	return nil
}

// embed web UI files into the executable

//go:embed distr/*
var content embed.FS

// AddSubrouter adds a subrouter to serve the ADK Web UI.
func (w *WebUISubLauncher) AddSubrouter(router *mux.Router, pathPrefix, backendAddress string) {
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
