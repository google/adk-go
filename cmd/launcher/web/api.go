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
	"flag"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/adkrest"
)

// APIConfig contains parametres for lauching ADK REST API
type APIConfig struct {
	FrontendAddress string
	PathPrefix      string
	SSEWriteTimeout time.Duration
}

func DefineAPIFlags(cfg *APIConfig) {
	flag.StringVar(&cfg.FrontendAddress, "adk_webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")
	flag.StringVar(&cfg.PathPrefix, "adk_webapi_path_prefix", "/api", "ADK REST API path prefix. Default is '/api'.")
	flag.DurationVar(&cfg.SSEWriteTimeout, "adk_webapi_sse_write_timeout", 120*time.Second, "SSE server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the SSE response after reading the headers & body")
}

// APISubLauncher can launch ADK REST API
type APISubLauncher struct {
	frontendAddress string
	pathPrefix      string
	sseWriteTimeout time.Duration
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewAPISubLauncher(cfg *APIConfig) *APISubLauncher {
	p := cfg.PathPrefix
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	p = strings.TrimSuffix(p, "/")

	return &APISubLauncher{
		frontendAddress: cfg.FrontendAddress,
		pathPrefix:      p,
		sseWriteTimeout: cfg.SSEWriteTimeout,
	}
}

// Adds CORS headers which allow calling ADK REST API from another web app (like ADK WebUI)
func corsWithArgs(frontendAddress string) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", frontendAddress)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// SetupSubrouters adds the API router to the parent router.
func (a *APISubLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// Create the ADK REST API handler
	apiHandler := adkrest.NewHandler(config, a.sseWriteTimeout)

	// Wrap it with CORS middleware
	corsHandler := corsWithArgs(a.frontendAddress)(apiHandler)

	// If prefix is empty, don't use PathPrefix("") because it's too greedy.
	// Instead, attach the handler to the main router directly.
	if a.pathPrefix == "" || a.pathPrefix == "/" {
		// This allows other routes (like /ui/) to match first if registered
		router.Methods("GET", "POST", "DELETE", "OPTIONS").Handler(corsHandler)
	} else {
		router.Methods("GET", "POST", "DELETE", "OPTIONS").
			PathPrefix(a.pathPrefix).
			Handler(http.StripPrefix(a.pathPrefix, corsHandler))
	}

	return nil
}
