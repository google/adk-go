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

// Package api provides a sublauncher that adds ADK REST API to the web server (using url /api/)
package api

import (
	"flag"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/adkrest"
)

// apiConfig contains parametres for lauching ADK REST API
type apiConfig struct {
	frontendAddress string
}

// apiLauncher can launch ADK REST API
type apiLauncher struct {
	flags  *flag.FlagSet
	config *apiConfig
}

// CommandLineSyntax returns the command-line syntax for the API launcher.
func (a *apiLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
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

// UserMessage implements web.Sublauncher. Prints message to the user
func (a *apiLauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       api:  you can access API using %s/api", webURL))
	printer(fmt.Sprintf("       api:      for instance: %s/api/list-apps", webURL))
}

// SetupSubrouters adds the API router to the parent router.
func (a *apiLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// Create the ADK REST API handler
	apiHandler := adkrest.NewHandler(config)

	// Wrap it with CORS middleware
	corsHandler := corsWithArgs(a.config.frontendAddress)(apiHandler)

	// Register it at the /api/ path
	router.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix("/api/").Handler(
		http.StripPrefix("/api", corsHandler),
	)

	return nil
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for API launcher.
func (a *apiLauncher) Keyword() string {
	return "api"
}

// Parse parses the command-line arguments for the API launcher.
func (a *apiLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse api flags: %v", err)
	}
	restArgs := a.flags.Args()
	return restArgs, nil
}

// SimpleDescription implements web.Sublauncher. Returns a simple description of the API launcher.
func (a *apiLauncher) SimpleDescription() string {
	return "starts ADK REST API server, accepting origins specified by webui_address (CORS)"
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher() weblauncher.Sublauncher {
	config := &apiConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.frontendAddress, "webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")

	return &apiLauncher{
		config: config,
		flags:  fs,
	}
}
