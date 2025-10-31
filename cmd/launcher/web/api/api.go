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

// package api allows to run ADK REST API alone
package api

import (
	"flag"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher/adk"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
	"google.golang.org/adk/internal/cli/util"
)

// apiConfig contains parametres for lauching ADK REST API
type apiConfig struct {
	frontendAddress string
}

// ApiLauncher can launch ADK REST API
type ApiLauncher struct {
	flags  *flag.FlagSet
	config *apiConfig
}

// CommandLineSyntax implements web.WebSublauncher.
func (a *ApiLauncher) CommandLineSyntax() string {
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

func (a *ApiLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       api:  you can access API using %s/api", webUrl))
	printer(fmt.Sprintf("       api:      for instance: %s/api/list-apps", webUrl))
}

// SetupSubrouters adds api router to the parent router
func (a *ApiLauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	rApi := router.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix("/api/").Subrouter()
	restapiweb.SetupRouter(rApi, adkConfig)
	rApi.Use(corsWithArgs(a.config.frontendAddress))
}

func (a *ApiLauncher) WrapHandlers(handler http.Handler, adkConfig *adk.Config) http.Handler {
	// api doesn't change the top level routes
	return handler
}

func (a *ApiLauncher) Keyword() string {
	return "api"
}

func (a *ApiLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse api flags: %v", err)
	}
	restArgs := a.flags.Args()
	return restArgs, nil
}

func (a *ApiLauncher) SimpleDescription() string {
	return "starts ADK REST API server, accepting origins specified by webui_address (CORS)"
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher() weblauncher.WebSublauncher {
	config := &apiConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.frontendAddress, "webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")

	return &ApiLauncher{
		config: config,
		flags:  fs,
	}
}
