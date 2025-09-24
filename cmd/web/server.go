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

// package web provides an ability to parse command line flags and easily run server for both ADK WEB UI and ADK REST API
package web

import (
	"flag"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
	"google.golang.org/adk/artifactservice"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/services"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
	"google.golang.org/adk/sessionservice"
)

// WebConfig is a struct with parameters to run a WebServer.
type WebConfig struct {
	LocalPort      int
	UIDistPath     string
	FrontEndServer string
	StartRestApi   bool
	StartWebUI     bool
}

// ParseArgs parses the arguments for the ADK API server.
func ParseArgs() *WebConfig {
	localPortFlag := flag.Int("port", 8080, "Port to listen on")
	frontendServerFlag := flag.String("front_address", "http://localhost:8001", "Front address to allow CORS requests from")
	startRespApi := flag.Bool("start_restapi", true, "Set to start a rest api endpoint '/api'")
	startWebUI := flag.Bool("start_webui", true, "Set to start a web ui endpoint '/ui'")
	webuiDist := flag.String("webui_path", "", "Points to a static web ui dist path with the built version of ADK Web UI")

	flag.Parse()
	if !flag.Parsed() {
		flag.Usage()
		panic("Failed to parse flags")
	}
	return &(WebConfig{
		LocalPort:      *localPortFlag,
		FrontEndServer: *frontendServerFlag,
		StartRestApi:   *startRespApi,
		StartWebUI:     *startWebUI,
		UIDistPath:     *webuiDist,
	})
}

func Logger(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		inner.ServeHTTP(w, r)

		log.Printf(
			"%s %s %s",
			r.Method,
			r.RequestURI,
			time.Since(start),
		)
	})
}

type ServeConfig struct {
	SessionService  sessionservice.Service
	AgentLoader     services.AgentLoader
	ArtifactService artifactservice.Service
}

// Serve initiates the http server and starts it according to WebConfig parameters
func Serve(c *WebConfig, serveConfig *ServeConfig) {
	serverConfig := config.ADKAPIRouterConfigs{
		SessionService:  serveConfig.SessionService,
		AgentLoader:     serveConfig.AgentLoader,
		ArtifactService: serveConfig.ArtifactService,
	}
	serverConfig.Cors = *cors.New(cors.Options{
		AllowedOrigins:   []string{c.FrontEndServer},
		AllowedMethods:   []string{http.MethodGet, http.MethodPost, http.MethodOptions, http.MethodDelete, http.MethodPut},
		AllowCredentials: true})

	rBase := mux.NewRouter().StrictSlash(true)
	rBase.Use(Logger)

	if c.StartWebUI {
		rUi := rBase.Methods("GET").PathPrefix("/ui/").Subrouter()
		rUi.Methods("GET").Handler(http.StripPrefix("/ui/", http.FileServer(http.Dir(c.UIDistPath))))
	}

	if c.StartRestApi {
		rApi := rBase.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix("/api/").Subrouter()
		rApi.Use(serverConfig.Cors.Handler)
		restapiweb.SetupRouter(rApi, &serverConfig)
	}

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(c.LocalPort), rBase))
}
