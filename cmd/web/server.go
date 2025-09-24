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
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/services"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
	"google.golang.org/adk/sessionservice"
)

// WebConfig is a struct with parameters to run a WebServer.
type WebConfig struct {
	LocalPort       int
	UIDistPath      string
	FrontendAddress string
	BackendAddress  string
	StartRestApi    bool
	StartWebUI      bool
}

// ParseArgs parses the arguments for the ADK API server.
func ParseArgs() *WebConfig {
<<<<<<< HEAD
	localPortFlag := flag.Int("port", 8080, "Port to listen on")
<<<<<<< HEAD
	frontendServerFlag := flag.String("front_address", "http://localhost:8001", "Front address to allow CORS requests from")
=======
	frontendServerFlag := flag.String("front_address", "localhost:8001", "Front address to allow CORS requests from as seen from the user browser")
	backendServerFlag := flag.String("backend_address", "http://localhost:8001/api", "Backend server as seen from the user browser")
>>>>>>> e7c16be (Added runtime generation of /assets/config/runtime-config.json)
=======
	localPortFlag := flag.Int("port", 8080, "Localhost port for the server")
<<<<<<< HEAD
	frontendServerFlag := flag.String("front_address", "localhost:8080", "Front address to allow CORS requests from as seen from the user browser. Please specify only hostname and (optionally) port")
	backendServerFlag := flag.String("backend_address", "http://localhost:8080/api", "Backend server as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'. ")
>>>>>>> 672ccad (Modified command line description)
	startRespApi := flag.Bool("start_restapi", true, "Set to start a rest api endpoint '/api'")
	startWebUI := flag.Bool("start_webui", true, "Set to start a web ui endpoint '/ui'")
	webuiDist := flag.String("webui_path", "",
		`Points to a static web ui dist path with the built version of ADK Web UI (cmd/web/distr/browser in the repo). 
=======
	frontendAddressFlag := flag.String("front_address", "localhost:8080", "Front address to allow CORS requests from as seen from the user browser. Please specify only hostname and (optionally) port")
	backendAddressFlag := flag.String("backend_address", "http://localhost:8080/api", "Backend server as seen from the user browser. Please specify the whole URL, i.e. 'http://localhost:8080/api'. ")
	startRespApiFlag := flag.Bool("start_restapi", true, "Set to start a rest api endpoint '/api'")
	startWebUIFlag := flag.Bool("start_webui", true, "Set to start a web ui endpoint '/ui'")
	webuiDistPathFlag := flag.String("webui_distr_path", "",
		`Points to a static web ui dist path with the built version of ADK Web UI (cmd/web/distr/browser in the ADK-GO repo). 
>>>>>>> a881883 (Modified the way the local server is run)
Normally it should be the version distributed with adk-go. You may use CLI command build webui to experiment with other versions.`)

	flag.Parse()
	if !flag.Parsed() {
		flag.Usage()
		panic("Failed to parse flags")
	}
	return &(WebConfig{
		LocalPort:       *localPortFlag,
		FrontendAddress: *frontendAddressFlag,
		BackendAddress:  *backendAddressFlag,
		StartRestApi:    *startRespApiFlag,
		StartWebUI:      *startWebUIFlag,
		UIDistPath:      *webuiDistPathFlag,
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

func corsWithArgs(c *WebConfig) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", c.FrontendAddress)
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

// Serve initiates the http server and starts it according to WebConfig parameters
func Serve(c *WebConfig, serveConfig *ServeConfig) {
	serverConfig := config.ADKAPIRouterConfigs{
		SessionService:  serveConfig.SessionService,
		AgentLoader:     serveConfig.AgentLoader,
		ArtifactService: serveConfig.ArtifactService,
	}
	serverConfig.Cors = *cors.New(cors.Options{
		AllowedOrigins:   []string{c.FrontendAddress},
		AllowedMethods:   []string{http.MethodGet, http.MethodPost, http.MethodOptions, http.MethodDelete, http.MethodPut},
		AllowCredentials: true})

	rBase := mux.NewRouter().StrictSlash(true)
	rBase.Use(Logger)

	if c.StartWebUI {

		rUi := rBase.Methods("GET").PathPrefix("/ui/").Subrouter()

		// generate runtime-config in the runtime
		runtimeConfigResponse := struct {
			BackendUrl string `json:"backendUrl"`
		}{BackendUrl: c.BackendAddress}
		rUi.Methods("GET").Path("/assets/config/runtime-config.json").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			handlers.EncodeJSONResponse(runtimeConfigResponse, http.StatusOK, w)
		})

		rUi.Methods("GET").Handler(http.StripPrefix("/ui/", http.FileServer(http.Dir(c.UIDistPath))))
	}

	if c.StartRestApi {
<<<<<<< HEAD
		rApi := rBase.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix("/api/").Subrouter()
		rApi.Use(serverConfig.Cors.Handler)
=======
		rApi := rBase.Methods("GET", "POST", "DELETE").PathPrefix("/api/").Subrouter()
		rApi.Use(corsWithArgs(c))
		// rApi= serverConfig.Cors.Handler(rApi)
		// rApi = serverConfig.Cors.Handler(rApi)
>>>>>>> e7c16be (Added runtime generation of /assets/config/runtime-config.json)
		restapiweb.SetupRouter(rApi, &serverConfig)
	}

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(c.LocalPort), rBase))
	// log.Fatal(http.ListenAndServe(":"+strconv.Itoa(c.LocalPort), serverConfig.Cors.Handler(rBase)))
}
