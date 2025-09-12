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

package main

import (
	"log"
	"net/http"
	"strconv"

	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
	"google.golang.org/adk/cmd/restapi/utils"
	"google.golang.org/adk/sessionservice"
)

func corsWithArgs(serverArgs utils.AdkAPIArgs) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "http://"+serverArgs.FrontAddress)
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

func main() {
	serverArgs := utils.ParseArgs()

	log.Printf("Starting server on port %d with front address %s", serverArgs.Port, serverArgs.FrontAddress)

	router := routers.NewRouter(
		routers.NewSessionsApiRouter(handlers.NewSessionsApiController(sessionservice.Mem())),
		routers.NewRuntimeApiRouter(&handlers.RuntimeApiController{}),
		routers.NewAppsApiRouter(&handlers.AppsApiController{}),
		routers.NewDebugApiRouter(&handlers.DebugApiController{}),
		routers.NewArtifactsApiRouter(&handlers.ArtifactsApiController{}),
	)
	router.Use(corsWithArgs(serverArgs))

	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(serverArgs.Port), router))
}
