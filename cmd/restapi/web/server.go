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
	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
)

func SetupRouter(router *mux.Router, routerConfig *config.ADKAPIRouterConfigs) *mux.Router {
	return setupRouter(router, routerConfig,
		routers.NewSessionsAPIRouter(&handlers.SessionsAPIController{}),
		routers.NewRuntimeAPIRouter(&handlers.RuntimeAPIController{}),
		routers.NewAppsAPIRouter(&handlers.AppsAPIController{}),
		routers.NewDebugAPIRouter(&handlers.DebugAPIController{}),
		routers.NewArtifactsAPIRouter(&handlers.ArtifactsAPIController{}))
}

func setupRouter(router *mux.Router, routerConfig *config.ADKAPIRouterConfigs, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
