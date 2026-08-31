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

package routers

import (
	"net/http"

	"google.golang.org/adk/v2/server/adkrest/controllers"
)

// VersionAPIRouter defines the route for the version endpoint.
type VersionAPIRouter struct {
	versionController *controllers.VersionAPIController
}

// NewVersionAPIRouter creates a new VersionAPIRouter.
func NewVersionAPIRouter(controller *controllers.VersionAPIController) *VersionAPIRouter {
	return &VersionAPIRouter{versionController: controller}
}

// Routes returns the routes for the version API.
func (r *VersionAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetVersion",
			Methods:     []string{http.MethodGet},
			Pattern:     "/version",
			HandlerFunc: r.versionController.VersionHandler,
		},
	}
}
