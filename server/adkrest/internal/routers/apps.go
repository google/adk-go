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

	"google.golang.org/adk/server/adkrest/controllers"
)

// AppsAPIRouter defines the routes for the Apps API.
type AppsAPIRouter struct {
	appsController *controllers.AppsAPIController
}

// NewAppsAPIRouter creates a new AppsAPIRouter.
func NewAppsAPIRouter(controller *controllers.AppsAPIController) *AppsAPIRouter {
	return &AppsAPIRouter{appsController: controller}
}

// Routes returns the routes for the Apps API.
func (r *AppsAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "ListApps",
			Methods:     []string{http.MethodGet},
			Pattern:     "/list-apps",
			HandlerFunc: r.appsController.ListAppsHandler,
		},
	}
}
