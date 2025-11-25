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

// ArtifactsAPIRouter defines the routes for the Artifacts API.
type ArtifactsAPIRouter struct {
	artifactsController *controllers.ArtifactsAPIController
}

// NewArtifactsAPIRouter creates a new ArtifactsAPIRouter.
func NewArtifactsAPIRouter(controller *controllers.ArtifactsAPIController) *ArtifactsAPIRouter {
	return &ArtifactsAPIRouter{artifactsController: controller}
}

// Routes returns the routes for the Artifacts API.
func (r *ArtifactsAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "ListArtifacts",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
			HandlerFunc: r.artifactsController.ListArtifactsHandler,
		},
		Route{
			Name:        "LoadArtifact",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			HandlerFunc: r.artifactsController.LoadArtifactHandler,
		},
		Route{
			Name:        "LoadArtifact",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version}",
			HandlerFunc: r.artifactsController.LoadArtifactVersionHandler,
		},
		Route{
			Name:        "DeleteArtifact",
			Methods:     []string{http.MethodDelete, http.MethodOptions},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			HandlerFunc: r.artifactsController.DeleteArtifactHandler,
		},
	}
}
