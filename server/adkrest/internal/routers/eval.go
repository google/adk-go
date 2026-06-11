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
	"google.golang.org/adk/server/restapi/handlers"
)

// EvalAPIRouter defines the routes for the Eval API.
type EvalAPIRouter struct {
	handler *handlers.EvalHandler
}

// NewEvalAPIRouter creates a new evaluation API router.
func NewEvalAPIRouter(handler *handlers.EvalHandler) *EvalAPIRouter {
	return &EvalAPIRouter{
		handler: handler,
	}
}

// Routes returns the routes for the Eval API.
func (r *EvalAPIRouter) Routes() Routes {
	// If no handler is configured, return unimplemented routes
	if r.handler == nil {
		return Routes{
			Route{
				Name:        "ListEvalSets",
				Methods:     []string{http.MethodGet},
				Pattern:     "/apps/{app_name}/eval_sets",
				HandlerFunc: controllers.Unimplemented,
			},
			Route{
				Name:        "CreateOrRunEvalSet",
				Methods:     []string{http.MethodPost, http.MethodOptions},
				Pattern:     "/apps/{app_name}/eval_sets/{eval_set_name}",
				HandlerFunc: controllers.Unimplemented,
			},
			Route{
				Name:        "ListEvalResults",
				Methods:     []string{http.MethodGet},
				Pattern:     "/apps/{app_name}/eval_results",
				HandlerFunc: controllers.Unimplemented,
			},
		}
	}

	// Return actual handler routes
	return Routes{
		Route{
			Name:        "ListEvalSets",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/eval_sets",
			HandlerFunc: r.handler.ListEvalSets,
		},
		Route{
			Name:        "CreateEvalSet",
			Methods:     []string{http.MethodPost},
			Pattern:     "/apps/{app_name}/eval_sets",
			HandlerFunc: r.handler.CreateEvalSet,
		},
		Route{
			Name:        "GetEvalSet",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/eval_sets/{eval_set_name}",
			HandlerFunc: r.handler.GetEvalSet,
		},
		Route{
			Name:        "RunEvalSet",
			Methods:     []string{http.MethodPost},
			Pattern:     "/apps/{app_name}/eval_sets/{eval_set_name}",
			HandlerFunc: r.handler.RunEvalSet,
		},
		Route{
			Name:        "DeleteEvalSet",
			Methods:     []string{http.MethodDelete},
			Pattern:     "/apps/{app_name}/eval_sets/{eval_set_name}",
			HandlerFunc: r.handler.DeleteEvalSet,
		},
		Route{
			Name:        "ListEvalResults",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/eval_results",
			HandlerFunc: r.handler.ListEvalResults,
		},
		Route{
			Name:        "GetEvalResult",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/eval_results/{result_id}",
			HandlerFunc: r.handler.GetEvalResult,
		},
	}
}
