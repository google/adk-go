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

	"google.golang.org/adk/cmd/restapi/errors"
	"google.golang.org/adk/cmd/restapi/handlers"
)

type RuntimeApiRouter struct {
	runtimeController *handlers.RuntimeApiController
}

func NewRuntimeApiRouter(controller *handlers.RuntimeApiController) *RuntimeApiRouter {
	return &RuntimeApiRouter{runtimeController: controller}

}

func (r *RuntimeApiRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "RunAgent",
			Method:      http.MethodPost,
			Pattern:     "/run",
			HandlerFunc: errors.FromErrorHandler(r.runtimeController.RunAgent),
		},
		Route{
			Name:        "RunAgentSse",
			Method:      http.MethodPost,
			Pattern:     "/run_sse",
			HandlerFunc: errors.FromErrorHandler(r.runtimeController.RunAgentSse),
		},
		Route{
			Name:        "RunAgentSseOptions",
			Method:      http.MethodOptions,
			Pattern:     "/run_sse",
			HandlerFunc: errors.FromErrorHandler(r.runtimeController.RunAgentSse),
		},
	}
}
