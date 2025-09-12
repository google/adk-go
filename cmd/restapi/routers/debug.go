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

	"google.golang.org/adk/cmd/restapi/handlers"
)

type DebugApiRouter struct {
	runtimeController *handlers.DebugApiController
}

func NewDebugApiRouter(controller *handlers.DebugApiController) *DebugApiRouter {
	return &DebugApiRouter{runtimeController: controller}

}

func (r *DebugApiRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetTraceDict",
			Method:      http.MethodGet,
			Pattern:     "/debug/trace/{event_id}",
			HandlerFunc: r.runtimeController.TraceDict,
		},
		Route{
			Name:        "GetEventGraph",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
			HandlerFunc: r.runtimeController.EventGraph,
		},
	}
}
