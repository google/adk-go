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

// SessionsAPIRouter defines the routes for the Sessions API.
type SessionsAPIRouter struct {
	sessionController *handlers.SessionsAPIController
}

// NewSessionsAPIRouter creates a new SessionsAPIRouter.
func NewSessionsAPIRouter(controller *handlers.SessionsAPIController) *SessionsAPIRouter {
	return &SessionsAPIRouter{sessionController: controller}
}

// Routes returns the routes for the Sessions API.
func (r *SessionsAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetSession",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.GetSessionHTTP,
		},
		Route{
			Name:        "CreateSession",
			Method:      http.MethodPost,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.CreateSessionHTTP,
		},
		Route{
			Name:        "CreateSessionWithId",
			Method:      http.MethodPost,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.CreateSessionHTTP,
		},
		Route{
			Name:        "DeleteSession",
			Method:      http.MethodDelete,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.DeleteSessionHTTP,
		},
		Route{
			Name:        "ListSessions",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.ListSessionsHTTP,
		},
	}
}
