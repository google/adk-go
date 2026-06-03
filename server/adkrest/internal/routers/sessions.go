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

// SessionsAPIRouter defines the routes for the Sessions API.
type SessionsAPIRouter struct {
	sessionController *controllers.SessionsAPIController
}

// NewSessionsAPIRouter creates a new SessionsAPIRouter.
func NewSessionsAPIRouter(controller *controllers.SessionsAPIController) *SessionsAPIRouter {
	return &SessionsAPIRouter{sessionController: controller}
}

// Routes returns the routes for the Sessions API.
func (r *SessionsAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetSession",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.GetSessionHandler,
		},
		Route{
			Name:        "CreateSession",
			Methods:     []string{http.MethodPost},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.CreateSessionHandler,
		},
		Route{
			Name:        "CreateSessionWithId",
			Methods:     []string{http.MethodPost},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.CreateSessionHandler,
		},
		Route{
			Name:        "DeleteSession",
			Methods:     []string{http.MethodDelete, http.MethodOptions},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.DeleteSessionHandler,
		},
		Route{
			Name:        "ListSessions",
			Methods:     []string{http.MethodGet},
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.ListSessionsHandler,
		},
	}
}
