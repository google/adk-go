package routers

import (
	"net/http"

	"google.golang.org/adk/cmd/restapi/handlers"
)

type SessionsApiRouter struct {
	sessionController *handlers.SessionsApiController
}

func NewSessionsApiRouter(controller *handlers.SessionsApiController) *SessionsApiRouter {
	return &SessionsApiRouter{sessionController: controller}
}

func (r *SessionsApiRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetSession",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.GetSession,
		},
		Route{
			Name:        "CreateSession",
			Method:      http.MethodPost,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.CreateSession,
		},
		Route{
			Name:        "CreateSessionWithId",
			Method:      http.MethodPost,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.CreateSession,
		},
		Route{
			Name:        "DeleteSession",
			Method:      http.MethodDelete,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			HandlerFunc: r.sessionController.CreateSession,
		},
		Route{
			Name:        "ListSessions",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions",
			HandlerFunc: r.sessionController.ListSessions,
		},
	}
}
