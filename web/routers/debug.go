package routers

import (
	"net/http"

	"google.golang.org/adk/web/handlers"
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
