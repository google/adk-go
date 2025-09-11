package routers

import (
	"net/http"

	"google.golang.org/adk/web/errors"
	"google.golang.org/adk/web/handlers"
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
			HandlerFunc: errors.FromErrorHandler(r.runtimeController.RunAgent),
		},
		Route{
			Name:        "RunAgentSse",
			Method:      http.MethodOptions,
			Pattern:     "/run_sse",
			HandlerFunc: errors.FromErrorHandler(r.runtimeController.RunAgent),
		},
	}
}
