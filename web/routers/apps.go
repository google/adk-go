package routers

import (
	"net/http"

	"google.golang.org/adk/web/handlers"
)

type AppsApiRouter struct {
	appsController *handlers.AppsApiController
}

func NewAppsApiRouter(controller *handlers.AppsApiController) *AppsApiRouter {
	return &AppsApiRouter{appsController: controller}

}

func (r *AppsApiRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "ListApps",
			Method:      http.MethodGet,
			Pattern:     "/apps",
			HandlerFunc: r.appsController.ListApps,
		},
	}
}
