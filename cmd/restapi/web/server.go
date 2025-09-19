package web

import (
	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
)

func SetupRouter(router *mux.Router, routerConfig *config.ADKAPIRouterConfigs) *mux.Router {
	return setupRouter(router, routerConfig,
		routers.NewSessionsAPIRouter(&handlers.SessionsAPIController{}),
		routers.NewRuntimeAPIRouter(&handlers.RuntimeAPIController{}),
		routers.NewAppsAPIRouter(&handlers.AppsAPIController{}),
		routers.NewDebugAPIRouter(&handlers.DebugAPIController{}),
		routers.NewArtifactsAPIRouter(&handlers.ArtifactsAPIController{}))
}

func setupRouter(router *mux.Router, routerConfig *config.ADKAPIRouterConfigs, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}
