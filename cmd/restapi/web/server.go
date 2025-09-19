package web

import (
	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/restapi/config"
	"google.golang.org/adk/cmd/restapi/handlers"
	"google.golang.org/adk/cmd/restapi/routers"
)

// func CorsWithArgs(routerConfig *config.ADKAPIRouterConfigs) func(next http.Handler) http.Handler {
// 	return func(next http.Handler) http.Handler {
// 		return routerConfig.Cors.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
// 			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
// 			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
// 			if r.Method == "OPTIONS" {
// 				w.WriteHeader(http.StatusOK)
// 				return
// 			}
// 			next.ServeHTTP(w, r)
// 		}))
// 	}
// }

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
	// router.Use(CorsWithArgs(routerConfig))
	return router
}
