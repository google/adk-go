package routers

import (
	"net/http"

	"google.golang.org/adk/web/handlers"
)

type ArtifactsApiRouter struct {
	artifactsController *handlers.ArtifactsApiController
}

func NewArtifactsApiRouter(controller *handlers.ArtifactsApiController) *ArtifactsApiRouter {
	return &ArtifactsApiRouter{artifactsController: controller}
}

func (r *ArtifactsApiRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "ListArtifacts",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
			HandlerFunc: r.artifactsController.ListArtifacts,
		},
		Route{
			Name:        "LoadArtifact",
			Method:      http.MethodGet,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			HandlerFunc: r.artifactsController.LoadArtifact,
		},
		Route{
			Name:        "DeleteSession",
			Method:      http.MethodDelete,
			Pattern:     "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			HandlerFunc: r.artifactsController.DeleteArtifact,
		},
	}
}
