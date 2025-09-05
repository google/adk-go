package web

import (
	"strings"
)

type SessionsApiController struct {}

func (*SessionsApiController) Routes(){

// Routes returns all the api routes for the DefaultAPIController
func (c *DefaultAPIController) Routes() Routes {
	return Routes{
		"ListAppsListAppsGet": Route{
			"ListAppsListAppsGet",
			strings.ToUpper("Get"),
			"/list-apps",
			c.ListAppsListAppsGet,
		},
		"GetSessionAppsAppNameUsersUserIdSessionsSessionIdGet": Route{
			"GetSessionAppsAppNameUsersUserIdSessionsSessionIdGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			c.GetSessionAppsAppNameUsersUserIdSessionsSessionIdGet,
		},
		"CreateSessionWithIdAppsAppNameUsersUserIdSessionsSessionIdPost": Route{
			"CreateSessionWithIdAppsAppNameUsersUserIdSessionsSessionIdPost",
			strings.ToUpper("Post"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			c.CreateSessionWithIdAppsAppNameUsersUserIdSessionsSessionIdPost,
		},
		"DeleteSessionAppsAppNameUsersUserIdSessionsSessionIdDelete": Route{
			"DeleteSessionAppsAppNameUsersUserIdSessionsSessionIdDelete",
			strings.ToUpper("Delete"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}",
			c.DeleteSessionAppsAppNameUsersUserIdSessionsSessionIdDelete,
		},
		"ListSessionsAppsAppNameUsersUserIdSessionsGet": Route{
			"ListSessionsAppsAppNameUsersUserIdSessionsGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions",
			c.ListSessionsAppsAppNameUsersUserIdSessionsGet,
		},
		"CreateSessionAppsAppNameUsersUserIdSessionsPost": Route{
			"CreateSessionAppsAppNameUsersUserIdSessionsPost",
			strings.ToUpper("Post"),
			"/apps/{app_name}/users/{user_id}/sessions",
			c.CreateSessionAppsAppNameUsersUserIdSessionsPost,
		},
		"LoadArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameGet": Route{
			"LoadArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			c.LoadArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameGet,
		},
		"DeleteArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameDelete": Route{
			"DeleteArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameDelete",
			strings.ToUpper("Delete"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
			c.DeleteArtifactAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameDelete,
		},
		"LoadArtifactVersionAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsVersionIdGet": Route{
			"LoadArtifactVersionAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsVersionIdGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}",
			c.LoadArtifactVersionAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsVersionIdGet,
		},
		"ListArtifactNamesAppsAppNameUsersUserIdSessionsSessionIdArtifactsGet": Route{
			"ListArtifactNamesAppsAppNameUsersUserIdSessionsSessionIdArtifactsGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
			c.ListArtifactNamesAppsAppNameUsersUserIdSessionsSessionIdArtifactsGet,
		},
		"ListArtifactVersionsAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsGet": Route{
			"ListArtifactVersionsAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsGet",
			strings.ToUpper("Get"),
			"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
			c.ListArtifactVersionsAppsAppNameUsersUserIdSessionsSessionIdArtifactsArtifactNameVersionsGet,
		},
		// "RunAgentRunPost": Route{
		// 	"RunAgentRunPost",
		// 	strings.ToUpper("Post"),
		// 	"/run",
		// 	c.RunAgentRunPost,
		// },
		"RunAgentSseRunSsePost": Route{
			"RunAgentSseRunSsePost",
			strings.ToUpper("Post"),
			"/run_sse",
			c.RunAgentSseRunSsePost,
		},
		"RunAgentSseRunSseOptions": Route{
			"RunAgentSseRunSseOptions",
			strings.ToUpper("Options"),
			"/run_sse",
			c.RunAgentSseRunSsePost,
		},
		"BuilderBuildBuilderSavePost": Route{
			"BuilderBuildBuilderSavePost",
			strings.ToUpper("Post"),
			"/builder/save",
			c.BuilderBuildBuilderSavePost,
		},
		"GetAgentBuilderBuilderAppAppNameGet": Route{
			"GetAgentBuilderBuilderAppAppNameGet",
			strings.ToUpper("Get"),
			"/builder/app/{app_name}",
			c.GetAgentBuilderBuilderAppAppNameGet,
		},
	}
}
}