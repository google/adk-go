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

	"google.golang.org/adk/v2/server/adkrest/controllers"
)

// AgentGraphAPIRouter defines the routes behind the web UI's agent-structure
// panel. They return the agent tree, and the DOT source names every tool the
// agent can call, so they are registered only under IncludeDebugAPI.
type AgentGraphAPIRouter struct {
	graphController *controllers.AgentGraphAPIController
}

// NewAgentGraphAPIRouter creates a new AgentGraphAPIRouter.
func NewAgentGraphAPIRouter(controller *controllers.AgentGraphAPIController) *AgentGraphAPIRouter {
	return &AgentGraphAPIRouter{graphController: controller}
}

// Routes returns the two implemented agent-graph endpoints.
func (r *AgentGraphAPIRouter) Routes() Routes {
	return Routes{
		Route{
			Name:        "GetAppInfo",
			Methods:     []string{http.MethodGet},
			Pattern:     DevPrefix + "/build_graph",
			HandlerFunc: r.graphController.BuildGraphHandler,
		},
		Route{
			Name:        "GetAgentGraphImage",
			Methods:     []string{http.MethodGet},
			Pattern:     DevPrefix + "/build_graph_image",
			HandlerFunc: r.graphController.BuildGraphImageHandler,
		},
	}
}

const builderDetail = "ADK Go agents are defined in Go code, so there is no server-side agent configuration to edit."

// AgentBuilderAPIRouter defines the web UI's agent builder routes.
//
// These are registered unconditionally. They read no agent and disclose
// nothing: the GET answers an empty 200, which is how the UI knows to disable
// its builder toggle, and the writes answer 501.
type AgentBuilderAPIRouter struct{}

// Routes returns the routes for the agent builder API.
//
// The write endpoints are not implemented: they would edit an agent's stored
// configuration, and ADK Go has none.
func (r *AgentBuilderAPIRouter) Routes() Routes {
	notImplemented := controllers.NewNotImplementedHandler("the agent builder", builderDetail)

	return Routes{
		Route{
			Name:        "GetAgentBuilderConfig",
			Methods:     []string{http.MethodGet},
			Pattern:     DevPrefix + "/builder",
			HandlerFunc: controllers.AgentBuilderConfigHandler,
		},
		Route{
			Name:        "SaveAgentBuilderConfig",
			Methods:     []string{http.MethodPost},
			Pattern:     DevPrefix + "/builder/save",
			HandlerFunc: notImplemented,
		},
		Route{
			Name:        "CancelAgentBuilderChanges",
			Methods:     []string{http.MethodPost},
			Pattern:     DevPrefix + "/builder/cancel",
			HandlerFunc: notImplemented,
		},
	}
}

const testsDetail = "ADK Go does not implement the web UI tests tab. Write Go tests against runner.Run instead."

// TestsAPIRouter defines the routes for the web UI's tests tab, which ADK Go
// does not implement.
type TestsAPIRouter struct{}

// Routes returns the routes for the Tests API.
//
// The two literal paths come before {test_name} so they are not read as the
// name of a test.
func (r *TestsAPIRouter) Routes() Routes {
	notImplemented := controllers.NewNotImplementedHandler("the tests tab", testsDetail)

	testRoutes := []struct {
		name    string
		methods []string
		pattern string
	}{
		{"ListTests", []string{http.MethodGet}, "/tests"},
		{"RebuildTests", []string{http.MethodPost}, "/tests/rebuild"},
		{"RunTests", []string{http.MethodPost}, "/tests/run"},
		{"GetTest", []string{http.MethodGet}, "/tests/{test_name}"},
		{"CreateTest", []string{http.MethodPut}, "/tests/{test_name}"},
		{"DeleteTest", []string{http.MethodDelete}, "/tests/{test_name}"},
	}

	routes := make(Routes, 0, len(testRoutes))
	for _, t := range testRoutes {
		routes = append(routes, Route{
			Name:        t.name,
			Methods:     t.methods,
			Pattern:     DevPrefix + t.pattern,
			HandlerFunc: notImplemented,
		})
	}
	return routes
}
