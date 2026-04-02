// Copyright 2026 Google LLC
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

// TriggersAPIRouter defines the routes for the triggers APIs.
type TriggersAPIRouter struct {
	controller     *controllers.TriggersAPIController
	triggerSources []string
}

// NewTriggersAPIRouter creates a new TriggersAPIRouter.
func NewTriggersAPIRouter(controller *controllers.TriggersAPIController, triggerSources []string) *TriggersAPIRouter {
	return &TriggersAPIRouter{
		controller:     controller,
		triggerSources: triggerSources,
	}
}

// Routes returns the routes for the triggers API depending on configured triggerSources.
func (r *TriggersAPIRouter) Routes() Routes {
	var routes Routes

	for _, source := range r.triggerSources {
		switch source {
		case "pubsub":
			routes = append(routes, Route{
				Name:        "PubSubTrigger",
				Methods:     []string{http.MethodPost},
				Pattern:     "/apps/{app_name}/trigger/pubsub",
				HandlerFunc: r.controller.PubSubTriggerHandler,
			})
		}
	}

	return routes
}
