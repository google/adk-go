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

// Package routers defines the HTTP routes for the ADK REST API.
package routers

import (
	"net/http"

	"github.com/gorilla/mux"
)

// A Route defines the parameters for an api endpoint
type Route struct {
	Name        string
	Methods     []string
	Pattern     string
	HandlerFunc http.HandlerFunc
}

// Routes is a list of defined api endpoints
type Routes []Route

// Router defines the required methods for retrieving api routes
type Router interface {
	Routes() Routes
}

// NewRouter creates a new router for any number of api routers
func NewRouter(routers ...Router) *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	SetupSubRouters(router)
	return router
}

// SetupSubRouters adds routes from subrouter to the naub router
func SetupSubRouters(router *mux.Router, subrouters ...Router) {
	for _, api := range subrouters {
		for _, route := range api.Routes() {
			var handler http.Handler = route.HandlerFunc

			router.
				Methods(route.Methods...).
				Path(route.Pattern).
				Name(route.Name).
				Handler(handler)
		}
	}
}
