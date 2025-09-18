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

// package routers defines the HTTP routes for the ADK-Web REST API.
package routers

import (
	"log"
	"net/http"
	"time"

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

func Logger(inner http.Handler, name string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		inner.ServeHTTP(w, r)

		log.Printf(
			"%s %s %s %s",
			r.Method,
			r.RequestURI,
			name,
			time.Since(start),
		)
		log.Printf("Request Header: %v", r.Header)
		log.Printf("Request Body: %v", r.Body)
		log.Printf("Response Header: %v", w.Header())
		log.Printf("Response: %v", w)
	})
}

// NewRouter creates a new router for any number of api routers
func NewRouter(routers ...Router) *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	for _, api := range routers {
		for _, route := range api.Routes() {
			var handler http.Handler = route.HandlerFunc

			// handler = Logger(handler, route.Name)

			router.
				Methods(route.Methods...).
				Path(route.Pattern).
				Name(route.Name).
				Handler(handler)
		}
	}
	return router
}
