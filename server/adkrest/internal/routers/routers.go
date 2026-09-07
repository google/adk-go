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
	"strings"

	"github.com/gorilla/mux"
)

// DevPrefix is the path prefix ADK v2 clients use for developer-only
// endpoints: evaluation, tracing, the agent builder and the tests tab.
//
// ADK split its API in two at v2. Production endpoints keep their original
// paths, while everything a developer tool needs moved under
// /dev/apps/{app_name}/. The ADK web UI has addressed these endpoints this way
// since then, so a route reachable only at the pre-v2 path is unreachable from
// the UI.
const DevPrefix = "/dev/apps/{app_name}"

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

// SetupSubRouters adds routes from subrouter to the main router.
//
// It also takes over the router's two fallbacks and its trailing-slash
// handling, because both need to know the routes registered here. See
// [newFallbackHandler].
func SetupSubRouters(router *mux.Router, subrouters ...Router) {
	// StrictSlash answers a trailing-slash URL with a redirect to the same
	// path without one. A client re-issues a redirected POST as a GET, so
	// POST /apps/x/users/y/sessions/ came back as a session list from the GET
	// handler on the same path, with the body discarded and no session
	// created. Matching without it and normalising the path in the fallback
	// instead runs the intended handler directly, method and body intact. It
	// is safe because no route below is registered with a trailing slash, so
	// none needs one to match.
	//
	// This overrides the setting on the router passed in, and mux offers no
	// way to read the previous value back, so it cannot be restored. Routes
	// registered on that router before this call keep whatever it was.
	router.StrictSlash(false)
	for _, api := range subrouters {
		for _, route := range api.Routes() {
			var handler http.Handler = route.HandlerFunc

			router.
				Methods(withHead(route.Methods)...).
				Path(route.Pattern).
				Name(route.Name).
				Handler(handler)
		}
	}
	fallback := newFallbackHandler(router)
	router.MethodNotAllowedHandler = fallback
	router.NotFoundHandler = fallback
}

// withHead adds HEAD wherever a route serves GET.
//
// net/http answers a HEAD by running the GET handler and discarding the body,
// so every GET route already serves HEAD correctly. Without this, gorilla/mux
// rejects the request before the handler runs and monitoring, link checkers and
// caches all see 405. Doing it here rather than in each Routes() literal means
// a new route cannot forget.
func withHead(methods []string) []string {
	for _, m := range methods {
		if m == http.MethodHead {
			return methods
		}
	}
	for _, m := range methods {
		if m == http.MethodGet {
			return append(append([]string{}, methods...), http.MethodHead)
		}
	}
	return methods
}

// probeMethods are the verbs a request is re-matched against when the router
// did not serve it, in the order they appear in an Allow header.
var probeMethods = []string{
	http.MethodGet, http.MethodHead, http.MethodPost,
	http.MethodPut, http.MethodPatch, http.MethodDelete,
	http.MethodOptions,
}

// newFallbackHandler answers every request no route served, as both the
// NotFoundHandler and the MethodNotAllowedHandler.
//
// It has to be both, because gorilla/mux cannot be trusted to tell the two
// apart. mux records a method mismatch on the route that matched the path, then
// clears it again in Route.Match as soon as any later route's method matcher
// succeeds — whatever path that route is on. Almost any verb some other route
// accepts therefore wipes the mismatch, and the request arrives at the
// NotFoundHandler: DELETE /version returned a bare 404 while PATCH /version
// returned 405, purely because the one PATCH route is registered earlier.
// Deciding here, from the routing table rather than from mux's verdict, makes
// the answer independent of registration order.
//
// The order of the three steps matters. A trailing slash is normalised first,
// since the path with the slash matches nothing and would otherwise be reported
// as a 404 for a resource that exists.
func newFallbackHandler(router *mux.Router) http.Handler {
	return http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		// Re-dispatch, rather than answer here, so the request reaches its real
		// handler with its method and body intact. This recurses at most once:
		// the rewritten path carries no trailing slash, so it cannot take this
		// branch again.
		if trimmed, ok := withoutTrailingSlash(req); ok {
			router.ServeHTTP(rw, trimmed)
			return
		}
		// Probing calls Match, which never runs a handler, so it cannot
		// re-enter this one.
		if allowed := allowedMethods(router, req); len(allowed) > 0 {
			// No route declares OPTIONS: preflight belongs in middleware, and
			// a route that answered it would run a mutating handler on a
			// preflight. But a resource that exists still has to describe
			// itself, so answer it here rather than reporting 405 while
			// listing every verb except the one that was asked for.
			if req.Method == http.MethodOptions {
				rw.Header().Set("Allow", strings.Join(append(allowed, http.MethodOptions), ", "))
				rw.WriteHeader(http.StatusNoContent)
				return
			}
			// The Allow header RFC 9110 requires on a 405. gorilla/mux omits
			// it, which leaves a client unable to discover what the resource
			// does support.
			rw.Header().Set("Allow", strings.Join(allowed, ", "))
			rw.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		http.NotFound(rw, req)
	})
}

// allowedMethods returns every method in [probeMethods] that the router serves
// on the request's path, by re-matching the request once per method.
func allowedMethods(router *mux.Router, req *http.Request) []string {
	var allowed []string
	for _, method := range probeMethods {
		probe := req.Clone(req.Context())
		probe.Method = method
		var match mux.RouteMatch
		// Match reports true when it fell back to the NotFoundHandler, so the
		// error has to be checked as well as the result.
		if router.Match(probe, &match) && match.MatchErr == nil {
			allowed = append(allowed, method)
		}
	}
	return allowed
}

// withoutTrailingSlash copies req with any trailing slashes removed from its
// path, reporting false when there is nothing to remove or nothing would be
// left.
func withoutTrailingSlash(req *http.Request) (*http.Request, bool) {
	// Count the slashes on the escaped path so an encoded %2F inside a path
	// variable is left alone.
	escaped := req.URL.EscapedPath()
	n := len(escaped) - len(strings.TrimRight(escaped, "/"))
	if n == 0 || n == len(escaped) {
		// Either no trailing slash, or a path that is nothing but slashes, in
		// which case trimming would leave no path to match.
		return nil, false
	}
	trimmed := req.Clone(req.Context())
	trimmed.URL.Path = req.URL.Path[:len(req.URL.Path)-n]
	if trimmed.URL.RawPath != "" {
		trimmed.URL.RawPath = req.URL.RawPath[:len(req.URL.RawPath)-n]
	}
	return trimmed, true
}
