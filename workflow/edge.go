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

package workflow

import "fmt"

// RouteKind tags the underlying type carried by a Route value.
type RouteKind uint8

const (
	routeKindNone RouteKind = iota
	RouteKindString
	RouteKindInt
	RouteKindBool
	RouteKindDefault
)

// Route is a typed value emitted by a node to drive conditional edges.
// Equivalent to adk-python's RouteValue (TypeAlias[bool|int|str]) with the
// addition of a sentinel DefaultRoute.
//
// Construct via the typed helpers (RouteString, RouteInt, RouteBool) or
// reference DefaultRoute as the fallback. Two Routes are equal iff their
// kinds match and their underlying values match.
type Route struct {
	kind RouteKind
	s    string
	i    int
	b    bool
}

// RouteString constructs a string-valued Route.
func RouteString(s string) Route { return Route{kind: RouteKindString, s: s} }

// RouteInt constructs an int-valued Route.
func RouteInt(i int) Route { return Route{kind: RouteKindInt, i: i} }

// RouteBool constructs a bool-valued Route.
func RouteBool(b bool) Route { return Route{kind: RouteKindBool, b: b} }

// DefaultRoute is the sentinel that matches an edge marked as the default
// branch when no other route value matched.
var DefaultRoute = Route{kind: RouteKindDefault}

// Kind reports the underlying type of the route.
func (r Route) Kind() RouteKind { return r.kind }

// String returns a printable representation of the route.
func (r Route) String() string {
	switch r.kind {
	case RouteKindString:
		return fmt.Sprintf("Route(%q)", r.s)
	case RouteKindInt:
		return fmt.Sprintf("Route(%d)", r.i)
	case RouteKindBool:
		return fmt.Sprintf("Route(%t)", r.b)
	case RouteKindDefault:
		return "Route(__DEFAULT__)"
	default:
		return "Route(<unset>)"
	}
}

// Match reports whether two routes are equal under the documented rules.
func (r Route) Match(o Route) bool {
	if r.kind != o.kind {
		return false
	}
	switch r.kind {
	case RouteKindString:
		return r.s == o.s
	case RouteKindInt:
		return r.i == o.i
	case RouteKindBool:
		return r.b == o.b
	case RouteKindDefault:
		return true
	}
	return false
}

// IsZero reports whether r was never constructed (the zero value).
func (r Route) IsZero() bool { return r.kind == routeKindNone }

// Edge connects two nodes. An empty Routes slice means the edge is
// unconditional. When Routes is non-empty, the edge fires only when the
// upstream node emits a matching route (or DefaultRoute for the catch-all).
type Edge struct {
	From, To Node
	Routes   []Route
}

// Connect is the convenience constructor for an Edge. Pass zero or more
// Route values to mark the edge conditional.
//
//	workflow.Connect(start, classifier)                         // unconditional
//	workflow.Connect(classifier, branchA, RouteString("a"))     // routed
//	workflow.Connect(classifier, fallback, DefaultRoute)        // default branch
func Connect(from, to Node, routes ...Route) Edge {
	return Edge{From: from, To: to, Routes: routes}
}

// RouteMap is the multi-target convenience constructor. It expands a single
// source into one Edge per (route, target) pair, mirroring adk-python's
// RoutingMap shorthand.
func RouteMap(from Node, m map[Route]Node) []Edge {
	out := make([]Edge, 0, len(m))
	for r, to := range m {
		out = append(out, Edge{From: from, To: to, Routes: []Route{r}})
	}
	return out
}
