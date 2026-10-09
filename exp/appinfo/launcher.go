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

package appinfo

import (
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/web"
	"google.golang.org/adk/v2/internal/cli/util"
	"google.golang.org/adk/v2/internal/originguard"
	"google.golang.org/adk/v2/server/authn"
)

// routeSuffix is the endpoint's path below the -path_prefix.
const routeSuffix = "/apps/{app_name}/app-info"

// sublauncher is the web sublauncher [NewLauncher] returns.
type sublauncher struct {
	flags      *flag.FlagSet
	pathPrefix string
}

// NewLauncher returns a web sublauncher that serves
// GET {path_prefix}/apps/{app_name}/app-info for the apps in
// [launcher.Config.AgentLoader]. Its keyword is "appinfo". Its one flag,
// -path_prefix, defaults to "/", which serves the endpoint at the root.
//
// The endpoint is served only when the sublauncher is passed to
// [web.NewLauncher] and also named on the command line:
//
//	go run . web api appinfo -path_prefix /api
//
// That serves it at /api/apps/{app_name}/app-info, beside the REST API under
// its default /api prefix.
//
// Pass it to web.NewLauncher before the api sublauncher. A router serves a
// request with the first route that matches it, and the api sublauncher's
// route matches every path under its prefix, or every path at all with
// -path_prefix /. Registered after it, the app-info route can be unreachable,
// and setup then fails with an error. The check probes the router with a plain
// GET, so it does not notice an earlier route that matches only on a Host,
// header or query.
//
// The route authenticates with [launcher.Config.Authenticator] and applies the
// REST API's Host check for [launcher.Config.BindHost]. It allows no
// cross-origin browser requests.
func NewLauncher() web.Sublauncher {
	l := &sublauncher{flags: flag.NewFlagSet("appinfo", flag.ContinueOnError)}
	l.flags.StringVar(&l.pathPrefix, "path_prefix", "/", `URL prefix to serve app-info under. "/" serves it at the root.`)
	return l
}

// Keyword implements [web.Sublauncher].
func (l *sublauncher) Keyword() string { return "appinfo" }

// Parse implements [web.Sublauncher].
func (l *sublauncher) Parse(args []string) ([]string, error) {
	if err := l.flags.Parse(args); err != nil {
		return nil, fmt.Errorf("failed to parse appinfo flags: %w", err)
	}
	// The router reads braces in a path as a route variable.
	if strings.ContainsAny(l.pathPrefix, "{}") {
		return nil, fmt.Errorf("appinfo -path_prefix must not contain { or }")
	}
	return l.flags.Args(), nil
}

// CommandLineSyntax implements [web.Sublauncher].
func (l *sublauncher) CommandLineSyntax() string { return util.FormatFlagUsage(l.flags) }

// SimpleDescription implements [web.Sublauncher].
func (l *sublauncher) SimpleDescription() string {
	return "serves the experimental GET {path_prefix}/apps/{app_name}/app-info, which describes each LLM agent's instruction and tools"
}

// UserMessage implements [web.Sublauncher].
func (l *sublauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       appinfo:  experimental, GET %s%s", webURL, l.route()))
}

// SetupSubrouters implements [web.Sublauncher].
func (l *sublauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// The REST API wraps its own routes in these two checks, but this route is
	// registered beside it rather than inside it.
	guard := originguard.New(originguard.Config{BindHost: config.BindHost})
	h := guard.Middleware(authn.Middleware(config.Authenticator)(Handler(config.AgentLoader)))
	route := router.Methods(http.MethodGet, http.MethodHead).Path(l.route()).Handler(h)

	probe := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: l.prefix() + "/apps/probe/app-info"},
		Header: http.Header{},
	}
	var match mux.RouteMatch
	if !router.Match(probe, &match) || match.Route != route {
		return fmt.Errorf("route %s is shadowed by a route registered before it: pass appinfo.NewLauncher() to web.NewLauncher before the sublauncher serving that prefix, such as api", l.route())
	}
	return nil
}

// prefix is -path_prefix with one leading slash and no trailing one, and empty
// for the root.
func (l *sublauncher) prefix() string {
	if p := strings.Trim(l.pathPrefix, "/"); p != "" {
		return "/" + p
	}
	return ""
}

func (l *sublauncher) route() string { return l.prefix() + routeSuffix }
