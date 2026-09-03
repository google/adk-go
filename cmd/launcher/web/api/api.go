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

// Package api provides a sublauncher that adds ADK REST API capabilities.
package api

import (
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
	weblauncher "google.golang.org/adk/v2/cmd/launcher/web"
	"google.golang.org/adk/v2/internal/cli/util"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/telemetry"
)

// apiConfig contains parametres for lauching ADK REST API
type apiConfig struct {
	frontendAddress string
	pathPrefix      string
	sseWriteTimeout time.Duration
	traceCapacity   int
	includeDebugAPI bool
}

// apiLauncher can launch ADK REST API
type apiLauncher struct {
	flags  *flag.FlagSet
	config *apiConfig
}

// apiMethods are the verbs a browser may use against the API.
//
// The mount itself no longer filters on them; see [registerAPIRoutes]. They are
// what a CORS preflight is told, which is a separate question from what the
// mount forwards: the preflight answers "may script on this origin send this
// verb", and these are the verbs the API is designed for.
var apiMethods = []string{
	http.MethodGet,
	http.MethodHead,
	http.MethodPost,
	http.MethodPut,
	http.MethodPatch,
	http.MethodDelete,
	http.MethodOptions,
}

// corsAllowMethods is the Access-Control-Allow-Methods value, built from
// apiMethods so the two cannot drift apart.
var corsAllowMethods = strings.Join(apiMethods, ", ")

// CommandLineSyntax returns the command-line syntax for the API launcher.
func (a *apiLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

// normalizeOrigin turns a -webui_address value into an RFC 6454 origin
// (scheme://host[:port]), which is the only form a browser accepts in
// Access-Control-Allow-Origin.
//
// A bare host or host:port gets http:// prepended. The flag names a local
// development web UI, which is served over plain HTTP, so http is the only
// useful default; pass a full https:// URL to override it. Any path, query or
// trailing slash is dropped, because an origin has none. "*" and the empty
// string pass through unchanged.
func normalizeOrigin(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" || addr == "*" {
		return addr
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	u, err := url.Parse(addr)
	if err != nil || u.Host == "" {
		// Not something we can read as an origin. Pass it through rather than
		// inventing a value: the operator sees their own input echoed back.
		return addr
	}
	return u.Scheme + "://" + u.Host
}

// corsWithArgs adds CORS headers which allow calling ADK REST API from another
// web app (like ADK WebUI). The configured address is normalised once, when the
// middleware is built, rather than on every request.
func corsWithArgs(frontendAddress string) func(next http.Handler) http.Handler {
	origin := normalizeOrigin(frontendAddress)
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				if origin != "*" {
					// The response body depends on the configured origin only,
					// but caches cannot know that, and a cached response
					// carrying this header must not be served to a request from
					// a different origin.
					w.Header().Add("Vary", "Origin")
				}
			}
			w.Header().Set("Access-Control-Allow-Methods", corsAllowMethods)
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// trimTrailingSlash drops a trailing slash from the request path before the
// REST router sees it.
//
// The REST router runs with StrictSlash(true), so it answers /list-apps/ with a
// redirect rather than a match. Two things go wrong with that under a mounted
// prefix. The redirect it writes is relative to the mount, so it loses the
// prefix and 404s. And a redirect turns a POST into a GET at the client, so
// POST /apps/x/users/y/sessions/ came back as a session list from the GET
// handler on the same path, with the request body discarded.
//
// Normalising here instead means the intended handler runs directly, with its
// method and body intact, and no redirect is involved at all. It is safe
// because no REST route is registered with a trailing slash, so no route needs
// one to match.
func trimTrailingSlash(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Count trailing slashes on the escaped path so an encoded %2F inside a
		// path variable is left alone.
		escaped := r.URL.EscapedPath()
		n := len(escaped) - len(strings.TrimRight(escaped, "/"))
		if n == len(escaped) {
			// The path is nothing but slashes ("/" or ""). Keep one, so the
			// router always sees a rooted path.
			n--
		}
		if n <= 0 && r.URL.Path != "" {
			next.ServeHTTP(w, r)
			return
		}
		r2 := r.Clone(r.Context())
		if n > 0 {
			r2.URL.Path = r.URL.Path[:len(r.URL.Path)-n]
			if r2.URL.RawPath != "" {
				r2.URL.RawPath = r.URL.RawPath[:len(r.URL.RawPath)-n]
			}
		}
		if r2.URL.Path == "" {
			r2.URL.Path = "/"
		}
		next.ServeHTTP(w, r2)
	})
}

// redirectRewriter restores the mount prefix on a redirect written by the
// handler underneath it, and keeps the method on a redirected unsafe request.
type redirectRewriter struct {
	http.ResponseWriter
	prefix      string
	safeMethod  bool // the request method is GET or HEAD
	wroteHeader bool
}

func (w *redirectRewriter) WriteHeader(code int) {
	if w.wroteHeader {
		w.ResponseWriter.WriteHeader(code)
		return
	}
	w.wroteHeader = true
	if code >= 300 && code < 400 {
		if loc := w.Header().Get("Location"); strings.HasPrefix(loc, "/") &&
			loc != w.prefix && !strings.HasPrefix(loc, w.prefix+"/") {
			w.Header().Set("Location", w.prefix+loc)
		}
		if !w.safeMethod {
			// 301 and 302 licence a client to re-issue the request as a GET,
			// which would land on a different handler. 308 and 307 mean the
			// same thing but keep the method and the body.
			switch code {
			case http.StatusMovedPermanently:
				code = http.StatusPermanentRedirect
			case http.StatusFound:
				code = http.StatusTemporaryRedirect
			}
		}
	}
	w.ResponseWriter.WriteHeader(code)
}

// Flush keeps SSE responses streaming through the wrapper.
func (w *redirectRewriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController reach the real writer, which the SSE
// handler needs for SetWriteDeadline.
func (w *redirectRewriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// rewriteRedirects re-adds pathPrefix to a redirect written below the mount.
//
// The handler under a mount is served a path with the prefix stripped, so any
// Location it derives from that path is missing the prefix and does not
// resolve. gorilla/mux emits one for an uncleaned path such as //list-apps.
// Rewriting the header here is the only place that knows the prefix.
func rewriteRedirects(pathPrefix string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(&redirectRewriter{
			ResponseWriter: w,
			prefix:         pathPrefix,
			safeMethod:     r.Method == http.MethodGet || r.Method == http.MethodHead,
		}, r)
	})
}

// registerAPIRoutes mounts handler on router under pathPrefix.
//
// The mount matches on path only, never on method. Method matching here would
// happen before the REST router is reached, so a verb the mount did not list
// was a blanket 405 for every API path no matter what the REST router served —
// that is how PATCH (updateSession), PUT (createTest, updateEvalCase) and HEAD
// came to be disabled. Worse, an outer 405 carries no Allow header, because
// this layer knows only that something is mounted here, not which verbs each
// path underneath serves. Forwarding everything puts the answer where that is
// known: the REST router replies 405 with an accurate Allow.
//
// Shared by SetupSubrouters and the tests so that both exercise the same
// registration.
func registerAPIRoutes(router *mux.Router, pathPrefix string, handler http.Handler) {
	// If prefix is empty, don't use PathPrefix("") because it's too greedy.
	// Instead, attach the handler to the main router directly.
	// This allows other routes (like /ui/) to match first if registered.
	if pathPrefix == "" || pathPrefix == "/" {
		router.NewRoute().Handler(trimTrailingSlash(handler))
		return
	}

	mounted := rewriteRedirects(pathPrefix, http.StripPrefix(pathPrefix, trimTrailingSlash(handler)))
	// PathPrefix(pathPrefix) alone is not segment-aware: it captures /apifoo
	// and hands the REST router /foo. Matching prefix+"/" and the bare prefix
	// separately keeps the mount on segment boundaries. The prefix route is
	// registered first so that a request for the mount root is not answered
	// with an outer StrictSlash redirect.
	router.PathPrefix(pathPrefix + "/").Handler(mounted)
	router.Path(pathPrefix).Handler(mounted)
}

// UserMessage implements web.Sublauncher. Prints message to the user
func (a *apiLauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       api:  you can access API using %s%s", webURL, a.config.pathPrefix))
	printer(fmt.Sprintf("       api:      for instance: %s%s/list-apps", webURL, a.config.pathPrefix))
}

// SetupSubrouters adds the API router to the parent router.
func (a *apiLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	// Create the ADK REST API handler
	restServer, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  config.SessionService,
		MemoryService:   config.MemoryService,
		AgentLoader:     config.AgentLoader,
		ArtifactService: config.ArtifactService,
		SSEWriteTimeout: a.config.sseWriteTimeout,
		PluginConfig:    config.PluginConfig,
		Compaction:      config.Compaction,
		DebugConfig: adkrest.DebugTelemetryConfig{
			TraceCapacity: a.config.traceCapacity,
		},
		DebugAPIConfig: adkrest.DebugAPIConfig{
			IncludeDebugAPI: a.config.includeDebugAPI,
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create REST server: %w", err)
	}

	config.TelemetryOptions = append(config.TelemetryOptions, telemetry.WithSpanProcessors(restServer.SpanProcessor()), telemetry.WithLogRecordProcessors(restServer.LogProcessor()))

	// Wrap it with CORS middleware
	corsHandler := corsWithArgs(a.config.frontendAddress)(restServer)

	registerAPIRoutes(router, a.config.pathPrefix, corsHandler)
	return nil
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for API launcher.
func (a *apiLauncher) Keyword() string {
	return "api"
}

// Parse parses the command-line arguments for the API launcher.
func (a *apiLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse api flags: %v", err)
	}
	p := a.config.pathPrefix
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	a.config.pathPrefix = strings.TrimSuffix(p, "/")

	restArgs := a.flags.Args()
	return restArgs, nil
}

// SimpleDescription implements web.Sublauncher. Returns a simple description of the API launcher.
func (a *apiLauncher) SimpleDescription() string {
	return "starts ADK REST API server, accepting origins specified by webui_address (CORS)"
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher() weblauncher.Sublauncher {
	config := &apiConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.frontendAddress, "webui_address", "localhost:8080", "ADK WebUI origin as seen from the user browser. It's used to allow CORS requests. Accepts a full origin such as 'http://localhost:8080'; a bare hostname and optional port is read as http. '*' allows any origin.")
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "ADK REST API path prefix. Default is '/api'.")
	fs.DurationVar(&config.sseWriteTimeout, "sse-write-timeout", 120*time.Second, "SSE server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the SSE response after reading the headers & body")
	fs.IntVar(&config.traceCapacity, "trace_capacity", 10000, "Maximum number of traces to keep in memory.")
	fs.BoolVar(&config.includeDebugAPI, "include_debug_api", false, "The debug api endpoint will be included in the API if and only if the flag is set to true.")

	return &apiLauncher{
		config: config,
		flags:  fs,
	}
}
