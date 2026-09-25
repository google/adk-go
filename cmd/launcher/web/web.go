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

// Package web provides a way to run ADK using a web server.
package web

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/internal/telemetry"
	"google.golang.org/adk/v2/cmd/launcher/universal"
	"google.golang.org/adk/v2/internal/cli/util"
	"google.golang.org/adk/v2/internal/originguard"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

const (
	logStartingWebServer = "Starting the web server: %+v"
	logWebServerStartsOn = "Web servers starts on %s"

	// defaultHost keeps the server off the network unless the caller opts in
	// with -host. See bindHost and the -host flag.
	defaultHost = "127.0.0.1"
)

// webConfig contains parameters for launching web server
type webConfig struct {
	port            int
	host            string
	allowOrigins    []string
	writeTimeout    time.Duration
	readTimeout     time.Duration
	idleTimeout     time.Duration
	shutdownTimeout time.Duration
	otelToCloud     bool
	useH2C          bool
}

// webLauncher can launch web server
type webLauncher struct {
	flags        *flag.FlagSet
	config       *webConfig
	sublaunchers []Sublauncher
	// maps keyword to sublauncher for the keywords parsed from command line
	activeSublaunchers map[string]Sublauncher
}

// Execute implements launcher.Launcher.
func (w *webLauncher) Execute(ctx context.Context, config *launcher.Config, args []string) error {
	if err := config.Validate(); err != nil {
		return err
	}
	remainingArgs, err := w.Parse(args)
	if err != nil {
		return fmt.Errorf("cannot parse args: %w", err)
	}
	// do not accept additional arguments
	err = universal.ErrorOnUnparsedArgs(remainingArgs)
	if err != nil {
		return fmt.Errorf("cannot parse all the arguments: %w", err)
	}
	return w.Run(ctx, config)
}

// Sublauncher defines an interface for extending the WebLauncher.
// Each sublauncher can add its own routes, wrap existing handlers, and parse its own command-line flags.
type Sublauncher interface {
	// Keyword is used to request usage of the Sublauncher from command-line
	Keyword() string
	// Parse after parsing command line args returns the remaining un-parsed arguments or error
	Parse(args []string) ([]string, error)
	// CommandLineSyntax returns a formatted string explaining command line syntax to end user
	CommandLineSyntax() string
	// SimpleDescription returns a short explanatory text displayed to end user
	SimpleDescription() string

	// SetupSubrouters adds sublauncher-specific routes to the router.
	SetupSubrouters(router *mux.Router, config *launcher.Config) error
	// UserMessage is a hook for sublaunchers to print a message to the user when the web server starts.
	UserMessage(webURL string, printer func(v ...any))
}

// CommandLineSyntax implements launcher.Launcher.
func (w *webLauncher) CommandLineSyntax() string {
	var b strings.Builder
	fmt.Fprint(&b, util.FormatFlagUsage(w.flags))
	fmt.Fprintf(&b, "  You may specify sublaunchers:\n")
	for _, l := range w.sublaunchers {
		fmt.Fprintf(&b, "    * %s - %s\n", l.Keyword(), l.SimpleDescription())
	}
	fmt.Fprintf(&b, "  Sublaunchers syntax:\n")
	for _, l := range w.sublaunchers {
		fmt.Fprintf(&b, "    %s\n  %s\n", l.Keyword(), l.CommandLineSyntax())
	}
	return b.String()
}

// Keyword implements launcher.SubLauncher.
func (w *webLauncher) Keyword() string {
	return "web"
}

// Parse implements launcher.SubLauncher. It parses the web launcher's flags
// and then iterates through the remaining arguments to find and parse arguments
// for any specified sublaunchers. It returns any arguments that are not processed.
func (w *webLauncher) Parse(args []string) ([]string, error) {
	keyToSublauncher := make(map[string]Sublauncher)
	for _, l := range w.sublaunchers {
		if _, ok := keyToSublauncher[l.Keyword()]; ok {
			return nil, fmt.Errorf("cannot create web launcher. Keywords for sublaunchers should be unique and they are not: '%s'", l.Keyword())
		}
		keyToSublauncher[l.Keyword()] = l
	}

	err := w.flags.Parse(args)
	if err != nil || !w.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse web flags: %v", err)
	}

	restArgs := w.flags.Args()
	w.activeSublaunchers = make(map[string]Sublauncher)

	for len(restArgs) > 0 {
		keyword := restArgs[0]
		if _, ok := w.activeSublaunchers[keyword]; ok {
			// already processed
			return restArgs, fmt.Errorf("the keyword %q is specified and processed more than once, which is not allowed", keyword)
		}

		if sublauncher, ok := keyToSublauncher[keyword]; ok {
			// skip the keyword and move on
			restArgs, err = sublauncher.Parse(restArgs[1:])
			if err != nil {
				return nil, fmt.Errorf("the %q launcher cannot parse arguments: %v", keyword, err)
			}
			w.activeSublaunchers[keyword] = sublauncher
		} else {
			// not known keyword, let it be processed elsewhere
			break
		}
	}
	return restArgs, nil
}

// applyServiceDefaults fills in in-memory services the caller left unset.
//
// Neither adkrest.NewServer nor runner.New defaults them; only
// runner.NewInMemory does, and the web launcher does not use it. A nil session
// or memory service reaches the request path and panics, which drops the
// connection without sending any HTTP response. The artifact handlers answer
// 503 instead, so defaulting that one replaces a clear diagnostic with a server
// that works until it restarts and then has lost everything. It is logged for
// that reason: cmd/launcher/prod runs through this same path, so a deployment
// that forgot to configure a service still says so on startup.
func applyServiceDefaults(config *launcher.Config) {
	var defaulted []string
	if config.SessionService == nil {
		config.SessionService = session.InMemoryService()
		defaulted = append(defaulted, "session")
	}
	if config.ArtifactService == nil {
		config.ArtifactService = artifact.InMemoryService()
		defaulted = append(defaulted, "artifact")
	}
	if config.MemoryService == nil {
		config.MemoryService = memory.InMemoryService()
		defaulted = append(defaulted, "memory")
	}
	for _, name := range defaulted {
		log.Printf("No %s service configured. Using an in-memory one, so whatever it holds is lost when the process exits.", name)
	}
}

// Run implements launcher.SubLauncher. It takes ownership of the telemetry
// providers initialized for execution and shuts them down on exit.
func (w *webLauncher) Run(ctx context.Context, config *launcher.Config) error {
	applyServiceDefaults(config)

	router := BuildBaseRouter()
	registerHealthRoute(router)

	// check if there are any active sublaunchers
	if len(w.activeSublaunchers) == 0 {
		availableSublaunchers := make([]string, len(w.sublaunchers))
		for i, l := range w.sublaunchers {
			availableSublaunchers[i] = l.Keyword()
		}
		return fmt.Errorf("no active sublaunchers found - please specify them in the command line. Possible values: %v", availableSublaunchers)
	}

	// Sublaunchers that build a server need the resolved bind address rather
	// than the raw flag, so an empty -host arms the same checks the default does.
	config.BindHost = w.bindHost()

	// Before any sublauncher runs, so that a server one of them builds, like
	// the REST server, is given the operator's origins too. Clipped so that
	// neither this append nor a sublauncher's writes into a slice the caller
	// owns.
	config.AllowedOrigins = append(slices.Clip(config.AllowedOrigins), w.config.allowOrigins...)

	// Setup subrouters
	for _, l := range w.sublaunchers {
		if _, isActive := w.activeSublaunchers[l.Keyword()]; isActive {
			if err := l.SetupSubrouters(router, config); err != nil {
				return fmt.Errorf("%s subrouter setup failed: %v", l.Keyword(), err)
			}
		}
	}

	// One guard over every route registered above, rather than one per
	// sublauncher, so that a sublauncher added later is covered without having
	// to remember it. Several of these routes run an agent or return session
	// data, and a rebound page can reach any of them.
	//
	// Built here, after setup, because sublaunchers add origins while they run.
	// gorilla/mux builds the middleware chain when a route matches, so this
	// still wraps the routes registered before it.
	//
	// The REST server keeps its own copy of these checks, because it is also
	// mounted outside this launcher. Its copy holds only the origins added
	// before the api sublauncher ran, so an origin that a later sublauncher adds
	// passes this guard and is still refused on the REST routes.
	router.Use(originguard.New(originguard.Config{
		BindHost:       config.BindHost,
		AllowedOrigins: config.AllowedOrigins,
	}).Middleware)

	telemetryService, err := telemetry.InitAndSetGlobalOtelProviders(ctx, config, w.config.otelToCloud)
	if err != nil {
		return fmt.Errorf("telemetry initialization failed: %v", err)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), w.config.shutdownTimeout)
		defer cancel()
		if err := telemetryService.Shutdown(shutdownCtx); err != nil {
			log.Printf("telemetry shutdown failed: %v", err)
		}
	}()

	log.Printf(logStartingWebServer, w.config)
	log.Println()
	webUrl := w.webURL()
	log.Printf(logWebServerStartsOn, webUrl)
	for _, l := range w.activeSublaunchers {
		l.UserMessage(webUrl, log.Println)
	}
	log.Println()

	srv := w.buildHTTPServer(router)

	errChan := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
		close(errChan)
	}()

	select {
	case <-ctx.Done():
		log.Println("Shutting down the web server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), w.config.shutdownTimeout)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	case err, ok := <-errChan:
		if !ok {
			return nil
		}
		return fmt.Errorf("server failed: %v", err)
	}
}

// webURL returns the user-facing URL for the configured host and port so the
// startup message reflects the actual bind address (including bracketed IPv6
// hosts such as "::1").
//
// The loopback hosts 127.0.0.1, ::1, 0.0.0.0 and :: are normalized to
// "localhost" in the URL shown/opened to the user. The ADK Web UI is served
// from http://localhost:8080/api, so presenting the server as
// http://127.0.0.1:8080 would be a different browser origin and fail CORS.
// This changes only the displayed URL - it does not change the server bind
// address, which is controlled by the -host flag and used by buildHTTPServer.
func (w *webLauncher) webURL() string {
	host := displayHost(w.bindHost())
	return fmt.Sprintf("http://%s", net.JoinHostPort(host, strconv.Itoa(w.config.port)))
}

// bindHost returns the host the server listens on. An empty -host is treated
// as the default. net.JoinHostPort("", port) yields ":port", which binds every
// interface, so leaving an empty value unresolved would reintroduce exactly the
// exposure this launcher refuses to default to.
func (w *webLauncher) bindHost() string {
	if w.config.host == "" {
		return defaultHost
	}
	return w.config.host
}

// displayHost maps loopback-ish listen hosts to "localhost" for the URL shown
// to the user, and otherwise returns the host unchanged. See webURL.
func displayHost(host string) string {
	switch host {
	case "127.0.0.1", "::1", "0.0.0.0", "::":
		return "localhost"
	default:
		return host
	}
}

func (w *webLauncher) buildHTTPServer(handler http.Handler) *http.Server {
	srv := &http.Server{
		Addr:         net.JoinHostPort(w.bindHost(), strconv.Itoa(w.config.port)),
		WriteTimeout: w.config.writeTimeout,
		ReadTimeout:  w.config.readTimeout,
		IdleTimeout:  w.config.idleTimeout,
		Handler:      handler,
	}

	if w.config.useH2C {
		// Enable both HTTP/1 and cleartext HTTP/2 on the same listener. Existing
		// REST, Web UI, A2A, and trigger routes continue to work over HTTP/1.1,
		// while custom web sublaunchers can register HTTP/2-capable handlers,
		// such as Connect handlers.
		protocols := new(http.Protocols)
		protocols.SetHTTP1(true)
		protocols.SetUnencryptedHTTP2(true)
		srv.Protocols = protocols
	}

	return srv
}

// SimpleDescription implements launcher.SubLauncher.
func (w *webLauncher) SimpleDescription() string {
	return "starts web server with additional sub-servers specified by sublaunchers"
}

// NewLauncher creates a new WebLauncher. It should be extended by providing
// one or more Sublaunchers that add the actual content and functionality.
func NewLauncher(sublaunchers ...Sublauncher) launcher.SubLauncher {
	config := &webConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.host, "host", defaultHost, "Host/IP to bind the web server to. Defaults to 127.0.0.1 (loopback only) so the server is not exposed to the network. Use 0.0.0.0 to listen on all interfaces, which may be required when running adk web inside a container. An empty value is treated as the default.")
	fs.IntVar(&config.port, "port", 8080, "Port for the web server")
	// Named, repeated and described like adk-python's --allow_origins, which
	// also accepts regex: entries. originguard matches literal origins only,
	// so the help offers none. launcher.Config.AllowedOrigins documents what
	// the list does.
	fs.Func("allow_origins", "Optional. Origins to allow for CORS (e.g., 'https://example.com'). Repeat the flag to allow more than one.", func(origin string) error {
		config.allowOrigins = append(config.allowOrigins, origin)
		return nil
	})
	fs.DurationVar(&config.writeTimeout, "write-timeout", 15*time.Second, "Server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the response after reading the headers & body")
	fs.DurationVar(&config.readTimeout, "read-timeout", 15*time.Second, "Server read timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for reading the whole request including body")
	fs.DurationVar(&config.idleTimeout, "idle-timeout", 60*time.Second, "Server idle timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for the next request (only when keep-alive is enabled)")
	fs.DurationVar(&config.shutdownTimeout, "shutdown-timeout", 15*time.Second, "Server shutdown timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for active requests to finish during shutdown")
	fs.BoolVar(&config.otelToCloud, "otel_to_cloud", false, "Enables/disables OpenTelemetry export to GCP: telemetry.googleapis.com. See adk-go/telemetry package for details about supported options, credentials and environment variables.")
	fs.BoolVar(&config.useH2C, "h2c", false, "Enable prior-knowledge cleartext HTTP/2 (h2c; no HTTP/1.1 Upgrade) on the web server listener. Cleartext is insecure; do not expose it to untrusted networks. Long-lived streaming responses may require increasing --write-timeout.")

	return &webLauncher{
		config:       config,
		flags:        fs,
		sublaunchers: sublaunchers,
	}
}

// logger is a middleware that logs the HTTP method, request URI, and the time taken to process the request.
func logger(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		inner.ServeHTTP(w, r)

		log.Printf(
			"%s %s %s",
			r.Method,
			r.RequestURI,
			time.Since(start),
		)
	})
}

// BuildBaseRouter returns the main router, which can be extended by sub-routers.
//
// It deliberately registers no routes of its own. mux serves the first route
// that matches, so anything registered here would silently shadow the same path
// registered by a caller afterwards.
func BuildBaseRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	router.Use(logger)
	return router
}

// registerHealthRoute serves health at the root as well as under the API
// prefix. Load balancers and container probes are configured with a fixed path
// and cannot be expected to know which sublaunchers happen to be enabled.
//
// Run calls this rather than BuildBaseRouter doing it, so that an embedder
// building its own server keeps /health for itself.
func registerHealthRoute(router *mux.Router) {
	router.HandleFunc("/health", healthHandler).Methods(http.MethodGet, http.MethodHead)
}

// healthHandler reports that the web server is up. It says nothing about the
// health of the agent or its downstream models. The body and Content-Type match
// adkrest's /api/health, so a probe can be pointed at either one.
func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if _, err := w.Write([]byte(`{"status":"ok"}` + "\n")); err != nil {
		log.Printf("failed to write health response: %v", err)
	}
}
