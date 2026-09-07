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
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/internal/telemetry"
	"google.golang.org/adk/v2/cmd/launcher/universal"
	"google.golang.org/adk/v2/internal/cli/util"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

// webConfig contains parameters for launching web server
type webConfig struct {
	port            int
	writeTimeout    time.Duration
	readTimeout     time.Duration
	idleTimeout     time.Duration
	shutdownTimeout time.Duration
	otelToCloud     bool
	useH2C          bool
	allowInMemory   bool
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

// applyServiceDefaults fills in in-memory services the caller left unset, and
// reports which ones it invented and whether the caller consented.
//
// Neither adkrest.NewServer nor runner.New defaults them; only
// runner.NewInMemory does, and the web launcher does not use it. A nil session
// or memory service reaches the request path and panics, which drops the
// connection without sending any HTTP response.
//
// The session service is always defaulted, as it was before the artifact and
// memory ones were added, because every request path needs it and a nil one
// panics.
//
// The other two are defaulted too, but only silently when allowInMemory says a
// volatile store is acceptable: the webui sublauncher is a developer tool and
// implies it, and -allow_in_memory_services sets it explicitly. Without that
// consent they are still created, so nothing breaks on upgrade, and unconsented
// reports true so the caller can warn that a future release will refuse instead.
//
// Refusing is the eventual goal rather than the current behaviour because
// cmd/launcher/prod runs through this same path. A production deployment that
// configured no storage should learn about it at startup, not when a restart
// has already dropped the data. adk-python falls back to
// InMemoryArtifactService for an absent URI, so the fallback is not wrong in
// itself; it belongs to a dev server rather than to every server.
func applyServiceDefaults(config *launcher.Config, allowInMemory bool) (defaulted []string, unconsented bool) {
	if config.SessionService == nil {
		config.SessionService = session.InMemoryService()
		log.Print("No session service configured. Using an in-memory one, so whatever it holds is lost when the process exits.")
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
	return defaulted, len(defaulted) > 0 && !allowInMemory
}

// logInMemoryServiceWarning restates which services fell back to memory, in the
// startup banner alongside the URLs.
//
// applyServiceDefaults already logs each one, but that runs before the routers
// are built and has scrolled past by the time the banner prints. Repeating it
// here is deliberate: this block is what an operator reads, and losing artifacts
// on the next restart is not something to learn from a line further up.
//
// When the caller did not ask for a volatile store, the warning also says the
// launcher will refuse to start in a future release, so a deployment can fix
// its configuration before that lands rather than after.
func logInMemoryServiceWarning(defaulted []string, unconsented bool) {
	if len(defaulted) == 0 {
		return
	}
	log.Printf("       WARNING: no %s service configured, so an in-memory one is in use.", strings.Join(defaulted, ", "))
	log.Printf("       WARNING:     everything it holds is lost when this process exits.")
	if unconsented {
		log.Printf("       WARNING:     a future release will refuse to start instead. Configure a")
		log.Printf("       WARNING:     service, or pass -allow_in_memory_services to keep this.")
	}
}

// buildRouter assembles the router Run serves: the base router, then every
// active sublauncher's routes, then the fallback health route.
//
// Separated from Run so a test can assert the registration order without
// binding a port, because the order is what decides which /health wins.
func (w *webLauncher) buildRouter(config *launcher.Config) (*mux.Router, error) {
	router := BuildBaseRouter()

	// check if there are any active sublaunchers
	if len(w.activeSublaunchers) == 0 {
		availableSublaunchers := make([]string, len(w.sublaunchers))
		for i, l := range w.sublaunchers {
			availableSublaunchers[i] = l.Keyword()
		}
		return nil, fmt.Errorf("no active sublaunchers found - please specify them in the command line. Possible values: %v", availableSublaunchers)
	}

	// Setup subrouters
	for _, l := range w.sublaunchers {
		if _, isActive := w.activeSublaunchers[l.Keyword()]; isActive {
			if err := l.SetupSubrouters(router, config); err != nil {
				return nil, fmt.Errorf("%s subrouter setup failed: %v", l.Keyword(), err)
			}
		}
	}

	// Registered after the sublaunchers, not before, because gorilla/mux
	// matches in registration order. Going first would shadow a /health that a
	// sublauncher serves itself, leaving that sublauncher's own readiness
	// signal unreachable while the probe still reported ok.
	registerHealthRoute(router)
	return router, nil
}

// webUIKeyword is the command-line keyword of the webui sublauncher. It is a
// literal rather than a reference because that package imports this one, so the
// dependency cannot run the other way.
const webUIKeyword = "webui"

// allowsInMemoryServices reports whether a volatile artifact and memory store
// is acceptable for this invocation. The webui sublauncher is a developer tool
// and implies it, so `adk web api webui` still runs with no configuration.
func (w *webLauncher) allowsInMemoryServices() bool {
	if w.config.allowInMemory {
		return true
	}
	_, hasWebUI := w.activeSublaunchers[webUIKeyword]
	return hasWebUI
}

// Run implements launcher.SubLauncher.
func (w *webLauncher) Run(ctx context.Context, config *launcher.Config) error {
	defaulted, unconsented := applyServiceDefaults(config, w.allowsInMemoryServices())

	router, err := w.buildRouter(config)
	if err != nil {
		return err
	}

	log.Printf("Starting the web server: %+v", w.config)
	log.Println()
	webUrl := fmt.Sprintf("http://localhost:%v", fmt.Sprint(w.config.port))
	log.Printf("Web servers starts on %s", webUrl)
	for _, l := range w.activeSublaunchers {
		l.UserMessage(webUrl, log.Println)
	}
	logInMemoryServiceWarning(defaulted, unconsented)
	log.Println()

	telemetryService, err := telemetry.InitAndSetGlobalOtelProviders(ctx, config, w.config.otelToCloud)
	if err != nil {
		return fmt.Errorf("telemetry initialization failed: %v", err)
	}

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
		serverErr := srv.Shutdown(shutdownCtx)
		telemetryErr := telemetryService.Shutdown(shutdownCtx)
		return errors.Join(serverErr, telemetryErr)
	case err, ok := <-errChan:
		if !ok {
			return nil
		}
		return fmt.Errorf("server failed: %v", err)
	}
}

func (w *webLauncher) buildHTTPServer(handler http.Handler) *http.Server {
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%v", fmt.Sprint(w.config.port)),
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
	fs.IntVar(&config.port, "port", 8080, "Localhost port for the server")
	fs.DurationVar(&config.writeTimeout, "write-timeout", 15*time.Second, "Server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the response after reading the headers & body")
	fs.DurationVar(&config.readTimeout, "read-timeout", 15*time.Second, "Server read timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for reading the whole request including body")
	fs.DurationVar(&config.idleTimeout, "idle-timeout", 60*time.Second, "Server idle timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for the next request (only when keep-alive is enabled)")
	fs.DurationVar(&config.shutdownTimeout, "shutdown-timeout", 15*time.Second, "Server shutdown timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for active requests to finish during shutdown")
	fs.BoolVar(&config.otelToCloud, "otel_to_cloud", false, "Enables/disables OpenTelemetry export to GCP: telemetry.googleapis.com. See adk-go/telemetry package for details about supported options, credentials and environment variables.")
	fs.BoolVar(&config.useH2C, "h2c", false, "Enable prior-knowledge cleartext HTTP/2 (h2c; no HTTP/1.1 Upgrade) on the web server listener. Cleartext is insecure; do not expose it to untrusted networks. Long-lived streaming responses may require increasing --write-timeout.")
	fs.BoolVar(&config.allowInMemory, "allow_in_memory_services", false, "Fall back to in-memory artifact and memory services when none is configured, instead of refusing to start. Their contents are lost when the process exits, so this is for local runs. Implied when the webui sublauncher is active.")

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
// building its own server keeps /health for itself, and calls it after the
// sublaunchers so that it is a fallback rather than an override.
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
