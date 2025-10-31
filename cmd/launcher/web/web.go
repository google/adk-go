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

// package web provides a way to run ADK using web server (extended by sublaunchers)
package web

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/universal"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/session"
)

// WebConfig contains parametres for lauching web server
type webConfig struct {
	port int
}

// WebLauncher can launch web server
type WebLauncher struct {
	flags        *flag.FlagSet
	config       *webConfig
	sublaunchers []WebSublauncher
	// maps keyword to sublauncher for the keywords parsed from command line
	activeSublaunchers map[string]WebSublauncher
}

// Execute implements launcher.Launcher.
func (w *WebLauncher) Execute(ctx context.Context, config *adk.Config, args []string) error {
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

// WebSublauncher defines an interface for extending the WebLauncher.
// Each sublauncher can add its own routes, wrap existing handlers, and parse its own command-line flags.
type WebSublauncher interface {
	Keyword() string
	Parse(args []string) ([]string, error)
	CommandLineSyntax() string
	SimpleDescription() string

	// SetupSubrouters adds sublauncher-specific routes to the router.
	SetupSubrouters(router *mux.Router, adkConfig *adk.Config)
	// WrapHandlers allows a sublauncher to wrap the main HTTP handler, for example to add middleware.
	WrapHandlers(handler http.Handler, adkConfig *adk.Config) http.Handler
	// UserMessage is a hook for sublaunchers to print a message to the user when the web server starts.
	UserMessage(webUrl string, printer func(v ...any))
}

// CommandLineSyntax implements launcher.Launcher.
func (w *WebLauncher) CommandLineSyntax() string {
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
func (w *WebLauncher) Keyword() string {
	return "web"
}

// Parse implements launcher.SubLauncher. It parses the web launcher's flags
// and then iterates through the remaining arguments to find and parse arguments
// for any specified sublaunchers. It returns any arguments that are not processed.
func (w *WebLauncher) Parse(args []string) ([]string, error) {

	keyToSublauncher := make(map[string]WebSublauncher)
	for _, l := range w.sublaunchers {
		if _, ok := keyToSublauncher[l.Keyword()]; ok {
			return nil, fmt.Errorf("cannot create universal launcher. Keywords for sublaunchers should be unique and they are not: '%s'", l.Keyword())
		}
		keyToSublauncher[l.Keyword()] = l
	}

	err := w.flags.Parse(args)
	if err != nil || !w.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse web flags: %v", err)
	}

	restArgs := w.flags.Args()
	w.activeSublaunchers = make(map[string]WebSublauncher)

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
				return nil, fmt.Errorf("tha %q launcher cannot parse arguments: %v", keyword, err)
			}
			w.activeSublaunchers[keyword] = sublauncher
		} else {
			// not known keyword, let it be processed elsewhere
			break
		}
	}
	return restArgs, nil
}

// Run implements launcher.SubLauncher.
func (w *WebLauncher) Run(ctx context.Context, config *adk.Config) error {
	if config.SessionService == nil {
		config.SessionService = session.InMemoryService()
	}

	router := BuildBaseRouter()

	// check if there are any active sublaunchers
	if len(w.activeSublaunchers) == 0 {
		availableSublaunchers := make([]string, len(w.sublaunchers))
		for i, l := range w.sublaunchers {
			availableSublaunchers[i] = l.Keyword()
		}
		return fmt.Errorf("no active sublaunchers found - please specify them in the command line. Possible values: %v", availableSublaunchers)
	}

	// Setup subrouters
	for _, l := range w.activeSublaunchers {
		l.SetupSubrouters(router, config)
	}

	// allow sublaunchers to modify top level handler (needed by a2a)
	var handler http.Handler = router
	for _, l := range w.activeSublaunchers {
		handler = l.WrapHandlers(handler, config)
	}

	log.Printf("Starting the web server: %+v", w.config)
	log.Println()
	webUrl := fmt.Sprintf("http://localhost:%v", fmt.Sprint(w.config.port))
	log.Printf("Web servers starts on %s", webUrl)
	for _, l := range w.activeSublaunchers {
		l.UserMessage(webUrl, log.Println)
	}
	log.Println()

	srv := http.Server{
		Addr:         fmt.Sprintf(":%v", fmt.Sprint(w.config.port)),
		WriteTimeout: time.Second * 15,
		ReadTimeout:  time.Second * 15,
		IdleTimeout:  time.Second * 60,
		Handler:      handler,
	}

	err := srv.ListenAndServe()
	if err != nil {
		return fmt.Errorf("server failed: %v", err)
	}

	return nil
}

// SimpleDescription implements launcher.SubLauncher.
func (w *WebLauncher) SimpleDescription() string {
	return "starts web server with additional sub-servers specified by sublaunchers"
}

// NewLauncher creates a new WebLauncher. It should be extended by providing
// one or more WebSublaunchers that add the actual content and functionality.
func NewLauncher(sublaunchers ...WebSublauncher) *WebLauncher {

	config := &webConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.IntVar(&config.port, "port", 8080, "Localhost port for the server")

	return &WebLauncher{
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
func BuildBaseRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	router.Use(logger)
	return router
}
