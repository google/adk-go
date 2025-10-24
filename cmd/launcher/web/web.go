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

// package web provides common web-related funcionalities
package web

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
)

// WebConfig contains parametres for lauching web server
type WebConfig struct {
	port int
}

// WebLauncher can launch web server
type WebLauncher struct {
	flags        *flag.FlagSet
	config       *WebConfig
	sublaunchers []WebSublauncher
}

type WebSublauncher interface {
	launcher.Sublauncher
	SetupSubrouters(router *mux.Router, adkConfig *adk.Config)
	ReplaceRouter(router *mux.Router, adkConfig *adk.Config) *mux.Router
	UserMessage(webUrl string, printer func(v ...any))
}

func (w *WebLauncher) FormatSyntax() string {
	return launcher.FormatFlagUsage(w.flags)
}

func (w *WebLauncher) Keyword() string {
	return "web"
}

// Parse processes its arguments, trying to find an appropiate sublauncher by keywords. Returns unprocessed arguments
func (w *WebLauncher) Parse(args []string) ([]string, error) {

	keyToSublauncher := make(map[string]launcher.Sublauncher)
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
	processedKeywords := make(map[string]launcher.Sublauncher)

	for {
		if len(restArgs) == 0 {
			break
		}
		keyword := restArgs[0]
		if _, ok := processedKeywords[keyword]; ok {
			// already processed
			return restArgs, fmt.Errorf("the keyword %q is specified and processed more than once, which is not allowed", keyword)
		}

		if sublauncher, ok := keyToSublauncher[keyword]; ok {
			// skip the keyword and move on
			restArgs, err = sublauncher.Parse(restArgs[1:])
			if err != nil {
				return nil, fmt.Errorf("tha %q launcher cannot parse arguments: %v", keyword, err)
			}
			processedKeywords[keyword] = sublauncher
		} else {
			// not known keyword, let it be processed elsewhere
			break
		}
	}
	return restArgs, nil
}

func (w *WebLauncher) ParseAndRun(ctx context.Context, config *adk.Config, args []string, parseRemaining func([]string) error) error {
	panic("unimplemented")
}

func (w *WebLauncher) Run(ctx context.Context, config *adk.Config) error {
	// Setup subrouters
	router := BuildBaseRouter()
	for _, l := range w.sublaunchers {
		l.SetupSubrouters(router, config)
	}

	// Allow to replace router
	for _, l := range w.sublaunchers {
		router = l.ReplaceRouter(router, config)
	}

	log.Printf("Starting the web server: %+v", w.config)
	log.Println()
	webUrl := fmt.Sprintf("http://localhost:%v", fmt.Sprint(w.config.port))
	log.Printf("Web servers starts on %s", webUrl)
	for _, l := range w.sublaunchers {
		l.UserMessage(webUrl, log.Println)
	}
	log.Println()
	return http.ListenAndServe(":"+fmt.Sprint(w.config.port), router)
}

func (w *WebLauncher) SimpleDescription() string {
	return "starts web server with additional sub-servers specified by sublaunchers"
}

// NewLauncher creates new web launcher. Should be extended by sublaunchers providing real content
func NewLauncher(sublaunchers ...WebSublauncher) *WebLauncher {

	config := &WebConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.IntVar(&config.port, "port", 8080, "Localhost port for the server")

	return &WebLauncher{
		config:       config,
		flags:        fs,
		sublaunchers: sublaunchers,
	}
}

func Logger(inner http.Handler) http.Handler {
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

// BuildBaseRouter returns the main router, which can be exteded by sub-routers
func BuildBaseRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	router.Use(Logger)
	return router
}
