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
	"slices"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	weblauncher "google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/adkrest"
	"google.golang.org/adk/telemetry"
)

// SupportedTriggers defines the allowed trigger sources for the ADK REST API.
var SupportedTriggers = []string{"pubsub"}

// apiConfig contains parametres for lauching ADK REST API
type apiConfig struct {
	frontendAddress   string
	pathPrefix        string
	sseWriteTimeout   time.Duration
	triggerSources    string
	triggerMaxRetries int
	triggerBaseDelay  time.Duration
	triggerMaxDelay   time.Duration
	triggerMaxRuns    int
}

// apiLauncher can launch ADK REST API
type apiLauncher struct {
	flags  *flag.FlagSet
	config *apiConfig
}

// CommandLineSyntax returns the command-line syntax for the API launcher.
func (a *apiLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(a.flags)
}

// Adds CORS headers which allow calling ADK REST API from another web app (like ADK WebUI)
func corsWithArgs(frontendAddress string) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", frontendAddress)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// UserMessage implements web.Sublauncher. Prints message to the user
func (a *apiLauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       api:  you can access API using %s%s", webURL, a.config.pathPrefix))
	printer(fmt.Sprintf("       api:      for instance: %s%s/list-apps", webURL, a.config.pathPrefix))
}

// SetupSubrouters adds the API router to the parent router.
func (a *apiLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	if a.config.triggerSources != "" {
		sources := strings.Split(a.config.triggerSources, ",")
		for _, source := range sources {
			if !slices.Contains(SupportedTriggers, source) {
				return fmt.Errorf("invalid trigger source: %q. Any subset of %s is allowed. Values should be comma-separated", source, strings.Join(SupportedTriggers, ", "))
			}
		}
		// De-duplicate the input sources.
		slices.Sort(sources)
		config.TriggerSources = slices.Compact(sources)
	}

	config.TriggerConfig = launcher.TriggerConfig{
		MaxRetries:        a.config.triggerMaxRetries,
		BaseDelay:         a.config.triggerBaseDelay,
		MaxDelay:          a.config.triggerMaxDelay,
		MaxConcurrentRuns: a.config.triggerMaxRuns,
	}

	// Create the ADK REST API handler
	restServer, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  config.SessionService,
		MemoryService:   config.MemoryService,
		AgentLoader:     config.AgentLoader,
		ArtifactService: config.ArtifactService,
		SSEWriteTimeout: a.config.sseWriteTimeout,
		PluginConfig:    config.PluginConfig,
		TriggerSources:  config.TriggerSources,
	})
	if err != nil {
		return fmt.Errorf("failed to create REST server: %w", err)
	}

	config.TelemetryOptions = append(config.TelemetryOptions, telemetry.WithSpanProcessors(restServer.SpanProcessor()), telemetry.WithLogRecordProcessors(restServer.LogProcessor()))

	// Wrap it with CORS middleware
	corsHandler := corsWithArgs(a.config.frontendAddress)(restServer)

	// If prefix is empty, don't use PathPrefix("") because it's too greedy.
	// Instead, attach the handler to the main router directly.
	if a.config.pathPrefix == "" || a.config.pathPrefix == "/" {
		// This allows other routes (like /ui/) to match first if registered
		router.Methods("GET", "POST", "DELETE", "OPTIONS").Handler(corsHandler)
	} else {
		router.Methods("GET", "POST", "DELETE", "OPTIONS").
			PathPrefix(a.config.pathPrefix).
			Handler(http.StripPrefix(a.config.pathPrefix, corsHandler))
	}
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
	if a.config.triggerMaxRetries < 0 {
		return nil, fmt.Errorf("trigger_max_retries must be >= 0")
	}
	if a.config.triggerBaseDelay < 0 {
		return nil, fmt.Errorf("trigger_base_delay must be >= 0")
	}
	if a.config.triggerMaxDelay < 0 {
		return nil, fmt.Errorf("trigger_max_delay must be >= 0")
	}
	if a.config.triggerMaxRuns < 0 {
		return nil, fmt.Errorf("trigger_max_concurrent_runs must be >= 0")
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
	fs.StringVar(&config.frontendAddress, "webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "ADK REST API path prefix. Default is '/api'.")
	fs.DurationVar(&config.sseWriteTimeout, "sse-write-timeout", 120*time.Second, "SSE server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the SSE response after reading the headers & body")
	fs.IntVar(&config.triggerMaxRetries, "trigger_max_retries", 3, "Maximum retries for HTTP 429 errors from triggers")
	fs.DurationVar(&config.triggerBaseDelay, "trigger_base_delay", 1*time.Second, "Base delay for trigger retry exponential backoff")
	fs.DurationVar(&config.triggerMaxDelay, "trigger_max_delay", 10*time.Second, "Maximum delay for trigger retry exponential backoff")
	fs.IntVar(&config.triggerMaxRuns, "trigger_max_concurrent_runs", 100, "Maximum concurrent trigger runs")
	fs.StringVar(&config.triggerSources, "trigger_sources", "", fmt.Sprintf("Comma-separated list of trigger sources to enable (any subset of %s)", strings.Join(SupportedTriggers, ", ")))

	return &apiLauncher{
		config: config,
		flags:  fs,
	}
}
