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

// Package eventarc provides a sublauncher that adds Eventarc trigger capabilities to ADK web server.
package eventarc

import (
	"flag"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
	"google.golang.org/adk/server/adkrest/controllers/triggers"
)

type eventarcConfig struct {
	pathPrefix        string
	triggerMaxRetries int
	triggerBaseDelay  time.Duration
	triggerMaxDelay   time.Duration
	triggerMaxRuns    int
}

type eventarcLauncher struct {
	flags  *flag.FlagSet
	config *eventarcConfig
}

// NewLauncher creates a new eventarc launcher. It extends Web launcher.
func NewLauncher() web.Sublauncher {
	config := &eventarcConfig{}

	fs := flag.NewFlagSet("eventarc", flag.ContinueOnError)
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "Path prefix for the Eventarc trigger endpoint. Default is '/api'.")
	fs.IntVar(&config.triggerMaxRetries, "trigger_max_retries", 3, "Maximum retries for HTTP 429 errors from triggers")
	fs.DurationVar(&config.triggerBaseDelay, "trigger_base_delay", 1*time.Second, "Base delay for trigger retry exponential backoff")
	fs.DurationVar(&config.triggerMaxDelay, "trigger_max_delay", 10*time.Second, "Maximum delay for trigger retry exponential backoff")
	fs.IntVar(&config.triggerMaxRuns, "trigger_max_concurrent_runs", 100, "Maximum concurrent trigger runs")

	return &eventarcLauncher{
		config: config,
		flags:  fs,
	}
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for eventarc launcher.
func (e *eventarcLauncher) Keyword() string {
	return "eventarc"
}

// Parse parses the command-line arguments for the eventarc launcher.
func (e *eventarcLauncher) Parse(args []string) ([]string, error) {
	err := e.flags.Parse(args)
	if err != nil || !e.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse eventarc flags: %v", err)
	}
	if e.config.triggerMaxRetries <= 0 {
		return nil, fmt.Errorf("trigger_max_retries must be > 0")
	}
	if e.config.triggerBaseDelay < 0 {
		return nil, fmt.Errorf("trigger_base_delay must be >= 0")
	}
	if e.config.triggerMaxDelay <= 0 {
		return nil, fmt.Errorf("trigger_max_delay must be > 0")
	}
	if e.config.triggerMaxRuns <= 0 {
		return nil, fmt.Errorf("trigger_max_concurrent_runs must be > 0")
	}

	prefix := e.config.pathPrefix
	if !strings.HasPrefix(prefix, "/") {
		prefix = "/" + prefix
	}
	e.config.pathPrefix = strings.TrimSuffix(prefix, "/")

	return e.flags.Args(), nil
}

// CommandLineSyntax returns the command-line syntax for the eventarc launcher.
func (e *eventarcLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(e.flags)
}

// SimpleDescription implements web.Sublauncher.
func (e *eventarcLauncher) SimpleDescription() string {
	return "starts ADK Eventarc trigger endpoint server"
}

// SetupSubrouters adds the Eventarc trigger endpoint to the parent router.
func (e *eventarcLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	triggerConfig := triggers.TriggerConfig{
		MaxRetries:        e.config.triggerMaxRetries,
		BaseDelay:         e.config.triggerBaseDelay,
		MaxDelay:          e.config.triggerMaxDelay,
		MaxConcurrentRuns: e.config.triggerMaxRuns,
	}

	controller := triggers.NewEventarcController(
		config.SessionService,
		config.AgentLoader,
		config.MemoryService,
		config.ArtifactService,
		config.PluginConfig,
		triggerConfig,
	)

	subrouter := router
	if e.config.pathPrefix != "" && e.config.pathPrefix != "/" {
		subrouter = router.PathPrefix(e.config.pathPrefix).Subrouter()
	}

	subrouter.HandleFunc("/apps/{app_name}/trigger/eventarc", controller.EventarcTriggerHandler).Methods(http.MethodPost)
	return nil
}

// UserMessage implements web.Sublauncher.
func (e *eventarcLauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       eventarc: Eventarc trigger endpoint is available at %s%s/apps/{app_name}/trigger/eventarc", webURL, e.config.pathPrefix))
}
