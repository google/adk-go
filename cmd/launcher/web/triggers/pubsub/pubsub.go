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

// Package pubsub provides a sublauncher that adds PubSub trigger capabilities to ADK web server.
package pubsub

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

type pubsubConfig struct {
	pathPrefix        string
	triggerMaxRetries int
	triggerBaseDelay  time.Duration
	triggerMaxDelay   time.Duration
	triggerMaxRuns    int
}

type pubsubLauncher struct {
	flags  *flag.FlagSet
	config *pubsubConfig
}

// NewLauncher creates a new pubsub launcher. It extends Web launcher.
func NewLauncher() web.Sublauncher {
	config := &pubsubConfig{}

	fs := flag.NewFlagSet("pubsub", flag.ContinueOnError)
	fs.StringVar(&config.pathPrefix, "path_prefix", "/api", "Path prefix for the PubSub trigger endpoint. Default is '/api'.")
	fs.IntVar(&config.triggerMaxRetries, "trigger_max_retries", 3, "Maximum retries for HTTP 429 errors from triggers")
	fs.DurationVar(&config.triggerBaseDelay, "trigger_base_delay", 1*time.Second, "Base delay for trigger retry exponential backoff")
	fs.DurationVar(&config.triggerMaxDelay, "trigger_max_delay", 10*time.Second, "Maximum delay for trigger retry exponential backoff")
	fs.IntVar(&config.triggerMaxRuns, "trigger_max_concurrent_runs", 100, "Maximum concurrent trigger runs")

	return &pubsubLauncher{
		config: config,
		flags:  fs,
	}
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for pubsub launcher.
func (p *pubsubLauncher) Keyword() string {
	return "pubsub"
}

// Parse parses the command-line arguments for the pubsub launcher.
func (p *pubsubLauncher) Parse(args []string) ([]string, error) {
	err := p.flags.Parse(args)
	if err != nil || !p.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse pubsub flags: %v", err)
	}
	if p.config.triggerMaxRetries <= 0 {
		return nil, fmt.Errorf("trigger_max_retries must be > 0")
	}
	if p.config.triggerBaseDelay < 0 {
		return nil, fmt.Errorf("trigger_base_delay must be >= 0")
	}
	if p.config.triggerMaxDelay <= 0 {
		return nil, fmt.Errorf("trigger_max_delay must be > 0")
	}
	if p.config.triggerMaxRuns <= 0 {
		return nil, fmt.Errorf("trigger_max_concurrent_runs must be > 0")
	}

	prefix := p.config.pathPrefix
	if !strings.HasPrefix(prefix, "/") {
		prefix = "/" + prefix
	}
	p.config.pathPrefix = strings.TrimSuffix(prefix, "/")

	return p.flags.Args(), nil
}

// CommandLineSyntax returns the command-line syntax for the pubsub launcher.
func (p *pubsubLauncher) CommandLineSyntax() string {
	return util.FormatFlagUsage(p.flags)
}

// SimpleDescription implements web.Sublauncher.
func (p *pubsubLauncher) SimpleDescription() string {
	return "starts ADK PubSub trigger endpoint server"
}

// SetupSubrouters adds the PubSub trigger endpoint to the parent router.
func (p *pubsubLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	triggerConfig := triggers.TriggerConfig{
		MaxRetries:        p.config.triggerMaxRetries,
		BaseDelay:         p.config.triggerBaseDelay,
		MaxDelay:          p.config.triggerMaxDelay,
		MaxConcurrentRuns: p.config.triggerMaxRuns,
	}

	controller := triggers.NewPubSubController(
		config.SessionService,
		config.AgentLoader,
		config.MemoryService,
		config.ArtifactService,
		config.PluginConfig,
		triggerConfig,
	)

	subrouter := router
	if p.config.pathPrefix != "" && p.config.pathPrefix != "/" {
		subrouter = router.PathPrefix(p.config.pathPrefix).Subrouter()
	}

	subrouter.HandleFunc("/apps/{app_name}/trigger/pubsub", controller.PubSubTriggerHandler).Methods(http.MethodPost)
	return nil
}

// UserMessage implements web.Sublauncher.
func (p *pubsubLauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("       pubsub:  PubSub trigger endpoint is available at %s%s/apps/{app_name}/trigger/pubsub", webURL, p.config.pathPrefix))
}
