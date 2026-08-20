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

// Package launcher provides ways to interact with agents.
package launcher

import (
	"context"
	"fmt"

	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
	"google.golang.org/adk/v2/telemetry"
)

// Validate reports a Config that cannot work, before anything starts serving.
//
// The compaction config is validated inside runner.New, and a runner is built
// per request, so without a check here an unusable setting produces a process
// that starts cleanly and then fails every request with an error naming nothing
// the operator can act on.
func (c *Config) Validate() error {
	if c == nil {
		return nil
	}
	if err := c.EventsCompactionConfig.Validate(); err != nil {
		return fmt.Errorf("invalid EventsCompactionConfig: %w", err)
	}
	return nil
}

// Launcher is the main interface for running an ADK application.
// It is responsible for parsing command-line arguments and executing the
// corresponding logic.
type Launcher interface {
	// Execute parses command-line arguments and runs the launcher.
	Execute(ctx context.Context, config *Config, args []string) error
	// CommandLineSyntax returns a string describing the command-line flags and arguments.
	CommandLineSyntax() string
}

// SubLauncher is an interface for launchers that can be composed within a parent
// launcher, like the universal launcher. Each SubLauncher corresponds to a
// specific mode of operation (e.g., 'console' or 'web').
type SubLauncher interface {
	// Keyword returns the command-line keyword that activates this sub-launcher.
	Keyword() string
	// Parse parses the arguments for the sub-launcher. It should return any unparsed arguments.
	Parse(args []string) ([]string, error)
	// CommandLineSyntax returns a string describing the command-line flags and arguments for the sub-launcher.
	CommandLineSyntax() string
	// SimpleDescription provides a brief, one-line description of the sub-launcher's function.
	SimpleDescription() string
	// Run executes the sub-launcher's main logic.
	Run(ctx context.Context, config *Config) error
}

// Config contains parameters for web & console execution: sessions, artifacts, agents etc
type Config struct {
	SessionService   session.Service
	ArtifactService  artifact.Service
	MemoryService    memory.Service
	AgentLoader      agent.Loader
	A2AOptions       []a2asrv.RequestHandlerOption
	PluginConfig     runner.PluginConfig
	TelemetryOptions []telemetry.Option

	// EventsCompactionConfig enables context compaction for the sessions the
	// runners created here drive, replacing older events with summaries. Nil,
	// the default, disables compaction.
	//
	// The sliding window reduces prompt size by a constant factor rather than
	// bounding it. Only tail retention bounds growth, and only when the sliding
	// window is off: with both enabled the sliding window consumes the events
	// tail retention would summarize and it never fires. Enable one. See
	// [compaction.Config].
	//
	// This setting is process-wide. One launcher can serve many applications
	// through its agent loader, and they all get this config or none of them
	// do, including the same Summarizer instance and so the same model. If
	// different applications need different compaction, or must not share a
	// summarizer, run them separately.
	EventsCompactionConfig *compaction.Config
}
