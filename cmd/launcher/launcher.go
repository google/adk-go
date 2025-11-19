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

// Package launcher provides ways to interact with agents
package launcher

import (
	"context"

	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
)

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
	SessionService  session.Service
	ArtifactService artifact.Service
	MemoryService   memory.Service
	AgentLoader     agent.Loader
	A2AOptions      []a2asrv.RequestHandlerOption
}
