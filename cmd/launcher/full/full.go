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

// Package full is a utility to quickly access all functionality of ADK while
// developing.  Out of the box, it is not suitable for production use; it serves
// the REST API without authentication.
//
// By default, it will launch ADK in console interaction mode.  Use
// `--adk=web` to launch the web API.
package full

import (
	"context"
	"flag"
	"fmt"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/console"
	"google.golang.org/adk/cmd/launcher/web"
)

// FlagConfig contains all flag values that need to be filled out in order to
// call full.Run.  It combines flags from the console and web sub-launchers.
type FlagConfig struct {
	RunWhat string
	Console console.Config
	Web     web.Config
}

func DefineFlagsVar(cfg *FlagConfig) {
	flag.StringVar(&cfg.RunWhat, "adk", "console", "Which interface to run.  One of \"console\" or \"web\"")
	console.DefineFlags(&cfg.Console)
	web.DefineFlags(&cfg.Web)
}

// DefineFlags uses the flag package to define command-line flags needed by the
// console and web sub-launchers, and returns a FlagConfig holding all of their
// bound values.
func DefineFlags() *FlagConfig {
	cfg := &FlagConfig{}
	DefineFlagsVar(cfg)
	return cfg
}

// Run launches the provided agent in either console or web mode.
func Run(ctx context.Context, flagConfig *FlagConfig, agentConfig *launcher.Config) error {
	switch flagConfig.RunWhat {
	case "console":
		clauncher := console.NewLauncher(&flagConfig.Console)
		if err := clauncher.Run(ctx, agentConfig); err != nil {
			return fmt.Errorf("while running console launcher: %w", err)
		}
		return nil
	case "web":
		wlauncher := web.NewLauncher(&flagConfig.Web)
		if err := wlauncher.Run(ctx, agentConfig); err != nil {
			return fmt.Errorf("while running web launcher: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("unrecognized launcher %q", flagConfig.RunWhat)
	}
}
