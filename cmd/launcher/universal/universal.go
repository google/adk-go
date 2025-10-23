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

// package launcher provides ways to interact with agents
package universal

import (
	"context"
	"fmt"
	"os"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/api"
	"google.golang.org/adk/cmd/launcher/apiweb"
	"google.golang.org/adk/cmd/launcher/console"
)

// Run builds the launcher according to command-line arguments and then executes it
func Run(ctx context.Context, config *adk.Config) error {
	args := os.Args[1:] // skip file name, safe

	// if there are no arguments - run console
	if len(args) == 0 {
		return console.Run(ctx, config)
	}

	var launcherToRun launcher.Launcher
	var err error

	switch args[0] {
	case "api":
		launcherToRun, _, err = api.BuildLauncher(args[1:])
	case "apiweb":
		launcherToRun, _, err = apiweb.BuildLauncher(args[1:])
	case "console":
		launcherToRun, _, err = console.BuildLauncher(args[1:])
	default:
		return fmt.Errorf("universal launcher requires either no arguments (which will run console version) or one of 'api', 'apiweb' or 'console', got: %s", args[0])
	}
	if err != nil {
		return fmt.Errorf("cannot build launcher for %s: %v", args[0], err)
	}

	err = launcherToRun.Run(ctx, config)
	if err != nil {
		return fmt.Errorf("run failed for %s launcher: %v", args[0], err)
	}
	return nil
}

// BuildLauncher uses command line argument to choose an appropiate launcher type and then builds it, returning the remaining un-parsed arguments
func BuildLauncher() (launcher.Launcher, []string, error) {
	args := os.Args[1:] // skip file name, safe

	if len(args) == 0 {
		return console.BuildLauncher(args)
	}
	// len(args) > 0
	switch args[0] {
	case "api":
		return api.BuildLauncher(args[1:])
	case "apiweb":
		return apiweb.BuildLauncher(args[1:])
	case "console":
		return console.BuildLauncher(args[1:])
	default:
		return nil, nil, fmt.Errorf("for the first argument want 'web', 'console' or nothing, got: %s", args[0])
	}
}
