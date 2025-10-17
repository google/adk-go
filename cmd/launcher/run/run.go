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

// package run provides the functionality of launching an agent in different ways (defined in the command line)
package run

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/console"
	"google.golang.org/adk/cmd/launcher/web"
)

// Run builds the launcher according to command-line arguments and then executes it
func Run(ctx context.Context, config *adk.Config) {
	l, _, err := BuildLauncher()
	if err != nil {
		log.Fatalf("cannot build launcher: %v", err)
	}
	err = (*l).Run(ctx, config)
	if err != nil {
		log.Fatalf("run failed: %v", err)
	}
}

// BuildLauncher uses command line argument to choose an appropiate launcher type and then builds it
func BuildLauncher() (*launcher.Launcher, []string, error) {
	args := os.Args[1:] // skip file name, safe

	if len(args) == 0 {
		return console.BuildLauncher(args)
	}
	// len(args) > 0
	switch args[0] {
	case "web":
		return web.BuildLauncher(args[1:])
	case "console":
		return console.BuildLauncher(args[1:])
	default:
		return nil, nil, fmt.Errorf("for the first argument want 'web', 'console' or nothing, got: %s", args[0])
	}
}
