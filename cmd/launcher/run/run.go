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
	"log"
	"os"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	"google.golang.org/adk/cmd/launcher/console"
	"google.golang.org/adk/cmd/launcher/web"
)

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

func BuildLauncher() (*launcher.Launcher, []string, error) {
	args := os.Args[1:] // skip file name

	if len(args) > 0 && args[0] == "web" {
		return web.BuildLauncher(args[1:])
	}
	return console.BuildLauncher(args[1:])
}
