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
	"strings"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
)

type LauncherConfig struct {
}

type Launcher struct {
	chosenLauncher   launcher.Launcher
	sublaunchers     []launcher.Launcher
	keyToSublauncher map[string]launcher.Launcher
}

// NewLauncher returns a new universal launcher. The first element on launcher list will be the default one if there are no arguments specified
func NewLauncher(sublaunchers ...launcher.Launcher) (*Launcher, error) {
	keyToSublauncher := make(map[string]launcher.Launcher)
	for _, l := range sublaunchers {
		if _, ok := keyToSublauncher[l.Keyword()]; ok {
			return nil, fmt.Errorf("cannot create universal launcher. Keywords for sublaunchers should be unique and they are not: '%s'", l.Keyword())
		}
		keyToSublauncher[l.Keyword()] = l
	}
	return &Launcher{
		sublaunchers:     sublaunchers,
		keyToSublauncher: keyToSublauncher,
	}, nil
}

func (l *Launcher) ParseAndRun(ctx context.Context, config *adk.Config, args []string, parseRemaining func([]string) error) error {
	remainingArgs, err := l.Parse(args)
	if err != nil {
		return err
	}
	if parseRemaining != nil {
		err = parseRemaining(remainingArgs)
		if err != nil {
			return err
		}
	}
	// args are parsed
	return l.Run(ctx, config)
}

func (l *Launcher) Run(ctx context.Context, config *adk.Config) error {
	return l.chosenLauncher.Run(ctx, config)
}

// Parse parses arguments and remembers which sublauncher should be run later
func (l *Launcher) Parse(args []string) ([]string, error) {
	if len(l.sublaunchers) == 0 {
		// no sub launchers
		return args, fmt.Errorf("there are no sub launchers to parse the arguments")
	}
	// default to the first one in the list
	l.chosenLauncher = l.sublaunchers[0]

	if len(args) == 0 {
		// execute the default one
		return l.chosenLauncher.Parse(args)
	}
	// there are arguments
	key := args[0]
	if keyLauncher, ok := l.keyToSublauncher[key]; ok {
		// match found, use it, continue parsing without the matching keyword
		l.chosenLauncher = keyLauncher
		return l.chosenLauncher.Parse(args[1:])
	}
	// no match found,
	return l.chosenLauncher.Parse(args)
}

func (l *Launcher) Keyword() string {
	// not important for universal launcher, it is not used
	return ""
}

func (l *Launcher) FormatSyntax() string {
	if len(l.sublaunchers) == 0 {
		// no sub launchers
		return l.SimpleDescription() + "\n\nThere are no sublaunchers to format syntax for."
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Arguments: Specify one of the following:\n")
	for _, l := range l.sublaunchers {
		fmt.Fprintf(&b, "  * %s - %s\n", l.Keyword(), l.SimpleDescription())
	}
	fmt.Fprintf(&b, "Details:\n")
	for _, l := range l.sublaunchers {
		fmt.Fprintf(&b, "  %s\n%s\n", l.Keyword(), l.FormatSyntax())
	}

	return b.String()
}

func (l *Launcher) SimpleDescription() string {
	return `Universal launcher acts as a router, routing command line arguments to one of it's sublaunchers. 
	The sublauncher is chosen by the first argument - a keyword. 
	If there are no arguments at all or the first one is not recognized by any of the sublaunchers, the first sublauncher is used.`
}

// // Run builds the launcher according to command-line arguments and then executes it
// func Run(ctx context.Context, config *adk.Config) error {
// 	args := os.Args[1:] // skip file name, safe

// 	// if there are no arguments - run console
// 	if len(args) == 0 {
// 		return console.Run(ctx, config)
// 	}

// 	// var launcherToRun launcher.Launcher
// 	// var err error

// 	launcherToRun, _, err := BuildLauncher()

// 	// switch args[0] {
// 	// case "api":
// 	// 	launcherToRun, _, err = api.BuildLauncher(args[1:])
// 	// case "apiweb":
// 	// 	launcherToRun, _, err = apiweb.BuildLauncher(args[1:])
// 	// case "console":
// 	// 	launcherToRun, _, err = console.BuildLauncher(args[1:])
// 	// default:
// 	// 	return fmt.Errorf("universal launcher requires either no arguments (which will run console version) or one of 'api', 'apiweb' or 'console', got: %s", args[0])
// 	// }
// 	if err != nil {
// 		return fmt.Errorf("cannot build launcher for %s: %v", args[0], err)
// 	}

// 	err = launcherToRun.Run(ctx, config)
// 	if err != nil {
// 		return fmt.Errorf("run failed for %s launcher: %v", args[0], err)
// 	}
// 	return nil
// }

// // BuildLauncher uses command line argument to choose an appropiate launcher type and then builds it, returning the remaining un-parsed arguments
// func BuildLauncher() (launcher.Launcher, []string, error) {
// 	args := os.Args[1:] // skip file name, safe

// 	if len(args) == 0 {
// 		return console.BuildLauncher(args)
// 	}
// 	// len(args) > 0
// 	switch args[0] {
// 	case "api":
// 		return api.BuildLauncher(args[1:])
// 	case "apiweb":
// 		return apiweb.BuildLauncher(args[1:])
// 	case "apiweba2a":
// 		return apiweba2a.BuildLauncher(args[1:])
// 	case "console":
// 		return console.BuildLauncher(args[1:])
// 	default:
// 		return nil, nil, fmt.Errorf("universal launcher requires either no arguments (which will run console version) or one of 'api', 'apiweb', 'apiweba2a' or 'console', got: %s", args[0])
// 	}
// }
