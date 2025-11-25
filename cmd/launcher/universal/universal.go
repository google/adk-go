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

// Package universal provides an umbrella over launchers (console and web).
// It allowes to choose one launcher by command-line parameters and uses it to parse the rest of arguments and then execute the launcher
package universal

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/adk/cmd/launcher"
)

// uniLauncher contains information about sublaunchers
type uniLauncher struct {
	chosenLauncher launcher.SubLauncher // the chosen launcher - after parsing command-line args
	sublaunchers   []launcher.SubLauncher
}

// Execute implements launcher.Launcher. Parses args and runs the chosen launcher. Returns error if there are non-parsed arguments.
func (l *uniLauncher) Execute(ctx context.Context, config *launcher.Config, args []string) error {
	return l.ParseAndRun(ctx, config, args, ErrorOnUnparsedArgs)
}

// NewLauncher returns a new universal launcher. The first element on launcher list will be the default one if there are no arguments specified
func NewLauncher(sublaunchers ...launcher.SubLauncher) launcher.Launcher {
	return &uniLauncher{
		sublaunchers: sublaunchers,
	}
}

// ParseAndRun parses arguments and runs the chosen sublauncher. It allows to provide a
// hook for processing any remaining arguments.
func (l *uniLauncher) ParseAndRun(ctx context.Context, config *launcher.Config, args []string, parseRemaining func([]string) error) error {
	remainingArgs, err := l.parse(args)
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
	return l.run(ctx, config)
}

// run executes the chosen sublauncher.
func (l *uniLauncher) run(ctx context.Context, config *launcher.Config) error {
	return l.chosenLauncher.Run(ctx, config)
}

// parse parses arguments and remembers which sublauncher should be run later
func (l *uniLauncher) parse(args []string) ([]string, error) {
	keyToSublauncher := make(map[string]launcher.SubLauncher)
	for _, l := range l.sublaunchers {
		if _, ok := keyToSublauncher[l.Keyword()]; ok {
			return nil, fmt.Errorf("cannot create universal launcher. Keywords for sublaunchers should be unique and they are not: '%s'", l.Keyword())
		}
		keyToSublauncher[l.Keyword()] = l
	}

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
	if keyLauncher, ok := keyToSublauncher[key]; ok {
		// match found, use it, continue parsing without the matching keyword
		l.chosenLauncher = keyLauncher
		return l.chosenLauncher.Parse(args[1:])
	}
	// no match found,
	return l.chosenLauncher.Parse(args)
}

// CommandLineSyntax implements launcher.Launcher.
func (l *uniLauncher) CommandLineSyntax() string {
	if len(l.sublaunchers) == 0 {
		// no sub launchers
		return l.simpleDescription() + "\n\nThere are no sublaunchers to format syntax for."
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Arguments: Specify one of the following:\n")
	for _, l := range l.sublaunchers {
		fmt.Fprintf(&b, "  * %s - %s\n", l.Keyword(), l.SimpleDescription())
	}
	fmt.Fprintf(&b, "Details:\n")
	for _, l := range l.sublaunchers {
		fmt.Fprintf(&b, "  %s\n%s\n", l.Keyword(), l.CommandLineSyntax())
	}

	return b.String()
}

// simpleDescription provides a brief explanation of the universal launcher.
func (l *uniLauncher) simpleDescription() string {
	return `Universal launcher acts as a router, routing command line arguments to one of its sublaunchers. 
	The sublauncher is chosen by the first argument - a keyword. 
	If there are no arguments at all or the first one is not recognized by any of the sublaunchers, the first sublauncher is used.`
}

// ErrorOnUnparsedArgs returns an error if there are any unparsed arguments left.
func ErrorOnUnparsedArgs(args []string) error {
	if len(args) > 0 {
		return fmt.Errorf("cannot parse following arguments: %v", args)
	}
	return nil
}
