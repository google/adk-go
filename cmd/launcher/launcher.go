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
package launcher

import (
	"context"
	"flag"
	"strings"

	"google.golang.org/adk/cmd/launcher/adk"
)

// execFile
//
//	launcher := universal.NewLauncher( console.NewLauncher())
//
// execFile console
//
//	launcher := universal.NewLauncher( console.NewLauncher())
//
// execFile console console-params
//
//	launcher := universal.NewLauncher( console.NewLauncher())
//
// execFile server sever-params api api-params
//
//	launcher := universal.NewLauncher( server.NewLauncher( api.NewLauncher() ) )
//
// execFile server sever-params webui webui-params
//
//	launcher := universal.NewLauncher( server.NewLauncher( webuil.NewLauncher() ) )
//
// execFile server sever-params a2a a2a-params
//
//	launcher := universal.NewLauncher( server.NewLauncher( a2a.NewLauncher() ))
//
// execFile server sever-params api api-params webui webui-params a2a a2a-params
//
//	   launcher := universal.NewLauncher( server.NewLauncher(api.NewLauncher(), webui.NewLauncher(), a2a.NewLauncher() ))
//	   launcher := universal.NewLauncher( console.NewLaucher(), server.NewLauncher(api.NewLauncher(), webui.NewLauncher(), a2a.NewLauncher() ))
//	   launcher := universal.NewLauncher( console.NewLaucher(), server.NewLauncher(api.NewLauncher(), a2a.NewLauncher() ))
//	   launcher := universal.NewLauncher( server.NewLauncher( a2a.NewLauncher() ))
//	   launcher := server.NewLauncher( a2a.NewLauncher() )
//	   launcher := server.NewLauncher( api.NewLauncher() )
//	   launcher := server.NewLauncher( api.NewLauncher(), webui.NewLauncher() )

//     all / :
//           func AllInServer() launcher.Launcher {
//				universal.NewLauncher( console.NewLaucher(), server.NewLauncher(api.NewLauncher(), webui.NewLauncher(), a2a.NewLauncher() ))
//           }
//     prod:
//           func ProdServer() launcher.Launcher {
//				universal.NewLauncher( server.NewLauncher(api.NewLauncher(), a2a.NewLauncher() ))
//           }
//
//
//    options:
//

//     webui.New( server.New())
//
//     server := server.New()
//     a2a.New(server)     ===   server.Extend(a2a.New())
//     webui.New(server)

//	args, err := launcher.Parse( os.Args() )

//	launcher.Launch()
//
// Launcher allowes to launch console or web application
//  execution:
//    parse command line
//    execute external parseRemaining
//    run

type Launcher interface {
	Sublauncher
	Run(ctx context.Context, config *adk.Config) error
	ParseAndRun(ctx context.Context, config *adk.Config, args []string, parseRemaining func([]string) error) error
}

type Sublauncher interface {
	//Run(ctx context.Context, config *adk.Config) error
	Keyword() string
	Parse(args []string) ([]string, error)
	FormatSyntax() string
	SimpleDescription() string
}

func FormatFlagUsage(fs *flag.FlagSet) string {
	var b strings.Builder
	o := fs.Output()
	fs.SetOutput(&b)
	fs.PrintDefaults()
	fs.SetOutput(o)
	return b.String()
}
