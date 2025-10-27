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

// package api allows to run ADK REST API alone
package api

import (
	"flag"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/adk"
	restapiweb "google.golang.org/adk/cmd/restapi/web"
)

// ApiConfig contains parametres for lauching ADK REST API
type ApiConfig struct {
	frontendAddress string
}

// ApiLauncher can launch ADK REST API
type ApiLauncher struct {
	flags  *flag.FlagSet
	config *ApiConfig
}

// Adds CORS headers which allow calling ADK REST API from another web app (like ADK WebUI)
func CorsWithArgs(frontendAddress string) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", frontendAddress)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func (a *ApiLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	printer(fmt.Sprintf("       api:  you can access API using %s/api", webUrl))
	printer(fmt.Sprintf("       api:      for instance: %s/api/list-apps", webUrl))
}

// SetupSubrouters adds api router to the parent router
func (a *ApiLauncher) SetupSubrouters(router *mux.Router, adkConfig *adk.Config) {
	rApi := router.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix("/api/").Subrouter()
	restapiweb.SetupRouter(rApi, adkConfig)
	rApi.Use(CorsWithArgs(a.config.frontendAddress))
}

// ReplaceRouter replaces parent router
func (a *ApiLauncher) SetupRoutes(router *mux.Router, adkConfig *adk.Config) {
	// api doesn't change the top level routes
}

func (a *ApiLauncher) FormatSyntax() string {
	return launcher.FormatFlagUsage(a.flags)
}

func (a *ApiLauncher) Keyword() string {
	return "api"
}

func (a *ApiLauncher) Parse(args []string) ([]string, error) {
	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse api flags: %v", err)
	}
	restArgs := a.flags.Args()
	return restArgs, nil
}

func (a *ApiLauncher) SimpleDescription() string {
	return "starts ADK REST API server, accepting origins specified by webui_address (CORS)"
}

// NewLauncher creates new api launcher. It extends Web launcher
func NewLauncher() *ApiLauncher {
	config := &ApiConfig{}

	fs := flag.NewFlagSet("web", flag.ContinueOnError)
	fs.StringVar(&config.frontendAddress, "webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")

	return &ApiLauncher{
		config: config,
		flags:  fs,
	}
}

// // ParseArgs returns a config from parsed arguments and the remaining un-parsed arguments
// func ParseArgs(args []string) (*ApiConfig, []string, error) {
// 	fs := flag.NewFlagSet("api", flag.ContinueOnError)

// 	localPortFlag := fs.Int("port", 8080, "Localhost port for the server")
// 	frontendAddressFlag := fs.String("webui_address", "localhost:8080", "ADK WebUI address as seen from the user browser. It's used to allow CORS requests. Please specify only hostname and (optionally) port.")

// 	err := fs.Parse(args)
// 	if err != nil || !fs.Parsed() {
// 		return &(ApiConfig{}), nil, fmt.Errorf("failed to parse flags: %v", err)
// 	}

// 	res := &ApiConfig{
// 		port:            *localPortFlag,
// 		frontendAddress: *frontendAddressFlag,
// 	}
// 	return res, fs.Args(), nil
// }

// // BuildLauncher parses command line args and returns ready-to-run console launcher.
// func BuildLauncher(args []string) (launcher.Launcher, []string, error) {
// 	apiConfig, argsLeft, err := ParseArgs(args)
// 	if err != nil {
// 		return nil, nil, fmt.Errorf("cannot parse arguments for api: %v: %w", args, err)
// 	}
// 	return &ApiLauncher{config: apiConfig}, argsLeft, nil
// }

// func AddSubrouter(router *mux.Router, pathPrefix string, adkConfig *adk.Config, frontendAddress string) {
// 	rApi := router.Methods("GET", "POST", "DELETE", "OPTIONS").PathPrefix(pathPrefix).Subrouter()
// 	restapiweb.SetupRouter(rApi, adkConfig)
// 	rApi.Use(web.CorsWithArgs(frontendAddress))
// }

// func (l ApiLauncher) Run(ctx context.Context, config *adk.Config) error {
// 	// we need some session service, add one if missing
// 	if config.SessionService == nil {
// 		config.SessionService = session.InMemoryService()
// 	}

// 	router := web.BuildBaseRouter()
// 	AddSubrouter(router, "/api/", config, l.config.frontendAddress)

// 	log.Printf("Starting the ADK REST API server: %+v", l.config)
// 	log.Println()
// 	log.Printf("Open %s", "http://localhost:"+strconv.Itoa(l.config.port))
// 	log.Println()
// 	return http.ListenAndServe(":"+strconv.Itoa(l.config.port), router)
// }

// // Run parses command line params, prepares api launcher and runs it
// func Run(ctx context.Context, config *adk.Config) error {
// 	// skip args[0] - executable file name
// 	// skip unparsed arguments returned by BuildLauncher
// 	launcherToRun, _, err := BuildLauncher(os.Args[1:])
// 	if err != nil {
// 		log.Fatalf("cannot build api launcher: %v", err)
// 	}
// 	err = launcherToRun.Run(ctx, config)
// 	if err != nil {
// 		log.Fatalf("run failed: %v", err)
// 	}
// 	return nil
// }
