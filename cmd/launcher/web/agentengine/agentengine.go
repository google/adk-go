// Copyright 2026 Google LLC
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

// Package agentengine provides a sublauncher that provides web interface as required by Agent Engine
package agentengine

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/internal/cli/util"
)

// agentEngineConfig contains parameters for launching ADK Agent Engine server
type agentEngineConfig struct{}

type agentEngineLauncher struct {
	flags        *flag.FlagSet // flags are used to parse command-line arguments
	config       *agentEngineConfig
	sublaunchers []web.Sublauncher
	// maps keyword to sublauncher for the keywords parsed from command line
	activeSublaunchers map[string]web.Sublauncher
	router             *mux.Router
}

// NewLauncher creates new Agent Engine launcher. It extends Web launcher
func NewLauncher(sublaunchers ...web.Sublauncher) web.Sublauncher {
	config := &agentEngineConfig{}

	fs := flag.NewFlagSet("agentengine", flag.ContinueOnError)

	return &agentEngineLauncher{
		config:       config,
		flags:        fs,
		sublaunchers: sublaunchers,
		router:       web.BuildBaseRouter(),
	}
}

// CommandLineSyntax implements web.Sublauncher. Returns the command-line syntax for the agentEngine launcher.
func (a *agentEngineLauncher) CommandLineSyntax() string {
	var b strings.Builder
	fmt.Fprint(&b, util.FormatFlagUsage(a.flags))
	fmt.Fprintf(&b, "  You may specify sublaunchers:\n")
	for _, l := range a.sublaunchers {
		fmt.Fprintf(&b, "        * %s - %s\n", l.Keyword(), l.SimpleDescription())
	}
	fmt.Fprintf(&b, "  Sublaunchers syntax:\n")
	for _, l := range a.sublaunchers {
		fmt.Fprintf(&b, "        %s\n  %s\n", l.Keyword(), l.CommandLineSyntax())
	}
	return b.String()
}

// Keyword implements web.Sublauncher. Returns the command-line keyword for A2A launcher.
func (a *agentEngineLauncher) Keyword() string {
	return "agentengine"
}

func (a *agentEngineLauncher) Parse(args []string) ([]string, error) {
	keyToSublauncher := make(map[string]web.Sublauncher)
	for _, l := range a.sublaunchers {
		if _, ok := keyToSublauncher[l.Keyword()]; ok {
			return nil, fmt.Errorf("cannot create agentengine launcher. Keywords for sublaunchers should be unique and they are not: '%s'", l.Keyword())
		}
		keyToSublauncher[l.Keyword()] = l
	}

	err := a.flags.Parse(args)
	if err != nil || !a.flags.Parsed() {
		return nil, fmt.Errorf("failed to parse agentengine flags: %v", err)
	}

	restArgs := a.flags.Args()
	a.activeSublaunchers = make(map[string]web.Sublauncher)

	for len(restArgs) > 0 {
		keyword := restArgs[0]
		if _, ok := a.activeSublaunchers[keyword]; ok {
			// already processed
			return restArgs, fmt.Errorf("the keyword %q is specified and processed more than once, which is not allowed", keyword)
		}

		if sublauncher, ok := keyToSublauncher[keyword]; ok {
			// skip the keyword and move on
			restArgs, err = sublauncher.Parse(restArgs[1:])
			if err != nil {
				return nil, fmt.Errorf("the %q launcher cannot parse arguments: %v", keyword, err)
			}
			a.activeSublaunchers[keyword] = sublauncher
		} else {
			// not known keyword, let it be processed elsewhere
			break
		}
	}
	return restArgs, nil
}

func (a *agentEngineLauncher) handleApiReasoningEngine(w http.ResponseWriter, r *http.Request) {
	_, _ = fmt.Fprintf(w, "Hello from handleApiReasoningEngine: Not implemented")
}

type payload struct {
	ClassMethod string `json:"class_method"`
	Input       any    `json:"input"`
}

func logReq(req *http.Request, body []byte) {
	rb, err := json.Marshal(struct {
		Method  string
		URL     string
		Headers any
		Body    any
	}{
		Method:  req.Method,
		URL:     req.URL.String(),
		Body:    string(body),
		Headers: req.Header,
	})
	if err != nil {
		log.Printf("json.Marshal() failed: %v", err)
		return
	}
	log.Printf("request: %v\n", string(rb))
}

type FakeWriter struct{}

// Header implements [http.ResponseWriter].
func (f *FakeWriter) Header() http.Header {
	res := make(http.Header)
	res.Set("Content-Type", "application/json")
	return res
}

// WriteHeader implements [http.ResponseWriter].
func (f *FakeWriter) WriteHeader(statusCode int) {
	log.Printf("WriteHeader(): statusCode= %v", statusCode)
}

var _ http.ResponseWriter = &FakeWriter{}

func (f *FakeWriter) Write(p []byte) (n int, err error) {
	log.Printf("Write(): p= %v", p)
	s := string(p)
	log.Printf("Write(): s= %v", s)

	return len(p), nil
}

func (a *agentEngineLauncher) handleApiSteamReasoningEngine(w http.ResponseWriter, r *http.Request) {
	_, _ = fmt.Fprintf(w, "Hello from handleApiSteamReasoningEngine")

	if r == nil {
		log.Printf("req is nil")
		return
	}
	if r.Body == nil {
		log.Printf("req.Body is nil")
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("io.ReadAll() failed: %v", err)
		return
	}

	log.Printf("Body string: %v", string(body))

	// {"class_method":"create_session","input":{"user_id":"u_123"}}
	var p payload
	err = json.Unmarshal(body, &p)
	if err != nil {
		log.Printf("json.Unmarshal() failed: %v", err)
		return
	}
	log.Printf("Payload: %+v", p)

	logReq(r, body)

	switch p.ClassMethod {
	case "create_session":
		log.Printf("Hello from create_session")
		req, err := http.NewRequest("POST", "/api/apps/{app_name}/users/{user_id}/sessions", nil)
		if err != nil {
			log.Printf("http.NewRequest() failed: %v", err)
			return
		}

		fw := &FakeWriter{}
		a.router.ServeHTTP(fw, req)
		// a.router

		// crr := &session.CreateRequest{
		// 	AppName: "app",
		// 	UserID:  "user",
		// }
		// ressession, err := s.ss.Create(r.Context(), crr)
		// if err != nil {
		// 	log.Printf("s.ss.Create() failed: %v", err)
		// 	return
		// }
		// log.Printf("ressession: %+v", ressession)
		// resp := fmt.Sprintf(`{"session_id":"%v}`, ressession.Session.ID())
		// w.Write([]byte(resp))

	default:
		log.Printf("unrecognized class method: %v", p.ClassMethod)
	}

}

// SetupSubrouters implements the web.Sublauncher interface.
func (a *agentEngineLauncher) SetupSubrouters(router *mux.Router, config *launcher.Config) error {
	router.HandleFunc("/api/reasoning_engine", a.handleApiReasoningEngine)
	router.HandleFunc("/api/stream_reasoning_engine", a.handleApiSteamReasoningEngine)

	// sublaunchers are hidden in a.router
	for _, l := range a.sublaunchers {
		if _, isActive := a.activeSublaunchers[l.Keyword()]; isActive {
			if err := l.SetupSubrouters(a.router, config); err != nil {
				return fmt.Errorf("%s subrouter setup failed: %v", l.Keyword(), err)
			}
		}
	}
	return nil
}

// SimpleDescription implements web.Sublauncher
func (a *agentEngineLauncher) SimpleDescription() string {
	// TODO(kdroste) description
	return "starts AgentEngine server which handles ??????????"
}

// UserMessage implements web.Sublauncher.
func (a *agentEngineLauncher) UserMessage(webUrl string, printer func(v ...any)) {
	// TODO(kdroste) description
	printer(fmt.Sprintf("       agentEngine:  you can access this server ????????????????????????????????: %s", webUrl))
}
