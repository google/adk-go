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

package appinfo_test

import (
	"log"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/web"
	"google.golang.org/adk/v2/cmd/launcher/web/a2a"
	"google.golang.org/adk/v2/cmd/launcher/web/agentengine"
	"google.golang.org/adk/v2/cmd/launcher/web/api"
	"google.golang.org/adk/v2/cmd/launcher/web/triggers/eventarc"
	"google.golang.org/adk/v2/cmd/launcher/web/triggers/pubsub"
	"google.golang.org/adk/v2/cmd/launcher/web/webui"
	"google.golang.org/adk/v2/exp/appinfo"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
)

// sublauncherArgs is one sublauncher and the arguments it is parsed with.
type sublauncherArgs struct {
	l    web.Sublauncher
	args []string
}

// setUp parses each sublauncher's arguments and mounts them on a base router
// in the order given, as the web launcher does, and returns the router and the
// first setup error.
func setUp(t *testing.T, config *launcher.Config, sublaunchers ...sublauncherArgs) (*mux.Router, error) {
	t.Helper()
	router := web.BuildBaseRouter()
	for _, s := range sublaunchers {
		if _, err := s.l.Parse(s.args); err != nil {
			t.Fatalf("%s: Parse(%q) failed: %v", s.l.Keyword(), s.args, err)
		}
	}
	for _, s := range sublaunchers {
		if err := s.l.SetupSubrouters(router, config); err != nil {
			return router, err
		}
	}
	return router, nil
}

func newConfig(t *testing.T) *launcher.Config {
	t.Helper()
	return &launcher.Config{
		SessionService: session.InMemoryService(),
		AgentLoader:    newLoader(t, llmagent.Config{Name: "concierge", Description: "Plans trips.", Instruction: "Coordinate."}),
	}
}

func get(router http.Handler, req *http.Request) int {
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec.Code
}

// TestLauncherCommandLine pins the keyword and flag the Agents CLI starts a Go
// agent with: web ... api -path_prefix / a2a appinfo.
func TestLauncherCommandLine(t *testing.T) {
	l := appinfo.NewLauncher()
	if got := l.Keyword(); got != "appinfo" {
		t.Errorf("Keyword() = %q, want %q", got, "appinfo")
	}
	rest, err := l.Parse([]string{"-path_prefix", "/x", "a2a"})
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if !slices.Equal(rest, []string{"a2a"}) {
		t.Errorf("Parse returned %q, want the next keyword left for the web launcher: [a2a]", rest)
	}
}

// TestLauncherRejectsBraces checks a prefix the router would read as a route
// variable is refused rather than served under every value of it.
func TestLauncherRejectsBraces(t *testing.T) {
	for _, prefix := range []string{"/{x}", "/a{b", "/a}b"} {
		if _, err := appinfo.NewLauncher().Parse([]string{"-path_prefix", prefix}); err == nil {
			t.Errorf("Parse(-path_prefix %q) succeeded, want an error", prefix)
		}
	}
}

// TestLauncherInTemplateOrder mounts the sublaunchers in the order the Agents
// CLI template passes them to web.NewLauncher. The agentengine route, a POST
// catch-all at the root here, is registered before app-info and matches its
// path but not its method, so it does not shadow it.
func TestLauncherInTemplateOrder(t *testing.T) {
	router, err := setUp(t, newConfig(t),
		sublauncherArgs{webui.NewLauncher(), nil},
		sublauncherArgs{a2a.NewLauncher(), nil},
		sublauncherArgs{pubsub.NewLauncher(), nil},
		sublauncherArgs{eventarc.NewLauncher(), nil},
		sublauncherArgs{agentengine.NewLauncher("concierge"), []string{"-path_prefix", "/"}},
		sublauncherArgs{appinfo.NewLauncher(), nil},
		sublauncherArgs{api.NewLauncher(), []string{"-path_prefix", "/"}},
	)
	if err != nil {
		t.Fatalf("SetupSubrouters failed: %v", err)
	}
	if code := get(router, httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil)); code != http.StatusOK {
		t.Errorf("GET app-info: status = %d, want %d", code, http.StatusOK)
	}
}

func TestLauncherPathPrefix(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		path string
	}{
		{name: "default is the root", args: nil, path: "/apps/concierge/app-info"},
		{name: "explicit root", args: []string{"-path_prefix", "/"}, path: "/apps/concierge/app-info"},
		{name: "prefix", args: []string{"-path_prefix", "/api"}, path: "/api/apps/concierge/app-info"},
		{name: "slashes normalized", args: []string{"-path_prefix", "api/"}, path: "/api/apps/concierge/app-info"},
		{name: "nested prefix", args: []string{"-path_prefix", "/a/b/"}, path: "/a/b/apps/concierge/app-info"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, err := setUp(t, newConfig(t), sublauncherArgs{appinfo.NewLauncher(), tc.args})
			if err != nil {
				t.Fatalf("SetupSubrouters failed: %v", err)
			}
			if code := get(router, httptest.NewRequest(http.MethodGet, tc.path, nil)); code != http.StatusOK {
				t.Errorf("GET %s: status = %d, want %d", tc.path, code, http.StatusOK)
			}
		})
	}
}

// TestLauncherOrderWithAPI covers registration order against the real api
// sublauncher, whose route matches every path under its prefix.
func TestLauncherOrderWithAPI(t *testing.T) {
	for _, tc := range []struct {
		name         string
		apiFirst     bool
		apiArgs      []string
		appInfoArgs  []string
		appInfoPath  string
		wantSetupErr bool
	}{
		{
			name:        "before api on the same prefix",
			apiArgs:     []string{"-path_prefix", "/"},
			appInfoPath: "/apps/concierge/app-info",
		},
		{
			name:         "after api on the same prefix",
			apiFirst:     true,
			apiArgs:      []string{"-path_prefix", "/"},
			wantSetupErr: true,
		},
		{
			name:         "after api on its default prefix",
			apiFirst:     true,
			appInfoArgs:  []string{"-path_prefix", "/api"},
			wantSetupErr: true,
		},
		{
			name:        "after api on another prefix",
			apiFirst:    true,
			appInfoPath: "/apps/concierge/app-info",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sublaunchers := []sublauncherArgs{{appinfo.NewLauncher(), tc.appInfoArgs}, {api.NewLauncher(), tc.apiArgs}}
			if tc.apiFirst {
				slices.Reverse(sublaunchers)
			}
			router, err := setUp(t, newConfig(t), sublaunchers...)
			if tc.wantSetupErr {
				if err == nil {
					t.Fatal("SetupSubrouters succeeded, want an error for a shadowed route")
				}
				return
			}
			if err != nil {
				t.Fatalf("SetupSubrouters failed: %v", err)
			}

			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.appInfoPath, nil))
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"rootAgentName":"concierge"`) {
				t.Errorf("GET %s: status = %d, body %q, want 200 with the app info", tc.appInfoPath, rec.Code, rec.Body)
			}
		})
	}
}

// TestLauncherAuthenticates checks the route applies the configured
// Authenticator, and that a refused request does not use up the experimental
// warning. HEAD is served too, as the REST API serves it for every GET route.
func TestLauncherAuthenticates(t *testing.T) {
	var logs strings.Builder
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(prevOut)
		log.SetFlags(prevFlags)
	})

	config := newConfig(t)
	config.Authenticator = authn.NewHeader("X-User")
	router, err := setUp(t, config, sublauncherArgs{appinfo.NewLauncher(), nil})
	if err != nil {
		t.Fatalf("SetupSubrouters failed: %v", err)
	}
	request := func(method string, authenticated bool) int {
		req := httptest.NewRequest(method, "/apps/concierge/app-info", nil)
		if authenticated {
			req.Header.Set("X-User", "alice")
		}
		return get(router, req)
	}

	if code := request(http.MethodGet, false); code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated GET: status = %d, want %d", code, http.StatusUnauthorized)
	}
	if strings.Contains(logs.String(), "[EXPERIMENTAL]") {
		t.Errorf("a refused request logged the warning:\n%s", logs.String())
	}
	for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodGet} {
		if code := request(method, true); code != http.StatusOK {
			t.Errorf("authenticated %s: status = %d, want %d", method, code, http.StatusOK)
		}
	}
	// Counted rather than compared whole: the base router logs every request.
	if got := strings.Count(logs.String(), "[EXPERIMENTAL]"); got != 1 {
		t.Errorf("the warning was logged %d times, want once:\n%s", got, logs.String())
	}
}

// TestLauncherHostAndOriginChecks checks the route refuses what the REST API
// refuses: a rebound Host on a loopback bind, and a cross-origin browser.
func TestLauncherHostAndOriginChecks(t *testing.T) {
	for _, tc := range []struct {
		name     string
		bindHost string
		host     string
		origin   string
		want     int
	}{
		{name: "loopback host", bindHost: "127.0.0.1", host: "localhost:8080", want: http.StatusOK},
		{name: "rebound host", bindHost: "127.0.0.1", host: "attacker.example", want: http.StatusForbidden},
		{name: "rebound host on a non-loopback bind", bindHost: "0.0.0.0", host: "attacker.example", want: http.StatusOK},
		{name: "same origin", bindHost: "127.0.0.1", host: "localhost:8080", origin: "http://localhost:8080", want: http.StatusOK},
		{name: "cross origin", bindHost: "127.0.0.1", host: "localhost:8080", origin: "http://attacker.example", want: http.StatusForbidden},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := newConfig(t)
			config.BindHost = tc.bindHost
			router, err := setUp(t, config, sublauncherArgs{appinfo.NewLauncher(), nil})
			if err != nil {
				t.Fatalf("SetupSubrouters failed: %v", err)
			}
			req := httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil)
			req.Host = tc.host
			if tc.origin != "" {
				req.Header.Set("Origin", tc.origin)
			}
			if code := get(router, req); code != tc.want {
				t.Errorf("status = %d, want %d", code, tc.want)
			}
		})
	}
}
