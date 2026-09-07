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

package web

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"go.opentelemetry.io/otel/sdk/resource"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/telemetry"
)

func TestH2CFlag(t *testing.T) {
	for _, tc := range []struct {
		name    string
		args    []string
		wantH2C bool
	}{
		{
			name: "disabled by default",
		},
		{
			name:    "enabled",
			args:    []string{"--h2c"},
			wantH2C: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			launcher := NewLauncher().(*webLauncher)
			if _, err := launcher.Parse(tc.args); err != nil {
				t.Fatalf("Parse(%v) failed: %v", tc.args, err)
			}

			srv := launcher.buildHTTPServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Request-Protocol", r.Proto)
				w.WriteHeader(http.StatusNoContent)
			}))
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatalf("net.Listen() failed: %v", err)
			}
			serveErr := make(chan error, 1)
			go func() {
				serveErr <- srv.Serve(listener)
			}()
			t.Cleanup(func() {
				if err := srv.Close(); err != nil {
					t.Errorf("server Close() failed: %v", err)
				}
				if err := <-serveErr; err != http.ErrServerClosed {
					t.Errorf("server Serve() error = %v, want %v", err, http.ErrServerClosed)
				}
			})

			url := "http://" + listener.Addr().String()
			assertProtocol(t, http.DefaultClient, url, 1)

			h2cProtocols := new(http.Protocols)
			h2cProtocols.SetUnencryptedHTTP2(true)
			h2cClient := &http.Client{
				Transport: &http.Transport{Protocols: h2cProtocols},
			}
			t.Cleanup(h2cClient.CloseIdleConnections)

			resp, err := h2cClient.Get(url)
			if !tc.wantH2C {
				if err == nil {
					if closeErr := resp.Body.Close(); closeErr != nil {
						t.Errorf("response body Close() failed: %v", closeErr)
					}
					t.Fatalf("h2c request unexpectedly succeeded with protocol %q", resp.Proto)
				}
				return
			}
			if err != nil {
				t.Fatalf("h2c request failed: %v", err)
			}
			defer func() {
				if err := resp.Body.Close(); err != nil {
					t.Errorf("response body Close() failed: %v", err)
				}
			}()
			if resp.ProtoMajor != 2 {
				t.Errorf("h2c response protocol = %q, want HTTP/2", resp.Proto)
			}
			if got := resp.Header.Get("X-Request-Protocol"); got != "HTTP/2.0" {
				t.Errorf("handler request protocol = %q, want %q", got, "HTTP/2.0")
			}
		})
	}
}

func assertProtocol(t *testing.T, client *http.Client, url string, wantMajor int) {
	t.Helper()

	resp, err := client.Get(url)
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Errorf("response body Close() failed: %v", err)
		}
	}()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("reading response body failed: %v", err)
	}
	if resp.ProtoMajor != wantMajor {
		t.Errorf("response protocol = %q, want HTTP/%d", resp.Proto, wantMajor)
	}
}

type telemetryFailSublauncher struct{}

func (telemetryFailSublauncher) Keyword() string { return "repro" }

func (telemetryFailSublauncher) Parse(args []string) ([]string, error)             { return args, nil }
func (telemetryFailSublauncher) CommandLineSyntax() string                         { return "" }
func (telemetryFailSublauncher) SimpleDescription() string                         { return "" }
func (telemetryFailSublauncher) UserMessage(webURL string, printer func(v ...any)) {}
func (telemetryFailSublauncher) SetupSubrouters(r *mux.Router, c *launcher.Config) error {
	return nil
}

// TestRunDoesNotLeakListenerWhenTelemetryInitFails covers issue #1350: when
// telemetry initialization fails, Run must not leave an HTTP listener bound.
func TestRunDoesNotLeakListenerWhenTelemetryInitFails(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen() failed: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if err := ln.Close(); err != nil {
		t.Fatalf("listener Close() failed: %v", err)
	}

	l := NewLauncher(telemetryFailSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"--port", fmt.Sprint(port), "repro"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}

	// A resource whose schema URL conflicts with resource.Default()'s makes
	// resource.Merge fail, so telemetry init fails without touching the network.
	bad := resource.NewWithAttributes("https://conflicting.invalid/schema/v1")
	config := &launcher.Config{
		TelemetryOptions: []telemetry.Option{telemetry.WithResource(bad)},
	}

	if err := l.Run(context.Background(), config); err == nil {
		t.Fatalf("Run() succeeded, want telemetry initialization failure")
	}

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			t.Fatalf("port %d still bound after Run() returned an error: listener leaked", port)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestApplyServiceDefaultsFillsEmptyConfig(t *testing.T) {
	config := &launcher.Config{}

	applyServiceDefaults(config, true)

	if config.SessionService == nil {
		t.Error("SessionService is nil after applyServiceDefaults, want a default in-memory service")
	}
	if config.ArtifactService == nil {
		t.Error("ArtifactService is nil after applyServiceDefaults, want a default in-memory service")
	}
	if config.MemoryService == nil {
		t.Error("MemoryService is nil after applyServiceDefaults, want a default in-memory service")
	}
}

// TestApplyServiceDefaultsKeepsSuppliedServices covers the partial cases too:
// defaulting one service must not clobber the two the caller did supply, and
// supplying one must not stop the other two from being defaulted.
func TestApplyServiceDefaultsKeepsSuppliedServices(t *testing.T) {
	for _, tc := range []struct {
		name           string
		supplySession  bool
		supplyArtifact bool
		supplyMemory   bool
	}{
		{
			name:           "all supplied",
			supplySession:  true,
			supplyArtifact: true,
			supplyMemory:   true,
		},
		{
			name:          "only session supplied",
			supplySession: true,
		},
		{
			name:           "only artifact supplied",
			supplyArtifact: true,
		},
		{
			name:         "only memory supplied",
			supplyMemory: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := &launcher.Config{}
			var (
				wantSession  session.Service
				wantArtifact artifact.Service
				wantMemory   memory.Service
			)
			if tc.supplySession {
				wantSession = session.InMemoryService()
				config.SessionService = wantSession
			}
			if tc.supplyArtifact {
				wantArtifact = artifact.InMemoryService()
				config.ArtifactService = wantArtifact
			}
			if tc.supplyMemory {
				wantMemory = memory.InMemoryService()
				config.MemoryService = wantMemory
			}

			applyServiceDefaults(config, true)

			assertService(t, "SessionService", config.SessionService, wantSession)
			assertService(t, "ArtifactService", config.ArtifactService, wantArtifact)
			assertService(t, "MemoryService", config.MemoryService, wantMemory)
		})
	}
}

// TestApplyServiceDefaultsLogsWhatItDefaulted covers the diagnostic rather than
// the wiring. cmd/launcher/prod runs through the same Run path, so a deployment
// that meant to configure a durable artifact or memory service and did not gets
// a server that looks healthy and loses everything on restart. The log line is
// the only thing that says so.
func TestApplyServiceDefaultsLogsWhatItDefaulted(t *testing.T) {
	for _, tc := range []struct {
		name    string
		config  *launcher.Config
		want    []string
		notWant []string
	}{
		{
			name:   "nothing supplied",
			config: &launcher.Config{},
			want:   []string{"session", "artifact", "memory"},
		},
		{
			name:    "only session supplied",
			config:  &launcher.Config{SessionService: session.InMemoryService()},
			want:    []string{"artifact", "memory"},
			notWant: []string{"session"},
		},
		{
			name: "all supplied",
			config: &launcher.Config{
				SessionService:  session.InMemoryService(),
				ArtifactService: artifact.InMemoryService(),
				MemoryService:   memory.InMemoryService(),
			},
			notWant: []string{"session", "artifact", "memory"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			flags := log.Flags()
			log.SetOutput(&buf)
			log.SetFlags(0)
			t.Cleanup(func() {
				log.SetOutput(os.Stderr)
				log.SetFlags(flags)
			})

			applyServiceDefaults(tc.config, true)

			got := buf.String()
			if len(tc.want) == 0 && got != "" {
				t.Fatalf("applyServiceDefaults logged %q, want nothing: every service was supplied", got)
			}
			for _, name := range tc.want {
				if line := "No " + name + " service configured"; !strings.Contains(got, line) {
					t.Errorf("applyServiceDefaults logged %q, want a line starting %q", got, line)
				}
			}
			for _, name := range tc.notWant {
				if line := "No " + name + " service configured"; strings.Contains(got, line) {
					t.Errorf("applyServiceDefaults logged %q, want no %q line: the caller supplied it", got, line)
				}
			}
		})
	}
}

// assertService checks that applyServiceDefaults left a service set. When the
// caller supplied one, want is that value and the check is pointer identity:
// the default must not replace it.
func assertService(t *testing.T, name string, got, want any) {
	t.Helper()

	if got == nil {
		t.Errorf("%s is nil after applyServiceDefaults, want a default in-memory service", name)
		return
	}
	if want != nil && got != want {
		t.Errorf("%s = %p, want the caller-supplied service %p", name, got, want)
	}
}

// TestApplyServiceDefaultsServesRESTRoutes pins the reason the defaults exist.
// Before them, an artifact route reached a nil service and panicked, which
// dropped the TCP connection without sending any HTTP response at all.
//
// The assertion is 200, not merely "some status": the artifact controller has
// since grown its own nil guard that answers 503, so accepting any status would
// let the defaults disappear unnoticed. The session controller has no such
// guard, so its route still panics outright without them.
func TestApplyServiceDefaultsServesRESTRoutes(t *testing.T) {
	config := &launcher.Config{}
	applyServiceDefaults(config, true)

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  config.SessionService,
		ArtifactService: config.ArtifactService,
		MemoryService:   config.MemoryService,
		AgentLoader:     config.AgentLoader,
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}

	for _, tc := range []struct {
		name string
		path string
	}{
		{
			name: "list artifacts",
			path: "/apps/a/users/u/sessions/s/artifacts",
		},
		{
			name: "list sessions",
			path: "/apps/a/users/u/sessions",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveWithoutPanic(t, server, httptest.NewRequest(http.MethodGet, tc.path, nil))
			if rec.Code != http.StatusOK {
				t.Errorf("GET %s status = %d (%s), want %d", tc.path, rec.Code, rec.Body.String(), http.StatusOK)
			}
		})
	}
}

// TestApplyServiceDefaultsMemoryServiceIsCallable exercises the defaulted
// memory service. No REST route reaches it — it is handed to the runner and
// used by the load_memory tool mid-run — so the consequence is checked at the
// call site: a nil service panics there instead of returning an empty result.
func TestApplyServiceDefaultsMemoryServiceIsCallable(t *testing.T) {
	config := &launcher.Config{}
	applyServiceDefaults(config, true)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("SearchMemory on the defaulted memory service panicked: %v", r)
		}
	}()

	resp, err := config.MemoryService.SearchMemory(t.Context(), &memory.SearchRequest{
		AppName: "a",
		UserID:  "u",
		Query:   "anything",
	})
	if err != nil {
		t.Fatalf("SearchMemory() failed: %v", err)
	}
	if resp == nil {
		t.Error("SearchMemory() response is nil, want an empty result")
	}
}

// serveWithoutPanic serves one request and turns a handler panic into a named
// test failure, so a regression reports the route it broke instead of taking
// the whole test binary down with it.
func serveWithoutPanic(t *testing.T, handler http.Handler, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s %s panicked: %v", req.Method, req.URL.Path, r)
		}
	}()

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec
}

func TestRegisterHealthRoute(t *testing.T) {
	router := BuildBaseRouter()
	registerHealthRoute(router)

	t.Run("GET", func(t *testing.T) {
		rec := serveWithoutPanic(t, router, httptest.NewRequest(http.MethodGet, "/health", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /health status = %d, want %d", rec.Code, http.StatusOK)
		}

		var got map[string]string
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("GET /health body %q is not JSON: %v", rec.Body.String(), err)
		}
		if got["status"] != "ok" {
			t.Errorf("GET /health body = %q, want status %q", rec.Body.String(), "ok")
		}
		// adkrest serves /api/health with this exact Content-Type. A probe that
		// checks the header must not care which of the two paths it is given.
		if got, want := rec.Header().Get("Content-Type"), "application/json"; got != want {
			t.Errorf("GET /health Content-Type = %q, want %q", got, want)
		}
	})

	t.Run("HEAD", func(t *testing.T) {
		rec := serveWithoutPanic(t, router, httptest.NewRequest(http.MethodHead, "/health", nil))
		if rec.Code != http.StatusOK {
			t.Errorf("HEAD /health status = %d, want %d", rec.Code, http.StatusOK)
		}
	})
}

// TestBuildBaseRouterLeavesHealthToTheCaller guards the reason the route lives
// in registerHealthRoute: mux serves the first matching route, so registering
// /health inside the exported constructor would silently shadow an embedder's
// own handler for that path.
func TestBuildBaseRouterLeavesHealthToTheCaller(t *testing.T) {
	router := BuildBaseRouter()
	router.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}).Methods(http.MethodGet)

	rec := serveWithoutPanic(t, router, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != http.StatusTeapot {
		t.Errorf("GET /health status = %d, want %d: the embedder's handler was shadowed", rec.Code, http.StatusTeapot)
	}
}

// healthSublauncher registers its own root /health, the way a deployment with a
// real readiness check does.
type healthSublauncher struct{}

func (healthSublauncher) Keyword() string                                   { return "ownhealth" }
func (healthSublauncher) Parse(args []string) ([]string, error)             { return args, nil }
func (healthSublauncher) CommandLineSyntax() string                         { return "" }
func (healthSublauncher) SimpleDescription() string                         { return "" }
func (healthSublauncher) UserMessage(webURL string, printer func(v ...any)) {}
func (healthSublauncher) SetupSubrouters(r *mux.Router, c *launcher.Config) error {
	r.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("draining"))
	}).Methods(http.MethodGet)
	return nil
}

// TestSublauncherHealthRouteWins pins the registration order in buildRouter.
//
// mux serves the first matching route, so registering the built-in /health
// before the sublaunchers would shadow a deployment's own readiness check and
// report ok for an instance that was reporting itself unhealthy. A load
// balancer would then keep sending it traffic.
func TestSublauncherHealthRouteWins(t *testing.T) {
	l := NewLauncher(healthSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"ownhealth"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}

	router, err := l.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if got, want := rec.Code, http.StatusServiceUnavailable; got != want {
		t.Errorf("GET /health status = %d, want %d from the sublauncher's own handler", got, want)
	}
	if got := rec.Body.String(); got != "draining" {
		t.Errorf("GET /health body = %q, want %q; the built-in health route shadowed the sublauncher's", got, "draining")
	}
}

// TestBuiltInHealthRouteIsAFallback is the other half: when no sublauncher
// claims /health, the built-in one must still answer, because probes are
// configured with a fixed path.
func TestBuiltInHealthRouteIsAFallback(t *testing.T) {
	l := NewLauncher(telemetryFailSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"repro"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}

	router, err := l.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if got, want := rec.Code, http.StatusOK; got != want {
		t.Errorf("GET /health status = %d, want %d", got, want)
	}
	if got, want := strings.TrimSpace(rec.Body.String()), `{"status":"ok"}`; got != want {
		t.Errorf("GET /health body = %q, want %q", got, want)
	}
}

func TestApplyServiceDefaultsReturnsWhatItDefaulted(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config *launcher.Config
		want   []string
	}{
		// The session service is always defaulted and is not reported here,
		// because it is not the one whose loss on restart matters.
		{"nothing supplied", &launcher.Config{}, []string{"artifact", "memory"}},
		{"only session supplied", &launcher.Config{SessionService: session.InMemoryService()}, []string{"artifact", "memory"}},
		{"all supplied", &launcher.Config{
			SessionService:  session.InMemoryService(),
			ArtifactService: artifact.InMemoryService(),
			MemoryService:   memory.InMemoryService(),
		}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log.SetOutput(io.Discard)
			t.Cleanup(func() { log.SetOutput(os.Stderr) })

			got, unconsented := applyServiceDefaults(tc.config, true)
			if unconsented {
				t.Error("unconsented = true, want false when in-memory is allowed")
			}
			if len(got) != len(tc.want) {
				t.Fatalf("applyServiceDefaults() = %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("applyServiceDefaults()[%d] = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// TestInMemoryServiceWarningNamesTheDataLoss covers the startup banner warning.
//
// A deployment that forgot to configure storage must find that out at startup,
// not after a restart has dropped the data. Without consent the warning also
// has to say the launcher will refuse in a future release, since that notice is
// the only thing standing between an operator and a broken upgrade later.
func TestInMemoryServiceWarningNamesTheDataLoss(t *testing.T) {
	capture := func(defaulted []string, unconsented bool) string {
		var buf bytes.Buffer
		flags := log.Flags()
		log.SetOutput(&buf)
		log.SetFlags(0)
		defer func() {
			log.SetOutput(os.Stderr)
			log.SetFlags(flags)
		}()
		logInMemoryServiceWarning(defaulted, unconsented)
		return buf.String()
	}

	t.Run("unconsented names the future refusal", func(t *testing.T) {
		out := capture([]string{"artifact", "memory"}, true)
		for _, want := range []string{
			"WARNING", "artifact, memory", "lost when this process exits",
			"refuse to start", "-allow_in_memory_services",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("warning output %q does not contain %q", out, want)
			}
		}
	})

	t.Run("consented omits the future refusal", func(t *testing.T) {
		out := capture([]string{"artifact"}, false)
		if !strings.Contains(out, "lost when this process exits") {
			t.Errorf("warning output %q does not mention the data loss", out)
		}
		if strings.Contains(out, "refuse to start") {
			t.Errorf("warning output %q threatens a refusal the caller already opted out of", out)
		}
	})

	t.Run("nothing defaulted logs nothing", func(t *testing.T) {
		if out := capture(nil, false); out != "" {
			t.Errorf("logInMemoryServiceWarning(nil) logged %q, want nothing", out)
		}
	})
}

// TestWebUIImpliesInMemoryServices keeps `adk web api webui` working with no
// configuration, because the dev UI is the case the fallback exists for.
func TestWebUIImpliesInMemoryServices(t *testing.T) {
	for _, tc := range []struct {
		name     string
		keywords []string
		flags    []string
		want     bool
	}{
		{"webui active", []string{"webui"}, nil, true},
		{"webui absent", []string{"repro"}, nil, false},
		{"flag set without webui", []string{"repro"}, []string{"--allow_in_memory_services"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := NewLauncher(keywordSublauncher{"webui"}, telemetryFailSublauncher{}).(*webLauncher)
			if _, err := l.Parse(append(tc.flags, tc.keywords...)); err != nil {
				t.Fatalf("Parse() failed: %v", err)
			}
			if got := l.allowsInMemoryServices(); got != tc.want {
				t.Errorf("allowsInMemoryServices() = %v, want %v", got, tc.want)
			}
		})
	}
}

// keywordSublauncher is a no-op sublauncher registered under a chosen keyword.
type keywordSublauncher struct{ keyword string }

func (k keywordSublauncher) Keyword() string                                 { return k.keyword }
func (keywordSublauncher) Parse(args []string) ([]string, error)             { return args, nil }
func (keywordSublauncher) CommandLineSyntax() string                         { return "" }
func (keywordSublauncher) SimpleDescription() string                         { return "" }
func (keywordSublauncher) UserMessage(webURL string, printer func(v ...any)) {}
func (keywordSublauncher) SetupSubrouters(r *mux.Router, c *launcher.Config) error {
	return nil
}
