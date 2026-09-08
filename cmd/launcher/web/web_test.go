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

// serveWithoutPanic serves one request and fails the test if the handler
// panics, rather than letting the panic take the package down.
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

// TestApplyServiceDefaultsFillsOnlySession pins the contract: the launcher
// supplies a session service and nothing else.
//
// Artifact and memory are the caller's to configure. Filling them in is what
// let a misconfigured deployment come up looking healthy and lose everything it
// had stored on the next restart.
func TestApplyServiceDefaultsFillsOnlySession(t *testing.T) {
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	config := &launcher.Config{}
	applyServiceDefaults(config)

	if config.SessionService == nil {
		t.Error("SessionService is nil, want an in-memory one")
	}
	if config.ArtifactService != nil {
		t.Error("ArtifactService was filled in, want it left to the caller")
	}
	if config.MemoryService != nil {
		t.Error("MemoryService was filled in, want it left to the caller")
	}
}

func TestApplyServiceDefaultsKeepsSuppliedServices(t *testing.T) {
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	supplied := session.InMemoryService()
	config := &launcher.Config{
		SessionService:  supplied,
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
	}
	applyServiceDefaults(config)

	if config.SessionService != supplied {
		t.Error("SessionService was replaced, want the supplied one kept")
	}
	if config.ArtifactService == nil || config.MemoryService == nil {
		t.Error("a supplied service was cleared")
	}
}

// TestUnconfiguredArtifactServiceAnswers503 is why leaving it nil is safe.
//
// Before the guard in the artifact handlers, a nil service was dereferenced,
// which panicked and dropped the connection with no HTTP response at all. That
// crash is what defaulting the service was working around. With the guard, the
// unconfigured case reports itself.
func TestUnconfiguredArtifactServiceAnswers503(t *testing.T) {
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	config := &launcher.Config{}
	applyServiceDefaults(config)

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  config.SessionService,
		ArtifactService: config.ArtifactService,
		MemoryService:   config.MemoryService,
		AgentLoader:     config.AgentLoader,
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}

	rec := serveWithoutPanic(t, server,
		httptest.NewRequest(http.MethodGet, "/apps/a/users/u/sessions/s/artifacts", nil))

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("GET artifacts status = %d (%s), want %d",
			rec.Code, rec.Body.String(), http.StatusServiceUnavailable)
	}
}

// TestUnconfiguredServicesStillServeSessions checks the rest of the API is
// unaffected by leaving artifact and memory unset.
func TestUnconfiguredServicesStillServeSessions(t *testing.T) {
	log.SetOutput(io.Discard)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	config := &launcher.Config{}
	applyServiceDefaults(config)

	server, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  config.SessionService,
		ArtifactService: config.ArtifactService,
		MemoryService:   config.MemoryService,
		AgentLoader:     config.AgentLoader,
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() failed: %v", err)
	}

	rec := serveWithoutPanic(t, server,
		httptest.NewRequest(http.MethodGet, "/apps/a/users/u/sessions", nil))

	if rec.Code != http.StatusOK {
		t.Errorf("GET sessions status = %d (%s), want %d",
			rec.Code, rec.Body.String(), http.StatusOK)
	}
}

func TestHealthFallbackAnswers(t *testing.T) {
	router := withHealthFallback(BuildBaseRouter())

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

// TestBuildBaseRouterLeavesHealthToTheCaller guards the reason the fallback
// lives in withHealthFallback: mux serves the first matching route, so
// registering /health inside the exported constructor would silently shadow an
// embedder's own handler for that path.
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

// catchAllSublauncher mounts a route that matches every path, the way an API
// sublauncher does when its path prefix is empty.
type catchAllSublauncher struct{}

func (catchAllSublauncher) Keyword() string                                   { return "catchall" }
func (catchAllSublauncher) Parse(args []string) ([]string, error)             { return args, nil }
func (catchAllSublauncher) CommandLineSyntax() string                         { return "" }
func (catchAllSublauncher) SimpleDescription() string                         { return "" }
func (catchAllSublauncher) UserMessage(webURL string, printer func(v ...any)) {}
func (catchAllSublauncher) SetupSubrouters(r *mux.Router, c *launcher.Config) error {
	r.NewRoute().HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte("catch-all"))
	})
	return nil
}

// TestHealthFallbackBeatsACatchAll covers a sublauncher that matches every
// path without meaning to own the probe path.
//
// Registering the health route after the sublaunchers puts it behind such a
// route, so the probe path 404s. Deciding outside the router avoids that,
// because a catch-all declares no path template and so does not claim /health.
func TestHealthFallbackBeatsACatchAll(t *testing.T) {
	l := NewLauncher(catchAllSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"catchall"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}
	handler, err := l.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}

	for _, method := range []string{http.MethodGet, http.MethodHead} {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(method, "/health", nil))
		if rec.Code != http.StatusOK {
			t.Errorf("%s /health status = %d, want %d; a catch-all is hiding the probe path",
				method, rec.Code, http.StatusOK)
		}
	}

	// Everything else still reaches the catch-all.
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/anything", nil))
	if body := rec.Body.String(); body != "catch-all" {
		t.Errorf("GET /anything body = %q, want the catch-all to still serve it", body)
	}
}

// TestSublauncherHealthOwnsEveryMethod covers the verb half of the shadowing
// bug.
//
// A sublauncher that declares /health for GET alone owns the path outright. If
// the fallback answered HEAD, a draining instance would report ok on the verb
// HAProxy and nginx probe with by default, which is the failure this is meant
// to prevent, surviving on one method.
func TestSublauncherHealthOwnsEveryMethod(t *testing.T) {
	l := NewLauncher(healthSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"ownhealth"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}
	handler, err := l.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodHead, "/health", nil))

	if rec.Code == http.StatusOK {
		t.Errorf("HEAD /health status = 200, want the sublauncher's own answer; "+
			"the fallback is reporting ok for an instance that declared itself unhealthy (body %q)",
			rec.Body.String())
	}
}

// TestHealthFallbackKeepsRouterBehaviour covers what answering in front of the
// router used to lose.
//
// The fallback is a route on an outer router, not a check before the inner one,
// so it still runs the logger middleware and still gets StrictSlash handling. A
// probe that stopped appearing in the request log, or a probe configured with a
// trailing slash, are both silent failures.
func TestHealthFallbackKeepsRouterBehaviour(t *testing.T) {
	var buf bytes.Buffer
	flags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(os.Stderr)
		log.SetFlags(flags)
	})

	inner := BuildBaseRouter()
	// A real route, because mux middleware runs only on a matched route. An
	// empty router matches nothing, so the logger would never run either way.
	inner.HandleFunc("/other", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := withHealthFallback(inner)

	t.Run("logger runs", func(t *testing.T) {
		buf.Reset()
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want %d", healthPath, rec.Code, http.StatusOK)
		}
		if !strings.Contains(buf.String(), healthPath) {
			t.Errorf("request log %q does not mention %s; the probe is invisible", buf.String(), healthPath)
		}
	})

	t.Run("trailing slash redirects", func(t *testing.T) {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath+"/", nil))
		if rec.Code != http.StatusMovedPermanently {
			t.Errorf("GET %s/ status = %d, want %d", healthPath, rec.Code, http.StatusMovedPermanently)
		}
		if got := rec.Header().Get("Location"); got != healthPath {
			t.Errorf("Location = %q, want %q", got, healthPath)
		}
	})

	t.Run("logger is not doubled", func(t *testing.T) {
		buf.Reset()
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/other", nil))
		if n := strings.Count(buf.String(), "/other"); n != 1 {
			t.Errorf("request logged %d times, want 1; log was %q", n, buf.String())
		}
	})
}

// scopedHealthSublauncher registers healthPath behind a host matcher, so it
// answers some requests for the path and rejects others.
type scopedHealthSublauncher struct{}

func (scopedHealthSublauncher) Keyword() string { return "scopedhealth" }

func (scopedHealthSublauncher) Parse(args []string) ([]string, error)             { return args, nil }
func (scopedHealthSublauncher) CommandLineSyntax() string                         { return "" }
func (scopedHealthSublauncher) SimpleDescription() string                         { return "" }
func (scopedHealthSublauncher) UserMessage(webURL string, printer func(v ...any)) {}
func (scopedHealthSublauncher) SetupSubrouters(r *mux.Router, c *launcher.Config) error {
	r.HandleFunc(healthPath, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}).Host("internal.example.com")
	return nil
}

// TestScopedHealthRouteDoesNotDisableTheFallback covers a route that claims the
// path only for some requests.
//
// Treating it as owning the path switches the fallback off for every request,
// including the ones its own matcher rejects, so an ordinary probe 404s.
func TestScopedHealthRouteDoesNotDisableTheFallback(t *testing.T) {
	l := NewLauncher(scopedHealthSublauncher{}).(*webLauncher)
	if _, err := l.Parse([]string{"scopedhealth"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}
	handler, err := l.buildRouter(&launcher.Config{})
	if err != nil {
		t.Fatalf("buildRouter() failed: %v", err)
	}

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, healthPath, nil))

	if rec.Code != http.StatusOK {
		t.Errorf("GET %s status = %d, want %d; a host-scoped route switched off the fallback "+
			"for requests it does not match", healthPath, rec.Code, http.StatusOK)
	}
}
