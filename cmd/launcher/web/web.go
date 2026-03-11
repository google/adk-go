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

// Package web provides a way to run ADK using a web server.
package web

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/internal/telemetry"
	"google.golang.org/adk/session"
)

// Config contains parameters for launching web server
type Config struct {
	Port            int
	WriteTimeout    time.Duration
	ReadTimeout     time.Duration
	IdleTimeout     time.Duration
	ShutdownTimeout time.Duration
	OTelToCloud     bool

	EnableA2A   bool
	EnableAPI   bool
	EnableWebUI bool

	A2A   A2AConfig
	API   APIConfig
	WebUI WebUIConfig
}

func DefineFlags(cfg *Config) {
	DefineA2AFlags(&cfg.A2A)
	DefineAPIFlags(&cfg.API)
	DefineWebUIFlags(&cfg.WebUI)

	flag.BoolVar(&cfg.EnableA2A, "adk_weba2a_enable", true, "Enable the ADK A2A server")
	flag.BoolVar(&cfg.EnableAPI, "adk_webapi_enable", true, "Enable the ADK REST API server")
	flag.BoolVar(&cfg.EnableWebUI, "adk_webui_enable", true, "Enable the ADK development web UI.  Not for production use")

	flag.IntVar(&cfg.Port, "adk_web_port", 8080, "Localhost port for the server")
	flag.DurationVar(&cfg.WriteTimeout, "adk_web_write_timeout", 15*time.Second, "Server write timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for writing the response after reading the headers & body")
	flag.DurationVar(&cfg.ReadTimeout, "adk_web_read_timeout", 15*time.Second, "Server read timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for reading the whole request including body")
	flag.DurationVar(&cfg.IdleTimeout, "adk_web_idle_timeout", 60*time.Second, "Server idle timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for the next request (only when keep-alive is enabled)")
	flag.DurationVar(&cfg.ShutdownTimeout, "adk_web_shutdown_timeout", 15*time.Second, "Server shutdown timeout (i.e. '10s', '2m' - see time.ParseDuration for details) - for waiting for active requests to finish during shutdown")
	flag.BoolVar(&cfg.OTelToCloud, "adk_web_otel_to_cloud", false, "Enables/disables OpenTelemetry export to GCP: telemetry.googleapis.com. See adk-go/telemetry package for details about supported options, credentials and environment variables.")
}

// Launcher can launch web server
type Launcher struct {
	config *Config

	apiSubLauncher   *APISubLauncher
	a2aSubLauncher   *A2ASubLauncher
	webUISubLauncher *WebUISubLauncher
}

// NewLauncher creates a new web launcher.
func NewLauncher(cfg *Config) *Launcher {
	l := &Launcher{
		config: cfg,
	}

	if cfg.EnableAPI {
		l.apiSubLauncher = NewAPISubLauncher(&cfg.API)
	}
	if cfg.EnableA2A {
		l.a2aSubLauncher = NewA2ASubLauncher(&cfg.A2A)
	}
	if cfg.EnableWebUI {
		l.webUISubLauncher = NewWebUISubLauncher(&cfg.WebUI)
	}

	return l
}

// Run implements launcher.SubLauncher.
func (w *Launcher) Run(ctx context.Context, config *launcher.Config) error {
	if config.SessionService == nil {
		config.SessionService = session.InMemoryService()
	}

	router := BuildBaseRouter()

	if w.config.EnableAPI {
		if err := w.apiSubLauncher.SetupSubrouters(router, config); err != nil {
			return fmt.Errorf("while setting up API subrouter: %w", err)
		}
	}
	if w.config.EnableA2A {
		if err := w.a2aSubLauncher.SetupSubrouters(router, config); err != nil {
			return fmt.Errorf("while setting up A2A subrouter: %w", err)
		}
	}
	if w.config.EnableWebUI {
		if err := w.webUISubLauncher.SetupSubrouters(router, config); err != nil {
			return fmt.Errorf("while setting up WebUI subrouter: %w", err)
		}
	}

	log.Printf("Starting the web server: %+v", w.config)
	log.Println()
	webURL := fmt.Sprintf("http://localhost:%v", fmt.Sprint(w.config.Port))
	log.Printf("Web servers starts on %s", webURL)
	if w.config.EnableAPI {
		log.Printf("       api:  you can access API using %s%s", webURL, w.config.API.PathPrefix)
		log.Printf("       api:      for instance: %s%s/list-apps", webURL, w.config.API.PathPrefix)
	}
	if w.config.EnableA2A {
		log.Printf("       a2a:  you can access A2A using jsonrpc protocol: %s", webURL)
	}
	if w.config.EnableWebUI {
		log.Printf("       webui:  you can access API using %s%s", webURL, WebUIPathPrefix)
	}
	log.Println()

	srv := http.Server{
		Addr:         fmt.Sprintf(":%v", fmt.Sprint(w.config.Port)),
		WriteTimeout: w.config.WriteTimeout,
		ReadTimeout:  w.config.ReadTimeout,
		IdleTimeout:  w.config.IdleTimeout,
		Handler:      router,
	}

	errChan := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- err
		}
		close(errChan)
	}()

	telemetryService, err := telemetry.InitAndSetGlobalOtelProviders(ctx, config, w.config.OTelToCloud)
	if err != nil {
		return fmt.Errorf("telemetry initialization failed: %v", err)
	}

	select {
	case <-ctx.Done():
		log.Println("Shutting down the web server...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), w.config.ShutdownTimeout)
		defer cancel()
		serverErr := srv.Shutdown(shutdownCtx)
		telemetryErr := telemetryService.Shutdown(shutdownCtx)
		return errors.Join(serverErr, telemetryErr)
	case err, ok := <-errChan:
		if !ok {
			return nil
		}
		return fmt.Errorf("server failed: %v", err)
	}
}

// logger is a middleware that logs the HTTP method, request URI, and the time taken to process the request.
func logger(inner http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		inner.ServeHTTP(w, r)

		log.Printf(
			"%s %s %s",
			r.Method,
			r.RequestURI,
			time.Since(start),
		)
	})
}

// BuildBaseRouter returns the main router, which can be extended by sub-routers.
func BuildBaseRouter() *mux.Router {
	router := mux.NewRouter().StrictSlash(true)
	router.Use(logger)
	return router
}
