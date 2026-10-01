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
	"encoding/json"
	"log"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/exp/appinfo"
	"google.golang.org/adk/v2/tool"
)

const path = "/apps/{app_name}/app-info"

// serve routes one request to the handler through a mux router, so the
// app_name route variable is set.
func serve(t *testing.T, loader agent.Loader, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	router := mux.NewRouter()
	router.Handle(path, appinfo.Handler(loader))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func newLoader(t *testing.T, cfg llmagent.Config) agent.Loader {
	t.Helper()
	root, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q) failed: %v", cfg.Name, err)
	}
	return agent.NewSingleLoader(root)
}

func TestHandler(t *testing.T) {
	sub, err := llmagent.New(llmagent.Config{Name: "hotel", Description: "Books hotels.", Instruction: "Book."})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	loader := newLoader(t, llmagent.Config{
		Name:        "concierge",
		Description: "Plans trips.",
		Instruction: "Coordinate.",
		SubAgents:   []agent.Agent{sub},
	})

	rec := serve(t, loader, httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, http.StatusOK, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var got appinfo.AppInfo
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding the response failed: %v", err)
	}
	if got.Name != "concierge" || got.RootAgentName != "concierge" || got.Language != "go" {
		t.Errorf("got name=%q rootAgentName=%q language=%q, want concierge, concierge, go", got.Name, got.RootAgentName, got.Language)
	}
	if len(got.Agents) != 2 || got.Agents["concierge"] == nil || got.Agents["hotel"] == nil {
		t.Errorf("agents = %v, want concierge and hotel", got.Agents)
	}
}

func TestHandlerUnknownApp(t *testing.T) {
	loader := newLoader(t, llmagent.Config{Name: "concierge", Description: "Plans trips.", Instruction: "Coordinate."})
	rec := serve(t, loader, httptest.NewRequest(http.MethodGet, "/apps/missing/app-info", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}

// nilAgentLoader serves every app name without error but has no agent for any,
// as a custom loader might.
type nilAgentLoader struct{}

func (nilAgentLoader) ListAgents() []string                  { return []string{"ghost"} }
func (nilAgentLoader) LoadAgent(string) (agent.Agent, error) { return nil, nil }
func (nilAgentLoader) RootAgent() agent.Agent                { return nil }

func TestHandlerLoaderReturnsNoAgent(t *testing.T) {
	rec := serve(t, nilAgentLoader{}, httptest.NewRequest(http.MethodGet, "/apps/ghost/app-info", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusNotFound)
	}
}

func TestHandlerWithoutLoader(t *testing.T) {
	rec := serve(t, nil, httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}
}

// unencodableTool declares a schema JSON cannot represent.
type unencodableTool struct{}

func (unencodableTool) Name() string        { return "unencodable" }
func (unencodableTool) Description() string { return "Has a NaN in its schema." }
func (unencodableTool) IsLongRunning() bool { return false }
func (unencodableTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:                 "unencodable",
		ParametersJsonSchema: map[string]any{"default": math.NaN()},
	}
}

// TestHandlerEncodingFailure pins that a response JSON cannot encode is a 500,
// not a 200 with an empty body.
func TestHandlerEncodingFailure(t *testing.T) {
	loader := newLoader(t, llmagent.Config{
		Name:        "odd",
		Description: "Holds a tool with an unencodable schema.",
		Instruction: "Work.",
		Tools:       []tool.Tool{unencodableTool{}},
	})
	rec := serve(t, loader, httptest.NewRequest(http.MethodGet, "/apps/odd/app-info", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d; body: %q", rec.Code, http.StatusInternalServerError, rec.Body)
	}
}

// TestHandlerWarnsOnce pins adk-python's behavior for an experimental feature:
// one warning on first use, however many requests follow and however
// concurrently, and a fresh warning for a handler built afresh.
func TestHandlerWarnsOnce(t *testing.T) {
	// The standard logger serializes its writes, so a plain builder is safe.
	var logs strings.Builder
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(prevOut)
		log.SetFlags(prevFlags)
	})

	loader := newLoader(t, llmagent.Config{Name: "concierge", Description: "Plans trips.", Instruction: "Coordinate."})
	router := mux.NewRouter()
	router.Handle(path, appinfo.Handler(loader))

	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil))
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d, want %d", rec.Code, http.StatusOK)
			}
		})
	}
	wg.Wait()

	const want = "[EXPERIMENTAL] /apps/{app_name}/app-info: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time.\n"
	if got := logs.String(); got != want {
		t.Errorf("log = %q, want the warning exactly once: %q", got, want)
	}

	logs.Reset()
	serve(t, loader, httptest.NewRequest(http.MethodGet, "/apps/concierge/app-info", nil))
	if got := logs.String(); got != want {
		t.Errorf("a new handler logged %q, want its own warning: %q", got, want)
	}
}
