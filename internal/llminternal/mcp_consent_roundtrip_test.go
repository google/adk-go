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

package llminternal_test

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/authconsent"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// consentingProvider needs consent until the test grants it, then mints a
// bearer token. It is the shape of a real interactive provider seen across the
// two turns of a round-trip.
type consentingProvider struct {
	mu        sync.Mutex
	consented bool
}

func (p *consentingProvider) Credential(context.Context) (auth.Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.consented {
		return nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example/auth", Nonce: "n0nce", Key: "mcp-key"}
	}
	return auth.BearerCredential{Token: "granted"}, nil
}

// mcpCallingModel calls cred_tool on its first turn and answers afterwards.
type mcpCallingModel struct {
	model.LLM
	calls int
}

func (m *mcpCallingModel) Name() string { return "mcp-calling-model" }

func (m *mcpCallingModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{genai.NewPartFromText("done")}}}, nil)
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "mcp-call-1", Name: "cred_tool", Args: map[string]any{},
		}}}}}, nil)
	}
}

// TestMCPConsentRoundTripThroughRunner drives the MCP consent path end to end
// over the real HTTP transport: listing succeeds while consent is pending, the
// tool's preflight pauses the run with adk_request_credential, and after the
// client's reply the call reaches the server with the granted credential.
func TestMCPConsentRoundTripThroughRunner(t *testing.T) {
	var (
		mu          sync.Mutex
		authHeaders []string
	)
	server := mcp.NewServer(&mcp.Implementation{Name: "consent", Version: "v1"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "cred_tool"}, func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "ran"}}}, struct{}{}, nil
	})
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		authHeaders = append(authHeaders, r.Header.Get("Authorization"))
		mu.Unlock()
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)

	provider := &consentingProvider{}
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: srv.URL, Auth: provider})
	if err != nil {
		t.Fatalf("mcptoolset.New() error = %v", err)
	}
	t.Cleanup(func() {
		if c, ok := ts.(interface{ Close() error }); ok {
			_ = c.Close()
		}
	})
	m := &mcpCallingModel{}
	a, err := llmagent.New(llmagent.Config{Name: "mcp_consent_agent", Model: m, Toolsets: []tool.Toolset{ts}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	ss := session.InMemoryService()
	if _, err := ss.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{Agent: a, SessionService: ss, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}
	run := func(content *genai.Content) []*session.Event {
		var evs []*session.Event
		for ev, err := range r.Run(t.Context(), "u", "s", content, agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("run error = %v", err)
			}
			evs = append(evs, ev)
		}
		return evs
	}

	// Turn 1: listing goes through unauthenticated, the call pauses for consent.
	turn1 := run(&genai.Content{Role: "user", Parts: []*genai.Part{genai.NewPartFromText("go")}})
	var consentCall *genai.FunctionCall
	for _, ev := range turn1 {
		for _, fc := range utils.FunctionCalls(ev.Content) {
			if fc.Name == authconsent.FunctionCallName {
				consentCall = fc
			}
		}
	}
	if consentCall == nil {
		t.Fatalf("turn 1 emitted no %s call; events: %d, model calls: %d", authconsent.FunctionCallName, len(turn1), m.calls)
	}
	if got := consentCall.Args[authconsent.FunctionCallIDArg]; got != "mcp-call-1" {
		t.Errorf("consent call binds %v, want mcp-call-1", got)
	}
	mu.Lock()
	for _, h := range authHeaders {
		if h != "" {
			t.Errorf("a request before consent carried Authorization %q", h)
		}
	}
	authHeaders = nil
	mu.Unlock()

	// Turn 2: the user consented; the client replies and the call reaches the server.
	provider.mu.Lock()
	provider.consented = true
	provider.mu.Unlock()
	reply := authconsent.AuthConfig{ExchangedAuthCredential: &authconsent.AuthCredential{
		AuthType: authconsent.AuthTypeOAuth2,
		OAuth2:   &authconsent.OAuth2Auth{AuthURI: "https://consent.example/auth", AuthResponseURI: "http://localhost/cb?code=abc"},
	}}
	raw, _ := json.Marshal(reply)
	var obj map[string]any
	_ = json.Unmarshal(raw, &obj)
	turn2 := run(&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: consentCall.ID, Name: authconsent.FunctionCallName, Response: obj,
	}}}})

	var ran bool
	for _, ev := range turn2 {
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr.Name == "cred_tool" {
				if _, isErr := fr.Response["error"]; !isErr {
					ran = true
				}
			}
		}
	}
	if !ran {
		t.Fatalf("turn 2 did not produce a successful cred_tool response; events: %d", len(turn2))
	}
	mu.Lock()
	defer mu.Unlock()
	var bearer bool
	for _, h := range authHeaders {
		if h == "Bearer granted" {
			bearer = true
		}
	}
	if !bearer {
		t.Errorf("after consent no request carried the granted credential; headers = %q", authHeaders)
	}
}
