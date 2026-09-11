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

package adka2a

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	iagent "google.golang.org/adk/v2/internal/agent"
	iremoteagent "google.golang.org/adk/v2/internal/agent/remoteagent"
	"google.golang.org/adk/v2/session"
)

// clientProviderFunc adapts a function to [iremoteagent.A2AClientProvider].
type clientProviderFunc func(context.Context, *a2a.AgentCard) (iremoteagent.A2AClient, error)

func (f clientProviderFunc) CreateClient(ctx context.Context, card *a2a.AgentCard) (iremoteagent.A2AClient, error) {
	return f(ctx, card)
}

// cancelOnlyExecutor answers CancelTask and nothing else.
type cancelOnlyExecutor struct{}

func (cancelOnlyExecutor) Execute(context.Context, *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {}
}

func (cancelOnlyExecutor) Cancel(ctx context.Context, reqCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, nil), nil)
	}
}

func (cancelOnlyExecutor) Cleanup(context.Context, *a2asrv.ExecutorContext, a2a.SendMessageResult, error) {
}

// TestCancelChildInputRequiredTasksAuthenticatesCancel covers the second place
// a CancelTask is issued against a remote subagent. The subagent's client can
// carry the a2a auth interceptor that remoteagent.NewA2A installs for
// A2AConfig.Auth, and that interceptor resolves nothing unless the call carries
// the credential scope — so without it the cancel goes out unauthenticated, a
// secured remote rejects it, and the child task is left running.
func TestCancelChildInputRequiredTasksAuthenticatesCancel(t *testing.T) {
	const (
		appName   = "app"
		agentName = "remote"
		contextID = "ctx-1"
		taskID    = "task-1"
		callID    = "call-1"
		token     = "scoped-token"
	)
	// toInvocationMeta derives both from the A2A context id.
	userID, sessionID := "A2A_USER_"+contextID, contextID

	var mu sync.Mutex
	authByMethod := map[string]string{}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(cancelOnlyExecutor{}))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		authByMethod[peekJSONRPCMethod(r)] = r.Header.Get("Authorization")
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	card := &a2a.AgentCard{
		Name:                 agentName,
		SupportedInterfaces:  []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL, a2a.TransportProtocolJSONRPC)},
		SecuritySchemes:      a2a.NamedSecuritySchemes{"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"}},
		SecurityRequirements: a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}}},
	}

	// Holds the credential under the scope the run loop would use, so the
	// cancel is authenticated only if the executor attaches the same one.
	store := a2aclient.NewInMemoryCredentialsStore()
	store.Set(iremoteagent.CredentialScope(appName, userID, sessionID, agentName), "bearer", token)
	factory := a2aclient.NewFactory(a2aclient.WithCallInterceptors(&a2aclient.AuthInterceptor{Service: store}))

	remoteCfg := &iremoteagent.A2AServerConfig{
		AgentCard:     card,
		OwnsAuthScope: true,
		ClientProvider: clientProviderFunc(func(ctx context.Context, c *a2a.AgentCard) (iremoteagent.A2AClient, error) {
			// The provider is entitled to the same scope its client gets, so a
			// caller can key per-session transport setup off it.
			if _, ok := a2aclient.SessionIDFrom(ctx); !ok {
				return nil, errors.New("no credential scope on the context handed to ClientProvider")
			}
			return factory.CreateFromCard(ctx, c)
		}),
	}

	ctx := t.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	// The event the cancel scan matches on: authored by the remote subagent,
	// carrying the pending call and the remote task id.
	event := session.NewEvent(ctx, "invocation")
	event.Author = agentName
	event.Content = &genai.Content{
		Role:  string(genai.RoleModel),
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: callID, Name: "ask"}}},
	}
	event.CustomMetadata = map[string]any{customMetaTaskIDKey: taskID, customMetaContextIDKey: contextID}
	if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("sessionService.AppendEvent() error = %v", err)
	}

	statusParts, err := ToA2AParts(event.Content.Parts, nil)
	if err != nil {
		t.Fatalf("ToA2AParts() error = %v", err)
	}
	status := a2a.TaskStatus{
		State:   a2a.TaskStateInputRequired,
		Message: a2a.NewMessage(a2a.MessageRoleAgent, statusParts...),
	}

	e := &Executor{}
	cfg := RunnerConfig{AppName: appName, Agent: newRemoteStateAgent(t, agentName, remoteCfg), SessionService: svc}
	reqCtx := &a2asrv.ExecutorContext{ContextID: contextID}
	subagents := findRemoteSubagents(cfg.Agent)
	if len(subagents) != 1 {
		t.Fatalf("findRemoteSubagents() found %d remote subagents, want 1", len(subagents))
	}

	// The stub remote has no such task, so the cancel itself fails. What is
	// under test is the header the request carried, which the recorder captured
	// before the handler ever looked the task up.
	_ = e.cancelChildInputRequiredTasks(ctx, reqCtx, status, cfg, subagents)

	mu.Lock()
	defer mu.Unlock()
	got, ok := authByMethod["CancelTask"]
	if !ok {
		t.Fatalf("no CancelTask reached the remote subagent (saw %v)", authByMethod)
	}
	if want := "Bearer " + token; got != want {
		t.Errorf("CancelTask Authorization = %q, want %q", got, want)
	}
}

// TestCancelChildInputRequiredTasksKeepsCallerScope covers a subagent whose
// client carries the caller's own auth interceptor rather than this SDK's.
// The caller owns the scope there, so overwriting it would resolve under a key
// their store has never heard of and drop their credential.
func TestCancelChildInputRequiredTasksKeepsCallerScope(t *testing.T) {
	const (
		appName   = "app"
		agentName = "remote"
		contextID = "ctx-1"
		taskID    = "task-1"
		callID    = "call-1"
		token     = "caller-token"
	)
	userID, sessionID := "A2A_USER_"+contextID, contextID

	var mu sync.Mutex
	authByMethod := map[string]string{}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(cancelOnlyExecutor{}))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		authByMethod[peekJSONRPCMethod(r)] = r.Header.Get("Authorization")
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	card := &a2a.AgentCard{
		Name:                 agentName,
		SupportedInterfaces:  []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL, a2a.TransportProtocolJSONRPC)},
		SecuritySchemes:      a2a.NamedSecuritySchemes{"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"}},
		SecurityRequirements: a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}}},
	}
	store := a2aclient.NewInMemoryCredentialsStore()
	store.Set("caller-tenant", "bearer", token)
	factory := a2aclient.NewFactory(a2aclient.WithCallInterceptors(&a2aclient.AuthInterceptor{Service: store}))

	// OwnsAuthScope stays false: this is the caller's own interceptor.
	remoteCfg := &iremoteagent.A2AServerConfig{
		AgentCard: card,
		ClientProvider: clientProviderFunc(func(ctx context.Context, c *a2a.AgentCard) (iremoteagent.A2AClient, error) {
			return factory.CreateFromCard(ctx, c)
		}),
	}

	ctx := a2aclient.AttachSessionID(t.Context(), "caller-tenant")
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatalf("sessionService.Create() error = %v", err)
	}
	event := session.NewEvent(ctx, "invocation")
	event.Author = agentName
	event.Content = &genai.Content{
		Role:  string(genai.RoleModel),
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: callID, Name: "ask"}}},
	}
	event.CustomMetadata = map[string]any{customMetaTaskIDKey: taskID, customMetaContextIDKey: contextID}
	if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatalf("sessionService.AppendEvent() error = %v", err)
	}
	statusParts, err := ToA2AParts(event.Content.Parts, nil)
	if err != nil {
		t.Fatalf("ToA2AParts() error = %v", err)
	}
	status := a2a.TaskStatus{State: a2a.TaskStateInputRequired, Message: a2a.NewMessage(a2a.MessageRoleAgent, statusParts...)}

	cfg := RunnerConfig{AppName: appName, Agent: newRemoteStateAgent(t, agentName, remoteCfg), SessionService: svc}
	subagents := findRemoteSubagents(cfg.Agent)
	_ = (&Executor{}).cancelChildInputRequiredTasks(ctx, &a2asrv.ExecutorContext{ContextID: contextID}, status, cfg, subagents)

	mu.Lock()
	defer mu.Unlock()
	got, ok := authByMethod["CancelTask"]
	if !ok {
		t.Fatalf("no CancelTask reached the remote subagent (saw %v)", authByMethod)
	}
	if want := "Bearer " + token; got != want {
		t.Errorf("CancelTask Authorization = %q, want %q; the caller's own scope must survive", got, want)
	}
}

// newRemoteStateAgent builds an agent carrying the remote-agent state that
// findRemoteSubagents looks for, without importing agent/remoteagent/v2 — that
// package imports this one.
func newRemoteStateAgent(t *testing.T, name string, remoteCfg *iremoteagent.A2AServerConfig) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name: name,
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	ia, ok := a.(iagent.Agent)
	if !ok {
		t.Fatalf("agent.New() returned %T, want an internal agent", a)
	}
	state := iagent.Reveal(ia)
	state.AgentType = iagent.TypeRemoteAgent
	state.Config = iremoteagent.RemoteAgentState{A2A: remoteCfg}
	return a
}

// peekJSONRPCMethod reads the JSON-RPC method and restores the body.
func peekJSONRPCMethod(r *http.Request) string {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return ""
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var rpc struct {
		Method string `json:"method"`
	}
	_ = json.Unmarshal(body, &rpc)
	return rpc.Method
}
