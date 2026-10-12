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

// scopedTransport stands in for the transport remoteagent.NewA2A installs for
// A2AConfig.Auth, which this package cannot build because of the import cycle.
// It reads the credential scope from the request's context, as the real one
// does to key its credential, but it is stricter: it applies the credential
// only if the executor attached the scope and the A2A client carried it down to
// the HTTP request, which is what makes it a test of that propagation.
type scopedTransport struct {
	scope a2aclient.SessionID
	token string
}

func (s scopedTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	out := r.Clone(r.Context())
	if sid, ok := a2aclient.SessionIDFrom(r.Context()); ok && sid == s.scope {
		out.Header.Set("Authorization", "Bearer "+s.token)
	}
	return http.DefaultTransport.RoundTrip(out)
}

// TestCancelChildInputRequiredTasksAuthenticatesCancel covers the second place
// a CancelTask is issued against a remote subagent. The subagent's client can
// carry the auth transport remoteagent.NewA2A installs for A2AConfig.Auth, which
// keys the credential on the scope the call carries — so without it the
// provider sees no identity, a secured remote rejects the cancel, and the child
// task is left running.
func TestCancelChildInputRequiredTasksAuthenticatesCancel(t *testing.T) {
	const (
		appName   = "app"
		agentName = "remote"
		contextID = "ctx-1"
		taskID    = "task-1"
		taskID2   = "task-2"
		callID    = "call-1"
		callID2   = "call-2"
		token     = "scoped-token"
	)
	// toInvocationMeta derives both from the A2A context id.
	userID, sessionID := "A2A_USER_"+contextID, contextID

	var mu sync.Mutex
	var cancelAuth []string
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(cancelOnlyExecutor{}))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if peekJSONRPCMethod(r) == "CancelTask" {
			mu.Lock()
			cancelAuth = append(cancelAuth, r.Header.Get("Authorization"))
			mu.Unlock()
		}
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()

	card := &a2a.AgentCard{
		Name:                 agentName,
		SupportedInterfaces:  []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL, a2a.TransportProtocolJSONRPC)},
		SecuritySchemes:      a2a.NamedSecuritySchemes{"bearer": a2a.HTTPAuthSecurityScheme{Scheme: "Bearer"}},
		SecurityRequirements: a2a.SecurityRequirementsOptions{{a2a.SecuritySchemeName("bearer"): a2a.SecuritySchemeScopes{}}},
	}

	client := &http.Client{Transport: scopedTransport{
		scope: iremoteagent.CredentialScope(appName, userID, sessionID, agentName),
		token: token,
	}}
	factory := a2aclient.NewFactory(a2aclient.WithJSONRPCTransport(client))

	// Stands in for the provider NewA2A builds for itself when Auth is set,
	// which is the only way OwnsAuthScope is ever true in production.
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
	// Two events the cancel scan matches on, each authored by the remote
	// subagent and carrying its own pending call and remote task id. Two,
	// because one leaves the executor's client cache always missing, and the
	// second cancel then goes through the cached client.
	var statusParts []*a2a.Part
	for _, seed := range []struct{ callID, taskID string }{{callID, taskID}, {callID2, taskID2}} {
		event := session.NewEvent(ctx, "invocation")
		event.Author = agentName
		event.Content = &genai.Content{
			Role:  string(genai.RoleModel),
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: seed.callID, Name: "ask"}}},
		}
		event.CustomMetadata = map[string]any{customMetaTaskIDKey: seed.taskID, customMetaContextIDKey: contextID}
		if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
			t.Fatalf("sessionService.AppendEvent() error = %v", err)
		}
		parts, err := ToA2AParts(event.Content.Parts, nil)
		if err != nil {
			t.Fatalf("ToA2AParts() error = %v", err)
		}
		statusParts = append(statusParts, parts...)
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
	if len(cancelAuth) != 2 {
		t.Fatalf("%d CancelTask requests reached the remote subagent, want 2 (saw %q)", len(cancelAuth), cancelAuth)
	}
	for i, got := range cancelAuth {
		if want := "Bearer " + token; got != want {
			t.Errorf("CancelTask #%d Authorization = %q, want %q", i+1, got, want)
		}
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

// TestCancelChildInputRequiredTasksCustomProviderContext pins what the
// ClientProvider doc and NewA2A's error now tell a caller about the second
// place their provider is called. The adka2a cancel passes the server's own
// request context, which is not an agent.InvocationContext and carries no
// scope, so a provider must check its type assertion. One that does, and falls
// back, still gets its CancelTask out.
func TestCancelChildInputRequiredTasksCustomProviderContext(t *testing.T) {
	const (
		appName   = "app"
		agentName = "remote"
		contextID = "ctx-1"
		taskID    = "task-1"
		callID    = "call-1"
	)
	userID, sessionID := "A2A_USER_"+contextID, contextID

	var mu sync.Mutex
	methods := map[string]int{}
	inner := a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(cancelOnlyExecutor{}))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		methods[peekJSONRPCMethod(r)]++
		mu.Unlock()
		inner.ServeHTTP(w, r)
	}))
	defer srv.Close()
	card := &a2a.AgentCard{
		Name:                agentName,
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(srv.URL, a2a.TransportProtocolJSONRPC)},
	}
	factory := a2aclient.NewFactory()

	var sawInvocation, sawScope bool
	remoteCfg := &iremoteagent.A2AServerConfig{
		AgentCard: card,
		// A custom provider: OwnsAuthScope stays false, as NewA2A leaves it.
		ClientProvider: clientProviderFunc(func(ctx context.Context, c *a2a.AgentCard) (iremoteagent.A2AClient, error) {
			_, sawInvocation = ctx.(agent.InvocationContext)
			_, sawScope = a2aclient.SessionIDFrom(ctx)
			return factory.CreateFromCard(ctx, c)
		}),
	}

	ctx := t.Context()
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
	_ = (&Executor{}).cancelChildInputRequiredTasks(ctx, &a2asrv.ExecutorContext{ContextID: contextID}, status, cfg, findRemoteSubagents(cfg.Agent))

	mu.Lock()
	defer mu.Unlock()
	if sawInvocation {
		t.Error("the provider got an agent.InvocationContext; the docs that tell callers to check the assertion here are out of date")
	}
	if sawScope {
		t.Error("the provider got a credential scope without Auth ownership; a caller's own interceptor could be fed a key it never chose")
	}
	if methods["CancelTask"] != 1 {
		t.Errorf("CancelTask reached the remote %d times, want 1", methods["CancelTask"])
	}
}
