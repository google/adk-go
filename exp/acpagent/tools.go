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

package acpagent

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// mcpServerName is the name under which the harness sees the tools.
const mcpServerName = "adk"

// errNoInvocation is the result of a tool call the harness makes while no ADK
// invocation of its session runs, for example while a question is pending.
var errNoInvocation = errors.New("no ADK invocation of this session is in progress")

// toolServer serves [Config.Tools] to the harness over MCP, on a loopback
// HTTP endpoint per A2A context. The endpoint is passed to the harness when it
// creates the ACP session, and its path holds a random token, so a tool runs
// only on behalf of the session it was given to.
//
// A tool runs in the ADK invocation which is in progress in its session, and
// its call and response become events of that invocation, so state and
// artifact changes are saved like those of any other tool.
type toolServer struct {
	agentName string
	tools     []toolinternal.FunctionTool
	baseURL   string
	srv       *http.Server

	mu        sync.Mutex
	endpoints map[string]*endpoint // token -> endpoint
	byContext map[string]*endpoint // A2A context ID -> endpoint
}

// endpoint is the MCP endpoint of one A2A context.
type endpoint struct {
	token     string
	contextID string
	server    *mcp.Server
	run       *invocationRun // guarded by toolServer.mu; nil between invocations
}

func newToolServer(agentName string, tools []tool.Tool) (*toolServer, error) {
	s := &toolServer{
		agentName: agentName,
		endpoints: map[string]*endpoint{},
		byContext: map[string]*endpoint{},
	}
	names := map[string]bool{}
	for _, t := range tools {
		ft, ok := t.(toolinternal.FunctionTool)
		if !ok {
			return nil, fmt.Errorf("tool %q: only function tools can be served to the harness, got %T", t.Name(), t)
		}
		if names[t.Name()] {
			return nil, fmt.Errorf("duplicate tool name %q", t.Name())
		}
		names[t.Name()] = true
		s.tools = append(s.tools, ft)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("listening for the tool server: %w", err)
	}
	s.baseURL = "http://" + ln.Addr().String() + "/mcp/"
	mux := http.NewServeMux()
	mux.Handle("/mcp/{token}", mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		s.mu.Lock()
		defer s.mu.Unlock()
		if ep := s.endpoints[r.PathValue("token")]; ep != nil {
			return ep.server
		}
		return nil
	}, nil))
	s.srv = &http.Server{Handler: mux}
	go s.srv.Serve(ln) //nolint:errcheck // Serve returns ErrServerClosed after Close.
	return s, nil
}

// Close stops the server and drops the connections of the harness.
func (s *toolServer) Close() error {
	return s.srv.Close()
}

// mcpServer returns the ACP McpServer the harness connects to for the A2A
// context, creating its endpoint on first use.
func (s *toolServer) mcpServer(contextID string) (json.RawMessage, error) {
	ep, err := s.endpoint(contextID)
	if err != nil {
		return nil, err
	}
	return json.Marshal(map[string]any{
		"type":    "http",
		"name":    mcpServerName,
		"url":     s.baseURL + ep.token,
		"headers": []any{},
	})
}

func (s *toolServer) endpoint(contextID string) (*endpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ep := s.byContext[contextID]; ep != nil {
		return ep, nil
	}
	ep := &endpoint{token: rand.Text(), contextID: contextID}
	ep.server = mcp.NewServer(&mcp.Implementation{Name: s.agentName}, nil)
	for _, t := range s.tools {
		schema, err := inputSchema(t.Declaration())
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", t.Name(), err)
		}
		ep.server.AddTool(&mcp.Tool{
			Name:        t.Name(),
			Description: t.Description(),
			InputSchema: schema,
		}, s.handler(ep, t))
	}
	s.endpoints[ep.token] = ep
	s.byContext[contextID] = ep
	return ep, nil
}

// bind makes run the invocation in which tools of the A2A context run.
func (s *toolServer) bind(contextID string, run *invocationRun) error {
	ep, err := s.endpoint(contextID)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ep.run = run
	return nil
}

// unbind detaches run from the contexts it was bound to.
func (s *toolServer) unbind(run *invocationRun) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, ep := range s.byContext {
		if ep.run == run {
			ep.run = nil
		}
	}
}

func (s *toolServer) handler(ep *endpoint, t toolinternal.FunctionTool) mcp.ToolHandler {
	return func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		s.mu.Lock()
		run := ep.run
		s.mu.Unlock()
		if run == nil {
			return errorResult(errNoInvocation), nil
		}
		args := map[string]any{}
		if len(req.Params.Arguments) > 0 {
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
				return errorResult(fmt.Errorf("tool %q arguments are not a JSON object: %w", t.Name(), err)), nil
			}
		}
		result, err := run.callTool(ctx, s.agentName, ep.contextID, t, args)
		if err != nil {
			return errorResult(err), nil
		}
		text, err := json.Marshal(result)
		if err != nil {
			return nil, fmt.Errorf("encoding the result of tool %q: %w", t.Name(), err)
		}
		return &mcp.CallToolResult{
			Content:           []mcp.Content{&mcp.TextContent{Text: string(text)}},
			StructuredContent: result,
		}, nil
	}
}

func errorResult(err error) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: err.Error()}},
		IsError: true,
	}
}

// inputSchema returns the JSON schema of the declaration's parameters.
func inputSchema(decl *genai.FunctionDeclaration) (any, error) {
	if decl == nil {
		return map[string]any{"type": "object"}, nil
	}
	if decl.ParametersJsonSchema != nil {
		return decl.ParametersJsonSchema, nil
	}
	if decl.Parameters == nil {
		return map[string]any{"type": "object"}, nil
	}
	// A genai.Schema is close to a JSON schema, except that its types are
	// upper case.
	b, err := json.Marshal(decl.Parameters)
	if err != nil {
		return nil, fmt.Errorf("encoding the parameters schema: %w", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(b, &schema); err != nil {
		return nil, fmt.Errorf("decoding the parameters schema: %w", err)
	}
	lowerTypes(schema)
	return schema, nil
}

func lowerTypes(v any) {
	switch v := v.(type) {
	case map[string]any:
		for k, child := range v {
			if s, ok := child.(string); ok && k == "type" {
				v[k] = strings.ToLower(s)
				continue
			}
			lowerTypes(child)
		}
	case []any:
		for _, child := range v {
			lowerTypes(child)
		}
	}
}

// invocationRun is an ADK invocation of the agent, which the tools the
// harness calls during the invocation run in.
type invocationRun struct {
	ic     agent.InvocationContext
	events chan toolEvent
	done   chan struct{} // closed when the invocation yields no more events
}

// toolEvent is an event of a tool call, and a channel closed once the
// invocation yielded it.
type toolEvent struct {
	event   *session.Event
	yielded chan struct{}
}

type runKey struct{}

// withRun runs the agent inner, yielding the events of the tools the harness
// calls between its events.
func withRun(ic agent.InvocationContext, s *toolServer, inner agent.Agent) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		run := &invocationRun{ic: ic, events: make(chan toolEvent), done: make(chan struct{})}
		ctx, cancel := context.WithCancel(ic)
		type result struct {
			event *session.Event
			err   error
		}
		results := make(chan result)
		go func() {
			defer close(results)
			// The client provider finds the run in the context and binds it
			// to the A2A context of the message.
			innerCtx := ic.WithContext(context.WithValue(ctx, runKey{}, run))
			for ev, err := range inner.Run(innerCtx) {
				select {
				case results <- result{ev, err}:
				case <-ctx.Done():
					return
				}
			}
		}()
		defer func() {
			close(run.done)
			s.unbind(run)
			cancel()
			for range results {
			}
		}()

		for {
			select {
			case r, ok := <-results:
				if !ok || !yield(r.event, r.err) {
					return
				}
			case te := <-run.events:
				ok := yield(te.event, nil)
				close(te.yielded)
				if !ok {
					return
				}
			}
		}
	}
}

// runFromContext returns the run in ctx, or nil if there is none.
func runFromContext(ctx context.Context) *invocationRun {
	run, _ := ctx.Value(runKey{}).(*invocationRun)
	return run
}

// callTool runs t with the invocation and emits its call and response as
// events of the invocation. The error is that of the tool or of the
// invocation, which the harness gets as the tool's error.
func (r *invocationRun) callTool(ctx context.Context, author, contextID string, t toolinternal.FunctionTool, args map[string]any) (map[string]any, error) {
	// The tool's context carries the invocation's values, and is canceled
	// when the invocation or the harness's request ends.
	toolCtx, cancel := context.WithCancel(r.ic)
	defer cancel()
	defer context.AfterFunc(ctx, cancel)()
	ic := r.ic.WithContext(toolCtx)

	callID := utils.GenerateFunctionCallID(ic)
	call := r.newEvent(author, contextID, genai.RoleModel, &genai.Part{FunctionCall: &genai.FunctionCall{ID: callID, Name: t.Name(), Args: args}})
	if err := r.emit(toolCtx, call); err != nil {
		return nil, err
	}

	actions := &session.EventActions{StateDelta: map[string]any{}}
	result, toolErr := t.Run(agent.NewToolContext(ic, callID, actions, nil), args)
	response := result
	if toolErr != nil {
		response = map[string]any{"error": toolErr.Error()}
	}
	resp := r.newEvent(author, contextID, genai.RoleUser, &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: callID, Name: t.Name(), Response: response}})
	resp.Actions = *actions
	if err := r.emit(toolCtx, resp); err != nil {
		return nil, err
	}
	return result, toolErr
}

func (r *invocationRun) newEvent(author, contextID, role string, part *genai.Part) *session.Event {
	ev := session.NewEvent(r.ic, r.ic.InvocationID())
	ev.Author = author
	ev.Content = &genai.Content{Role: role, Parts: []*genai.Part{part}}
	// remoteagent continues the A2A context of the agent's last event, which
	// may be this one.
	ev.CustomMetadata = map[string]any{adka2a.ToADKMetaKey("context_id"): contextID}
	return ev
}

// emit yields ev from the invocation and waits until it was yielded, so that
// the session has the tool's changes before the harness continues.
func (r *invocationRun) emit(ctx context.Context, ev *session.Event) error {
	te := toolEvent{event: ev, yielded: make(chan struct{})}
	select {
	case r.events <- te:
	case <-r.done:
		return errNoInvocation
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-te.yielded:
		return nil
	case <-r.done:
		return errNoInvocation
	}
}
