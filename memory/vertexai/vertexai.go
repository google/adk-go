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

// Package vertexai provides a Vertex AI Memory Bank implementation of
// memory.Service.
//
// Memory Bank is the long-term memory feature in Vertex AI Agent Engine. This
// package connects ADK's generic memory.Service interface to that hosted
// service, so callers can ask Memory Bank to extract durable facts from past
// conversations and later search those facts for a user.
//
// Basic setup:
//
//	ctx := context.Background()
//
//	mem, err := vertexai.NewMemoryService(ctx, vertexai.VertexAIServiceConfig{
//		ProjectID:       "my-project",
//		Location:        "us-central1",
//		ReasoningEngine: "1234567890",
//	})
//	if err != nil {
//		// Handle client creation or authentication error.
//	}
//
// NewMemoryService returns vertexai.Service, which extends ADK's generic
// memory.Service with Vertex AI-specific helpers. It can still be passed
// anywhere ADK expects memory.Service. Production code usually authenticates
// with Application Default Credentials. Tests can pass google.golang.org/api/option
// values such as option.WithoutAuthentication.
//
// Most Agent Engine applications use this package together with the Vertex AI
// session service. In that setup, Memory Bank can read a persisted Vertex AI
// session directly. The application only needs to pass the reasoning engine
// app name and session ID:
//
//	result, err := mem.GenerateMemoriesFromVertexSession(
//		ctx,
//		cbCtx.AppName(),
//		cbCtx.SessionID(),
//		vertexai.WithWaitForCompletion(false),
//	)
//	if err != nil {
//		// Handle GenerateMemories failure.
//	}
//	// result.OperationName can be logged if asynchronous generation fails later.
//
// This uses Memory Bank's vertex_session_source. It is the best fit when your
// session service is session/vertexai because Memory Bank reads the persisted
// session events directly from Agent Engine. You do not need to fetch the full
// session object before generating memories.
//
// AddSessionToMemory is still implemented for ADK's generic memory.Service
// interface. Use it when you are working with generic memory.Service code and
// already have a session.Session value:
//
//	// sess must identify a session that already exists in Vertex AI Sessions.
//	if err := mem.AddSessionToMemory(ctx, sess); err != nil {
//		// Handle GenerateMemories failure.
//	}
//
// Search uses the generic memory.Service method:
//
//	resp, err := mem.SearchMemory(ctx, &memory.SearchRequest{
//		AppName: "1234567890",
//		UserID:  "user-1",
//		Query:   "What answer style does the user prefer?",
//	})
//	if err != nil {
//		// Handle retrieval failure.
//	}
//	for _, entry := range resp.Memories {
//		// entry.Content contains the remembered fact as GenAI content.
//	}
//
// ADK's built-in load_memory and preload_memory tools also use this generic
// SearchMemory path. Pass this service into your runner or launcher
// configuration as the memory service to make those tools work:
//
//	launcher.Config{
//		SessionService: sessionSvc,
//		MemoryService:  mem,
//	}
//
// By default, direct memory generation and search both use the user scope
// {"user_id": userID}. Vertex session generation lets Memory Bank derive the
// default user scope from the persisted session. In both cases, generated
// memories are visible to the built-in memory tools when the same user asks a
// later question.
//
// For advanced Vertex AI behavior, use the returned vertexai.Service value:
//
//	vertexMem, err := vertexai.NewMemoryService(ctx, cfg)
//	if err != nil {
//		// Handle error.
//	}
//
// To generate memories from a Vertex session with request controls, use
// GenerateMemoriesFromVertexSession:
//
//	result, err := vertexMem.GenerateMemoriesFromVertexSession(
//		ctx,
//		"1234567890",
//		"session-1",
//		vertexai.WithEventTimeRange(start, end),
//		vertexai.WithDisableConsolidation(false),
//		vertexai.WithWaitForCompletion(true),
//	)
//	if err != nil {
//		// Handle error.
//	}
//	for _, generated := range result.GeneratedMemories {
//		// generated.Action is CREATED, UPDATED, or DELETED.
//	}
//
// Memory Bank also supports direct source data. Use GenerateMemoriesFromEvents
// when the conversation exists as ADK session events but was not stored in
// Vertex AI Sessions:
//
//	result, err := vertexMem.GenerateMemoriesFromEvents(
//		ctx,
//		"1234567890",
//		"user-1",
//		[]*session.Event{event1, event2},
//	)
//
// Use GenerateMemoriesFromContents when you already have GenAI content messages:
//
//	result, err := vertexMem.GenerateMemoriesFromContents(
//		ctx,
//		"1234567890",
//		"user-1",
//		[]*genai.Content{
//			genai.NewContentFromText("Remember that I prefer concise answers.", genai.RoleUser),
//		},
//	)
//
// Use GenerateMemoriesFromFacts when facts have already been extracted. This
// maps to Memory Bank's direct_memories_source and accepts at most five facts in
// one request:
//
//	result, err := vertexMem.GenerateMemoriesFromFacts(
//		ctx,
//		"1234567890",
//		"user-1",
//		[]string{"The user prefers concise answers."},
//	)
//
// For lower-latency production flows, generate memories asynchronously:
//
//	result, err := vertexMem.GenerateMemoriesFromFacts(
//		ctx,
//		"1234567890",
//		"user-1",
//		[]string{"The user prefers concise answers."},
//		vertexai.WithWaitForCompletion(false),
//	)
//	// result.OperationName can be logged for debugging. GeneratedMemories is
//	// empty because the operation is still running.
//
// Custom scopes should be used deliberately. Memory Bank requires exact scope
// matching: a memory generated with {"user_id": "user-1", "tenant_id": "t1"}
// will not be found by a search that only uses {"user_id": "user-1"}.
//
//	scope := map[string]string{"user_id": "user-1", "tenant_id": "t1"}
//
//	_, err = vertexMem.GenerateMemoriesFromFacts(
//		ctx,
//		"1234567890",
//		"user-1",
//		[]string{"The user prefers concise answers."},
//		vertexai.WithScope(scope),
//	)
//	if err != nil {
//		// Handle error.
//	}
//
//	resp, err := vertexMem.SearchMemoryWithScope(ctx, &memory.SearchRequest{
//		AppName: "1234567890",
//		Query:   "answer style",
//	}, scope)
//
// The built-in load_memory and preload_memory tools search only the default
// user scope. If your application generates memories with WithScope, provide a
// custom memory-loading tool or wrap the memory service so generic SearchMemory
// applies the same scope consistently.
//
// Some Memory Bank features documented for Python, such as memory topic
// allow-lists and metadata merge strategy, are not exposed by the Go
// cloud.google.com/go/aiplatform v1.125.0 GenerateMemoriesRequest type yet.
package vertexai

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
)

type vertexAIService struct {
	client *vertexAIClient
}

// Service is a Vertex AI Memory Bank service.
//
// It embeds memory.Service so it works anywhere ADK expects the generic memory
// service, and adds methods for Memory Bank data sources and controls that do
// not fit in the generic interface.
type Service interface {
	memory.Service

	GenerateMemoriesFromVertexSession(context.Context, string, string, ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error)
	AddSessionToMemoryWithOptions(context.Context, session.Session, ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error)
	GenerateMemoriesFromEvents(context.Context, string, string, []*session.Event, ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error)
	GenerateMemoriesFromContents(context.Context, string, string, []*genai.Content, ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error)
	GenerateMemoriesFromFacts(context.Context, string, string, []string, ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error)
	SearchMemoryWithScope(context.Context, *memory.SearchRequest, map[string]string) (*memory.SearchResponse, error)
}

// AddSessionToMemoryOption configures a single GenerateMemories request.
//
// These options map to fields that are available on the generated Go
// GenerateMemoriesRequest type. Memory topic allow-lists and metadata merge
// strategy are documented for the Python client, but are not exposed by the
// Go v1.125.0 request type yet.
type AddSessionToMemoryOption func(*addSessionToMemoryOptions)

type addSessionToMemoryOptions struct {
	scope                map[string]string
	disableConsolidation bool
	startTime            time.Time
	endTime              time.Time
	waitForCompletion    bool
}

// GeneratedMemory describes one memory changed by GenerateMemories.
//
// Memory Bank can create, update, or delete memories during consolidation. The
// Action field reports which of those changes happened.
type GeneratedMemory struct {
	Name   string
	Fact   string
	Action string
}

// GenerateMemoriesResult contains information returned by a GenerateMemories
// operation.
//
// If WaitForCompletion is false, GeneratedMemories will be empty because the
// operation is still running in Vertex AI.
type GenerateMemoriesResult struct {
	OperationName     string
	GeneratedMemories []GeneratedMemory
}

// VertexAIServiceConfig contains the Google Cloud resource settings needed to
// call Vertex AI Memory Bank.
//
// A reasoning engine is the Agent Engine runtime resource that owns the Memory
// Bank data. This package follows the same convention as session/vertexai:
// callers can either set ReasoningEngine once here, or leave it empty and pass
// the reasoning engine ID/resource name in Session.AppName or SearchRequest.AppName.
type VertexAIServiceConfig struct {
	// ProjectID is the Google Cloud project with Vertex AI API enabled.
	ProjectID string
	// Location is the Google Cloud region where the reasoning engine is running.
	Location string
	// ReasoningEngine is the runtime in Agent Engine that stores memories.
	// Optional: if unset, the service uses SearchRequest.AppName or Session.AppName
	// as either a reasoning engine numeric ID or full resource name.
	ReasoningEngine string
}

// NewMemoryService creates a memory.Service backed by Vertex AI Memory Bank.
//
// opts are forwarded to the generated Vertex AI client. Tests can pass client
// options such as option.WithoutAuthentication or replay dial options; production
// callers usually rely on Application Default Credentials.
func NewMemoryService(ctx context.Context, cfg VertexAIServiceConfig, opts ...option.ClientOption) (Service, error) {
	client, err := newVertexAIClient(ctx, cfg.Location, cfg.ProjectID, cfg.ReasoningEngine, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Vertex AI memory client: %w", err)
	}

	return &vertexAIService{client: client}, nil
}

// WithScope overrides the Memory Bank scope for generated memories.
//
// When this option is not provided for a Vertex session source, Memory Bank
// derives the default scope from the session as {"user_id": session.user_id}.
// Use this option for custom isolation boundaries, such as user plus session or
// user plus tenant. Retrieval must use the exact same scope to find the memory.
//
// Important: Memory Bank uses exact scope matching. The scope you use when
// generating a memory must be the same scope you use when searching for it.
// For example, a memory generated with {"user_id": "u1", "tenant_id": "t1"}
// will not be returned by a search that only uses {"user_id": "u1"}.
//
// The built-in load_memory and preload_memory tools do not know about
// Vertex-specific custom scopes. They call the generic memory.Service
// SearchMemory method, which searches the default user scope
// {"user_id": current_user_id}. That default works when memories are generated
// without WithScope, but it will not find memories generated with a custom
// scope.
//
// If your application uses WithScope, make sure the retrieval path uses the same
// scope too. Common ways to do that are to provide your own memory-loading tool
// that calls SearchMemoryWithScope, or to wrap the memory service so its generic
// SearchMemory method applies your app's custom scope consistently.
func WithScope(scope map[string]string) AddSessionToMemoryOption {
	return func(opts *addSessionToMemoryOptions) {
		opts.scope = scope
	}
}

// WithDisableConsolidation controls whether Memory Bank should merge extracted
// facts with existing memories in the same scope.
//
// By default, Memory Bank consolidates facts so duplicate or contradictory
// memories can be updated or removed. Setting this to true asks Memory Bank to
// add all generated memories as new memories.
func WithDisableConsolidation(disable bool) AddSessionToMemoryOption {
	return func(opts *addSessionToMemoryOptions) {
		opts.disableConsolidation = disable
	}
}

// WithEventTimeRange limits which session events Memory Bank uses as source
// material.
//
// start is inclusive and end is exclusive. Pass a zero time for either value to
// leave that side of the range unbounded.
func WithEventTimeRange(start, end time.Time) AddSessionToMemoryOption {
	return func(opts *addSessionToMemoryOptions) {
		opts.startTime = start
		opts.endTime = end
	}
}

// WithWaitForCompletion controls whether AddSessionToMemoryWithOptions waits
// for the long-running GenerateMemories operation to finish.
//
// Waiting is useful for tests, debugging, and workflows that need to inspect
// generated memory actions. Production agents often set this to false to avoid
// adding memory generation latency to the current user interaction.
func WithWaitForCompletion(wait bool) AddSessionToMemoryOption {
	return func(opts *addSessionToMemoryOptions) {
		opts.waitForCompletion = wait
	}
}

// AddSessionToMemory asks Memory Bank to generate long-term memories from a
// Vertex AI session.
//
// The session must already exist in Vertex AI Sessions. Memory Bank reads the
// persisted session events, extracts useful facts, and stores them under the
// user's scope. This method waits for the long-running GenerateMemories
// operation to finish before returning.
func (s *vertexAIService) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil || sess.AppName() == "" || sess.UserID() == "" || sess.ID() == "" {
		return fmt.Errorf("session, app_name, user_id and session_id are required")
	}
	if _, err := s.client.generateMemoriesFromVertexSession(ctx, sess.AppName(), sess.ID(), defaultAddSessionToMemoryOptions()); err != nil {
		return fmt.Errorf("failed to generate memories: %w", err)
	}
	return nil
}

// GenerateMemoriesFromVertexSession asks Memory Bank to generate memories from
// a Vertex AI session resource.
//
// This is the lightest Vertex-session API because Memory Bank only needs the
// reasoning engine, represented by appName, and the session ID. It does not
// need the full ADK session object. Use this from callbacks when you already
// have AppName and SessionID available and want to avoid an extra session Get
// call.
func (s *vertexAIService) GenerateMemoriesFromVertexSession(ctx context.Context, appName, sessionID string, opts ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error) {
	if appName == "" || sessionID == "" {
		return nil, fmt.Errorf("app_name and session_id are required")
	}

	res, err := s.client.generateMemoriesFromVertexSession(ctx, appName, sessionID, applyAddSessionToMemoryOptions(opts...))
	if err != nil {
		return nil, fmt.Errorf("failed to generate memories from Vertex session: %w", err)
	}
	return res, nil
}

// AddSessionToMemoryWithOptions asks Memory Bank to generate memories from a
// session with Vertex AI-specific request controls.
//
// The generic memory.Service interface intentionally stays small, so options
// like explicit scope, event time windows, and asynchronous generation live on
// this Vertex AI implementation. If you only have AppName and SessionID, prefer
// GenerateMemoriesFromVertexSession to avoid fetching the full session object.
func (s *vertexAIService) AddSessionToMemoryWithOptions(ctx context.Context, sess session.Session, opts ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error) {
	if sess == nil || sess.AppName() == "" || sess.UserID() == "" || sess.ID() == "" {
		return nil, fmt.Errorf("session, app_name, user_id and session_id are required")
	}

	res, err := s.client.generateMemoriesFromVertexSession(ctx, sess.AppName(), sess.ID(), applyAddSessionToMemoryOptions(opts...))
	if err != nil {
		return nil, fmt.Errorf("failed to generate memories: %w", err)
	}
	return res, nil
}

// GenerateMemoriesFromEvents asks Memory Bank to extract memories from direct
// ADK session events instead of reading a Vertex AI session.
//
// Use this when the conversation is available in your application, but was not
// stored in Vertex AI Sessions. The events are converted to Memory Bank's
// direct_contents_source format. By default, generated memories use the same
// user scope as SearchMemory: {"user_id": userID}.
//
// WithEventTimeRange is only meaningful for Vertex session sources and is
// rejected here.
func (s *vertexAIService) GenerateMemoriesFromEvents(ctx context.Context, appName, userID string, events []*session.Event, opts ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error) {
	if appName == "" || userID == "" {
		return nil, fmt.Errorf("app_name and user_id are required")
	}
	if len(events) == 0 {
		return nil, fmt.Errorf("events are required")
	}

	res, err := s.client.generateMemoriesFromEvents(ctx, appName, userID, events, applyAddSessionToMemoryOptions(opts...))
	if err != nil {
		return nil, fmt.Errorf("failed to generate memories from events: %w", err)
	}
	return res, nil
}

// GenerateMemoriesFromContents asks Memory Bank to extract memories from direct
// GenAI contents.
//
// This is the closest Go API to Memory Bank's direct_contents_source. It is
// useful when the caller has already built content messages and does not have
// ADK session.Event values. By default, generated memories use
// {"user_id": userID}.
//
// WithEventTimeRange is only meaningful for Vertex session sources and is
// rejected here.
func (s *vertexAIService) GenerateMemoriesFromContents(ctx context.Context, appName, userID string, contents []*genai.Content, opts ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error) {
	if appName == "" || userID == "" {
		return nil, fmt.Errorf("app_name and user_id are required")
	}
	if len(contents) == 0 {
		return nil, fmt.Errorf("contents are required")
	}

	res, err := s.client.generateMemoriesFromContents(ctx, appName, userID, contents, applyAddSessionToMemoryOptions(opts...))
	if err != nil {
		return nil, fmt.Errorf("failed to generate memories from contents: %w", err)
	}
	return res, nil
}

// GenerateMemoriesFromFacts asks Memory Bank to consolidate pre-extracted facts.
//
// This maps to Memory Bank's direct_memories_source. Use it when your
// application, another model, or a human has already decided exactly what fact
// should be remembered. Memory Bank will still consolidate these facts with
// existing memories in the same scope unless WithDisableConsolidation(true) is
// used.
//
// Memory Bank allows at most five direct memories in one request. By default,
// generated memories use {"user_id": userID}. WithEventTimeRange is only
// meaningful for Vertex session sources and is rejected here.
func (s *vertexAIService) GenerateMemoriesFromFacts(ctx context.Context, appName, userID string, facts []string, opts ...AddSessionToMemoryOption) (*GenerateMemoriesResult, error) {
	if appName == "" || userID == "" {
		return nil, fmt.Errorf("app_name and user_id are required")
	}
	if len(facts) == 0 {
		return nil, fmt.Errorf("facts are required")
	}
	if len(facts) > 5 {
		return nil, fmt.Errorf("at most 5 facts are allowed")
	}

	res, err := s.client.generateMemoriesFromFacts(ctx, appName, userID, facts, applyAddSessionToMemoryOptions(opts...))
	if err != nil {
		return nil, fmt.Errorf("failed to generate memories from facts: %w", err)
	}
	return res, nil
}

// SearchMemory retrieves memories for the user that are relevant to req.Query.
//
// Memory Bank keeps memories isolated by scope. This implementation uses the
// same default scope as the Vertex AI docs and Python ADK examples:
// {"user_id": req.UserID}. Only memories in that exact scope can be returned.
func (s *vertexAIService) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil || req.AppName == "" || req.UserID == "" || req.Query == "" {
		return nil, fmt.Errorf("app_name, user_id and query are required")
	}

	memories, err := s.client.retrieveMemories(ctx, req, scopeForUser(req.UserID))
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve memories: %w", err)
	}
	return &memory.SearchResponse{Memories: memories}, nil
}

// SearchMemoryWithScope retrieves memories from an explicit Memory Bank scope.
//
// Use this when memories were generated with WithScope. Memory Bank requires an
// exact scope match for retrieval; a memory generated with {"user_id": "u",
// "tenant_id": "t"} will not be returned by searching only {"user_id": "u"}.
func (s *vertexAIService) SearchMemoryWithScope(ctx context.Context, req *memory.SearchRequest, scope map[string]string) (*memory.SearchResponse, error) {
	if req == nil || req.AppName == "" || req.Query == "" {
		return nil, fmt.Errorf("app_name and query are required")
	}
	if len(scope) == 0 {
		return nil, fmt.Errorf("scope is required")
	}

	memories, err := s.client.retrieveMemories(ctx, req, scope)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve memories: %w", err)
	}
	return &memory.SearchResponse{Memories: memories}, nil
}

var (
	_ memory.Service = (*vertexAIService)(nil)
	_ Service        = (*vertexAIService)(nil)
)

func defaultAddSessionToMemoryOptions() addSessionToMemoryOptions {
	return addSessionToMemoryOptions{
		waitForCompletion: true,
	}
}

func applyAddSessionToMemoryOptions(opts ...AddSessionToMemoryOption) addSessionToMemoryOptions {
	out := defaultAddSessionToMemoryOptions()
	for _, opt := range opts {
		opt(&out)
	}
	return out
}
