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

package vertexai

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
)

const (
	engineResourceTemplate  = "projects/%s/locations/%s/reasoningEngines/%s"
	sessionResourceTemplate = engineResourceTemplate + "/sessions/%s"
)

// memoryBankClient is a small interface around the generated Google Cloud
// client. Keeping this interface narrow makes the service easy to unit test
// without making live calls to Vertex AI.
type memoryBankClient interface {
	Close() error
	GenerateMemories(context.Context, *aiplatformpb.GenerateMemoriesRequest) (generateMemoriesOperation, error)
	RetrieveMemories(context.Context, *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error)
}

// generateMemoriesOperation hides the generated long-running operation type.
// Tests can replace it with a small fake that records whether Wait was called.
type generateMemoriesOperation interface {
	Name() string
	Wait(context.Context) (*aiplatformpb.GenerateMemoriesResponse, error)
}

type vertexAIClient struct {
	location        string
	projectID       string
	reasoningEngine string
	rpcClient       memoryBankClient
}

// newVertexAIClient creates the low-level Memory Bank client used by the
// service.
//
// The generated Google Cloud client handles authentication, endpoint selection,
// retries, and long-running operations. This wrapper stores the resource
// identity pieces so the rest of the package can build request names in one
// consistent way.
func newVertexAIClient(ctx context.Context, location, projectID, reasoningEngine string, opts ...option.ClientOption) (*vertexAIClient, error) {
	rpcClient, err := aiplatform.NewMemoryBankClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("could not establish connection to the aiplatform server: %w", err)
	}
	return &vertexAIClient{location: location, projectID: projectID, reasoningEngine: reasoningEngine, rpcClient: &memoryBankClientAdapter{client: rpcClient}}, nil
}

// Close closes the underlying Vertex AI client.
func (c *vertexAIClient) Close() error {
	return c.rpcClient.Close()
}

// generateMemoriesFromVertexSession builds a GenerateMemories request that uses
// a Vertex AI session as the source data.
//
// This maps to Memory Bank's vertex_session_source. Memory Bank will read the
// session events that were already stored in Vertex AI Sessions, so the session
// ID alone is not enough; the request must include the full session resource
// name. Start and end times are only supported for this source type.
func (c *vertexAIClient) generateMemoriesFromVertexSession(ctx context.Context, appName, sessionID string, opts addSessionToMemoryOptions) (*GenerateMemoriesResult, error) {
	reasoningEngine, err := c.getReasoningEngineID(appName)
	if err != nil {
		return nil, err
	}

	sessionSource := &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource{
		Session: fmt.Sprintf(sessionResourceTemplate, c.projectID, c.location, reasoningEngine, sessionID),
	}
	if !opts.startTime.IsZero() {
		sessionSource.StartTime = timestamppb.New(opts.startTime)
	}
	if !opts.endTime.IsZero() {
		sessionSource.EndTime = timestamppb.New(opts.endTime)
	}

	req, err := c.newGenerateMemoriesRequest(appName, "", opts)
	if err != nil {
		return nil, err
	}
	req.Source = &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource_{
		VertexSessionSource: sessionSource,
	}

	// Memory Bank can generate memories directly from a Vertex AI session. The
	// session name must be the full resource path, not just the session ID.
	return c.generateMemoriesFromRequest(ctx, req, opts)
}

// generateMemoriesFromEvents converts ADK session events into direct content
// messages and sends them to Memory Bank.
//
// Use this path when a conversation is available locally but was not persisted
// as a Vertex AI session. Events without content are ignored because Memory
// Bank needs message content to extract memories.
func (c *vertexAIClient) generateMemoriesFromEvents(ctx context.Context, appName, userID string, events []*session.Event, opts addSessionToMemoryOptions) (*GenerateMemoriesResult, error) {
	contents := make([]*genai.Content, 0, len(events))
	for _, event := range events {
		if event == nil || event.Content == nil {
			continue
		}
		contents = append(contents, event.Content)
	}
	if len(contents) == 0 {
		return nil, fmt.Errorf("events must contain at least one content message")
	}
	return c.generateMemoriesFromContents(ctx, appName, userID, contents, opts)
}

// generateMemoriesFromContents sends GenAI content messages using Memory Bank's
// direct_contents_source.
//
// This source type is useful when the caller already has chat history as
// genai.Content values. Event time ranges are rejected here because the Memory
// Bank API only defines start/end time fields on vertex_session_source.
func (c *vertexAIClient) generateMemoriesFromContents(ctx context.Context, appName, userID string, contents []*genai.Content, opts addSessionToMemoryOptions) (*GenerateMemoriesResult, error) {
	if hasEventTimeRange(opts) {
		return nil, fmt.Errorf("event time range is only supported for Vertex session sources")
	}

	directEvents := make([]*aiplatformpb.GenerateMemoriesRequest_DirectContentsSource_Event, 0, len(contents))
	for _, content := range contents {
		if content == nil {
			continue
		}
		pbContent, err := genaiContentToAiplatformContent(content)
		if err != nil {
			return nil, err
		}
		directEvents = append(directEvents, &aiplatformpb.GenerateMemoriesRequest_DirectContentsSource_Event{
			Content: pbContent,
		})
	}
	if len(directEvents) == 0 {
		return nil, fmt.Errorf("contents must contain at least one non-empty content message")
	}

	req, err := c.newGenerateMemoriesRequest(appName, userID, opts)
	if err != nil {
		return nil, err
	}
	req.Source = &aiplatformpb.GenerateMemoriesRequest_DirectContentsSource_{
		DirectContentsSource: &aiplatformpb.GenerateMemoriesRequest_DirectContentsSource{
			Events: directEvents,
		},
	}
	return c.generateMemoriesFromRequest(ctx, req, opts)
}

// generateMemoriesFromFacts uploads already-extracted facts using Memory Bank's
// direct_memories_source.
//
// Unlike direct_contents_source, this does not ask Memory Bank to extract facts
// from a conversation. It starts from facts the caller already trusts, then lets
// Memory Bank consolidate them with existing memories for the same scope.
func (c *vertexAIClient) generateMemoriesFromFacts(ctx context.Context, appName, userID string, facts []string, opts addSessionToMemoryOptions) (*GenerateMemoriesResult, error) {
	if hasEventTimeRange(opts) {
		return nil, fmt.Errorf("event time range is only supported for Vertex session sources")
	}

	directMemories := make([]*aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource_DirectMemory, 0, len(facts))
	for _, fact := range facts {
		if fact == "" {
			continue
		}
		directMemories = append(directMemories, &aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource_DirectMemory{
			Fact: fact,
		})
	}
	if len(directMemories) == 0 {
		return nil, fmt.Errorf("facts must contain at least one non-empty fact")
	}

	req, err := c.newGenerateMemoriesRequest(appName, userID, opts)
	if err != nil {
		return nil, err
	}
	req.Source = &aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource_{
		DirectMemoriesSource: &aiplatformpb.GenerateMemoriesRequest_DirectMemoriesSource{
			DirectMemories: directMemories,
		},
	}
	return c.generateMemoriesFromRequest(ctx, req, opts)
}

// newGenerateMemoriesRequest builds the source-independent part of a
// GenerateMemories request.
//
// Every source type needs the same parent reasoning engine, consolidation
// setting, and optional scope. If the caller did not pass WithScope and userID
// is available, direct sources default to {"user_id": userID} so generic ADK
// memory search and built-in memory tools can find the generated memories.
// Vertex session generation passes an empty userID here so Memory Bank can use
// the scope embedded in the session source by default.
func (c *vertexAIClient) newGenerateMemoriesRequest(appName, userID string, opts addSessionToMemoryOptions) (*aiplatformpb.GenerateMemoriesRequest, error) {
	reasoningEngine, err := c.getReasoningEngineID(appName)
	if err != nil {
		return nil, err
	}

	req := &aiplatformpb.GenerateMemoriesRequest{
		Parent:               fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, reasoningEngine),
		DisableConsolidation: opts.disableConsolidation,
	}
	if opts.scope != nil {
		req.Scope = opts.scope
	} else if userID != "" {
		req.Scope = scopeForUser(userID)
	}
	return req, nil
}

// generateMemoriesFromRequest sends the request and handles the long-running
// operation.
//
// When waitForCompletion is false, the caller gets the operation name
// immediately and GeneratedMemories remains empty. When waiting is enabled, the
// final response is mapped into the package's small GenerateMemoriesResult type.
func (c *vertexAIClient) generateMemoriesFromRequest(ctx context.Context, req *aiplatformpb.GenerateMemoriesRequest, opts addSessionToMemoryOptions) (*GenerateMemoriesResult, error) {
	op, err := c.rpcClient.GenerateMemories(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("error generating memories: %w", err)
	}

	result := &GenerateMemoriesResult{OperationName: op.Name()}
	if !opts.waitForCompletion {
		return result, nil
	}

	resp, err := op.Wait(ctx)
	if err != nil {
		return nil, fmt.Errorf("error waiting for GenerateMemories operation: %w", err)
	}
	result.GeneratedMemories = generatedMemoriesFromResponse(resp)
	return result, nil
}

// hasEventTimeRange reports whether a caller asked to limit source events by
// time.
//
// Memory Bank only supports this on vertex_session_source. Direct sources are
// already explicit slices of content or facts, so they reject this option.
func hasEventTimeRange(opts addSessionToMemoryOptions) bool {
	return !opts.startTime.IsZero() || !opts.endTime.IsZero()
}

// retrieveMemories runs a similarity search in one exact Memory Bank scope and
// converts the returned Memory resources into ADK memory entries.
//
// The scope must be the same scope used at generation time. Memory Bank does not
// do partial scope matching, so {"user_id": "u1"} and {"user_id": "u1",
// "tenant_id": "t1"} are different collections of memories.
func (c *vertexAIClient) retrieveMemories(ctx context.Context, req *memory.SearchRequest, scope map[string]string) ([]memory.Entry, error) {
	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}

	// Similarity search asks Memory Bank for memories that are semantically
	// close to the user's query instead of returning every memory in the scope.
	resp, err := c.rpcClient.RetrieveMemories(ctx, &aiplatformpb.RetrieveMemoriesRequest{
		Parent: fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, reasoningEngine),
		Scope:  scope,
		RetrievalParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams_{
			SimilaritySearchParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams{
				SearchQuery: req.Query,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("error retrieving memories: %w", err)
	}

	entries := make([]memory.Entry, 0, len(resp.GetRetrievedMemories()))
	for _, retrieved := range resp.GetRetrievedMemories() {
		m := retrieved.GetMemory()
		if m == nil {
			continue
		}
		// Memory Bank stores each memory as a text fact. ADK memory.Entry expects
		// model content, so wrap the fact in a genai.Content value.
		entries = append(entries, memory.Entry{
			ID:             m.GetName(),
			Content:        genai.NewContentFromText(m.GetFact(), genai.RoleModel),
			Author:         "memory",
			Timestamp:      memoryTimestamp(m),
			CustomMetadata: map[string]any{"scope": m.GetScope()},
		})
	}
	return entries, nil
}

// scopeForUser returns the Memory Bank scope used by this service.
//
// Scope is part of Memory Bank's isolation model: retrieval only returns
// memories whose scope exactly matches the requested scope.
func scopeForUser(userID string) map[string]string {
	return map[string]string{"user_id": userID}
}

// memoryTimestamp picks the best timestamp available on a Memory Bank memory.
// Update time is preferred because generated memories may be consolidated and
// refined over time after their initial creation.
func memoryTimestamp(m *aiplatformpb.Memory) time.Time {
	if m.GetUpdateTime() != nil {
		return m.GetUpdateTime().AsTime()
	}
	if m.GetCreateTime() != nil {
		return m.GetCreateTime().AsTime()
	}
	return time.Time{}
}

// generatedMemoriesFromResponse converts Memory Bank's operation response into
// the public result type returned by this package.
//
// Each generated memory includes the final fact and the action Memory Bank took
// during consolidation, such as CREATED, UPDATED, or DELETED.
func generatedMemoriesFromResponse(resp *aiplatformpb.GenerateMemoriesResponse) []GeneratedMemory {
	out := make([]GeneratedMemory, 0, len(resp.GetGeneratedMemories()))
	for _, generated := range resp.GetGeneratedMemories() {
		m := generated.GetMemory()
		if m == nil {
			continue
		}
		out = append(out, GeneratedMemory{
			Name:   m.GetName(),
			Fact:   m.GetFact(),
			Action: generated.GetAction().String(),
		})
	}
	return out
}

// genaiContentToAiplatformContent converts the GenAI SDK content type into the
// Vertex AI proto content type used by Memory Bank.
//
// The two types represent the same concept but live in different Go packages.
// Unsupported or empty parts are skipped; the caller gets an error if nothing
// supported remains because Memory Bank requires each direct event to contain
// content.
func genaiContentToAiplatformContent(content *genai.Content) (*aiplatformpb.Content, error) {
	if content == nil {
		return nil, nil
	}

	parts := make([]*aiplatformpb.Part, 0, len(content.Parts))
	for _, part := range content.Parts {
		pbPart, err := genaiPartToAiplatformPart(part)
		if err != nil {
			return nil, err
		}
		if pbPart != nil {
			parts = append(parts, pbPart)
		}
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("content must contain at least one supported part")
	}
	return &aiplatformpb.Content{
		Role:  content.Role,
		Parts: parts,
	}, nil
}

// genaiPartToAiplatformPart converts one GenAI part into the matching Vertex AI
// proto part.
//
// Memory generation only needs conversation content, so this converter handles
// the common part types used in chat history: text, inline data, function calls,
// and function responses. Unknown part types return nil so callers can ignore
// them instead of failing an entire request.
func genaiPartToAiplatformPart(part *genai.Part) (*aiplatformpb.Part, error) {
	if part == nil {
		return nil, nil
	}

	out := &aiplatformpb.Part{
		Thought:          part.Thought,
		ThoughtSignature: part.ThoughtSignature,
	}
	switch {
	case part.Text != "":
		out.Data = &aiplatformpb.Part_Text{Text: part.Text}
	case part.InlineData != nil:
		out.Data = &aiplatformpb.Part_InlineData{
			InlineData: &aiplatformpb.Blob{
				MimeType: part.InlineData.MIMEType,
				Data:     part.InlineData.Data,
			},
		}
	case part.FunctionCall != nil:
		args, err := structpb.NewStruct(part.FunctionCall.Args)
		if err != nil {
			return nil, fmt.Errorf("failed to convert function call args: %w", err)
		}
		out.Data = &aiplatformpb.Part_FunctionCall{
			FunctionCall: &aiplatformpb.FunctionCall{
				Id:   part.FunctionCall.ID,
				Name: part.FunctionCall.Name,
				Args: args,
			},
		}
	case part.FunctionResponse != nil:
		response, err := structpb.NewStruct(part.FunctionResponse.Response)
		if err != nil {
			return nil, fmt.Errorf("failed to convert function response: %w", err)
		}
		out.Data = &aiplatformpb.Part_FunctionResponse{
			FunctionResponse: &aiplatformpb.FunctionResponse{
				Id:       part.FunctionResponse.ID,
				Name:     part.FunctionResponse.Name,
				Response: response,
			},
		}
	default:
		return nil, nil
	}
	return out, nil
}

var reasoningEnginePattern = regexp.MustCompile(`^projects/(?:[a-zA-Z0-9-_]+)/locations/(?:[a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$`)

// getReasoningEngineID accepts the same forms as session/vertexai:
// a configured ID, a numeric app name, or a full reasoning engine resource
// name. Returning just the numeric ID keeps resource path formatting centralized.
func (c *vertexAIClient) getReasoningEngineID(appName string) (string, error) {
	if c.reasoningEngine != "" {
		return c.reasoningEngine, nil
	}
	if _, err := strconv.Atoi(appName); err == nil {
		return appName, nil
	}
	matches := reasoningEnginePattern.FindStringSubmatch(appName)
	if len(matches) < 2 {
		return "", fmt.Errorf("app name %q is not valid. It should be the full ReasoningEngine resource name or the reasoning engine numeric ID", appName)
	}
	return matches[1], nil
}

type memoryBankClientAdapter struct {
	client *aiplatform.MemoryBankClient
}

// Close closes the generated Google Cloud Memory Bank client.
func (c *memoryBankClientAdapter) Close() error {
	return c.client.Close()
}

// GenerateMemories adapts the generated client's variadic call-option method to
// the small interface used by this package's tests.
func (c *memoryBankClientAdapter) GenerateMemories(ctx context.Context, req *aiplatformpb.GenerateMemoriesRequest) (generateMemoriesOperation, error) {
	op, err := c.client.GenerateMemories(ctx, req)
	if err != nil {
		return nil, err
	}
	return &generateMemoriesOperationAdapter{op: op}, nil
}

// RetrieveMemories forwards retrieval requests to the generated Google Cloud
// client.
func (c *memoryBankClientAdapter) RetrieveMemories(ctx context.Context, req *aiplatformpb.RetrieveMemoriesRequest) (*aiplatformpb.RetrieveMemoriesResponse, error) {
	return c.client.RetrieveMemories(ctx, req)
}

type generateMemoriesOperationAdapter struct {
	op *aiplatform.GenerateMemoriesOperation
}

// Name returns the server-side long-running operation name.
func (o *generateMemoriesOperationAdapter) Name() string {
	return o.op.Name()
}

// Wait blocks until the GenerateMemories operation completes and returns its
// final response.
func (o *generateMemoriesOperationAdapter) Wait(ctx context.Context) (*aiplatformpb.GenerateMemoriesResponse, error) {
	return o.op.Wait(ctx)
}
