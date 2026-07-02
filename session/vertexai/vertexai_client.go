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
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	vertexaiutil "google.golang.org/adk/v2/util/vertexai"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
)

type vertexAiClient struct {
	agentEngineData *vertexaiutil.AgentEngineData
	rpcClient       *aiplatform.SessionClient
}

func newVertexAiClient(ctx context.Context, location, projectID, reasoningEngine string, opts ...option.ClientOption) (*vertexAiClient, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("could not establish connection to the aiplatform server: %w", err)
	}
	return &vertexAiClient{
		agentEngineData: &vertexaiutil.AgentEngineData{
			Location:        location,
			ProjectID:       projectID,
			ReasoningEngine: reasoningEngine,
		},
		rpcClient: rpcClient,
	}, nil
}

// Ensure you close it when your application shuts down
func (c *vertexAiClient) Close() error {
	return c.rpcClient.Close()
}

func (c *vertexAiClient) createSession(ctx context.Context, req *session.CreateRequest) (*localSession, error) {
	pbSession := &aiplatformpb.Session{
		UserId: req.UserID,
	}
	// Convert and set the initial state if provided
	if len(req.State) > 0 {
		stateStruct, err := toStructPB(req.State)
		if err != nil {
			return nil, fmt.Errorf("failed to convert state to structpb: %w", err)
		}
		pbSession.SessionState = stateStruct
	}

	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}
	aeData := &vertexaiutil.AgentEngineData{
		Location:        c.agentEngineData.Location,
		ProjectID:       c.agentEngineData.ProjectID,
		ReasoningEngine: reasoningEngine,
	}
	rpcReq := &aiplatformpb.CreateSessionRequest{
		Parent:    vertexaiutil.AgentEngineResource(aeData),
		Session:   pbSession,
		SessionId: req.SessionID,
	}
	lro, err := c.rpcClient.CreateSession(ctx, rpcReq)
	if err != nil {
		return nil, fmt.Errorf("error creating session: %w", err)
	}

	sessionID, err := sessionIDByOperationName(lro.Name())
	if err != nil {
		return nil, fmt.Errorf("error creating session: %w", err)
	}
	createdSession, err := c.waitForOperation(ctx, req.AppName, req.UserID, sessionID)
	if err != nil {
		return nil, fmt.Errorf("LRO for CreateSession failed: %w", err)
	}
	return createdSession, nil
}

func isNotFoundError(err error) bool {
	// status.Code returns codes.Unknown if it's not a gRPC error,
	// otherwise it returns the specific gRPC code.
	return status.Code(err) == codes.NotFound
}

// TODO replace with LRO wait when it's fixed
// waitForOperation polls the LRO until it is done.
func (c *vertexAiClient) waitForOperation(ctx context.Context, appName, userId, sessionID string) (*localSession, error) {
	const (
		maxRetries = 10
		baseDelay  = time.Second
		maxDelay   = 5 * time.Second
	)

	for i := range maxRetries {
		// Get the latest status of the operation.
		ls, err := c.getSession(ctx, &session.GetRequest{AppName: appName, UserID: userId, SessionID: sessionID})
		if err != nil {
			// Basic retry on "not found" which might be due to propagation
			if i < maxRetries-1 && isNotFoundError(err) {
				delay := min(time.Duration(i*i)*baseDelay, maxDelay)
				time.Sleep(delay)
				continue
			}
			return nil, fmt.Errorf("error getting operation '%s': %w", sessionID, err)
		} else {
			return ls, nil
		}
	}
	return nil, fmt.Errorf("LRO '%s' timed out after %d retries", sessionID, maxRetries)
}

func (c *vertexAiClient) getSession(ctx context.Context, req *session.GetRequest) (*localSession, error) {
	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}
	sessRpcReq := &aiplatformpb.GetSessionRequest{
		Name: sessionNameByID(req.SessionID, c, reasoningEngine),
	}
	sessRpcResp, err := c.rpcClient.GetSession(ctx, sessRpcReq)
	if err != nil {
		return nil, fmt.Errorf("error fetching session: %w", err)
	}

	if sessRpcResp == nil {
		return nil, fmt.Errorf("session %+v not found", req.SessionID)
	}
	if sessRpcResp.UserId != req.UserID {
		return nil, fmt.Errorf("session %s does not belong to user %s", req.SessionID, req.UserID)
	}

	return &localSession{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: req.SessionID,
		updatedAt: sessRpcResp.UpdateTime.AsTime(),
		state:     filterNilValues(sessRpcResp.SessionState.AsMap()),
	}, nil
}

func (c *vertexAiClient) listSessions(ctx context.Context, req *session.ListRequest) ([]session.Session, error) {
	sessions := make([]session.Session, 0)

	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}

	aeData := vertexaiutil.AgentEngineData{
		Location:        c.agentEngineData.Location,
		ProjectID:       c.agentEngineData.ProjectID,
		ReasoningEngine: reasoningEngine,
	}

	rpcReq := &aiplatformpb.ListSessionsRequest{
		Parent: vertexaiutil.AgentEngineResource(&aeData),
	}
	if req.UserID != "" {
		rpcReq.Filter = fmt.Sprintf("userId=\"%s\"", req.UserID)
	}
	it := c.rpcClient.ListSessions(ctx, rpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error creating session list: %w", err)
		}
		id, err := sessionIdBySessionName(rpcResp.Name)
		if err != nil {
			return nil, fmt.Errorf("error creating session list: %w", err)
		}
		session := &localSession{
			appName:   req.AppName,
			userID:    rpcResp.UserId,
			sessionID: id,
			state:     filterNilValues(rpcResp.SessionState.AsMap()),
			updatedAt: rpcResp.UpdateTime.AsTime(),
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
}

func filterNilValues(originalMap map[string]any) map[string]any {
	if originalMap == nil {
		return nil
	}

	filteredMap := make(map[string]any)
	for key, value := range originalMap {
		if value != nil {
			filteredMap[key] = value
		}
	}
	return filteredMap
}

func (c *vertexAiClient) deleteSession(ctx context.Context, req *session.DeleteRequest) error {
	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return err
	}
	lro, err := c.rpcClient.DeleteSession(ctx, &aiplatformpb.DeleteSessionRequest{
		Name: sessionNameByID(req.SessionID, c, reasoningEngine),
	})
	if err != nil {
		return fmt.Errorf("error deleting session: %w", err)
	}
	return lro.Wait(ctx)
}

func (c *vertexAiClient) appendEvent(ctx context.Context, appName, sessionID string, event *session.Event) error {
	// ignore partial events
	if event.Partial {
		return nil
	}

	reasoningEngine, err := c.getReasoningEngineID(appName)
	if err != nil {
		return err
	}

	eventActions, err := createAiplatformpbEventActions(event)
	if err != nil {
		return fmt.Errorf("failed to convert event actions: %w", err)
	}

	content, err := createAiplatformpbContent(event)
	if err != nil {
		return fmt.Errorf("error creating content: %w", err)
	}

	metadata, err := createAiplatformpbMetadata(event)
	if err != nil {
		return fmt.Errorf("error creating metadata: %w", err)
	}

	// The legacy column-backed fields are still written below as a fallback
	// for readers that ignore raw_event.
	var rawEvent *structpb.Struct
	if eventNeedsRawEvent(event) {
		rawEvent, err = eventToRawEvent(event)
		if err != nil {
			return fmt.Errorf("error creating raw event: %w", err)
		}
	}

	_, err = c.rpcClient.AppendEvent(ctx, &aiplatformpb.AppendEventRequest{
		Name: sessionNameByID(sessionID, c, reasoningEngine),
		Event: &aiplatformpb.SessionEvent{
			Timestamp: &timestamppb.Timestamp{
				Seconds: event.Timestamp.Unix(),
				Nanos:   int32(event.Timestamp.Nanosecond()),
			},
			Author:        event.Author,
			InvocationId:  event.InvocationID,
			Content:       content,
			Actions:       eventActions,
			EventMetadata: metadata,
			ErrorCode:     event.ErrorCode,
			ErrorMessage:  event.ErrorMessage,
			RawEvent:      rawEvent,
		},
	})
	if err != nil {
		return fmt.Errorf("error appending event: %w", err)
	}

	return nil
}

// eventNeedsRawEvent reports whether the event carries state that has no
// dedicated SessionEvent column and would be lost without raw_event.
// Gating raw_event on this keeps plain events on their legacy wire format,
// so the recorded replay fixtures stay valid.
func eventNeedsRawEvent(event *session.Event) bool {
	return event.Output != nil ||
		event.NodeInfo != nil ||
		event.IsolationScope != "" ||
		event.RequestedInput != nil ||
		len(event.Routes) > 0
}

// eventToRawEvent serializes a session.Event into a structpb.Struct for
// the SessionEvent.raw_event field. Uses Go's JSON encoding; not yet
// byte-compatible with adk-python's camelCase dump (cross-runtime parity
// is tracked separately).
//
// Integers in the any-typed Output and StateDelta come back as float64
// (structpb numbers and json.Unmarshal into any are both float64). This
// matches the SQL backend, so the lossiness is framework-wide; store
// values needing exact integer fidelity as strings.
func eventToRawEvent(event *session.Event) (*structpb.Struct, error) {
	b, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("marshaling event: %w", err)
	}
	s := &structpb.Struct{}
	if err := s.UnmarshalJSON(b); err != nil {
		return nil, fmt.Errorf("converting event to structpb: %w", err)
	}
	return s, nil
}

// eventFromRawEvent reconstructs a session.Event from a raw_event struct
// written by eventToRawEvent. Identity fields (ID, Timestamp,
// InvocationID, Author) are authoritative on the SessionEvent envelope,
// so callers overwrite them after this returns.
func eventFromRawEvent(raw *structpb.Struct) (*session.Event, error) {
	b, err := json.Marshal(raw.AsMap())
	if err != nil {
		return nil, fmt.Errorf("marshaling raw event map: %w", err)
	}
	event := &session.Event{}
	if err := json.Unmarshal(b, event); err != nil {
		return nil, fmt.Errorf("unmarshaling raw event: %w", err)
	}
	return event, nil
}

func (c *vertexAiClient) listSessionEvents(ctx context.Context, appName, sessionID string, after time.Time, numRecentEvents int) ([]*session.Event, error) {
	reasoningEngine, err := c.getReasoningEngineID(appName)
	if err != nil {
		return nil, err
	}
	events := make([]*session.Event, 0)
	eventsRpcReq := &aiplatformpb.ListEventsRequest{
		Parent: sessionNameByID(sessionID, c, reasoningEngine),
	}
	if !after.IsZero() {
		eventsRpcReq.Filter = fmt.Sprintf("timestamp>=%q", after.Format("2006-01-02T15:04:05-07:00"))
	}
	it := c.rpcClient.ListEvents(ctx, eventsRpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error fetching session events: %w", err)
		}

		id, err := sessionIdBySessionName(rpcResp.Name)
		if err != nil {
			return nil, fmt.Errorf("error fetching session events: %w", err)
		}

		// Prefer raw_event; fall back to legacy field reconstruction for
		// events written before raw_event support.
		if rpcResp.RawEvent != nil {
			event, err := eventFromRawEvent(rpcResp.RawEvent)
			if err != nil {
				return nil, fmt.Errorf("error fetching session events: %w", err)
			}
			// Identity fields are authoritative on the envelope.
			event.ID = id
			event.Timestamp = rpcResp.Timestamp.AsTime()
			event.InvocationID = rpcResp.InvocationId
			event.Author = rpcResp.Author
			events = append(events, event)
			continue
		}

		content := aiplatformToGenaiContent(rpcResp)
		event := &session.Event{
			ID:           id,
			Timestamp:    rpcResp.Timestamp.AsTime(),
			InvocationID: rpcResp.InvocationId,
			Author:       rpcResp.Author,
			Actions:      aiplatformToSessionEventActions(rpcResp.Actions),
			LLMResponse: model.LLMResponse{
				Content:      content,
				ErrorCode:    rpcResp.ErrorCode,
				ErrorMessage: rpcResp.ErrorMessage,
			},
		}
		if rpcResp.EventMetadata != nil {
			event.Branch = rpcResp.EventMetadata.Branch
			event.TurnComplete = rpcResp.EventMetadata.TurnComplete
			event.Partial = rpcResp.EventMetadata.Partial
			event.Interrupted = rpcResp.EventMetadata.Interrupted
			event.LongRunningToolIDs = rpcResp.EventMetadata.LongRunningToolIds
			event.GroundingMetadata = createGroundingMetadata(rpcResp.EventMetadata.GroundingMetadata)
			if rpcResp.EventMetadata.CustomMetadata != nil {
				event.CustomMetadata = rpcResp.EventMetadata.CustomMetadata.AsMap()
			}
		}
		events = append(events, event)
	}
	if numRecentEvents > 0 {
		if numRecentEvents > len(events) {
			return events, nil
		}
		return events[len(events)-numRecentEvents:], nil
	}
	return events, nil
}

func createAiplatformpbEventActions(event *session.Event) (*aiplatformpb.EventActions, error) {
	if len(event.Actions.StateDelta) == 0 && len(event.Actions.ArtifactDelta) == 0 {
		return nil, nil
	}

	actions := &aiplatformpb.EventActions{}
	if len(event.Actions.StateDelta) > 0 {
		sessionState, err := toStructPB(event.Actions.StateDelta)
		if err != nil {
			return nil, fmt.Errorf("failed to convert state to structpb: %w", err)
		}
		actions.StateDelta = sessionState
	}
	if len(event.Actions.ArtifactDelta) > 0 {
		actions.ArtifactDelta = make(map[string]int32, len(event.Actions.ArtifactDelta))
		for name, version := range event.Actions.ArtifactDelta {
			actions.ArtifactDelta[name] = int32(version)
		}
	}
	return actions, nil
}

func aiplatformToSessionEventActions(actions *aiplatformpb.EventActions) session.EventActions {
	if actions == nil {
		return session.EventActions{}
	}

	eventActions := session.EventActions{}
	if actions.StateDelta != nil {
		eventActions.StateDelta = filterNilValues(actions.StateDelta.AsMap())
	}
	if len(actions.ArtifactDelta) > 0 {
		eventActions.ArtifactDelta = make(map[string]int64, len(actions.ArtifactDelta))
		for name, version := range actions.ArtifactDelta {
			eventActions.ArtifactDelta[name] = int64(version)
		}
	}
	return eventActions
}

func sessionIdBySessionName(sn string) (string, error) {
	idx := strings.LastIndex(sn, "/")
	if idx == -1 {
		return "", fmt.Errorf("invalid session name format %q: missing separator '/'", sn)
	}

	id := sn[idx+1:]
	if id == "" {
		return "", fmt.Errorf("invalid session name %q: empty session ID", sn)
	}

	return id, nil
}

func sessionIDByOperationName(on string) (string, error) {
	const sessionPrefix = "/sessions/"
	const opsSuffix = "/operations/"

	idxSession := strings.LastIndex(on, sessionPrefix)
	if idxSession == -1 {
		return "", fmt.Errorf("invalid operation name %q: missing %q", on, sessionPrefix)
	}

	// Calculate where the ID actually begins
	idStart := idxSession + len(sessionPrefix)

	idxOps := strings.LastIndex(on, opsSuffix)
	if idxOps == -1 {
		return "", fmt.Errorf("invalid operation name %q: missing %q", on, opsSuffix)
	}

	// ensure the start comes before the end
	// If idStart > idxOps, it means "/operations/" appeared before "/sessions/"
	// or they overlap in a weird way, which would cause a panic on slicing.
	if idStart > idxOps {
		return "", fmt.Errorf("invalid operation name %q: structure malformed or segments out of order", on)
	}

	id := on[idStart:idxOps]
	if id == "" {
		return "", fmt.Errorf("invalid operation name %q: empty session ID", on)
	}

	return id, nil
}

func sessionNameByID(id string, c *vertexAiClient, reasoningEngine string) string {
	aeData := &vertexaiutil.AgentEngineData{
		Location:        c.agentEngineData.Location,
		ProjectID:       c.agentEngineData.ProjectID,
		ReasoningEngine: reasoningEngine,
	}
	return vertexaiutil.SessionResource(aeData, id)
}

// (?:...) tells Go "match this, but don't save it in the results array".
// We keep the (\d+) at the end as a capturing group.
var reasoningEnginePattern = regexp.MustCompile(`^projects/(?:[a-zA-Z0-9-_]+)/locations/(?:[a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$`)

func (c *vertexAiClient) getReasoningEngineID(appName string) (string, error) {
	if c.agentEngineData.ReasoningEngine != "" {
		return c.agentEngineData.ReasoningEngine, nil
	}

	// Check if appName consists only of digits
	if _, err := strconv.Atoi(appName); err == nil {
		return appName, nil
	}

	// Execute the Regex
	matches := reasoningEnginePattern.FindStringSubmatch(appName)

	// With non-capturing groups, 'matches' will strictly have 2 elements if successful:
	// matches[0]: The full string (e.g., "projects/my-p/locations/...")
	// matches[1]: The first capturing group (the ID)
	if len(matches) < 2 {
		return "", fmt.Errorf("app name %q is not valid. It should be the full ReasoningEngine resource name or the reasoning engine numeric ID", appName)
	}

	return matches[1], nil
}

func aiplatformToGenaiContent(rpcResp *aiplatformpb.SessionEvent) *genai.Content {
	// TODO add logic for other types of parts
	var content *genai.Content
	if rpcResp.Content != nil {
		var parts []*genai.Part
		role := rpcResp.Content.Role
		for _, respPart := range rpcResp.Content.Parts {
			part := &genai.Part{}
			part.Thought = respPart.Thought
			part.ThoughtSignature = respPart.ThoughtSignature
			switch v := respPart.Data.(type) {
			case *aiplatformpb.Part_Text:
				part.Text = v.Text
			case *aiplatformpb.Part_InlineData:
				part.InlineData = &genai.Blob{
					MIMEType: v.InlineData.MimeType,
					Data:     v.InlineData.Data,
				}
			case *aiplatformpb.Part_FunctionCall:
				argsMap := v.FunctionCall.Args.AsMap() // Converts *structpb.Struct -> map[string]any
				part.FunctionCall = &genai.FunctionCall{
					ID:   v.FunctionCall.Id,
					Name: v.FunctionCall.Name,
					Args: argsMap,
				}
			case *aiplatformpb.Part_FunctionResponse:
				responseMap := v.FunctionResponse.Response.AsMap() // Converts *structpb.Struct -> map[string]any
				part.FunctionResponse = &genai.FunctionResponse{
					ID:       v.FunctionResponse.Id,
					Name:     v.FunctionResponse.Name,
					Response: responseMap,
				}
			}
			parts = append(parts, part)
		}
		content = &genai.Content{
			Parts: parts,
			Role:  role,
		}
	}
	return content
}

func createAiplatformpbContent(event *session.Event) (*aiplatformpb.Content, error) {
	// TODO add logic for other types of parts
	var content *aiplatformpb.Content
	if event.Content != nil {
		parts := make([]*aiplatformpb.Part, 0)
		for _, part := range event.Content.Parts {
			aiplatformPart := &aiplatformpb.Part{}
			aiplatformPart.Thought = part.Thought
			aiplatformPart.ThoughtSignature = part.ThoughtSignature
			if part.Text != "" {
				aiplatformPart.Data = &aiplatformpb.Part_Text{Text: part.Text}
			}
			if part.InlineData != nil {
				aiplatformPart.Data = &aiplatformpb.Part_InlineData{
					InlineData: &aiplatformpb.Blob{
						Data:     part.InlineData.Data,
						MimeType: part.InlineData.MIMEType,
					},
				}
			}
			if part.FunctionCall != nil {
				args, err := toStructPB(part.FunctionCall.Args)
				if err != nil {
					return nil, fmt.Errorf("failed to convert function call to structpb: %w", err)
				}
				aiplatformPart.Data = &aiplatformpb.Part_FunctionCall{
					FunctionCall: &aiplatformpb.FunctionCall{
						Id:   part.FunctionCall.ID,
						Name: part.FunctionCall.Name,
						Args: args,
					},
				}
			}
			if part.FunctionResponse != nil {
				response, err := toStructPB(part.FunctionResponse.Response)
				if err != nil {
					return nil, fmt.Errorf("failed to convert function response to structpb: %w", err)
				}
				aiplatformPart.Data = &aiplatformpb.Part_FunctionResponse{
					FunctionResponse: &aiplatformpb.FunctionResponse{
						Id:       part.FunctionResponse.ID,
						Name:     part.FunctionResponse.Name,
						Response: response,
					},
				}
			}
			parts = append(parts, aiplatformPart)
		}
		content = &aiplatformpb.Content{
			Parts: parts,
			Role:  event.Content.Role,
		}
	}
	return content, nil
}

func createAiplatformpbMetadata(event *session.Event) (*aiplatformpb.EventMetadata, error) {
	if event == nil {
		return nil, nil
	}
	metadata := &aiplatformpb.EventMetadata{
		Partial:            event.Partial,
		TurnComplete:       event.TurnComplete,
		Interrupted:        event.Interrupted,
		LongRunningToolIds: event.LongRunningToolIDs,
		Branch:             event.Branch,
	}
	if event.CustomMetadata != nil {
		customMetadata, err := toStructPB(event.CustomMetadata)
		if err != nil {
			return nil, fmt.Errorf("failed to convert event customMetadata to structpb: %w", err)
		}
		metadata.CustomMetadata = customMetadata
	}
	if event.GroundingMetadata != nil {
		metadata.GroundingMetadata = &aiplatformpb.GroundingMetadata{
			WebSearchQueries:             event.GroundingMetadata.WebSearchQueries,
			RetrievalQueries:             event.GroundingMetadata.RetrievalQueries,
			GoogleMapsWidgetContextToken: &event.GroundingMetadata.GoogleMapsWidgetContextToken,
		}
		if event.GroundingMetadata.SearchEntryPoint != nil {
			metadata.GroundingMetadata.SearchEntryPoint = &aiplatformpb.SearchEntryPoint{
				RenderedContent: event.GroundingMetadata.SearchEntryPoint.RenderedContent,
				SdkBlob:         event.GroundingMetadata.SearchEntryPoint.SDKBlob,
			}
		}
		if event.GroundingMetadata.RetrievalMetadata != nil {
			metadata.GroundingMetadata.RetrievalMetadata = &aiplatformpb.RetrievalMetadata{
				GoogleSearchDynamicRetrievalScore: event.GroundingMetadata.RetrievalMetadata.GoogleSearchDynamicRetrievalScore,
			}
		}
		var groundingChunks []*aiplatformpb.GroundingChunk
		for _, gc := range event.GroundingMetadata.GroundingChunks {
			if gc.Maps != nil {
				maps := &aiplatformpb.GroundingChunk_Maps{
					Uri:     &gc.Maps.URI,
					Title:   &gc.Maps.Title,
					Text:    &gc.Maps.Text,
					PlaceId: &gc.Maps.PlaceID,
				}
				if gc.Maps.PlaceAnswerSources != nil {
					var reviewSnippets []*aiplatformpb.GroundingChunk_Maps_PlaceAnswerSources_ReviewSnippet
					for _, source := range gc.Maps.PlaceAnswerSources.ReviewSnippets {
						snippet := &aiplatformpb.GroundingChunk_Maps_PlaceAnswerSources_ReviewSnippet{
							ReviewId:      source.Review,
							GoogleMapsUri: source.GoogleMapsURI,
							Title:         source.Title,
						}
						reviewSnippets = append(reviewSnippets, snippet)
					}
					maps.PlaceAnswerSources = &aiplatformpb.GroundingChunk_Maps_PlaceAnswerSources{
						ReviewSnippets: reviewSnippets,
					}
				}
				aiplGc := &aiplatformpb.GroundingChunk{
					ChunkType: &aiplatformpb.GroundingChunk_Maps_{
						Maps: maps,
					},
				}
				groundingChunks = append(groundingChunks, aiplGc)
			}
			if gc.RetrievedContext != nil {
				retrievedContext := &aiplatformpb.GroundingChunk_RetrievedContext{
					Uri:          &gc.RetrievedContext.URI,
					Title:        &gc.RetrievedContext.Title,
					Text:         &gc.RetrievedContext.Text,
					DocumentName: &gc.RetrievedContext.DocumentName,
				}
				if gc.RetrievedContext.RAGChunk != nil && gc.RetrievedContext.RAGChunk.PageSpan != nil {
					retrievedContext.ContextDetails = &aiplatformpb.GroundingChunk_RetrievedContext_RagChunk{
						RagChunk: &aiplatformpb.RagChunk{
							Text: gc.RetrievedContext.RAGChunk.Text,
							PageSpan: &aiplatformpb.RagChunk_PageSpan{
								FirstPage: gc.RetrievedContext.RAGChunk.PageSpan.FirstPage,
								LastPage:  gc.RetrievedContext.RAGChunk.PageSpan.LastPage,
							},
						},
					}
				}
				aiplGc := &aiplatformpb.GroundingChunk{
					ChunkType: &aiplatformpb.GroundingChunk_RetrievedContext_{
						RetrievedContext: retrievedContext,
					},
				}
				groundingChunks = append(groundingChunks, aiplGc)
			}
			if gc.Web != nil {
				web := &aiplatformpb.GroundingChunk_Web{
					Uri:   &gc.Web.URI,
					Title: &gc.Web.Title,
				}
				aiplGc := &aiplatformpb.GroundingChunk{
					ChunkType: &aiplatformpb.GroundingChunk_Web_{
						Web: web,
					},
				}
				groundingChunks = append(groundingChunks, aiplGc)
			}
		}
		metadata.GroundingMetadata.GroundingChunks = groundingChunks

		var groundingSupports []*aiplatformpb.GroundingSupport
		for _, gs := range event.GroundingMetadata.GroundingSupports {
			aiplGs := &aiplatformpb.GroundingSupport{
				GroundingChunkIndices: gs.GroundingChunkIndices,
				ConfidenceScores:      gs.ConfidenceScores,
			}
			if gs.Segment != nil {
				aiplGs.Segment = &aiplatformpb.Segment{
					PartIndex:  gs.Segment.PartIndex,
					StartIndex: gs.Segment.StartIndex,
					EndIndex:   gs.Segment.EndIndex,
					Text:       gs.Segment.Text,
				}
			}
			groundingSupports = append(groundingSupports, aiplGs)
		}
		metadata.GroundingMetadata.GroundingSupports = groundingSupports
	}
	return metadata, nil
}

func createGroundingMetadata(metadata *aiplatformpb.GroundingMetadata) *genai.GroundingMetadata {
	if metadata == nil {
		return nil
	}

	out := &genai.GroundingMetadata{
		WebSearchQueries: metadata.WebSearchQueries,
		RetrievalQueries: metadata.RetrievalQueries,
	}

	// Handle string pointer for Context Token
	out.GoogleMapsWidgetContextToken = derefString(metadata.GoogleMapsWidgetContextToken)

	// Search Entry Point
	if metadata.SearchEntryPoint != nil {
		out.SearchEntryPoint = &genai.SearchEntryPoint{
			RenderedContent: metadata.SearchEntryPoint.RenderedContent,
			SDKBlob:         metadata.SearchEntryPoint.SdkBlob,
		}
	}

	// Retrieval Metadata
	if metadata.RetrievalMetadata != nil {
		out.RetrievalMetadata = &genai.RetrievalMetadata{
			GoogleSearchDynamicRetrievalScore: metadata.RetrievalMetadata.GoogleSearchDynamicRetrievalScore,
		}
	}

	// Grounding Chunks
	if len(metadata.GroundingChunks) > 0 {
		var chunks []*genai.GroundingChunk
		for _, chunk := range metadata.GroundingChunks {
			newChunk := &genai.GroundingChunk{}

			// Handle 'Maps' Chunk
			if maps := chunk.GetMaps(); maps != nil {
				newMaps := &genai.GroundingChunkMaps{
					URI:     derefString(maps.Uri),
					Title:   derefString(maps.Title),
					Text:    derefString(maps.Text),
					PlaceID: derefString(maps.PlaceId),
				}

				if maps.PlaceAnswerSources != nil {
					newMaps.PlaceAnswerSources = &genai.GroundingChunkMapsPlaceAnswerSources{}
					for _, snippet := range maps.PlaceAnswerSources.ReviewSnippets {
						newSnippet := &genai.GroundingChunkMapsPlaceAnswerSourcesReviewSnippet{
							Review:        snippet.ReviewId,
							GoogleMapsURI: snippet.GoogleMapsUri,
						}
						newMaps.PlaceAnswerSources.ReviewSnippets = append(newMaps.PlaceAnswerSources.ReviewSnippets, newSnippet)
					}
				}
				newChunk.Maps = newMaps
			}

			// Handle 'RetrievedContext' Chunk
			if rc := chunk.GetRetrievedContext(); rc != nil {
				newRC := &genai.GroundingChunkRetrievedContext{
					URI:          derefString(rc.Uri),
					Title:        derefString(rc.Title),
					Text:         derefString(rc.Text),
					DocumentName: derefString(rc.DocumentName),
				}

				// Handle RAG Chunk (oneof in pb, usually a nested struct in genai)
				if rag := rc.GetRagChunk(); rag != nil {
					newRC.RAGChunk = &genai.RAGChunk{
						Text: rag.Text,
					}
					if rag.PageSpan != nil {
						newRC.RAGChunk.PageSpan = &genai.RAGChunkPageSpan{
							FirstPage: rag.PageSpan.FirstPage,
							LastPage:  rag.PageSpan.LastPage,
						}
					}
				}
				newChunk.RetrievedContext = newRC
			}

			// Handle 'Web' Chunk
			if web := chunk.GetWeb(); web != nil {
				newChunk.Web = &genai.GroundingChunkWeb{
					URI:   derefString(web.Uri),
					Title: derefString(web.Title),
				}
			}

			chunks = append(chunks, newChunk)
		}
		out.GroundingChunks = chunks
	}

	// Grounding Supports
	if len(metadata.GroundingSupports) > 0 {
		var supports []*genai.GroundingSupport
		for _, gs := range metadata.GroundingSupports {
			newSupport := &genai.GroundingSupport{
				GroundingChunkIndices: gs.GroundingChunkIndices,
				ConfidenceScores:      gs.ConfidenceScores,
			}

			if gs.Segment != nil {
				newSupport.Segment = &genai.Segment{
					PartIndex:  gs.Segment.PartIndex,
					StartIndex: gs.Segment.StartIndex,
					EndIndex:   gs.Segment.EndIndex,
					Text:       gs.Segment.Text,
				}
			}
			supports = append(supports, newSupport)
		}
		out.GroundingSupports = supports
	}

	return out
}

// toStructPB converts an arbitrary Go value into a protobuf Struct.
// It uses JSON marshaling as an intermediary step to safely serialize
// the input data before constructing the *structpb.Struct.
// Returns an error if any part of the JSON round-trip or conversion fails.
func toStructPB(value any) (*structpb.Struct, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal value: %w", err)
	}
	res := &structpb.Struct{}
	if err := res.UnmarshalJSON(data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON data to structpb: %w", err)
	}
	return res, nil
}

// derefString is a helper to safely dereference string pointers
// Returns empty string if pointer is nil
func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
