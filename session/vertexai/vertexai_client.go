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
	"maps"
	"regexp"
	"strconv"
	"strings"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
)

const (
	engineResourceTemplate  = "projects/%s/locations/%s/reasoningEngines/%s"
	sessionResourceTemplate = engineResourceTemplate + "/sessions/%s"
	connectionErrorTemplate = "could not establish connection to the aiplatform server: %s"
)

type vertexAiClient struct {
	location        string
	projectID       string
	reasoningEngine string
	rpcClient       *aiplatform.SessionClient
}

func newVertexAiClient(ctx context.Context, location string, projectID string, reasoningEngine string) (*vertexAiClient, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return nil, fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	return &vertexAiClient{location, projectID, reasoningEngine, rpcClient}, nil
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
		stateStruct, err := structpb.NewStruct(req.State)
		if err != nil {
			return nil, fmt.Errorf("failed to convert state to structpb: %w", err)
		}
		pbSession.SessionState = stateStruct
	}

	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}
	rpcReq := &aiplatformpb.CreateSessionRequest{
		Parent:  fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, reasoningEngine),
		Session: pbSession,
	}
	lro, err := c.rpcClient.CreateSession(ctx, rpcReq)
	if err != nil {
		return nil, err
	}

	initState := maps.Clone(req.State)

	return &localSession{
		appName:   req.AppName,
		userID:    req.UserID,
		sessionID: sessionIDByOperationName(lro.Name()),
		events:    make([]*session.Event, 0),
		state:     initState,
	}, nil
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
		return nil, err
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
		state:     sessRpcResp.SessionState.AsMap(),
	}, nil
}

func (c *vertexAiClient) listSessions(ctx context.Context, req *session.ListRequest) ([]session.Session, error) {
	sessions := make([]session.Session, 0)

	reasoningEngine, err := c.getReasoningEngineID(req.AppName)
	if err != nil {
		return nil, err
	}
	rpcReq := &aiplatformpb.ListSessionsRequest{
		Parent: fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, reasoningEngine),
	}
	it := c.rpcClient.ListSessions(ctx, rpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		// FIXME this should be a filter in the request
		if rpcResp.UserId != req.UserID {
			continue
		}
		session := &localSession{
			appName:   req.AppName,
			userID:    rpcResp.UserId,
			sessionID: sessionIdBySessionName(rpcResp.Name),
			state:     rpcResp.SessionState.AsMap(),
			updatedAt: rpcResp.UpdateTime.AsTime(),
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
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
		return err
	}
	return lro.Wait(ctx)
}

func (c *vertexAiClient) appendEvent(ctx context.Context, appName, sessionID string, event *session.Event) error {
	reasoningEngine, err := c.getReasoningEngineID(appName)
	if err != nil {
		return err
	}

	// TODO add logic for other types of parts
	parts := make([]*aiplatformpb.Part, 0)
	role := genai.RoleUser
	if event.Content != nil {
		role = event.Content.Role
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				aiplatformPart := aiplatformpb.Part{
					Data: &aiplatformpb.Part_Text{Text: part.Text},
				}
				parts = append(parts, &aiplatformPart)
			}
		}
	}

	_, err = c.rpcClient.AppendEvent(ctx, &aiplatformpb.AppendEventRequest{
		Name: sessionNameByID(sessionID, c, reasoningEngine),
		Event: &aiplatformpb.SessionEvent{
			Timestamp: &timestamppb.Timestamp{
				Seconds: event.Timestamp.Unix(),
				Nanos:   int32(event.Timestamp.Nanosecond()),
			},
			Author:       event.Author,
			InvocationId: event.InvocationID,
			Content: &aiplatformpb.Content{
				Parts: parts,
				Role:  role,
			},
		},
	})
	if err != nil {
		return err
	}

	return nil
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
		// TODO after parameter support
		// eventsRpcReq.Filter = fmt.Sprintf("timestamp>=%q", after.Format("2006-01-02T15:04:05-07:00"))
		return nil, fmt.Errorf("timestamp filter is not supported for VertexAISessionService")
	}
	it := c.rpcClient.ListEvents(ctx, eventsRpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		parts := make([]*genai.Part, 0)
		role := genai.RoleUser
		// TODO add logic for other types of parts
		if rpcResp.Content != nil {
			role = rpcResp.Content.Role
			for _, respPart := range rpcResp.Content.Parts {
				part := genai.NewPartFromText(respPart.GetText())
				parts = append(parts, part)
			}
		}
		event := &session.Event{
			ID:           sessionIdBySessionName(rpcResp.Name),
			Timestamp:    rpcResp.Timestamp.AsTime(),
			InvocationID: rpcResp.InvocationId,
			Author:       rpcResp.Author,
			Actions: session.EventActions{
				StateDelta: make(map[string]any),
			},
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: parts,
					Role:  role,
				},
			},
		}
		events = append(events, event)
	}
	if numRecentEvents != 0 {
		return events[len(events)-numRecentEvents:], nil
	}
	return events, nil
}

func sessionIdBySessionName(sn string) string {
	return sn[strings.LastIndex(sn, "/")+1:]
}

func sessionIDByOperationName(on string) string {
	return on[strings.LastIndex(on, "/sessions/")+10 : strings.LastIndex(on, "/operations/")]
}

func sessionNameByID(id string, c *vertexAiClient, reasoningEngine string) string {
	return fmt.Sprintf(sessionResourceTemplate, c.projectID, c.location, reasoningEngine, id)
}

func (c *vertexAiClient) getReasoningEngineID(appName string) (string, error) {
	if c.reasoningEngine != "" {
		return c.reasoningEngine, nil
	}

	// Check if appName consists only of digits
	_, err := strconv.Atoi(appName)
	if err == nil {
		return appName, nil
	}

	// Regex pattern to match the full resource name
	pattern := `^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$`
	re, err := regexp.Compile(pattern)
	if err != nil {
		// This should not happen with a valid constant pattern
		return "", fmt.Errorf("internal error compiling regex: %w", err)
	}

	matches := re.FindStringSubmatch(appName)

	if len(matches) == 0 {
		return "", fmt.Errorf("app name %s is not valid. It should either be the full ReasoningEngine resource name, or the reasoning engine id", appName)
	}

	// The last group is the ID
	return matches[len(matches)-1], nil
}
