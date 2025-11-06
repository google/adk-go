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

package session

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/api/iterator"
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
}

func newVertexAiClient(location string, projectID string, reasoningEngine string) (*vertexAiClient, error) {
	return &vertexAiClient{location, projectID, reasoningEngine}, nil
}

func (c *vertexAiClient) createSession(ctx context.Context, req *CreateRequest) (*session, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return nil, fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	pbSession := &aiplatformpb.Session{
		UserId: req.UserID,
	}
	rpcReq := &aiplatformpb.CreateSessionRequest{
		Parent:  fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, c.reasoningEngine),
		Session: pbSession,
	}
	lro, err := rpcClient.CreateSession(ctx, rpcReq)
	if err != nil {
		return nil, err
	}

	return &session{
		id: id{
			appName:   req.AppName,
			userID:    req.UserID,
			sessionID: sessionIDByOperationName(lro.Name()),
		},
		events: make([]*Event, 0),
		state:  make(map[string]any),
	}, nil
}

func (c *vertexAiClient) getSession(ctx context.Context, req *GetRequest) (*session, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return nil, fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	sessRpcReq := &aiplatformpb.GetSessionRequest{
		Name: sessionNameByID(req.SessionID, c),
	}
	sessRpcResp, err := rpcClient.GetSession(ctx, sessRpcReq)
	if err != nil {
		return nil, err
	}

	return &session{
		id: id{
			appName:   req.AppName,
			userID:    req.UserID,
			sessionID: req.SessionID,
		},
		updatedAt: sessRpcResp.UpdateTime.AsTime(),
	}, nil
}

func (c *vertexAiClient) listSessions(ctx context.Context, req *ListRequest) ([]Session, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return nil, fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	sessions := make([]Session, 0)
	rpcReq := &aiplatformpb.ListSessionsRequest{
		Parent: fmt.Sprintf(engineResourceTemplate, c.projectID, c.location, c.reasoningEngine),
	}
	it := rpcClient.ListSessions(ctx, rpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		session := &session{
			id: id{
				appName:   req.AppName,
				userID:    req.UserID,
				sessionID: sessionIdBySessionName(rpcResp.Name),
			},
		}
		sessions = append(sessions, session)
	}
	return sessions, nil
}

func (c *vertexAiClient) deleteSession(ctx context.Context, req *DeleteRequest) error {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	lro, err := rpcClient.DeleteSession(ctx, &aiplatformpb.DeleteSessionRequest{
		Name: sessionNameByID(req.SessionID, c),
	})
	if err != nil {
		return err
	}
	return lro.Wait(ctx)
}

func (c *vertexAiClient) appendEvent(ctx context.Context, sessionID string, event *Event) error {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	_, err = rpcClient.AppendEvent(ctx, &aiplatformpb.AppendEventRequest{
		Name: sessionNameByID(sessionID, c),
		Event: &aiplatformpb.SessionEvent{
			Timestamp: &timestamppb.Timestamp{
				Seconds: event.Timestamp.Unix(),
				Nanos:   int32(event.Timestamp.Nanosecond()),
			},
			Author:       event.Author,
			InvocationId: event.InvocationID,
		},
	})
	if err != nil {
		return err
	}

	return nil
}

func (c *vertexAiClient) listSessionEvents(ctx context.Context, sessionID string) ([]*Event, error) {
	rpcClient, err := aiplatform.NewSessionClient(ctx)
	if err != nil {
		return nil, fmt.Errorf(connectionErrorTemplate, err.Error())
	}
	defer func() { _ = rpcClient.Close() }()

	events := make([]*Event, 0)
	eventsRpcReq := &aiplatformpb.ListEventsRequest{
		Parent: sessionNameByID(sessionID, c),
	}
	it := rpcClient.ListEvents(ctx, eventsRpcReq)
	for {
		rpcResp, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		event := &Event{
			ID:           sessionIdBySessionName(rpcResp.Name),
			Timestamp:    rpcResp.Timestamp.AsTime(),
			InvocationID: rpcResp.InvocationId,
			Author:       rpcResp.Author,
			Actions: EventActions{
				StateDelta: make(map[string]any),
			},
		}
		events = append(events, event)
	}
	return events, nil
}

func sessionIdBySessionName(sn string) string {
	return sn[strings.LastIndex(sn, "/")+1:]
}

func sessionIDByOperationName(on string) string {
	return on[strings.LastIndex(on, "/sessions/")+10 : strings.LastIndex(on, "/operations/")]
}

func sessionNameByID(id string, c *vertexAiClient) string {
	return fmt.Sprintf(sessionResourceTemplate, c.projectID, c.location, c.reasoningEngine, id)
}
