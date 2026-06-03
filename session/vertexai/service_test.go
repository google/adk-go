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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"cloud.google.com/go/rpcreplay"
	"github.com/google/uuid"
	"google.golang.org/api/option"
	"google.golang.org/grpc"

	"google.golang.org/adk/session"
	"google.golang.org/adk/session/session_test"
)

const (
	ProjectID = "adk-go-test"
	Location  = "us-central1"
	EngineId  = "5576569044451983360"
	EngineId2 = "8602987994044956672"
	UserID    = "test-user"
)

func Test_vertexaiService(t *testing.T) {
	opts := session_test.SuiteOptions{
		SupportsUserProvidedSessionID: false,
		ProvidesServerAssignedEventID: true,
		AppName:                       EngineId,
	} // VertexAI forbids custom IDs
	session_test.RunServiceTests(t, opts, func(t *testing.T) session.Service {
		name := strings.ReplaceAll(t.Name(), "/", "_")
		s, _ := emptyService(t, name, false)
		return s
	})
}

func Test_vertexaiService_AppendEvent_StructuralValidation(t *testing.T) {
	tests := []struct {
		name    string
		session *localSession
		event   *session.Event
		wantErr bool
		offline bool
	}{
		{
			name:    "missing_session_id",
			session: &localSession{appName: EngineId, userID: UserID},
			event:   &session.Event{},
			wantErr: true,
			offline: true,
		},
		{
			name:    "nil_event",
			session: &localSession{appName: EngineId2, userID: "user2", sessionID: "session2"},
			event:   nil,
			wantErr: true,
			offline: true,
		},
		{
			name:    "missing_author",
			session: &localSession{appName: EngineId2, userID: "user2", sessionID: "session2"},
			event: &session.Event{
				Timestamp:    time.Now(),
				InvocationID: uuid.NewString(),
			},
			wantErr: true,
		},
		{
			name:    "missing_invocation_id",
			session: &localSession{appName: EngineId2, userID: "user2", sessionID: "session2"},
			event: &session.Event{
				Timestamp: time.Now(),
				Author:    UserID,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := emptyService(t, tt.name, tt.offline)
			ctx := t.Context()
			err := s.AppendEvent(ctx, tt.session, tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("AppendEvent() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func emptyService(t *testing.T, name string, offline bool) (session.Service, map[string]string) {
	t.Helper()
	replayFile := sanitizeFilename(name)

	var opts []option.ClientOption
	teardown := func() {}
	var err error

	if offline {
		opts = []option.ClientOption{option.WithoutAuthentication()}
	} else {
		var rawOpts []option.ClientOption
		var rawTeardown func()
		rawOpts, rawTeardown, err = setupReplay(t, replayFile)
		if err != nil {
			t.Fatalf("Failed to setup replay: %v", err)
		}
		opts = rawOpts
		teardown = rawTeardown
	}

	v, err := NewSessionService(t.Context(), VertexAIServiceConfig{
		Location:  Location,
		ProjectID: ProjectID,
	}, opts...)
	if err != nil {
		t.Fatalf("%s", err)
	}

	t.Cleanup(func() {
		t.Log("CLEANUP")
		if !offline {
			deleteAll(t, v)
		}
		defer teardown()
	})

	return v, make(map[string]string, 0)
}

func deleteAll(t *testing.T, v session.Service) {
	deleteAllFromApp(t, v, EngineId)
	deleteAllFromApp(t, v, EngineId2)
}

func deleteAllFromApp(t *testing.T, v session.Service, app string) {
	cleanupCtx := context.Background()
	sessionsResp, err := v.List(cleanupCtx, &session.ListRequest{
		AppName: app,
	})
	if err != nil {
		t.Errorf("error listing session for delete all: %s", err)
	}

	for _, s := range sessionsResp.Sessions {
		err := v.Delete(cleanupCtx, &session.DeleteRequest{
			AppName:   s.AppName(),
			UserID:    s.UserID(),
			SessionID: s.ID(),
		})
		if err != nil {
			t.Errorf("error deleting session for delete all: %s", err)
		}
	}
}

func setupReplay(t *testing.T, filename string) ([]option.ClientOption, func(), error) {
	filePath := filepath.Join("testdata", filename)
	var grpcOpts []grpc.DialOption
	var teardown func() error

	if os.Getenv("UPDATE_REPLAYS") == "true" {
		t.Logf("Recording payload to %s", filePath)
		_ = os.MkdirAll("testdata", 0o755)

		rec, err := rpcreplay.NewRecorder(filePath, nil)
		if err != nil {
			return nil, nil, err
		}
		grpcOpts = rec.DialOptions()
		teardown = rec.Close
	} else {
		rep, err := rpcreplay.NewReplayer(filePath)
		if err != nil {
			return nil, nil, err
		}
		grpcOpts = rep.DialOptions()
		teardown = rep.Close
	}

	var clientOpts []option.ClientOption
	for _, opt := range grpcOpts {
		clientOpts = append(clientOpts, option.WithGRPCDialOption(opt))
		if os.Getenv("UPDATE_REPLAYS") != "true" {
			clientOpts = append(clientOpts, option.WithoutAuthentication())
		}
	}

	return clientOpts, func() {
		if err := teardown(); err != nil {
			t.Errorf("Failed to close replayer/recorder: %v", err)
		}
	}, nil
}

func sanitizeFilename(name string) string {
	safe := strings.ReplaceAll(name, " ", "_")
	safe = strings.ReplaceAll(safe, ",", "_")
	safe = strings.ReplaceAll(safe, "/", "-")
	return safe + ".replay"
}
