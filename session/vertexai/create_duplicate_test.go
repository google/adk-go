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

package vertexai

import (
	"errors"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/session"
)

// A Create of a taken session ID fails with Agent Engine's own error, and
// that error does not wrap session.ErrAlreadyExists, as adk-python's Vertex
// AI service raises no AlreadyExistsError. It replays the recording of the
// shared suite's duplicate case, whose second CreateSession Agent Engine
// answered with INVALID_ARGUMENT.
func TestCreate_duplicateReturnsAPIErrorWithoutSentinel(t *testing.T) {
	s, _ := emptyService(t, "Test_vertexaiService_Create_when_already_exists,_it_fails", false)
	req := &session.CreateRequest{AppName: EngineID, UserID: "testUserID"}
	got, err := s.Create(t.Context(), req)
	if err != nil {
		t.Fatalf("first Create() error = %v", err)
	}

	_, err = s.Create(t.Context(), &session.CreateRequest{AppName: EngineID, UserID: "testUserID", SessionID: got.Session.ID()})

	if code := status.Code(err); code != codes.InvalidArgument {
		t.Errorf("Create(duplicate) error = %v, want one wrapping an INVALID_ARGUMENT status", err)
	}
	if errors.Is(err, session.ErrAlreadyExists) {
		t.Errorf("Create(duplicate) error = %v, want an error that does not wrap session.ErrAlreadyExists", err)
	}
}
