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

package session_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/session"
)

// serviceWithoutUserState implements session.Service and not
// session.UserStateReader, like a third-party service written before the
// read existed.
type serviceWithoutUserState struct {
	session.Service
}

func TestGetUserState_UnsupportedServices(t *testing.T) {
	resp, err := session.GetUserState(t.Context(), serviceWithoutUserState{}, &session.GetUserStateRequest{AppName: "app", UserID: "u1"})

	if !errors.Is(err, errors.ErrUnsupported) {
		t.Errorf("GetUserState() error = %v, want an error wrapping errors.ErrUnsupported", err)
	}
	if resp != nil {
		t.Errorf("GetUserState() response = %+v, want nil", resp)
	}
}

func TestGetUserState_NilArgumentsAreNotUnsupported(t *testing.T) {
	tests := []struct {
		name string
		svc  session.Service
		req  *session.GetUserStateRequest
	}{
		{name: "nil service", svc: nil, req: &session.GetUserStateRequest{AppName: "app", UserID: "u1"}},
		// The service does not implement session.UserStateReader, so a nil
		// request that got past validation would be reported as unsupported.
		{name: "nil request", svc: serviceWithoutUserState{}, req: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := session.GetUserState(t.Context(), tt.svc, tt.req)

			if err == nil || errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("GetUserState() error = %v, want a non-nil error not wrapping errors.ErrUnsupported", err)
			}
			if resp != nil {
				t.Errorf("GetUserState() response = %+v, want nil", resp)
			}
		})
	}
}
