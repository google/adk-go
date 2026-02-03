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

package sessioninternal

import (
	"fmt"
	"iter"
	"time"

	"google.golang.org/adk/session"
)

// MutableSession implements session.Session
type MutableSession struct {
	service       session.Service
	storedSession session.Session
}

// NewMutableSession creates and returns session.Session implementation.
func NewMutableSession(service session.Service, storedSession session.Session) *MutableSession {
	return &MutableSession{
		service:       service,
		storedSession: storedSession,
	}
}

func (s *MutableSession) State() session.State {
	return s
}

func (s *MutableSession) AppName() string {
	return s.storedSession.AppName()
}

func (s *MutableSession) UserID() string {
	return s.storedSession.UserID()
}

func (s *MutableSession) ID() string {
	return s.storedSession.ID()
}

func (s *MutableSession) Events() session.Events {
	return s.storedSession.Events()
}

func (s *MutableSession) LastUpdateTime() time.Time {
	return s.storedSession.LastUpdateTime()
}

func (s *MutableSession) Get(key string) (any, error) {
	value, err := s.storedSession.State().Get(key)
	if err != nil {
		return nil, fmt.Errorf("failed to get key %q from state: %w", key, err)
	}
	return value, nil
}

func (s *MutableSession) All() iter.Seq2[string, any] {
	return s.storedSession.State().All()
}

func (s *MutableSession) Set(key string, value any) error {
	mutableState, ok := s.storedSession.State().(MutableState)
	if !ok {
		return fmt.Errorf("this session state is not mutable")
	}
	if err := mutableState.Set(key, value); err != nil {
		return fmt.Errorf("failed to set key %q in state: %w", key, err)
	}
	return nil
}
