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

package runner

import (
	"fmt"
	"iter"
	"time"

	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
)

// mutableSession implements session.Session
type mutableSession struct {
	service       sessionservice.Service
	storedSession sessionservice.StoredSession
}

func (s *mutableSession) State() session.State {
	return s
}

func (s *mutableSession) ID() session.ID {
	return s.storedSession.ID()
}

func (s *mutableSession) Events() session.Events {
	return s.storedSession.Events()
}

func (s *mutableSession) Updated() time.Time {
	return s.storedSession.Updated()
}

func (s *mutableSession) Get(key string) (any, error) {
	value, err := s.storedSession.State().Get(key)
	if err != nil {
		return nil, fmt.Errorf("failed to get key %q from state: %w", key, err)
	}
	return value, nil
}

func (s *mutableSession) All() iter.Seq2[string, any] {
	return s.storedSession.State().All()
}

func (s *mutableSession) Set(key string, value any) error {
	mutableState, ok := s.storedSession.State().(sessioninternal.MutableState)
	if !ok {
		return fmt.Errorf("this session state is not mutable")

	}
	if err := mutableState.Set(key, value); err != nil {
		return fmt.Errorf("failed to set key %q in state: %w", key, err)
	}
	return nil
}
