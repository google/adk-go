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

package codeexecution

import (
	"iter"
	"time"

	"google.golang.org/adk/session"
)

type mockState struct {
	data map[string]any
}

func (m *mockState) Set(key string, val any) error {
	m.data[key] = val
	return nil
}

func (m *mockState) Get(key string) (any, error) {
	return m.data[key], nil
}

func (m *mockState) All() iter.Seq2[string, any] {
	return nil
}

var _ session.State = (*mockState)(nil)

type mockSession struct {
	state *mockState
}

func (m *mockSession) ID() string                { return "mock-session-id" }
func (m *mockSession) AppName() string           { return "mock-app" }
func (m *mockSession) UserID() string            { return "mock-user" }
func (m *mockSession) State() session.State      { return m.state }
func (m *mockSession) Events() session.Events    { return nil }
func (m *mockSession) LastUpdateTime() time.Time { return time.Now() }

var _ session.Session = (*mockSession)(nil)
