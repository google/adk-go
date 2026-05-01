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

package main

import (
	"context"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// stubState is a tiny in-memory session.State.
type stubState struct {
	m map[string]any
}

func newStubState(m map[string]any) *stubState {
	if m == nil {
		m = map[string]any{}
	}
	return &stubState{m: m}
}

func (s *stubState) Get(key string) (any, error) {
	if v, ok := s.m[key]; ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}
func (s *stubState) Set(key string, val any) error { s.m[key] = val; return nil }
func (s *stubState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.m {
			if !yield(k, v) {
				return
			}
		}
	}
}

type stubReadonlyContext struct {
	context.Context
	state *stubState
}

func newStubReadonlyContext(m map[string]any) *stubReadonlyContext {
	return &stubReadonlyContext{Context: context.Background(), state: newStubState(m)}
}

func (s *stubReadonlyContext) AgentName() string                    { return "researcher" }
func (s *stubReadonlyContext) AppName() string                      { return "demo" }
func (s *stubReadonlyContext) Branch() string                       { return "" }
func (s *stubReadonlyContext) InvocationID() string                 { return "inv-1" }
func (s *stubReadonlyContext) ReadonlyState() session.ReadonlyState { return s.state }
func (s *stubReadonlyContext) SessionID() string                    { return "sess-1" }
func (s *stubReadonlyContext) UserContent() *genai.Content          { return nil }
func (s *stubReadonlyContext) UserID() string                       { return "u" }

var _ agent.ReadonlyContext = (*stubReadonlyContext)(nil)
