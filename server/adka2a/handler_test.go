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

package adka2a

import (
	"net/http"
	"testing"

	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func TestNewHandler(t *testing.T) {
	agent, err := newEventReplayAgent(nil, nil)
	if err != nil {
		t.Fatalf("newEventReplayAgent() error = %v", err)
	}

	sessionService := session.InMemoryService()
	config := HandlerConfig{
		ExecutorConfig: ExecutorConfig{
			RunnerConfig: runner.Config{
				AppName:        agent.Name(),
				Agent:          agent,
				SessionService: sessionService,
			},
		},
	}

	handler := NewHandler(config)
	if handler == nil {
		t.Fatal("NewHandler() returned nil")
	}

	if _, ok := handler.(http.Handler); !ok {
		t.Fatal("NewHandler() did not return an http.Handler")
	}
}
