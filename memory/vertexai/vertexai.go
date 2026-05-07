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

// package vertexai provides support for using MemoryBank provided by VertexAI
package vertexai

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	vertexaiutil "google.golang.org/adk/util/vertexai"
)

type vertexAIService struct {
	client                        *vertexAIClient
	stateKeySessionLastUpdateTime string
}

// ServiceConfig allows you to specify the instance of MemoryBank (by specifying an AgentEngine instance).
// Specifies also a way to convert session events to memories (see StateKeySessionLastUpdateTime)
type ServiceConfig struct {
	vertexaiutil.AgentEngineData
	// StateKeySessionLastUpdateTime controlls the process of the generation of memories.
	// If set to "", the whole session is used to generate the memories.
	// If provided, the value is treated as a key for Session State. Retrieved value (time.Time is expected) is used to filter the events to the most recent ones.
	// Warning! The value for State under this key should be set as soon as possible (for instance, during the BeforeRunCallback)
	StateKeySessionLastUpdateTime string
	WaitForCompletion             bool
}

// NewService creates a new instance of Memory Service supported by VertexAI MemoryBank.
func NewService(ctx context.Context, config *ServiceConfig) (memory.Service, error) {
	client, err := newVertexAIClient(ctx, &vertexAIClientConfig{
		AgentEngineData:   config.AgentEngineData,
		waitForCompletion: config.WaitForCompletion,
	})
	if err != nil {
		return nil, fmt.Errorf("newVertexAIClient failed: %w", err)
	}
	return &vertexAIService{
		client:                        client,
		stateKeySessionLastUpdateTime: config.StateKeySessionLastUpdateTime,
	}, nil
}

var _ memory.Service = &vertexAIService{}

// AddSessionToMemory implements [memory.Service].
func (v *vertexAIService) AddSessionToMemory(ctx context.Context, s session.Session) error {
	var err error
	if v.stateKeySessionLastUpdateTime == "" {
		// add the whole session
		err = v.client.addWholeSession(ctx, s)
		if err != nil {
			return fmt.Errorf("v.client.addWholeSession failed: %w", err)
		}
		return nil
	}

	// add only events newer than given
	t, err := s.State().Get(v.stateKeySessionLastUpdateTime)
	if err != nil {
		return fmt.Errorf("state.Get(%s) failed: %w", v.stateKeySessionLastUpdateTime, err)
	}

	tm, ok := t.(time.Time)
	if !ok {
		return fmt.Errorf("want type time.Time, got : %T (%v)", t, t)
	}
	err = v.client.addEventsNewerThan(ctx, s, tm)
	if err != nil {
		fmt.Printf("Err: %v\n", err)
		return fmt.Errorf("v.client.addEventsNewerThan failed: %w", err)
	}

	return err
}

// SearchMemory implements [memory.Service].
func (v *vertexAIService) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	return v.client.searchMemory(ctx, req)
}
