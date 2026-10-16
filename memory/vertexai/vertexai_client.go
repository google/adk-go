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
	"context"
	"fmt"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
	vertexaiutil "google.golang.org/adk/v2/util/vertexai"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
)

type vertexAIClient struct {
	config          vertexAIClientConfig
	client          *aiplatform.MemoryBankClient
	agentEngineData *vertexaiutil.AgentEngineData
	parent          string
}

type vertexAIClientConfig struct {
	vertexaiutil.AgentEngineData
	waitForCompletion bool
}

func newVertexAIClient(ctx context.Context, config *vertexAIClientConfig) (*vertexAIClient, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	c, err := aiplatform.NewMemoryBankClient(ctx, option.WithEndpoint(aiplatformutil.HostPortURL(config.Location)))
	if err != nil {
		return nil, fmt.Errorf("aiplatform.NewMemoryBankClient failed: %w", err)
	}
	return &vertexAIClient{
		config:          *config,
		client:          c,
		agentEngineData: &config.AgentEngineData,
		parent:          vertexaiutil.AgentEngineResource(&config.AgentEngineData),
	}, nil
}

// addWholeSession adds the whole session to the Memory
func (v *vertexAIClient) addWholeSession(ctx context.Context, s session.Session) error {
	return v.addSession(ctx, s, nil)
}

// addEventsNewerThan uses time to filter out the old event. The new ones are used to generate memories
func (v *vertexAIClient) addEventsNewerThan(ctx context.Context, s session.Session, start time.Time) error {
	return v.addSession(ctx, s, &start)
}

// addSession adds the whole session or just events created after `start` to the Memory
func (v *vertexAIClient) addSession(ctx context.Context, s session.Session, start *time.Time) error {
	sr := vertexaiutil.SessionResource(v.agentEngineData, s.ID())

	vss := &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource{
		Session: sr,
	}
	if start != nil {
		vss.StartTime = timestamppb.New(*start)
	}
	req := &aiplatformpb.GenerateMemoriesRequest{
		Parent: v.parent,
		Source: &aiplatformpb.GenerateMemoriesRequest_VertexSessionSource_{VertexSessionSource: vss},
		Scope:  createScope(s.AppName(), s.UserID()),
	}

	op, err := v.client.GenerateMemories(ctx, req)
	if err != nil {
		return fmt.Errorf("v.client.GenerateMemories failed: %w", err)
	}
	if v.config.waitForCompletion {
		_, err = op.Wait(ctx)
		if err != nil && err.Error() == "unsupported result type <nil>: <nil>" {
			// accept it
			err = nil
		}
		if err != nil {
			return fmt.Errorf("op.Wait for GenerateMemories (whole session) failed: %w", err)
		}
	}
	return nil
}

// searchMemory uses provided query to find the relevant memories for the given user
func (v *vertexAIClient) searchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	r, err := v.client.RetrieveMemories(ctx,
		&aiplatformpb.RetrieveMemoriesRequest{
			RetrievalParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams_{
				SimilaritySearchParams: &aiplatformpb.RetrieveMemoriesRequest_SimilaritySearchParams{
					SearchQuery: req.Query,
				},
			},
			Parent: v.parent,
			Scope:  createScope(req.AppName, req.UserID),
		},
	)
	if err != nil {
		return nil, fmt.Errorf("v.client.RetrieveMemories failed: %w", err)
	}

	res := &memory.SearchResponse{
		Memories: []memory.Entry{},
	}

	for _, m := range r.RetrievedMemories {
		res.Memories = append(res.Memories, memory.Entry{
			Content: genai.NewContentFromText(m.Memory.Fact, genai.RoleUser),
		})
	}

	return res, nil
}

// createScope builds the MemoryBank scope that partitions memories by
// (app_name, user_id). Memory Bank matches the scope map exactly on retrieval,
// so including app_name keeps each application's memories in its own partition;
// without it, applications sharing a MemoryBank for the same user write into and
// read from a single shared partition. This is a namespacing convention rather
// than an authenticated confidentiality boundary — on the REST path the app name
// comes from the request and the handler does not authenticate the caller — and
// it matches the (app_name, user_id) keying used by the in-memory memory service
// (memory/inmemory.go) and the adk-python MemoryBank implementation.
func createScope(appName, userID string) map[string]string {
	return map[string]string{"app_name": appName, "user_id": userID}
}
