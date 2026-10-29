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

package loadartifactstool_test

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/loadartifactstool"
)

type countingArtifactService struct {
	artifact.Service
	listCalls atomic.Int64
}

func (s *countingArtifactService) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	s.listCalls.Add(1)
	return s.Service.List(ctx, req)
}

// BenchmarkLoadArtifactsToolLoopTurn measures loadartifactstool.ProcessRequest
// across a 4-model-call tool loop within a single invocation turn.
func BenchmarkLoadArtifactsToolLoopTurn(b *testing.B) {
	ctx := b.Context()
	svc := &countingArtifactService{Service: artifact.InMemoryService()}
	artifacts := &artifactinternal.Artifacts{
		Service:   svc,
		AppName:   "bench-app",
		UserID:    "bench-user",
		SessionID: "bench-session",
	}
	for i := range 5 {
		name := fmt.Sprintf("artifact-%d.txt", i)
		if _, err := artifacts.Save(ctx, name, genai.NewPartFromText("payload")); err != nil {
			b.Fatalf("Save(%q) failed: %v", name, err)
		}
	}

	lat := loadartifactstool.New()
	rp, ok := lat.(toolinternal.RequestProcessor)
	if !ok {
		b.Fatal("loadartifactstool.New() does not implement RequestProcessor")
	}

	svc.listCalls.Store(0)
	b.ReportAllocs()
	b.ResetTimer()

	for i := range b.N {
		invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:    artifacts,
			InvocationID: fmt.Sprintf("inv-%d", i),
		})
		tc := agent.NewToolContext(invCtx, "", nil, nil)
		for range 4 {
			req := &model.LLMRequest{}
			if err := rp.ProcessRequest(tc, req); err != nil {
				b.Fatalf("ProcessRequest failed: %v", err)
			}
		}
	}

	b.StopTimer()
	b.ReportMetric(float64(svc.listCalls.Load())/float64(b.N), "artifact-lists/turn")
}
