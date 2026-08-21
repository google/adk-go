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

package retrievaltool_test

import (
	"context"
	"testing"

	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/sessioninternal"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	toolpkg "google.golang.org/adk/tool"
	"google.golang.org/adk/tool/retrievaltool"
	"google.golang.org/genai"
)

func TestVertexAIRAG_ProcessRequest(t *testing.T) {
	similarityTopK := int32(5)
	vectorDistanceThreshold := 0.8
	ragStore := &genai.VertexRAGStore{
		RAGCorpora:              []string{"projects/123456789/locations/us-central1/ragCorpora/1234567890"},
		SimilarityTopK:          &similarityTopK,
		VectorDistanceThreshold: &vectorDistanceThreshold,
	}

	tool, err := retrievaltool.NewVertexAIRAG("test_rag", "Test RAG tool", ragStore)
	if err != nil {
		t.Fatalf("NewVertexAIRAG() failed: %v", err)
	}

	req := &model.LLMRequest{
		Model: "gemini-2.0-flash",
	}

	requestProcessor, ok := tool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("tool does not implement RequestProcessor")
	}

	toolCtx := createToolContext(t)
	err = requestProcessor.ProcessRequest(toolCtx, req)
	if err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	if req.Config == nil {
		t.Fatal("req.Config is nil")
	}

	if len(req.Config.Tools) == 0 {
		t.Fatal("req.Config.Tools is empty")
	}

	// Find the retrieval tool
	var foundRetrievalTool bool
	for _, genaiTool := range req.Config.Tools {
		if genaiTool.Retrieval != nil {
			foundRetrievalTool = true
			store := genaiTool.Retrieval.VertexRAGStore
			if store == nil {
				t.Fatal("VertexRAGStore is nil")
			}

			if len(store.RAGCorpora) == 0 {
				t.Error("RAGCorpora is empty")
			} else {
				expectedCorpus := "projects/123456789/locations/us-central1/ragCorpora/1234567890"
				if store.RAGCorpora[0] != expectedCorpus {
					t.Errorf("Expected corpus %s, got %s", expectedCorpus, store.RAGCorpora[0])
				}
			}

			if store.SimilarityTopK == nil {
				t.Error("SimilarityTopK is nil")
			} else if *store.SimilarityTopK != similarityTopK {
				t.Errorf("Expected SimilarityTopK %d, got %d", similarityTopK, *store.SimilarityTopK)
			}

			if store.VectorDistanceThreshold == nil {
				t.Error("VectorDistanceThreshold is nil")
			} else if *store.VectorDistanceThreshold != vectorDistanceThreshold {
				t.Errorf("Expected VectorDistanceThreshold %f, got %f", vectorDistanceThreshold, *store.VectorDistanceThreshold)
			}
		}
	}
	if !foundRetrievalTool {
		t.Error("Retrieval tool not found in request config")
	}
}

// createToolContext creates a tool context for testing
func createToolContext(t *testing.T) toolpkg.Context {
	t.Helper()

	sessionService := session.InMemoryService()
	createResponse, err := sessionService.Create(context.Background(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	s := createResponse.Session
	sessionImpl := sessioninternal.NewMutableSession(sessionService, s)

	ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Session: sessionImpl,
	})

	return toolinternal.NewToolContext(ctx, "", &session.EventActions{})
}
