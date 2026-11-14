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

package vertexairagtool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/vertexairagtool"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

const testCorpus = "projects/test-project/locations/us-central1/ragCorpora/test-corpus"

func TestNew(t *testing.T) {
	testCases := []struct {
		name        string
		cfg         vertexairagtool.Config
		wantErrText string // substring; empty means no error expected
	}{
		{
			name:        "missing description",
			cfg:         vertexairagtool.Config{RAGResources: []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}}},
			wantErrText: "Config.Description is required",
		},
		{
			name:        "nil resources",
			cfg:         vertexairagtool.Config{Description: "d"},
			wantErrText: "Config.RAGResources is required",
		},
		{
			name:        "empty resources",
			cfg:         vertexairagtool.Config{Description: "d", RAGResources: []*genai.VertexRAGStoreRAGResource{}},
			wantErrText: "Config.RAGResources is required",
		},
		{
			name:        "nil resource element",
			cfg:         vertexairagtool.Config{Description: "d", RAGResources: []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}, nil}},
			wantErrText: "Config.RAGResources cannot be nil",
		},
		{
			name: "minimal config",
			cfg: vertexairagtool.Config{
				Description:  "Retrieves documentation.",
				RAGResources: []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}},
			},
		},
		{
			name: "full config",
			cfg: vertexairagtool.Config{
				Description:             "Retrieves documentation.",
				RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"file-1"}}},
				SimilarityTopK:          new(int32(5)),
				VectorDistanceThreshold: new(0.4),
			},
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := vertexairagtool.New(tt.cfg)
			if tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("New() error = nil, want error containing %q", tt.wantErrText)
				}
				if !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("New() error = %q, want it to contain %q", err, tt.wantErrText)
				}
				return
			}
			if err != nil {
				t.Fatalf("New() unexpected error = %v", err)
			}
			if got.Name() != tt.cfg.Name {
				t.Errorf("Name() = %q, want %q", got.Name(), tt.cfg.Name)
			}
			if got.Description() != tt.cfg.Description {
				t.Errorf("Description() = %q, want %q", got.Description(), tt.cfg.Description)
			}
			if got.IsLongRunning() {
				t.Error("IsLongRunning() = true, want false")
			}
			if _, ok := got.(toolinternal.RequestProcessor); !ok {
				t.Error("tool does not implement toolinternal.RequestProcessor")
			}
			// The tool is a model-side built-in only; it must not be offered to
			// the model as a callable function.
			if _, ok := got.(toolinternal.FunctionTool); ok {
				t.Error("tool implements toolinternal.FunctionTool, want built-in only")
			}
		})
	}
}

func TestProcessRequest(t *testing.T) {
	cfg := vertexairagtool.Config{
		Description:             "Retrieves documentation.",
		RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"file-1"}}},
		SimilarityTopK:          new(int32(3)),
		VectorDistanceThreshold: new(0.5),
	}
	retrieval := &genai.Tool{Retrieval: &genai.Retrieval{VertexRAGStore: &genai.VertexRAGStore{
		RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"file-1"}}},
		SimilarityTopK:          new(int32(3)),
		VectorDistanceThreshold: new(0.5),
	}}}

	testCases := []struct {
		name        string
		req         *model.LLMRequest
		wantTools   []*genai.Tool
		wantErrText string // substring; empty means no error expected
	}{
		{
			name:      "gemini adds retrieval to request without config",
			req:       &model.LLMRequest{Model: "gemini-2.5-flash"},
			wantTools: []*genai.Tool{retrieval},
		},
		{
			name: "gemini adds retrieval alongside existing tools",
			req: &model.LLMRequest{
				Model: "gemini-2.5-flash",
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
				},
			},
			wantTools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}, retrieval},
		},
		{
			name:      "path-based gemini model resolves and adds retrieval",
			req:       &model.LLMRequest{Model: "projects/p/locations/global/publishers/google/models/gemini-2.5-flash"},
			wantTools: []*genai.Tool{retrieval},
		},
		{
			name:        "non-gemini model is unsupported",
			req:         &model.LLMRequest{Model: "gpt-4o"},
			wantErrText: "ragtool tool is not supported for model gpt-4o",
		},
		{
			name:        "bare gemini is unsupported",
			req:         &model.LLMRequest{Model: "gemini"},
			wantErrText: "ragtool tool is not supported for model gemini",
		},
		{
			name:        "empty model is unsupported",
			req:         &model.LLMRequest{},
			wantErrText: "ragtool tool is not supported for model",
		},
		{
			name:        "nil request errors",
			req:         nil,
			wantErrText: "llm request is nil",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			tl, err := vertexairagtool.New(cfg)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			rp := tl.(toolinternal.RequestProcessor)

			err = rp.ProcessRequest(nil, tt.req)
			if tt.wantErrText != "" {
				if err == nil {
					t.Fatalf("ProcessRequest() error = nil, want error containing %q", tt.wantErrText)
				}
				if !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("ProcessRequest() error = %q, want it to contain %q", err, tt.wantErrText)
				}
				if tt.req != nil && tt.req.Config != nil {
					t.Errorf("ProcessRequest() set req.Config = %+v on error, want nil", tt.req.Config)
				}
				return
			}
			if err != nil {
				t.Fatalf("ProcessRequest() unexpected error = %v", err)
			}
			if diff := cmp.Diff(tt.wantTools, tt.req.Config.Tools); diff != "" {
				t.Errorf("ProcessRequest() tools mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestProcessRequest_RAGRetrievalConfig(t *testing.T) {
	testCases := []struct {
		name string
		cfg  *genai.RAGRetrievalConfig
	}{
		{name: "nil"},
		{name: "empty", cfg: &genai.RAGRetrievalConfig{}},
		{name: "empty filter", cfg: &genai.RAGRetrievalConfig{Filter: &genai.RAGRetrievalConfigFilter{}}},
		{name: "empty hybrid search", cfg: &genai.RAGRetrievalConfig{HybridSearch: &genai.RAGRetrievalConfigHybridSearch{}}},
		{name: "empty ranking", cfg: &genai.RAGRetrievalConfig{Ranking: &genai.RAGRetrievalConfigRanking{}}},
		{
			name: "zero values",
			cfg: &genai.RAGRetrievalConfig{
				Filter: &genai.RAGRetrievalConfigFilter{
					VectorDistanceThreshold:   new(float64(0)),
					VectorSimilarityThreshold: new(float64(0)),
				},
				HybridSearch: &genai.RAGRetrievalConfigHybridSearch{Alpha: new(float32(0))},
				TopK:         new(int32(0)),
			},
		},
		{
			name: "llm ranker only",
			cfg: &genai.RAGRetrievalConfig{Ranking: &genai.RAGRetrievalConfigRanking{
				LlmRanker: &genai.RAGRetrievalConfigRankingLlmRanker{ModelName: "gemini-2.5-flash"},
			}},
		},
		{
			name: "rank service only",
			cfg: &genai.RAGRetrievalConfig{Ranking: &genai.RAGRetrievalConfigRanking{
				RankService: &genai.RAGRetrievalConfigRankingRankService{ModelName: "semantic-ranker-512@latest"},
			}},
		},
		{name: "full", cfg: retrievalConfig()},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			tl, err := vertexairagtool.New(vertexairagtool.Config{
				Description:        "d",
				RAGResources:       []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}},
				RAGRetrievalConfig: tt.cfg,
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if diff := cmp.Diff(tt.cfg, processStore(t, tl).RAGRetrievalConfig); diff != "" {
				t.Errorf("ProcessRequest() retrieval config mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// processStore runs ProcessRequest on a fresh Gemini request and returns the
// Vertex RAG store it added.
func processStore(t *testing.T, tl tool.Tool) *genai.VertexRAGStore {
	t.Helper()
	req := &model.LLMRequest{Model: "gemini-2.5-flash"}
	if err := tl.(toolinternal.RequestProcessor).ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
	if len(req.Config.Tools) != 1 || req.Config.Tools[0].Retrieval == nil {
		t.Fatalf("ProcessRequest() tools = %+v, want one Retrieval tool", req.Config.Tools)
	}
	return req.Config.Tools[0].Retrieval.VertexRAGStore
}

// wantStore is the store built from the config used by the copy tests.
func wantStore() *genai.VertexRAGStore {
	return &genai.VertexRAGStore{
		RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"file-1"}}},
		RAGRetrievalConfig:      retrievalConfig(),
		SimilarityTopK:          new(int32(3)),
		VectorDistanceThreshold: new(0.5),
	}
}

func retrievalConfig() *genai.RAGRetrievalConfig {
	return &genai.RAGRetrievalConfig{
		Filter: &genai.RAGRetrievalConfigFilter{
			MetadataFilter:            "category = \"docs\"",
			VectorDistanceThreshold:   new(0.4),
			VectorSimilarityThreshold: new(0.6),
		},
		HybridSearch: &genai.RAGRetrievalConfigHybridSearch{Alpha: new(float32(0.7))},
		Ranking: &genai.RAGRetrievalConfigRanking{
			LlmRanker:   &genai.RAGRetrievalConfigRankingLlmRanker{ModelName: "gemini-2.5-flash"},
			RankService: &genai.RAGRetrievalConfigRankingRankService{ModelName: "semantic-ranker-512@latest"},
		},
		TopK: new(int32(5)),
	}
}

// TestNew_CopiesConfig verifies that changing the caller's Config after New
// does not change what the tool sends.
func TestNew_CopiesConfig(t *testing.T) {
	fileIDs := []string{"file-1"}
	resources := []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: fileIDs}}
	topK := int32(3)
	threshold := 0.5
	retrieval := retrievalConfig()
	tl, err := vertexairagtool.New(vertexairagtool.Config{
		Description:             "d",
		RAGResources:            resources,
		RAGRetrievalConfig:      retrieval,
		SimilarityTopK:          &topK,
		VectorDistanceThreshold: &threshold,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	fileIDs[0] = "changed-file"
	resources[0].RAGCorpus = "projects/other/locations/us-east1/ragCorpora/mutated"
	resources[0] = &genai.VertexRAGStoreRAGResource{RAGCorpus: "projects/other/locations/us-east1/ragCorpora/replaced"}
	topK = 99
	threshold = 0.9
	retrieval.Filter.MetadataFilter = "changed"
	*retrieval.Filter.VectorDistanceThreshold = 0.9
	*retrieval.Filter.VectorSimilarityThreshold = 0.1
	*retrieval.HybridSearch.Alpha = 0.2
	retrieval.Ranking.LlmRanker.ModelName = "changed-llm"
	retrieval.Ranking.RankService.ModelName = "changed-ranker"
	*retrieval.TopK = 99

	if diff := cmp.Diff(wantStore(), processStore(t, tl)); diff != "" {
		t.Errorf("ProcessRequest() store mismatch (-want +got):\n%s", diff)
	}
}

// TestProcessRequest_CopiesStorePerRequest verifies that modifying the store
// added to one request does not change what later requests send.
func TestProcessRequest_CopiesStorePerRequest(t *testing.T) {
	tl, err := vertexairagtool.New(vertexairagtool.Config{
		Description:             "d",
		RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"file-1"}}},
		RAGRetrievalConfig:      retrievalConfig(),
		SimilarityTopK:          new(int32(3)),
		VectorDistanceThreshold: new(0.5),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	first := processStore(t, tl)
	first.RAGResources[0].RAGCorpus = "projects/other/locations/us-east1/ragCorpora/mutated"
	first.RAGResources[0].RAGFileIDs[0] = "changed-file"
	first.RAGResources = append(first.RAGResources, &genai.VertexRAGStoreRAGResource{RAGCorpus: "extra"})
	*first.SimilarityTopK = 99
	*first.VectorDistanceThreshold = 0.9
	first.RAGRetrievalConfig.Filter.MetadataFilter = "changed"
	*first.RAGRetrievalConfig.Filter.VectorDistanceThreshold = 0.9
	*first.RAGRetrievalConfig.Filter.VectorSimilarityThreshold = 0.1
	*first.RAGRetrievalConfig.HybridSearch.Alpha = 0.2
	first.RAGRetrievalConfig.Ranking.LlmRanker.ModelName = "changed-llm"
	first.RAGRetrievalConfig.Ranking.RankService.ModelName = "changed-ranker"
	*first.RAGRetrievalConfig.TopK = 99

	if diff := cmp.Diff(wantStore(), processStore(t, tl)); diff != "" {
		t.Errorf("second ProcessRequest() store mismatch (-want +got):\n%s", diff)
	}
}
