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

// Package vertexairagtool provides a tool that grounds an agent's responses on a
// Vertex AI RAG (Retrieval-Augmented Generation) corpus.
//
// The tool is a model-side built-in: before each model call it adds a Vertex RAG
// retrieval entry to the request, and the Gemini backend performs the retrieval
// against the configured corpus. This requires the Vertex AI backend; the
// Gemini API backend does not support Vertex RAG grounding. Non-Gemini models
// are not supported.
//
// EXPERIMENTAL: this package is not yet stable API. Anything here may change or
// be removed in a later version without notice.
package vertexairagtool

import (
	"errors"
	"fmt"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// Config configures a Vertex AI RAG retrieval tool. Description and at least one
// RAGResources entry are required.
type Config struct {
	// Name for the tool
	Name string
	// Description tells the model what the corpus holds and when to consult it.
	Description string

	// RAGResources selects a corpus and, optionally, specific files within it.
	// Format for RAGCorpus: "projects/{project}/locations/{location}/ragCorpora/{corpus}".
	RAGResources []*genai.VertexRAGStoreRAGResource

	// RAGRetrievalConfig configures filtering, hybrid search, ranking and context
	// count. Nil leaves the backend defaults in place. These options extend the
	// configuration supported by the Python retrieval tool.
	RAGRetrievalConfig *genai.RAGRetrievalConfig

	// SimilarityTopK caps how many results retrieval returns. Nil leaves the
	// backend default in place.
	SimilarityTopK *int32

	// VectorDistanceThreshold drops results whose vector distance exceeds it.
	// Nil leaves the backend default in place.
	VectorDistanceThreshold *float64
}

// New returns a tool that grounds a Gemini model's responses on the configured
// Vertex AI RAG corpus. It returns an error when Description or RAGResources is
// empty.
func New(cfg Config) (tool.Tool, error) {
	if cfg.Description == "" {
		return nil, errors.New("vertexairag: Config.Description is required")
	}
	if len(cfg.RAGResources) == 0 {
		return nil, errors.New("vertexairag: Config.RAGResources is required")
	}

	for _, r := range cfg.RAGResources {
		if r == nil {
			return nil, errors.New("vertexairag: Config.RAGResources cannot be nil")
		}
	}

	// Copy so that later changes to cfg do not reach the tool.
	store := cloneStore(&genai.VertexRAGStore{
		RAGResources:            cfg.RAGResources,
		RAGRetrievalConfig:      cfg.RAGRetrievalConfig,
		SimilarityTopK:          cfg.SimilarityTopK,
		VectorDistanceThreshold: cfg.VectorDistanceThreshold,
	})

	return &ragRetrieval{
		description: cfg.Description,
		name:        cfg.Name,
		store:       store,
	}, nil
}

// ragRetrieval is a Vertex AI RAG retrieval tool. store is built in New and
// never handed out directly, so the value is safe for concurrent use.
type ragRetrieval struct {
	description string
	name        string
	store       *genai.VertexRAGStore
}

// Name implements tool.Tool.
func (r *ragRetrieval) Name() string { return r.name }

// Description implements tool.Tool.
func (r *ragRetrieval) Description() string { return r.description }

// IsLongRunning implements tool.Tool.
func (r *ragRetrieval) IsLongRunning() bool { return false }

// ProcessRequest adds the Vertex RAG retrieval entry to the request so a Gemini
// model grounds its answer on the corpus. It returns an error for a non-Gemini
// model, which cannot use the model-side built-in.
func (r *ragRetrieval) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if req == nil {
		return errors.New("vertexairag: llm request is nil")
	}

	if !googlellm.IsGeminiModel(req.Model) {
		return fmt.Errorf("vertexairag: ragtool tool is not supported for model %v", req.Model)
	}

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	// Each request gets its own copy, so code that modifies one request's tools
	// cannot change what later requests send.
	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		Retrieval: &genai.Retrieval{VertexRAGStore: cloneStore(r.store)},
	})
	return nil
}

// cloneStore returns a deep copy of the fields of s that New sets:
// RAGResources, RAGRetrievalConfig, SimilarityTopK and VectorDistanceThreshold.
// Other fields are not copied. Every element of s.RAGResources must be non-nil.
func cloneStore(s *genai.VertexRAGStore) *genai.VertexRAGStore {
	res := make([]*genai.VertexRAGStoreRAGResource, len(s.RAGResources))
	for i, r := range s.RAGResources {
		res[i] = &genai.VertexRAGStoreRAGResource{
			RAGCorpus:  r.RAGCorpus,
			RAGFileIDs: slices.Clone(r.RAGFileIDs),
		}
	}
	cfg := clonePtr(s.RAGRetrievalConfig)
	if cfg != nil {
		cfg.TopK = clonePtr(cfg.TopK)
		cfg.Filter = clonePtr(cfg.Filter)
		if cfg.Filter != nil {
			cfg.Filter.VectorDistanceThreshold = clonePtr(cfg.Filter.VectorDistanceThreshold)
			cfg.Filter.VectorSimilarityThreshold = clonePtr(cfg.Filter.VectorSimilarityThreshold)
		}
		cfg.HybridSearch = clonePtr(cfg.HybridSearch)
		if cfg.HybridSearch != nil {
			cfg.HybridSearch.Alpha = clonePtr(cfg.HybridSearch.Alpha)
		}
		cfg.Ranking = clonePtr(cfg.Ranking)
		if cfg.Ranking != nil {
			cfg.Ranking.LlmRanker = clonePtr(cfg.Ranking.LlmRanker)
			cfg.Ranking.RankService = clonePtr(cfg.Ranking.RankService)
		}
	}
	return &genai.VertexRAGStore{
		RAGResources:            res,
		RAGRetrievalConfig:      cfg,
		SimilarityTopK:          clonePtr(s.SimilarityTopK),
		VectorDistanceThreshold: clonePtr(s.VectorDistanceThreshold),
	}
}

// clonePtr returns a pointer to a copy of *p, or nil when p is nil.
func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	return new(*p)
}
