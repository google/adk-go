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

// Package vertexairag provides a tool that grounds an agent's responses on a
// Vertex AI RAG (Retrieval-Augmented Generation) corpus.
//
// For a Gemini model the tool is a model-side built-in: before each model call
// it adds a Vertex RAG retrieval entry to the request, and the Gemini backend
// performs the retrieval against the configured corpus. This requires the
// Vertex AI backend; the Gemini API backend does not support Vertex RAG
// grounding.
//
// A non-Gemini model cannot use the model-side built-in. adk-python falls back
// to a function tool that queries the Vertex AI RAG API directly
// (src/google/adk/tools/retrieval/vertex_ai_rag_retrieval.py); that fallback is
// not implemented here yet, so for a non-Gemini model the tool reports the
// model as unsupported.
//
// EXPERIMENTAL: this package is not yet stable API. Anything here may change or
// be removed in a later version without notice.
package vertexairag

import (
	"errors"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// Config configures a Vertex AI RAG retrieval tool. The zero value is not
// usable: Name, Description and at least one corpus source are required.
type Config struct {
	// Name is the tool name presented to the model.
	Name string
	// Description tells the model what the corpus holds and when to consult it.
	Description string

	// RagCorpora lists fully-qualified corpus resource names, formatted as
	// "projects/{project}/locations/{location}/ragCorpora/{corpus}".
	//
	// The backend deprecates this in favor of RagResources but still accepts
	// it; adk-python supports both. Set at least one of RagCorpora or
	// RagResources.
	RagCorpora []string
	// RagResources selects a corpus and, optionally, specific files within it.
	// Set at least one of RagCorpora or RagResources.
	RagResources []*genai.VertexRAGStoreRAGResource

	// SimilarityTopK caps how many results retrieval returns. Nil leaves the
	// backend default in place.
	SimilarityTopK *int32
	// VectorDistanceThreshold drops results whose vector distance exceeds it.
	// Nil leaves the backend default in place.
	VectorDistanceThreshold *float64
}

// New returns a tool that grounds a Gemini model's responses on the configured
// Vertex AI RAG corpus. It returns an error when Name or Description is empty or
// when neither RagCorpora nor RagResources is set.
func New(cfg Config) (tool.Tool, error) {
	if cfg.Name == "" {
		return nil, errors.New("vertexairag: Config.Name is required")
	}
	if cfg.Description == "" {
		return nil, errors.New("vertexairag: Config.Description is required")
	}
	if len(cfg.RagCorpora) == 0 && len(cfg.RagResources) == 0 {
		return nil, errors.New("vertexairag: at least one of Config.RagCorpora or Config.RagResources is required")
	}

	return &ragRetrieval{
		name:        cfg.Name,
		description: cfg.Description,
		store: &genai.VertexRAGStore{
			RAGCorpora:              cfg.RagCorpora,
			RAGResources:            cfg.RagResources,
			SimilarityTopK:          cfg.SimilarityTopK,
			VectorDistanceThreshold: cfg.VectorDistanceThreshold,
		},
	}, nil
}

// ragRetrieval is a model-side Vertex AI RAG grounding tool. store is built once
// in New and read-only afterwards, so the value is safe for concurrent use.
type ragRetrieval struct {
	name        string
	description string
	store       *genai.VertexRAGStore
}

// Name implements tool.Tool.
func (t *ragRetrieval) Name() string { return t.name }

// Description implements tool.Tool.
func (t *ragRetrieval) Description() string { return t.description }

// IsLongRunning implements tool.Tool.
func (t *ragRetrieval) IsLongRunning() bool { return false }

// ProcessRequest adds the Vertex RAG retrieval entry to a Gemini request so the
// model grounds its answer on the corpus. It reports a non-Gemini model as
// unsupported, mirroring adk-python's grounding branch
// (vertex_ai_rag_retrieval.py: process_llm_request). Unlike adk-python it does
// not yet provide the non-Gemini function-tool fallback.
func (t *ragRetrieval) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if req == nil {
		return errors.New("vertexairag: llm request is nil")
	}
	if !googlellm.IsGeminiModel(req.Model) {
		return fmt.Errorf("vertexairag: Vertex AI RAG retrieval is not supported for model %q", req.Model)
	}

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		Retrieval: &genai.Retrieval{VertexRAGStore: t.store},
	})
	return nil
}
