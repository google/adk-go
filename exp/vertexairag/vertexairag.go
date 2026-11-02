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
// A non-Gemini model cannot use the model-side built-in, so the tool is exposed
// as a function the model calls: it queries the Vertex AI RAG RetrieveContexts
// API directly and returns the matching passages. This mirrors adk-python's
// VertexAiRagRetrieval
// (src/google/adk/tools/retrieval/vertex_ai_rag_retrieval.py).
//
// EXPERIMENTAL: this package is not yet stable API. Anything here may change or
// be removed in a later version without notice.
package vertexairag

import (
	"context"
	"errors"
	"fmt"
	"strings"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
)

// noMatchMessage is returned to the model when retrieval finds nothing. Matching
// nothing is a normal outcome, not an error, so the model can act on it and
// continue the turn.
const noMatchMessage = "No matching result found for the query."

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

	// Project and Location address the Vertex AI RAG backend for the non-Gemini
	// retrieval fallback. They are optional when a fully-qualified corpus name
	// is configured, since the tool derives them from it, and a corpus name
	// wins over them. They are unused on the Gemini grounding path.
	Project  string
	Location string
}

// New returns a tool that grounds a Gemini model's responses on the configured
// Vertex AI RAG corpus, and for a non-Gemini model retrieves matching passages
// from it directly. It returns an error when Name or Description is empty or
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
		project:   cfg.Project,
		location:  cfg.Location,
		newClient: newVertexRagClient,
	}, nil
}

// ragRetrieval is a Vertex AI RAG retrieval tool. store, project and location
// are built once in New and read-only afterwards, so the value is safe for
// concurrent use.
type ragRetrieval struct {
	name        string
	description string
	store       *genai.VertexRAGStore
	project     string
	location    string

	// newClient builds the retrieval client. It is a field so tests can supply
	// a fake in place of the real Vertex AI RAG client.
	newClient func(ctx context.Context, location string) (ragClient, error)
}

// Name implements tool.Tool.
func (t *ragRetrieval) Name() string { return t.name }

// Description implements tool.Tool.
func (t *ragRetrieval) Description() string { return t.description }

// IsLongRunning implements tool.Tool.
func (t *ragRetrieval) IsLongRunning() bool { return false }

// Declaration is the function-tool surface used on the non-Gemini fallback path:
// a single required string parameter, query. It mirrors adk-python's
// BaseRetrievalTool._get_declaration.
func (t *ragRetrieval) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.description,
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"query": {
					Type:        "STRING",
					Description: "The query to retrieve.",
				},
			},
			Required: []string{"query"},
		},
	}
}

// ProcessRequest wires the tool into the request according to the model. For a
// Gemini model it adds the Vertex RAG retrieval entry so the model grounds its
// answer on the corpus. For a non-Gemini model, which cannot use the model-side
// built-in, it registers the tool as a function the model calls (handled by
// Run). This mirrors adk-python's vertex_ai_rag_retrieval.process_llm_request.
func (t *ragRetrieval) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if req == nil {
		return errors.New("vertexairag: llm request is nil")
	}

	if !googlellm.IsGeminiModel(req.Model) {
		return toolutils.PackTool(req, t)
	}

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		Retrieval: &genai.Retrieval{VertexRAGStore: t.store},
	})
	return nil
}

// Run retrieves passages from the Vertex AI RAG corpus for the query in args. It
// is called on the non-Gemini fallback path; on the Gemini path the model
// performs retrieval itself and Run is not called. Finding no match is not an
// error: it returns noMatchMessage so the model can continue.
func (t *ragRetrieval) Run(toolCtx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("vertexairag: unexpected args type, got %T", args)
	}
	queryRaw, exists := m["query"]
	if !exists {
		return nil, errors.New("vertexairag: missing required parameter: query")
	}
	query, ok := queryRaw.(string)
	if !ok {
		return nil, fmt.Errorf("vertexairag: query must be a string, got %T", queryRaw)
	}

	project, location, err := t.resolveProjectLocation()
	if err != nil {
		return nil, err
	}

	client, err := t.newClient(toolCtx, location)
	if err != nil {
		return nil, fmt.Errorf("vertexairag: creating Vertex AI RAG client: %w", err)
	}
	defer func() { _ = client.Close() }()

	resp, err := client.RetrieveContexts(toolCtx, t.retrieveRequest(project, location, query))
	if err != nil {
		return nil, fmt.Errorf("vertexairag: retrieving contexts: %w", err)
	}

	var texts []string
	for _, c := range resp.GetContexts().GetContexts() {
		if text := c.GetText(); text != "" {
			texts = append(texts, text)
		}
	}
	if len(texts) == 0 {
		return map[string]any{"result": noMatchMessage}, nil
	}
	return map[string]any{"result": texts}, nil
}

// retrieveRequest builds the RetrieveContexts request. SimilarityTopK is
// deliberately left off the store and set on the query: the RetrieveContexts API
// reserves that field on the store, so sending it there is rejected.
func (t *ragRetrieval) retrieveRequest(project, location, query string) *aiplatformpb.RetrieveContextsRequest {
	var ragResources []*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource
	for _, r := range t.store.RAGResources {
		ragResources = append(ragResources, &aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource{
			RagCorpus:  r.RAGCorpus,
			RagFileIds: r.RAGFileIDs,
		})
	}
	store := &aiplatformpb.RetrieveContextsRequest_VertexRagStore{
		RagCorpora:              t.store.RAGCorpora,
		RagResources:            ragResources,
		VectorDistanceThreshold: t.store.VectorDistanceThreshold,
	}

	q := &aiplatformpb.RagQuery{
		Query: &aiplatformpb.RagQuery_Text{Text: query},
	}
	if t.store.SimilarityTopK != nil {
		q.SimilarityTopK = *t.store.SimilarityTopK
	}

	return &aiplatformpb.RetrieveContextsRequest{
		Parent:     fmt.Sprintf("projects/%s/locations/%s", project, location),
		DataSource: &aiplatformpb.RetrieveContextsRequest_VertexRagStore_{VertexRagStore: store},
		Query:      q,
	}
}

// resolveProjectLocation resolves the project and location for retrieval. A
// fully-qualified corpus name identifies one endpoint, so it wins; Config fills
// in what a bare corpus name leaves unspecified. It mirrors adk-python's
// _project_and_location, except it does not read environment variables.
func (t *ragRetrieval) resolveProjectLocation() (string, string, error) {
	project, location := t.project, t.location
	if p, l, ok := parseCorpusProjectLocation(t.firstCorpusName()); ok {
		project, location = p, l
	}
	if project == "" || location == "" {
		return "", "", errors.New("vertexairag: could not resolve project and location; set Config.Project and Config.Location, or use a fully-qualified corpus name")
	}
	return project, location, nil
}

// firstCorpusName returns the first configured corpus name, preferring
// RagCorpora, or "" when none is set.
func (t *ragRetrieval) firstCorpusName() string {
	if len(t.store.RAGCorpora) > 0 {
		return t.store.RAGCorpora[0]
	}
	if len(t.store.RAGResources) > 0 {
		return t.store.RAGResources[0].RAGCorpus
	}
	return ""
}

// parseCorpusProjectLocation extracts the project and location from a
// fully-qualified corpus name ("projects/{project}/locations/{location}/..."),
// reporting ok=false when name is not in that form.
func parseCorpusProjectLocation(name string) (project, location string, ok bool) {
	if !strings.HasPrefix(name, "projects/") {
		return "", "", false
	}
	parts := strings.Split(name, "/")
	if len(parts) >= 4 && parts[0] == "projects" && parts[2] == "locations" {
		return parts[1], parts[3], true
	}
	return "", "", false
}

// ragClient is the slice of the Vertex AI RAG service the tool needs.
type ragClient interface {
	RetrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error)
	Close() error
}

// newVertexRagClient builds the real client against the region's endpoint.
func newVertexRagClient(ctx context.Context, location string) (ragClient, error) {
	c, err := aiplatform.NewVertexRagClient(ctx, option.WithEndpoint(aiplatformutil.HostPortURL(location)))
	if err != nil {
		return nil, err
	}
	return vertexRagClientAdapter{c: c}, nil
}

// vertexRagClientAdapter adapts *aiplatform.VertexRagClient to ragClient,
// dropping the gax call-option variadic the tool never uses so the package needs
// no direct dependency on gax.
type vertexRagClientAdapter struct {
	c *aiplatform.VertexRagClient
}

func (a vertexRagClientAdapter) RetrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
	return a.c.RetrieveContexts(ctx, req)
}

func (a vertexRagClientAdapter) Close() error { return a.c.Close() }
