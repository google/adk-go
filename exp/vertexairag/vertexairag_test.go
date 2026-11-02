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

package vertexairag_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/vertexairag"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

const testCorpus = "projects/test-project/locations/us-central1/ragCorpora/test-corpus"

// mustNewTool builds a tool with a fixed, valid config for ProcessRequest and
// Declaration tests.
func mustNewTool(t *testing.T) tool.Tool {
	t.Helper()
	topK := int32(3)
	tl, err := vertexairag.New(vertexairag.Config{
		Name:           "rag_retrieval",
		Description:    "Retrieves documentation.",
		RagCorpora:     []string{testCorpus},
		SimilarityTopK: &topK,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return tl
}

func TestNew(t *testing.T) {
	topK := int32(5)
	tests := []struct {
		name    string
		cfg     vertexairag.Config
		wantErr bool
	}{
		{
			name:    "missing name",
			cfg:     vertexairag.Config{Description: "d", RagCorpora: []string{testCorpus}},
			wantErr: true,
		},
		{
			name:    "missing description",
			cfg:     vertexairag.Config{Name: "rag_retrieval", RagCorpora: []string{testCorpus}},
			wantErr: true,
		},
		{
			name:    "no corpus source",
			cfg:     vertexairag.Config{Name: "rag_retrieval", Description: "d"},
			wantErr: true,
		},
		{
			name: "valid with corpora",
			cfg: vertexairag.Config{
				Name:           "rag_retrieval",
				Description:    "Retrieves documentation.",
				RagCorpora:     []string{testCorpus},
				SimilarityTopK: &topK,
			},
		},
		{
			name: "valid with resources",
			cfg: vertexairag.Config{
				Name:         "rag_retrieval",
				Description:  "Retrieves documentation.",
				RagResources: []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := vertexairag.New(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
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
		})
	}
}

func TestDeclaration(t *testing.T) {
	ft, ok := mustNewTool(t).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("tool does not implement toolinternal.FunctionTool")
	}
	decl := ft.Declaration()
	if decl.Name != "rag_retrieval" {
		t.Errorf("Declaration().Name = %q, want %q", decl.Name, "rag_retrieval")
	}
	if diff := cmp.Diff([]string{"query"}, decl.Parameters.Required); diff != "" {
		t.Errorf("Declaration().Parameters.Required mismatch (-want +got):\n%s", diff)
	}
	q, ok := decl.Parameters.Properties["query"]
	if !ok {
		t.Fatal("Declaration() has no \"query\" parameter")
	}
	if q.Type != genai.TypeString {
		t.Errorf("query parameter type = %q, want %q", q.Type, genai.TypeString)
	}
}

func TestProcessRequest_Gemini(t *testing.T) {
	topK := int32(3)
	wantStore := &genai.VertexRAGStore{
		RAGCorpora:     []string{testCorpus},
		SimilarityTopK: &topK,
	}

	tests := []struct {
		name      string
		req       *model.LLMRequest
		wantTools []*genai.Tool
		wantErr   bool
	}{
		{
			name: "empty request",
			req:  &model.LLMRequest{Model: "gemini-2.5-flash"},
			wantTools: []*genai.Tool{
				{Retrieval: &genai.Retrieval{VertexRAGStore: wantStore}},
			},
		},
		{
			name: "appends to existing tools",
			req: &model.LLMRequest{
				Model: "gemini-2.5-flash",
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
				},
			},
			wantTools: []*genai.Tool{
				{GoogleSearch: &genai.GoogleSearch{}},
				{Retrieval: &genai.Retrieval{VertexRAGStore: wantStore}},
			},
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rp := mustNewTool(t).(toolinternal.RequestProcessor)

			err := rp.ProcessRequest(nil, tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ProcessRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.wantTools, tt.req.Config.Tools); diff != "" {
				t.Errorf("ProcessRequest() tools mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestProcessRequest_NonGemini verifies the fallback path: a non-Gemini model
// cannot use the model-side built-in, so the tool is registered as a function
// the model can call (handled by Run) rather than a grounding entry.
func TestProcessRequest_NonGemini(t *testing.T) {
	rp := mustNewTool(t).(toolinternal.RequestProcessor)
	req := &model.LLMRequest{Model: "claude-3-5-sonnet"}

	if err := rp.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}

	if _, ok := req.Tools["rag_retrieval"]; !ok {
		t.Errorf("req.Tools does not contain the registered tool; got keys %v", keys(req.Tools))
	}

	var gotDecl bool
	for _, gt := range req.Config.Tools {
		for _, fd := range gt.FunctionDeclarations {
			if fd.Name == "rag_retrieval" {
				gotDecl = true
			}
		}
	}
	if !gotDecl {
		t.Error("req.Config.Tools has no function declaration named \"rag_retrieval\"")
	}
	// The grounding entry must not be present on the non-Gemini path.
	for _, gt := range req.Config.Tools {
		if gt.Retrieval != nil {
			t.Error("req.Config.Tools unexpectedly contains a Retrieval grounding entry")
		}
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
