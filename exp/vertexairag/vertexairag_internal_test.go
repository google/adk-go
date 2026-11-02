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

package vertexairag

import (
	"context"
	"errors"
	"testing"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

const testCorpus = "projects/test-project/locations/us-central1/ragCorpora/test-corpus"

// fakeRagClient is a test double for the Vertex AI RAG client. It records the
// request it receives and returns a canned response.
type fakeRagClient struct {
	gotReq *aiplatformpb.RetrieveContextsRequest
	resp   *aiplatformpb.RetrieveContextsResponse
	err    error
	closed bool
}

func (f *fakeRagClient) RetrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
	f.gotReq = req
	return f.resp, f.err
}

func (f *fakeRagClient) Close() error {
	f.closed = true
	return nil
}

// toolWithClient builds a tool whose retrieval client is c and whose corpus is
// the fully-qualified testCorpus (so project/location resolve).
func toolWithClient(c ragClient) *ragRetrieval {
	return &ragRetrieval{
		name:        "rag_retrieval",
		description: "d",
		store: &genai.VertexRAGStore{
			RAGResources: []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus}},
		},
		newClient: func(ctx context.Context, location string) (ragClient, error) { return c, nil },
	}
}

func TestRun_Success(t *testing.T) {
	topK := int32(7)
	vdt := 0.4
	fake := &fakeRagClient{
		resp: &aiplatformpb.RetrieveContextsResponse{
			Contexts: &aiplatformpb.RagContexts{
				Contexts: []*aiplatformpb.RagContexts_Context{
					{Text: "first"},
					{Text: ""}, // empty text is dropped
					{Text: "second"},
				},
			},
		},
	}
	rt := &ragRetrieval{
		name:        "rag_retrieval",
		description: "d",
		store: &genai.VertexRAGStore{
			RAGResources:            []*genai.VertexRAGStoreRAGResource{{RAGCorpus: testCorpus, RAGFileIDs: []string{"f1"}}},
			SimilarityTopK:          &topK,
			VectorDistanceThreshold: &vdt,
		},
		newClient: func(ctx context.Context, location string) (ragClient, error) { return fake, nil },
	}

	got, err := rt.Run(nil, map[string]any{"query": "what is adk"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	want := map[string]any{"result": []string{"first", "second"}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
	if !fake.closed {
		t.Error("client was not closed")
	}

	req := fake.gotReq
	if req == nil {
		t.Fatal("RetrieveContexts was not called")
	}
	if want := "projects/test-project/locations/us-central1"; req.Parent != want {
		t.Errorf("Parent = %q, want %q", req.Parent, want)
	}
	if got := req.Query.GetText(); got != "what is adk" {
		t.Errorf("query text = %q, want %q", got, "what is adk")
	}
	// SimilarityTopK belongs on the query, not the store.
	if req.Query.SimilarityTopK != topK {
		t.Errorf("query SimilarityTopK = %d, want %d", req.Query.SimilarityTopK, topK)
	}
	store := req.GetVertexRagStore()
	if store == nil {
		t.Fatal("request has no VertexRagStore")
	}
	if store.VectorDistanceThreshold == nil || *store.VectorDistanceThreshold != vdt {
		t.Errorf("store VectorDistanceThreshold = %v, want %v", store.VectorDistanceThreshold, vdt)
	}
	if len(store.RagResources) != 1 || store.RagResources[0].RagCorpus != testCorpus {
		t.Fatalf("store RagResources = %+v, want one with corpus %q", store.RagResources, testCorpus)
	}
	if diff := cmp.Diff([]string{"f1"}, store.RagResources[0].RagFileIds); diff != "" {
		t.Errorf("RagFileIds mismatch (-want +got):\n%s", diff)
	}
}

func TestRun_NoMatch(t *testing.T) {
	fake := &fakeRagClient{resp: &aiplatformpb.RetrieveContextsResponse{}}
	got, err := toolWithClient(fake).Run(nil, map[string]any{"query": "q"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	want := map[string]any{"result": noMatchMessage}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
}

func TestRun_RetrieveError(t *testing.T) {
	fake := &fakeRagClient{err: errors.New("backend unavailable")}
	if _, err := toolWithClient(fake).Run(nil, map[string]any{"query": "q"}); err == nil {
		t.Fatal("Run() error = nil, want non-nil")
	}
}

func TestRun_NewClientError(t *testing.T) {
	rt := &ragRetrieval{
		store:     &genai.VertexRAGStore{RAGCorpora: []string{testCorpus}},
		newClient: func(ctx context.Context, location string) (ragClient, error) { return nil, errors.New("no creds") },
	}
	if _, err := rt.Run(nil, map[string]any{"query": "q"}); err == nil {
		t.Fatal("Run() error = nil, want non-nil")
	}
}

func TestRun_BadArgs(t *testing.T) {
	tests := []struct {
		name string
		args any
	}{
		{"not a map", "query"},
		{"missing query", map[string]any{"other": "x"}},
		{"query not a string", map[string]any{"query": 42}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeRagClient{resp: &aiplatformpb.RetrieveContextsResponse{}}
			if _, err := toolWithClient(fake).Run(nil, tt.args); err == nil {
				t.Fatal("Run() error = nil, want non-nil")
			}
			if fake.gotReq != nil {
				t.Error("RetrieveContexts should not be called on bad args")
			}
		})
	}
}

func TestResolveProjectLocation(t *testing.T) {
	tests := []struct {
		name        string
		rt          *ragRetrieval
		wantProject string
		wantLoc     string
		wantErr     bool
	}{
		{
			name:        "from corpus",
			rt:          &ragRetrieval{store: &genai.VertexRAGStore{RAGCorpora: []string{testCorpus}}},
			wantProject: "test-project",
			wantLoc:     "us-central1",
		},
		{
			name:        "from config when corpus is bare",
			rt:          &ragRetrieval{store: &genai.VertexRAGStore{RAGCorpora: []string{"bare-corpus"}}, project: "p2", location: "l2"},
			wantProject: "p2",
			wantLoc:     "l2",
		},
		{
			name:        "corpus wins over config",
			rt:          &ragRetrieval{store: &genai.VertexRAGStore{RAGCorpora: []string{testCorpus}}, project: "p2", location: "l2"},
			wantProject: "test-project",
			wantLoc:     "us-central1",
		},
		{
			name:    "unresolved",
			rt:      &ragRetrieval{store: &genai.VertexRAGStore{RAGCorpora: []string{"bare-corpus"}}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			project, loc, err := tt.rt.resolveProjectLocation()
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolveProjectLocation() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if project != tt.wantProject || loc != tt.wantLoc {
				t.Errorf("resolveProjectLocation() = (%q, %q), want (%q, %q)", project, loc, tt.wantProject, tt.wantLoc)
			}
		})
	}
}

func TestParseCorpusProjectLocation(t *testing.T) {
	tests := []struct {
		name   string
		in     string
		wantP  string
		wantL  string
		wantOK bool
	}{
		{"fully qualified corpus", testCorpus, "test-project", "us-central1", true},
		{"minimal location path", "projects/p/locations/l", "p", "l", true},
		{"bare name", "bare-corpus", "", "", false},
		{"wrong prefix", "foo/p/locations/l", "", "", false},
		{"too few segments", "projects/p/locations", "", "", false},
		{"empty", "", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, l, ok := parseCorpusProjectLocation(tt.in)
			if p != tt.wantP || l != tt.wantL || ok != tt.wantOK {
				t.Errorf("parseCorpusProjectLocation(%q) = (%q, %q, %v), want (%q, %q, %v)", tt.in, p, l, ok, tt.wantP, tt.wantL, tt.wantOK)
			}
		})
	}
}
