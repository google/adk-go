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
	"encoding/json"
	"errors"
	"io"
	"iter"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type uploadedFile struct {
	corpusName  string
	displayName string
	content     string
}

type fakeRAGBackend struct {
	uploads       []uploadedFile
	uploadFn      func(ctx context.Context, corpusName, displayName string, content io.Reader) error
	listCalls     []*aiplatformpb.ListRagFilesRequest
	listFn        func(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error)
	retrieveCalls []*aiplatformpb.RetrieveContextsRequest
	retrieveFn    func(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error)
}

func (f *fakeRAGBackend) uploadFile(ctx context.Context, corpusName, displayName string, content io.Reader) error {
	if f.uploadFn != nil {
		return f.uploadFn(ctx, corpusName, displayName, content)
	}
	b, err := io.ReadAll(content)
	if err != nil {
		return err
	}
	f.uploads = append(f.uploads, uploadedFile{
		corpusName:  corpusName,
		displayName: displayName,
		content:     string(b),
	})
	return nil
}

func (f *fakeRAGBackend) listFiles(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
	f.listCalls = append(f.listCalls, req)
	if f.listFn != nil {
		return f.listFn(ctx, req)
	}
	return nil, "", errors.New("cannot list files")
}

func (f *fakeRAGBackend) retrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
	f.retrieveCalls = append(f.retrieveCalls, req)
	if f.retrieveFn != nil {
		return f.retrieveFn(ctx, req)
	}
	return &aiplatformpb.RetrieveContextsResponse{}, nil
}

func ragContext(displayName, text string) *aiplatformpb.RagContexts_Context {
	line, err := json.Marshal(transcriptEvent{
		Author:    "user",
		Timestamp: 1,
		Text:      text,
	})
	if err != nil {
		panic(err)
	}
	return &aiplatformpb.RagContexts_Context{
		SourceDisplayName: displayName,
		Text:              string(line),
	}
}

func ragFile(fileID, displayName string) *aiplatformpb.RagFile {
	return &aiplatformpb.RagFile{
		Name:        "projects/test-project/locations/us-central1/ragCorpora/1/ragFiles/" + fileID,
		DisplayName: displayName,
	}
}

func noEnv(string) string { return "" }

// newTestService resolves cfg without reading the process environment, so
// tests do not depend on GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION.
func newTestService(cfg Config, backend ragBackend) *service {
	return newServiceWithBackend(resolveConfig(cfg, noEnv), backend)
}

func memoryTexts(resp *memory.SearchResponse) []string {
	var texts []string
	for _, m := range resp.Memories {
		if m.Content != nil && len(m.Content.Parts) > 0 && m.Content.Parts[0] != nil {
			texts = append(texts, m.Content.Parts[0].Text)
		}
	}
	return texts
}

type fakeSession struct {
	appName   string
	userID    string
	sessionID string
	events    []*session.Event
}

func (s *fakeSession) ID() string             { return s.sessionID }
func (s *fakeSession) AppName() string        { return s.appName }
func (s *fakeSession) UserID() string         { return s.userID }
func (s *fakeSession) Events() session.Events { return s }
func (s *fakeSession) All() iter.Seq[*session.Event] {
	return slices.Values(s.events)
}
func (s *fakeSession) Len() int                  { return len(s.events) }
func (s *fakeSession) At(i int) *session.Event   { return s.events[i] }
func (s *fakeSession) State() session.State      { panic("not implemented") }
func (s *fakeSession) LastUpdateTime() time.Time { panic("not implemented") }

func defaultSession() session.Session {
	return &fakeSession{
		appName:   "demo.app",
		userID:    "alice.smith",
		sessionID: "session.secret",
		events: []*session.Event{
			{
				ID:        "event-1",
				Author:    "user",
				Timestamp: time.Unix(1, 0),
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "sensitive memory"}},
					},
				},
			},
		},
	}
}

func TestSearchMemoryRetrievalConfig(t *testing.T) {
	// reference: test_search_memory_forwards_similarity_top_k
	for _, tt := range []struct {
		name          string
		topK          int32
		threshold     float64
		wantTopK      int32
		wantThreshold float64
	}{
		{
			name:          "forwards configured similarity top k and default threshold",
			topK:          7,
			threshold:     0,
			wantTopK:      7,
			wantThreshold: 10,
		},
		{
			name:          "leaves top k unset when zero and forwards custom threshold",
			topK:          0,
			threshold:     0.42,
			wantTopK:      0,
			wantThreshold: 0.42,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeRAGBackend{}
			svc := newTestService(Config{
				RAGCorpus:               "unused",
				SimilarityTopK:          tt.topK,
				VectorDistanceThreshold: tt.threshold,
			}, fake)

			_, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
				AppName: "demo",
				UserID:  "alice",
				Query:   "memory",
			})
			if err != nil {
				t.Fatalf("SearchMemory() error = %v", err)
			}

			if len(fake.retrieveCalls) != 1 {
				t.Fatalf("retrieveContexts calls = %d, want 1", len(fake.retrieveCalls))
			}
			call := fake.retrieveCalls[0]
			query := call.GetQuery()
			if query.GetText() != "memory" {
				t.Errorf("Query.Text = %q, want %q", query.GetText(), "memory")
			}
			if got := query.GetRagRetrievalConfig().GetTopK(); got != tt.wantTopK {
				t.Errorf("RagRetrievalConfig.TopK = %d, want %d", got, tt.wantTopK)
			}
			if got := query.GetRagRetrievalConfig().GetFilter().GetVectorDistanceThreshold(); got != tt.wantThreshold {
				t.Errorf("RagRetrievalConfig.Filter.VectorDistanceThreshold = %v, want %v", got, tt.wantThreshold)
			}
			// Ensure deprecated fields remain unset.
			if got := query.GetSimilarityTopK(); got != 0 {
				t.Errorf("deprecated Query.SimilarityTopK = %d, want 0", got)
			}
			if call.GetVertexRagStore().VectorDistanceThreshold != nil {
				t.Errorf("deprecated VertexRagStore.VectorDistanceThreshold = %v, want nil", call.GetVertexRagStore().GetVectorDistanceThreshold())
			}
		})
	}
}

func TestSearchMemoryScoping(t *testing.T) {
	// reference: test_search_memory_scopes_retrieval_to_tenant_files
	t.Run("scopes retrieval to tenant files across pages", func(t *testing.T) {
		page := 0
		fake := &fakeRAGBackend{
			listFn: func(_ context.Context, _ *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
				page++
				switch page {
				case 1:
					return []*aiplatformpb.RagFile{
						ragFile("alice-1", buildSourceDisplayName("demo", "alice", "session-1")),
						ragFile("bob-1", buildSourceDisplayName("demo", "bob", "session-2")),
					}, "page-2", nil
				default:
					return []*aiplatformpb.RagFile{
						ragFile("alice-2", "demo.alice.legacy-session"),
						ragFile("other-app-1", buildSourceDisplayName("other", "alice", "session-3")),
					}, "", nil
				}
			},
			retrieveFn: func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
				return &aiplatformpb.RetrieveContextsResponse{
					Contexts: &aiplatformpb.RagContexts{
						Contexts: []*aiplatformpb.RagContexts_Context{
							ragContext(buildSourceDisplayName("demo", "alice", "session-1"), "ALICE_MEMORY"),
						},
					},
				}, nil
			},
		}
		svc := newTestService(Config{
			RAGCorpus:      "corpus",
			SimilarityTopK: 5,
		}, fake)

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "memory",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if diff := cmp.Diff([]string{"ALICE_MEMORY"}, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}
		if len(fake.listCalls) != 2 {
			t.Fatalf("listFiles calls = %d, want 2", len(fake.listCalls))
		}
		if fake.listCalls[0].GetPageSize() != 100 {
			t.Errorf("page 1 PageSize = %d, want 100", fake.listCalls[0].GetPageSize())
		}
		if fake.listCalls[1].GetPageToken() != "page-2" {
			t.Errorf("page 2 PageToken = %q, want %q", fake.listCalls[1].GetPageToken(), "page-2")
		}
		if len(fake.retrieveCalls) != 1 {
			t.Fatalf("retrieveContexts calls = %d, want 1", len(fake.retrieveCalls))
		}
		resources := fake.retrieveCalls[0].GetVertexRagStore().GetRagResources()
		if len(resources) != 1 {
			t.Fatalf("RagResources len = %d, want 1", len(resources))
		}
		if resources[0].GetRagCorpus() != "corpus" {
			t.Errorf("RagCorpus = %q, want %q", resources[0].GetRagCorpus(), "corpus")
		}
		if diff := cmp.Diff([]string{"alice-1", "alice-2"}, resources[0].GetRagFileIds()); diff != "" {
			t.Errorf("RagFileIds mismatch (-want +got):\n%s", diff)
		}
		if got := fake.retrieveCalls[0].GetQuery().GetRagRetrievalConfig().GetTopK(); got != 5 {
			t.Errorf("RagRetrievalConfig.TopK = %d, want 5", got)
		}
	})

	// reference: test_search_memory_skips_retrieval_without_tenant_files
	t.Run("skips retrieval when tenant has no files", func(t *testing.T) {
		fake := &fakeRAGBackend{
			listFn: func(_ context.Context, _ *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
				return []*aiplatformpb.RagFile{
					ragFile("bob-1", buildSourceDisplayName("demo", "bob", "session-2")),
				}, "", nil
			},
		}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "memory",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(resp.Memories) != 0 {
			t.Errorf("Memories len = %d, want 0", len(resp.Memories))
		}
		if len(fake.retrieveCalls) != 0 {
			t.Errorf("retrieveContexts calls = %d, want 0", len(fake.retrieveCalls))
		}
	})

	// reference: test_search_memory_retrieves_unscoped_when_listing_fails
	t.Run("retrieves unscoped and filters response when listing fails", func(t *testing.T) {
		fake := &fakeRAGBackend{
			retrieveFn: func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
				return &aiplatformpb.RetrieveContextsResponse{
					Contexts: &aiplatformpb.RagContexts{
						Contexts: []*aiplatformpb.RagContexts_Context{
							ragContext(buildSourceDisplayName("demo", "alice", "session-1"), "ALICE_MEMORY"),
							ragContext(buildSourceDisplayName("demo", "bob", "session-2"), "BOB_MEMORY"),
							ragContext(buildSourceDisplayName("other", "alice", "session-3"), "OTHER_APP_MEMORY"),
						},
					},
				}, nil
			},
		}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "memory",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if diff := cmp.Diff([]string{"ALICE_MEMORY"}, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}
		resources := fake.retrieveCalls[0].GetVertexRagStore().GetRagResources()
		if len(resources[0].GetRagFileIds()) != 0 {
			t.Errorf("RagFileIds = %v, want empty", resources[0].GetRagFileIds())
		}
	})

	// reference: test_search_memory_retrieves_unscoped_when_corpus_is_too_large
	t.Run("retrieves unscoped when corpus exceeds ten pages", func(t *testing.T) {
		fake := &fakeRAGBackend{
			listFn: func(_ context.Context, _ *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
				return []*aiplatformpb.RagFile{
					ragFile("alice-1", buildSourceDisplayName("demo", "alice", "session-1")),
				}, "another-page", nil
			},
		}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		_, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "memory",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if len(fake.listCalls) != maxRAGFilePages {
			t.Errorf("listFiles calls = %d, want %d", len(fake.listCalls), maxRAGFilePages)
		}
		resources := fake.retrieveCalls[0].GetVertexRagStore().GetRagResources()
		if len(resources[0].GetRagFileIds()) != 0 {
			t.Errorf("RagFileIds = %v, want empty", resources[0].GetRagFileIds())
		}
	})

	// reference: test_search_memory_rejects_ambiguous_legacy_display_names
	t.Run("rejects ambiguous legacy display names", func(t *testing.T) {
		fake := &fakeRAGBackend{
			retrieveFn: func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
				return &aiplatformpb.RetrieveContextsResponse{
					Contexts: &aiplatformpb.RagContexts{
						Contexts: []*aiplatformpb.RagContexts_Context{
							ragContext("demo.alice.smith.session_secret", "SECRET_FROM_ALICE_SMITH"),
							ragContext(buildSourceDisplayName("demo", "alice", "session_ok"), "NORMAL_ALICE_MEMORY"),
							ragContext("demo.alice.legacy_session", "LEGACY_ALICE_MEMORY"),
							ragContext("demo.bob.session_other", "BOB_MEMORY"),
							ragContext("adk-memory-v1.not-three-parts", "MALFORMED_PREFIX"),
							ragContext("adk-memory-v1.!!!.YWxpY2U.c2Vzcw", "INVALID_BASE64"),
							ragContext("adk-memory-v1."+base64RawInvalidUTF8()+".YWxpY2U.c2Vzcw", "INVALID_UTF8"),
						},
					},
				}, nil
			},
		}
		svc := newTestService(Config{RAGCorpus: "unused"}, fake)

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "secret",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		want := []string{"NORMAL_ALICE_MEMORY", "LEGACY_ALICE_MEMORY"}
		if diff := cmp.Diff(want, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_add_and_search_memory_uses_unambiguous_display_names
	t.Run("round trips dotted identifiers with unambiguous display names", func(t *testing.T) {
		fake := &fakeRAGBackend{}
		svc := newTestService(Config{RAGCorpus: "unused"}, fake)

		if err := svc.AddSessionToMemory(t.Context(), defaultSession()); err != nil {
			t.Fatalf("AddSessionToMemory() error = %v", err)
		}
		if len(fake.uploads) != 1 {
			t.Fatalf("uploads = %d, want 1", len(fake.uploads))
		}
		displayName := fake.uploads[0].displayName
		if !strings.HasPrefix(displayName, sourceDisplayNamePrefix) {
			t.Errorf("displayName = %q, want prefix %q", displayName, sourceDisplayNamePrefix)
		}
		if displayName == "demo.app.alice.smith.session.secret" {
			t.Errorf("displayName = %q, must not use raw dot concatenation", displayName)
		}

		fake.retrieveFn = func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
			return &aiplatformpb.RetrieveContextsResponse{
				Contexts: &aiplatformpb.RagContexts{
					Contexts: []*aiplatformpb.RagContexts_Context{
						{
							SourceDisplayName: displayName,
							Text:              fake.uploads[0].content,
						},
					},
				},
			}, nil
		}

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo.app",
			UserID:  "alice.smith",
			Query:   "sensitive",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if diff := cmp.Diff([]string{"sensitive memory"}, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns error when request is nil or retrieve fails", func(t *testing.T) {
		fake := &fakeRAGBackend{
			retrieveFn: func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
				return nil, errors.New("retrieve boom")
			},
		}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		if _, err := svc.SearchMemory(t.Context(), nil); err == nil {
			t.Fatal("SearchMemory(nil) succeeded, want error")
		}
		_, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "memory",
		})
		if err == nil || !strings.Contains(err.Error(), "retrieve boom") {
			t.Fatalf("SearchMemory() error = %v, want retrieve boom", err)
		}
	})
}

func base64RawInvalidUTF8() string {
	// 0xff is invalid UTF-8; base64url of []byte{0xff} without padding is "_w".
	return "_w"
}

func TestAddSessionToMemory(t *testing.T) {
	// reference: test_add_session_cleans_temp_file_after_partial_upload_failure
	t.Run("stops at first upload failure across multiple corpora", func(t *testing.T) {
		var attempted []string
		fake := &fakeRAGBackend{
			uploadFn: func(_ context.Context, corpusName, _ string, _ io.Reader) error {
				attempted = append(attempted, corpusName)
				if corpusName == "second" {
					return errors.New("upload failed")
				}
				return nil
			},
		}
		svc := newTestService(Config{RAGCorpus: "first"}, fake)
		svc.ragResources = []*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource{
			{RagCorpus: "first"},
			{RagCorpus: "second"},
			{RagCorpus: "third"},
		}

		err := svc.AddSessionToMemory(t.Context(), defaultSession())
		if err == nil || !strings.Contains(err.Error(), "upload failed") {
			t.Fatalf("AddSessionToMemory() error = %v, want upload failed", err)
		}
		if diff := cmp.Diff([]string{"first", "second"}, attempted); diff != "" {
			t.Errorf("attempted corpora mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_add_session_leaves_no_temp_file_when_corpus_missing
	t.Run("returns error when corpus is missing", func(t *testing.T) {
		fake := &fakeRAGBackend{}
		svc := newTestService(Config{RAGCorpus: ""}, fake)

		err := svc.AddSessionToMemory(t.Context(), defaultSession())
		if err == nil || !strings.Contains(err.Error(), "rag_corpus must be set") {
			t.Fatalf("AddSessionToMemory() error = %v, want rag_corpus must be set", err)
		}
		if len(fake.uploads) != 0 {
			t.Errorf("uploads = %d, want 0", len(fake.uploads))
		}
	})

	t.Run("writes HTML characters without escaping", func(t *testing.T) {
		fake := &fakeRAGBackend{}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)
		sess := &fakeSession{
			appName:   "demo",
			userID:    "alice",
			sessionID: "session-1",
			events: []*session.Event{{
				Author:    "user",
				Timestamp: time.Unix(1, 0),
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{Parts: []*genai.Part{{Text: "<b>salt & pepper</b>"}}},
				},
			}},
		}

		if err := svc.AddSessionToMemory(t.Context(), sess); err != nil {
			t.Fatalf("AddSessionToMemory() error = %v", err)
		}
		want := `{"author":"user","timestamp":1,"text":"<b>salt & pepper</b>"}`
		if len(fake.uploads) != 1 || fake.uploads[0].content != want {
			t.Errorf("uploaded transcript = %+v, want content %q", fake.uploads, want)
		}
	})

	t.Run("returns error when session is nil", func(t *testing.T) {
		fake := &fakeRAGBackend{}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		if err := svc.AddSessionToMemory(t.Context(), nil); err == nil {
			t.Fatal("AddSessionToMemory(nil) succeeded, want error")
		}
	})

	// reference: test_add_session_to_memory_skips_thought_parts
	t.Run("skips thought parts and formats multiline multi-part text", func(t *testing.T) {
		fake := &fakeRAGBackend{}
		svc := newTestService(Config{RAGCorpus: "corpus"}, fake)

		sess := &fakeSession{
			appName:   "demo",
			userID:    "alice",
			sessionID: "session-1",
			events: []*session.Event{
				{
					ID:        "thought-only",
					Author:    "model",
					Timestamp: time.Unix(1, 0),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{Text: "Internal thought only", Thought: true},
							},
						},
					},
				},
				{
					ID:        "mixed",
					Author:    "model",
					Timestamp: time.Unix(2, 500_000_000),
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{Text: "Consider a zeppelin.", Thought: true},
								{Text: "Take the train\nto Paris"},
								{Text: "Bring a coat"},
							},
						},
					},
				},
				{
					ID:     "zero-time",
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "user",
							Parts: []*genai.Part{{Text: "Untimed note"}},
						},
					},
				},
			},
		}

		if err := svc.AddSessionToMemory(t.Context(), sess); err != nil {
			t.Fatalf("AddSessionToMemory() error = %v", err)
		}
		if len(fake.uploads) != 1 {
			t.Fatalf("uploads = %d, want 1", len(fake.uploads))
		}
		lines := strings.Split(fake.uploads[0].content, "\n")
		if len(lines) != 2 {
			t.Fatalf("uploaded lines = %d (%q), want 2", len(lines), fake.uploads[0].content)
		}
		var first, second transcriptEvent
		if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
			t.Fatalf("unmarshal line 0: %v", err)
		}
		if first.Author != "model" || first.Timestamp != 2.5 || first.Text != "Take the train to Paris.Bring a coat" {
			t.Errorf("line 0 = %+v, want author=model timestamp=2.5 text=%q", first, "Take the train to Paris.Bring a coat")
		}
		if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
			t.Fatalf("unmarshal line 1: %v", err)
		}
		if second.Author != "user" || second.Timestamp != 0 || second.Text != "Untimed note" {
			t.Errorf("line 1 = %+v, want author=user timestamp=0 text=%q", second, "Untimed note")
		}

		fake.retrieveFn = func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
			return &aiplatformpb.RetrieveContextsResponse{
				Contexts: &aiplatformpb.RagContexts{
					Contexts: []*aiplatformpb.RagContexts_Context{
						{
							SourceDisplayName: fake.uploads[0].displayName,
							Text:              fake.uploads[0].content,
						},
					},
				},
			}, nil
		}

		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo",
			UserID:  "alice",
			Query:   "Paris",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		wantTexts := []string{"Untimed note", "Take the train to Paris.Bring a coat"}
		if diff := cmp.Diff(wantTexts, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}
		if !resp.Memories[1].Timestamp.Equal(time.Unix(2, 500_000_000)) {
			t.Errorf("Memories[1].Timestamp = %v, want %v", resp.Memories[1].Timestamp, time.Unix(2, 500_000_000))
		}
	})
}

func TestSearchMemoryMergesOverlappingContextEvents(t *testing.T) {
	fake := &fakeRAGBackend{
		retrieveFn: func(_ context.Context, _ *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
			sess1 := buildSourceDisplayName("demo", "alice", "sess-1")
			sess2 := buildSourceDisplayName("demo", "alice", "sess-2")
			return &aiplatformpb.RetrieveContextsResponse{
				Contexts: &aiplatformpb.RagContexts{
					Contexts: []*aiplatformpb.RagContexts_Context{
						{
							// Context 1 for sess-1: timestamps 30 and 10 (out of order).
							SourceDisplayName: sess1,
							Text: strings.Join([]string{
								`{"author":"user","timestamp":30,"text":"s1-t30"}`,
								`{"author":"user","timestamp":10,"text":"s1-t10"}`,
							}, "\n"),
						},
						{
							// Context for sess-2: timestamp 5.
							SourceDisplayName: sess2,
							Text:              `{"author":"model","timestamp":"5.5","text":"s2-t5.5"}`,
						},
						{
							// Context 2 for sess-1: disjoint from Context 1 directly, but overlaps via Context 3 (timestamp 20).
							SourceDisplayName: sess1,
							Text:              `{"author":"model","timestamp":20,"text":"s1-t20-later"}`,
						},
						{
							// Context 3 for sess-1: links Context 1 (timestamp 30) and Context 2 (timestamp 20).
							SourceDisplayName: sess1,
							Text: strings.Join([]string{
								`{"author":"user","timestamp":30,"text":"s1-t30-dup"}`,
								`{"author":"model","timestamp":20,"text":"s1-t20"}`,
							}, "\n"),
						},
						{
							// Context 4 for sess-1: disjoint timestamp 50, plus malformed JSON lines skipped.
							SourceDisplayName: sess1,
							Text: strings.Join([]string{
								`not-json`,
								`{"author":123,"timestamp":1,"text":"bad-author"}`,
								`{"author":"user","timestamp":"NaN","text":"s1-nan"}`,
								`{"author":"user","timestamp":"Infinity","text":"s1-inf"}`,
								`{"author":"user","timestamp":true,"text":"s1-bool-ts"}`,
								`{"author":"user","timestamp":1,"text":456}`,
								`{"author":"user","timestamp":50,"text":"s1-t50"}`,
							}, "\n"),
						},
					},
				},
			}, nil
		},
	}

	svc := newTestService(Config{RAGCorpus: "corpus"}, fake)
	resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
		AppName: "demo",
		UserID:  "alice",
		Query:   "anything",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}

	// sess-1 is seen first, with its overlapping contexts (1, 2, 3) merged into
	// [t10, t20, t30] and its disjoint context (4) kept as [t50]; sess-2 follows
	// with [t5.5].
	want := []string{"s1-t10", "s1-t20", "s1-t30", "s1-t50", "s2-t5.5"}
	if diff := cmp.Diff(want, memoryTexts(resp)); diff != "" {
		t.Errorf("SearchMemory() merged texts mismatch (-want +got):\n%s", diff)
	}
}

func TestMergeEventLists(t *testing.T) {
	ev := func(author string, ts float64) parsedEvent {
		return parsedEvent{author: author, timestampSeconds: ts}
	}
	authors := func(lists [][]parsedEvent) [][]string {
		var out [][]string
		for _, l := range lists {
			var names []string
			for _, e := range l {
				names = append(names, e.author)
			}
			out = append(out, names)
		}
		return out
	}

	for _, tt := range []struct {
		name  string
		input [][]parsedEvent
		want  [][]string
	}{
		{
			name:  "keeps every new event that shares a timestamp",
			input: [][]parsedEvent{{ev("a", 1)}, {ev("a", 1), ev("x", 0), ev("y", 0)}},
			want:  [][]string{{"a", "x", "y"}},
		},
		{
			name:  "drops events whose timestamp is already present",
			input: [][]parsedEvent{{ev("a", 1), ev("b", 2)}, {ev("b-dup", 2), ev("c", 3)}},
			want:  [][]string{{"a", "b", "c"}},
		},
		{
			// The last list joins first and brings timestamp 2, so the earlier
			// "b" list then overlaps and adds nothing new.
			name:  "merges transitively and keeps disjoint lists apart",
			input: [][]parsedEvent{{ev("a", 1)}, {ev("z", 9)}, {ev("b", 2)}, {ev("a-dup", 1), ev("b-dup", 2)}},
			want:  [][]string{{"a", "b-dup"}, {"z"}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if diff := cmp.Diff(tt.want, authors(mergeEventLists(tt.input))); diff != "" {
				t.Errorf("mergeEventLists() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestResolveConfig(t *testing.T) {
	envFrom := func(m map[string]string) func(string) string {
		return func(k string) string { return m[k] }
	}

	t.Run("resolves project and location from env and expands bare corpus id", func(t *testing.T) {
		env := envFrom(map[string]string{
			"GOOGLE_CLOUD_PROJECT":  "env-proj",
			"GOOGLE_CLOUD_LOCATION": "us-east1",
		})
		got := resolveConfig(Config{RAGCorpus: "12345"}, env)
		want := Config{
			RAGCorpus:               "projects/env-proj/locations/us-east1/ragCorpora/12345",
			Project:                 "env-proj",
			Location:                "us-east1",
			VectorDistanceThreshold: 10,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("resolveConfig() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("extracts project and location from full corpus resource name", func(t *testing.T) {
		fullCorpus := "projects/corpus-proj/locations/europe-west1/ragCorpora/67890"
		got := resolveConfig(Config{RAGCorpus: fullCorpus}, noEnv)
		want := Config{
			RAGCorpus:               fullCorpus,
			Project:                 "corpus-proj",
			Location:                "europe-west1",
			VectorDistanceThreshold: 10,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("resolveConfig() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("explicit config overrides env and corpus prefix", func(t *testing.T) {
		env := envFrom(map[string]string{
			"GOOGLE_CLOUD_PROJECT":  "env-proj",
			"GOOGLE_CLOUD_LOCATION": "env-loc",
		})
		fullCorpus := "projects/corpus-proj/locations/corpus-loc/ragCorpora/67890"
		in := Config{
			RAGCorpus:               fullCorpus,
			Project:                 "cfg-proj",
			Location:                "cfg-loc",
			VectorDistanceThreshold: 2.5,
		}
		if diff := cmp.Diff(in, resolveConfig(in, env)); diff != "" {
			t.Errorf("resolveConfig() mismatch (-want +got):\n%s", diff)
		}
	})
}

const (
	testCorpus     = "projects/test-proj/locations/us-central1/ragCorpora/corpus-1"
	uploadPath     = "/upload/v1beta1/" + testCorpus + "/ragFiles:upload"
	listFilesPath  = "/v1beta1/" + testCorpus + "/ragFiles"
	retrievePath   = "/v1beta1/projects/test-proj/locations/us-central1:retrieveContexts"
	testUploadName = "adk-memory-v1.ZGVtby5hcHA.YWxpY2Uuc21pdGg.c2Vzc2lvbi5zZWNyZXQ"
)

// fakeVertexServer stands in for the Vertex AI HTTP API. It records the
// upload it receives and serves it back from list and retrieve.
type fakeVertexServer struct {
	t                  *testing.T
	uploadReply        string
	emptyPageNextToken string

	mu          sync.Mutex
	uploads     []uploadedFile
	listCount   int
	retrieveReq *aiplatformpb.RetrieveContextsRequest
}

func (f *fakeVertexServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodPost && r.URL.Path == uploadPath:
		f.serveUpload(w, r)
	case r.Method == http.MethodGet && r.URL.Path == listFilesPath:
		f.listCount++
		if f.emptyPageNextToken != "" {
			nextToken := f.emptyPageNextToken
			if f.listCount > maxRAGFilePages {
				nextToken = ""
			}
			f.writeProto(w, &aiplatformpb.ListRagFilesResponse{NextPageToken: nextToken})
			return
		}
		var files []*aiplatformpb.RagFile
		for i, u := range f.uploads {
			files = append(files, &aiplatformpb.RagFile{
				Name:        testCorpus + "/ragFiles/file-" + string(rune('a'+i)),
				DisplayName: u.displayName,
			})
		}
		f.writeProto(w, &aiplatformpb.ListRagFilesResponse{RagFiles: files})
	case r.Method == http.MethodPost && r.URL.Path == retrievePath:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			f.t.Errorf("read retrieve body: %v", err)
		}
		req := &aiplatformpb.RetrieveContextsRequest{}
		if err := protojson.Unmarshal(body, req); err != nil {
			f.t.Errorf("decode retrieve body: %v", err)
		}
		f.retrieveReq = req
		var contexts []*aiplatformpb.RagContexts_Context
		for _, u := range f.uploads {
			contexts = append(contexts, &aiplatformpb.RagContexts_Context{
				SourceDisplayName: u.displayName,
				Text:              u.content,
			})
		}
		f.writeProto(w, &aiplatformpb.RetrieveContextsResponse{
			Contexts: &aiplatformpb.RagContexts{Contexts: contexts},
		})
	default:
		f.t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		http.NotFound(w, r)
	}
}

func (f *fakeVertexServer) serveUpload(w http.ResponseWriter, r *http.Request) {
	if got := r.Header.Get("X-Goog-Upload-Protocol"); got != "multipart" {
		f.t.Errorf("X-Goog-Upload-Protocol = %q, want %q", got, "multipart")
	}
	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		f.t.Errorf("Content-Type = %q, want multipart/form-data", r.Header.Get("Content-Type"))
		http.Error(w, "No file found in request.", http.StatusBadRequest)
		return
	}
	var displayName, content, fileName string
	var partNames []string
	mr := multipart.NewReader(r.Body, params["boundary"])
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			f.t.Errorf("read multipart body: %v", err)
			return
		}
		b, err := io.ReadAll(part)
		if err != nil {
			f.t.Errorf("read part %q: %v", part.FormName(), err)
			return
		}
		partNames = append(partNames, part.FormName())
		switch part.FormName() {
		case "metadata":
			var md struct {
				RagFile struct {
					DisplayName string `json:"display_name"`
				} `json:"rag_file"`
			}
			if err := json.Unmarshal(b, &md); err != nil {
				f.t.Errorf("decode metadata part %q: %v", b, err)
			}
			displayName = md.RagFile.DisplayName
		case "file":
			fileName = part.FileName()
			content = string(b)
		}
	}
	if diff := cmp.Diff([]string{"metadata", "file"}, partNames); diff != "" {
		f.t.Errorf("multipart parts mismatch (-want +got):\n%s", diff)
	}
	if !strings.HasSuffix(fileName, ".txt") {
		f.t.Errorf("file part name = %q, want a .txt file", fileName)
	}
	f.uploads = append(f.uploads, uploadedFile{
		corpusName:  testCorpus,
		displayName: displayName,
		content:     content,
	})
	reply := f.uploadReply
	if reply == "" {
		reply = `{"ragFile":{"name":"` + testCorpus + `/ragFiles/file-a"}}`
	}
	_, _ = w.Write([]byte(reply))
}

func (f *fakeVertexServer) writeProto(w http.ResponseWriter, m proto.Message) {
	b, err := protojson.Marshal(m)
	if err != nil {
		f.t.Errorf("marshal response: %v", err)
	}
	_, _ = w.Write(b)
}

func TestNewService(t *testing.T) {
	cfg := Config{
		RAGCorpus: "corpus-1",
		Project:   "test-proj",
		Location:  "us-central1",
	}

	t.Run("rejects nil client option", func(t *testing.T) {
		_, err := NewService(t.Context(), cfg, nil)
		if err == nil {
			t.Fatal("NewService() with nil option succeeded, want error")
		}
	})

	for _, tt := range []struct {
		name string
		tls  bool
		opts func(srv *httptest.Server) []option.ClientOption
	}{
		{
			name: "adds and searches memory with endpoint option",
			opts: func(srv *httptest.Server) []option.ClientOption {
				return []option.ClientOption{option.WithoutAuthentication(), option.WithEndpoint(srv.URL)}
			},
		},
		{
			name: "adds and searches memory with HTTP client option",
			opts: func(srv *httptest.Server) []option.ClientOption {
				return []option.ClientOption{option.WithHTTPClient(srv.Client()), option.WithEndpoint(srv.URL)}
			},
		},
		{
			name: "adds and searches memory with host:port endpoint option",
			tls:  true,
			opts: func(srv *httptest.Server) []option.ClientOption {
				return []option.ClientOption{option.WithHTTPClient(srv.Client()), option.WithEndpoint(srv.Listener.Addr().String())}
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeVertexServer{t: t}
			var srv *httptest.Server
			if tt.tls {
				srv = httptest.NewTLSServer(fake)
			} else {
				srv = httptest.NewServer(fake)
			}
			defer srv.Close()

			ctx, cancel := context.WithCancel(t.Context())
			svc, err := NewService(ctx, cfg, tt.opts(srv)...)
			if err != nil {
				t.Fatalf("NewService() error = %v", err)
			}
			// Canceling the constructor context must not break later calls.
			cancel()

			if err := svc.AddSessionToMemory(t.Context(), defaultSession()); err != nil {
				t.Fatalf("AddSessionToMemory() error = %v", err)
			}
			resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
				AppName: "demo.app",
				UserID:  "alice.smith",
				Query:   "sensitive",
			})
			if err != nil {
				t.Fatalf("SearchMemory() error = %v", err)
			}
			if diff := cmp.Diff([]string{"sensitive memory"}, memoryTexts(resp)); diff != "" {
				t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
			}

			fake.mu.Lock()
			defer fake.mu.Unlock()
			wantUploads := []uploadedFile{{
				corpusName:  testCorpus,
				displayName: testUploadName,
				content:     `{"author":"user","timestamp":1,"text":"sensitive memory"}`,
			}}
			if diff := cmp.Diff(wantUploads, fake.uploads, cmp.AllowUnexported(uploadedFile{})); diff != "" {
				t.Errorf("uploads mismatch (-want +got):\n%s", diff)
			}
			if fake.retrieveReq == nil {
				t.Fatal("retrieveContexts was not called")
			}
			resources := fake.retrieveReq.GetVertexRagStore().GetRagResources()
			if len(resources) != 1 || resources[0].GetRagCorpus() != testCorpus {
				t.Fatalf("RagResources = %v, want one resource for %q", resources, testCorpus)
			}
			if diff := cmp.Diff([]string{"file-a"}, resources[0].GetRagFileIds()); diff != "" {
				t.Errorf("RagFileIds mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("surfaces in-band upload error status", func(t *testing.T) {
		fake := &fakeVertexServer{t: t, uploadReply: `{"error":{"code":13,"message":"internal failure"}}`}
		srv := httptest.NewServer(fake)
		defer srv.Close()

		svc, err := NewService(t.Context(), cfg, option.WithoutAuthentication(), option.WithEndpoint(srv.URL))
		if err != nil {
			t.Fatalf("NewService() error = %v", err)
		}
		err = svc.AddSessionToMemory(t.Context(), defaultSession())
		if err == nil || !strings.Contains(err.Error(), "status code 13") {
			t.Fatalf("AddSessionToMemory() error = %v, want status code 13", err)
		}
	})

	t.Run("surfaces HTTP upload error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "No file found in request.", http.StatusBadRequest)
		}))
		defer srv.Close()

		svc, err := NewService(t.Context(), cfg, option.WithoutAuthentication(), option.WithEndpoint(srv.URL))
		if err != nil {
			t.Fatalf("NewService() error = %v", err)
		}
		err = svc.AddSessionToMemory(t.Context(), defaultSession())
		if err == nil || !strings.Contains(err.Error(), "400") {
			t.Fatalf("AddSessionToMemory() error = %v, want HTTP 400", err)
		}
	})

	t.Run("caps list requests when server returns empty pages with next page token", func(t *testing.T) {
		fake := &fakeVertexServer{t: t, emptyPageNextToken: "more"}
		srv := httptest.NewServer(fake)
		defer srv.Close()

		svc, err := NewService(t.Context(), cfg, option.WithoutAuthentication(), option.WithEndpoint(srv.URL))
		if err != nil {
			t.Fatalf("NewService() error = %v", err)
		}
		if err := svc.AddSessionToMemory(t.Context(), defaultSession()); err != nil {
			t.Fatalf("AddSessionToMemory() error = %v", err)
		}
		resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{
			AppName: "demo.app",
			UserID:  "alice.smith",
			Query:   "sensitive",
		})
		if err != nil {
			t.Fatalf("SearchMemory() error = %v", err)
		}
		if diff := cmp.Diff([]string{"sensitive memory"}, memoryTexts(resp)); diff != "" {
			t.Errorf("SearchMemory() texts mismatch (-want +got):\n%s", diff)
		}

		fake.mu.Lock()
		defer fake.mu.Unlock()
		if fake.listCount != maxRAGFilePages {
			t.Errorf("listCount = %d, want %d", fake.listCount, maxRAGFilePages)
		}
		resources := fake.retrieveReq.GetVertexRagStore().GetRagResources()
		if len(resources) != 1 || len(resources[0].GetRagFileIds()) != 0 {
			t.Errorf("RagResources = %v, want one unscoped resource", resources)
		}
	})
}
