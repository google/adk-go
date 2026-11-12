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
	"errors"
	"io"
	"net"
	"testing"
	"time"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/v2/tool/exampletool"
)

const testStoreName = "projects/p/locations/l/exampleStores/s"

type fakeExampleStoreClient struct {
	gotReq *aiplatformpb.SearchExamplesRequest
	resp   *aiplatformpb.SearchExamplesResponse
	err    error
	closed int
}

func (f *fakeExampleStoreClient) SearchExamples(_ context.Context, req *aiplatformpb.SearchExamplesRequest) (*aiplatformpb.SearchExamplesResponse, error) {
	f.gotReq = req
	if f.err != nil {
		return nil, f.err
	}
	return f.resp, nil
}

func (f *fakeExampleStoreClient) Close() error {
	f.closed++
	return nil
}

type fakeExampleStoreServer struct {
	aiplatformpb.UnimplementedExampleStoreServiceServer
	gotReq *aiplatformpb.SearchExamplesRequest
	resp   *aiplatformpb.SearchExamplesResponse
}

func (s *fakeExampleStoreServer) SearchExamples(_ context.Context, req *aiplatformpb.SearchExamplesRequest) (*aiplatformpb.SearchExamplesResponse, error) {
	s.gotReq = req
	return s.resp, nil
}

func makeExpectedContent(role string, parts ...*aiplatformpb.Part) *aiplatformpb.ContentsExample_ExpectedContent {
	return &aiplatformpb.ContentsExample_ExpectedContent{
		Content: &aiplatformpb.Content{
			Role:  role,
			Parts: parts,
		},
	}
}

func makeResult(searchKey string, score float32, expectedContents ...*aiplatformpb.ContentsExample_ExpectedContent) *aiplatformpb.SearchExamplesResponse_SimilarExample {
	return &aiplatformpb.SearchExamplesResponse_SimilarExample{
		SimilarityScore: score,
		Example: &aiplatformpb.Example{
			ExampleType: &aiplatformpb.Example_StoredContentsExample{
				StoredContentsExample: &aiplatformpb.StoredContentsExample{
					SearchKey: searchKey,
					ContentsExample: &aiplatformpb.ContentsExample{
						ExpectedContents: expectedContents,
					},
				},
			},
		},
	}
}

func TestProvider_ConfigValidation(t *testing.T) {
	// reference: test_get_examples_derives_project_and_location_from_the_store_name
	t.Run("extracts location from valid store resource name", func(t *testing.T) {
		for _, storeName := range []string{
			"projects/my-proj_1/locations/us-central1/exampleStores/store.1",
			"projects/example.com:my-proj/locations/us-central1/exampleStores/store.1",
		} {
			loc, err := parseLocation(&ProviderConfig{
				ExampleStoreName: storeName,
			})
			if err != nil {
				t.Fatalf("parseLocation(%q) error = %v", storeName, err)
			}
			if loc != "us-central1" {
				t.Errorf("parseLocation(%q) = %q, want %q", storeName, loc, "us-central1")
			}
		}
	})

	t.Run("NewProvider builds client for regional endpoint and closes it", func(t *testing.T) {
		listener := bufconn.Listen(1024 * 1024)
		srv := grpc.NewServer()
		fakeSrv := &fakeExampleStoreServer{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult("hello", 0.9, makeExpectedContent("model", &aiplatformpb.Part{
						Data: &aiplatformpb.Part_Text{Text: "world"},
					})),
				},
			},
		}
		aiplatformpb.RegisterExampleStoreServiceServer(srv, fakeSrv)
		go func() {
			_ = srv.Serve(listener)
		}()
		t.Cleanup(func() {
			srv.Stop()
			_ = listener.Close()
		})

		var gotDialTarget string
		var initCtxErr error
		origFactory := newExampleStoreClient
		t.Cleanup(func() {
			newExampleStoreClient = origFactory
		})
		newExampleStoreClient = func(ctx context.Context, opts ...option.ClientOption) (exampleStoreClient, error) {
			initCtxErr = ctx.Err()
			allOpts := append([]option.ClientOption{
				option.WithoutAuthentication(),
				option.WithGRPCDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
				option.WithGRPCDialOption(grpc.WithContextDialer(func(_ context.Context, addr string) (net.Conn, error) {
					gotDialTarget = addr
					return listener.Dial()
				})),
			}, opts...)
			return origFactory(ctx, allOpts...)
		}

		canceledCtx, cancel := context.WithCancel(t.Context())
		cancel()

		const storeName = "projects/example.com:my-proj/locations/asia-east1/exampleStores/store-1"
		p, err := NewProvider(canceledCtx, &ProviderConfig{ExampleStoreName: storeName})
		if err != nil {
			t.Fatalf("NewProvider() error = %v", err)
		}
		if initCtxErr != nil {
			t.Errorf("client init context Err() = %v, want nil (detached from canceled caller context)", initCtxErr)
		}
		closer, ok := p.(io.Closer)
		if !ok {
			t.Fatal("returned Provider does not implement io.Closer")
		}
		t.Cleanup(func() {
			if err := closer.Close(); err != nil {
				t.Errorf("Close() error = %v", err)
			}
		})

		examples, err := p.Examples(t.Context(), "hi")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if gotDialTarget != "asia-east1-aiplatform.googleapis.com:443" {
			t.Errorf("dialed target = %q, want %q", gotDialTarget, "asia-east1-aiplatform.googleapis.com:443")
		}
		if fakeSrv.gotReq.GetExampleStore() != storeName {
			t.Errorf("SearchExamples ExampleStore = %q, want %q", fakeSrv.gotReq.GetExampleStore(), storeName)
		}
		if len(examples) != 1 || examples[0].Output[0].Parts[0].Text != "world" {
			t.Errorf("Examples() = %+v, want 1 example with output 'world'", examples)
		}

		// A canceled per-call context fails SearchExamples.
		if _, err := p.Examples(canceledCtx, "hi"); err == nil {
			t.Error("Examples() with canceled context succeeded, want error")
		}
	})

	t.Run("Close delegates to underlying client", func(t *testing.T) {
		client := &fakeExampleStoreClient{}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		closer, ok := p.(io.Closer)
		if !ok {
			t.Fatal("Provider does not implement io.Closer")
		}
		if err := closer.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if client.closed != 1 {
			t.Errorf("client.closed = %d, want 1", client.closed)
		}
	})

	t.Run("rejects invalid config or resource name", func(t *testing.T) {
		cases := []struct {
			name   string
			config *ProviderConfig
		}{
			{name: "nil config", config: nil},
			{name: "empty store name", config: &ProviderConfig{ExampleStoreName: ""}},
			{name: "bare store ID", config: &ProviderConfig{ExampleStoreName: "exampleStores/s"}},
			{name: "missing exampleStores segment", config: &ProviderConfig{ExampleStoreName: "projects/p/locations/l"}},
			{name: "extra trailing segment", config: &ProviderConfig{ExampleStoreName: "projects/p/locations/l/exampleStores/s/extra"}},
			{name: "invalid characters in location", config: &ProviderConfig{ExampleStoreName: "projects/p/locations/l#bad/exampleStores/s"}},
			{name: "empty project", config: &ProviderConfig{ExampleStoreName: "projects//locations/l/exampleStores/s"}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := NewProvider(t.Context(), tc.config); err == nil {
					t.Errorf("NewProvider(%+v) succeeded, want error", tc.config)
				}
				if _, err := newProvider(tc.config, &fakeExampleStoreClient{}); err == nil {
					t.Errorf("newProvider(%+v) succeeded, want error", tc.config)
				}
			})
		}
	})

	t.Run("rejects nil client in newProvider", func(t *testing.T) {
		if _, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, nil); err == nil {
			t.Fatal("newProvider with nil client succeeded, want error")
		}
	})
}

func TestProvider_Examples(t *testing.T) {
	// reference: test_get_examples_searches_the_configured_store
	t.Run("searches the configured store", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}

		examples, err := p.Examples(t.Context(), "what is the weather?")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 0 {
			t.Errorf("len(examples) = %d, want 0", len(examples))
		}

		wantReq := &aiplatformpb.SearchExamplesRequest{
			ExampleStore: testStoreName,
			TopK:         10,
			Parameters: &aiplatformpb.SearchExamplesRequest_StoredContentsExampleParameters{
				StoredContentsExampleParameters: &aiplatformpb.StoredContentsExampleParameters{
					Query: &aiplatformpb.StoredContentsExampleParameters_ContentSearchKey_{
						ContentSearchKey: &aiplatformpb.StoredContentsExampleParameters_ContentSearchKey{
							Contents: []*aiplatformpb.Content{
								{
									Role: genai.RoleUser,
									Parts: []*aiplatformpb.Part{
										{Data: &aiplatformpb.Part_Text{Text: "what is the weather?"}},
									},
								},
							},
							SearchKeyGenerationMethod: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod{
								Method: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod_LastEntry_{
									LastEntry: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod_LastEntry{},
								},
							},
						},
					},
				},
			},
		}
		if diff := cmp.Diff(wantReq, client.gotReq, protocmp.Transform()); diff != "" {
			t.Errorf("SearchExamplesRequest mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_get_examples_returns_empty_list_without_results
	t.Run("returns empty slice without results", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{}},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 0 {
			t.Errorf("got %v, want empty slice", examples)
		}
	})

	// reference: test_get_examples_tolerates_unset_results
	t.Run("tolerates unset results", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: nil,
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 0 {
			t.Errorf("got %v, want empty slice", examples)
		}
	})

	// reference: test_get_examples_converts_text_part
	t.Run("converts text part", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"what is the weather?",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{
							Data: &aiplatformpb.Part_Text{Text: "it is sunny"},
						}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		want := []*exampletool.Example{
			{
				Input:  genai.NewContentFromText("what is the weather?", genai.RoleUser),
				Output: []*genai.Content{genai.NewContentFromText("it is sunny", "model")},
			},
		}
		if diff := cmp.Diff(want, examples); diff != "" {
			t.Errorf("Examples() mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_get_examples_filters_results_below_similarity_threshold
	t.Run("filters results below similarity threshold", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult("too dissimilar", 0.49),
					makeResult("similar enough", 0.5),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 1 || examples[0].Input.Parts[0].Text != "similar enough" {
			t.Errorf("got examples %+v, want only 'similar enough'", examples)
		}
	})

	// reference: test_get_examples_converts_function_call_part
	t.Run("converts function call part", func(t *testing.T) {
		args, err := structpb.NewStruct(map[string]any{"city": "London"})
		if err != nil {
			t.Fatalf("structpb.NewStruct() error = %v", err)
		}
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{
							Data: &aiplatformpb.Part_FunctionCall{
								FunctionCall: &aiplatformpb.FunctionCall{
									Id:   "call-1",
									Name: "get_weather",
									Args: args,
								},
							},
						}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		fc := examples[0].Output[0].Parts[0].FunctionCall
		if fc == nil {
			t.Fatal("FunctionCall is nil")
		}
		if fc.ID != "call-1" || fc.Name != "get_weather" {
			t.Errorf("FunctionCall = %+v, want ID=call-1 Name=get_weather", fc)
		}
		if diff := cmp.Diff(map[string]any{"city": "London"}, fc.Args); diff != "" {
			t.Errorf("FunctionCall.Args mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_get_examples_converts_function_response_part
	t.Run("converts function response part", func(t *testing.T) {
		respStruct, err := structpb.NewStruct(map[string]any{"temperature": float64(12)})
		if err != nil {
			t.Fatalf("structpb.NewStruct() error = %v", err)
		}
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("user", &aiplatformpb.Part{
							Data: &aiplatformpb.Part_FunctionResponse{
								FunctionResponse: &aiplatformpb.FunctionResponse{
									Id:       "call-1",
									Name:     "get_weather",
									Response: respStruct,
								},
							},
						}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		fr := examples[0].Output[0].Parts[0].FunctionResponse
		if fr == nil {
			t.Fatal("FunctionResponse is nil")
		}
		if fr.ID != "call-1" || fr.Name != "get_weather" {
			t.Errorf("FunctionResponse = %+v, want ID=call-1 Name=get_weather", fr)
		}
		if diff := cmp.Diff(map[string]any{"temperature": float64(12)}, fr.Response); diff != "" {
			t.Errorf("FunctionResponse.Response mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_get_examples_preserves_inline_data_part
	t.Run("preserves inline data part", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{
							Data: &aiplatformpb.Part_InlineData{
								InlineData: &aiplatformpb.Blob{
									MimeType: "image/png",
									Data:     []byte{0x89, 'P', 'N', 'G'},
								},
							},
						}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		blob := examples[0].Output[0].Parts[0].InlineData
		if blob == nil {
			t.Fatal("InlineData is nil")
		}
		if blob.MIMEType != "image/png" || string(blob.Data) != "\x89PNG" {
			t.Errorf("InlineData = %+v, want image/png \\x89PNG", blob)
		}
	})

	t.Run("converts file data, video metadata, executable code, code execution result, and thought fields", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent(
							"model",
							&aiplatformpb.Part{
								Data: &aiplatformpb.Part_FileData{
									FileData: &aiplatformpb.FileData{
										MimeType: "video/mp4",
										FileUri:  "gs://bucket/clip.mp4",
									},
								},
								Metadata: &aiplatformpb.Part_VideoMetadata{
									VideoMetadata: &aiplatformpb.VideoMetadata{
										StartOffset: durationpb.New(5 * time.Second),
										EndOffset:   durationpb.New(15 * time.Second),
									},
								},
							},
							&aiplatformpb.Part{
								Data: &aiplatformpb.Part_ExecutableCode{
									ExecutableCode: &aiplatformpb.ExecutableCode{
										Code:     "print(42)",
										Language: aiplatformpb.ExecutableCode_PYTHON,
									},
								},
							},
							&aiplatformpb.Part{
								Data: &aiplatformpb.Part_CodeExecutionResult{
									CodeExecutionResult: &aiplatformpb.CodeExecutionResult{
										Outcome: aiplatformpb.CodeExecutionResult_OUTCOME_OK,
										Output:  "42\n",
									},
								},
							},
							&aiplatformpb.Part{
								Thought:          true,
								ThoughtSignature: []byte("sig-1"),
								Data: &aiplatformpb.Part_Text{
									Text: "reasoning step",
								},
							},
						),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		wantParts := []*genai.Part{
			{
				FileData: &genai.FileData{
					MIMEType: "video/mp4",
					FileURI:  "gs://bucket/clip.mp4",
				},
				VideoMetadata: &genai.VideoMetadata{
					StartOffset: 5 * time.Second,
					EndOffset:   15 * time.Second,
				},
			},
			{
				ExecutableCode: &genai.ExecutableCode{
					Code:     "print(42)",
					Language: genai.LanguagePython,
				},
			},
			{
				CodeExecutionResult: &genai.CodeExecutionResult{
					Outcome: genai.OutcomeOK,
					Output:  "42\n",
				},
			},
			{
				Text:             "reasoning step",
				Thought:          true,
				ThoughtSignature: []byte("sig-1"),
			},
		}
		if diff := cmp.Diff(wantParts, examples[0].Output[0].Parts); diff != "" {
			t.Errorf("Parts mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("keeps data-less parts with thought, signature, or video metadata and drops empty parts", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent(
							"model",
							nil,
							&aiplatformpb.Part{},
							&aiplatformpb.Part{
								Thought: true,
							},
							&aiplatformpb.Part{
								ThoughtSignature: []byte("sig-only"),
							},
							&aiplatformpb.Part{
								Metadata: &aiplatformpb.Part_VideoMetadata{
									VideoMetadata: &aiplatformpb.VideoMetadata{
										StartOffset: durationpb.New(2 * time.Second),
										EndOffset:   durationpb.New(8 * time.Second),
									},
								},
							},
						),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		wantParts := []*genai.Part{
			{
				Thought: true,
			},
			{
				ThoughtSignature: []byte("sig-only"),
			},
			{
				VideoMetadata: &genai.VideoMetadata{
					StartOffset: 2 * time.Second,
					EndOffset:   8 * time.Second,
				},
			},
		}
		if diff := cmp.Diff(wantParts, examples[0].Output[0].Parts); diff != "" {
			t.Errorf("Parts mismatch (-want +got):\n%s", diff)
		}
	})

	// reference: test_get_examples_preserves_multi_step_expected_output
	t.Run("preserves multi-step expected output", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: "step 1"}}),
						makeExpectedContent("model", &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: "step 2"}}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples[0].Output) != 2 || examples[0].Output[0].Parts[0].Text != "step 1" || examples[0].Output[1].Parts[0].Text != "step 2" {
			t.Errorf("Output = %+v, want [step 1, step 2]", examples[0].Output)
		}
	})

	// reference: test_get_examples_skips_expected_contents_without_content
	t.Run("skips expected contents without content", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					makeResult(
						"search key",
						1.0,
						nil,
						&aiplatformpb.ContentsExample_ExpectedContent{Content: nil},
						makeExpectedContent("model", &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: "kept"}}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples[0].Output) != 1 || examples[0].Output[0].Parts[0].Text != "kept" {
			t.Errorf("Output = %+v, want [kept]", examples[0].Output)
		}
	})

	// reference: test_get_examples_skips_result_without_example
	t.Run("skips result without example", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					nil,
					{SimilarityScore: 1.0, Example: nil},
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: "kept"}}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 1 || examples[0].Output[0].Parts[0].Text != "kept" {
			t.Errorf("examples = %+v, want 1 example with 'kept'", examples)
		}
	})

	// reference: test_get_examples_skips_result_without_stored_contents_example
	t.Run("skips result without stored contents example", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					{
						SimilarityScore: 1.0,
						Example:         &aiplatformpb.Example{},
					},
					makeResult(
						"search key",
						1.0,
						makeExpectedContent("model", &aiplatformpb.Part{Data: &aiplatformpb.Part_Text{Text: "kept"}}),
					),
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 1 || examples[0].Output[0].Parts[0].Text != "kept" {
			t.Errorf("examples = %+v, want 1 example with 'kept'", examples)
		}
	})

	// reference: test_get_examples_tolerates_unset_contents_example
	t.Run("tolerates unset contents example", func(t *testing.T) {
		client := &fakeExampleStoreClient{
			resp: &aiplatformpb.SearchExamplesResponse{
				Results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
					{
						SimilarityScore: 1.0,
						Example: &aiplatformpb.Example{
							ExampleType: &aiplatformpb.Example_StoredContentsExample{
								StoredContentsExample: &aiplatformpb.StoredContentsExample{
									SearchKey:       "prompt",
									ContentsExample: nil,
								},
							},
						},
					},
				},
			},
		}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		examples, err := p.Examples(t.Context(), "query")
		if err != nil {
			t.Fatalf("Examples() error = %v", err)
		}
		if len(examples) != 1 || examples[0].Input.Parts[0].Text != "prompt" || len(examples[0].Output) != 0 {
			t.Errorf("examples = %+v, want 1 example with Input='prompt' and empty Output", examples)
		}
	})

	t.Run("propagates search error", func(t *testing.T) {
		wantErr := errors.New("rpc failed")
		client := &fakeExampleStoreClient{err: wantErr}
		p, err := newProvider(&ProviderConfig{ExampleStoreName: testStoreName}, client)
		if err != nil {
			t.Fatalf("newProvider() error = %v", err)
		}
		_, err = p.Examples(t.Context(), "query")
		if !errors.Is(err, wantErr) {
			t.Fatalf("Examples() error = %v, want %v", err, wantErr)
		}
	})
}
