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

package vertexai_test

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net"
	"strings"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/exampletool"
	"google.golang.org/adk/v2/tool/exampletool/vertexai"
)

// fakeExampleStore is an in-process Example Store service. It answers every
// SearchExamples request with the same results and prints the query text of
// each request it receives.
type fakeExampleStore struct {
	aiplatformpb.UnimplementedExampleStoreServiceServer
	results []*aiplatformpb.SearchExamplesResponse_SimilarExample
}

func (s *fakeExampleStore) SearchExamples(_ context.Context, req *aiplatformpb.SearchExamplesRequest) (*aiplatformpb.SearchExamplesResponse, error) {
	query := req.GetStoredContentsExampleParameters().GetContentSearchKey().GetContents()[0].GetParts()[0].GetText()
	fmt.Printf("store received: %s top_k=%d query=%q\n", req.GetExampleStore(), req.GetTopK(), query)
	return &aiplatformpb.SearchExamplesResponse{Results: s.results}, nil
}

// storedExample returns a search result whose stored example pairs searchKey
// with one model turn that calls functionName with args.
func storedExample(score float32, searchKey, functionName string, args map[string]any) *aiplatformpb.SearchExamplesResponse_SimilarExample {
	argsStruct, err := structpb.NewStruct(args)
	if err != nil {
		log.Fatal(err)
	}
	return &aiplatformpb.SearchExamplesResponse_SimilarExample{
		SimilarityScore: score,
		Example: &aiplatformpb.Example{
			ExampleType: &aiplatformpb.Example_StoredContentsExample{
				StoredContentsExample: &aiplatformpb.StoredContentsExample{
					SearchKey: searchKey,
					ContentsExample: &aiplatformpb.ContentsExample{
						ExpectedContents: []*aiplatformpb.ContentsExample_ExpectedContent{{
							Content: &aiplatformpb.Content{
								Role: "model",
								Parts: []*aiplatformpb.Part{{
									Data: &aiplatformpb.Part_FunctionCall{FunctionCall: &aiplatformpb.FunctionCall{
										Name: functionName,
										Args: argsStruct,
									}},
								}},
							},
						}},
					},
				},
			},
		},
	}
}

// startFakeExampleStore serves store on an in-memory listener and returns a
// gRPC connection to it and a function that stops the server.
func startFakeExampleStore(store *fakeExampleStore) (*grpc.ClientConn, func()) {
	listener := bufconn.Listen(1024 * 1024)
	srv := grpc.NewServer()
	aiplatformpb.RegisterExampleStoreServiceServer(srv, store)
	go func() {
		if err := srv.Serve(listener); err != nil {
			log.Print(err)
		}
	}()
	conn, err := grpc.NewClient("passthrough:///example-store",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return listener.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatal(err)
	}
	return conn, srv.Stop
}

// This example connects a Provider to an Example Store through caller client
// options. option.WithGRPCConn replaces the regional endpoint and Application
// Default Credentials, so the provider talks to an in-process fake store. The
// store returns one result above the 0.5 similarity threshold and one below
// it; Examples keeps the first and turns its search key into the example's
// user input.
func ExampleNewProviderWithOptions() {
	conn, stop := startFakeExampleStore(&fakeExampleStore{
		results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
			storedExample(0.9, "Where is order 4417?", "check_order_status", map[string]any{"order_id": "4417"}),
			storedExample(0.3, "Cancel my subscription.", "cancel_subscription", nil),
		},
	})
	defer stop()

	ctx := context.Background()
	p, err := vertexai.NewProviderWithOptions(ctx, &vertexai.ProviderConfig{
		ExampleStoreName: "projects/my-project/locations/us-central1/exampleStores/support",
	}, option.WithGRPCConn(conn))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("project:", p.Project(), "location:", p.Location())

	examples, err := p.Examples(ctx, "Where is my order?")
	if err != nil {
		log.Fatal(err)
	}
	for _, ex := range examples {
		fmt.Println("input:", ex.Input.Parts[0].Text)
		for _, out := range ex.Output {
			for _, part := range out.Parts {
				fmt.Printf("output: %s %s(%v)\n", out.Role, part.FunctionCall.Name, part.FunctionCall.Args)
			}
		}
	}

	// Close closes the client, and with it the connection passed in
	// option.WithGRPCConn.
	if err := p.Close(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("connection shut down:", conn.GetState() == connectivity.Shutdown)
	// Output:
	// project: my-project location: us-central1
	// store received: projects/my-project/locations/us-central1/exampleStores/support top_k=10 query="Where is my order?"
	// input: Where is order 4417?
	// output: model check_order_status(map[order_id:4417])
	// connection shut down: true
}

// recordingModel stands in for a real model: it answers every request with the
// same text and keeps the last request it received.
type recordingModel struct {
	lastReq *model.LLMRequest
}

func (*recordingModel) Name() string { return "gemini-2.5-flash" }

func (m *recordingModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.lastReq = req
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText("Let me look that up.", genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
}

// This example puts a Provider behind the example tool of package
// exampletool. On each turn the tool sends the text of the user message to the
// Example Store and appends the examples it returns to the system instruction
// of the model request, after the agent's own instruction. The model here is a
// stand-in that records the request;
// its name, gemini-2.5-flash, makes the tool write the function call in a plain
// code fence.
func Example_exampleTool() {
	conn, stop := startFakeExampleStore(&fakeExampleStore{
		results: []*aiplatformpb.SearchExamplesResponse_SimilarExample{
			storedExample(0.8, "Where is order 4417?", "check_order_status", map[string]any{"order_id": "4417"}),
		},
	})
	defer stop()

	ctx := context.Background()
	p, err := vertexai.NewProviderWithOptions(ctx, &vertexai.ProviderConfig{
		ExampleStoreName: "projects/my-project/locations/us-central1/exampleStores/support",
	}, option.WithGRPCConn(conn))
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := p.Close(); err != nil {
			log.Print(err)
		}
	}()

	examples, err := exampletool.New(exampletool.ExampleToolConfig{Provider: p})
	if err != nil {
		log.Fatal(err)
	}
	llm := &recordingModel{}
	a, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Model:       llm,
		Instruction: "Help the user with their order.",
		Tools:       []tool.Tool{examples},
	})
	if err != nil {
		log.Fatal(err)
	}

	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("Where is order 981?", genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}

	var sys strings.Builder
	for _, part := range llm.lastReq.Config.SystemInstruction.Parts {
		sys.WriteString(part.Text)
	}
	// The examples come after the agent's own instruction.
	_, block, _ := strings.Cut(sys.String(), "<EXAMPLES>")
	fmt.Println("<EXAMPLES>" + block)
	// Output:
	// store received: projects/my-project/locations/us-central1/exampleStores/support top_k=10 query="Where is order 981?"
	// <EXAMPLES>
	// Begin few-shot
	// The following are examples of user queries and model responses using the available tools.
	//
	// EXAMPLE 1:
	// Begin example
	// [user]
	// Where is order 4417?
	// [model]
	// ```
	// check_order_status(order_id='4417')
	// ```
	// End example
	//
	// End few-shot
	// <EXAMPLES>
}
