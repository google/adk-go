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

package agentanalytics

import (
	"context"
	"fmt"
	"io"
	"iter"
	"net"
	"sync/atomic"
	"testing"
	"time"

	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

type discardBenchLogger struct{}

func (discardBenchLogger) Printf(string, ...any) {}
func (discardBenchLogger) Println(...any)        {}

type benchStaticLLM struct {
	resp *model.LLMResponse
}

func (m *benchStaticLLM) Name() string { return "gemini-2.5-flash" }

func (m *benchStaticLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(m.resp, nil)
	}
}

type slowAppendWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	delay             time.Duration
	turnSeq           *atomic.Int64
	rpcsOnRequestPath *atomic.Int64
	closing           *atomic.Bool
}

func (s *slowAppendWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		if err := stream.Context().Err(); err != nil {
			return err
		}
		_, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		startSeq := s.turnSeq.Load()
		if !s.closing.Load() && s.delay > 0 {
			time.Sleep(s.delay)
		}
		endSeq := s.turnSeq.Load()
		if startSeq&1 == 1 && endSeq == startSeq {
			s.rpcsOnRequestPath.Add(1)
		}
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
}

// BenchmarkRunTurnWithAnalyticsFlush measures a runner turn with the BigQuery
// Agent Analytics plugin registered against a fake BigQuery Storage Write
// server that sleeps 50ms per AppendRows RPC (#17a).
func BenchmarkRunTurnWithAnalyticsFlush(b *testing.B) {
	ctx := context.Background()
	var turnSeq atomic.Int64
	var rpcsOnRequestPath atomic.Int64
	var closing atomic.Bool

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		b.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, &slowAppendWriteServer{
		delay:             50 * time.Millisecond,
		turnSeq:           &turnSeq,
		rpcsOnRequestPath: &rpcsOnRequestPath,
		closing:           &closing,
	})
	go func() { _ = gSrv.Serve(lis) }()
	b.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		b.Fatalf("dial: %v", err)
	}
	b.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		b.Fatalf("NewBigQueryWriteClient: %v", err)
	}

	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ProjectID = "bench-project"
	cfg.DatasetID = "bench_dataset"
	cfg.TableName = "bench_table"
	cfg.BatchSize = 500
	cfg.QueueMaxSize = 100000
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 200 * time.Millisecond
	cfg.DisableCreateViews = true
	cfg.Logger = discardBenchLogger{}

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, writeClient)
	if err != nil {
		b.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients: %v", err)
	}
	b.Cleanup(func() {
		closing.Store(true)
		_ = p.Close()
	})

	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "Benchmark assistant",
		Model: &benchStaticLLM{resp: &model.LLMResponse{
			Content:      genai.NewContentFromText("Here is the answer.", genai.RoleModel),
			TurnComplete: true,
		}},
		Instruction: "You are a helpful assistant.",
	})
	if err != nil {
		b.Fatalf("llmagent.New: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bench-app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*baseplugin.Plugin{p}},
	})
	if err != nil {
		b.Fatalf("runner.New: %v", err)
	}

	userMsg := genai.NewContentFromText("Hello, how are you?", genai.RoleUser)
	for _, err := range r.Run(ctx, "user-1", "warmup", userMsg, agent.RunConfig{}) {
		if err != nil {
			b.Fatalf("warmup Run: %v", err)
		}
	}

	rpcsOnRequestPath.Store(0)
	turnSeq.Store(0)
	b.ReportAllocs()
	b.ResetTimer()
	i := 0
	for b.Loop() {
		i++
		sessID := fmt.Sprintf("s-%d", i)
		turnSeq.Add(1) // odd while inside r.Run
		for _, err := range r.Run(ctx, "user-1", sessID, userMsg, agent.RunConfig{}) {
			if err != nil {
				b.Fatalf("Run: %v", err)
			}
		}
		turnSeq.Add(1) // even outside r.Run
	}
	b.StopTimer()
	closing.Store(true)
	b.ReportMetric(float64(rpcsOnRequestPath.Load())/float64(b.N), "append-rows-rpcs-on-request-path/turn")
}

type benchToolArgs struct {
	Query      string   `json:"query"`
	MaxResults int      `json:"max_results,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	IncludeRaw bool     `json:"include_raw,omitempty"`
}

type benchToolResult struct {
	Items []string `json:"items"`
	Count int      `json:"count"`
}

// BenchmarkBeforeModelPrompt measures beforeModel on a 100-message conversation
// history with system instruction and 10 function tools (#17b).
func BenchmarkBeforeModelPrompt(b *testing.B) {
	ctx := context.Background()
	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ProjectID = "bench-project"
	cfg.DatasetID = "bench_dataset"
	cfg.TableName = "bench_table"
	cfg.BatchSize = 1000000
	cfg.QueueMaxSize = 1000000
	cfg.BatchFlushIntv = time.Hour
	cfg.DisableFlushOnRunEnd = true
	cfg.DisableCreateViews = true
	cfg.Logger = discardBenchLogger{}

	bp, err := NewBatchProcessor(ctx, nil, "projects/p/datasets/d/tables/t/streams/_default", cfg)
	if err != nil {
		b.Fatalf("NewBatchProcessor: %v", err)
	}
	stopDrain := make(chan struct{})
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		for {
			select {
			case <-bp.queue:
			case <-stopDrain:
				return
			}
		}
	}()
	b.Cleanup(func() {
		close(stopDrain)
		<-drainDone
		bp.closed.Store(true)
		_ = bp.drainQueue()
	})

	traces := newTraceManager()
	traces.logf = cfg.logf
	ap := &analyticsPlugin{
		config:         cfg,
		processor:      bp,
		allowlist:      newSet(cfg.EventAllowlist...),
		denylist:       newSet(cfg.EventDenylist...),
		deniedColumns:  map[string]struct{}{},
		customMetadata: newCustomMetadataFilter(cfg.CustomMetadataAllowlist),
		traces:         traces,
	}

	const numMessages = 100
	contents := make([]*genai.Content, numMessages)
	for i := range numMessages {
		role := genai.Role(genai.RoleUser)
		if i%2 == 1 {
			role = genai.Role(genai.RoleModel)
		}
		contents[i] = genai.NewContentFromText(
			fmt.Sprintf("Message %d in the conversation history discussing performance characteristics and telemetry spans.", i),
			role,
		)
	}

	const numTools = 10
	toolsMap := make(map[string]any, numTools)
	for i := range numTools {
		name := fmt.Sprintf("search_tool_%d", i)
		ft, err := functiontool.New(functiontool.Config{
			Name:        name,
			Description: fmt.Sprintf("Searches domain index %d for matching documents and metadata.", i),
		}, func(_ agent.Context, _ benchToolArgs) (benchToolResult, error) {
			return benchToolResult{}, nil
		})
		if err != nil {
			b.Fatalf("functiontool.New: %v", err)
		}
		toolsMap[name] = ft
	}

	req := &model.LLMRequest{
		Model:    "gemini-2.5-flash",
		Contents: contents,
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are an expert assistant answering questions accurately.", genai.RoleUser),
		},
		Tools: toolsMap,
	}

	mCtx := &mockInvocationContext{
		ctx:          ctx,
		agentName:    "assistant",
		sessionID:    "sess-1",
		invocationID: "inv-1",
		userID:       "user-1",
		appName:      "bench-app",
		path:         "seq@0/assistant@1",
	}

	if _, err := ap.beforeModel(mCtx, req); err != nil {
		b.Fatalf("warmup beforeModel: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := ap.beforeModel(mCtx, req); err != nil {
			b.Fatalf("beforeModel: %v", err)
		}
	}
}

// BenchmarkContextPath measures contextPath across a workflow-node ADK callback
// context (non-empty path), a root-agent ADK callback context (empty path,
// full chain traversal), and an InvocationContext fallback (#17c).
func BenchmarkContextPath(b *testing.B) {
	ctx := context.Background()
	var wfCapturedCtx agent.Context
	wfSub, err := llmagent.New(llmagent.Config{
		Name: "assistant",
		Model: &benchStaticLLM{resp: &model.LLMResponse{
			Content:      genai.NewContentFromText("ok", genai.RoleModel),
			TurnComplete: true,
		}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(c agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				wfCapturedCtx = c
				return nil, nil
			},
		},
	})
	if err != nil {
		b.Fatalf("llmagent.New: %v", err)
	}
	node, err := workflow.NewAgentNode(wfSub, workflow.NodeConfig{})
	if err != nil {
		b.Fatalf("workflow.NewAgentNode: %v", err)
	}
	wf, err := workflowagent.New(workflowagent.Config{
		Name: "wf",
		Edges: []workflow.Edge{
			{From: workflow.Start, To: node},
		},
	})
	if err != nil {
		b.Fatalf("workflowagent.New: %v", err)
	}
	rWf, err := runner.New(runner.Config{
		AppName:           "bench-app",
		Agent:             wf,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		b.Fatalf("runner.New: %v", err)
	}
	for _, err := range rWf.Run(ctx, "u1", "s1", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			b.Fatalf("Run(wf): %v", err)
		}
	}
	if wfCapturedCtx == nil {
		b.Fatal("BeforeModelCallback did not capture workflow agent.Context")
	}
	if got := contextPath(wfCapturedCtx); got == "" {
		b.Fatalf("contextPath(wfCapturedCtx) = %q, want non-empty node path", got)
	}

	var rootCapturedCtx agent.Context
	rootSub, err := llmagent.New(llmagent.Config{
		Name: "root_assistant",
		Model: &benchStaticLLM{resp: &model.LLMResponse{
			Content:      genai.NewContentFromText("ok", genai.RoleModel),
			TurnComplete: true,
		}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(c agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				rootCapturedCtx = c
				return nil, nil
			},
		},
	})
	if err != nil {
		b.Fatalf("llmagent.New(root): %v", err)
	}
	rRoot, err := runner.New(runner.Config{
		AppName:           "bench-app",
		Agent:             rootSub,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		b.Fatalf("runner.New(root): %v", err)
	}
	for _, err := range rRoot.Run(ctx, "u1", "s2", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			b.Fatalf("Run(root): %v", err)
		}
	}
	if rootCapturedCtx == nil {
		b.Fatal("BeforeModelCallback did not capture root agent.Context")
	}

	b.Run("workflow_callback_context", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if p := contextPath(wfCapturedCtx); p == "" {
				b.Fatal("empty path")
			}
		}
	})

	b.Run("root_callback_context", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if p := contextPath(rootCapturedCtx); p != "" {
				b.Fatalf("got %q, want empty path", p)
			}
		}
	})

	mCtx := &mockInvocationContext{
		ctx:          ctx,
		agentName:    "assistant",
		sessionID:    "sess-1",
		invocationID: "inv-1",
		userID:       "user-1",
		appName:      "bench-app",
		path:         "seq@0/assistant@1",
	}
	b.Run("invocation_context_fallback", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if p := contextPath(mCtx); p == "" {
				b.Fatal("empty path")
			}
		}
	})
}
