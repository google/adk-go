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
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

type mockTransport struct {
	roundTrip func(*http.Request) (*http.Response, error)
}

func (m *mockTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return m.roundTrip(r)
}

// fakeBigQueryWriteServer implements a fake gRPC BigQueryWriteServer for testing.
type fakeBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	requests chan *storagepb.AppendRowsRequest
}

func (s *fakeBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		if s.requests != nil {
			s.requests <- req
		}
		// Send a minimal response back.
		if err := stream.Send(&storagepb.AppendRowsResponse{}); err != nil {
			return err
		}
	}
}

func TestNewBigQueryAgentAnalyticsPlugin_Disabled(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.Enabled = false

	p, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if p == nil {
		t.Fatal("Expected plugin, got nil")
	}
	if p.Name() != "bigquery_agent_analytics" {
		t.Errorf("Expected name 'bigquery_agent_analytics', got %s", p.Name())
	}
}

func TestNewBigQueryAgentAnalyticsPlugin_Enabled(t *testing.T) {
	p, _, _ := setupTestPlugin(t)
	if p == nil {
		t.Fatal("Expected plugin, got nil")
	}
	if p.Name() != "bigquery_agent_analytics" {
		t.Errorf("Expected name 'bigquery_agent_analytics', got %s", p.Name())
	}

	// Verify callbacks are registered
	if p.OnUserMessageCallback() == nil {
		t.Error("OnUserMessageCallback is nil")
	}
	if p.BeforeRunCallback() == nil {
		t.Error("BeforeRunCallback is nil")
	}
	if p.AfterRunCallback() == nil {
		t.Error("AfterRunCallback is nil")
	}
	if p.OnEventCallback() == nil {
		t.Error("OnEventCallback is nil")
	}
}

type mockInvocationContext struct {
	agent.Context
	ctx          context.Context
	agentName    string
	sessionID    string
	invocationID string
	userID       string
}

func (m *mockInvocationContext) Deadline() (deadline time.Time, ok bool) { return m.ctx.Deadline() }
func (m *mockInvocationContext) Done() <-chan struct{}                   { return m.ctx.Done() }
func (m *mockInvocationContext) Err() error                              { return m.ctx.Err() }
func (m *mockInvocationContext) Value(key any) any                       { return m.ctx.Value(key) }

func (m *mockInvocationContext) AgentName() string    { return m.agentName }
func (m *mockInvocationContext) SessionID() string    { return m.sessionID }
func (m *mockInvocationContext) InvocationID() string { return m.invocationID }
func (m *mockInvocationContext) UserID() string       { return m.userID }
func (m *mockInvocationContext) Branch() string       { return "" }

func (m *mockInvocationContext) UserContent() *genai.Content          { return nil }
func (m *mockInvocationContext) ReadonlyState() session.ReadonlyState { return nil }
func (m *mockInvocationContext) AppName() string                      { return "" }
func (m *mockInvocationContext) Artifacts() agent.Artifacts           { return nil }
func (m *mockInvocationContext) State() session.State                 { return nil }

func TestOnUserMessageCallback_AppendsToWriter(t *testing.T) {
	p, requestsChan, mCtx := setupTestPlugin(t)

	userMsgCb := p.OnUserMessageCallback()
	if userMsgCb == nil {
		t.Fatal("OnUserMessageCallback is nil")
	}

	_, err := userMsgCb(mCtx, &genai.Content{Parts: []*genai.Part{{Text: "hello"}}})
	if err != nil {
		t.Fatalf("OnUserMessageCallback error: %v", err)
	}

	afterRunCb := p.AfterRunCallback()
	if afterRunCb == nil {
		t.Fatal("AfterRunCallback is nil")
	}
	afterRunCb(mCtx)

	// Verify that a request was received
	select {
	case req := <-requestsChan:
		if req == nil {
			t.Error("Received nil request")
		}
	case <-time.After(5 * time.Second):
		t.Error("Timed out waiting for request")
	}
}

func TestBeforeRunCallback_AppendsToWriter(t *testing.T) {
	p, requestsChan, mCtx := setupTestPlugin(t)

	beforeRunCb := p.BeforeRunCallback()
	if beforeRunCb == nil {
		t.Fatal("BeforeRunCallback is nil")
	}

	_, err := beforeRunCb(mCtx)
	if err != nil {
		t.Fatalf("BeforeRunCallback error: %v", err)
	}

	afterRunCb := p.AfterRunCallback()
	if afterRunCb == nil {
		t.Fatal("AfterRunCallback is nil")
	}
	afterRunCb(mCtx)

	select {
	case req := <-requestsChan:
		if req == nil {
			t.Error("Received nil request")
		}
	case <-time.After(5 * time.Second):
		t.Error("Timed out waiting for request")
	}
}

func setupTestPlugin(t *testing.T) (*baseplugin.Plugin, chan *storagepb.AppendRowsRequest, *mockInvocationContext) {
	ctx := context.Background()
	config := DefaultConfig()
	config.Enabled = true
	config.ProjectID = "test-project"
	config.DatasetID = "test-dataset"
	config.TableName = "test-table"

	mockTransport := &mockTransport{
		roundTrip: func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}, nil
		},
	}
	httpClient := &http.Client{Transport: mockTransport}
	bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("Failed to create bigquery client: %v", err)
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	gSrv := grpc.NewServer()
	requestsChan := make(chan *storagepb.AppendRowsRequest, 10)
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: requestsChan})
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial test server: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("Failed to create bigquery client: %v", err)
	}

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	mCtx := &mockInvocationContext{
		ctx:          ctx,
		agentName:    "test-agent",
		sessionID:    "sess-id",
		invocationID: "inv-id",
		userID:       "user-id",
	}

	return p, requestsChan, mCtx
}

func TestOnEventCallback_AppendsToWriter(t *testing.T) {
	p, requestsChan, mCtx := setupTestPlugin(t)

	eventCb := p.OnEventCallback()
	if eventCb == nil {
		t.Fatal("OnEventCallback is nil")
	}

	_, err := eventCb(mCtx, &session.Event{
		Author: "test-author",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{{Text: "event-data"}}},
		},
	})
	if err != nil {
		t.Fatalf("OnEventCallback error: %v", err)
	}

	p.AfterRunCallback()(mCtx)

	select {
	case req := <-requestsChan:
		if req == nil {
			t.Error("Received nil request")
		}
	case <-time.After(5 * time.Second):
		t.Error("Timed out waiting for request")
	}
}

func TestOnModelErrorCallback_AppendsToWriter(t *testing.T) {
	p, requestsChan, mCtx := setupTestPlugin(t)

	modelErrorCb := p.OnModelErrorCallback()
	if modelErrorCb == nil {
		t.Fatal("OnModelErrorCallback is nil")
	}

	_, err := modelErrorCb(mCtx, &model.LLMRequest{}, errors.New("test error"))
	if err != nil {
		t.Fatalf("OnModelErrorCallback error: %v", err)
	}

	p.AfterRunCallback()(mCtx)

	select {
	case req := <-requestsChan:
		if req == nil {
			t.Error("Received nil request")
		}
	case <-time.After(5 * time.Second):
		t.Error("Timed out waiting for request")
	}
}

func TestAfterModelCallback_PartialChunksDoNotEachLogUsage(t *testing.T) {
	run := func(t *testing.T, chunks []*model.LLMResponse) (totalRows int) {
		p, requestsChan, mCtx := setupTestPlugin(t)
		cb := p.AfterModelCallback()
		for _, c := range chunks {
			if _, err := cb(mCtx, c, nil); err != nil {
				t.Fatalf("AfterModelCallback error: %v", err)
			}
		}
		p.AfterRunCallback()(mCtx) // logs 1 INVOCATION_END row too, then flushes
		for {
			select {
			case req := <-requestsChan:
				totalRows += int(req.GetArrowRows().GetRows().GetRowCount())
			case <-time.After(time.Second):
				return
			}
		}
	}
	usage := func(c int32) *genai.GenerateContentResponseUsageMetadata {
		return &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 100, CandidatesTokenCount: c}
	}
	baseline := run(t, []*model.LLMResponse{
		{Partial: false, Content: &genai.Content{Parts: []*genai.Part{{Text: "final"}}}, UsageMetadata: usage(15)},
	})
	streaming := run(t, []*model.LLMResponse{
		{Partial: true, UsageMetadata: usage(5)},
		{Partial: true, UsageMetadata: usage(10)},
		{Partial: true, UsageMetadata: usage(15)},
		{Partial: false, UsageMetadata: usage(15)},
	})
	if streaming != baseline {
		t.Errorf("expected streaming turn to log the same number of rows as the baseline turn (%d), got %d", baseline, streaming)
	}
}

func TestLogEvent_ExtractsTraceInfo(t *testing.T) {
	p, requestsChan, mCtx := setupTestPlugin(t)

	traceID, _ := trace.TraceIDFromHex("4bf92f3577b34da6a3ce929d0e0e4736")
	spanID, _ := trace.SpanIDFromHex("00f067aa0ba902b7")
	spanCtx := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: traceID,
		SpanID:  spanID,
	})

	ctx := trace.ContextWithSpanContext(mCtx.ctx, spanCtx)
	mCtx.ctx = ctx

	userMsgCb := p.OnUserMessageCallback()
	_, err := userMsgCb(mCtx, &genai.Content{Parts: []*genai.Part{{Text: "hello"}}})
	if err != nil {
		t.Fatalf("OnUserMessageCallback error: %v", err)
	}

	p.AfterRunCallback()(mCtx)

	select {
	case req := <-requestsChan:
		if req == nil {
			t.Error("Received nil request")
		}
	case <-time.After(5 * time.Second):
		t.Error("Timed out waiting for request")
	}
}

func TestNewBigQueryAgentAnalyticsPlugin_CreateTable_WithPartitioning(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.Enabled = true
	config.ProjectID = "test-project"
	config.DatasetID = "test-dataset"
	config.TableName = "test-table"

	createCalled := false
	var requestBody string

	mockTransport := &mockTransport{
		roundTrip: func(r *http.Request) (*http.Response, error) {
			// Table metadata request: returns 404 Not Found to trigger creation
			if r.Method == "GET" && strings.Contains(r.URL.Path, "/datasets/test-dataset/tables/test-table") {
				return &http.Response{
					StatusCode: http.StatusNotFound,
					Body:       io.NopCloser(strings.NewReader(`{"error":{"code":404,"message":"Not found"}}`)),
				}, nil
			}
			// Table creation request
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/datasets/test-dataset/tables") {
				createCalled = true
				bodyBytes, _ := io.ReadAll(r.Body)
				requestBody = string(bodyBytes)
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}, nil
		},
	}
	httpClient := &http.Client{Transport: mockTransport}
	bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("Failed to create bigquery client: %v", err)
	}

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("failed to dial test server: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("Failed to create BigQuery write client: %v", err)
	}

	_, err = NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
	if err != nil {
		t.Fatalf("Plugin initialization error: %v", err)
	}

	if !createCalled {
		t.Error("Expected table creation to be called")
	}

	if !strings.Contains(requestBody, "timePartitioning") {
		t.Errorf("Expected request body to contain 'timePartitioning', got: %s", requestBody)
	}
	if !strings.Contains(requestBody, "DAY") {
		t.Errorf("Expected partitioning type to be 'DAY', got request body: %s", requestBody)
	}
	if !strings.Contains(requestBody, "timestamp") {
		t.Errorf("Expected partitioning field to be 'timestamp', got request body: %s", requestBody)
	}
}

func TestHybridContentParser(t *testing.T) {
	ctx := context.Background()

	// Test Case 1: Inline parsing (offloader is nil)
	t.Run("InlineParsingNoOffloader", func(t *testing.T) {
		parser := newHybridContentParser(nil, "conn-id", 100) // maxLen = 100

		// Test string
		strInput := "This is a very long string that should be truncated because it exceeds the max length of 100 characters."
		res, parts, trunc := parser.parse(ctx, strInput, "trace-1", "span-1")
		if !trunc {
			t.Error("Expected truncated to be true")
		}
		strRes, ok := res.(string)
		if !ok {
			t.Fatalf("Expected result to be string, got %T", res)
		}
		if !strings.Contains(strRes, "truncated") {
			t.Errorf("Expected result to be truncated, got: %s", strRes)
		}
		if len(parts) != 0 {
			t.Errorf("Expected parts to be empty, got %d", len(parts))
		}

		// Test genai.Content
		contentInput := &genai.Content{
			Parts: []*genai.Part{
				{Text: "Short text"},
				{Text: "Another somewhat longer text that will still fit but combined with the other it might exceed if we joined them?"},
			},
		}
		res, parts, _ = parser.parse(ctx, contentInput, "trace-1", "span-1")
		resMap, ok := res.(map[string]any)
		if !ok {
			t.Fatalf("Expected result to be map, got %T", res)
		}
		summary := resMap["text_summary"].(string)
		if !strings.Contains(summary, "Short text") {
			t.Errorf("Expected summary to contain 'Short text', got: %s", summary)
		}
		if len(parts) != 2 {
			t.Errorf("Expected 2 parts, got %d", len(parts))
		}
		// When offloader is nil, part 2 should just be inline text even if long, but truncated to maxLen
		if parts[1]["storage_mode"] != "INLINE" {
			t.Errorf("Expected storage mode INLINE, got %s", parts[1]["storage_mode"])
		}
	})

	// Test Case 2: Offloading with mock GCS server
	t.Run("OffloadingWithGCS", func(t *testing.T) {
		var uploadedData []byte
		var uploadedContentType string
		var uploadedPath string

		// Start mock GCS server
		ts := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// GCS JSON API upload endpoint is /upload/storage/v1/b/bucket/o
			// The client might send a POST request.
			if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/upload/storage/v1/b/test-bucket/o") {
				uploadedContentType = r.Header.Get("Content-Type")
				uploadedPath = r.URL.Query().Get("name")
				var err error
				uploadedData, err = io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("failed to read request body: %v", err)
				}

				// Return GCS Object response
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"kind": "storage#object", "name": "test-path", "bucket": "test-bucket"}`))
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer ts.Close()

		// Create GCS offloader
		offloader, err := newGCSOffloader(
			ctx, "test-project", "test-bucket",
			option.WithHTTPClient(ts.Client()),
			option.WithEndpoint(ts.URL),
		)
		if err != nil {
			t.Fatalf("failed to create GCS offloader: %v", err)
		}
		defer func() { _ = offloader.close() }()

		// Use hybridContentParser with offloader, maxLen=50, inlineTextLimit = 10 (which is smaller than our test text)
		parser := &hybridContentParser{
			offloader:       offloader,
			connectionID:    "conn-123",
			maxLen:          50,
			inlineTextLimit: 10,
		}

		// Test large text part
		contentInput := &genai.Content{
			Parts: []*genai.Part{
				{Text: "This is a large text that exceeds the limit of 10"},
			},
		}

		_, parts, _ := parser.parseContentObject(ctx, contentInput, "trace-x", "span-y")
		if len(parts) != 1 {
			t.Fatalf("Expected 1 part, got %d", len(parts))
		}

		part := parts[0]
		if part["storage_mode"] != "GCS_REFERENCE" {
			t.Errorf("Expected storage mode GCS_REFERENCE, got %v", part["storage_mode"])
		}

		uri := part["uri"].(string)
		if !strings.HasPrefix(uri, "gs://test-bucket/2") || !strings.HasSuffix(uri, "span-y_p0.txt") {
			t.Errorf("Unexpected GCS URI: %s", uri)
		}

		if !strings.Contains(string(uploadedData), "This is a large text") {
			t.Errorf("Uploaded data mismatch, got %s", string(uploadedData))
		}

		if !strings.Contains(uploadedContentType, "multipart/related") && !strings.Contains(uploadedContentType, "text/plain") {
			t.Errorf("Expected uploaded Content-Type to contain 'multipart/related' or 'text/plain', got '%s'", uploadedContentType)
		}

		if !strings.Contains(uploadedPath, "trace-x") || !strings.Contains(uploadedPath, "span-y") {
			t.Errorf("Expected uploaded path to contain trace-x and span-y, got '%s'", uploadedPath)
		}

		objRef, ok := part["object_ref"].(map[string]any)
		if !ok || objRef == nil {
			t.Fatal("Expected object_ref in part metadata")
		}
		if objRef["authorizer"] != "conn-123" {
			t.Errorf("Expected connection_id to be conn-123, got %v", objRef["authorizer"])
		}
	})
}
