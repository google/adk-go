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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type mockTransport struct {
	roundTrip func(*http.Request) (*http.Response, error)
}

func (m *mockTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if m.roundTrip != nil {
		resp, err := m.roundTrip(r)
		if err != nil {
			return nil, err
		}
		if resp != nil {
			if strings.Contains(r.URL.Path, "/jobs") && resp.StatusCode == http.StatusOK {
				bodyBytes, _ := io.ReadAll(resp.Body)
				_ = resp.Body.Close()
				bodyStr := strings.TrimSpace(string(bodyBytes))
				if bodyStr == "" || bodyStr == "{}" {
					resp.Body = io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`))
				} else {
					resp.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				}
			}
			return resp, nil
		}
	}
	if strings.Contains(r.URL.Path, "/jobs") {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
		}, nil
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("{}")),
	}, nil
}

// fakeBigQueryWriteServer implements a fake gRPC BigQueryWriteServer for testing.
type fakeBigQueryWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	requests      chan *storagepb.AppendRowsRequest
	streamsOpened atomic.Int32
}

func (s *fakeBigQueryWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	s.streamsOpened.Add(1)
	for {
		if err := stream.Context().Err(); err != nil {
			return err
		}
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
	ctx            context.Context
	agentName      string
	sessionID      string
	invocationID   string
	userID         string
	appName        string
	branch         string
	path           string
	functionCallID string
	state          mapState
}

func (m *mockInvocationContext) Deadline() (deadline time.Time, ok bool) { return m.ctx.Deadline() }
func (m *mockInvocationContext) Done() <-chan struct{}                   { return m.ctx.Done() }
func (m *mockInvocationContext) Err() error                              { return m.ctx.Err() }
func (m *mockInvocationContext) Value(key any) any                       { return m.ctx.Value(key) }

func (m *mockInvocationContext) AgentName() string    { return m.agentName }
func (m *mockInvocationContext) SessionID() string    { return m.sessionID }
func (m *mockInvocationContext) InvocationID() string { return m.invocationID }
func (m *mockInvocationContext) UserID() string       { return m.userID }
func (m *mockInvocationContext) Branch() string       { return m.branch }
func (m *mockInvocationContext) Path() string         { return m.path }

func (m *mockInvocationContext) UserContent() *genai.Content { return nil }

func (m *mockInvocationContext) ReadonlyState() session.ReadonlyState {
	if m.state == nil {
		return nil
	}
	return m.state
}
func (m *mockInvocationContext) AppName() string            { return m.appName }
func (m *mockInvocationContext) Artifacts() agent.Artifacts { return nil }
func (m *mockInvocationContext) State() session.State       { return nil }
func (m *mockInvocationContext) FunctionCallID() string     { return m.functionCallID }

// mapState is a session.ReadonlyState backed by a map.
type mapState map[string]any

func (s mapState) Get(key string) (any, error) {
	v, ok := s[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s mapState) All() iter.Seq2[string, any] { return maps.All(s) }

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
	t.Helper()
	return setupTestPluginWithConfig(t, nil)
}

// setupTestPluginWithConfig creates a plugin that writes to a fake BigQuery
// Storage Write server, whose requests are delivered on the returned channel.
// mutate, if set, adjusts the test configuration first. The plugin is closed
// when the test ends.
func setupTestPluginWithConfig(t *testing.T, mutate func(*Config)) (*baseplugin.Plugin, chan *storagepb.AppendRowsRequest, *mockInvocationContext) {
	t.Helper()
	p, requestsChan, mCtx := newTestPluginWithConfig(t, mutate)
	t.Cleanup(func() { _ = p.Close() })
	return p, requestsChan, mCtx
}

// newTestPluginWithConfig is setupTestPluginWithConfig for tests that close
// the plugin themselves.
func newTestPluginWithConfig(t *testing.T, mutate func(*Config)) (*baseplugin.Plugin, chan *storagepb.AppendRowsRequest, *mockInvocationContext) {
	t.Helper()
	ctx := context.Background()
	config := DefaultConfig()
	config.Enabled = true
	config.ProjectID = "test-project"
	config.DatasetID = "test_dataset"
	config.TableName = "test-table"
	if mutate != nil {
		mutate(&config)
	}

	mockTransport := &mockTransport{
		roundTrip: func(r *http.Request) (*http.Response, error) {
			if strings.Contains(r.URL.Path, "/jobs") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
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
	requestsChan := make(chan *storagepb.AppendRowsRequest, 100)
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
				return totalRows
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

// TestHasGeneratedContent verifies which streamed chunks count as the first
// token of a model call: only those with a part that holds generated content.
// The chunks that carry only metadata, such as token usage or the role of the
// content, do not.
func TestHasGeneratedContent(t *testing.T) {
	content := func(parts ...*genai.Part) *model.LLMResponse {
		return &model.LLMResponse{Partial: true, Content: &genai.Content{Role: "model", Parts: parts}}
	}
	tests := []struct {
		name string
		res  *model.LLMResponse
		want bool
	}{
		{name: "nil response", res: nil, want: false},
		{name: "usage only", res: &model.LLMResponse{Partial: true, UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}}, want: false},
		{name: "no content", res: &model.LLMResponse{Partial: true}, want: false},
		{name: "role only", res: &model.LLMResponse{Partial: true, Content: &genai.Content{Role: "model"}}, want: false},
		{name: "empty parts", res: content(), want: false},
		{name: "nil part", res: content(nil), want: false},
		{name: "empty part", res: content(&genai.Part{}), want: false},
		{name: "text", res: content(&genai.Part{Text: "Hel"}), want: true},
		{name: "inline data", res: content(&genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}), want: true},
		{name: "file data", res: content(&genai.Part{FileData: &genai.FileData{FileURI: "gs://bucket/object", MIMEType: "image/png"}}), want: true},
		{name: "function call", res: content(&genai.Part{FunctionCall: &genai.FunctionCall{Name: "lookup"}}), want: true},
		{name: "function response", res: content(&genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "lookup"}}), want: true},
		{name: "executable code", res: content(&genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "print(1)"}}), want: true},
		{name: "code execution result", res: content(&genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "1"}}), want: true},
		{name: "text after empty parts", res: content(nil, &genai.Part{}, &genai.Part{Text: "Hel"}), want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hasGeneratedContent(tt.res); got != tt.want {
				t.Errorf("hasGeneratedContent() = %v, want %v", got, tt.want)
			}
		})
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
	config.DatasetID = "test_dataset"
	config.TableName = "test-table"

	createCalled := false
	var requestBody string

	mockTransport := &mockTransport{
		roundTrip: func(r *http.Request) (*http.Response, error) {
			// Table metadata request: returns 404 Not Found to trigger creation
			if r.Method == "GET" && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
				return &http.Response{
					StatusCode: http.StatusNotFound,
					Body:       io.NopCloser(strings.NewReader(`{"error":{"code":404,"message":"Not found"}}`)),
				}, nil
			}
			// Table creation request
			if r.Method == "POST" && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables") {
				createCalled = true
				bodyBytes, _ := io.ReadAll(r.Body)
				requestBody = string(bodyBytes)
				r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			}
			if strings.Contains(r.URL.Path, "/jobs") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
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

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
	if err != nil {
		t.Fatalf("Plugin initialization error: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	if !createCalled {
		t.Fatalf("Expected table creation to be called")
	}

	var got struct {
		TimePartitioning *struct {
			Field string `json:"field"`
			Type  string `json:"type"`
		} `json:"timePartitioning"`
	}
	if err := json.Unmarshal([]byte(requestBody), &got); err != nil {
		t.Fatalf("Failed to parse create-table request body: %v", err)
	}
	if got.TimePartitioning == nil {
		t.Fatalf("Expected request body to contain timePartitioning, got: %s", requestBody)
	}
	if want := "timestamp"; got.TimePartitioning.Field != want {
		t.Errorf("timePartitioning.field = %q, want %q", got.TimePartitioning.Field, want)
	}
	if want := "DAY"; got.TimePartitioning.Type != want {
		t.Errorf("timePartitioning.type = %q, want %q", got.TimePartitioning.Type, want)
	}
}

// eventIDPattern matches the 32 lowercase hex digits of adk-python's
// uuid4().hex.
var eventIDPattern = regexp.MustCompile(`^[0-9a-f]{32}$`)

// textContent returns a message with a single text part.
func textContent(role, text string) *genai.Content {
	return &genai.Content{Role: role, Parts: []*genai.Part{{Text: text}}}
}

// textPart returns the content_parts entry of a text part, as decoded from
// Arrow.
func textPart(text string, index int64) map[string]any {
	return map[string]any{"mime_type": "text/plain", "text": text, "part_index": index, "storage_mode": "INLINE"}
}

// callbackErrors collects the errors returned by plugin callbacks.
type callbackErrors []error

func (e *callbackErrors) add(_ any, err error) { *e = append(*e, err) }

func (e callbackErrors) check(t *testing.T) {
	t.Helper()
	if err := errors.Join(e...); err != nil {
		t.Fatalf("callback error: %v", err)
	}
}

func TestEventsSchema_ColumnOrder(t *testing.T) {
	want := []string{
		"timestamp", "event_id", "event_type", "agent", "session_id", "invocation_id", "user_id",
		"trace_id", "span_id", "parent_span_id", "content", "content_parts", "attributes",
		"latency_ms", "status", "error_message", "is_truncated",
	}
	var bqColumns []string
	for _, f := range EventsSchema() {
		bqColumns = append(bqColumns, f.Name)
	}
	if diff := cmp.Diff(want, bqColumns); diff != "" {
		t.Errorf("EventsSchema() columns mismatch (-want +got):\n%s", diff)
	}
	var arrowColumns []string
	for _, f := range ArrowSchema().Fields() {
		arrowColumns = append(arrowColumns, f.Name)
	}
	if diff := cmp.Diff(want, arrowColumns); diff != "" {
		t.Errorf("ArrowSchema() columns mismatch (-want +got):\n%s", diff)
	}
	if got := EventsSchema()[1].Type; got != bq.StringFieldType {
		t.Errorf("event_id type = %v, want %v", got, bq.StringFieldType)
	}
}

func TestConfig_TableNameDefault(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "DefaultConfig", config: DefaultConfig(), want: "agent_events"},
		{name: "empty TableName", config: applyDefaults(Config{}), want: "agent_events"},
		{name: "explicit TableName", config: applyDefaults(Config{TableName: "custom"}), want: "custom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.config.TableName; got != tt.want {
				t.Errorf("TableName = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestConfig_DefaultOnFeatures(t *testing.T) {
	cleared := DefaultConfig()
	cleared.LogMultiModalContent = false
	cleared.DisableSessionMetadata = true
	cleared.DisableSensitiveKeyRedaction = true
	cleared.DisableFlushOnRunEnd = true
	cleared.DisableCreateViews = true
	disabled := Config{
		DisableMultiModalContent:     true,
		DisableSessionMetadata:       true,
		DisableSensitiveKeyRedaction: true,
		DisableFlushOnRunEnd:         true,
		DisableCreateViews:           true,
	}
	defaultDisabled := DefaultConfig()
	defaultDisabled.DisableMultiModalContent = true
	defaultDisabled.DisableSessionMetadata = true
	defaultDisabled.DisableSensitiveKeyRedaction = true
	defaultDisabled.DisableFlushOnRunEnd = true
	defaultDisabled.DisableCreateViews = true

	tests := []struct {
		name   string
		config Config
		want   bool
	}{
		{name: "DefaultConfig", config: DefaultConfig(), want: true},
		{name: "Config literal with LogMultiModalContent", config: Config{LogMultiModalContent: true}, want: true},
		{name: "DefaultConfig with the features cleared", config: cleared, want: false},
		{name: "Config literal with Disable fields", config: disabled, want: false},
		{name: "DefaultConfig with Disable fields", config: defaultDisabled, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := map[string]bool{
				"logMultiModalContent": tt.config.logMultiModalContent(),
				"logSessionMetadata":   tt.config.logSessionMetadata(),
				"redactSensitiveKeys":  tt.config.redactSensitiveKeys(),
				"flushOnRunEnd":        tt.config.flushOnRunEnd(),
				"createViews":          tt.config.createViews(),
			}
			for name, enabled := range got {
				if enabled != tt.want {
					t.Errorf("%s() = %v, want %v", name, enabled, tt.want)
				}
			}
		})
	}
}

func TestCallbacks_WriteRows(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.CustomTags = map[string]any{"team": "search", "labels": map[string]any{"tier": "gold"}}
	})
	mCtx.appName = "test-app"
	mCtx.branch = "root.test-agent"
	mCtx.functionCallID = "call-1"
	mCtx.state = mapState{"user_pref": "dark", "temp:scratch": "scratch-value", "api_key": "sk-test"}

	lookupTool := &fakeTool{name: "lookup", description: "Looks things up", longRunning: true}
	toolArgs := map[string]any{"query": "weather", "big": int64(9007199254740993)}
	llmRequest := &model.LLMRequest{
		Model:    "gemini-test",
		Contents: []*genai.Content{textContent("user", "hello")},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: textContent("", "be brief"),
			Temperature:       genai.Ptr[float32](0.5),
			Labels:            map[string]string{"env": "test"},
		},
	}
	llmResponse := &model.LLMResponse{
		Content:       textContent("model", "hi there"),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CandidatesTokenCount: 5, TotalTokenCount: 15},
		ModelVersion:  "gemini-test-001",
		FinishReason:  genai.FinishReasonStop,
		TurnComplete:  true,
	}
	ev := &session.Event{
		ID:             "event-1",
		Author:         "test-agent",
		Branch:         "root.test-agent",
		IsolationScope: "wf@0/review@2",
		NodeInfo:       &session.NodeInfo{Path: "wf@0/review@2", OutputFor: []string{"wf@0"}},
		Routes:         []string{"approve"},
		Actions:        session.EventActions{ArtifactDelta: map[string]int64{"report.txt": 3}, Escalate: true},
		LLMResponse:    model.LLMResponse{Content: textContent("model", "event text")},
	}

	start := time.Now()
	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello")))
	errs.add(p.BeforeRunCallback()(mCtx))
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeModelCallback()(mCtx, llmRequest))
	errs.add(p.AfterModelCallback()(mCtx, llmResponse, nil))
	errs.add(p.OnModelErrorCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}, errors.New("model unavailable")))
	errs.add(p.BeforeToolCallback()(mCtx, lookupTool, toolArgs))
	errs.add(p.AfterToolCallback()(mCtx, lookupTool, toolArgs, map[string]any{"answer": "sunny"}, nil))
	errs.add(p.OnToolErrorCallback()(mCtx, lookupTool, toolArgs, errors.New("lookup failed")))
	errs.add(p.OnEventCallback()(mCtx, ev))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

	want := []struct {
		eventType string
		status    string
		errMsg    string
		// content is the JSON of the content column; empty if the row has
		// none.
		content string
		parts   []map[string]any
		// attrs maps dot-separated attributes paths to their JSON, on top of
		// the attributes every row has.
		attrs map[string]string
	}{
		{
			eventType: EventUserMessageReceived,
			content:   `{"text_summary":"hello"}`,
			parts:     []map[string]any{textPart("hello", 0)},
			attrs:     map[string]string{"adk.author": `"user"`},
		},
		{eventType: EventInvocationStarting},
		{eventType: EventAgentStarting},
		{
			eventType: EventLLMRequest,
			content:   `{"prompt":[{"role":"user","content":"hello"}],"system_prompt":"be brief"}`,
			parts:     []map[string]any{textPart("hello", 0), textPart("be brief", 0)},
			attrs: map[string]string{
				"model":      `"gemini-test"`,
				"llm_config": `{"temperature":0.5}`,
				"labels":     `{"env":"test"}`,
			},
		},
		{
			eventType: EventLLMResponse,
			content:   `{"response":"text: 'hi there'","usage":{"prompt":10,"completion":5,"total":15}}`,
			parts:     []map[string]any{textPart("hi there", 0)},
			attrs: map[string]string{
				"model_version":  `"gemini-test-001"`,
				"finish_reason":  `"STOP"`,
				"partial":        `false`,
				"turn_complete":  `true`,
				"interrupted":    `false`,
				"usage_metadata": `{"prompt_token_count":10,"candidates_token_count":5,"total_token_count":15}`,
				"cache_type":     `"none"`,
			},
		},
		{
			eventType: EventLLMError,
			status:    "ERROR",
			errMsg:    "model unavailable",
			attrs:     map[string]string{"model": `"gemini-test"`},
		},
		{
			eventType: EventToolStarting,
			content:   `{"tool":"lookup","args":{"query":"weather","big":9007199254740993},"tool_origin":"UNKNOWN"}`,
			attrs: map[string]string{
				"tool_description":     `"Looks things up"`,
				"is_long_running":      `true`,
				"adk.function_call_id": `"call-1"`,
			},
		},
		{
			eventType: EventToolCompleted,
			content:   `{"tool":"lookup","result":{"answer":"sunny"},"tool_origin":"UNKNOWN"}`,
			attrs:     map[string]string{"adk.function_call_id": `"call-1"`},
		},
		{
			eventType: EventToolError,
			status:    "ERROR",
			errMsg:    "lookup failed",
			content:   `{"tool":"lookup","args":{"query":"weather","big":9007199254740993},"tool_origin":"UNKNOWN"}`,
			attrs:     map[string]string{"adk.function_call_id": `"call-1"`},
		},
		{
			eventType: EventAgentResponse,
			content:   `{"response":"text: 'event text'"}`,
			parts:     []map[string]any{textPart("event text", 0)},
			attrs: map[string]string{
				"event_author":        `"test-agent"`,
				"source_event_id":     `"event-1"`,
				"source_event_author": `"test-agent"`,
				"source_event_branch": `"root.test-agent"`,
				"adk.source_event_id": `"event-1"`,
				"adk.branch":          `"root.test-agent"`,
				"adk.node":            `{"path":"wf@0/review@2","run_id":"2","parent_run_id":"0","output_for":["wf@0"]}`,
				"adk.isolation_scope": `"wf@0/review@2"`,
				"adk.scope":           `{"id":"wf@0/review@2","kind":"node_run"}`,
				"adk.route":           `["approve"]`,
				"adk.artifact_delta":  `{"report.txt":3}`,
				"adk.escalate":        `true`,
			},
		},
		{eventType: EventAgentCompleted},
		{eventType: EventInvocationCompleted},
	}

	var wantTypes []string
	for _, w := range want {
		wantTypes = append(wantTypes, w.eventType)
	}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}

	seenIDs := map[string]bool{}
	for i, w := range want {
		row := rows[i]
		t.Run(w.eventType, func(t *testing.T) {
			eventID, _ := row["event_id"].(string)
			if !eventIDPattern.MatchString(eventID) {
				t.Errorf("event_id = %q, want 32 lowercase hex digits", eventID)
			}
			if seenIDs[eventID] {
				t.Errorf("event_id %q is not unique", eventID)
			}
			seenIDs[eventID] = true

			ts, ok := row["timestamp"].(time.Time)
			if !ok || ts.Before(start.Truncate(time.Microsecond)) || ts.After(time.Now()) {
				t.Errorf("timestamp = %v, want a time after %v", row["timestamp"], start)
			}

			wantStatus := "OK"
			if w.status != "" {
				wantStatus = w.status
			}
			wantColumns := map[string]any{
				"event_type":    w.eventType,
				"agent":         "test-agent",
				"session_id":    "sess-id",
				"invocation_id": "inv-id",
				"user_id":       "user-id",
				"status":        wantStatus,
				"is_truncated":  false,
			}
			if w.errMsg != "" {
				wantColumns["error_message"] = w.errMsg
			}
			gotColumns := map[string]any{}
			for column := range maps.Keys(wantColumns) {
				if v, ok := row[column]; ok {
					gotColumns[column] = v
				}
			}
			if v, ok := row["error_message"]; ok {
				gotColumns["error_message"] = v
			}
			if diff := cmp.Diff(wantColumns, gotColumns); diff != "" {
				t.Errorf("columns mismatch (-want +got):\n%s", diff)
			}

			gotContent, hasContent := jsonColumnValue(t, row, "content")
			switch {
			case w.content == "" && hasContent:
				t.Errorf("content = %v, want no content", gotContent)
			case w.content != "" && !hasContent:
				t.Error("content column is missing")
			case w.content != "":
				checkJSON(t, "content", gotContent, w.content)
			}
			gotParts, _ := row["content_parts"].([]map[string]any)
			if diff := cmp.Diff(w.parts, gotParts); diff != "" {
				t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
			}

			attrs := jsonObjectColumn(t, row, "attributes")
			wantAttrs := map[string]string{
				"adk.schema_version": `"1"`,
				"adk.event_id":       strconv.Quote(eventID),
				"adk.app_name":       `"test-app"`,
				"adk.invocation_id":  `"inv-id"`,
				"adk.session_id":     `"sess-id"`,
				"adk.user_id":        `"user-id"`,
				"adk.author":         `"test-agent"`,
				"root_agent_name":    `"test-agent"`,
				"custom_tags":        `{"team":"search","labels":{"tier":"gold"}}`,
				"session_metadata": `{"session_id":"sess-id","app_name":"test-app","user_id":"user-id",` +
					`"state":{"user_pref":"dark","temp:scratch":"[REDACTED]","api_key":"[REDACTED]"}}`,
			}
			maps.Copy(wantAttrs, w.attrs)
			for path, wantJSON := range wantAttrs {
				checkJSONAt(t, attrs, wantJSON, strings.Split(path, ".")...)
			}
			if v, ok := lookup(attrs, "adk", "adk_version"); ok && (v == "" || v == "(devel)" || v == "0.0.0") {
				t.Errorf("attributes.adk.adk_version = %v, want either absent or a real version", v)
			}
			_, hasBranch := lookup(attrs, "adk", "branch")
			if wantBranch := w.eventType == EventAgentResponse; hasBranch != wantBranch {
				t.Errorf("attributes.adk.branch present = %v, want %v", hasBranch, wantBranch)
			}
			_, hasCallID := lookup(attrs, "adk", "function_call_id")
			if wantCallID := strings.HasPrefix(w.eventType, "TOOL_"); hasCallID != wantCallID {
				t.Errorf("attributes.adk.function_call_id present = %v, want %v", hasCallID, wantCallID)
			}
		})
	}

	// The checks above compare numbers as float64; the column itself must
	// keep every digit.
	if raw, _ := onlyRowOfType(t, rows, EventToolStarting)["content"].(string); !strings.Contains(raw, `"big":9007199254740993`) {
		t.Errorf("TOOL_STARTING content %s lost the precision of a 64-bit integer", raw)
	}
}

func TestAttributes_OptionalSections(t *testing.T) {
	const (
		traceHex = "4bf92f3577b34da6a3ce929d0e0e4736"
		spanHex  = "00f067aa0ba902b7"
	)
	traceID, _ := trace.TraceIDFromHex(traceHex)
	spanID, _ := trace.SpanIDFromHex(spanHex)
	spanCtx := trace.NewSpanContext(trace.SpanContextConfig{TraceID: traceID, SpanID: spanID})

	// want maps an attributes path to its JSON value; "" means that the
	// path must be absent.
	tests := []struct {
		name   string
		mutate func(*Config)
		inSpan bool
		want   map[string]string
	}{
		{
			name:   "defaults",
			inSpan: true,
			want: map[string]string{
				"session_metadata": `{"session_id":"sess-id","app_name":"test-app","user_id":"user-id"}`,
				"adk.session_id":   `"sess-id"`,
				"adk.user_id":      `"user-id"`,
				"custom_tags":      "",
				"otel":             "",
			},
		},
		{
			name:   "DisableSessionMetadata",
			mutate: func(c *Config) { c.DisableSessionMetadata = true },
			want: map[string]string{
				"session_metadata":  "",
				"adk.session_id":    "",
				"adk.user_id":       "",
				"adk.invocation_id": `"inv-id"`,
			},
		},
		{
			name:   "CustomTags",
			mutate: func(c *Config) { c.CustomTags = map[string]any{"team": "search"} },
			want:   map[string]string{"custom_tags": `{"team":"search"}`},
		},
		{
			name:   "EnableOTelCorrelation inside a span",
			mutate: func(c *Config) { c.EnableOTelCorrelation = true },
			inSpan: true,
			want:   map[string]string{"otel": fmt.Sprintf(`{"span_id":%q,"trace_id":%q}`, spanHex, traceHex)},
		},
		{
			name:   "EnableOTelCorrelation outside a span",
			mutate: func(c *Config) { c.EnableOTelCorrelation = true },
			want:   map[string]string{"otel": ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, tt.mutate)
			mCtx.appName = "test-app"
			if tt.inSpan {
				mCtx.ctx = trace.ContextWithSpanContext(mCtx.ctx, spanCtx)
			}
			if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello")); err != nil {
				t.Fatalf("OnUserMessageCallback error: %v", err)
			}
			p.AfterRunCallback()(mCtx)

			row := onlyRowOfType(t, collectRows(t, requests, untilEventType(EventInvocationCompleted)), EventUserMessageReceived)
			// The row belongs to the invocation span. Inside an ambient
			// span, the invocation continues the ambient trace under the
			// ambient span; outside one, it starts a trace of its own.
			gotTrace, _ := row["trace_id"].(string)
			gotSpan, _ := row["span_id"].(string)
			gotParent, hasParent := row["parent_span_id"].(string)
			switch {
			case tt.inSpan && (gotTrace != traceHex || gotParent != spanHex):
				t.Errorf("trace_id, parent_span_id = %q, %q, want %q, %q", gotTrace, gotParent, traceHex, spanHex)
			case !tt.inSpan && (!eventIDPattern.MatchString(gotTrace) || hasParent):
				t.Errorf("trace_id, parent_span_id = %q, %q, want 32 hex digits and no parent", gotTrace, gotParent)
			}
			if !spanIDPattern.MatchString(gotSpan) || gotSpan == spanHex {
				t.Errorf("span_id = %q, want 16 hex digits of a span of the invocation", gotSpan)
			}
			attrs := jsonObjectColumn(t, row, "attributes")
			for path, wantJSON := range tt.want {
				keys := strings.Split(path, ".")
				if wantJSON == "" {
					if v, ok := lookup(attrs, keys...); ok {
						t.Errorf("attributes.%s = %v, want it absent", path, v)
					}
					continue
				}
				checkJSONAt(t, attrs, wantJSON, keys...)
			}
		})
	}
}

func TestSensitiveKeyRedaction(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		want    string
		wantURI string
	}{
		{
			name:    "redacted by default",
			want:    redactedValue,
			wantURI: "https://example.com/i.png?api_key=%5BREDACTED%5D#access_token=%5BREDACTED%5D",
		},
		{
			name:    "DisableSensitiveKeyRedaction keeps values",
			mutate:  func(c *Config) { c.DisableSensitiveKeyRedaction = true },
			want:    "sk-live",
			wantURI: "https://example.com/i.png?api_key=sk-live#access_token=sk-live",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, tt.mutate)
			mCtx.state = mapState{"temp:token": "sk-live", "theme": "dark"}
			args := map[string]any{
				"api_key": "sk-live",
				"query":   "weather",
				"headers": map[string]any{"Authorization": "sk-live", "Accept": "json"},
				"items":   []any{map[string]any{"client-secret": "sk-live"}},
				"json":    `{"api_key":"sk-live"}`,
			}
			var errs callbackErrors
			errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, args))
			errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
				Role: "user",
				Parts: []*genai.Part{
					{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "https://example.com/i.png?api_key=sk-live#access_token=sk-live"}},
				},
			}))
			errs.add(p.OnModelErrorCallback()(mCtx, &model.LLMRequest{}, errors.New(
				`auth failed: api_key=sk-live Bearer sk-live https://example.com/v1?key=sk-live {"client_secret":"sk-live"}`,
			)))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			row := onlyRowOfType(t, rows, EventToolStarting)
			checkJSONAt(t, jsonObjectColumn(t, row, "content"), fmt.Sprintf(
				`{"api_key":%[1]q,"query":"weather","headers":{"Authorization":%[1]q,"Accept":"json"},"items":[{"client-secret":%[1]q}],"json":%[2]q}`,
				tt.want, fmt.Sprintf(`{"api_key":%q}`, tt.want),
			), "args")
			checkJSONAt(t, jsonObjectColumn(t, row, "attributes"),
				fmt.Sprintf(`{"temp:token":%q,"theme":"dark"}`, tt.want), "session_metadata", "state")

			userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
			parts, _ := userRow["content_parts"].([]map[string]any)
			if len(parts) != 1 || parts[0]["uri"] != tt.wantURI {
				t.Errorf("content_parts uri = %v, want %q", parts, tt.wantURI)
			}

			errRow := onlyRowOfType(t, rows, EventLLMError)
			errMsg, _ := errRow["error_message"].(string)
			if tt.want == redactedValue && strings.Contains(errMsg, "sk-live") {
				t.Errorf("error_message %q leaked credential sk-live", errMsg)
			}
			if tt.want != redactedValue && !strings.Contains(errMsg, "sk-live") {
				t.Errorf("error_message %q did not preserve sk-live when redaction was disabled", errMsg)
			}
		})
	}
}

// TestSensitiveKeyRedaction_GoNamesAndStatePrefixes checks that sensitive keys
// match regardless of case and separators, so camelCase keys and Go field names
// are redacted, and that session keys with a user: or app: prefix match on the
// name after the prefix.
func TestSensitiveKeyRedaction_GoNamesAndStatePrefixes(t *testing.T) {
	const secret = "SECRET123"
	p, requests, mCtx := setupTestPlugin(t)
	mCtx.state = mapState{"user:api_key": secret, "app:clientSecret": secret, "user:theme": "dark"}
	args := map[string]any{"accessToken": secret, "x-goog-api-key": secret, "tokenCount": 3}
	result := map[string]any{"config": struct {
		APIKey       string
		ClientSecret string
		Region       string
	}{APIKey: secret, ClientSecret: secret, Region: "us"}}
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, args))
	errs.add(p.AfterToolCallback()(mCtx, &fakeTool{name: "lookup"}, args, result, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	startRow := onlyRowOfType(t, rows, EventToolStarting)
	checkJSONAt(t, jsonObjectColumn(t, startRow, "content"),
		`{"accessToken":"[REDACTED]","x-goog-api-key":"[REDACTED]","tokenCount":3}`, "args")
	checkJSONAt(t, jsonObjectColumn(t, startRow, "attributes"),
		`{"user:api_key":"[REDACTED]","app:clientSecret":"[REDACTED]","user:theme":"dark"}`, "session_metadata", "state")
	checkJSONAt(t, jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolCompleted), "content"),
		`{"config":{"APIKey":"[REDACTED]","ClientSecret":"[REDACTED]","Region":"us"}}`, "result")
	for _, row := range rows {
		if s := fmt.Sprint(row); strings.Contains(s, secret) {
			t.Errorf("%v row leaked %s: %s", row["event_type"], secret, s)
		}
	}
}

// TestCredentialsEnvelopeIsRedactedWhole checks that a credentials envelope,
// such as an AWS STS result, is redacted as a whole in tool arguments and
// results, whatever its fields are named, and that an AWS secret access key
// or session token is redacted outside an envelope too.
func TestCredentialsEnvelopeIsRedactedWhole(t *testing.T) {
	const secret = "SECRET123"
	envelope := func() map[string]any {
		return map[string]any{
			"AccessKeyId":     "EXAMPLEKEYID",
			"SecretAccessKey": secret,
			"SessionToken":    secret,
			"Expiration":      "2026-10-14T00:00:00Z",
		}
	}
	p, requests, mCtx := setupTestPlugin(t)
	args := map[string]any{"role": "reader", "credential": envelope()}
	result := map[string]any{
		"Credentials": envelope(),
		"keys":        map[string]any{"AccessKeyId": "EXAMPLEKEYID", "secret_access_key": secret, "session-token": secret, "aws_secret_access_key": secret, "AWS_SESSION_TOKEN": secret},
	}
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "assume_role"}, args))
	errs.add(p.AfterToolCallback()(mCtx, &fakeTool{name: "assume_role"}, args, result, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	checkJSONAt(t, jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolStarting), "content"),
		`{"role":"reader","credential":"[REDACTED]"}`, "args")
	checkJSONAt(t, jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolCompleted), "content"),
		`{"Credentials":"[REDACTED]","keys":{"AccessKeyId":"EXAMPLEKEYID","secret_access_key":"[REDACTED]","session-token":"[REDACTED]","aws_secret_access_key":"[REDACTED]","AWS_SESSION_TOKEN":"[REDACTED]"}}`, "result")
	for _, row := range rows {
		if s := fmt.Sprint(row); strings.Contains(s, secret) {
			t.Errorf("%v row leaked %s: %s", row["event_type"], secret, s)
		}
	}
}

func TestToolArgs_UnloggableValuesKeepTheRow(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	cyclic := map[string]any{"name": "loop"}
	cyclic["self"] = cyclic
	args := map[string]any{
		"big":    int64(9007199254740993),
		"bad":    panicMarshaler{},
		"cyclic": cyclic,
		"nan":    math.NaN(),
	}
	if _, err := p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, args); err != nil {
		t.Fatalf("BeforeToolCallback error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	row := onlyRowOfType(t, collectRows(t, requests, untilEventType(EventInvocationCompleted)), EventToolStarting)
	checkJSONAt(t, jsonObjectColumn(t, row, "content"),
		`{"big":9007199254740993,"bad":"[UNSUPPORTED_OBJECT]","cyclic":{"name":"loop","self":"[CIRCULAR_REFERENCE]"},"nan":"NaN"}`,
		"args")
	if raw, _ := row["content"].(string); !strings.Contains(raw, `"big":9007199254740993`) {
		t.Errorf("content %s lost the precision of a 64-bit integer", raw)
	}
	if got := row["is_truncated"]; got != true {
		t.Errorf("is_truncated = %v, want true for a dropped value", got)
	}
}

func TestContentFormatter(t *testing.T) {
	const secret = "formatter-secret-detail"
	failing := func(any, string) (any, error) { return nil, errors.New(secret) }
	tests := []struct {
		name          string
		formatter     ContentFormatter
		debug         bool
		denyContent   bool
		wantContent   string
		wantFmtFailed bool
		wantErrMsg    string
		wantParts     bool
		wantLog       string
		wantNoLog     string
	}{
		{
			name: "replaces the content",
			formatter: func(_ any, eventType string) (any, error) {
				return map[string]any{"event": eventType, "redacted": true}, nil
			},
			wantContent: `{"event":"USER_MESSAGE_RECEIVED","redacted":true}`,
		},
		{
			name:        "passes the content through",
			formatter:   func(payload any, _ string) (any, error) { return payload, nil },
			wantContent: `{"text_summary":"hello"}`,
			wantParts:   true,
		},
		{
			name:          "error fails closed",
			formatter:     failing,
			wantContent:   `"[FORMATTER_FAILED]"`,
			wantFmtFailed: true,
			wantErrMsg:    "content_formatter raised *errors.errorString",
			wantLog:       "ContentFormatter failed for event USER_MESSAGE_RECEIVED",
			wantNoLog:     secret,
		},
		{
			name:          "debug mode logs error type and length without raw error text",
			formatter:     failing,
			debug:         true,
			wantContent:   `"[FORMATTER_FAILED]"`,
			wantFmtFailed: true,
			wantErrMsg:    "content_formatter raised *errors.errorString",
			wantLog:       fmt.Sprintf("err_len=%d", len(secret)),
			wantNoLog:     secret,
		},
		{
			name:          "panic fails closed",
			formatter:     func(any, string) (any, error) { panic(secret) },
			wantContent:   `"[FORMATTER_FAILED]"`,
			wantFmtFailed: true,
			wantErrMsg:    "content_formatter panicked",
			wantLog:       "ContentFormatter panicked for event USER_MESSAGE_RECEIVED",
			wantNoLog:     secret,
		},
		{
			name:          "error with content denied sets formatter_failed with diagnostic error_message",
			formatter:     failing,
			denyContent:   true,
			wantFmtFailed: true,
			wantErrMsg:    "content_formatter raised *errors.errorString",
			wantLog:       "ContentFormatter failed for event USER_MESSAGE_RECEIVED",
			wantNoLog:     secret,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &recordingLogger{}
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.ContentFormatter = tt.formatter
				c.DebugContentFormatterErrors = tt.debug
				c.Logger = logger
				if tt.denyContent {
					c.PayloadColumnDenylist = []string{columnContent}
				}
			})
			gotMsg, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello"))
			if err != nil {
				t.Fatalf("OnUserMessageCallback error: %v", err)
			}
			if gotMsg != nil {
				t.Errorf("OnUserMessageCallback returned %v, want nil so subsequent plugins run", gotMsg)
			}
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			if _, ok := onlyRowOfType(t, rows, EventInvocationCompleted)["content"]; ok {
				t.Error("INVOCATION_COMPLETED row has content; the formatter must not run for events without content")
			}
			row := onlyRowOfType(t, rows, EventUserMessageReceived)
			if tt.wantContent == "" {
				if _, ok := row["content"]; ok {
					t.Errorf("content = %v, want absent when content column is denied", row["content"])
				}
			} else {
				gotContent, _ := jsonColumnValue(t, row, "content")
				checkJSON(t, "content", gotContent, tt.wantContent)
			}
			if got := row["status"]; got != "OK" {
				t.Errorf("status = %v, want OK", got)
			}
			if tt.wantErrMsg != "" {
				if gotErrMsg, ok := row["error_message"]; !ok || gotErrMsg != tt.wantErrMsg {
					t.Errorf("error_message = (%v, %v), want (%q, true)", gotErrMsg, ok, tt.wantErrMsg)
				}
			} else {
				if gotErrMsg, ok := row["error_message"]; ok {
					t.Errorf("error_message = %q, want absent", gotErrMsg)
				}
			}
			if got := row["is_truncated"]; got != false {
				t.Errorf("is_truncated = %v, want false", got)
			}
			if _, got := row["content_parts"]; got != tt.wantParts {
				t.Errorf("content_parts present = %v, want %v", got, tt.wantParts)
			}
			attrs := jsonObjectColumn(t, row, "attributes")
			gotFmtFailed, hasFmtFailed := lookup(attrs, "formatter_failed")
			if hasFmtFailed != tt.wantFmtFailed || (tt.wantFmtFailed && gotFmtFailed != true) {
				t.Errorf("attributes.formatter_failed = (%v, %v), want (%v, %v)", gotFmtFailed, hasFmtFailed, true, tt.wantFmtFailed)
			}
			if tt.wantLog != "" && !logger.contains(tt.wantLog) {
				t.Errorf("log does not contain %q:\n%s", tt.wantLog, logger.String())
			}
			if tt.wantNoLog != "" && logger.contains(tt.wantNoLog) {
				t.Errorf("log contains %q:\n%s", tt.wantNoLog, logger.String())
			}
		})
	}
}

func TestContentFormatter_PreservesExistingErrorMessage(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.ContentFormatter = func(any, string) (any, error) { return nil, errors.New("formatter secret") }
	})
	if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "partial")}, errors.New("model failed")); err != nil {
		t.Fatalf("AfterModelCallback error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	row := onlyRowOfType(t, collectRows(t, requests, untilInvocationEnd), EventLLMResponse)
	if got := row["status"]; got != "ERROR" {
		t.Errorf("status = %v, want ERROR", got)
	}
	const wantErr = "model failed; content_formatter raised *errors.errorString"
	if got := row["error_message"]; got != wantErr {
		t.Errorf("error_message = %q, want %q", got, wantErr)
	}
	checkJSONAt(t, jsonObjectColumn(t, row, "attributes"), `true`, "formatter_failed")
}

func TestShouldLogEvent(t *testing.T) {
	tests := []struct {
		name      string
		allow     []string
		deny      []string
		eventType string
		want      bool
	}{
		{name: "no filters", eventType: EventLLMRequest, want: true},
		{name: "allowlisted", allow: []string{EventLLMRequest}, eventType: EventLLMRequest, want: true},
		{name: "not allowlisted", allow: []string{EventToolStarting}, eventType: EventLLMRequest, want: false},
		{name: "denylisted", deny: []string{EventLLMRequest}, eventType: EventLLMRequest, want: false},
		{name: "denylist overrides allowlist", allow: []string{EventLLMRequest}, deny: []string{EventLLMRequest}, eventType: EventLLMRequest, want: false},
		{name: "denylist leaves other events", deny: []string{EventLLMRequest}, eventType: EventToolStarting, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &analyticsPlugin{allowlist: newSet(tt.allow...), denylist: newSet(tt.deny...)}
			if got := p.shouldLogEvent(tt.eventType); got != tt.want {
				t.Errorf("shouldLogEvent(%q) = %v, want %v", tt.eventType, got, tt.want)
			}
		})
	}
}

func TestEventFiltering_DenylistOverridesAllowlist(t *testing.T) {
	logger := &recordingLogger{}
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.Logger = logger
		c.EventAllowlist = []string{EventLLMRequest, EventToolStarting, "TOOL_STARTED"}
		c.EventDenylist = []string{EventLLMRequest}
	})
	var errs callbackErrors
	errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{Model: "gemini-test"}))
	errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	// Rows arrive in order, so a logged LLM_REQUEST row would arrive first.
	rows := collectRows(t, requests, untilEventType(EventToolStarting))
	if diff := cmp.Diff([]string{EventToolStarting}, eventTypes(rows)); diff != "" {
		t.Errorf("event types mismatch (-want +got):\n%s", diff)
	}
	if want := `unknown event_type "TOOL_STARTED"`; !logger.contains(want) {
		t.Errorf("log does not contain %q:\n%s", want, logger.String())
	}
	if logger.contains(`unknown event_type "` + EventLLMRequest) {
		t.Errorf("log warns about a known event type:\n%s", logger.String())
	}
}

func TestNewPlugin_RejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr string
	}{
		{
			name:    "unknown payload column",
			mutate:  func(c *Config) { c.PayloadColumnDenylist = []string{"error_message"} },
			wantErr: `unsupported PayloadColumnDenylist entry "error_message"`,
		},
		{
			name: "custom metadata without the attributes column",
			mutate: func(c *Config) {
				c.PayloadColumnDenylist = []string{columnAttributes}
				c.CustomMetadataAllowlist = []string{"a2a:*"}
			},
			wantErr: "CustomMetadataAllowlist cannot be used",
		},
		{
			name:    "MaxContentLen less than -1",
			mutate:  func(c *Config) { c.MaxContentLen = -2 },
			wantErr: "MaxContentLen must be -1 or >= 1",
		},
		{
			name:    "negative BatchSize",
			mutate:  func(c *Config) { c.BatchSize = -1 },
			wantErr: "BatchSize and QueueMaxSize must be >= 1",
		},
		{
			name:    "negative BatchFlushIntv",
			mutate:  func(c *Config) { c.BatchFlushIntv = -time.Second },
			wantErr: "BatchFlushIntv and ShutdownTimeout must be > 0",
		},
		{
			name:    "negative ShutdownTimeout",
			mutate:  func(c *Config) { c.ShutdownTimeout = -time.Second },
			wantErr: "BatchFlushIntv and ShutdownTimeout must be > 0",
		},
		{
			name:    "negative QueueMaxSize",
			mutate:  func(c *Config) { c.QueueMaxSize = -1 },
			wantErr: "BatchSize and QueueMaxSize must be >= 1",
		},
		{
			name:    "negative RetryConfig.MaxRetries",
			mutate:  func(c *Config) { c.RetryConfig.MaxRetries = -1 },
			wantErr: "invalid RetryConfig",
		},
		{
			name:    "Multiplier less than 1",
			mutate:  func(c *Config) { c.RetryConfig.Multiplier = 0.5 },
			wantErr: "invalid RetryConfig",
		},
		{
			name: "MaxDelay less than InitialDelay",
			mutate: func(c *Config) {
				c.RetryConfig.InitialDelay = 10 * time.Second
				c.RetryConfig.MaxDelay = 5 * time.Second
			},
			wantErr: "invalid RetryConfig",
		},
		{
			name:    "invalid ViewPrefix with SQL injection characters",
			mutate:  func(c *Config) { c.ViewPrefix = "v` OR 1=1 --" },
			wantErr: "invalid ViewPrefix",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GOOGLE_APPLICATION_CREDENTIALS",
				filepath.Join(t.TempDir(), "missing-credentials.json"))
			ctx := context.Background()
			config := DefaultConfig()
			config.ProjectID = "test-project"
			tt.mutate(&config)
			// Validation happens before any client is created or used.
			if _, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, config); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v, want an error containing %q", err, tt.wantErr)
			}
			if _, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, nil, nil); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want an error containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestPayloadColumns(t *testing.T) {
	tests := []struct {
		name        string
		mutate      func(*Config)
		wantContent bool
		wantParts   bool
		wantAttrs   bool
	}{
		{name: "default", wantContent: true, wantParts: true, wantAttrs: true},
		{
			name:        "LogMultiModalContent cleared",
			mutate:      func(c *Config) { c.LogMultiModalContent = false },
			wantContent: true, wantAttrs: true,
		},
		{
			name:        "DisableMultiModalContent",
			mutate:      func(c *Config) { c.DisableMultiModalContent = true },
			wantContent: true, wantAttrs: true,
		},
		{
			name:        "content_parts denied",
			mutate:      func(c *Config) { c.PayloadColumnDenylist = []string{columnContentParts} },
			wantContent: true, wantAttrs: true,
		},
		{
			name:      "content denied",
			mutate:    func(c *Config) { c.PayloadColumnDenylist = []string{columnContent} },
			wantParts: true, wantAttrs: true,
		},
		{
			name:      "content and content_parts denied",
			mutate:    func(c *Config) { c.PayloadColumnDenylist = []string{columnContent, columnContentParts} },
			wantAttrs: true,
		},
		{
			name:        "attributes denied",
			mutate:      func(c *Config) { c.PayloadColumnDenylist = []string{columnAttributes} },
			wantContent: true, wantParts: true,
		},
		{
			name:        "latency_ms denied",
			mutate:      func(c *Config) { c.PayloadColumnDenylist = []string{columnLatencyMS} },
			wantContent: true, wantParts: true, wantAttrs: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatterCalls := 0
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.ContentFormatter = func(payload any, _ string) (any, error) {
					formatterCalls++
					return payload, nil
				}
				if tt.mutate != nil {
					tt.mutate(c)
				}
			})
			if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello")); err != nil {
				t.Fatalf("OnUserMessageCallback error: %v", err)
			}
			p.AfterRunCallback()(mCtx)

			row := onlyRowOfType(t, collectRows(t, requests, untilEventType(EventInvocationCompleted)), EventUserMessageReceived)
			for column, want := range map[string]bool{
				columnContent:      tt.wantContent,
				columnContentParts: tt.wantParts,
				columnAttributes:   tt.wantAttrs,
			} {
				if _, got := row[column]; got != want {
					t.Errorf("%s present = %v, want %v", column, got, want)
				}
			}
			// The formatter runs only when a content column is written.
			if got, want := formatterCalls > 0, tt.wantContent || tt.wantParts; got != want {
				t.Errorf("ContentFormatter called = %v, want %v", got, want)
			}
		})
	}
}

func TestCustomMetadataFilter(t *testing.T) {
	md := map[string]any{"trace_tag": "t", "a2a:task_id": "task-1", "a2a:context_id": "ctx-1", "private": "p"}
	tests := []struct {
		name      string
		allowlist []string
		want      map[string]any
	}{
		{name: "empty allowlist captures nothing"},
		{name: "exact entry", allowlist: []string{"trace_tag"}, want: map[string]any{"trace_tag": "t"}},
		{name: "prefix entry", allowlist: []string{"a2a:*"}, want: map[string]any{"a2a:task_id": "task-1", "a2a:context_id": "ctx-1"}},
		{name: "entries without * match exactly", allowlist: []string{"a2a:"}},
		{name: "no match", allowlist: []string{"other"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newCustomMetadataFilter(tt.allowlist).apply(md)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("apply() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCustomMetadataAllowlist(t *testing.T) {
	tests := []struct {
		name      string
		allowlist []string
		want      string // JSON of attributes.custom_metadata; empty if absent
	}{
		{name: "allowlisted keys are written", allowlist: []string{"a2a:*"}, want: `{"a2a:task_id":"task-1"}`},
		{name: "nothing is written by default"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.CustomMetadataAllowlist = tt.allowlist })
			md := map[string]any{"a2a:task_id": "task-1", "private": "p"}
			var errs callbackErrors
			gotEv, evErr := p.OnEventCallback()(mCtx, &session.Event{
				Author:      "test-agent",
				LLMResponse: model.LLMResponse{Content: textContent("model", "hi"), CustomMetadata: md},
			})
			errs.add(gotEv, evErr)
			if gotEv != nil {
				t.Errorf("OnEventCallback returned %v, want nil so subsequent plugins run", gotEv)
			}
			errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{CustomMetadata: md}, nil))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			for _, eventType := range []string{EventAgentResponse, EventLLMResponse} {
				attrs := jsonObjectColumn(t, onlyRowOfType(t, rows, eventType), "attributes")
				got, ok := lookup(attrs, "custom_metadata")
				switch {
				case tt.want == "" && ok:
					t.Errorf("%s: custom_metadata = %v, want none", eventType, got)
				case tt.want != "":
					checkJSONAt(t, attrs, tt.want, "custom_metadata")
				}
			}
		})
	}
}

func TestFlushOnRunEnd(t *testing.T) {
	tests := []struct {
		name           string
		disable        bool
		wantFlushAtEnd bool
	}{
		{name: "flushes when the run ends", wantFlushAtEnd: true},
		{name: "DisableFlushOnRunEnd defers the write", disable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				// Only the run-end flush and Close write rows.
				c.BatchFlushIntv = time.Hour
				c.DisableFlushOnRunEnd = tt.disable
			})
			if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello")); err != nil {
				t.Fatalf("OnUserMessageCallback error: %v", err)
			}
			p.AfterRunCallback()(mCtx)
			if !tt.wantFlushAtEnd {
				expectNoRequest(t, requests, 300*time.Millisecond)
				if err := p.Close(); err != nil {
					t.Fatalf("Close() error: %v", err)
				}
			}
			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			if diff := cmp.Diff([]string{EventUserMessageReceived, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
				t.Errorf("event types mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFinalResponseToolNames(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.FinalResponseToolNames = []string{"final_answer"} })
	mCtx.functionCallID = "call-7"
	final := &fakeTool{name: "final_answer"}
	answer := map[string]any{"answer": "42"}
	var errs callbackErrors
	errs.add(p.AfterToolCallback()(mCtx, final, answer, map[string]any{}, nil))
	errs.add(p.AfterToolCallback()(mCtx, final, answer, nil, errors.New("rejected")))
	errs.add(p.AfterToolCallback()(mCtx, &fakeTool{name: "lookup"}, answer, map[string]any{}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	wantTypes := []string{EventToolCompleted, EventAgentResponse, EventToolError, EventToolCompleted, EventInvocationCompleted}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	row := onlyRowOfType(t, rows, EventAgentResponse)
	checkJSON(t, "content", jsonObjectColumn(t, row, "content"), `{"response":{"answer":"42"}}`)
	attrs := jsonObjectColumn(t, row, "attributes")
	checkJSONAt(t, attrs, `"final_answer"`, "source_tool")
	checkJSONAt(t, attrs, `"call-7"`, "adk", "function_call_id")
	if got := row["status"]; got != "OK" {
		t.Errorf("status = %v, want OK", got)
	}
}

// TestToolFailure_WritesErrorRows replays the callbacks ADK runs for a failed
// tool call: BeforeToolCallback, OnToolErrorCallback, then
// AfterToolCallback with the same error. The call ends with a single
// TOOL_ERROR row, as in adk-python, which runs no after_tool_callback for
// failed calls.
func TestToolFailure_WritesErrorRows(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	mCtx.functionCallID = "call-1"
	lookupTool := &fakeTool{name: "lookup"}
	args := map[string]any{"query": "weather"}
	toolErr := errors.New("lookup failed")
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, lookupTool, args))
	errs.add(p.OnToolErrorCallback()(mCtx, lookupTool, args, toolErr))
	errs.add(p.AfterToolCallback()(mCtx, lookupTool, args, nil, toolErr))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventToolStarting, EventToolError, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	row := onlyRowOfType(t, rows, EventToolError)
	if got := row["status"]; got != "ERROR" {
		t.Errorf("status = %v, want ERROR", got)
	}
	if got := row["error_message"]; got != "lookup failed" {
		t.Errorf("error_message = %v, want %q", got, "lookup failed")
	}
	checkJSON(t, "content", jsonObjectColumn(t, row, "content"),
		`{"tool":"lookup","args":{"query":"weather"},"tool_origin":"UNKNOWN"}`)
	if got, want := row["span_id"], onlyRowOfType(t, rows, EventToolStarting)["span_id"]; got != want {
		t.Errorf("TOOL_ERROR span_id = %v, want the TOOL_STARTING span_id %v", got, want)
	}
	if _, ok := row[columnLatencyMS]; !ok {
		t.Error("TOOL_ERROR row has no latency_ms")
	}
}

// TestToolConfirmationRequired_CompletesCall replays the callbacks ADK runs
// for a call of a tool that waits for the user's confirmation: the tool
// fails with tool.ErrConfirmationRequired before it runs, so ADK runs
// BeforeToolCallback, OnToolErrorCallback, then AfterToolCallback with the
// same error. As in adk-python, whose tools return the confirmation request
// as their result, the call ends with a TOOL_COMPLETED row whose result is
// the error that ADK sends the model, and no TOOL_ERROR row. A final
// response tool writes its AGENT_RESPONSE row when the confirmed call runs.
func TestToolConfirmationRequired_CompletesCall(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.FinalResponseToolNames = []string{"deploy"} })
	mCtx.functionCallID = "call-1"
	deployTool := &fakeTool{name: "deploy"}
	args := map[string]any{"region": "us-central1"}
	toolErr := fmt.Errorf("error tool %q %w", "deploy", tool.ErrConfirmationRequired)
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, deployTool, args))
	if res, err := p.OnToolErrorCallback()(mCtx, deployTool, args, toolErr); res != nil || err != nil {
		t.Errorf("OnToolErrorCallback() = %v, %v; want nil, nil", res, err)
	}
	errs.add(p.AfterToolCallback()(mCtx, deployTool, args, nil, toolErr))
	// The user confirms the call, and ADK runs the tool.
	errs.add(p.BeforeToolCallback()(mCtx, deployTool, args))
	errs.add(p.AfterToolCallback()(mCtx, deployTool, args, map[string]any{"cluster_id": "c-123"}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	wantTypes := []string{
		EventToolStarting, EventToolCompleted, // waits for confirmation
		EventToolStarting, EventToolCompleted, EventAgentResponse, // confirmed
		EventInvocationCompleted,
	}
	if diff := cmp.Diff(wantTypes, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	row := rowsOfType(rows, EventToolCompleted)[0]
	if got := row["status"]; got != "OK" {
		t.Errorf("status = %v, want OK", got)
	}
	if got, ok := row["error_message"]; ok {
		t.Errorf("error_message = %v, want none", got)
	}
	checkJSON(t, "content", jsonObjectColumn(t, row, "content"),
		`{"tool":"deploy","result":{"error":"error tool \"deploy\" requires confirmation, please approve or reject"},"tool_origin":"UNKNOWN"}`)
	checkJSONAt(t, jsonObjectColumn(t, row, "attributes"), `"call-1"`, "adk", "function_call_id")
	if got, want := row["span_id"], rowsOfType(rows, EventToolStarting)[0]["span_id"]; got != want {
		t.Errorf("TOOL_COMPLETED span_id = %v, want the TOOL_STARTING span_id %v", got, want)
	}
	if _, ok := row[columnLatencyMS]; !ok {
		t.Error("TOOL_COMPLETED row has no latency_ms")
	}
	checkJSON(t, "AGENT_RESPONSE content", jsonObjectColumn(t, onlyRowOfType(t, rows, EventAgentResponse), "content"),
		`{"response":{"region":"us-central1"}}`)
}

func TestToolCallbacks_NilToolDoesNotPanic(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, nil, map[string]any{"q": "1"}))
	errs.add(p.OnToolErrorCallback()(mCtx, nil, map[string]any{"q": "1"}, errors.New("tool not found")))
	errs.add(p.AfterToolCallback()(mCtx, nil, map[string]any{"q": "1"}, map[string]any{"ok": true}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventToolStarting, EventToolError, EventToolCompleted, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	startRow := onlyRowOfType(t, rows, EventToolStarting)
	errRow := onlyRowOfType(t, rows, EventToolError)
	doneRow := onlyRowOfType(t, rows, EventToolCompleted)
	if errRow["span_id"] != startRow["span_id"] || doneRow["span_id"] != startRow["span_id"] {
		t.Errorf("span_ids = (%v, %v, %v), want all equal to TOOL_STARTING span_id",
			startRow["span_id"], errRow["span_id"], doneRow["span_id"])
	}
	checkJSON(t, "TOOL_STARTING content", jsonObjectColumn(t, startRow, "content"),
		`{"tool":"","args":{"q":"1"},"tool_origin":"UNKNOWN"}`)
	checkJSON(t, "TOOL_ERROR content", jsonObjectColumn(t, errRow, "content"),
		`{"tool":"","args":{"q":"1"},"tool_origin":"UNKNOWN"}`)
	checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, doneRow, "content"),
		`{"tool":"","result":{"ok":true},"tool_origin":"UNKNOWN"}`)
}

func TestAfterModelCallback_Status(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	responses := []struct {
		res *model.LLMResponse
		err error
	}{
		{res: &model.LLMResponse{ModelVersion: "ok"}},
		{res: &model.LLMResponse{
			ModelVersion: "function-call",
			Content:      &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "lookup"}}}},
		}},
		{res: &model.LLMResponse{ModelVersion: "callback-error"}, err: errors.New("stream broken")},
		{res: &model.LLMResponse{ModelVersion: "error-message", ErrorCode: "SAFETY", ErrorMessage: "blocked by safety filters"}},
		{res: &model.LLMResponse{ModelVersion: "error-code", ErrorCode: "MAX_TOKENS"}},
		{res: &model.LLMResponse{ModelVersion: "message-only", ErrorMessage: "quota exceeded"}},
		{res: &model.LLMResponse{ModelVersion: "partial", Partial: true}},
	}
	var errs callbackErrors
	for _, r := range responses {
		errs.add(p.AfterModelCallback()(mCtx, r.res, r.err))
	}
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	want := map[string]map[string]any{
		"ok":             {"status": "OK"},
		"function-call":  {"status": "OK", "content": `{"response":"call: lookup"}`},
		"callback-error": {"status": "ERROR", "error_message": "stream broken"},
		"error-message":  {"status": "OK", "error_message": "blocked by safety filters"},
		"error-code":     {"status": "OK"},
		"message-only":   {"status": "OK", "error_message": "quota exceeded"},
	}
	rows := collectRows(t, requests, untilEventType(EventInvocationError))
	invErr := onlyRowOfType(t, rows, EventInvocationError)
	if invErr["error_message"] != "stream broken" {
		t.Errorf("INVOCATION_ERROR error_message = %v, want %q", invErr["error_message"], "stream broken")
	}
	got := map[string]map[string]any{}
	for _, row := range rowsOfType(rows, EventLLMResponse) {
		version, _ := lookup(jsonObjectColumn(t, row, "attributes"), "model_version")
		columns := map[string]any{"status": row["status"]}
		if msg, ok := row["error_message"]; ok {
			columns["error_message"] = msg
		}
		if c, ok := row["content"]; ok {
			columns["content"] = c
		}
		got[fmt.Sprint(version)] = columns
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LLM_RESPONSE rows mismatch (-want +got):\n%s", diff)
	}
}

func TestTruncation(t *testing.T) {
	tests := []struct {
		name        string
		mutate      func(*Config)
		wantSummary string
		wantErrMsg  string
		wantTrunc   bool
	}{
		{
			name:        "MaxContentLen truncates",
			mutate:      func(c *Config) { c.MaxContentLen = 5 },
			wantSummary: "hello...[TRUNCATED]",
			wantErrMsg:  "model...[TRUNCATED]",
			wantTrunc:   true,
		},
		{
			name: "CaptureFullMessageContent keeps the full text",
			mutate: func(c *Config) {
				c.MaxContentLen = 5
				c.CaptureFullMessageContent = true
			},
			wantSummary: "hello world",
			wantErrMsg:  "model unavailable",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, tt.mutate)
			var errs callbackErrors
			errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", "hello world")))
			errs.add(p.OnModelErrorCallback()(mCtx, &model.LLMRequest{}, errors.New("model unavailable")))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			row := onlyRowOfType(t, rows, EventUserMessageReceived)
			checkJSONAt(t, jsonObjectColumn(t, row, "content"), strconv.Quote(tt.wantSummary), "text_summary")
			if got := row["is_truncated"]; got != tt.wantTrunc {
				t.Errorf("is_truncated = %v, want %v", got, tt.wantTrunc)
			}
			if got := onlyRowOfType(t, rows, EventLLMError)["error_message"]; got != tt.wantErrMsg {
				t.Errorf("LLM_ERROR error_message = %v, want %q", got, tt.wantErrMsg)
			}
		})
	}
}

func TestLogEvent_RecoversFromPanics(t *testing.T) {
	logger := &recordingLogger{}
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.Logger = logger })
	broken := &fakeTool{name: "broken", panicOnDescription: true}
	var errs callbackErrors
	errs.add(p.BeforeToolCallback()(mCtx, broken, nil))
	errs.add(p.AfterToolCallback()(mCtx, broken, nil, map[string]any{"ok": true}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventToolCompleted, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Errorf("event types mismatch (-want +got):\n%s", diff)
	}
	if want := "dropped TOOL_STARTING event: recovered panic of type string"; !logger.contains(want) {
		t.Errorf("log does not contain %q:\n%s", want, logger.String())
	}
	if logger.contains("tool description unavailable") {
		t.Errorf("log contains the panic value:\n%s", logger.String())
	}
}

// noAgentContext is an invocation context whose agent accessor panics, as
// ADK contexts do in invocations without an agent.
type noAgentContext struct {
	*mockInvocationContext
}

func (noAgentContext) AgentName() string { panic("invocation has no agent") }

func TestIdentity_AgentFallsBackToEventAuthor(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	ctx := noAgentContext{mCtx}
	ev := &session.Event{
		Author:      "event-author",
		LLMResponse: model.LLMResponse{Content: textContent("model", "hi")},
	}
	if _, err := p.OnEventCallback()(ctx, ev); err != nil {
		t.Fatalf("OnEventCallback error: %v", err)
	}
	p.AfterRunCallback()(ctx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventAgentResponse)
	if got := row["agent"]; got != "event-author" {
		t.Errorf("agent = %v, want the event author", got)
	}
	if got := row["session_id"]; got != "sess-id" {
		t.Errorf("session_id = %v, want sess-id", got)
	}
	if got, ok := onlyRowOfType(t, rows, EventInvocationCompleted)["agent"]; ok {
		t.Errorf("INVOCATION_COMPLETED agent = %v, want none", got)
	}
}

func TestIdentityOf_PlainContext(t *testing.T) {
	if got := identityOf(context.Background()); got != (identity{}) {
		t.Errorf("identityOf(context.Background()) = %+v, want the zero identity", got)
	}
}

func TestLLMRequestAttributes_Tools(t *testing.T) {
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "from_config",
				Description: "Declared in the config",
				Parameters:  &genai.Schema{Type: genai.TypeObject},
			}}}},
		},
		Tools: map[string]any{
			"lookup": &declaredTool{
				fakeTool: fakeTool{name: "lookup", description: "Looks things up"},
				decl: &genai.FunctionDeclaration{
					Name:                 "lookup",
					Description:          "Declaration description",
					Parameters:           &genai.Schema{Type: genai.TypeString},
					ParametersJsonSchema: map[string]any{"type": "object"},
				},
			},
			"from_config": &fakeTool{name: "from_config"},
		},
	}
	tools, ok := llmRequestAttributes(req)["tools"]
	if !ok {
		t.Fatal("llmRequestAttributes() has no tools")
	}
	b, _, err := SmartTruncate(tools, -1)
	if err != nil {
		t.Fatalf("SmartTruncate() error: %v", err)
	}
	var got any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("json.Unmarshal(%s) error: %v", b, err)
	}
	checkJSON(t, "tools", got, `[`+
		`{"name":"from_config","description":"Declared in the config","parameters":{"type":"OBJECT"}},`+
		`{"name":"lookup","description":"Looks things up","parameters":{"type":"object"}}]`)
}

func TestParseNodePath(t *testing.T) {
	tests := []struct {
		path            string
		wantRunID       string
		wantParentRunID any
	}{
		{path: ""},
		{path: "agent"},
		{path: "agent@1", wantRunID: "1"},
		{path: "wf@0/agent@3", wantRunID: "3", wantParentRunID: "0"},
		{path: "wf/agent@3", wantRunID: "3"},
		{path: "wf@0/agent", wantParentRunID: "0"},
		{path: "root/wf@0/agent@3", wantRunID: "3", wantParentRunID: "0"},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			runID, parentRunID := parseNodePath(tt.path)
			if runID != tt.wantRunID || parentRunID != tt.wantParentRunID {
				t.Errorf("parseNodePath(%q) = (%q, %v), want (%q, %v)", tt.path, runID, parentRunID, tt.wantRunID, tt.wantParentRunID)
			}
		})
	}
}

func TestDeriveScope(t *testing.T) {
	tests := []struct {
		scope    string
		wantKind string
	}{
		{scope: "wf@0/review@2", wantKind: scopeKindNodeRun},
		{scope: "review@2", wantKind: scopeKindNodeRun},
		{scope: "adk-call-123", wantKind: scopeKindFunctionCall},
		{scope: "wf@0/adk-call-123", wantKind: scopeKindFunctionCall},
	}
	for _, tt := range tests {
		t.Run(tt.scope, func(t *testing.T) {
			want := map[string]any{"id": tt.scope, "kind": tt.wantKind}
			if diff := cmp.Diff(want, deriveScope(tt.scope)); diff != "" {
				t.Errorf("deriveScope(%q) mismatch (-want +got):\n%s", tt.scope, diff)
			}
		})
	}
}

func TestNewBigQueryAgentAnalyticsPlugin_UpgradesExistingSchema(t *testing.T) {
	type bqFieldJSON struct {
		Name   string        `json:"name"`
		Type   string        `json:"type"`
		Mode   string        `json:"mode,omitempty"`
		Fields []bqFieldJSON `json:"fields,omitempty"`
	}
	toJSONFields := func(schema bq.Schema, omit string, overrideType map[string]string) []bqFieldJSON {
		var convert func(bq.Schema) []bqFieldJSON
		convert = func(s bq.Schema) []bqFieldJSON {
			var out []bqFieldJSON
			for _, f := range s {
				if f.Name == omit {
					continue
				}
				typ := string(f.Type)
				if over, ok := overrideType[f.Name]; ok {
					typ = over
				}
				mode := "NULLABLE"
				if f.Required {
					mode = "REQUIRED"
				} else if f.Repeated {
					mode = "REPEATED"
				}
				out = append(out, bqFieldJSON{
					Name:   f.Name,
					Type:   typ,
					Mode:   mode,
					Fields: convert(f.Schema),
				})
			}
			return out
		}
		return convert(schema)
	}

	tests := []struct {
		name           string
		existing       []bqFieldJSON
		disableUpgrade bool
		patchStatus    int
		retryUpgraded  bool
		wantPatch      bool
		wantPatched    []string
		wantErrMatch   string
	}{
		{
			name:        "legacy 16-column table without event_id is upgraded in place",
			existing:    toJSONFields(EventsSchema(), "event_id", nil),
			patchStatus: http.StatusOK,
			wantPatch:   true,
			wantPatched: []string{
				"timestamp", "event_type", "agent", "session_id", "invocation_id", "user_id",
				"trace_id", "span_id", "parent_span_id", "content", "content_parts", "attributes",
				"latency_ms", "status", "error_message", "is_truncated", "event_id",
			},
		},
		{
			name:        "existing table with empty schema is populated with full schema",
			existing:    nil,
			patchStatus: http.StatusOK,
			wantPatch:   true,
			wantPatched: []string{
				"timestamp", "event_id", "event_type", "agent", "session_id", "invocation_id", "user_id",
				"trace_id", "span_id", "parent_span_id", "content", "content_parts", "attributes",
				"latency_ms", "status", "error_message", "is_truncated",
			},
		},
		{
			name:          "412 precondition failed re-fetches metadata and succeeds when schema is already upgraded",
			existing:      toJSONFields(EventsSchema(), "event_id", nil),
			patchStatus:   http.StatusPreconditionFailed,
			retryUpgraded: true,
			wantPatch:     true,
		},
		{
			name:           "DisableSchemaUpgrade skips PATCH on legacy table",
			existing:       toJSONFields(EventsSchema(), "event_id", nil),
			disableUpgrade: true,
			patchStatus:    http.StatusOK,
			wantPatch:      false,
		},
		{
			name:        "up-to-date 17-column table does not trigger PATCH",
			existing:    toJSONFields(EventsSchema(), "", nil),
			patchStatus: http.StatusOK,
			wantPatch:   false,
		},
		{
			name:         "incompatible existing column type fails initialization",
			existing:     toJSONFields(EventsSchema(), "", map[string]string{"event_id": "INTEGER"}),
			patchStatus:  http.StatusOK,
			wantPatch:    false,
			wantErrMatch: "incompatible BigQuery schema field",
		},
		{
			name:         "PATCH failure returns an error",
			existing:     toJSONFields(EventsSchema(), "event_id", nil),
			patchStatus:  http.StatusForbidden,
			wantPatch:    true,
			wantErrMatch: "failed to upgrade BigQuery table",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			config := DefaultConfig()
			config.ProjectID = "test-project"
			config.DatasetID = "test_dataset"
			config.TableName = "test-table"
			config.DisableSchemaUpgrade = tt.disableUpgrade

			getBodyBytes, err := json.Marshal(map[string]any{
				"etag":   "etag-1",
				"schema": map[string]any{"fields": tt.existing},
			})
			if err != nil {
				t.Fatalf("json.Marshal existing schema: %v", err)
			}
			upgradedBodyBytes, err := json.Marshal(map[string]any{
				"etag":   "etag-2",
				"schema": map[string]any{"fields": toJSONFields(EventsSchema(), "", nil)},
			})
			if err != nil {
				t.Fatalf("json.Marshal upgraded schema: %v", err)
			}

			var patchCalled bool
			var patchedColumns []string
			var patchedLabels map[string]string
			mockTransport := &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
						if patchCalled && tt.retryUpgraded {
							return &http.Response{
								StatusCode: http.StatusOK,
								Body:       io.NopCloser(bytes.NewReader(upgradedBodyBytes)),
							}, nil
						}
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(bytes.NewReader(getBodyBytes)),
						}, nil
					}
					if (r.Method == http.MethodPatch || r.Method == http.MethodPut) && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
						patchCalled = true
						bodyBytes, _ := io.ReadAll(r.Body)
						var body struct {
							Labels map[string]string `json:"labels"`
							Schema struct {
								Fields []bqFieldJSON `json:"fields"`
							} `json:"schema"`
						}
						if err := json.Unmarshal(bodyBytes, &body); err == nil {
							patchedLabels = body.Labels
							for _, f := range body.Schema.Fields {
								patchedColumns = append(patchedColumns, f.Name)
							}
						}
						if tt.patchStatus != http.StatusOK {
							return &http.Response{
								StatusCode: tt.patchStatus,
								Body:       io.NopCloser(strings.NewReader(fmt.Sprintf(`{"error":{"code":%d,"message":"patch status"}}`, tt.patchStatus))),
							}, nil
						}
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(bytes.NewReader(bodyBytes)),
						}, nil
					}
					if strings.Contains(r.URL.Path, "/jobs") {
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			}

			bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(&http.Client{Transport: mockTransport}))
			if err != nil {
				t.Fatalf("bq.NewClient: %v", err)
			}
			lis, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("listen: %v", err)
			}
			gSrv := grpc.NewServer()
			storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
			go func() { _ = gSrv.Serve(lis) }()
			t.Cleanup(gSrv.Stop)
			conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatalf("dial: %v", err)
			}
			t.Cleanup(func() { _ = conn.Close() })
			writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
			if err != nil {
				t.Fatalf("NewBigQueryWriteClient: %v", err)
			}

			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
			if p != nil {
				t.Cleanup(func() { _ = p.Close() })
			}
			if tt.wantErrMatch != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrMatch) {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients error = %v, want match %q", err, tt.wantErrMatch)
				}
			} else if err != nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients unexpected error: %v", err)
			}
			if patchCalled != tt.wantPatch {
				t.Errorf("PATCH called = %v, want %v", patchCalled, tt.wantPatch)
			}
			if patchCalled && patchedLabels["adk_schema_version"] != SchemaVersion {
				t.Errorf("patched labels = %v, want adk_schema_version=%q", patchedLabels, SchemaVersion)
			}
			if len(tt.wantPatched) > 0 {
				if diff := cmp.Diff(tt.wantPatched, patchedColumns); diff != "" {
					t.Errorf("patched columns mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// TestNewBigQueryAgentAnalyticsPlugin_TableSetup checks how construction
// treats table metadata and creation errors: only a missing table (404) is
// created, a table that another writer created first (409) is read again and
// upgraded, and any other error fails construction and closes the write
// client.
func TestNewBigQueryAgentAnalyticsPlugin_TableSetup(t *testing.T) {
	var legacySchema bq.Schema
	for _, f := range EventsSchema() {
		if f.Name != eventIDField {
			legacySchema = append(legacySchema, f)
		}
	}
	legacyFields, err := legacySchema.ToJSONFields()
	if err != nil {
		t.Fatalf("ToJSONFields: %v", err)
	}
	legacyTable := fmt.Sprintf(`{"etag":"etag-1","schema":{"fields":%s}}`, legacyFields)

	tests := []struct {
		name           string
		disableUpgrade bool
		// getStatuses holds the status of each table metadata request in
		// turn; http.StatusOK returns a table without event_id.
		getStatuses []int
		// postStatus is the status of the table creation request.
		postStatus int
		wantGets   int
		wantPost   bool
		wantPatch  bool
		wantErr    string
		// wantErrCode is the status of the BigQuery API error that the
		// construction error wraps.
		wantErrCode int
	}{
		{
			name:        "metadata error other than 404 fails without creating the table",
			getStatuses: []int{http.StatusForbidden},
			wantGets:    1,
			wantErr:     "failed to get BigQuery table test-table metadata",
			wantErrCode: http.StatusForbidden,
		},
		{
			name:        "409 from create reads the table again and upgrades it",
			getStatuses: []int{http.StatusNotFound, http.StatusOK},
			postStatus:  http.StatusConflict,
			wantGets:    2,
			wantPost:    true,
			wantPatch:   true,
		},
		{
			name:           "409 from create with DisableSchemaUpgrade keeps the table as is",
			disableUpgrade: true,
			getStatuses:    []int{http.StatusNotFound},
			postStatus:     http.StatusConflict,
			wantGets:       1,
			wantPost:       true,
		},
		{
			name:        "create error other than 409 fails",
			getStatuses: []int{http.StatusNotFound},
			postStatus:  http.StatusForbidden,
			wantGets:    1,
			wantPost:    true,
			wantErr:     "failed to create BigQuery table test-table",
			wantErrCode: http.StatusForbidden,
		},
		{
			name:        "metadata error after a 409 from create fails",
			getStatuses: []int{http.StatusNotFound, http.StatusForbidden},
			postStatus:  http.StatusConflict,
			wantGets:    2,
			wantPost:    true,
			wantErr:     "failed to get BigQuery table test-table metadata",
			wantErrCode: http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			config := DefaultConfig()
			config.ProjectID = "test-project"
			config.DatasetID = "test_dataset"
			config.TableName = "test-table"
			config.DisableSchemaUpgrade = tt.disableUpgrade
			// Views are created in the background once the table is ready;
			// this test covers only the table requests.
			config.DisableCreateViews = true

			respond := func(status int, body string) *http.Response {
				return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body))}
			}
			apiError := func(status int) *http.Response {
				return respond(status, fmt.Sprintf(`{"error":{"code":%d,"message":"status %d"}}`, status, status))
			}
			const tablePath = "/datasets/test_dataset/tables/test-table"
			var gets, posts, patches int
			var patchedColumns []string
			transport := &mockTransport{roundTrip: func(r *http.Request) (*http.Response, error) {
				switch {
				case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, tablePath):
					gets++
					if gets > len(tt.getStatuses) {
						t.Errorf("unexpected table metadata request #%d", gets)
						return apiError(http.StatusBadRequest), nil
					}
					if status := tt.getStatuses[gets-1]; status != http.StatusOK {
						return apiError(status), nil
					}
					return respond(http.StatusOK, legacyTable), nil
				case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/datasets/test_dataset/tables"):
					posts++
					return apiError(tt.postStatus), nil
				case (r.Method == http.MethodPatch || r.Method == http.MethodPut) && strings.HasSuffix(r.URL.Path, tablePath):
					patches++
					body, _ := io.ReadAll(r.Body)
					var patch struct {
						Schema struct {
							Fields []struct {
								Name string `json:"name"`
							} `json:"fields"`
						} `json:"schema"`
					}
					if err := json.Unmarshal(body, &patch); err != nil {
						t.Errorf("unmarshal patch request: %v", err)
					}
					for _, f := range patch.Schema.Fields {
						patchedColumns = append(patchedColumns, f.Name)
					}
					return respond(http.StatusOK, string(body)), nil
				}
				t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				return apiError(http.StatusBadRequest), nil
			}}
			bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("bq.NewClient: %v", err)
			}
			conn, writeClient := newTestWriteClient(t)

			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
			if p != nil {
				t.Cleanup(func() { _ = p.Close() })
			}
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want an error containing %q", err, tt.wantErr)
				}
				var apiErr *googleapi.Error
				if !errors.As(err, &apiErr) || apiErr.Code != tt.wantErrCode {
					t.Errorf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want it to wrap a BigQuery API error with code %d", err, tt.wantErrCode)
				}
				if state := conn.GetState(); state != connectivity.Shutdown {
					t.Errorf("write client connection state = %v, want %v after a failed construction", state, connectivity.Shutdown)
				}
			} else if err != nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() unexpected error: %v", err)
			}
			if gets != tt.wantGets {
				t.Errorf("table metadata requests = %d, want %d", gets, tt.wantGets)
			}
			if gotPost := posts > 0; gotPost != tt.wantPost {
				t.Errorf("table created = %v, want %v", gotPost, tt.wantPost)
			}
			if gotPatch := patches > 0; gotPatch != tt.wantPatch {
				t.Errorf("table schema patched = %v, want %v", gotPatch, tt.wantPatch)
			}
			if tt.wantPatch && !slices.Contains(patchedColumns, eventIDField) {
				t.Errorf("patched columns = %v, want them to include %s", patchedColumns, eventIDField)
			}
		})
	}
}

// newTestWriteClient returns a write client connected to a fake BigQuery
// Storage Write server, and its connection, whose state shows whether the
// client was closed.
func newTestWriteClient(t *testing.T) (*grpc.ClientConn, *bqstorage.BigQueryWriteClient) {
	t.Helper()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewBigQueryWriteClient: %v", err)
	}
	return conn, writeClient
}

// TestNewBigQueryAgentAnalyticsPluginWithClients_ClosesUnusedWriteClient
// checks that construction closes the write client it was given when it
// returns before table setup: on an invalid config, and when the plugin is
// disabled and so never writes.
func TestNewBigQueryAgentAnalyticsPluginWithClients_ClosesUnusedWriteClient(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
		// wantErr is empty when construction must succeed.
		wantErr string
	}{
		{
			name:    "invalid BatchSize",
			mutate:  func(c *Config) { c.BatchSize = -1 },
			wantErr: "BatchSize and QueueMaxSize must be >= 1",
		},
		{
			name:    "invalid PayloadColumnDenylist",
			mutate:  func(c *Config) { c.PayloadColumnDenylist = []string{"bogus"} },
			wantErr: `unsupported PayloadColumnDenylist entry "bogus"`,
		},
		{
			name:   "disabled",
			mutate: func(c *Config) { c.Enabled = false },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			config := DefaultConfig()
			config.ProjectID = "test-project"
			config.DatasetID = "test_dataset"
			config.TableName = "test-table"
			tt.mutate(&config)
			transport := &mockTransport{roundTrip: func(r *http.Request) (*http.Response, error) {
				t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				return nil, errors.New("unexpected request")
			}}
			bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("bq.NewClient: %v", err)
			}
			conn, writeClient := newTestWriteClient(t)

			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
			if p != nil {
				t.Cleanup(func() { _ = p.Close() })
			}
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want an error containing %q", err, tt.wantErr)
				}
			} else if err != nil || p == nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() = (%v, %v), want a plugin", p, err)
			}
			if state := conn.GetState(); state != connectivity.Shutdown {
				t.Errorf("write client connection state = %v, want %v", state, connectivity.Shutdown)
			}
		})
	}
}

func TestPayloadColumns_ProjectsTableAndArrowSchema(t *testing.T) {
	ctx := context.Background()
	config := DefaultConfig()
	config.ProjectID = "test-project"
	config.DatasetID = "test_dataset"
	config.TableName = "test-table"
	config.PayloadColumnDenylist = []string{columnContent, columnContentParts, columnLatencyMS}

	var createdColumns []string
	var createdLabels map[string]string
	mockTransport := &mockTransport{
		roundTrip: func(r *http.Request) (*http.Response, error) {
			if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
				return &http.Response{
					StatusCode: http.StatusNotFound,
					Body:       io.NopCloser(strings.NewReader(`{"error":{"code":404,"message":"Not found"}}`)),
				}, nil
			}
			if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables") {
				bodyBytes, _ := io.ReadAll(r.Body)
				var body struct {
					Labels map[string]string `json:"labels"`
					Schema struct {
						Fields []struct {
							Name string `json:"name"`
						} `json:"fields"`
					} `json:"schema"`
				}
				if err := json.Unmarshal(bodyBytes, &body); err == nil {
					createdLabels = body.Labels
					for _, f := range body.Schema.Fields {
						createdColumns = append(createdColumns, f.Name)
					}
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			}
			if strings.Contains(r.URL.Path, "/jobs") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
				}, nil
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader("{}")),
			}, nil
		},
	}

	bqClient, err := bq.NewClient(ctx, config.ProjectID, option.WithHTTPClient(&http.Client{Transport: mockTransport}))
	if err != nil {
		t.Fatalf("bq.NewClient: %v", err)
	}
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 10)
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: requests})
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewBigQueryWriteClient: %v", err)
	}

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
	if err != nil {
		t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	wantColumns := []string{
		"timestamp", "event_id", "event_type", "agent", "session_id", "invocation_id", "user_id",
		"trace_id", "span_id", "parent_span_id", "attributes", "status", "error_message", "is_truncated",
	}
	if diff := cmp.Diff(wantColumns, createdColumns); diff != "" {
		t.Errorf("created table schema columns mismatch (-want +got):\n%s", diff)
	}
	wantLabels := map[string]string{"adk_schema_version": SchemaVersion}
	if diff := cmp.Diff(wantLabels, createdLabels); diff != "" {
		t.Errorf("created table labels mismatch (-want +got):\n%s", diff)
	}

	mCtx := &mockInvocationContext{
		ctx:          ctx,
		agentName:    "test-agent",
		sessionID:    "sess-id",
		invocationID: "inv-id",
		userID:       "user-id",
	}
	if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "hello")); err != nil {
		t.Fatalf("OnUserMessageCallback: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	req := <-requests
	if diff := cmp.Diff(wantColumns, arrowWriterSchemaColumns(t, req)); diff != "" {
		t.Errorf("Arrow writer schema columns mismatch (-want +got):\n%s", diff)
	}
}

func TestStructuredContentPartsAndSanitizedTextInCallbacks(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.MaxContentLen = 60
	})

	structuredContent := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: `{"api_key":"sk-secret-key","note":"hello world"}`},
			{FunctionCall: &genai.FunctionCall{
				ID:   "call-42",
				Name: "lookup",
				Args: map[string]any{"api_key": "sk-secret", "query": "weather"},
			}},
			{FunctionResponse: &genai.FunctionResponse{
				ID:       "call-42",
				Name:     "lookup",
				Response: map[string]any{"token": "tok-secret", "answer": "sunny"},
			}},
			{ExecutableCode: &genai.ExecutableCode{
				Language: genai.LanguagePython,
				Code:     "print(42)",
			}},
			{CodeExecutionResult: &genai.CodeExecutionResult{
				Outcome: genai.OutcomeOK,
				Output:  "42\n",
			}},
		},
	}

	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", `{"api_key":"sk-user-secret","tail":`)))
	errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{
		Model:    "gemini-test",
		Contents: []*genai.Content{textContent("user", `{"api_key":"sk-req-secret","msg":"hi"}`)},
	}))
	errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: structuredContent}, nil))
	errs.add(p.OnEventCallback()(mCtx, &session.Event{
		Author:      "test-agent",
		LLMResponse: model.LLMResponse{Content: textContent("model", `{"api_key":"sk-secret-key","note":"hello world"}`)},
	}))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

	userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
	if raw, _ := userRow["content"].(string); strings.Contains(raw, "sk-user-secret") {
		t.Errorf("USER_MESSAGE_RECEIVED content %s leaked secret from malformed JSON", raw)
	}
	if userRow["is_truncated"] != true {
		t.Errorf("USER_MESSAGE_RECEIVED is_truncated = %v, want true", userRow["is_truncated"])
	}

	reqRow := onlyRowOfType(t, rows, EventLLMRequest)
	if raw, _ := reqRow["content"].(string); strings.Contains(raw, "sk-req-secret") {
		t.Errorf("LLM_REQUEST content %s leaked secret from JSON text", raw)
	}

	resRow := onlyRowOfType(t, rows, EventLLMResponse)
	if raw, _ := resRow["content"].(string); strings.Contains(raw, "sk-secret") {
		t.Errorf("LLM_RESPONSE content %s leaked secret", raw)
	}
	parts, _ := resRow["content_parts"].([]map[string]any)
	if len(parts) != 5 {
		t.Fatalf("LLM_RESPONSE content_parts len = %d, want 5: %v", len(parts), parts)
	}
	if parts[1]["mime_type"] != "application/json" {
		t.Errorf("LLM_RESPONSE parts[1] mime_type = %v", parts[1]["mime_type"])
	}
	if attrs, _ := parts[1]["part_attributes"].(string); strings.Contains(attrs, "sk-secret") || !strings.Contains(attrs, redactedValue) {
		t.Errorf("LLM_RESPONSE parts[1] part_attributes = %s, want redacted args", attrs)
	}
	if parts[2]["mime_type"] != "application/json" {
		t.Errorf("LLM_RESPONSE parts[2] mime_type = %v", parts[2]["mime_type"])
	}
	if attrs, _ := parts[2]["part_attributes"].(string); strings.Contains(attrs, "tok-secret") || !strings.Contains(attrs, redactedValue) {
		t.Errorf("LLM_RESPONSE parts[2] part_attributes = %s, want redacted response", attrs)
	}
	if parts[3]["mime_type"] != "text/plain" || parts[3]["text"] != "print(42)" {
		t.Errorf("LLM_RESPONSE parts[3] = %v", parts[3])
	}
	if parts[4]["mime_type"] != "text/plain" || parts[4]["text"] != "42\n" {
		t.Errorf("LLM_RESPONSE parts[4] = %v", parts[4])
	}

	agentRespRow := onlyRowOfType(t, rows, EventAgentResponse)
	if raw, _ := agentRespRow["content"].(string); strings.Contains(raw, "sk-secret") {
		t.Errorf("AGENT_RESPONSE content %s leaked secret", raw)
	}
	respParts, _ := agentRespRow["content_parts"].([]map[string]any)
	if len(respParts) != 1 {
		t.Fatalf("AGENT_RESPONSE content_parts len = %d, want 1: %v", len(respParts), respParts)
	}
	if text, _ := respParts[0]["text"].(string); strings.Contains(text, "sk-secret") {
		t.Errorf("AGENT_RESPONSE parts[0] text = %s leaked secret", text)
	}

	proseCases := []string{
		"[Urgent] please help me reset my router",
		"[1] First footnote",
		"[docs](https://example.com) has the details",
		"{{template}} rendered",
		"[x] task one",
		`"see {this}" she said`,
	}
	for _, prose := range proseCases {
		t.Run("prose_"+prose, func(t *testing.T) {
			var proseErrs callbackErrors
			proseErrs.add(p.OnUserMessageCallback()(mCtx, textContent("user", prose)))
			proseErrs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", prose)}, nil))
			proseErrs.check(t)
			p.AfterRunCallback()(mCtx)

			proseRows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			uRow := onlyRowOfType(t, proseRows, EventUserMessageReceived)
			if raw, _ := uRow["content"].(string); strings.Contains(raw, unparseableJSONPlaceholder) || !strings.Contains(raw, prose[:10]) {
				t.Errorf("USER_MESSAGE_RECEIVED content = %s, want preserved prose %q", raw, prose)
			}
			if uRow["is_truncated"] != false {
				t.Errorf("USER_MESSAGE_RECEIVED is_truncated = %v, want false for prose %q", uRow["is_truncated"], prose)
			}
			uParts, _ := uRow["content_parts"].([]map[string]any)
			if len(uParts) != 1 || uParts[0]["text"] != prose {
				t.Errorf("USER_MESSAGE_RECEIVED content_parts = %v, want text %q", uParts, prose)
			}
			mRow := onlyRowOfType(t, proseRows, EventLLMResponse)
			if raw, _ := mRow["content"].(string); strings.Contains(raw, unparseableJSONPlaceholder) {
				t.Errorf("LLM_RESPONSE content = %s, want preserved prose %q", raw, prose)
			}
			if mRow["is_truncated"] != false {
				t.Errorf("LLM_RESPONSE is_truncated = %v, want false for prose %q", mRow["is_truncated"], prose)
			}
		})
	}
}

func TestQuotedFragmentCost(t *testing.T) {
	tests := []struct {
		name string
		text string
	}{
		{
			name: "broken fragments",
			text: `"a" ` + strings.Repeat("\"x\n", 64) + strings.Repeat("b", 100_000),
		},
		{
			// A key word lets the text through the credential pattern
			// prefilter, so the patterns run over all of it, for content
			// and again for content_parts. At 50 KB that stays well within
			// the bound under -race, while scanning the rest of the text
			// again for each of the 64 fragments takes many times longer.
			name: "broken fragments with a key word",
			text: `"a" ` + strings.Repeat("\"token\n", 64) + strings.Repeat("b", 50_000),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, _, mCtx := setupTestPlugin(t)
			start := time.Now()
			_, _ = p.OnUserMessageCallback()(mCtx, textContent("user", tt.text))
			if d := time.Since(start); d > 3*time.Second {
				t.Errorf("OnUserMessageCallback took %v for a %d KB message", d, len(tt.text)/1000)
			}
		})
	}
}

func TestQuotedJSONSecretSurvives(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	body := `{"access_token": "SECRET123"}`
	_, _ = p.OnToolErrorCallback()(mCtx, &fakeTool{name: "exchange"}, nil, fmt.Errorf("token exchange failed: %q", body))
	p.AfterRunCallback()(mCtx)
	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	row := onlyRowOfType(t, rows, EventToolError)
	if msg := fmt.Sprint(row["error_message"]); strings.Contains(msg, "SECRET123") {
		t.Errorf("error_message = %s", msg)
	}
}

// TestCredentialShapesNeverReachARow checks that a value under a sensitive
// key is redacted from error_message, content and content_parts, whatever
// quoting or nesting the key or the value uses.
func TestCredentialShapesNeverReachARow(t *testing.T) {
	const secret = "SECRET123"
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "JSON quoted with %q",
			text: fmt.Sprintf("token exchange failed: %q", `{"access_token": "SECRET123"}`),
			want: `token exchange failed: "{\"access_token\":\"[REDACTED]\"}"`,
		},
		{
			name: "value quoted with escaped quotes",
			text: `failed: password=\"SECRET123\"`,
			want: `failed: password=\"[REDACTED]\"`,
		},
		{
			name: "single-quoted key",
			text: `bad config: {'api_key': 'SECRET123'}`,
			want: `bad config: {'api_key':'[REDACTED]'}`,
		},
		{
			name: "credentials envelope is redacted whole",
			text: `auth failed: {"credentials": {"access_token": "SECRET123", "expires_in": 3600}}`,
			want: `auth failed: {"credentials":"[REDACTED]"}`,
		},
		{
			name: "AWS STS credentials envelope",
			text: `assume role failed: {"Credentials": {"AccessKeyId": "EXAMPLEKEYID", "SecretAccessKey": "SECRET123", "SessionToken": "SECRET123", "Expiration": "2026-10-14T00:00:00Z"}}`,
			want: `assume role failed: {"Credentials":"[REDACTED]"}`,
		},
		{
			name: "object under a sensitive key",
			text: `auth failed: {"token": {"value": "SECRET123"}}`,
			want: `auth failed: {"token":"[REDACTED]"}`,
		},
		{
			name: "object quoted with %q whose string field holds a closing brace",
			text: fmt.Sprintf("auth failed: %q", `{"token": {"label": "a},", "val": "SECRET123"}}`),
			want: `auth failed: "{\"token\":[REDACTED]}"`,
		},
		{
			name: "bare value with a parenthesis",
			text: "failed: password=abc)SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value starting with a parenthesis",
			text: "failed: password=)SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value starting with slashes",
			text: "failed: password: //SECRET123",
			want: "failed: password:[REDACTED]",
		},
		{
			name: "bare value with a closing brace",
			text: "failed: password=ab}SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "bare value ending with a closing brace",
			text: "failed: password=SECRET123}",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "quoted value ending with a closing brace",
			text: `bad config: {'password': 'SECRET123}'}`,
			want: `bad config: {'password':'[REDACTED]'}`,
		},
		{
			name: "camelCase keys",
			text: "auth failed: accessToken=SECRET123 apiKey: SECRET123",
			want: "auth failed: accessToken=[REDACTED] apiKey:[REDACTED]",
		},
		{
			name: "Go struct printed with %+v",
			text: fmt.Sprintf("bad config: %+v", struct{ APIKey, ClientSecret, Region string }{"SECRET123", "SECRET123", "us"}),
			want: "bad config: {APIKey:[REDACTED] ClientSecret:[REDACTED] Region:us}",
		},
		{
			name: "Go struct printed with %+v ending with credential fields",
			text: fmt.Sprintf("bad config: %+v", struct{ APIKey, ClientSecret string }{"SECRET123", "SECRET123"}),
			want: "bad config: {APIKey:[REDACTED] ClientSecret:[REDACTED]}",
		},
		{
			name: "Go struct printed with %+v whose credential field holds a closing brace",
			text: fmt.Sprintf("bad config: %+v", struct{ Password string }{"ab}SECRET123"}),
			want: "bad config: {Password:[REDACTED]}",
		},
		{
			name: "nested Go struct printed with %+v",
			text: fmt.Sprintf("bad config: %+v", struct{ Auth struct{ Token string } }{Auth: struct{ Token string }{Token: "SECRET123"}}),
			want: "bad config: {Auth:{Token:[REDACTED]}}",
		},
		{
			name: "slice of Go structs printed with %+v",
			text: fmt.Sprintf("bad config: %+v", []struct{ APIKey string }{{"SECRET123"}}),
			want: "bad config: [{APIKey:[REDACTED]}]",
		},
		{
			name: "AWS credentials printed with %+v",
			text: fmt.Sprintf("assume role failed: %+v", struct{ AccessKeyID, SecretAccessKey, SessionToken string }{"EXAMPLEKEYID", "SECRET123", "SECRET123"}),
			want: "assume role failed: {AccessKeyID:EXAMPLEKEYID SecretAccessKey:[REDACTED] SessionToken:[REDACTED]}",
		},
		{
			name: "x_goog_api_key",
			text: "request failed: x_goog_api_key=SECRET123",
			want: "request failed: x_goog_api_key=[REDACTED]",
		},
		{
			name: "session user key",
			text: "failed: user:api_key=SECRET123",
			want: "failed: user:api_key=[REDACTED]",
		},
		{
			name: "embedded JSON with a camelCase key",
			text: `token exchange failed: {"accessToken": "SECRET123", "expiresIn": 3600}`,
			want: `token exchange failed: {"accessToken":"[REDACTED]", "expiresIn": 3600}`,
		},
		{
			name: "command-line flag",
			text: "mysql -u root --password=SECRET123 failed",
			want: "mysql -u root --password=[REDACTED] failed",
		},
		{
			name: "command-line flag with a hyphenated key",
			text: "--api-key=SECRET123",
			want: "--api-key=[REDACTED]",
		},
		{
			name: "hyphenated header ending in token",
			text: "X-Auth-Token: SECRET123",
			want: "X-Auth-Token:[REDACTED]",
		},
		{
			name: "upper-case hyphenated header ending in token",
			text: "PRIVATE-TOKEN: SECRET123",
			want: "PRIVATE-TOKEN:[REDACTED]",
		},
		{
			name: "credentials block in YAML",
			text: "config:\ncredentials:\n  token: SECRET123\n  region: us",
			want: "config:\ncredentials:\n  token:[REDACTED]\n  region: us",
		},
		{
			name: "quoted value that starts with a key-value pair",
			text: `bad config: password: 'token: SECRET123 SECRET123'`,
			want: `bad config: password:'[REDACTED]'`,
		},
		{
			name: "credentials in a Python dict whose value starts with a key-value pair",
			text: `bad config: {'credentials': 'secret=correct horse SECRET123 staple'}`,
			want: `bad config: {'credentials':'[REDACTED]'}`,
		},
		{
			name: "bare value that ends with a key-value pair",
			text: "failed: password=SECRET123-token:SECRET123",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "key-shaped bare value with no value after it",
			text: "failed: password=SECRET123-token:",
			want: "failed: password=[REDACTED]",
		},
		{
			name: "key-shaped bare value followed by a closing brace",
			text: "failed: password=SECRET123-token: }",
			want: "failed: password=[REDACTED] }",
		},
		{
			name: "key-shaped value of a Go struct field followed by the closing brace",
			text: "failed: {Password:SECRET123-token: }",
			want: "failed: {Password:[REDACTED] }",
		},
		{
			name: "base64 value that ends with a key name, one pad and a closing brace",
			text: "failed: api_key=SECRET123AbC+xYz/Sig= }",
			want: "failed: api_key=[REDACTED] }",
		},
		{
			name: "key-shaped bare value followed by a closing bracket",
			text: "failed: password=SECRET123-token: ]",
			want: "failed: password=[REDACTED] ]",
		},
		{
			name: "key-shaped value of a Go struct field followed by the closing brace and another struct",
			text: "failed: [{Password:SECRET123-token: } {User:bob}]",
			want: "failed: [{Password:[REDACTED] } {User:bob}]",
		},
		{
			name: "nested key whose value starts with a closing bracket",
			text: "failed: credentials: token = ]SECRET123",
			want: "failed: credentials: token=[REDACTED]",
		},
		{
			name: "nested key whose quoted value has leading whitespace",
			text: `failed: credentials: token: " SECRET123"`,
			want: `failed: credentials: token:"[REDACTED]"`,
		},
		{
			name: "nested key in a Go struct whose quoted value has leading whitespace",
			text: `failed: {Credentials:token: " SECRET123"}`,
			want: `failed: {Credentials:token:"[REDACTED]"}`,
		},
		{
			name: "key-shaped value of a Go struct field followed by an unclosed quote",
			text: `failed: {Password:SECRET123-token: "}`,
			want: `failed: {Password:[REDACTED] "}`,
		},
		{
			name: "nested key glued to a quoted value with spaces",
			text: `failed: credentials: password="correct SECRET123 battery"`,
			want: `failed: credentials: password="[REDACTED]"`,
		},
		{
			name: "nested key in a Go struct glued to a quoted value with leading whitespace",
			text: `failed: {Credentials:token=" SECRET123"}`,
			want: `failed: {Credentials:token="[REDACTED]"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			var errs callbackErrors
			errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", tt.text)))
			errs.add(p.AfterToolCallback()(mCtx, &fakeTool{name: "lookup"}, nil, map[string]any{"output": tt.text}, nil))
			errs.add(p.OnToolErrorCallback()(mCtx, &fakeTool{name: "exchange"}, nil, errors.New(tt.text)))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
			if got, _ := lookup(jsonObjectColumn(t, userRow, "content"), "text_summary"); got != tt.want {
				t.Errorf("USER_MESSAGE_RECEIVED content.text_summary = %v, want %q", got, tt.want)
			}
			if parts, _ := userRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["text"] != tt.want {
				t.Errorf("USER_MESSAGE_RECEIVED content_parts = %v, want text %q", parts, tt.want)
			}
			toolRow := onlyRowOfType(t, rows, EventToolCompleted)
			if got, _ := lookup(jsonObjectColumn(t, toolRow, "content"), "result", "output"); got != tt.want {
				t.Errorf("TOOL_COMPLETED content.result.output = %v, want %q", got, tt.want)
			}
			if got := onlyRowOfType(t, rows, EventToolError)["error_message"]; got != tt.want {
				t.Errorf("TOOL_ERROR error_message = %v, want %q", got, tt.want)
			}
			for _, row := range rows {
				if s := fmt.Sprint(row); strings.Contains(s, secret) {
					t.Errorf("%v row leaked %s: %s", row["event_type"], secret, s)
				}
			}
		})
	}
}

// TestValueEndingInKeyName checks that a credential whose value only ends in
// a sensitive key name, as in the text that fmt.Errorf("password=%s: %w", ...)
// writes before a wrapped error, is redacted from error_message, content,
// content_parts, tool arguments and results, and event attributes, and that
// no row holds a credential. When the text is a JSON object or array whose
// string also holds a password after an escaped line break, error_message
// redacts each pair, while content, content_parts, tool arguments and
// results, and event attributes are written as [REDACTED_SENSITIVE_TEXT].
func TestValueEndingInKeyName(t *testing.T) {
	marshalN := func(s string, n int) string {
		for range n {
			b, err := json.Marshal(s)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			s = string(b)
		}
		return s
	}
	secrets := []string{"SECRET123", "hunter2", "s3cr3t", "bob pw", "OTHER pass"}
	tests := []struct {
		name string
		text string
		// want is the error_message, and also the content, content_parts and
		// attributes text unless wantContent is set.
		want        string
		wantContent string
		// wantResponse is the AGENT_RESPONSE content.response, if it is not
		// "text: '" + the content + "'". The response is redacted again
		// once the text is quoted, and a redacted value that runs to the
		// end of the text then also takes the closing "'".
		wantResponse string
	}{
		{
			name: "password before a wrapped error",
			text: "connect user=bob password=dev-SECRET123-password: dial tcp 10.0.0.1:5432: connection refused",
			want: "connect user=bob password=[REDACTED] tcp 10.0.0.1:5432: connection refused",
		},
		{
			name: "quoted password before an error",
			text: `login failed: password="SECRET123-secret": invalid`,
			want: `login failed: password="[REDACTED]"`,
		},
		{
			name:         "API key before a wrapped error",
			text:         "api_key=prodSECRET123-api-key: permission denied",
			want:         "api_key=[REDACTED]",
			wantResponse: "text: 'api_key=[REDACTED]",
		},
		{
			name: "token in prose",
			text: "my login is token=abc-SECRET123-token: please check",
			want: "my login is token=[REDACTED] check",
		},
		{
			name: "temp: value ending in a key name before a wrapped error",
			text: "password=temp:SECRET123-token: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "temp: value before a wrapped error",
			text: "password=temp:SECRET123: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "user:temp: value before a wrapped error",
			text: "password=user:temp:SECRET123: dial tcp",
			want: "password=[REDACTED] tcp",
		},
		{
			name: "quoted temp: value before a wrapped error",
			text: `password="temp:SECRET123": dial tcp`,
			want: `password="[REDACTED]" tcp`,
		},
		{
			name:         "temp: key on the line of credentials",
			text:         "credentials: temp:foo: SECRET123",
			want:         "credentials:[REDACTED]",
			wantResponse: "text: 'credentials:[REDACTED]",
		},
		{
			name:         "temp: key on an indented line under credentials",
			text:         "credentials:\n  temp:foo: SECRET123",
			want:         "credentials:\n  temp:foo:[REDACTED]",
			wantResponse: "text: 'credentials:[REDACTED]",
		},
		{
			name:        "password after an escaped line break in a JSON message",
			text:        `{"msg":"retry\npassword=hunter2 vault: secret: kv/data/db-password: \"s3cr3t pass\""}`,
			want:        `{"msg":"retry\npassword=[REDACTED] vault: secret:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "password after an escaped line break in a JSON error",
			text:        `{"error":"connect failed\npassword=hunter2 credentials: user/password:\"bob pw\""}`,
			want:        `{"error":"connect failed\npassword=[REDACTED] credentials:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "password after an escaped line break in a JSON object whose last field is a bare pair",
			text:        `{"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\"","code":"token=abc"}`,
			want:        `{"code":"token=[REDACTED]","msg":"retry\npassword=[REDACTED] vault: secret:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON error with a password after an escaped line break",
			text:        `POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefix with bracket tag before JSON error with a password after an escaped line break",
			text:        `POST /v1/login [400]: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "bracket-led log line before JSON error with only an escaped-newline password",
			text:        `[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123"}`,
			want:        unparseableJSONPlaceholder,
			wantContent: unparseableJSONPlaceholder,
		},
		{
			name:        "bracket-prefixed JSON error with a password after an escaped line break and a raw marker",
			text:        `[ERROR] POST /v1/login: 400: {"msg":"retry\npassword=SECRET123 vault: secret: kv/data/db-password: \"OTHER pass\""}`,
			want:        unparseableJSONPlaceholder,
			wantContent: unparseableJSONPlaceholder,
		},
		{
			name:        "JSON error followed by trailing status text with a password after an escaped line break",
			text:        `POST /v1/login: {"msg":"retry\npassword=SECRET123"} (status 400)`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object whose string ends with a value ending in a key name before a word",
			text:        `{"msg":"retry\npassword=dev-SECRET123-password: dial"}`,
			want:        `{"msg":"retry\npassword=[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object with a password after an escaped line break and a final value ending in a key name",
			text:        `{"msg":"retry\npassword=hunter2 token=abc-SECRET123-token: dial"}`,
			want:        `{"msg":"retry\npassword=[REDACTED] token=[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON array whose string ends with a value ending in a key name before a word",
			text:        `["retry\npassword=dev-SECRET123-password: dial"]`,
			want:        `["retry\npassword=[REDACTED]"]`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object whose string ends with a value ending in a key name",
			text:        `{"msg":"retry\npassword=dev-SECRET123-password"}`,
			want:        `{"msg":"retry\npassword=[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON array whose string ends with a value ending in a key name",
			text:        `["retry\npassword=dev-SECRET123-password"]`,
			want:        `["retry\npassword=[REDACTED]"]`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "stray quote before JSON error with an escaped-newline password",
			text:        `bad token " in body: {"msg":"retry\npassword=SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "escaped quote before JSON error with an escaped-newline password",
			text:        `flag \"foo: {"msg":"retry\npassword=SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "non-JSON backslash before escaped quote and escaped-newline password",
			text:        `err: "dir C:\work\"retry\npassword=SECRET123"`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "5-deep marshaled JSON string with an escaped-newline password",
			text:        "upstream: " + marshalN(`"retry\npassword=SECRET123"`, 5),
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "9-deep marshaled JSON string with an escaped-newline password",
			text:        "upstream: " + marshalN(`"retry\npassword=SECRET123"`, 9),
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "truncated unclosed JSON error body with an escaped-newline password",
			text:        `upstream error: {"error":"login failed\npassword=SECRET123`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "single-quoted Python dict with an escaped-newline password",
			text:        `{'error': 'retry\npassword=SECRET123'}`,
			want:        unparseableJSONPlaceholder,
			wantContent: unparseableJSONPlaceholder,
		},
		{
			name:        "prefixed single-quoted Python dict with an escaped-newline password",
			text:        `bad config: {'error': 'retry\npassword=SECRET123'}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "unquoted escaped log line with an escaped-newline password",
			text:        `err=retry\npassword=SECRET123`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where earlier token swallows escaped newline before value ending in key name",
			text:        `{"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`,
			want:        `{"msg":"token=[REDACTED]\npassword:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows escaped newline before value ending in key name",
			text:        `POST /v1/login: 400: {"msg":"token=abc\npassword: dev-SECRET123-password: dial"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where earlier compound key value swallows escaped newline before password",
			text:        `{"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`,
			want:        `{"msg":"secret:[REDACTED]\npassword:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier compound key value swallows escaped newline before password",
			text:        `POST /v1/login: 400: {"msg":"secret: kv/data/db-password: other\npassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where password colon is followed by escaped newline and space",
			text:        `{"msg":"password:\n SECRET123"}`,
			want:        `{"msg":"password:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where password colon is followed by escaped newline and space",
			text:        `POST /v1/login: 400: {"msg":"password:\n SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed single-quoted Python dict where earlier token swallows escaped newline before password",
			text:        `bad config: {'error': 'token=abc\npassword: SECRET123'}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "truncated JSON error body where earlier token swallows escaped newline before password",
			text:        `upstream error: {"error":"token=abc\npassword: SECRET123`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "bare token value ending with escaped quote before password key",
			text:        `token=abc\"password\": SECRET123`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed quoted string where earlier token swallows escaped newline before password",
			text:        `upstream failed: "token=abc\npassword: SECRET123"`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows percent-encoded newline before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc%0Apassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows double-escaped unicode newline before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc\\u000apassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows escaped newline and unicode colon before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc\npassword\u003a SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where earlier token swallows unicode-escaped backslash and n before password",
			text:        `{"msg":"token=abc\u005cnpassword: SECRET123"}`,
			want:        `{"msg":"[REDACTED_SENSITIVE_TEXT]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows unicode-escaped backslash and n before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc\u005cnpassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where earlier token swallows percent-escaped backslash and n before password",
			text:        `{"msg":"token=abc%5Cnpassword: SECRET123"}`,
			want:        `{"msg":"[REDACTED_SENSITIVE_TEXT]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows hex-escaped backslash and n before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc\x5cnpassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where earlier token swallows unicode-escaped backslash and unicode n before password",
			text:        `POST /v1/login: 400: {"msg":"token=abc\u005c\u006epassword: SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where password colon is followed by escaped backspace and space",
			text:        `{"msg":"password:\b SECRET123"}`,
			want:        `{"msg":"password:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "prefixed JSON object where password colon is followed by escaped backspace and space",
			text:        `POST /v1/login: 400: {"msg":"password:\b SECRET123"}`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "plain text where password colon is followed by escaped backspace and space",
			text:        `password:\b SECRET123`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where password colon is followed by unicode backspace and space",
			text:        `{"msg":"password:\u0008 SECRET123"}`,
			want:        `{"msg":"password:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where password colon is followed by unicode vertical tab and space",
			text:        `{"msg":"password:\u000b SECRET123"}`,
			want:        `{"msg":"password:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:        "JSON object where password colon is followed by unicode NUL and space",
			text:        `{"msg":"password:\u0000 SECRET123"}`,
			want:        `{"msg":"password:[REDACTED]"}`,
			wantContent: redactedSensitiveTextPlaceholder,
		},
		{
			name:         "Go struct field ending in key name followed by space, closing brace, backspace byte and secret",
			text:         "failed: {Password:abc-token: }\bSECRET123",
			want:         "failed: {Password:[REDACTED]",
			wantContent:  "failed: {Password:[REDACTED]",
			wantResponse: "text: 'failed: {Password:[REDACTED]",
		},
		{
			name:        "Go struct field ending in key name followed by space, backspace byte and closing brace",
			text:        "failed: {Password:abc-token: \b}",
			want:        "failed: {Password:[REDACTED] \b}",
			wantContent: "failed: {Password:[REDACTED] \b}",
		},
		{
			name:        "caller-supplied redaction placeholders splitting secret after escaped newline",
			text:        `password:\n [REDACTED]a[REDACTED]u[REDACTED]t[REDACTED]h[REDACTED] authorization=1 authorization=2`,
			want:        redactedSensitiveTextPlaceholder,
			wantContent: redactedSensitiveTextPlaceholder,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			var errs callbackErrors
			errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", tt.text)))
			errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{
				Model:    "gemini-test",
				Contents: []*genai.Content{textContent("user", tt.text)},
			}))
			errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, map[string]any{"input": tt.text}))
			errs.add(p.AfterToolCallback()(mCtx, &fakeTool{name: "lookup"}, nil, map[string]any{"output": tt.text}, nil))
			errs.add(p.OnToolErrorCallback()(mCtx, &fakeTool{name: "db"}, map[string]any{"input": tt.text}, errors.New(tt.text)))
			errs.add(p.OnEventCallback()(mCtx, &session.Event{
				ID:          "event-1",
				Author:      "test-agent",
				Branch:      tt.text,
				LLMResponse: model.LLMResponse{Content: textContent("model", tt.text)},
			}))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			wantContent := tt.want
			if tt.wantContent != "" {
				wantContent = tt.wantContent
			}
			wantResponse := "text: '" + wantContent + "'"
			if tt.wantResponse != "" {
				wantResponse = tt.wantResponse
			}
			userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
			if got, _ := lookup(jsonObjectColumn(t, userRow, "content"), "text_summary"); got != wantContent {
				t.Errorf("USER_MESSAGE_RECEIVED content.text_summary = %v, want %q", got, wantContent)
			}
			if parts, _ := userRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["text"] != wantContent {
				t.Errorf("USER_MESSAGE_RECEIVED content_parts = %v, want text %q", parts, wantContent)
			}
			llmRow := onlyRowOfType(t, rows, EventLLMRequest)
			if prompt, _ := lookup(jsonObjectColumn(t, llmRow, "content"), "prompt"); len(prompt.([]any)) != 1 || prompt.([]any)[0].(map[string]any)["content"] != wantContent {
				t.Errorf("LLM_REQUEST content.prompt = %v, want content %q", prompt, wantContent)
			}
			if parts, _ := llmRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["text"] != wantContent {
				t.Errorf("LLM_REQUEST content_parts = %v, want text %q", parts, wantContent)
			}
			startRow := onlyRowOfType(t, rows, EventToolStarting)
			if got, _ := lookup(jsonObjectColumn(t, startRow, "content"), "args", "input"); got != wantContent {
				t.Errorf("TOOL_STARTING content.args.input = %v, want %q", got, wantContent)
			}
			toolRow := onlyRowOfType(t, rows, EventToolCompleted)
			if got, _ := lookup(jsonObjectColumn(t, toolRow, "content"), "result", "output"); got != wantContent {
				t.Errorf("TOOL_COMPLETED content.result.output = %v, want %q", got, wantContent)
			}
			errRow := onlyRowOfType(t, rows, EventToolError)
			if got := errRow["error_message"]; got != tt.want {
				t.Errorf("TOOL_ERROR error_message = %v, want %q", got, tt.want)
			}
			if got, _ := lookup(jsonObjectColumn(t, errRow, "content"), "args", "input"); got != wantContent {
				t.Errorf("TOOL_ERROR content.args.input = %v, want %q", got, wantContent)
			}
			evRow := onlyRowOfType(t, rows, EventAgentResponse)
			if got, _ := lookup(jsonObjectColumn(t, evRow, "content"), "response"); got != wantResponse {
				t.Errorf("AGENT_RESPONSE content.response = %v, want %q", got, wantResponse)
			}
			if parts, _ := evRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["text"] != wantContent {
				t.Errorf("AGENT_RESPONSE content_parts = %v, want text %q", parts, wantContent)
			}
			if got, _ := lookup(jsonObjectColumn(t, evRow, "attributes"), "adk", "branch"); got != wantContent {
				t.Errorf("AGENT_RESPONSE attributes.adk.branch = %v, want %q", got, wantContent)
			}
			for _, row := range rows {
				s := fmt.Sprint(row)
				for _, secret := range secrets {
					if strings.Contains(s, secret) {
						t.Errorf("%v row leaked %s: %s", row["event_type"], secret, s)
					}
				}
			}
		})
	}
}

// TestFileDataURICredentialsAreRedacted checks that credentials in the path,
// a nested URL, a query value or the fragment of a FileData URI do not reach
// content_parts.uri.
func TestFileDataURICredentialsAreRedacted(t *testing.T) {
	uris := []struct {
		uri  string
		want string
	}{
		{
			uri:  "https://example.com/app#/cb?access_token=SECRET123",
			want: "https://example.com/app#/cb?access_token=%5BREDACTED%5D",
		},
		{
			uri:  "https://example.com/login?next=https://api.example.com/cb?access_token=SECRET123",
			want: "https://example.com/login?next=https%3A%2F%2Fapi.example.com%2Fcb%3Faccess_token%3D%5BREDACTED%5D",
		},
		{
			uri:  "https://example.com/token:SECRET123/file.png",
			want: "https://example.com/token:%5BREDACTED%5D/file.png",
		},
		{
			uri:  "https://example.com/a?a=1;access_token=SECRET123",
			want: "https://example.com/a?a=1%3Baccess_token%3D%5BREDACTED%5D",
		},
	}
	p, requests, mCtx := setupTestPlugin(t)
	content := &genai.Content{Role: "user"}
	for _, u := range uris {
		content.Parts = append(content.Parts, &genai.Part{FileData: &genai.FileData{MIMEType: "image/png", FileURI: u.uri}})
	}
	var errs callbackErrors
	errs.add(p.OnUserMessageCallback()(mCtx, content))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
	parts, _ := userRow["content_parts"].([]map[string]any)
	if len(parts) != len(uris) {
		t.Fatalf("content_parts = %v, want %d parts", parts, len(uris))
	}
	for i, u := range uris {
		if got := parts[i]["uri"]; got != u.want {
			t.Errorf("content_parts[%d].uri = %v, want %q", i, got, u.want)
		}
	}
	if got := userRow["is_truncated"]; got != true {
		t.Errorf("is_truncated = %v, want true", got)
	}
}

// TestTextWithoutCredentialsIsUnchanged checks that text that mentions an
// authorization scheme or a sensitive word, but carries no credential, is
// written unchanged and is not marked as truncated.
func TestTextWithoutCredentialsIsUnchanged(t *testing.T) {
	texts := []string{
		"Write a basic HTTP server in Go",
		"I'm the bearer of bad news",
		"basic auth not configured",
		"[Basic usage](https://example.com/docs)",
		`[Q] What does "token" mean?`,
		"See https://example.com/search?q=basic%20recipes for ideas",
		`"Hello," she said [laughing]. "Bye."`,
		`"What does [this] mean?"`,
	}
	for _, text := range texts {
		t.Run(text, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			var errs callbackErrors
			errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", text)))
			errs.add(p.OnToolErrorCallback()(mCtx, &fakeTool{name: "lookup"}, nil, errors.New(text)))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
			if got, _ := lookup(jsonObjectColumn(t, userRow, "content"), "text_summary"); got != text {
				t.Errorf("USER_MESSAGE_RECEIVED content.text_summary = %v, want %q", got, text)
			}
			if parts, _ := userRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["text"] != text {
				t.Errorf("USER_MESSAGE_RECEIVED content_parts = %v, want text %q", parts, text)
			}
			if got := userRow["is_truncated"]; got != false {
				t.Errorf("USER_MESSAGE_RECEIVED is_truncated = %v, want false", got)
			}
			if got := onlyRowOfType(t, rows, EventToolError)["error_message"]; got != text {
				t.Errorf("TOOL_ERROR error_message = %v, want %q", got, text)
			}
		})
	}

	t.Run("file URI", func(t *testing.T) {
		const uri = "https://example.com/search?q=basic%20recipes"
		p, requests, mCtx := setupTestPlugin(t)
		var errs callbackErrors
		errs.add(p.OnUserMessageCallback()(mCtx, &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{FileData: &genai.FileData{MIMEType: "text/html", FileURI: uri}}},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
		if parts, _ := userRow["content_parts"].([]map[string]any); len(parts) != 1 || parts[0]["uri"] != uri {
			t.Errorf("content_parts = %v, want uri %q", parts, uri)
		}
		if got := userRow["is_truncated"]; got != false {
			t.Errorf("is_truncated = %v, want false", got)
		}
	})

	t.Run("large user message with pattern words and backslashes", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			text   string
			prefix string
		}{
			{
				name:   "52KB prose",
				text:   strings.Repeat("design review notes; ", 2_500) + strings.Repeat(`\`, 256) + " end",
				prefix: "design review notes; ",
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				start := time.Now()
				var errs callbackErrors
				errs.add(p.OnUserMessageCallback()(mCtx, textContent("user", tc.text)))
				errs.check(t)
				if d := time.Since(start); d > 3*time.Second {
					t.Errorf("OnUserMessageCallback took %v for a %d-byte text", d, len(tc.text))
				}
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
				userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
				gotSummary, _ := lookup(jsonObjectColumn(t, userRow, "content"), "text_summary")
				if s, _ := gotSummary.(string); !strings.HasPrefix(s, tc.prefix) {
					t.Errorf("USER_MESSAGE_RECEIVED content.text_summary = %q, want prefix %q", s, tc.prefix)
				}
			})
		}
	})
}

func TestToolErrorDeduplication_RecoveredMissingAndScoped(t *testing.T) {
	p, requests, mCtx1 := setupTestPlugin(t)
	mCtx1.sessionID = "sess-1"
	mCtx1.invocationID = "inv-1"
	mCtx1.functionCallID = "call-shared"

	mCtx2 := &mockInvocationContext{
		ctx:            mCtx1.ctx,
		agentName:      "test-agent",
		sessionID:      "sess-2",
		invocationID:   "inv-2",
		userID:         "user-2",
		functionCallID: "call-shared",
	}

	lookupTool := &fakeTool{name: "lookup"}
	args := map[string]any{"q": "weather"}
	toolErr := errors.New("lookup failed")

	var errs callbackErrors
	// 1. Interleaved tool errors across two invocations with the same callID ("call-shared"):
	// Each invocation must log exactly 1 TOOL_ERROR (2 total), without deleting the other's marker.
	errs.add(p.OnToolErrorCallback()(mCtx1, lookupTool, args, toolErr))
	errs.add(p.OnToolErrorCallback()(mCtx2, lookupTool, args, toolErr))
	errs.add(p.AfterToolCallback()(mCtx1, lookupTool, args, nil, toolErr))
	errs.add(p.AfterToolCallback()(mCtx2, lookupTool, args, nil, toolErr))

	// 2. Recovered tool error on mCtx1 (OnToolError followed by AfterTool with err == nil),
	// followed by a fallback-only AfterTool error on the same tool with empty callID:
	// The recovered marker must be consumed so the subsequent fallback error is NOT suppressed.
	mCtx1.functionCallID = ""
	errs.add(p.OnToolErrorCallback()(mCtx1, lookupTool, args, errors.New("transient error")))
	errs.add(p.AfterToolCallback()(mCtx1, lookupTool, args, map[string]any{"ok": true}, nil))
	errs.add(p.AfterToolCallback()(mCtx1, lookupTool, args, nil, errors.New("fallback error")))

	// 3. Missing-tool error on mCtx2 (OnToolError without AfterTool) before AfterRunCallback:
	// AfterRunCallback must clean up the leftover marker for mCtx2 so a later run with the same IDs can log fallback errors.
	mCtx2.functionCallID = ""
	errs.add(p.OnToolErrorCallback()(mCtx2, &fakeTool{name: "missing"}, args, errors.New("not found")))
	errs.check(t)
	p.AfterRunCallback()(mCtx2)

	// Now a fallback-only AfterTool error on "missing" in mCtx2 must still be logged!
	errs.add(p.AfterToolCallback()(mCtx2, &fakeTool{name: "missing"}, args, nil, errors.New("fallback after run cleanup")))
	errs.check(t)
	p.AfterRunCallback()(mCtx1)

	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rowsOfType(rows, EventInvocationCompleted)) >= 2
	})
	toolErrors := rowsOfType(rows, EventToolError)
	if len(toolErrors) != 6 {
		t.Fatalf("TOOL_ERROR count = %d, want 6 (got types: %v)", len(toolErrors), eventTypes(rows))
	}
}

type fakeSession struct {
	session.Session
	id      string
	userID  string
	appName string
}

func (s fakeSession) ID() string           { return s.id }
func (s fakeSession) UserID() string       { return s.userID }
func (s fakeSession) AppName() string      { return s.appName }
func (s fakeSession) State() session.State { return nil }

type fakeAgent struct {
	agent.Agent
	name string
}

func (a fakeAgent) Name() string { return a.name }

type bareInvocationContext struct {
	agent.InvocationContext
	ctx          context.Context
	invocationID string
	branch       string
	ag           agent.Agent
	sess         session.Session
}

func (b bareInvocationContext) Deadline() (time.Time, bool) { return b.ctx.Deadline() }
func (b bareInvocationContext) Done() <-chan struct{}       { return b.ctx.Done() }
func (b bareInvocationContext) Err() error                  { return b.ctx.Err() }
func (b bareInvocationContext) Value(key any) any           { return b.ctx.Value(key) }
func (b bareInvocationContext) InvocationID() string        { return b.invocationID }
func (b bareInvocationContext) Branch() string              { return b.branch }
func (b bareInvocationContext) Agent() agent.Agent          { return b.ag }
func (b bareInvocationContext) Session() session.Session    { return b.sess }

func TestIdentityOf_BareInvocationContext(t *testing.T) {
	ctx := bareInvocationContext{
		ctx:          context.Background(),
		invocationID: "inv-bare",
		branch:       "root.bare",
		ag:           fakeAgent{name: "bare-agent"},
		sess:         fakeSession{id: "sess-bare", userID: "user-bare", appName: "app-bare"},
	}
	got := identityOf(ctx)
	if got.agentName != "bare-agent" || got.sessionID != "sess-bare" || got.invocationID != "inv-bare" ||
		got.userID != "user-bare" || got.appName != "app-bare" || got.branch != "root.bare" {
		t.Errorf("identityOf(bareInvocationContext) = %+v", got)
	}
}

func TestADKVersionFromBuildInfo(t *testing.T) {
	tests := []struct {
		name string
		info *debug.BuildInfo
		want string
	}{
		{name: "nil build info", info: nil, want: ""},
		{name: "devel main module without deps", info: &debug.BuildInfo{Main: debug.Module{Path: "google.golang.org/adk/v2", Version: "(devel)"}}, want: ""},
		{name: "tagged main module", info: &debug.BuildInfo{Main: debug.Module{Path: "google.golang.org/adk/v2", Version: "v2.1.0"}}, want: "2.1.0"},
		{
			name: "tagged dependency",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app", Version: "(devel)"},
				Deps: []*debug.Module{{Path: "google.golang.org/adk/v2", Version: "v2.0.3"}},
			},
			want: "2.0.3",
		},
		{
			name: "replaced dependency uses replacement version",
			info: &debug.BuildInfo{
				Main: debug.Module{Path: "example.com/app", Version: "(devel)"},
				Deps: []*debug.Module{{
					Path:    "google.golang.org/adk/v2",
					Version: "v2.0.0",
					Replace: &debug.Module{Path: "google.golang.org/adk/v2", Version: "v2.0.4-rc1"},
				}},
			},
			want: "2.0.4-rc1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := adkVersionFromBuildInfo(tt.info); got != tt.want {
				t.Errorf("adkVersionFromBuildInfo() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCacheType(t *testing.T) {
	tests := []struct {
		name  string
		usage *genai.GenerateContentResponseUsageMetadata
		want  string
	}{
		{name: "no usage", want: "none"},
		{name: "no cached tokens", usage: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10}, want: "none"},
		{name: "cached tokens", usage: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 10, CachedContentTokenCount: 1}, want: "implicit"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cacheType(tt.usage); got != tt.want {
				t.Errorf("cacheType(%+v) = %q, want %q", tt.usage, got, tt.want)
			}
		})
	}
}

func TestBigQueryAgentAnalyticsPlugin_DropStats(t *testing.T) {
	if _, ok := AsBigQueryAgentAnalyticsPlugin(nil); ok {
		t.Error("AsBigQueryAgentAnalyticsPlugin(nil) = true, want false")
	}
	var nilPlugin *BigQueryAgentAnalyticsPlugin
	if got := nilPlugin.DropStats(); got != (DropStats{}) {
		t.Errorf("nil BigQueryAgentAnalyticsPlugin.DropStats() = %+v, want zero DropStats", got)
	}

	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.ContentFormatter = func(_ any, eventType string) (any, error) {
			if eventType == EventUserMessageReceived {
				return nil, errors.New("formatter error")
			}
			return "ok", nil
		}
	})

	bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
	if !ok || bqaap == nil {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) failed")
	}
	if got := bqaap.DropStats(); got != (DropStats{}) {
		t.Errorf("initial DropStats() = %+v, want zero DropStats", got)
	}

	if _, err := p.OnUserMessageCallback()(mCtx, textContent("user", "secret")); err != nil {
		t.Fatalf("OnUserMessageCallback error: %v", err)
	}
	p.AfterRunCallback()(mCtx)
	_ = collectRows(t, requests, untilEventType(EventInvocationCompleted))

	want := DropStats{
		FormatterFailureDrops: 1,
		TotalDrops:            0,
	}
	if diff := cmp.Diff(want, bqaap.DropStats()); diff != "" {
		t.Errorf("BigQueryAgentAnalyticsPlugin.DropStats() mismatch (-want +got):\n%s", diff)
	}

	counters, _ := bqaap.stats.(*dropCounters)
	// Verify a recovered panic inside logEvent increments AppendFailureDrops and TotalDrops once.
	ap := &analyticsPlugin{
		config:    DefaultConfig(),
		processor: &BatchProcessor{counters: counters},
	}
	ap.logEvent(context.Background(), identity{}, EventUserMessageReceived, nil, func(*eventData) {
		panic("boom in callback")
	})
	wantAfterPanic := DropStats{
		AppendFailureDrops:    1,
		FormatterFailureDrops: 1,
		TotalDrops:            1,
	}
	if diff := cmp.Diff(wantAfterPanic, bqaap.DropStats()); diff != "" {
		t.Errorf("DropStats() after recovered panic in logEvent mismatch (-want +got):\n%s", diff)
	}

	// Verify a recovered panic inside logSessionEvent also increments AppendFailureDrops and TotalDrops once.
	ap.logSessionEvent(context.Background(), identity{}, EventStateDelta, rowSpan{}, func(*eventData) {
		panic("boom in logSessionEvent")
	})
	wantAfterSessionPanic := DropStats{
		AppendFailureDrops:    2,
		FormatterFailureDrops: 1,
		TotalDrops:            2,
	}
	if diff := cmp.Diff(wantAfterSessionPanic, bqaap.DropStats()); diff != "" {
		t.Errorf("DropStats() after recovered panic in logSessionEvent mismatch (-want +got):\n%s", diff)
	}

	// Verify a disabled analyticsPlugin does not increment AppendFailureDrops on panic.
	disabledCfg := DefaultConfig()
	disabledCfg.Enabled = false
	disabledAP := &analyticsPlugin{
		config:    disabledCfg,
		processor: &BatchProcessor{counters: counters},
	}
	disabledAP.logEvent(context.Background(), identity{}, EventUserMessageReceived, nil, func(*eventData) {
		panic("boom in disabled plugin")
	})
	if diff := cmp.Diff(wantAfterSessionPanic, bqaap.DropStats()); diff != "" {
		t.Errorf("DropStats() after panic in disabled plugin mismatch (-want +got):\n%s", diff)
	}
}

func TestNewBigQueryAgentAnalyticsPluginWithClients_ClosesWriteClientOnError(t *testing.T) {
	ctx := context.Background()

	newWriteClient := func(t *testing.T) *bqstorage.BigQueryWriteClient {
		t.Helper()
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		wc, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}
		return wc
	}

	t.Run("rejects nil writeClient when plugin is enabled", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		if _, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, nil); err == nil || !strings.Contains(err.Error(), "writeClient must not be nil") {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients(writeClient=nil) error = %v, want error containing %q", err, "writeClient must not be nil")
		}
	})

	t.Run("succeeds with bqClient == nil for pre-provisioned append-only deployments", func(t *testing.T) {
		wc := newWriteClient(t)
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, wc)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients(bqClient=nil) error = %v, want nil", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close() error = %v", err)
		}
	})

	t.Run("closes writeClient when table metadata returns non-retryable HTTP 400", func(t *testing.T) {
		httpClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(_ *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       io.NopCloser(strings.NewReader(`{"error":{"code":400,"message":"bad request"}}`)),
					}, nil
				},
			},
		}
		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })

		wc := newWriteClient(t)
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"

		if _, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, wc); err == nil {
			t.Fatal("NewBigQueryAgentAnalyticsPluginWithClients() error = nil, want HTTP 400 error")
		}
		// Verify writeClient was closed by attempting to open an AppendRows stream on it.
		if _, err := wc.AppendRows(ctx); err == nil {
			t.Error("writeClient remained open after constructor error (resource leak)")
		}
	})

	t.Run("closes writeClient when validateConfig fails", func(t *testing.T) {
		wc := newWriteClient(t)
		cfg := DefaultConfig()
		cfg.PayloadColumnDenylist = []string{"not_a_valid_column"}

		if _, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, wc); err == nil {
			t.Fatal("NewBigQueryAgentAnalyticsPluginWithClients() error = nil, want validation error")
		}
		if _, err := wc.AppendRows(ctx); err == nil {
			t.Error("writeClient remained open after validateConfig error (resource leak)")
		}
	})
}

func TestNewBigQueryAgentAnalyticsPluginWithClients_UpgradesExistingOutdatedTableViaHTTP(t *testing.T) {
	ctx := context.Background()

	var patchBody string
	var patchIfMatch string
	patchCalls := 0

	httpClient := &http.Client{
		Transport: &mockTransport{
			roundTrip: func(r *http.Request) (*http.Response, error) {
				if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Etag": []string{"etag-http-1"}},
						Body: io.NopCloser(strings.NewReader(`{
							"etag": "etag-http-1",
							"schema": {
								"fields": [
									{"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"}
								]
							}
						}`)),
					}, nil
				}
				if r.Method == http.MethodPatch && strings.Contains(r.URL.Path, "/datasets/test_dataset/tables/test-table") {
					patchCalls++
					patchIfMatch = r.Header.Get("If-Match")
					b, _ := io.ReadAll(r.Body)
					patchBody = string(b)
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader(`{"etag":"etag-http-2"}`)),
					}, nil
				}
				if strings.Contains(r.URL.Path, "/jobs") {
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader(`{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`)),
					}, nil
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			},
		},
	}

	bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(httpClient))
	if err != nil {
		t.Fatalf("bq.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = bqClient.Close() })

	srv := &committedStreamTestServer{
		createdStreamName: "projects/test-project/datasets/test_dataset/tables/test-table/streams/committed-http-1",
	}
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, srv)
	go func() { _ = gSrv.Serve(lis) }()
	t.Cleanup(gSrv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewBigQueryWriteClient: %v", err)
	}

	cfg := DefaultConfig()
	cfg.ProjectID = "test-project"
	cfg.DatasetID = "test_dataset"
	cfg.TableName = "test-table"
	cfg.ExactlyOnceDelivery = true

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
	if err != nil {
		t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
	}

	if patchCalls != 1 {
		t.Fatalf("PATCH calls = %d, want 1", patchCalls)
	}
	if patchIfMatch != "etag-http-1" {
		t.Errorf("If-Match header = %q, want %q", patchIfMatch, "etag-http-1")
	}

	var parsed struct {
		Labels map[string]string `json:"labels"`
		Schema struct {
			Fields []struct {
				Name string `json:"name"`
			} `json:"fields"`
		} `json:"schema"`
	}
	if err := json.Unmarshal([]byte(patchBody), &parsed); err != nil {
		t.Fatalf("json.Unmarshal(patchBody) error = %v", err)
	}
	if got := parsed.Labels[schemaVersionKey]; got != SchemaVersion {
		t.Errorf("patched label %q = %q, want %q", schemaVersionKey, got, SchemaVersion)
	}
	if len(parsed.Schema.Fields) != len(EventsSchema()) {
		t.Errorf("patched field count = %d, want %d", len(parsed.Schema.Fields), len(EventsSchema()))
	}

	if err := p.Close(); err != nil {
		t.Fatalf("p.Close() error = %v", err)
	}

	srv.mu.Lock()
	defer srv.mu.Unlock()
	if len(srv.createReqs) != 1 {
		t.Fatalf("CreateWriteStream calls = %d, want 1", len(srv.createReqs))
	}
	if got := srv.createReqs[0].GetWriteStream().GetType(); got != storagepb.WriteStream_COMMITTED {
		t.Errorf("CreateWriteStream type = %v, want %v", got, storagepb.WriteStream_COMMITTED)
	}
	if len(srv.finalizeReqs) != 1 || srv.finalizeReqs[0].GetName() != srv.createdStreamName {
		t.Errorf("FinalizeWriteStream reqs = %+v, want [%q]", srv.finalizeReqs, srv.createdStreamName)
	}
}

func TestBigQueryAgentAnalyticsPlugin_PluginRegistryCleanupAndDisabledConstructors(t *testing.T) {
	ctx := context.Background()

	t.Run("enabled plugin remains reachable via AsBigQueryAgentAnalyticsPlugin after Close", func(t *testing.T) {
		p, _, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.QueueMaxSize = 1
		})
		if _, ok := AsBigQueryAgentAnalyticsPlugin(p); !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin() = false before Close, want true")
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close() error = %v", err)
		}
		// A caller holding only *baseplugin.Plugin can still call AsBigQueryAgentAnalyticsPlugin(p) after Close().
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok || bqaap == nil {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin() = false after Close, want true while p is reachable")
		}
		_, _ = p.OnUserMessageCallback()(mCtx, textContent("user", "after_close"))
		want := DropStats{
			ShutdownDrops: 1,
			TotalDrops:    1,
		}
		if diff := cmp.Diff(want, bqaap.DropStats()); diff != "" {
			t.Errorf("DropStats() after Close mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("disabled via NewBigQueryAgentAnalyticsPluginWithConfig remains reachable after Close", func(t *testing.T) {
		p, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, Config{Enabled: false})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v", err)
		}
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok || bqaap == nil {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin() = false before Close, want true")
		}
		if got := bqaap.DropStats(); got != (DropStats{}) {
			t.Errorf("DropStats() = %+v, want zero DropStats", got)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close() error = %v", err)
		}
		if _, ok := AsBigQueryAgentAnalyticsPlugin(p); !ok {
			t.Error("AsBigQueryAgentAnalyticsPlugin() = false after Close, want true while p is reachable")
		}
	})

	t.Run("disabled via NewBigQueryAgentAnalyticsPluginWithClients remains reachable after Close", func(t *testing.T) {
		p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, Config{Enabled: false}, nil, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if _, ok := AsBigQueryAgentAnalyticsPlugin(p); !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin() = false before Close, want true")
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close() error = %v", err)
		}
		if _, ok := AsBigQueryAgentAnalyticsPlugin(p); !ok {
			t.Error("AsBigQueryAgentAnalyticsPlugin() = false after Close, want true while p is reachable")
		}
	})

	t.Run("unreachable plugin entry is removed from pluginRegistry after GC", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			// newPlugin returns a plugin that the caller drops.
			newPlugin func(t *testing.T) *baseplugin.Plugin
			// wantStats reports whether the registry entry holds drop
			// counters, which only an enabled plugin has.
			wantStats bool
		}{
			{
				name: "disabled plugin",
				newPlugin: func(t *testing.T) *baseplugin.Plugin {
					p, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, Config{Enabled: false})
					if err != nil {
						t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v", err)
					}
					return p
				},
			},
			{
				name: "enabled plugin after Close",
				newPlugin: func(t *testing.T) *baseplugin.Plugin {
					_, writeClient := newTestWriteClient(t)
					cfg := DefaultConfig()
					cfg.ProjectID = "test-project"
					cfg.DatasetID = "test_dataset"
					p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, writeClient)
					if err != nil {
						t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
					}
					if err := p.Close(); err != nil {
						t.Fatalf("p.Close() error = %v", err)
					}
					return p
				},
				wantStats: true,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				newWeakKey := func() weak.Pointer[baseplugin.Plugin] {
					p := tc.newPlugin(t)
					key := weak.Make(p)
					v, ok := pluginRegistry.Load(key)
					if !ok {
						t.Fatal("pluginRegistry missing entry before GC")
					}
					state, _ := v.(*pluginState)
					if gotStats := state != nil && state.counters != nil; gotStats != tc.wantStats {
						t.Fatalf("pluginRegistry entry holds drop counters = %t, want %t", gotStats, tc.wantStats)
					}
					return key
				}
				key := newWeakKey()
				deadline := time.Now().Add(2 * time.Second)
				for time.Now().Before(deadline) {
					runtime.GC()
					if _, ok := pluginRegistry.Load(key); !ok {
						return
					}
					time.Sleep(5 * time.Millisecond)
				}
				t.Fatal("pluginRegistry still contained entry for unreachable plugin after GC")
			})
		}
	})

	t.Run("closed disabled plugins are garbage collected and cleaned from pluginRegistry", func(t *testing.T) {
		const count = 16
		weakKeys := make([]weak.Pointer[baseplugin.Plugin], count)
		var cleanedCount atomic.Int32

		func() {
			for i := 0; i < count; i++ {
				p, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, Config{Enabled: false})
				if err != nil {
					t.Fatalf("failed to create disabled plugin: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("failed to close disabled plugin: %v", err)
				}
				weakKeys[i] = weak.Make(p)
				runtime.AddCleanup(p, func(_ int) {
					cleanedCount.Add(1)
				}, 1)
			}
		}()

		for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
			runtime.GC()
			runtime.Gosched()
			if int(cleanedCount.Load()) == count {
				allClean := true
				for _, wk := range weakKeys {
					if _, ok := pluginRegistry.Load(wk); ok || wk.Value() != nil {
						allClean = false
						break
					}
				}
				if allClean {
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
		}

		if int(cleanedCount.Load()) != count {
			t.Fatalf("expected all %d disabled plugins to be collected, got %d", count, cleanedCount.Load())
		}
		for i, wk := range weakKeys {
			if wk.Value() != nil {
				t.Fatalf("weakKey[%d] still valid after GC", i)
			}
			if _, ok := pluginRegistry.Load(wk); ok {
				t.Fatalf("pluginRegistry still retains key for instance %d after GC", i)
			}
		}
	})

	t.Run("closed enabled plugins are garbage collected and cleaned from pluginRegistry", func(t *testing.T) {
		var weakKey weak.Pointer[baseplugin.Plugin]
		var cleaned atomic.Bool

		func() {
			mockTransport := &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			}
			httpClient := &http.Client{Transport: mockTransport}
			bqClient, err := bq.NewClient(ctx, "test-proj", option.WithHTTPClient(httpClient))
			if err != nil {
				t.Fatalf("failed to create BQ client: %v", err)
			}
			defer func() { _ = bqClient.Close() }()

			lis, err := net.Listen("tcp", "localhost:0")
			if err != nil {
				t.Fatalf("failed to listen: %v", err)
			}
			defer func() { _ = lis.Close() }()
			gSrv := grpc.NewServer()
			storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)})
			go func() { _ = gSrv.Serve(lis) }()
			defer gSrv.Stop()

			conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatalf("failed to dial: %v", err)
			}
			defer func() { _ = conn.Close() }()

			writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
			if err != nil {
				t.Fatalf("failed to create write client: %v", err)
			}

			config := DefaultConfig()
			config.Enabled = true
			config.ProjectID = "test-proj"
			config.DatasetID = "test_dataset"
			config.TableName = "test-table"
			config.DisableSchemaUpgrade = true
			config.DisableCreateViews = true

			p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
			if err != nil {
				t.Fatalf("failed to create plugin: %v", err)
			}
			if err := p.Close(); err != nil {
				t.Fatalf("p.Close() error = %v", err)
			}
			weakKey = weak.Make(p)
			runtime.AddCleanup(p, func(_ int) {
				cleaned.Store(true)
			}, 1)
		}()

		for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
			runtime.GC()
			runtime.Gosched()
			if _, ok := pluginRegistry.Load(weakKey); cleaned.Load() && weakKey.Value() == nil && !ok {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}

		if !cleaned.Load() || weakKey.Value() != nil {
			t.Fatal("expected closed plugin to be collected and not retained by pluginRegistry")
		}
		if _, ok := pluginRegistry.Load(weakKey); ok {
			t.Fatal("expected pluginRegistry entry to be deleted after GC")
		}
	})

	// An application can hand the plugin an object that holds the plugin, such
	// as a per-tenant struct used as the Logger. The registry holds its values
	// for as long as the plugin lives, so a value that reached such an object
	// would keep the plugin alive for the life of the process. That holds for
	// a plugin that is dropped without Close too, even though its
	// BatchProcessor goroutine then keeps running, as it does on main.
	for _, tc := range []struct {
		name string
		// newPlugin returns a plugin that refers to h, and a function that
		// closes what the caller owns once the plugin is closed.
		newPlugin func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func())
		// leaveOpen drops the plugin without closing it.
		leaveOpen bool
	}{
		{
			name: "closed enabled plugin with circular Logger back-reference is collected",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				return newHolderTestPlugin(t, ctx, &mockTransport{}, func(c *Config) { c.Logger = h })
			},
		},
		{
			name: "closed enabled plugin with ContentFormatter method receiver back-reference is collected",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				return newHolderTestPlugin(t, ctx, &mockTransport{}, func(c *Config) { c.ContentFormatter = h.format })
			},
		},
		{
			name: "closed disabled plugin with circular Logger back-reference is collected",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				p, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, Config{Enabled: false, Logger: h})
				if err != nil {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v", err)
				}
				return p, nil
			},
		},
		{
			name: "closed enabled plugin with constructor context back-reference is collected after Close cancels view creation",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				jobStarted := make(chan struct{})
				var started atomic.Bool
				transport := &mockTransport{roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs") {
						if started.CompareAndSwap(false, true) {
							close(jobStarted)
						}
						<-r.Context().Done()
						return nil, r.Context().Err()
					}
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
				}}
				p, closeOwned := newHolderTestPlugin(t, context.WithValue(ctx, pluginHolderKey{}, h), transport, nil)
				select {
				case <-jobStarted:
				case <-time.After(3 * time.Second):
					t.Fatal("timed out waiting for startup view creation to send a job")
				}
				return p, closeOwned
			},
		},
		{
			name: "closed enabled plugin with BigQuery client transport back-reference is collected",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				return newHolderTestPlugin(t, ctx, &mockTransport{roundTrip: h.roundTrip}, nil)
			},
		},
		{
			name: "unclosed enabled plugin with DisableCreateViews and BigQuery client transport back-reference is collected",
			newPlugin: func(t *testing.T, h *pluginHolder) (*baseplugin.Plugin, func()) {
				return newHolderTestPlugin(t, ctx, &mockTransport{roundTrip: h.roundTrip}, func(c *Config) { c.DisableCreateViews = true })
			},
			leaveOpen: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var key weak.Pointer[baseplugin.Plugin]
			func() {
				h := &pluginHolder{}
				p, closeOwned := tc.newPlugin(t, h)
				h.plugin = p
				if !tc.leaveOpen {
					if err := p.Close(); err != nil {
						t.Fatalf("Close() error = %v", err)
					}
				}
				if closeOwned != nil {
					closeOwned()
				}
				key = weak.Make(p)
			}()
			if !waitPluginCollected(key) {
				_, registered := pluginRegistry.Load(key)
				t.Fatalf("after GC: plugin collected = %v, registry entry removed = %v; want both", key.Value() == nil, !registered)
			}
		})
	}
}

// pluginHolder stands for an application object that holds a plugin and is
// also handed to it: as its Logger, as the receiver of its ContentFormatter,
// in the context that it is constructed with, or in the transport of its
// BigQuery client.
type pluginHolder struct {
	plugin *baseplugin.Plugin
}

func (h *pluginHolder) Printf(string, ...any) {}

func (h *pluginHolder) Println(...any) {}

func (h *pluginHolder) format(payload any, _ string) (any, error) {
	return payload, nil
}

func (h *pluginHolder) roundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
}

type pluginHolderKey struct{}

// newHolderTestPlugin constructs an enabled plugin with ctx. The plugin
// creates its views in the background through a BigQuery client whose
// requests go to transport, and writes rows to a fake BigQuery Storage Write
// server. mutate, if set, adjusts the config. The returned function closes
// what the caller owns: the BigQuery client, the connection and the server.
func newHolderTestPlugin(t *testing.T, ctx context.Context, transport http.RoundTripper, mutate func(*Config)) (*baseplugin.Plugin, func()) {
	t.Helper()
	bqClient, err := bq.NewClient(context.Background(), "test-project", option.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatalf("bq.NewClient: %v", err)
	}
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	gSrv := grpc.NewServer()
	storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
	go func() { _ = gSrv.Serve(lis) }()
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	writeClient, err := bqstorage.NewBigQueryWriteClient(context.Background(), option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewBigQueryWriteClient: %v", err)
	}

	cfg := DefaultConfig()
	cfg.Enabled = true
	cfg.ProjectID = "test-project"
	cfg.DatasetID = "test_dataset"
	cfg.TableName = "test-table"
	if mutate != nil {
		mutate(&cfg)
	}
	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
	if err != nil {
		t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
	}
	return p, func() {
		_ = bqClient.Close()
		_ = conn.Close()
		gSrv.Stop()
	}
}

// waitPluginCollected runs the garbage collector until the plugin that key
// points to has been collected and its pluginRegistry entry removed, and
// reports whether both happened within five seconds.
func waitPluginCollected(key weak.Pointer[baseplugin.Plugin]) bool {
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		runtime.GC()
		if _, ok := pluginRegistry.Load(key); !ok && key.Value() == nil {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func TestAddContent_FormatterFailureWhenContentDeniedRecordsDropForContentParts(t *testing.T) {
	ctx := context.Background()

	t.Run("content denied but content_parts enabled records formatter failure drop and omits content_parts", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.PayloadColumnDenylist = []string{columnContent}
		cfg.ContentFormatter = func(any, string) (any, error) {
			return nil, errors.New("boom")
		}

		bp, err := NewBatchProcessor(ctx, nil, "test_stream", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		denied, err := validateConfig(cfg)
		if err != nil {
			t.Fatalf("validateConfig() error = %v", err)
		}
		p := &analyticsPlugin{
			config:        cfg,
			processor:     bp,
			deniedColumns: denied,
		}

		row := make(map[string]any)
		ed := &eventData{content: textContent("user", "hello")}
		truncated := p.addContent(row, EventUserMessageReceived, ed)
		if truncated {
			t.Error("addContent() truncated = true, want false when ContentFormatter fails")
		}
		if ed.attributes["formatter_failed"] != true {
			t.Errorf("ed.attributes[\"formatter_failed\"] = %v, want true", ed.attributes["formatter_failed"])
		}
		if _, exists := row[columnContent]; exists {
			t.Errorf("row[%q] = %v, want omitted when columnContent is denied", columnContent, row[columnContent])
		}
		if _, exists := row[columnContentParts]; exists {
			t.Errorf("row[%q] = %v, want omitted when ContentFormatter fails", columnContentParts, row[columnContentParts])
		}
		want := DropStats{
			FormatterFailureDrops: 1,
			TotalDrops:            0,
		}
		if diff := cmp.Diff(want, bp.DropStats()); diff != "" {
			t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("both content and content_parts denied skips formatter and records zero drops", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.PayloadColumnDenylist = []string{columnContent, columnContentParts}
		cfg.ContentFormatter = func(any, string) (any, error) {
			return nil, errors.New("boom")
		}

		bp, err := NewBatchProcessor(ctx, nil, "test_stream", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		denied, err := validateConfig(cfg)
		if err != nil {
			t.Fatalf("validateConfig() error = %v", err)
		}
		p := &analyticsPlugin{
			config:        cfg,
			processor:     bp,
			deniedColumns: denied,
		}

		row := make(map[string]any)
		if truncated := p.addContent(row, EventUserMessageReceived, &eventData{content: textContent("user", "hello")}); truncated {
			t.Error("addContent() truncated = true, want false when both content and content_parts are denied")
		}
		if got := bp.DropStats(); got != (DropStats{}) {
			t.Errorf("DropStats() = %+v, want zero drops when both content and content_parts are denied", got)
		}
	})

	t.Run("content denied and multimodal content off still runs formatter and records failure on LLM_REQUEST", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.PayloadColumnDenylist = []string{columnContent}
		cfg.DisableMultiModalContent = true
		cfg.ContentFormatter = func(any, string) (any, error) {
			return nil, errors.New("formatter boom")
		}

		bp, err := NewBatchProcessor(ctx, nil, "test_stream", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		denied, err := validateConfig(cfg)
		if err != nil {
			t.Fatalf("validateConfig() error = %v", err)
		}
		p := &analyticsPlugin{
			config:        cfg,
			processor:     bp,
			deniedColumns: denied,
		}

		row := make(map[string]any)
		ed := &eventData{content: &model.LLMRequest{Model: "gemini-2.5-flash"}}
		if truncated := p.addContent(row, EventLLMRequest, ed); truncated {
			t.Error("addContent() truncated = true, want false when ContentFormatter fails")
		}
		if ed.attributes["formatter_failed"] != true {
			t.Errorf("ed.attributes[\"formatter_failed\"] = %v, want true", ed.attributes["formatter_failed"])
		}
		if !strings.Contains(ed.errMsg, "content_formatter raised") {
			t.Errorf("ed.errMsg = %q, want formatter failure note", ed.errMsg)
		}
		want := DropStats{
			FormatterFailureDrops: 1,
			TotalDrops:            0,
		}
		if diff := cmp.Diff(want, bp.DropStats()); diff != "" {
			t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestNewBigQueryAgentAnalyticsPluginWithConfig_BQClientLifecycle(t *testing.T) {
	ctx := context.Background()

	newMockGRPCWriteConn := func(t *testing.T) *grpc.ClientConn {
		t.Helper()
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })
		return conn
	}

	okHTTPClient := &http.Client{
		Transport: &mockTransport{
			roundTrip: func(r *http.Request) (*http.Response, error) {
				if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/tables/") {
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader(`{"type":"TABLE","labels":{"adk_schema_version":"2"}}`)),
					}, nil
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			},
		},
	}

	t.Run("closes internal bqClient when writeClient creation fails", func(t *testing.T) {
		var closeCount int
		origClose := closeBQClient
		closeBQClient = func(c *bq.Client) error {
			closeCount++
			_ = origClose(c)
			return errors.New("simulated bq close failure")
		}
		t.Cleanup(func() { closeBQClient = origClose })

		logger := &recordingLogger{}
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DisableCreateViews = true
		cfg.Logger = logger
		cfg.ClientOptions = []option.ClientOption{
			option.WithHTTPClient(okHTTPClient),
			option.WithCredentialsFile(filepath.Join(t.TempDir(), "missing-creds.json")),
		}

		_, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, cfg)
		if err == nil {
			t.Fatal("NewBigQueryAgentAnalyticsPluginWithConfig() error = nil, want writeClient error")
		}
		if closeCount != 1 {
			t.Errorf("closeBQClient called %d times, want 1", closeCount)
		}
		if !logger.contains("failed to close bigquery client") {
			t.Errorf("expected close error to be logged, got:\n%s", logger.String())
		}
	})

	t.Run("closes internal bqClient when table verification fails", func(t *testing.T) {
		var closeCount int
		origClose := closeBQClient
		closeBQClient = func(c *bq.Client) error {
			closeCount++
			return origClose(c)
		}
		t.Cleanup(func() { closeBQClient = origClose })

		failHTTPClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       io.NopCloser(strings.NewReader(`{"error":{"code":400,"message":"bad request"}}`)),
					}, nil
				},
			},
		}
		bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(failHTTPClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = bqClient.Close() })
		writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(newMockGRPCWriteConn(t)))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}

		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DisableCreateViews = true

		_, err = newBigQueryAgentAnalyticsPlugin(ctx, cfg, bqClient, writeClient, true)
		if err == nil {
			t.Fatal("newBigQueryAgentAnalyticsPlugin() error = nil, want table check error")
		}
		if closeCount != 1 {
			t.Errorf("closeBQClient called %d times on table verification failure, want 1", closeCount)
		}
	})

	t.Run("closes internal bqClient on plugin Close but does not close caller-supplied bqClient", func(t *testing.T) {
		var closeCount int
		origClose := closeBQClient
		closeBQClient = func(c *bq.Client) error {
			closeCount++
			return origClose(c)
		}
		t.Cleanup(func() { closeBQClient = origClose })

		internalBQ, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(okHTTPClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = internalBQ.Close() })
		internalWrite, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(newMockGRPCWriteConn(t)))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}

		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.DisableCreateViews = true

		p, err := newBigQueryAgentAnalyticsPlugin(ctx, cfg, internalBQ, internalWrite, true)
		if err != nil {
			t.Fatalf("newBigQueryAgentAnalyticsPlugin(ownBQClient=true) error = %v", err)
		}
		if closeCount != 0 {
			t.Fatalf("closeBQClient called %d times before Close(), want 0", closeCount)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close() error = %v", err)
		}
		if closeCount != 1 {
			t.Errorf("closeBQClient called %d times after Close(), want 1", closeCount)
		}

		// Public NewBigQueryAgentAnalyticsPluginWithConfig must pass ownBQClient=true
		// so p.Close() closes the internally constructed bq.Client.
		httpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/tables/") {
				_, _ = io.WriteString(w, `{"type":"TABLE","labels":{"adk_schema_version":"2"}}`)
				return
			}
			_, _ = io.WriteString(w, "{}")
		}))
		t.Cleanup(httpSrv.Close)

		publicCfg := DefaultConfig()
		publicCfg.ProjectID = "test-project"
		publicCfg.DatasetID = "test_dataset"
		publicCfg.DisableCreateViews = true
		publicCfg.ClientOptions = []option.ClientOption{
			option.WithEndpoint(httpSrv.URL),
			option.WithoutAuthentication(),
			option.WithGRPCDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
		}
		pPublic, err := NewBigQueryAgentAnalyticsPluginWithConfig(ctx, publicCfg)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v", err)
		}
		if closeCount != 1 {
			t.Fatalf("closeBQClient called %d times before closing WithConfig plugin, want 1", closeCount)
		}
		if err := pPublic.Close(); err != nil {
			t.Fatalf("pPublic.Close() error = %v", err)
		}
		if closeCount != 2 {
			t.Errorf("closeBQClient called %d times after closing WithConfig plugin, want 2", closeCount)
		}

		// Caller-supplied bqClient via NewBigQueryAgentAnalyticsPluginWithClients must NOT be closed by p.Close().
		callerBQ, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(okHTTPClient))
		if err != nil {
			t.Fatalf("bq.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = callerBQ.Close() })
		callerWrite, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(newMockGRPCWriteConn(t)))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}

		p2, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, callerBQ, callerWrite)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if err := p2.Close(); err != nil {
			t.Fatalf("p2.Close() error = %v", err)
		}
		if closeCount != 2 {
			t.Errorf("closeBQClient called %d times after closing WithClients plugin, want 2 (caller owns bqClient)", closeCount)
		}
	})
}

func TestPlugin_CloseOnceAndAfterRunPanic(t *testing.T) {
	t.Run("Close executes shutdown once across concurrent and repeated calls", func(t *testing.T) {
		ctx := context.Background()
		lis := bufconn.Listen(1 << 20)
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(
			"passthrough:///bufnet",
			grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		okHTTPClient := &http.Client{
			Transport: &mockTransport{
				roundTrip: func(r *http.Request) (*http.Response, error) {
					if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/tables/") {
						return &http.Response{
							StatusCode: http.StatusOK,
							Body:       io.NopCloser(strings.NewReader(`{"type":"TABLE","labels":{"adk_schema_version":"2"}}`)),
						}, nil
					}
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader("{}")),
					}, nil
				},
			},
		}

		var closeCalls atomic.Int32
		sentinelErr := errors.New("simulated bq close failure")
		origClose := closeBQClient
		closeBQClient = func(c *bq.Client) error {
			closeCalls.Add(1)
			_ = origClose(c)
			return sentinelErr
		}
		t.Cleanup(func() { closeBQClient = origClose })

		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test-dataset"
		cfg.TableName = "agent_events"
		cfg.DisableCreateViews = true

		bqClient, err := bq.NewClient(ctx, cfg.ProjectID, option.WithHTTPClient(okHTTPClient))
		if err != nil {
			t.Fatalf("bq.NewClient() error = %v", err)
		}
		writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient() error = %v", err)
		}

		adkPlugin, err := newBigQueryAgentAnalyticsPlugin(ctx, cfg, bqClient, writeClient, true)
		if err != nil {
			t.Fatalf("newBigQueryAgentAnalyticsPlugin() error = %v", err)
		}

		const workers = 16
		errs := make([]error, workers)
		start := make(chan struct{})
		var wg sync.WaitGroup
		for i := range workers {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				<-start
				errs[idx] = adkPlugin.Close()
			}(i)
		}
		close(start)
		wg.Wait()

		repeatErr := adkPlugin.Close()
		if got := closeCalls.Load(); got != 1 {
			t.Fatalf("closeBQClient called %d times across %d concurrent + 1 sequential Close() calls, want 1", got, workers)
		}
		for i, cErr := range errs {
			if !errors.Is(cErr, sentinelErr) {
				t.Errorf("concurrent Close()[%d] error = %v, want %v", i, cErr, sentinelErr)
			}
			if cErr != repeatErr {
				t.Errorf("concurrent Close()[%d] returned distinct error value %v vs sequential Close() %v", i, cErr, repeatErr)
			}
		}
	})

	t.Run("afterRun panic before terminal span still ends invocation and flushes run-end batch", func(t *testing.T) {
		ctx := context.Background()
		lis := bufconn.Listen(1 << 20)
		gSrv := grpc.NewServer()
		requests := make(chan *storagepb.AppendRowsRequest, 100)
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: requests})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(
			"passthrough:///bufnet",
			grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient() error = %v", err)
		}
		t.Cleanup(func() { _ = writeClient.Close() })

		cfg := DefaultConfig()
		cfg.ProjectID = "test-proj"
		cfg.DatasetID = "test-ds"
		cfg.TableName = "test-tbl"
		cfg.BatchSize = 100
		cfg.BatchFlushIntv = time.Hour
		cfg.DisableCreateViews = true

		processor, err := NewBatchProcessor(ctx, writeClient, "projects/test-proj/datasets/test-ds/tables/test-tbl/streams/_default", cfg)
		if err != nil {
			t.Fatalf("NewBatchProcessor() error = %v", err)
		}
		t.Cleanup(processor.Close)

		tm := newTraceManager()
		ap := &analyticsPlugin{
			config:    cfg,
			processor: processor,
			traces:    tm,
		}

		mCtx := &mockInvocationContext{
			ctx:          ctx,
			agentName:    "test-agent",
			sessionID:    "sess-1",
			invocationID: "inv-afterrun-panic",
			userID:       "user-1",
		}
		if _, err := ap.beforeRun(mCtx); err != nil {
			t.Fatalf("beforeRun() error = %v", err)
		}
		if _, err := ap.onUserMessage(mCtx, textContent("user", "hello")); err != nil {
			t.Fatalf("onUserMessage() error = %v", err)
		}

		tm.mu.Lock()
		_, liveBefore := tm.invocations["inv-afterrun-panic"]
		tm.mu.Unlock()
		if !liveBefore {
			t.Fatal("expected invocation trace to be active before afterRun")
		}

		var panickedOnce atomic.Bool
		tm.now = func() time.Time {
			if panickedOnce.CompareAndSwap(false, true) {
				panic("boom inside popUnclosedAgentSpans")
			}
			return time.Now()
		}

		ap.afterRun(mCtx)

		tm.mu.Lock()
		_, stillLive := tm.invocations["inv-afterrun-panic"]
		_, inCompleted := tm.completed["inv-afterrun-panic"]
		tm.mu.Unlock()
		if stillLive {
			t.Error("invocation remained in traces.invocations after panic in afterRun; want endInvocation cleanup")
		}
		if !inCompleted {
			t.Error("invocation missing from traces.completed after panic in afterRun; want endInvocation completion record")
		}

		select {
		case req := <-requests:
			rows := decodeArrowRows(t, req)
			if len(rows) < 2 {
				t.Fatalf("flushed %d rows on afterRun panic, want >= 2 queued rows; event types: %v", len(rows), eventTypes(rows))
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for flushRunEnd batch after panic in afterRun")
		}

		if got := processor.DropStats().AppendFailureDrops; got != 1 {
			t.Errorf("AppendFailureDrops = %d, want 1 from recovered afterRun panic", got)
		}
	})
}

func TestClose_SecondCallAfterCleanCloseSucceeds(t *testing.T) {
	p, _, _ := setupTestPluginWithConfig(t, nil)

	if err := p.Close(); err != nil {
		t.Fatalf("first Close() error = %v, want nil", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("second Close() error = %v, want nil", err)
	}
}

func TestNestedRetiredRecovery_Runner(t *testing.T) {
	makeBranch := func(t *testing.T, name, childName string, childErr error, recoverText string) agent.Agent {
		t.Helper()
		child, err := agent.New(agent.Config{
			Name: childName,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = childName
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", childName+" answer")
					if !yield(ev, nil) {
						return
					}
					yield(nil, childErr)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%s): %v", childName, err)
		}
		par := newParallelAgent(t, name+"_par", child)
		cfg := agent.Config{
			Name:      name,
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		}
		if recoverText != "" {
			cfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return textContent("model", recoverText), nil
				},
			}
		}
		inner, err := agent.New(cfg)
		if err != nil {
			t.Fatalf("agent.New(%s): %v", name, err)
		}
		return inner
	}

	makeOuter := func(t *testing.T, recoverText string, subs ...agent.Agent) agent.Agent {
		t.Helper()
		cfg := agent.Config{
			Name:      "outer",
			SubAgents: subs,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, sub := range subs {
						for ev, runErr := range sub.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
				}
			},
		}
		if recoverText != "" {
			cfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return textContent("model", recoverText), nil
				},
			}
		}
		outer, err := agent.New(cfg)
		if err != nil {
			t.Fatalf("agent.New(outer): %v", err)
		}
		return outer
	}

	makeBufferingRoot := func(t *testing.T, outer agent.Agent) agent.Agent {
		t.Helper()
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{outer},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered []*session.Event
					for ev, runErr := range outer.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil {
							buffered = append(buffered, ev)
						}
					}
					for _, ev := range buffered {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		return root
	}

	t.Run("inner recovers and outer forwards", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		inner := makeBranch(t, "inner", "child", errors.New("child branch failure"), "inner recovered")
		outer := makeOuter(t, "", inner)
		root := makeBufferingRoot(t, outer)

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("inner does not recover and outer recovers", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		inner := makeBranch(t, "inner", "child", errors.New("child branch failure"), "")
		outer := makeOuter(t, "outer recovered", inner)
		root := makeBufferingRoot(t, outer)

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s2", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("neither inner nor outer recovers", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		inner := makeBranch(t, "inner", "child", errors.New("child unrecovered failure"), "")
		outer := makeOuter(t, "", inner)
		root := makeBufferingRoot(t, outer)

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s3", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if invErr["status"] != "ERROR" {
			t.Errorf("INVOCATION_ERROR status = %v, want ERROR", invErr["status"])
		}
	})

	t.Run("unrelated sibling failure stays intact when inner1 recovers", func(t *testing.T) {
		for _, recoverFirst := range []bool{true, false} {
			name := "recover first"
			if !recoverFirst {
				name = "fail silently first"
			}
			t.Run(name, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				err1 := errors.New("child1 recoverable error")
				err2 := errors.New("child2 silent error")
				inner1 := makeBranch(t, "inner1", "child1", err1, "inner1 recovered")
				inner2 := makeBranch(t, "inner2", "child2", err2, "")
				subs := []agent.Agent{inner1, inner2}
				if !recoverFirst {
					subs = []agent.Agent{inner2, inner1}
				}
				outer := makeOuter(t, "", subs...)
				root := makeBufferingRoot(t, outer)

				for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s4", textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error = %v, want nil", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
				}
				invErr := onlyRowOfType(t, rows, EventInvocationError)
				msg, _ := invErr["error_message"].(string)
				if strings.Contains(msg, err1.Error()) || !strings.Contains(msg, err2.Error()) {
					t.Errorf("INVOCATION_ERROR error_message = %q, want %q and not %q", msg, err2.Error(), err1.Error())
				}
			})
		}
	})
}

func TestDelayedRecoveryDoesNotClearLaterActivation_Runner(t *testing.T) {
	makeSwitchableBranch := func(t *testing.T, name string, runs []struct {
		childName   string
		childErr    error
		runText     string
		recoverText string
	},
	) agent.Agent {
		t.Helper()
		var mu sync.Mutex
		var idx int
		type runData struct {
			par         agent.Agent
			runText     string
			recoverText string
		}
		prepared := make([]runData, 0, len(runs))
		subAgents := make([]agent.Agent, 0, len(runs))
		for i, rc := range runs {
			childErr := rc.childErr
			childName := rc.childName
			child, err := agent.New(agent.Config{
				Name: childName,
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = childName
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", childName+" answer")
						if !yield(ev, nil) {
							return
						}
						if childErr != nil {
							yield(nil, childErr)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(%s): %v", childName, err)
			}
			par := newParallelAgent(t, fmt.Sprintf("%s_par_%d", name, i), child)
			prepared = append(prepared, runData{par: par, runText: rc.runText, recoverText: rc.recoverText})
			subAgents = append(subAgents, par)
		}
		var active *runData
		ag, err := agent.New(agent.Config{
			Name:      name,
			SubAgents: subAgents,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				mu.Lock()
				cur := &prepared[idx]
				active = cur
				idx++
				mu.Unlock()
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range cur.par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if cur.runText != "" {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = name
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", cur.runText)
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					mu.Lock()
					cur := active
					mu.Unlock()
					if cur == nil || cur.recoverText == "" {
						return nil, nil
					}
					return textContent("model", cur.recoverText), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(%s): %v", name, err)
		}
		return ag
	}

	makeDualRoot := func(t *testing.T, child agent.Agent, delayAuthors ...string) agent.Agent {
		t.Helper()
		delaySet := make(map[string]bool, len(delayAuthors))
		for _, a := range delayAuthors {
			delaySet[a] = true
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{child},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered []*session.Event
					// Run 1
					for ev, runErr := range child.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && delaySet[ev.Author] {
							buffered = append(buffered, ev)
							continue
						}
						if ev != nil {
							if !yield(ev, nil) {
								return
							}
						}
					}
					// Run 2
					for ev, runErr := range child.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && delaySet[ev.Author] {
							buffered = append(buffered, ev)
							continue
						}
						if ev != nil {
							if !yield(ev, nil) {
								return
							}
						}
					}
					for _, ev := range buffered {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		return root
	}

	t.Run("immediate delivery control preserves later unrecovered failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		err1 := errors.New("run1 recoverable error")
		err2 := errors.New("run2 unrecovered error")
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: err1, recoverText: "run1 recovered"},
			{childName: "child2", childErr: err2, recoverText: ""},
		})
		root := makeDualRoot(t, inner)

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-ctrl", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if strings.Contains(msg, err1.Error()) || !strings.Contains(msg, err2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q and not %q", msg, err2.Error(), err1.Error())
		}
	})

	t.Run("delayed recovery from earlier run does not clear later unrecovered activation", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		err1 := errors.New("run1 recoverable error")
		err2 := errors.New("run2 unrecovered error")
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: err1, recoverText: "run1 recovered"},
			{childName: "child2", childErr: err2, recoverText: ""},
		})
		root := makeDualRoot(t, inner, "inner")

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-delay-inner", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, err1.Error()) && !strings.Contains(msg, err2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q or %q preserved", msg, err1.Error(), err2.Error())
		}
	})

	t.Run("two buffered responses from recovered first run do not clear later unrecovered activation", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		err1 := errors.New("run1 recoverable error")
		err2 := errors.New("run2 unrecovered error")
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: err1, runText: "run1 progress", recoverText: "run1 recovered"},
			{childName: "child2", childErr: err2, recoverText: ""},
		})
		outer, err := agent.New(agent.Config{
			Name:      "outer",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(outer): %v", err)
		}
		root := makeDualRoot(t, outer, "inner")

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-two-buffered", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, err1.Error()) && !strings.Contains(msg, err2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q or %q preserved", msg, err1.Error(), err2.Error())
		}
	})

	t.Run("buffered response from clean first run does not clear later unrecovered activation", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		err2 := errors.New("run2 unrecovered after clean run1")
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: nil, recoverText: "run1 clean summary"},
			{childName: "child2", childErr: err2, recoverText: ""},
		})
		root := makeDualRoot(t, inner, "inner")

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-clean-first", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, err2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, err2.Error())
		}
	})

	t.Run("built-in LoopAgent clean first iteration allows second iteration recovery", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: nil, recoverText: "iter1 clean"},
			{childName: "child2", childErr: errors.New("iter2 recoverable error"), recoverText: "iter2 recovered"},
		})
		loop, err := loopagent.New(loopagent.Config{
			AgentConfig: agent.Config{
				Name:      "loop",
				SubAgents: []agent.Agent{inner},
			},
			MaxIterations: 2,
		})
		if err != nil {
			t.Fatalf("loopagent.New: %v", err)
		}

		for _, err := range newTestRunner(t, loop, p).Run(t.Context(), "u1", "s-loop-clean-first", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("delayed nested wrapper recovery does not clear later unrecovered wrapper+inner run", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		err1 := errors.New("run1 nested recoverable error")
		err2 := errors.New("run2 nested unrecovered error")
		inner := makeSwitchableBranch(t, "inner", []struct {
			childName   string
			childErr    error
			runText     string
			recoverText string
		}{
			{childName: "child1", childErr: err1, recoverText: ""},
			{childName: "child2", childErr: err2, recoverText: ""},
		})
		var outerRun int
		outer, err := agent.New(agent.Config{
			Name:      "outer",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				outerRun++
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if outerRun == 1 {
						return textContent("model", "outer run1 recovered"), nil
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(outer): %v", err)
		}
		root := makeDualRoot(t, outer, "outer")

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-delay-outer", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, err1.Error()) && !strings.Contains(msg, err2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q or %q preserved", msg, err1.Error(), err2.Error())
		}
	})

	t.Run("buffered response from first run flushed while second activation is open preserves error", func(t *testing.T) {
		runOpenFlushCase := func(t *testing.T, flushRun1, rootRecovers bool) {
			t.Helper()
			p, requests, _ := setupTestPlugin(t)
			err2 := errors.New("run2 unrecovered while open")
			helper, err := agent.New(agent.Config{
				Name: "helper",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "helper"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "tick")
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(helper): %v", err)
			}
			child1, err := agent.New(agent.Config{
				Name: "child1",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "child1"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "c1-ok")
						yield(ev, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(child1): %v", err)
			}
			child2, err := agent.New(agent.Config{
				Name: "child2",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "child2"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "c2-ok")
						if !yield(ev, nil) {
							return
						}
						yield(nil, err2)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(child2): %v", err)
			}
			par1 := newParallelAgent(t, "par1", child1)
			par2 := newParallelAgent(t, "par2", child2)

			var innerRun int
			inner, err := agent.New(agent.Config{
				Name:      "inner",
				SubAgents: []agent.Agent{par1, par2, helper},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					innerRun++
					curRun := innerRun
					return func(yield func(*session.Event, error) bool) {
						if curRun == 1 {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "inner"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "run1-a")
							if !yield(ev, nil) {
								return
							}
							for ev, runErr := range par1.Run(ctx) {
								if runErr == nil && ev != nil && !yield(ev, nil) {
									return
								}
							}
							return
						}
						for ev, runErr := range par2.Run(ctx) {
							if runErr != nil {
								continue
							}
							if ev != nil && !yield(ev, nil) {
								return
							}
						}
						for ev, runErr := range helper.Run(ctx) {
							if runErr == nil && ev != nil && !yield(ev, nil) {
								return
							}
						}
					}
				},
				AfterAgentCallbacks: []agent.AfterAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						if innerRun == 1 {
							return textContent("model", "run1-b"), nil
						}
						return nil, nil
					},
				},
			})
			if err != nil {
				t.Fatalf("agent.New(inner): %v", err)
			}

			var rootCallbacks []agent.AfterAgentCallback
			if rootRecovers {
				rootCallbacks = []agent.AfterAgentCallback{
					func(agent.Context) (*genai.Content, error) {
						return textContent("model", "root final recovery"), nil
					},
				}
			}
			root, err := agent.New(agent.Config{
				Name:                "root",
				SubAgents:           []agent.Agent{inner},
				AfterAgentCallbacks: rootCallbacks,
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						var buffered *session.Event
						// Run 1: deliver run1-a immediately and buffer run1-b.
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if ev != nil && ev.Author == "inner" && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "run1-b" {
								buffered = ev
								continue
							}
							if ev != nil && !yield(ev, nil) {
								return
							}
						}
						// Run 2: flush run1-b when helper ticks while inner2 is still open.
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if ev != nil && ev.Author == "helper" && flushRun1 && buffered != nil {
								flushed := buffered
								buffered = nil
								if !yield(flushed, nil) {
									return
								}
							}
							if ev != nil && !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}

			for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open2-flush", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error = %v, want nil", err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if rootRecovers {
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
					t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
				}
				return
			}
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			msg, _ := invErr["error_message"].(string)
			if !strings.Contains(msg, err2.Error()) {
				t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, err2.Error())
			}
		}

		t.Run("control no flush", func(t *testing.T) { runOpenFlushCase(t, false, false) })
		t.Run("flush run1 buffered while inner2 open", func(t *testing.T) { runOpenFlushCase(t, true, false) })
		t.Run("root final recovery after flush", func(t *testing.T) { runOpenFlushCase(t, true, true) })
	})

	t.Run("buffered LLM response from first run flushed while open second LlmAgent has model error preserves failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errModel2 := errors.New("bot2 model unrecovered error")
		readyToFlush := make(chan struct{})
		flushDone := make(chan struct{})
		var readyOnce, flushOnce sync.Once
		botLLM := &firstCallResponseThenErrLLM{
			first: &model.LLMResponse{Content: textContent("model", "bot1-buffered-final"), TurnComplete: true},
			err:   errModel2,
		}
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Model: botLLM,
			OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
				func(ctx agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
					readyOnce.Do(func() { close(readyToFlush) })
					select {
					case <-flushDone:
					case <-ctx.Done():
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered *session.Event
					// Run 1: buffer bot1's model response.
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Author == "bot" {
							buffered = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					// Run 2: run bot2 concurrently so root flushes bot1's buffered
					// event while bot2 is open on agentStack with errModel2 recorded
					// in unrecoveredErrors (root swallows bot2's returned error).
					streamBufferedConcurrentRun(ctx, bot, readyToFlush, buffered, func() {
						flushOnce.Do(func() { close(flushDone) })
					}, true, yield)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open2-llm-flush", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("built-in LoopAgent allows open second activation recovery", func(t *testing.T) {
		t.Run("LoopAgent iter2 open recovery in Run clears iter2 failure", func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			inner := makeSwitchableBranch(t, "inner", []struct {
				childName   string
				childErr    error
				runText     string
				recoverText string
			}{
				{childName: "child1", childErr: nil, runText: "iter1-ok"},
				{childName: "child2", childErr: errors.New("iter2 open failure"), runText: "iter2-recovered-in-run"},
			})
			loop, err := loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{
					Name:      "loop",
					SubAgents: []agent.Agent{inner},
				},
				MaxIterations: 2,
			})
			if err != nil {
				t.Fatalf("loopagent.New: %v", err)
			}

			for _, err := range newTestRunner(t, loop, p).Run(t.Context(), "u1", "s-loop-open2", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error = %v, want nil", err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
			}
			if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
				t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
			}
		})

		t.Run("LoopAgent iter2 open recovery after iter1 custom wrapper retired", func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			inner := makeSwitchableBranch(t, "inner", []struct {
				childName   string
				childErr    error
				runText     string
				recoverText string
			}{
				{childName: "child1", childErr: nil, runText: "iter1-ok"},
				{childName: "child2", childErr: errors.New("iter2 open recoverable"), runText: "iter2-recovered-in-run"},
			})
			var outerRun int
			bufOuter, err := agent.New(agent.Config{
				Name:      "bufOuter",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					outerRun++
					curRun := outerRun
					return func(yield func(*session.Event, error) bool) {
						var buffered []*session.Event
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil || ev == nil {
								continue
							}
							if curRun == 1 {
								buffered = append(buffered, ev)
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						for _, ev := range buffered {
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(bufOuter): %v", err)
			}
			loopRoot, err := loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{
					Name:      "loopRoot",
					SubAgents: []agent.Agent{bufOuter},
				},
				MaxIterations: 2,
			})
			if err != nil {
				t.Fatalf("loopagent.New: %v", err)
			}

			for _, err := range newTestRunner(t, loopRoot, p).Run(t.Context(), "u1", "s-loop-wrap-open2", textContent("user", "go"), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("runner.Run error = %v, want nil", err)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v, agentErrorCounts=%v", n, eventTypes(rows), agentErrorCounts(rows))
			}
			if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
				t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
			}
		})
	})

	t.Run("buffered FunctionResponse from first run flushed while open second LlmAgent tool is running preserves TOOL_COMPLETED span and latency", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		readyToFlush := make(chan struct{})
		flushDone := make(chan struct{})
		var readyOnce, flushOnce sync.Once
		var toolRuns atomic.Int32
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			runNum := toolRuns.Add(1)
			ctx.Actions().SkipSummarization = true
			if runNum == 2 {
				readyOnce.Do(func() { close(readyToFlush) })
				select {
				case <-flushDone:
				case <-ctx.Done():
				}
			}
			return map[string]any{"res": fmt.Sprintf("r%d", runNum)}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_0", Name: "lookup", Args: map[string]any{}}}},
					},
				}, nil)
			}},
			Tools: []tool.Tool{lookupTool},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && len(functionResponseCalls(ev.Content)) > 0 {
							buffered = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					streamBufferedConcurrentRun(ctx, bot, readyToFlush, buffered, func() {
						flushOnce.Do(func() { close(flushDone) })
					}, false, yield)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open2-tool-flush", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
		starts := rowsOfType(rows, EventToolStarting)
		ends := rowsOfType(rows, EventToolCompleted)
		if len(starts) != 2 || len(ends) != 2 {
			t.Fatalf("tool rows = (starts=%d, ends=%d), want (2, 2); event types=%v", len(starts), len(ends), eventTypes(rows))
		}
		for idx := range 2 {
			if got, want := ends[idx]["span_id"], starts[idx]["span_id"]; got != want {
				t.Errorf("TOOL_COMPLETED[%d] span_id = %v, want TOOL_STARTING[%d] span_id %v", idx, got, idx, want)
			}
			if _, ok := ends[idx][columnLatencyMS]; !ok {
				t.Errorf("TOOL_COMPLETED[%d] row missing %s", idx, columnLatencyMS)
			}
		}
	})

	t.Run("clean LlmAgent BeforeModelCallbacks cache hit allows second run recovery", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			BeforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(_ agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{Content: textContent("model", "cached-1"), TurnComplete: true}, nil
				},
			},
			Model: &funcStreamLLM{fn: func(_ context.Context, _ func(*model.LLMResponse, error) bool) {
				t.Error("bot1 Model should not be called on BeforeModelCallbacks cache hit")
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errChild2 := errors.New("bot2 child recoverable failure")
		failingChild, err := agent.New(agent.Config{
			Name: "p8_failing_child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "p8_failing_child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(p8_failing_child): %v", err)
		}
		par2 := newParallelAgent(t, "p8_par2", failingChild)
		bot2, err := agent.New(agent.Config{
			Name:      "bot",
			SubAgents: []agent.Agent{par2},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "bot"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "bot2-recovered")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-clean-before-model-break", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("clean LlmAgent earlier AfterModelCallback plugin with OnModelErrorCallbacks fallback allows second run recovery", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		earlierAfterModel, err := baseplugin.New(baseplugin.Config{
			Name: "earlier_after_model",
			AfterModelCallback: func(_ agent.Context, res *model.LLMResponse, err error) (*model.LLMResponse, error) {
				if err != nil || res == nil {
					return nil, err
				}
				return res, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(earlier_after_model): %v", err)
		}
		errModel1 := errors.New("bot1 transient model failure")
		errModel2 := errors.New("bot2 transient model failure")
		var modelCalls atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if modelCalls.Add(1) == 1 {
					yield(nil, errModel1)
					return
				}
				yield(nil, errModel2)
			}},
			OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
				func(_ agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
					return &model.LLMResponse{Content: textContent("model", "bot-fallback"), TurnComplete: true}, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range bot.Run(ctx) {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							if ev != nil && !yield(ev, nil) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, earlierAfterModel, p).Run(t.Context(), "u1", "s-clean-earlier-after-model", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("clean LlmAgent earlier AfterToolCallback plugin with SkipSummarization allows second run recovery", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		earlierAfterTool, err := baseplugin.New(baseplugin.Config{
			Name: "earlier_after_tool",
			AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, _ map[string]any, err error) (map[string]any, error) {
				if err == nil {
					return map[string]any{"res": "intercepted-1"}, nil
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(earlier_after_tool): %v", err)
		}
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			ctx.Actions().SkipSummarization = true
			return map[string]any{"res": "r1"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{}}}},
					},
				}, nil)
			}},
			Tools: []tool.Tool{lookupTool},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errChild2 := errors.New("bot2 child recoverable failure")
		failingChild, err := agent.New(agent.Config{
			Name: "p14_failing_child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "p14_failing_child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(p14_failing_child): %v", err)
		}
		par2 := newParallelAgent(t, "p14_par2", failingChild)
		bot2, err := agent.New(agent.Config{
			Name:      "bot",
			SubAgents: []agent.Agent{par2},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "bot"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "bot2-recovered")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, earlierAfterTool, p).Run(t.Context(), "u1", "s-clean-earlier-after-tool", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("later AfterAgentCallback plugin event buffered from clean LlmAgent does not clear second activation failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var cbCalls atomic.Int32
		laterAfterAgent, err := baseplugin.New(baseplugin.Config{
			Name: "later_after_agent",
			AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
				if ctx.AgentName() == "bot" && cbCalls.Add(1) == 1 {
					return textContent("model", "plugin-tail-1"), nil
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(later_after_agent): %v", err)
		}
		errModel2 := errors.New("bot2 unrecovered model error")
		readyToFlush := make(chan struct{})
		flushDone := make(chan struct{})
		var readyOnce, flushOnce sync.Once
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &firstCallResponseThenErrLLM{
				first: &model.LLMResponse{Content: textContent("model", "bot1-model"), TurnComplete: true},
				err:   errModel2,
			},
			OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
				func(ctx agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
					readyOnce.Do(func() { close(readyToFlush) })
					select {
					case <-flushDone:
					case <-ctx.Done():
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "plugin-tail-1" {
							buffered = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					streamBufferedConcurrentRun(ctx, bot, readyToFlush, buffered, func() {
						flushOnce.Do(func() { close(flushDone) })
					}, true, yield)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p, laterAfterAgent).Run(t.Context(), "u1", "s-later-after-agent", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("multi-turn LlmAgent buffered FunctionResponse preserves later sequential run failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(_ agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{"res": "tool-ok"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		errModel2 := errors.New("bot2 sequential multi-turn error")
		var callCount atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				switch callCount.Add(1) {
				case 1:
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call_seq_1",
									Name: "lookup",
									Args: map[string]any{},
								},
							}},
						},
						TurnComplete: true,
					}, nil)
				case 2:
					yield(&model.LLMResponse{Content: textContent("model", "bot1-turn2-final"), TurnComplete: true}, nil)
				default:
					yield(nil, errModel2)
				}
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFR *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
							bufferedFR = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							if bufferedFR != nil {
								yield(bufferedFR, nil)
								bufferedFR = nil
							}
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-multiturn-seq-fr", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("multi-turn LlmAgent buffered FunctionResponse preserves concurrent open activation failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(_ agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{"res": "tool-ok"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		errModel2 := errors.New("bot2 concurrent multi-turn error")
		readyToFlush := make(chan struct{})
		flushDone := make(chan struct{})
		var readyOnce, flushOnce sync.Once
		var callCount atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				switch callCount.Add(1) {
				case 1:
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call_conc_1",
									Name: "lookup",
									Args: map[string]any{},
								},
							}},
						},
						TurnComplete: true,
					}, nil)
				case 2:
					yield(&model.LLMResponse{Content: textContent("model", "bot1-turn2-final"), TurnComplete: true}, nil)
				default:
					yield(nil, errModel2)
				}
			}},
			OnModelErrorCallbacks: []llmagent.OnModelErrorCallback{
				func(ctx agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
					readyOnce.Do(func() { close(readyToFlush) })
					select {
					case <-flushDone:
					case <-ctx.Done():
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFR *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
							bufferedFR = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					streamBufferedConcurrentRun(ctx, bot, readyToFlush, bufferedFR, func() {
						flushOnce.Do(func() { close(flushDone) })
					}, true, yield)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-multiturn-conc-fr", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("earlier BeforeModelCallback benign plugin with sequential recovery allows invocation completion", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		benignBeforeModel, err := baseplugin.New(baseplugin.Config{
			Name: "benign_before_model",
			BeforeModelCallback: func(_ agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(benign_before_model): %v", err)
		}
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			BeforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(_ agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					return &model.LLMResponse{Content: textContent("model", "cached-1"), TurnComplete: true}, nil
				},
			},
			Model: &funcStreamLLM{fn: func(_ context.Context, _ func(*model.LLMResponse, error) bool) {
				t.Error("bot1 Model should not be called on BeforeModelCallbacks cache hit")
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errChild2 := errors.New("bot2 child recoverable failure")
		failingChild, err := agent.New(agent.Config{
			Name: "h3_failing_child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "h3_failing_child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(h3_failing_child): %v", err)
		}
		par2 := newParallelAgent(t, "h3_par2", failingChild)
		bot2, err := agent.New(agent.Config{
			Name:      "bot",
			SubAgents: []agent.Agent{par2},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "bot"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "bot2-recovered")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, benignBeforeModel, p).Run(t.Context(), "u1", "s-h3-benign-before-model", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
			t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
		}
	})

	t.Run("LlmAgent AfterAgentCallbacks OutputSchema ModeTask and custom predecessor with sequential recovery allow invocation completion", func(t *testing.T) {
		for _, modeCase := range []struct {
			name   string
			task   bool
			custom bool
		}{
			{name: "AfterAgentCallbacks and OutputSchema"},
			{name: "ModeTask", task: true},
			{name: "custom_bot1", custom: true},
		} {
			t.Run(modeCase.name, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				var bot1 agent.Agent
				var err error
				var taskCalls atomic.Int32
				if modeCase.custom {
					bot1, err = agent.New(agent.Config{
						Name: "bot",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								ev := session.NewEvent(ctx, ctx.InvocationID())
								ev.Author = "bot"
								ev.Branch = ctx.Branch()
								ev.Content = textContent("model", "custom-bot1-ok")
								yield(ev, nil)
							}
						},
					})
				} else if modeCase.task {
					taskSub, subErr := llmagent.New(llmagent.Config{
						Name: "task_worker_r8",
						Mode: llmagent.ModeTask,
						Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
							yield(&model.LLMResponse{
								Content: &genai.Content{
									Role: "model",
									Parts: []*genai.Part{{
										FunctionCall: &genai.FunctionCall{
											ID:   "finish-r8",
											Name: "finish_task",
											Args: map[string]any{"result": "done"},
										},
									}},
								},
								TurnComplete: true,
							}, nil)
						}},
					})
					if subErr != nil {
						t.Fatalf("llmagent.New(taskSub): %v", subErr)
					}
					bot1, err = llmagent.New(llmagent.Config{
						Name:      "bot",
						Mode:      llmagent.ModeChat,
						SubAgents: []agent.Agent{taskSub},
						Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
							if taskCalls.Add(1) == 1 {
								yield(&model.LLMResponse{
									Content: &genai.Content{
										Role: "model",
										Parts: []*genai.Part{{
											FunctionCall: &genai.FunctionCall{
												ID:   "call-task-r8",
												Name: "task_worker_r8",
												Args: map[string]any{"request": "complete task"},
											},
										}},
									},
									TurnComplete: true,
								}, nil)
								return
							}
							yield(&model.LLMResponse{Content: textContent("model", "delegation done"), TurnComplete: true}, nil)
						}},
					})
				} else {
					bot1, err = llmagent.New(llmagent.Config{
						Name:         "bot",
						OutputSchema: &genai.Schema{Type: genai.TypeObject},
						AfterAgentCallbacks: []agent.AfterAgentCallback{
							func(_ agent.Context) (*genai.Content, error) {
								return nil, nil
							},
						},
						BeforeModelCallbacks: []llmagent.BeforeModelCallback{
							func(_ agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
								return &model.LLMResponse{Content: textContent("model", `{"ok":true}`), TurnComplete: true}, nil
							},
						},
						Model: &funcStreamLLM{fn: func(_ context.Context, _ func(*model.LLMResponse, error) bool) {
							t.Error("bot1 Model should not be called on BeforeModelCallbacks cache hit")
						}},
					})
				}
				if err != nil {
					t.Fatalf("llmagent.New(bot1): %v", err)
				}
				errChild2 := errors.New("bot2 child recoverable failure")
				failingChild, err := agent.New(agent.Config{
					Name: "h3_cfg_failing_child",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "h3_cfg_failing_child"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "child partial")
							if !yield(ev, nil) {
								return
							}
							yield(nil, errChild2)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(h3_cfg_failing_child): %v", err)
				}
				par2 := newParallelAgent(t, "h3_cfg_par2", failingChild)
				bot2, err := agent.New(agent.Config{
					Name:      "bot",
					SubAgents: []agent.Agent{par2},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range par2.Run(ctx) {
								if runErr != nil {
									continue
								}
								if ev != nil && !yield(ev, nil) {
									return
								}
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "bot"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "bot2-recovered")
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(bot2): %v", err)
				}
				root, err := agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{bot1},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range bot1.Run(ctx) {
								if runErr != nil {
									yield(nil, runErr)
									return
								}
								if ev != nil && !yield(ev, nil) {
									return
								}
							}
							for ev, runErr := range bot2.Run(ctx) {
								if runErr != nil {
									continue
								}
								if ev != nil && !yield(ev, nil) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}

				for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-h3-cfg-"+modeCase.name, textContent("user", "go"), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("runner.Run error = %v, want nil", err)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				if got := onlyRowOfType(t, rows, EventInvocationCompleted)["status"]; got != "OK" {
					t.Errorf("INVOCATION_COMPLETED status = %v, want OK", got)
				}
			})
		}
	})

	t.Run("single LlmAgent failing functiontool with Turn-2 model error preserves TOOL_ERROR and error when buffered FunctionCall flushes", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errTool := errors.New("tool execution failed")
		errModel2 := errors.New("turn2 model failure after tool error")
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(_ agent.Context, _ struct{}) (map[string]any, error) {
			return nil, errTool
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		var modelCalls atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if modelCalls.Add(1) == 1 {
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call_fc_tool_err",
									Name: "lookup",
									Args: map[string]any{},
								},
							}},
						},
						TurnComplete: true,
					}, nil)
					return
				}
				yield(nil, errModel2)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFC, bufferedFR *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							if !errors.Is(runErr, errModel2) {
								t.Fatalf("unexpected bot.Run error: %v", runErr)
							}
							if bufferedFC != nil {
								yield(bufferedFC, nil)
							}
							if bufferedFR != nil {
								yield(bufferedFR, nil)
							}
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 {
							if ev.Content.Parts[0].FunctionCall != nil {
								bufferedFC = ev
								continue
							}
							if ev.Content.Parts[0].FunctionResponse != nil {
								bufferedFR = ev
								continue
							}
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-single-fc-tool-err", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventToolError)); n != 1 {
			t.Fatalf("got %d TOOL_ERROR rows, want 1; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("single LlmAgent BeforeToolCallbacks error with Turn-2 model error preserves TOOL_ERROR and error when buffered FunctionCall flushes", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errBeforeTool := errors.New("before_tool callback failure")
		errModel2 := errors.New("turn2 model failure after before_tool error")
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(_ agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{"res": "should-not-run"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		var modelCalls atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			BeforeToolCallbacks: []llmagent.BeforeToolCallback{
				func(_ agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
					return nil, errBeforeTool
				},
			},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if modelCalls.Add(1) == 1 {
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call_fc_cb_err",
									Name: "lookup",
									Args: map[string]any{},
								},
							}},
						},
						TurnComplete: true,
					}, nil)
					return
				}
				yield(nil, errModel2)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFC, bufferedFR *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							if !errors.Is(runErr, errModel2) {
								t.Fatalf("unexpected bot.Run error: %v", runErr)
							}
							if bufferedFC != nil {
								yield(bufferedFC, nil)
							}
							if bufferedFR != nil {
								yield(bufferedFR, nil)
							}
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 {
							if ev.Content.Parts[0].FunctionCall != nil {
								bufferedFC = ev
								continue
							}
							if ev.Content.Parts[0].FunctionResponse != nil {
								bufferedFR = ev
								continue
							}
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-single-fc-cb-err", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventToolError)); n != 1 {
			t.Fatalf("got %d TOOL_ERROR rows, want 1; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("single multi-turn LlmAgent buffered FunctionResponse preserves Turn-2 model error", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errModel2 := errors.New("single multi-turn turn2 model failure")
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(_ agent.Context, _ struct{}) (map[string]any, error) {
			return map[string]any{"res": "tool-ok"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		var modelCalls atomic.Int32
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if modelCalls.Add(1) == 1 {
					yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{{
								FunctionCall: &genai.FunctionCall{
									ID:   "call_single_mt_fr",
									Name: "lookup",
									Args: map[string]any{},
								},
							}},
						},
						TurnComplete: true,
					}, nil)
					return
				}
				yield(nil, errModel2)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFR *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							if bufferedFR != nil {
								yield(bufferedFR, nil)
							}
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionResponse != nil {
							bufferedFR = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-single-mt-fr-err", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errModel2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errModel2.Error())
		}
	})

	t.Run("single LlmAgent buffered plain text flushed during frozen-clock stream error preserves error", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errStream := errors.New("stream_err_after_turncomplete_text")
		bot, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if !yield(&model.LLMResponse{
					Content:      textContent("model", "buffered-plain-text"),
					TurnComplete: true,
				}, nil) {
					return
				}
				yield(nil, errStream)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedText *session.Event
					for ev, runErr := range bot.Run(ctx) {
						if runErr != nil {
							if bufferedText != nil {
								yield(bufferedText, nil)
							}
							return
						}
						if ev != nil && ev.Content != nil {
							bufferedText = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		fixedTime := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixedTime })
		for _, err := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-frozen-clock-plain-text", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errStream.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want %q", msg, errStream.Error())
		}
	})

	t.Run("buffered plugin AfterAgentCallback tail from first LlmAgent preserves silent second run failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		tailPlugin, err := baseplugin.New(baseplugin.Config{
			Name: "tail_after_agent",
			AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
				if ctx.AgentName() == "bot" {
					return textContent("model", "bot1-plugin-tail"), nil
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(tail_after_agent): %v", err)
		}
		var calls atomic.Int32
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				if calls.Add(1) == 1 {
					yield(&model.LLMResponse{Content: textContent("model", "bot1-ok"), TurnComplete: true}, nil)
					return
				}
				t.Error("bot1 Model called more than once")
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errSilent2 := errors.New("bot2 silent failure")
		bot2, err := agent.New(agent.Config{
			Name: "bot",
			Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, errSilent2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedTail *session.Event
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "bot1-plugin-tail" {
							bufferedTail = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							if bufferedTail != nil {
								yield(bufferedTail, nil)
							}
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p, tailPlugin).Run(t.Context(), "u1", "s-plugin-tail-silent2", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})

	t.Run("buffered agent AfterAgentCallbacks event from first LlmAgent preserves silent second run failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{Content: textContent("model", "bot1-model"), TurnComplete: true}, nil)
			}},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					return textContent("model", "bot1-agent-tail"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errSilent2 := errors.New("bot2 silent failure")
		bot2, err := agent.New(agent.Config{
			Name: "bot",
			Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, errSilent2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedTail *session.Event
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "bot1-agent-tail" {
							bufferedTail = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							if bufferedTail != nil {
								yield(bufferedTail, nil)
							}
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-agent-tail-silent2", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})

	t.Run("SkipSummarization tool buffered FunctionCall flushed during silent second run failure preserves error", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		lookupTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Look up item",
		}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			ctx.Actions().SkipSummarization = true
			return map[string]any{"res": "done"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		bot1, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Tools: []tool.Tool{lookupTool},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   "call_skip_sum",
								Name: "lookup",
								Args: map[string]any{},
							},
						}},
					},
					TurnComplete: true,
				}, nil)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errSilent2 := errors.New("bot2 silent failure")
		bot2, err := agent.New(agent.Config{
			Name: "bot",
			Run: func(_ agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, errSilent2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedFC *session.Event
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].FunctionCall != nil {
							bufferedFC = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							if bufferedFC != nil {
								yield(bufferedFC, nil)
							}
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-skipsum-fc-silent2", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})

	t.Run("responded LlmAgent with buffered AfterAgentCallback tail preserves silent second run failure", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		bot1, err := llmagent.New(llmagent.Config{
			Name: "bot",
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					return textContent("model", "bot1-extra-tail"), nil
				},
			},
			Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
				yield(&model.LLMResponse{
					Content:      textContent("model", "bot1-primary"),
					TurnComplete: true,
				}, nil)
			}},
		})
		if err != nil {
			t.Fatalf("llmagent.New(bot1): %v", err)
		}
		errSilent := errors.New("silent bot2 failure")
		bot2, err := agent.New(agent.Config{
			Name: "bot",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					yield(nil, errSilent)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(bot2): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{bot1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedTail *session.Event
					for ev, runErr := range bot1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "bot1-extra-tail" {
							bufferedTail = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range bot2.Run(ctx) {
						if runErr != nil {
							if bufferedTail != nil {
								yield(bufferedTail, nil)
							}
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-responded-tail-silent2", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
}

// streamBufferedConcurrentRun runs second in a goroutine while forwarding any
// events synchronously with per-event backpressure until readyToFlush signals,
// yields flushEv while second is open, invokes onFlush, and then drains the
// remaining output from second.
func streamBufferedConcurrentRun(
	ctx agent.InvocationContext,
	second agent.Agent,
	readyToFlush <-chan struct{},
	flushEv *session.Event,
	onFlush func(),
	swallowSecondErr bool,
	yield func(*session.Event, error) bool,
) {
	type botOut struct {
		ev  *session.Event
		err error
		ack chan bool
	}
	ch := make(chan botOut)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		defer close(ch)
		for ev, runErr := range second.Run(ctx) {
			var ack chan bool
			if ev != nil {
				ack = make(chan bool, 1)
			}
			select {
			case ch <- botOut{ev: ev, err: runErr, ack: ack}:
			case <-done:
				return
			}
			if ack != nil {
				select {
				case cont := <-ack:
					if !cont {
						return
					}
				case <-done:
					return
				}
			}
		}
	})
	defer func() {
		onFlush()
		close(done)
		wg.Wait()
	}()
	handleOut := func(out botOut) bool {
		if out.err != nil {
			if out.ack != nil {
				out.ack <- false
			}
			if swallowSecondErr {
				return true
			}
			yield(nil, out.err)
			return false
		}
		if out.ev != nil {
			cont := yield(out.ev, nil)
			if out.ack != nil {
				out.ack <- cont
			}
			return cont
		}
		if out.ack != nil {
			out.ack <- true
		}
		return true
	}
	for ready := false; !ready; {
		select {
		case out, ok := <-ch:
			if !ok {
				return
			}
			if !handleOut(out) {
				return
			}
		case <-readyToFlush:
			ready = true
		case <-ctx.Done():
			ready = true
		}
	}
	if flushEv != nil && !yield(flushEv, nil) {
		return
	}
	onFlush()
	for out := range ch {
		if !handleOut(out) {
			return
		}
	}
}

func TestBufferedPreFailureContent_Runner(t *testing.T) {
	runCase := func(t *testing.T, withOuter bool) {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("child branch failure after progress")

		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "inner_par", child)

		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					progress := session.NewEvent(ctx, ctx.InvocationID())
					progress.Author = "inner"
					progress.Branch = ctx.Branch()
					progress.Content = textContent("model", "starting child")
					if !yield(progress, nil) {
						return
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}

		target := inner
		if withOuter {
			outer, err := agent.New(agent.Config{
				Name:      "outer",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(outer): %v", err)
			}
			target = outer
		}

		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{target},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered []*session.Event
					for ev, runErr := range target.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Author == "inner" {
							buffered = append(buffered, ev)
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for _, ev := range buffered {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-pre-failure-progress", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errChild.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, errChild.Error())
		}
	}

	t.Run("nested outer wrapper", func(t *testing.T) {
		runCase(t, true)
	})
	t.Run("direct child of root", func(t *testing.T) {
		runCase(t, false)
	})
}

func TestLaterLiveResponseDoesNotClearRetiredFailure_Runner(t *testing.T) {
	p, requests, _ := setupTestPlugin(t)
	err1 := errors.New("run1 unrecovered parallel failure")

	child1, err := agent.New(agent.Config{
		Name: "child1",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child1"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child1 partial")
				if !yield(ev, nil) {
					return
				}
				yield(nil, err1)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child1): %v", err)
	}
	child2, err := agent.New(agent.Config{
		Name: "child2",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child2"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child2 ok")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child2): %v", err)
	}
	par1 := newParallelAgent(t, "inner_par_1", child1)
	par2 := newParallelAgent(t, "inner_par_2", child2)

	var runIdx int
	inner, err := agent.New(agent.Config{
		Name:      "inner",
		SubAgents: []agent.Agent{par1, par2},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			runIdx++
			cur := runIdx
			return func(yield func(*session.Event, error) bool) {
				par := par1
				if cur == 2 {
					par = par2
				}
				for ev, runErr := range par.Run(ctx) {
					if runErr != nil {
						continue
					}
					if !yield(ev, nil) {
						return
					}
				}
				if cur == 2 {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "run2 normal output")
					yield(ev, nil)
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(inner): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{inner},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for range 2 {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	for _, err := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-later-live-response", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error = %v, want nil", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
	}
	invErr := onlyRowOfType(t, rows, EventInvocationError)
	msg, _ := invErr["error_message"].(string)
	if !strings.Contains(msg, err1.Error()) {
		t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, err1.Error())
	}
}

func runZeroTimestampBufferedCleanRun(t *testing.T, runCtx context.Context, emitImmediateProgress bool, bufferedID string, initActions bool) {
	t.Helper()
	runZeroTimestampBufferedCleanRunWith(t, zeroTimestampBufferedCleanRun{
		runCtx:                runCtx,
		emitImmediateProgress: emitImmediateProgress,
		bufferedID:            bufferedID,
		initActions:           initActions,
	})
}

// zeroTimestampBufferedCleanRun configures runZeroTimestampBufferedCleanRunWith:
// a custom root runs a custom inner twice and holds inner's zero-timestamp
// event from the clean first run until after the unrecovered second run.
type zeroTimestampBufferedCleanRun struct {
	runCtx context.Context
	// emitImmediateProgress makes the first run respond with a
	// session.NewEvent event before the held one.
	emitImmediateProgress bool
	bufferedID            string
	initActions           bool
	// innerAfterAgentCallback gives inner a no-op AfterAgentCallback, and
	// laterAfterAgentPlugin registers a plugin with a no-op
	// AfterAgentCallback after this one: either lets ADK create events of
	// inner after its activation retired.
	innerAfterAgentCallback bool
	laterAfterAgentPlugin   bool
	// tick, when set, is advanced by root before it delivers the held event,
	// so that on a coarse clock that runCtx derives from it everything else
	// shares one tick and the delivery falls in the next one.
	tick *atomic.Int64
}

func runZeroTimestampBufferedCleanRunWith(t *testing.T, cfg zeroTimestampBufferedCleanRun) {
	t.Helper()
	runCtx, emitImmediateProgress, bufferedID, initActions := cfg.runCtx, cfg.emitImmediateProgress, cfg.bufferedID, cfg.initActions
	p, requests, _ := setupTestPlugin(t)
	err2 := errors.New("run2 unrecovered after zero-timestamp clean run1")

	child1, err := agent.New(agent.Config{
		Name: "child1",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child1"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child1 ok")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child1): %v", err)
	}
	child2, err := agent.New(agent.Config{
		Name: "child2",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child2"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child2 partial")
				if !yield(ev, nil) {
					return
				}
				yield(nil, err2)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child2): %v", err)
	}
	par1 := newParallelAgent(t, "inner_par_1", child1)
	par2 := newParallelAgent(t, "inner_par_2", child2)

	var runIdx int
	innerCfg := agent.Config{
		Name:      "inner",
		SubAgents: []agent.Agent{par1, par2},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			runIdx++
			cur := runIdx
			return func(yield func(*session.Event, error) bool) {
				par := par1
				if cur == 2 {
					par = par2
				}
				for ev, runErr := range par.Run(ctx) {
					if runErr != nil {
						continue
					}
					if !yield(ev, nil) {
						return
					}
				}
				if cur == 1 {
					if emitImmediateProgress {
						prog := session.NewEvent(ctx, ctx.InvocationID())
						prog.Author = "inner"
						prog.Branch = ctx.Branch()
						prog.Content = textContent("model", "run1 immediate progress")
						if !yield(prog, nil) {
							return
						}
					}
					var actions session.EventActions
					if initActions {
						actions = session.EventActions{
							StateDelta:    map[string]any{},
							ArtifactDelta: map[string]int64{},
						}
					}
					ev := &session.Event{
						ID:           bufferedID,
						InvocationID: ctx.InvocationID(),
						Author:       "inner",
						Branch:       ctx.Branch(),
						Actions:      actions,
						LLMResponse:  model.LLMResponse{Content: textContent("model", "run1 clean summary")},
					}
					yield(ev, nil)
				}
			}
		},
	}
	if cfg.innerAfterAgentCallback {
		innerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{func(agent.Context) (*genai.Content, error) { return nil, nil }}
	}
	inner, err := agent.New(innerCfg)
	if err != nil {
		t.Fatalf("agent.New(inner): %v", err)
	}

	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{inner},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var buffered []*session.Event
				for range 2 {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Author == "inner" && ev.Timestamp.IsZero() {
							buffered = append(buffered, ev)
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
				if cfg.tick != nil {
					cfg.tick.Add(1)
				}
				for _, ev := range buffered {
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}

	plugins := []*baseplugin.Plugin{p}
	if cfg.laterAfterAgentPlugin {
		later, err := baseplugin.New(baseplugin.Config{
			Name:               "later_after_agent",
			AfterAgentCallback: func(agent.Context) (*genai.Content, error) { return nil, nil },
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}
		plugins = append(plugins, later)
	}
	for _, err := range newTestRunner(t, root, plugins...).Run(runCtx, "u1", "s-zero-ts-clean", textContent("user", "go"), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run error = %v, want nil", err)
		}
	}
	rows := collectRows(t, requests, untilInvocationEnd)
	if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
		t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
	}
	invErr := onlyRowOfType(t, rows, EventInvocationError)
	msg, _ := invErr["error_message"].(string)
	if !strings.Contains(msg, err2.Error()) {
		t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, err2.Error())
	}
}

func TestZeroTimestampBufferedCleanRun_Runner(t *testing.T) {
	t.Run("single zero-timestamp buffered event", func(t *testing.T) {
		runZeroTimestampBufferedCleanRun(t, t.Context(), false, "clean-run-1", false)
	})
	t.Run("single zero-timestamp buffered event with initialized Actions maps", func(t *testing.T) {
		runZeroTimestampBufferedCleanRun(t, t.Context(), false, "clean-run-1-actions", true)
	})
	t.Run("responded first run with additional buffered zero-timestamp UUIDv4 event", func(t *testing.T) {
		runZeroTimestampBufferedCleanRun(t, t.Context(), true, uuid.NewString(), false)
	})
	// An after-agent callback, of inner or of a later plugin, lets ADK create
	// events of inner after its activation retired. The held event is
	// stamped when the runner delivers it, after both activations retired,
	// so it must not clear the unrecovered failure of the second run either:
	// it carries no StateDelta of an AfterAgentCallback.
	for _, tc := range []struct {
		name                  string
		responded             bool
		innerCallback, plugin bool
	}{
		{name: "unresponded first run with buffered zero-timestamp UUIDv4 event"},
		{name: "unresponded first run with buffered zero-timestamp UUIDv4 event and inner AfterAgentCallback", innerCallback: true},
		{name: "unresponded first run with buffered zero-timestamp UUIDv4 event and later AfterAgentCallback plugin", plugin: true},
		{name: "responded first run with buffered zero-timestamp UUIDv4 event and inner AfterAgentCallback", responded: true, innerCallback: true},
		{name: "responded first run with buffered zero-timestamp UUIDv4 event and later AfterAgentCallback plugin", responded: true, plugin: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runZeroTimestampBufferedCleanRunWith(t, zeroTimestampBufferedCleanRun{
				runCtx:                  t.Context(),
				emitImmediateProgress:   tc.responded,
				bufferedID:              uuid.NewString(),
				innerAfterAgentCallback: tc.innerCallback,
				laterAfterAgentPlugin:   tc.plugin,
			})
		})
	}
	// On a coarse clock root can deliver the held event in a tick after the
	// unrecovered second run retired: only an AfterAgentCallback event of a
	// later tick counts as the second run's.
	for _, tc := range []struct {
		name                  string
		innerCallback, plugin bool
	}{
		{name: "coarse clock delivery in a tick after the unrecovered run retired with inner AfterAgentCallback", innerCallback: true},
		{name: "coarse clock delivery in a tick after the unrecovered run retired with later AfterAgentCallback plugin", plugin: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var tick atomic.Int64
			base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
			runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
				return base.Add(time.Duration(tick.Load()) * time.Millisecond)
			})
			runZeroTimestampBufferedCleanRunWith(t, zeroTimestampBufferedCleanRun{
				runCtx:                  runCtx,
				bufferedID:              uuid.NewString(),
				innerAfterAgentCallback: tc.innerCallback,
				laterAfterAgentPlugin:   tc.plugin,
				tick:                    &tick,
			})
		})
	}
	t.Run("pre-failure zero-timestamp struct literal with initialized Actions maps held across parallel failure fails closed", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var tick atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return base.Add(time.Duration(tick.Add(1)) * time.Millisecond)
		})

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					held := &session.Event{
						ID:           "pre-failure-actions-literal",
						InvocationID: ctx.InvocationID(),
						Author:       "inner",
						Branch:       ctx.Branch(),
						Actions: session.EventActions{
							StateDelta:    map[string]any{},
							ArtifactDelta: map[string]int64{},
						},
						LLMResponse: model.LLMResponse{Content: textContent("model", "stale pre-failure literal")},
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					yield(held, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-held-zero-ts-actions", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, errChild.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, errChild.Error())
		}
	})
}

func TestLaterPluginAfterAgentRecovery_Runner(t *testing.T) {
	newSteppedClockCtx := func(parent context.Context) (context.Context, func(time.Duration)) {
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())
		ctx := platform.WithTimeProvider(parent, func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		step := func(d time.Duration) {
			nowUnixNano.Add(int64(d))
		}
		return ctx, step
	}

	runRecoveryCase := func(t *testing.T, laterPlugin bool) {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		runCtx, stepClock := newSteppedClockCtx(t.Context())
		errChild := errors.New("parallel branch failure")

		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					if !yield(ev, nil) {
						return
					}
					stepClock(time.Millisecond)
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)

		rootCfg := agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					stepClock(time.Millisecond)
				}
			},
		}
		plugins := []*baseplugin.Plugin{p}
		if laterPlugin {
			tailPlugin, err := baseplugin.New(baseplugin.Config{
				Name: "tail_recovery_plugin",
				AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
					if ctx.AgentName() == "root" {
						stepClock(10 * time.Millisecond)
						return textContent("model", "recovered-by-later-plugin"), nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(tail_recovery_plugin): %v", err)
			}
			plugins = append(plugins, tailPlugin)
		} else {
			rootCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					stepClock(10 * time.Millisecond)
					return textContent("model", "recovered-by-agent-callback"), nil
				},
			}
		}

		root, err := agent.New(rootCfg)
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, err := range newTestRunner(t, root, plugins...).Run(runCtx, "u1", "s-later-plugin-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	}

	t.Run("laterPlugin_false", func(t *testing.T) {
		runRecoveryCase(t, false)
	})
	t.Run("laterPlugin_true", func(t *testing.T) {
		runRecoveryCase(t, true)
	})

	t.Run("subrunner shadowing control", func(t *testing.T) {
		outerP, outerRequests, _ := setupTestPlugin(t)
		innerP, innerRequests, _ := setupTestPlugin(t)
		runCtx, stepClock := newSteppedClockCtx(t.Context())

		errInner := errors.New("inner parallel branch failure")
		innerFail, err := agent.New(agent.Config{
			Name: "inner_fail",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner_fail"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "inner_fail answer")
					if !yield(ev, nil) {
						return
					}
					stepClock(time.Millisecond)
					yield(nil, errInner)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner_fail): %v", err)
		}
		innerPar := newParallelAgent(t, "inner_par", innerFail)
		innerBot, err := agent.New(agent.Config{
			Name:      "inner_bot",
			SubAgents: []agent.Agent{innerPar},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range innerPar.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					stepClock(time.Millisecond)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner_bot): %v", err)
		}
		innerRoot, err := agent.New(agent.Config{
			Name:      "inner_root",
			SubAgents: []agent.Agent{innerBot},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range innerBot.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					// Emit a post-retirement inner_bot-authored event; because innerRunner
					// has no AfterAgentCallback on inner_bot or in its own PluginManager,
					// outerTailPlugin must not be inherited to mark inner_bot recoverable.
					stepClock(10 * time.Millisecond)
					lateEv := session.NewEvent(ctx, ctx.InvocationID())
					lateEv.Author = "inner_bot"
					lateEv.Branch = ctx.Branch()
					lateEv.Content = textContent("model", "late-inner-bot-event")
					yield(lateEv, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner_root): %v", err)
		}
		innerRunner := newTestRunner(t, innerRoot, innerP)

		outerTailPlugin, err := baseplugin.New(baseplugin.Config{
			Name: "outer_tail_after_agent",
			AfterAgentCallback: func(_ agent.Context) (*genai.Content, error) {
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(outer_tail_after_agent): %v", err)
		}
		outerRoot, err := agent.New(agent.Config{
			Name: "outer_root",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, runErr := range innerRunner.Run(ctx, "u1", "s-shadow-inner", textContent("user", "go"), agent.RunConfig{}) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "outer_root"
					ev.Content = textContent("model", "outer-complete")
					yield(ev, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(outer_root): %v", err)
		}

		for _, err := range newTestRunner(t, outerRoot, outerP, outerTailPlugin).Run(runCtx, "u1", "s-shadow-outer", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("outer runner.Run error = %v, want nil", err)
			}
		}

		innerRows := collectRows(t, innerRequests, untilInvocationEnd)
		if n := len(rowsOfType(innerRows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("inner got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(innerRows))
		}
		invErr := onlyRowOfType(t, innerRows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errInner.Error()) {
			t.Errorf("inner INVOCATION_ERROR error_message = %q, want substring %q", msg, errInner.Error())
		}

		outerRows := collectRows(t, outerRequests, untilInvocationEnd)
		_ = onlyRowOfType(t, outerRows, EventInvocationCompleted)
	})
}

// TestAfterAgentRecoveryWithCopiedActionMaps_Runner runs root -> inner ->
// parallel -> child, where the child emits content and fails and inner drains
// the error and recovers with content from its AfterAgentCallback, while a
// plugin registered before this one replaces action maps of every event in its
// OnEventCallback with copies. ADK builds the recovery event after this
// plugin retired inner, so only an action map of inner's after-agent callback
// context shows that the event is inner's: a copied StateDelta or ArtifactDelta
// alone keeps the recovery, with immediate or buffered delivery, with or
// without earlier progress from inner. With both maps copied the event carries
// no trace of the callback, and the failure stays reported (see Config).
func TestAfterAgentRecoveryWithCopiedActionMaps_Runner(t *testing.T) {
	run := func(t *testing.T, buffered, innerProgress bool, copyMaps func(*session.EventActions)) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if innerProgress {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner progress")
						if !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return textContent("model", "inner callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if buffered {
							held = append(held, ev)
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		copier, err := baseplugin.New(baseplugin.Config{
			Name: "action_map_copier",
			OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
				if ev != nil {
					copyMaps(&ev.Actions)
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New(action_map_copier): %v", err)
		}
		recoveries := 0
		for ev, runErr := range newTestRunner(t, root, copier, p).Run(t.Context(), "u1", "s-copied-action-maps", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
			if ev != nil && ev.Author == "inner" && ev.Content != nil && len(ev.Content.Parts) == 1 && ev.Content.Parts[0].Text == "inner callback recovery" {
				recoveries++
			}
		}
		if recoveries != 1 {
			t.Fatalf("caller got %d recovery responses, want 1", recoveries)
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	copyStateDelta := func(a *session.EventActions) { a.StateDelta = maps.Clone(a.StateDelta) }
	copyArtifactDelta := func(a *session.EventActions) { a.ArtifactDelta = maps.Clone(a.ArtifactDelta) }

	for _, delivery := range []string{"immediate", "buffered"} {
		for _, progress := range []bool{false, true} {
			for _, tc := range []struct {
				name string
				copy func(*session.EventActions)
			}{
				{"copied StateDelta", copyStateDelta},
				{"copied ArtifactDelta", copyArtifactDelta},
			} {
				t.Run(fmt.Sprintf("%s/%s delivery/inner progress %v", tc.name, delivery, progress), func(t *testing.T) {
					rows := run(t, delivery == "buffered", progress, tc.copy)
					if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
						t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
					}
					_ = onlyRowOfType(t, rows, EventInvocationCompleted)
				})
			}
		}
	}

	t.Run("both action maps copied fails closed", func(t *testing.T) {
		rows := run(t, false, false, func(a *session.EventActions) {
			copyStateDelta(a)
			copyArtifactDelta(a)
		})
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
}

func TestCustomUUIDProviderRecovery_Runner(t *testing.T) {
	t.Run("post-failure recovery succeeds with sequential custom UUIDs without extra provider calls", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var uuidCalls atomic.Int32
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())

		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		runCtx = platform.WithUUIDProvider(runCtx, func() string {
			return fmt.Sprintf("event-%d", uuidCalls.Add(1))
		})

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					if !yield(ev, nil) {
						return
					}
					nowUnixNano.Add(int64(time.Millisecond))
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		var runnerEventCount atomic.Int32
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					nowUnixNano.Add(int64(10 * time.Millisecond))
					recovery := session.NewEvent(ctx, ctx.InvocationID())
					recovery.Author = "root"
					recovery.Content = textContent("model", "recovered-with-custom-uuid")
					yield(recovery, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for ev, err := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-custom-uuid-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
			if ev != nil {
				runnerEventCount.Add(1)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)

		// Expected platform.NewUUID calls: 1 InvocationID + 1 user event + yielded agent events.
		wantCalls := 2 + runnerEventCount.Load()
		if got := uuidCalls.Load(); got != wantCalls {
			t.Fatalf("UUIDProvider call count = %d, want %d (InvocationID + user + %d yielded events)", got, wantCalls, runnerEventCount.Load())
		}
		if hasEventCreationProvenance(runCtx, &session.Event{ID: "event-1"}) != true || uuidCalls.Load() != wantCalls {
			t.Fatalf("hasEventCreationProvenance altered provider state: calls=%d want %d", uuidCalls.Load(), wantCalls)
		}
	})

	t.Run("retired agent AfterAgentCallback recovery succeeds with custom UUIDs", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var uuidCalls atomic.Int32
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())

		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		runCtx = platform.WithUUIDProvider(runCtx, func() string {
			return fmt.Sprintf("event-%d", uuidCalls.Add(1))
		})

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					if !yield(ev, nil) {
						return
					}
					nowUnixNano.Add(int64(time.Millisecond))
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					nowUnixNano.Add(int64(time.Millisecond))
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					nowUnixNano.Add(int64(10 * time.Millisecond))
					return textContent("model", "inner-callback-recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		var runnerEventCount atomic.Int32
		for ev, err := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-custom-uuid-retired-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
			if ev != nil {
				runnerEventCount.Add(1)
			}
		}

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)

		wantCalls := 2 + runnerEventCount.Load()
		if got := uuidCalls.Load(); got != wantCalls {
			t.Fatalf("UUIDProvider call count = %d, want %d", got, wantCalls)
		}
	})

	t.Run("WithUUIDProvider(childCtx, nil) overrides outer provider", func(t *testing.T) {
		var uuidCalls atomic.Int32
		parentCtx := platform.WithUUIDProvider(t.Context(), func() string {
			return fmt.Sprintf("event-%d", uuidCalls.Add(1))
		})
		childCtx := platform.WithUUIDProvider(parentCtx, nil)
		runZeroTimestampBufferedCleanRun(t, childCtx, false, "clean-run-1", false)
		if got := uuidCalls.Load(); got != 0 {
			t.Fatalf("outer UUIDProvider called %d times under child nil override, want 0", got)
		}
	})

	t.Run("zero-timestamp buffered empty ID rejected under WithUUIDProvider", func(t *testing.T) {
		var uuidCalls atomic.Int32
		customCtx := platform.WithUUIDProvider(t.Context(), func() string {
			return fmt.Sprintf("event-%d", uuidCalls.Add(1))
		})
		runZeroTimestampBufferedCleanRun(t, customCtx, false, "", false)
	})

	t.Run("responded first run with zero-timestamp buffered custom ID preserves error under WithUUIDProvider", func(t *testing.T) {
		var uuidCalls atomic.Int32
		customCtx := platform.WithUUIDProvider(t.Context(), func() string {
			return fmt.Sprintf("event-%d", uuidCalls.Add(1))
		})
		runZeroTimestampBufferedCleanRun(t, customCtx, true, "clean-run-1", false)
	})

	t.Run("invocation-scoped WithUUIDProvider recovers with stripped Actions", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var uuidCalls atomic.Int32
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())

		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		runCtx = platform.WithUUIDProvider(runCtx, func() string {
			return fmt.Sprintf("inv-ev-%d", uuidCalls.Add(1))
		})

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					if !yield(ev, nil) {
						return
					}
					nowUnixNano.Add(int64(time.Millisecond))
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		var runnerEventCount atomic.Int32
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					nowUnixNano.Add(int64(10 * time.Millisecond))
					recovery := session.NewEvent(ctx, ctx.InvocationID())
					recovery.Author = "root"
					recovery.Content = textContent("model", "recovered-stripped-actions")
					recovery.Actions = session.EventActions{}
					yield(recovery, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for ev, err := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-inv-uuid-stripped", textContent("user", "go"), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("runner.Run error = %v, want nil", err)
			}
			if ev != nil {
				runnerEventCount.Add(1)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		wantCalls := 2 + runnerEventCount.Load()
		if got := uuidCalls.Load(); got != wantCalls {
			t.Fatalf("UUIDProvider call count = %d, want %d", got, wantCalls)
		}
	})

	t.Run("child-scoped WithUUIDProvider recovers via AfterAgentCallback", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var uuidCalls atomic.Int32
		var innerEventCount atomic.Int32
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())

		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		provider := func() string {
			return fmt.Sprintf("child-ev-%d", uuidCalls.Add(1))
		}

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					innerEventCount.Add(1)
					if !yield(ev, nil) {
						return
					}
					nowUnixNano.Add(int64(time.Millisecond))
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(ctx agent.Context) (*genai.Content, error) {
					nowUnixNano.Add(int64(time.Millisecond))
					innerEventCount.Add(1)
					return textContent("model", "child-scoped callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					childCtx := ctx.WithContext(platform.WithUUIDProvider(ctx, provider))
					for ev, runErr := range inner.Run(childCtx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-child-uuid-cb", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		if got, want := uuidCalls.Load(), innerEventCount.Load(); got != want {
			t.Fatalf("child UUIDProvider call count = %d, want %d", got, want)
		}
	})

	t.Run("child-scoped WithUUIDProvider recovers via session.NewEvent", func(t *testing.T) {
		for _, stripActions := range []bool{false, true} {
			name := "preserving Actions"
			if stripActions {
				name = "with stripped Actions"
			}
			t.Run(name, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				var uuidCalls atomic.Int32
				var innerEventCount atomic.Int32
				var nowUnixNano atomic.Int64
				base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
				nowUnixNano.Store(base.UnixNano())

				runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
					return time.Unix(0, nowUnixNano.Load()).UTC()
				})
				provider := func() string {
					return fmt.Sprintf("child-ev-%d", uuidCalls.Add(1))
				}

				errChild := errors.New("parallel child boom")
				failBranch, err := agent.New(agent.Config{
					Name: "fail_branch",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "fail_branch"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "fail_branch answer")
							innerEventCount.Add(1)
							if !yield(ev, nil) {
								return
							}
							nowUnixNano.Add(int64(time.Millisecond))
							yield(nil, errChild)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(fail_branch): %v", err)
				}
				par := newParallelAgent(t, "par", failBranch)
				inner, err := agent.New(agent.Config{
					Name:      "inner",
					SubAgents: []agent.Agent{par},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range par.Run(ctx) {
								if runErr != nil {
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							nowUnixNano.Add(int64(time.Millisecond))
							recEv := session.NewEvent(ctx, ctx.InvocationID())
							recEv.Author = "inner"
							recEv.Branch = ctx.Branch()
							recEv.Content = textContent("model", "child-scoped live recovery")
							if stripActions {
								recEv.Actions = session.EventActions{}
							}
							innerEventCount.Add(1)
							yield(recEv, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(inner): %v", err)
				}
				root, err := agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{inner},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							childCtx := ctx.WithContext(platform.WithUUIDProvider(ctx, provider))
							for ev, runErr := range inner.Run(childCtx) {
								if !yield(ev, runErr) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}

				for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-child-uuid-live", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error = %v, want nil", runErr)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationCompleted)
				if got, want := uuidCalls.Load(), innerEventCount.Load(); got != want {
					t.Fatalf("child UUIDProvider call count = %d, want %d", got, want)
				}
			})
		}
	})

	t.Run("call-site-only WithUUIDProvider without agent or invocation attachment fails closed", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var uuidCalls atomic.Int32
		var nowUnixNano atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		nowUnixNano.Store(base.UnixNano())

		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Unix(0, nowUnixNano.Load()).UTC()
		})
		eventUUIDProvider := func() string {
			return fmt.Sprintf("custom-ev-%d", uuidCalls.Add(1))
		}

		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch answer")
					if !yield(ev, nil) {
						return
					}
					nowUnixNano.Add(int64(time.Millisecond))
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					nowUnixNano.Add(int64(time.Millisecond))
					recEv := session.NewEvent(platform.WithUUIDProvider(ctx, eventUUIDProvider), ctx.InvocationID())
					if recEv.ID != "custom-ev-1" {
						t.Errorf("recovery event ID = %q, want custom-ev-1", recEv.ID)
					}
					recEv.Author = "inner"
					recEv.Branch = ctx.Branch()
					recEv.Content = textContent("model", "event-scoped recovery")
					yield(recEv, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-event-uuid-scoped", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, errChild.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, errChild.Error())
		}
		if got := uuidCalls.Load(); got != 1 {
			t.Fatalf("event-scoped UUIDProvider called %d times, want 1", got)
		}
	})
}

func TestFixedTimeProviderRecovery_Runner(t *testing.T) {
	fixedTime := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	fixedCtx := func(t *testing.T) context.Context {
		t.Helper()
		return platform.WithTimeProvider(t.Context(), func() time.Time {
			return fixedTime
		})
	}

	advancingCtx := func(t *testing.T) context.Context {
		t.Helper()
		var tick atomic.Int64
		return platform.WithTimeProvider(t.Context(), func() time.Time {
			return fixedTime.Add(time.Duration(tick.Add(1)) * time.Millisecond)
		})
	}

	for _, tc := range []struct {
		name      string
		author    string
		ctxFn     func(*testing.T) context.Context
		wantError bool
	}{
		{"live root same-tick non-callback fails closed under fixed clock", "root", fixedCtx, true},
		{"live root non-callback recovery succeeds with advancing clock", "root", advancingCtx, false},
		{"live inner same-tick non-callback fails closed under fixed clock", "inner", fixedCtx, true},
		{"live inner non-callback recovery succeeds with advancing clock", "inner", advancingCtx, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			errChild := errors.New("parallel child boom")
			failBranch, err := agent.New(agent.Config{
				Name: "fail_branch",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "fail_branch"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "fail_branch partial")
						if !yield(ev, nil) {
							return
						}
						yield(nil, errChild)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(fail_branch): %v", err)
			}
			par := newParallelAgent(t, "par", failBranch)
			inner, err := agent.New(agent.Config{
				Name:      "inner",
				SubAgents: []agent.Agent{par},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						if tc.author == "inner" {
							recEv := session.NewEvent(ctx, ctx.InvocationID())
							recEv.Author = "inner"
							recEv.Branch = ctx.Branch()
							recEv.Content = textContent("model", "inner recovery")
							yield(recEv, nil)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(inner): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
						if tc.author == "root" {
							recEv := session.NewEvent(ctx, ctx.InvocationID())
							recEv.Author = "root"
							recEv.Branch = ctx.Branch()
							recEv.Content = textContent("model", "root recovery")
							yield(recEv, nil)
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}

			for _, runErr := range newTestRunner(t, root, p).Run(tc.ctxFn(t), "u1", "s-fixed-live", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error = %v, want nil", runErr)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if tc.wantError {
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationError)
			} else {
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationCompleted)
			}
		})
	}

	t.Run("held pre-failure live same-tick progress fails closed", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			ctxFn       func(*testing.T) context.Context
			createAfter bool
			wantError   bool
		}{
			{"stale_progress/advancing_clock", advancingCtx, false, true},
			{"stale_progress/fixed_clock", fixedCtx, false, true},
			{"genuine_recovery/advancing_clock", advancingCtx, true, false},
			{"genuine_recovery/fixed_clock", fixedCtx, true, true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				errChild := errors.New("parallel child boom")
				failBranch, err := agent.New(agent.Config{
					Name: "fail_branch",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "fail_branch"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "fail_branch partial")
							if !yield(ev, nil) {
								return
							}
							yield(nil, errChild)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(fail_branch): %v", err)
				}
				par := newParallelAgent(t, "par", failBranch)
				inner, err := agent.New(agent.Config{
					Name:      "inner",
					SubAgents: []agent.Agent{par},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							var held *session.Event
							if !tc.createAfter {
								held = session.NewEvent(ctx, ctx.InvocationID())
								held.Author = "inner"
								held.Branch = ctx.Branch()
								held.Content = textContent("model", "stale pre-failure progress held by inner")
							}
							for ev, runErr := range par.Run(ctx) {
								if runErr != nil {
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							if tc.createAfter {
								held = session.NewEvent(ctx, ctx.InvocationID())
								held.Author = "inner"
								held.Branch = ctx.Branch()
								held.Content = textContent("model", "genuine post-failure recovery")
							}
							yield(held, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(inner): %v", err)
				}
				root, err := agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{inner},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range inner.Run(ctx) {
								if !yield(ev, runErr) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}
				for _, runErr := range newTestRunner(t, root, p).Run(tc.ctxFn(t), "u1", "s-live-held-stale", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error = %v, want nil", runErr)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if tc.wantError {
					if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
						t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
					}
					_ = onlyRowOfType(t, rows, EventInvocationError)
				} else {
					if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
						t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
					}
					_ = onlyRowOfType(t, rows, EventInvocationCompleted)
				}
			})
		}
	})

	t.Run("same-tick AfterAgentCallback clears unrespondedAtPop so subsequent activation can disambiguate buffered recovery", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var phase atomic.Int32
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return fixedTime.Add(time.Duration(phase.Load()) * 10 * time.Millisecond)
		})
		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		var iterCount atomic.Int32
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				cur := iterCount.Add(1)
				return func(yield func(*session.Event, error) bool) {
					if cur == 1 {
						return
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					phase.Store(2)
					recEv := session.NewEvent(ctx, ctx.InvocationID())
					recEv.Author = "inner"
					recEv.Branch = ctx.Branch()
					recEv.Content = textContent("model", "iter2 buffered recovery")
					yield(recEv, nil)
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					if iterCount.Load() == 1 {
						return textContent("model", "iter1 callback response"), nil
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
					phase.Store(1)
					var buffered []*session.Event
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil || ev == nil {
							continue
						}
						if ev.Author == "inner" {
							buffered = append(buffered, ev)
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					for _, ev := range buffered {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-unresp-same-tick", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	for _, bufferedByRoot := range []bool{false, true} {
		name := "retired inner recovers via agent AfterAgentCallback (immediate)"
		if bufferedByRoot {
			name = "retired inner recovers via agent AfterAgentCallback (buffered root delivery)"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			errChild := errors.New("parallel child boom")
			failBranch, err := agent.New(agent.Config{
				Name: "fail_branch",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "fail_branch"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "fail_branch partial")
						if !yield(ev, nil) {
							return
						}
						yield(nil, errChild)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(fail_branch): %v", err)
			}
			par := newParallelAgent(t, "par", failBranch)
			inner, err := agent.New(agent.Config{
				Name:      "inner",
				SubAgents: []agent.Agent{par},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
				AfterAgentCallbacks: []agent.AfterAgentCallback{
					func(ctx agent.Context) (*genai.Content, error) {
						return textContent("model", "inner callback recovery"), nil
					},
				},
			})
			if err != nil {
				t.Fatalf("agent.New(inner): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						var buffered []*session.Event
						for ev, runErr := range inner.Run(ctx) {
							if bufferedByRoot {
								if ev != nil {
									buffered = append(buffered, ev)
								}
								continue
							}
							if !yield(ev, runErr) {
								return
							}
						}
						for _, ev := range buffered {
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}

			for _, runErr := range newTestRunner(t, root, p).Run(fixedCtx(t), "u1", "s-fixed-cb", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error = %v, want nil", runErr)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
	}

	t.Run("retired inner recovers via later plugin AfterAgentCallback", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		laterPlugin, err := baseplugin.New(baseplugin.Config{
			Name: "later-recovery",
			AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
				if ctx.AgentName() != "inner" {
					return nil, nil
				}
				return textContent("model", "later plugin recovery"), nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}
		errChild := errors.New("parallel child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p, laterPlugin).Run(fixedCtx(t), "u1", "s-fixed-later-plugin", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	for _, tc := range []struct {
		name             string
		withOuter        bool
		withNoOpCallback bool
	}{
		{name: "stale same-tick pre-failure progress (withOuter=false)", withOuter: false, withNoOpCallback: false},
		{name: "stale same-tick pre-failure progress (withOuter=true)", withOuter: true, withNoOpCallback: false},
		{name: "stale same-tick pre-failure progress with no-op AfterAgentCallback (withOuter=false)", withOuter: false, withNoOpCallback: true},
		{name: "stale same-tick pre-failure progress with no-op AfterAgentCallback (withOuter=true)", withOuter: true, withNoOpCallback: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			errChild := errors.New("parallel child boom")
			failBranch, err := agent.New(agent.Config{
				Name: "fail_branch",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "fail_branch"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "fail_branch partial")
						if !yield(ev, nil) {
							return
						}
						yield(nil, errChild)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(fail_branch): %v", err)
			}
			par := newParallelAgent(t, "par", failBranch)

			var afterCBs []agent.AfterAgentCallback
			if tc.withNoOpCallback {
				afterCBs = []agent.AfterAgentCallback{
					func(ctx agent.Context) (*genai.Content, error) {
						return nil, nil
					},
				}
			}
			inner, err := agent.New(agent.Config{
				Name:      "inner",
				SubAgents: []agent.Agent{par},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						prog := session.NewEvent(ctx, ctx.InvocationID())
						prog.Author = "inner"
						prog.Branch = ctx.Branch()
						prog.Content = textContent("model", "pre-failure progress")
						if !yield(prog, nil) {
							return
						}
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
				},
				AfterAgentCallbacks: afterCBs,
			})
			if err != nil {
				t.Fatalf("agent.New(inner): %v", err)
			}

			var root agent.Agent
			if tc.withOuter {
				outer, err := agent.New(agent.Config{
					Name:      "outer",
					SubAgents: []agent.Agent{inner},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							var buffered []*session.Event
							for ev, runErr := range inner.Run(ctx) {
								if runErr != nil || ev == nil {
									continue
								}
								if ev.Author == "inner" {
									buffered = append(buffered, ev)
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							for _, ev := range buffered {
								if !yield(ev, nil) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(outer): %v", err)
				}
				root, err = agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{outer},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range outer.Run(ctx) {
								if !yield(ev, runErr) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}
			} else {
				root, err = agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{inner},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							var buffered []*session.Event
							for ev, runErr := range inner.Run(ctx) {
								if runErr != nil || ev == nil {
									continue
								}
								if ev.Author == "inner" {
									buffered = append(buffered, ev)
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							for _, ev := range buffered {
								if !yield(ev, nil) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}
			}

			for _, runErr := range newTestRunner(t, root, p).Run(fixedCtx(t), "u1", "s-fixed-stale", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error = %v, want nil", runErr)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationError)
		})
	}

	t.Run("nested buffered AfterAgentCallback recovery across retired parent under fixed clock", func(t *testing.T) {
		for _, bufferMode := range []string{"inner", "all"} {
			t.Run("buffer_"+bufferMode, func(t *testing.T) {
				p, requests, _ := setupTestPlugin(t)
				errChild := errors.New("parallel child boom")
				failBranch, err := agent.New(agent.Config{
					Name: "fail_branch",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "fail_branch"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "fail_branch partial")
							if !yield(ev, nil) {
								return
							}
							yield(nil, errChild)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(fail_branch): %v", err)
				}
				par := newParallelAgent(t, "par", failBranch)
				inner, err := agent.New(agent.Config{
					Name:      "inner",
					SubAgents: []agent.Agent{par},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range par.Run(ctx) {
								if runErr != nil {
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
						}
					},
					AfterAgentCallbacks: []agent.AfterAgentCallback{
						func(agent.Context) (*genai.Content, error) {
							return textContent("model", "inner callback recovery"), nil
						},
					},
				})
				if err != nil {
					t.Fatalf("agent.New(inner): %v", err)
				}
				outer, err := agent.New(agent.Config{
					Name:      "outer",
					SubAgents: []agent.Agent{inner},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for ev, runErr := range inner.Run(ctx) {
								if !yield(ev, runErr) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(outer): %v", err)
				}
				top, err := agent.New(agent.Config{
					Name:      "top",
					SubAgents: []agent.Agent{outer},
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							var buffered []*session.Event
							for ev, runErr := range outer.Run(ctx) {
								if runErr != nil {
									yield(nil, runErr)
									return
								}
								hold := ev != nil && (bufferMode == "all" || ev.Author == "inner")
								if hold {
									buffered = append(buffered, ev)
									continue
								}
								if ev != nil && !yield(ev, nil) {
									return
								}
							}
							for _, ev := range buffered {
								if !yield(ev, nil) {
									return
								}
							}
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New(top): %v", err)
				}

				for _, runErr := range newTestRunner(t, top, p).Run(fixedCtx(t), "u1", "s-nested-buf-cb", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error = %v, want nil", runErr)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
					t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationCompleted)
			})
		}
	})

	t.Run("repeated sequential recovered iterations fail closed under constant clock", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")
		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := iterCount.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					return textContent("model", "inner-callback-recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(fixedCtx(t), "u1", "s-fixed-seq-cb", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
}

func TestStructLiteralRecoveryEvents_Runner(t *testing.T) {
	baseTS := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	cases := []struct {
		name            string
		useNewEvent     bool
		eventID         string
		includeNonZeroT bool
		initActions     bool
		wantSuccess     bool
	}{
		{name: "bare struct literal non-UUID ID with timestamp fails closed", eventID: "manual-non-uuid", includeNonZeroT: true, wantSuccess: false},
		{name: "bare struct literal empty ID with timestamp fails closed", eventID: "", includeNonZeroT: true, wantSuccess: false},
		{name: "bare struct literal non-UUID ID with zero timestamp fails closed", eventID: "manual-zero-ts", includeNonZeroT: false, wantSuccess: false},
		{name: "bare struct literal non-UUID ID with zero timestamp and initialized Actions maps fails closed", eventID: "manual-zero-ts-actions", includeNonZeroT: false, initActions: true, wantSuccess: false},
		{name: "session.NewEvent recovery succeeds", useNewEvent: true, wantSuccess: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var tick atomic.Int64
			runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
				return baseTS.Add(time.Duration(tick.Add(1)) * time.Second)
			})

			errChild := errors.New("parallel child boom")
			failBranch, err := agent.New(agent.Config{
				Name: "fail_branch",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "fail_branch"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "fail_branch partial")
						if !yield(ev, nil) {
							return
						}
						yield(nil, errChild)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(fail_branch): %v", err)
			}
			par := newParallelAgent(t, "par", failBranch)
			inner, err := agent.New(agent.Config{
				Name:      "inner",
				SubAgents: []agent.Agent{par},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						if tc.useNewEvent {
							recEv := session.NewEvent(ctx, ctx.InvocationID())
							recEv.Author = "inner"
							recEv.Branch = ctx.Branch()
							recEv.Content = textContent("model", "session.NewEvent recovery")
							yield(recEv, nil)
							return
						}
						var ts time.Time
						if tc.includeNonZeroT {
							ts = platform.Now(ctx)
						}
						var actions session.EventActions
						if tc.initActions {
							actions = session.EventActions{
								StateDelta:    map[string]any{},
								ArtifactDelta: map[string]int64{},
							}
						}
						yield(&session.Event{
							ID:           tc.eventID,
							Timestamp:    ts,
							InvocationID: ctx.InvocationID(),
							Author:       "inner",
							Branch:       ctx.Branch(),
							Actions:      actions,
							LLMResponse:  model.LLMResponse{Content: textContent("model", "struct literal recovery")},
						}, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(inner): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}

			for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-struct-literal", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error = %v, want nil", runErr)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			if tc.wantSuccess {
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationCompleted)
			} else {
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationError)
			}
		})
	}
}

func TestAfterAgentActionMapsGCReuse_Runner(t *testing.T) {
	// probeActionMapCollision allocates events until one gets a StateDelta or
	// ArtifactDelta map at the address that target recorded for the same
	// field, as a freed map's memory would allow.
	probeActionMapCollision := func(ctx agent.InvocationContext, target *actionMaps) (*session.Event, bool) {
		if target == nil || *target == (actionMaps{}) {
			return nil, false
		}
		deadline := time.Now().Add(2 * time.Second)
		for round := 0; round < 20 && time.Now().Before(deadline); round++ {
			var keep []*session.Event
			runtime.GC()
			runtime.GC()
			for range 2000 {
				cand := session.NewEvent(ctx, ctx.InvocationID())
				if target.shares(eventActionMaps(cand)) {
					return cand, true
				}
				keep = append(keep, cand)
			}
			runtime.KeepAlive(keep)
		}
		return nil, false
	}

	t.Run("sequential activations buffer and force GC without reusing retired callback action maps", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		var tick atomic.Int64
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return base.Add(time.Duration(tick.Add(1)) * time.Millisecond)
		})
		var iter1Actions atomic.Pointer[actionMaps]
		var innerAfterCalls atomic.Int32
		probe, err := baseplugin.New(baseplugin.Config{
			Name: "probe-capture-after-agent-actions",
			AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
				if ctx.AgentName() == "inner" && innerAfterCalls.Add(1) == 1 {
					am := callbackAfterAgentActionMaps(ctx)
					iter1Actions.Store(&am)
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}
		errChild := errors.New("iteration 1 child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		var runIdx atomic.Int32
		var collided atomic.Bool
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				cur := runIdx.Add(1)
				return func(yield func(*session.Event, error) bool) {
					if cur == 1 {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						return
					}
					ev, hit := probeActionMapCollision(ctx, iter1Actions.Load())
					if hit {
						collided.Store(true)
					} else {
						ev = session.NewEvent(ctx, ctx.InvocationID())
					}
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "iteration 2 clean answer")
					yield(ev, nil)
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) { return nil, nil },
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var buffered []*session.Event
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil || ev == nil {
								continue
							}
							if ev.Author == "inner" {
								buffered = append(buffered, ev)
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
					for _, ev := range buffered {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p, probe).Run(runCtx, "u1", "s-gc-ptr-seq", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		if collided.Load() {
			t.Fatalf("retired callback action maps %+v were freed and reused by GC while retired activation remained guarded", *iter1Actions.Load())
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})

	t.Run("single activation under fixed clock pins callback action maps across forced GC", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		fixedTime := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time { return fixedTime })
		var iter1Actions atomic.Pointer[actionMaps]
		probe, err := baseplugin.New(baseplugin.Config{
			Name: "probe-capture-after-agent-actions",
			AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
				if ctx.AgentName() == "inner" {
					am := callbackAfterAgentActionMaps(ctx)
					iter1Actions.Store(&am)
				}
				return nil, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}
		errChild := errors.New("single child boom")
		failBranch, err := agent.New(agent.Config{
			Name: "fail_branch",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "fail_branch"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "fail_branch partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(fail_branch): %v", err)
		}
		par := newParallelAgent(t, "par", failBranch)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) { return nil, nil },
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		var collided atomic.Bool
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil || ev == nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					ev, hit := probeActionMapCollision(ctx, iter1Actions.Load())
					if hit {
						collided.Store(true)
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "forged colliding same-tick event")
						yield(ev, nil)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p, probe).Run(runCtx, "u1", "s-gc-ptr-single", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		if collided.Load() {
			t.Fatalf("retired callback action maps %+v were freed and reused by GC during single fixed-clock activation", *iter1Actions.Load())
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
}

func TestSequentialRecoveredParallelIterations_Runner(t *testing.T) {
	t.Run("AfterAgentCallback recovery across two sequential iterations", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := iterCount.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					return textContent("model", "inner-callback-recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-cb", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	t.Run("live session.NewEvent recovery across two sequential iterations", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := iterCount.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					time.Sleep(time.Millisecond)
					recEv := session.NewEvent(ctx, ctx.InvocationID())
					recEv.Author = "inner"
					recEv.Branch = ctx.Branch()
					recEv.Content = textContent("model", "inner-live-recovery")
					yield(recEv, nil)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-live", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	// root holds every event of both iterations and replays them in order
	// after the second one: each child event was created before the second
	// child started, so it is the first child's, and the first child, which
	// the first recovery retires as a drained descendant, must not later look
	// like a peer that could have created the second child's event.
	runFullBufferReplay := func(t *testing.T, callbackRecovery, recoverFirst bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				n := childIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child partial %d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, fmt.Errorf("parallel branch failure %d", n))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		var innerIter atomic.Int32
		recovers := func() bool { return recoverFirst || innerIter.Load() == 2 }
		innerCfg := agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				innerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if !callbackRecovery && recovers() {
						recEv := session.NewEvent(ctx, ctx.InvocationID())
						recEv.Author = "inner"
						recEv.Branch = ctx.Branch()
						recEv.Content = textContent("model", "inner live recovery")
						yield(recEv, nil)
					}
				}
			},
		}
		if callbackRecovery {
			innerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if !recovers() {
						return nil, nil
					}
					return textContent("model", "inner callback recovery"), nil
				},
			}
		}
		inner, err := agent.New(innerCfg)
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if runErr != nil {
								yield(nil, runErr)
								return
							}
							held = append(held, ev)
						}
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-full-buffer", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	for _, tc := range []struct {
		name             string
		callbackRecovery bool
	}{
		{"live session.NewEvent recovery", false},
		{"AfterAgentCallback recovery", true},
	} {
		t.Run("full buffer replayed in order across two recovered iterations with "+tc.name, func(t *testing.T) {
			rows := runFullBufferReplay(t, tc.callbackRecovery, true)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
		t.Run("full buffer replayed in order keeps an unrecovered first iteration failed with "+tc.name, func(t *testing.T) {
			rows := runFullBufferReplay(t, tc.callbackRecovery, false)
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			msg, _ := invErr["error_message"].(string)
			if !strings.Contains(msg, "parallel branch failure 1") || strings.Contains(msg, "parallel branch failure 2") {
				t.Errorf("INVOCATION_ERROR error_message = %q, want the unrecovered first iteration's failure only", msg)
			}
		})
	}

	// root holds events of the first iteration and delivers them only once
	// the second iteration yields its first event, while the second inner,
	// its parallel agent and its child are open: the held events are the
	// first inner's, which retired before the second one started. Only the
	// first child fails. With recovery set, the first inner recovers from the
	// failure with a live event or from its AfterAgentCallback; without it,
	// the first inner only reports progress before the failure. root holds
	// the first inner's own events, or with holdAll every event of the first
	// iteration, so that the first child is still open too.
	runHeldIntoNextIteration := func(t *testing.T, callbackRecovery, recovery, holdAll bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				n := childIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child partial %d", n))
					if !yield(ev, nil) || n > 1 {
						return
					}
					yield(nil, errors.New("parallel branch failure 1"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		var innerIter atomic.Int32
		innerCfg := agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				first := innerIter.Add(1) == 1
				return func(yield func(*session.Event, error) bool) {
					if first && !recovery {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner progress before the failure")
						if !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if first && recovery && !callbackRecovery {
						time.Sleep(time.Millisecond)
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner live recovery")
						yield(ev, nil)
					}
				}
			},
		}
		if callbackRecovery {
			innerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if !recovery || innerIter.Load() != 1 {
						return nil, nil
					}
					return textContent("model", "inner callback recovery"), nil
				},
			}
		}
		inner, err := agent.New(innerCfg)
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if holdAll || ev.Author == "inner" {
							held = append(held, ev)
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						for _, h := range held {
							if !yield(h, nil) {
								return
							}
						}
						held = nil
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-held-into-next", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	for _, tc := range []struct {
		name             string
		callbackRecovery bool
	}{
		{"live session.NewEvent", false},
		{"AfterAgentCallback", true},
	} {
		for _, held := range []struct {
			name    string
			holdAll bool
		}{
			{"the first inner's events", false},
			{"every first-iteration event", true},
		} {
			t.Run(fmt.Sprintf("%s recovery held with %s until the next iteration is open clears the failure", tc.name, held.name), func(t *testing.T) {
				rows := runHeldIntoNextIteration(t, tc.callbackRecovery, true, held.holdAll)
				if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
					t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
					t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
				}
				_ = onlyRowOfType(t, rows, EventInvocationCompleted)
			})
			t.Run(fmt.Sprintf("progress before the failure held with %s until the next iteration is open keeps the failure with %s callbacks", held.name, tc.name), func(t *testing.T) {
				rows := runHeldIntoNextIteration(t, tc.callbackRecovery, false, held.holdAll)
				if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
					t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
				}
				invErr := onlyRowOfType(t, rows, EventInvocationError)
				if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "parallel branch failure 1") {
					t.Errorf("INVOCATION_ERROR error_message = %q, want the first iteration's failure", msg)
				}
			})
		}
	}

	t.Run("unrecovered first iteration negative control", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errIter1 := errors.New("parallel branch failure iter 1")
		errIter2 := errors.New("parallel branch failure iter 2")

		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				cur := childIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", cur))
					if !yield(ev, nil) {
						return
					}
					if cur == 1 {
						yield(nil, errIter1)
						return
					}
					yield(nil, errIter2)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		var innerIter atomic.Int32
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				innerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					if innerIter.Load() == 2 {
						return textContent("model", "iter2-only-recovery"), nil
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-unrecovered", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		msg, _ := invErr["error_message"].(string)
		if !strings.Contains(msg, errIter1.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q, want substring %q", msg, errIter1.Error())
		}
		if strings.Contains(msg, errIter2.Error()) {
			t.Errorf("INVOCATION_ERROR error_message = %q unexpectedly named recovered iter 2 error %q", msg, errIter2.Error())
		}
	})

	t.Run("delayed buffered branch event from recovered first iteration fails closed", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				cur := iterCount.Add(1)
				return func(yield func(*session.Event, error) bool) {
					if cur == 1 {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "child"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "iter1-buffered-child-event")
						if !yield(ev, nil) {
							return
						}
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(_ agent.Context) (*genai.Content, error) {
					return textContent("model", "inner-callback-recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var bufferedChildEv *session.Event
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && ev.Author == "child" {
							bufferedChildEv = ev
							continue
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					for ev, runErr := range inner.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev != nil && !yield(ev, nil) {
							return
						}
					}
					if bufferedChildEv != nil {
						yield(bufferedChildEv, nil)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-seq-par-delayed-child", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})

	runCleanThenRecovered := func(t *testing.T, liveRecovery, includeNoOpCallback bool) {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch iter 2 failure")

		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				cur := childIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-msg-%d", cur))
					if !yield(ev, nil) {
						return
					}
					if cur >= 2 {
						yield(nil, errChild)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		var innerIter atomic.Int32
		innerCfg := agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				it := innerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if liveRecovery && it >= 2 {
						time.Sleep(time.Millisecond)
						recEv := session.NewEvent(ctx, ctx.InvocationID())
						recEv.Author = "inner"
						recEv.Branch = ctx.Branch()
						recEv.Content = textContent("model", "inner-live-recovery")
						yield(recEv, nil)
					}
				}
			},
		}
		if !liveRecovery || includeNoOpCallback {
			innerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if !liveRecovery && innerIter.Load() >= 2 {
						return textContent("model", "inner-cb-recovery"), nil
					}
					return nil, nil
				},
			}
		}
		inner, err := agent.New(innerCfg)
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-clean-then-rec", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	}

	t.Run("clean first iteration followed by AfterAgentCallback recovery", func(t *testing.T) {
		runCleanThenRecovered(t, false, true)
	})
	t.Run("clean first iteration followed by live session.NewEvent recovery (with no-op callback)", func(t *testing.T) {
		runCleanThenRecovered(t, true, true)
	})
	t.Run("clean first iteration followed by live session.NewEvent recovery (without callbacks)", func(t *testing.T) {
		runCleanThenRecovered(t, true, false)
	})

	t.Run("live session.NewEvent recovery across two sequential parallel runs inside open custom agent", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := iterCount.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for iter := 1; iter <= 2; iter++ {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						time.Sleep(time.Millisecond)
						recEv := session.NewEvent(ctx, ctx.InvocationID())
						recEv.Author = "root"
						recEv.Branch = ctx.Branch()
						recEv.Content = textContent("model", fmt.Sprintf("root-live-recovery-%d", iter))
						if !yield(recEv, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open-root-seq-par", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	t.Run("quantized advancing clock sequential live recovery", func(t *testing.T) {
		base := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
		var tickMs atomic.Int64
		runCtx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return base.Add(time.Duration(tickMs.Load()) * time.Millisecond)
		})

		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var iterCount atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := iterCount.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					tickMs.Add(5)
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					tickMs.Add(10)
					recEv := session.NewEvent(ctx, ctx.InvocationID())
					recEv.Author = "inner"
					recEv.Branch = ctx.Branch()
					recEv.Content = textContent("model", "inner-live-recovery")
					yield(recEv, nil)
					tickMs.Add(10)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(runCtx, "u1", "s-coarse-seq-live", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	t.Run("open parent live session.NewEvent recovery settles failed inner activation across sequential iterations", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := childIter.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		var innerIter atomic.Int32
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				innerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if innerIter.Load() == 2 {
						return textContent("model", "inner2-cb-recovery"), nil
					}
					return nil, nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					// Iteration 1: inner does not recover; root stays open and yields live recovery.
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
					time.Sleep(time.Millisecond)
					recEv := session.NewEvent(ctx, ctx.InvocationID())
					recEv.Author = "root"
					recEv.Branch = ctx.Branch()
					recEv.Content = textContent("model", "root-live-recovery-1")
					if !yield(recEv, nil) {
						return
					}
					// Iteration 2: inner recovers via AfterAgentCallback while root remains open.
					time.Sleep(time.Millisecond)
					for ev, runErr := range inner.Run(ctx) {
						if !yield(ev, runErr) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open-parent-live-settle", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})

	runOuterInnerSequential := func(t *testing.T, outerLive bool) {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		errChild := errors.New("parallel branch failure")

		var childIter atomic.Int32
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := childIter.Add(1)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", fmt.Sprintf("child-partial-%d", n))
					if !yield(ev, nil) {
						return
					}
					yield(nil, errChild)
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)

		var innerIter atomic.Int32
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				it := innerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if it == 2 {
						time.Sleep(time.Millisecond)
						recEv := session.NewEvent(ctx, ctx.InvocationID())
						recEv.Author = "inner"
						recEv.Branch = ctx.Branch()
						recEv.Content = textContent("model", "inner2-live-recovery")
						yield(recEv, nil)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}

		var outerIter atomic.Int32
		outerCfg := agent.Config{
			Name:      "outer",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				outerIter.Add(1)
				return func(yield func(*session.Event, error) bool) {
					if !outerLive {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
						return
					}
					for it := range 2 {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
						if it == 0 {
							time.Sleep(time.Millisecond)
							recEv := session.NewEvent(ctx, ctx.InvocationID())
							recEv.Author = "outer"
							recEv.Branch = ctx.Branch()
							recEv.Content = textContent("model", "outer1-live-recovery")
							if !yield(recEv, nil) {
								return
							}
							time.Sleep(time.Millisecond)
						}
					}
				}
			},
		}
		if !outerLive {
			outerCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if outerIter.Load() == 1 {
						return textContent("model", "outer1-callback-recovery"), nil
					}
					return nil, nil
				},
			}
		}
		outer, err := agent.New(outerCfg)
		if err != nil {
			t.Fatalf("agent.New(outer): %v", err)
		}

		rootIters := 2
		if outerLive {
			rootIters = 1
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{outer},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for range rootIters {
						for ev, runErr := range outer.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
						time.Sleep(time.Millisecond)
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}

		sessionID := "s-outer-cb-then-inner-live"
		if outerLive {
			sessionID = "s-outer-live-then-inner-live"
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", sessionID, textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	}

	t.Run("outer AfterAgentCallback recovery followed by inner live session.NewEvent recovery", func(t *testing.T) {
		runOuterInnerSequential(t, false)
	})
	t.Run("outer live session.NewEvent recovery followed by inner live session.NewEvent recovery", func(t *testing.T) {
		runOuterInnerSequential(t, true)
	})
}

// TestAfterAgentRecoveryWhileConcurrentPeerOpen_Runner runs root -> inner ->
// parallel -> child, where root runs inner twice concurrently: the first run's
// child emits content and fails, and the first inner drains the error, while
// the second run of inner, which runs no sub-agent, stays open until the first
// run is over. The event that ADK builds from the content of the first
// inner's AfterAgentCallback carries action maps of that callback's context,
// so it is the first inner's even though another activation of inner is open,
// and it clears the failure, immediately or after a delay. Content of the
// second inner's AfterAgentCallback does not.
func TestAfterAgentRecoveryWhileConcurrentPeerOpen_Runner(t *testing.T) {
	type runKey struct{}
	run := func(t *testing.T, delay time.Duration, firstRecovers, secondRecovers bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		secondOpen := make(chan struct{})
		release := make(chan struct{})
		var releaseOnce sync.Once
		releaseSecond := func() { releaseOnce.Do(func() { close(release) }) }
		defer releaseSecond()

		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("parallel branch failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if ctx.Value(runKey{}) == 2 {
						close(secondOpen)
						<-release
						return
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(cctx agent.Context) (*genai.Content, error) {
					first := cctx.Value(runKey{}) == 1
					if (first && !firstRecovers) || (!first && !secondRecovers) {
						return nil, nil
					}
					time.Sleep(delay)
					return textContent("model", "inner callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					type item struct {
						ev  *session.Event
						err error
						ack chan bool
					}
					items := make(chan item)
					done := make(chan struct{})
					defer close(done)
					var wg sync.WaitGroup
					wg.Add(2)
					go func() {
						wg.Wait()
						close(items)
					}()
					start := func(n int) {
						go func() {
							defer wg.Done()
							if n == 1 {
								// The second run stays open until the first one,
								// including its after-agent event, is over.
								defer releaseSecond()
							}
							runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, n))
							for ev, runErr := range inner.Run(runCtx) {
								ack := make(chan bool, 1)
								select {
								case items <- item{ev: ev, err: runErr, ack: ack}:
								case <-done:
									return
								}
								select {
								case cont := <-ack:
									if !cont {
										return
									}
								case <-done:
									return
								}
							}
						}()
					}
					start(1)
					startedSecond := false
					for it := range items {
						cont := yield(it.ev, it.err)
						if !startedSecond {
							// The first run's child emitted its content and waits
							// for the ack, so the second run opens while the child
							// and the first inner are still open.
							startedSecond = true
							start(2)
							select {
							case <-secondOpen:
							case <-time.After(10 * time.Second):
								yield(nil, errors.New("second run of inner did not open"))
								return
							}
						}
						it.ack <- cont
						if !cont {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-concurrent-peer-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	for _, delay := range []time.Duration{0, 2 * time.Millisecond} {
		t.Run(fmt.Sprintf("first inner recovers while the second is open/callback delay %v", delay), func(t *testing.T) {
			rows := run(t, delay, true, false)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
	}
	t.Run("recovery of the second inner keeps the first inner's failure", func(t *testing.T) {
		rows := run(t, 0, false, true)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "parallel branch failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want the first run's failure", msg)
		}
	})
}

func TestCleanParallelIterationsBoundRetiredActivations_Runner(t *testing.T) {
	for _, tc := range []struct {
		name              string
		withChildCallback bool
		directChild       bool
	}{
		{name: "without child AfterAgentCallback", withChildCallback: false, directChild: false},
		{name: "with child AfterAgentCallback", withChildCallback: true, directChild: false},
		{name: "direct clean child with AfterAgentCallback", withChildCallback: true, directChild: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			lis := bufconn.Listen(1 << 20)
			gSrv := grpc.NewServer()
			requests := make(chan *storagepb.AppendRowsRequest, 256)
			storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: requests})
			go func() { _ = gSrv.Serve(lis) }()
			t.Cleanup(gSrv.Stop)

			conn, err := grpc.NewClient(
				"passthrough:///bufnet",
				grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
				grpc.WithTransportCredentials(insecure.NewCredentials()),
			)
			if err != nil {
				t.Fatalf("dial: %v", err)
			}
			t.Cleanup(func() { _ = conn.Close() })

			writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
			if err != nil {
				t.Fatalf("NewBigQueryWriteClient() error = %v", err)
			}
			t.Cleanup(func() { _ = writeClient.Close() })

			cfg := DefaultConfig()
			cfg.ProjectID = "test-proj"
			cfg.DatasetID = "test-ds"
			cfg.TableName = "test-tbl"
			cfg.BatchSize = 100
			cfg.BatchFlushIntv = time.Hour
			cfg.DisableCreateViews = true

			processor, err := NewBatchProcessor(ctx, writeClient, "projects/test-proj/datasets/test-ds/tables/test-tbl/streams/_default", cfg)
			if err != nil {
				t.Fatalf("NewBatchProcessor() error = %v", err)
			}
			processor.Start()
			t.Cleanup(processor.Close)

			tm := newTraceManager()
			ap := &analyticsPlugin{
				config:    cfg,
				processor: processor,
				traces:    tm,
			}
			p, err := baseplugin.New(baseplugin.Config{
				Name:                  pluginName,
				OnUserMessageCallback: ap.onUserMessage,
				BeforeRunCallback:     ap.beforeRun,
				AfterRunCallback:      ap.afterRun,
				OnEventCallback:       ap.onEvent,
				BeforeAgentCallback:   ap.beforeAgent,
				AfterAgentCallback:    ap.afterAgent,
				BeforeModelCallback:   ap.beforeModel,
				AfterModelCallback:    ap.afterModel,
				OnModelErrorCallback:  ap.onModelError,
				BeforeToolCallback:    ap.beforeTool,
				AfterToolCallback:     ap.afterTool,
				OnToolErrorCallback:   ap.onToolError,
			})
			if err != nil {
				t.Fatalf("baseplugin.New() error = %v", err)
			}
			t.Cleanup(func() { _ = p.Close() })
			tm.selfPlugin = weak.Make(p)

			childCfg := agent.Config{
				Name: "child",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "child"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "ok")
						yield(ev, nil)
					}
				},
			}
			if tc.withChildCallback {
				childCfg.AfterAgentCallbacks = []agent.AfterAgentCallback{
					func(agent.Context) (*genai.Content, error) { return nil, nil },
				}
			}
			child, err := agent.New(childCfg)
			if err != nil {
				t.Fatalf("agent.New(child): %v", err)
			}
			target := child
			if !tc.directChild {
				target = newParallelAgent(t, "par", child)
			}

			var retiredLen int
			var retainedDeltaPtrs, retainedDeltaRefs int
			// retainedLatest reports that the entry that keeps action maps, if
			// any, is the latest one of child.
			retainedLatest := true
			root, err := agent.New(agent.Config{
				Name:      "root",
				SubAgents: []agent.Agent{target},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for range 50 {
							for ev, runErr := range target.Run(ctx) {
								if runErr != nil {
									yield(nil, runErr)
									return
								}
								if ev != nil && !yield(ev, nil) {
									return
								}
							}
						}
						tm.mu.Lock()
						inv := tm.invocations[ctx.InvocationID()]
						tm.mu.Unlock()
						if inv != nil {
							inv.mu.Lock()
							retiredLen = len(inv.retiredActivations)
							var latest uint64
							for _, ra := range inv.retiredActivations {
								if ra.swallower.name == "child" {
									latest = max(latest, ra.retiredAtSeq)
								}
							}
							for _, ra := range inv.retiredActivations {
								if ra.afterAgentActions != (actionMaps{}) {
									retainedDeltaPtrs++
									retainedLatest = retainedLatest && ra.swallower.name == "child" && ra.retiredAtSeq == latest
								}
								if ra.afterAgentActionsRef != nil {
									retainedDeltaRefs++
								}
							}
							inv.mu.Unlock()
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}

			for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-bound-50", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error = %v, want nil", runErr)
				}
			}
			if retiredLen == 0 || retiredLen > 8 {
				t.Fatalf("after 50 clean parallel iterations under open root: len(retiredActivations) = %d, want in [1, 8]", retiredLen)
			}
			// A child that root runs on its own goroutine ran its after-agent
			// callbacks there, and that goroutine started each next child
			// only once they returned, so only the latest child, whose
			// callbacks nothing showed to have returned yet, keeps its maps
			// while root still runs: root could hold their event. par waits
			// for the goroutine of its child, so its retirement shows that.
			wantRetained := 0
			if tc.directChild {
				wantRetained = 1
			}
			if retainedDeltaPtrs != wantRetained || retainedDeltaRefs != wantRetained || !retainedLatest {
				t.Fatalf("retained after-agent callback action maps (maps=%d, refs=%d, latest child only %v), want (%d, %d, true)", retainedDeltaPtrs, retainedDeltaRefs, retainedLatest, wantRetained, wantRetained)
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
	}
}

// peerRowsUnderOtherInner checks that every row of eventType sits under an
// activation of inner other than the one that ran par, which is the retired
// activation whose after-agent event an open peer could claim.
func peerRowsUnderOtherInner(t *testing.T, rows []map[string]any, eventType string) {
	t.Helper()
	var inner1Span string
	innerSpans := map[string]bool{}
	for _, r := range rowsOfType(rows, EventAgentStarting) {
		span, _ := r["span_id"].(string)
		switch r["agent"] {
		case "par":
			inner1Span, _ = r["parent_span_id"].(string)
		case "inner":
			innerSpans[span] = true
		}
	}
	got := rowsOfType(rows, eventType)
	if len(got) == 0 {
		t.Fatalf("got no %s rows; event types=%v", eventType, eventTypes(rows))
	}
	for _, r := range got {
		parent, _ := r["parent_span_id"].(string)
		if parent == inner1Span || !innerSpans[parent] {
			t.Errorf("%s row parent_span_id = %q, want the open peer's span (inner spans %v, retired inner %q)", eventType, parent, innerSpans, inner1Span)
		}
	}
}

// TestAfterAgentRecoveryWithOpenPeerAwaitingEvent_Runner covers a custom
// inner whose child under par fails and whose AfterAgentCallback recovers,
// and whose recovery event root holds until another activation of inner, an
// llmagent, is open and waits for an event of its own: the event of a model
// call that AfterModelCallback holds, the rest of a streamed response after a
// partial event, or the event of a tool call that AfterToolCallback holds.
// The held event carries action maps of the first inner's after-agent
// callback context, so it is the retired inner's, whatever the open peer
// waits for, and it clears the failure. Without recovery, the failure stays.
func TestAfterAgentRecoveryWithOpenPeerAwaitingEvent_Runner(t *testing.T) {
	const (
		awaitModel   = "awaiting_model"
		awaitPartial = "partial"
		awaitTool    = "awaiting_tool"
	)
	run := func(t *testing.T, await string, concurrent, innerProgress, recovers bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		peerPaused := make(chan struct{})
		ev1Delivered := make(chan struct{})
		var pauseOnce, deliverOnce sync.Once
		pause := func() { pauseOnce.Do(func() { close(peerPaused) }) }
		deliver := func() { deliverOnce.Do(func() { close(ev1Delivered) }) }
		defer deliver()
		waitDelivered := func(ctx context.Context) {
			select {
			case <-ev1Delivered:
			case <-ctx.Done():
			case <-time.After(10 * time.Second):
			}
		}

		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("parallel branch failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner1, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if innerProgress {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner progress")
						if !yield(ev, nil) {
							return
						}
					}
					if concurrent {
						select {
						case <-peerPaused:
						case <-time.After(10 * time.Second):
							yield(nil, errors.New("the open peer did not pause"))
							return
						}
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if !recovers {
						return nil, nil
					}
					if await == awaitTool {
						return &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
							ID: "recover_0", Name: "recover", Response: map[string]any{"result": "inner callback recovery"},
						}}}}, nil
					}
					return textContent("model", "inner callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Look up item"}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			ctx.Actions().SkipSummarization = true
			return map[string]any{"res": "r1"}, nil
		})
		if err != nil {
			t.Fatalf("functiontool.New(lookup): %v", err)
		}
		var modelCalls atomic.Int32
		peerCfg := llmagent.Config{
			Name: "inner",
			Model: &funcStreamLLM{fn: func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
				switch {
				case await == awaitTool && modelCalls.Add(1) == 1:
					yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_0", Name: "lookup", Args: map[string]any{}}}}}}, nil)
				case await == awaitPartial:
					if !yield(&model.LLMResponse{Content: textContent("model", "peer "), Partial: true}, nil) {
						return
					}
					waitDelivered(ctx)
					yield(&model.LLMResponse{Content: textContent("model", "peer answer"), TurnComplete: true}, nil)
				default:
					yield(&model.LLMResponse{Content: textContent("model", "peer answer"), TurnComplete: true}, nil)
				}
			}},
			Tools: []tool.Tool{lookup},
		}
		switch await {
		case awaitModel:
			peerCfg.AfterModelCallbacks = []llmagent.AfterModelCallback{
				func(ctx agent.Context, res *model.LLMResponse, _ error) (*model.LLMResponse, error) {
					if res != nil && !res.Partial {
						pause()
						waitDelivered(ctx)
					}
					return nil, nil
				},
			}
		case awaitTool:
			peerCfg.AfterToolCallbacks = []llmagent.AfterToolCallback{
				func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
					pause()
					waitDelivered(ctx)
					return nil, nil
				},
			}
		}
		// The peer is another activation of inner that root runs itself, so
		// it is not a sub-agent in the tree, whose names are unique.
		inner2, err := llmagent.New(peerCfg)
		if err != nil {
			t.Fatalf("llmagent.New(inner): %v", err)
		}
		isRecovery := func(ev *session.Event) bool {
			if ev == nil || ev.Author != "inner" || ev.Content == nil {
				return false
			}
			for _, part := range ev.Content.Parts {
				if part.Text == "inner callback recovery" || (part.FunctionResponse != nil && part.FunctionResponse.Name == "recover") {
					return true
				}
			}
			return false
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner1},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					type item struct {
						ev  *session.Event
						err error
						ack chan struct{}
					}
					peerItems := make(chan item)
					done := make(chan struct{})
					defer close(done)
					peerStarted := false
					startPeer := func() {
						peerStarted = true
						go func() {
							defer close(peerItems)
							for ev, runErr := range inner2.Run(ctx) {
								// The peer resumes, and reads the event again,
								// only once root has yielded it, because the
								// runner writes to an event that root yields.
								ack := make(chan struct{})
								select {
								case peerItems <- item{ev: ev, err: runErr, ack: ack}:
								case <-done:
									return
								case <-ctx.Done():
									return
								}
								select {
								case <-ack:
								case <-done:
									return
								case <-ctx.Done():
									return
								}
							}
						}()
					}
					forward := func(it item) bool {
						defer close(it.ack)
						return yield(it.ev, it.err)
					}
					var held *session.Event
					for ev, runErr := range inner1.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if isRecovery(ev) {
							held = ev
							continue
						}
						if !yield(ev, nil) {
							return
						}
						if concurrent && !peerStarted {
							// The first inner waits in its run, after its
							// progress, until the peer pauses.
							startPeer()
						}
					}
					if !peerStarted {
						startPeer()
					}
					deliverHeld := func() bool {
						defer deliver()
						return held == nil || yield(held, nil)
					}
					if await == awaitPartial {
						it, ok := <-peerItems
						if !ok || it.err != nil || it.ev == nil || !it.ev.Partial {
							// Closing done on return releases the peer.
							yield(nil, fmt.Errorf("first event of the peer = %+v, %v, want a partial event", it.ev, it.err))
							return
						}
						if !forward(it) {
							return
						}
					} else {
						// The peer's events before it pauses, such as its
						// function call, are delivered as they come.
						timeout := time.After(10 * time.Second)
					waitPause:
						for {
							select {
							case <-peerPaused:
								break waitPause
							case it, ok := <-peerItems:
								if !ok {
									yield(nil, errors.New("the open peer ended without pausing"))
									return
								}
								if !forward(it) {
									return
								}
							case <-timeout:
								yield(nil, errors.New("the open peer did not pause"))
								return
							}
						}
					}
					if !deliverHeld() {
						return
					}
					for it := range peerItems {
						if !forward(it) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-open-peer-awaiting-event", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	wantRecovered := func(t *testing.T, rows []map[string]any, await string) {
		t.Helper()
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		peerRowsUnderOtherInner(t, rows, EventLLMResponse)
		if await == awaitTool {
			peerRowsUnderOtherInner(t, rows, EventToolCompleted)
		}
	}
	wantFailure := func(t *testing.T, rows []map[string]any) {
		t.Helper()
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "parallel branch failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want the first inner's failure", msg)
		}
	}
	for _, progress := range []bool{false, true} {
		t.Run(fmt.Sprintf("sequential/awaiting_model/inner_progress_%v", progress), func(t *testing.T) {
			wantRecovered(t, run(t, awaitModel, false, progress, true), awaitModel)
		})
	}
	t.Run("concurrent/awaiting_model", func(t *testing.T) {
		wantRecovered(t, run(t, awaitModel, true, true, true), awaitModel)
	})
	t.Run("sequential/partial", func(t *testing.T) {
		wantRecovered(t, run(t, awaitPartial, false, false, true), awaitPartial)
	})
	t.Run("sequential/awaiting_tool/function_response_recovery", func(t *testing.T) {
		wantRecovered(t, run(t, awaitTool, false, false, true), awaitTool)
	})
	t.Run("sequential/awaiting_model/no_recovery_keeps_the_failure", func(t *testing.T) {
		wantFailure(t, run(t, awaitModel, false, false, false))
	})
	t.Run("sequential/awaiting_tool/no_recovery_keeps_the_failure", func(t *testing.T) {
		wantFailure(t, run(t, awaitTool, false, false, false))
	})
}

// TestAfterAgentRecoveryAfterConcurrentPeerRetired_Runner covers two
// activations of a custom inner, the first of which drains the failure of
// its child under par and recovers in its AfterAgentCallback, while the second
// starts before the first one's recovery event is created and retires after
// root reads it, so that the event falls in the second one's run. Root
// delivers the recovery event only after both activations retired. The event
// carries action maps of the first inner's after-agent callback context, so
// it is the first inner's although the second one could have created it by
// timestamp, whether the two ran concurrently or the second started after
// the first one retired but before its AfterAgentCallback ran, and it clears
// the failure. With both maps copied, nothing shows whose the event is, and
// the failure stays. When both activations drain a failure of their own and
// only one recovers, the other one's failure, which was swallowed before the
// recovery event was created, stays. The two child activations share agent,
// branch and node path, so neither fails before root has delivered the
// events of both: an event that one created while the other ran and that
// reaches the plugin after the other retired under root could be the retired
// one's, held by root, and records no response, so the recovered failure
// stays reported, as the held-child cases pin.
func TestAfterAgentRecoveryAfterConcurrentPeerRetired_Runner(t *testing.T) {
	type runKey struct{}
	type config struct {
		popThenPush   bool
		ev2First      bool
		innerProgress bool
		// recovers[n] reports whether the AfterAgentCallback of run n+1
		// recovers.
		recovers [2]bool
		// bothFail makes the second run drain a failure of its own too.
		bothFail bool
		// holdRecoveredChild, with bothFail, makes the recovering run's child
		// start after root delivered the other child's event and create its
		// own while the other child runs, and root hold that event until both
		// inner activations retired.
		holdRecoveredChild bool
		copyMaps           bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		inner1Started := make(chan struct{})
		inner2Started := make(chan struct{})
		inner1Popped := make(chan struct{})
		ev1Received := make(chan struct{})
		child1Swallowed := make(chan struct{})
		// childStarted[n-1] is closed when the child of run n starts, and
		// childSeen[n-1] once root delivered or held that child's event.
		childStarted := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
		childSeen := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
		var ev1Once, swallowOnce sync.Once
		var childSeenOnce [2]sync.Once
		receivedEv1 := func() { ev1Once.Do(func() { close(ev1Received) }) }
		swallowed := func() { swallowOnce.Do(func() { close(child1Swallowed) }) }
		sawChild := func(n int) { childSeenOnce[n-1].Do(func() { close(childSeen[n-1]) }) }
		defer receivedEv1()
		defer swallowed()
		defer sawChild(1)
		defer sawChild(2)
		wait := func(ch <-chan struct{}) {
			select {
			case <-ch:
			case <-time.After(10 * time.Second):
			}
		}
		runOf := func(ctx context.Context) int {
			n, _ := ctx.Value(runKey{}).(int)
			return n
		}

		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := runOf(ctx)
					if c.bothFail {
						close(childStarted[n-1])
						if c.holdRecoveredChild && c.recovers[n-1] {
							// The event falls in the other child's run.
							wait(childStarted[2-n])
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					if c.bothFail {
						// Neither child retires before root delivered or held
						// the events of both.
						wait(childSeen[0])
						wait(childSeen[1])
					}
					if c.recovers[n-1] {
						yield(nil, errors.New("recovered child failure"))
					} else {
						yield(nil, errors.New("child-1 failure"))
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := runOf(ctx)
					answer := func() bool {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner-2 answer")
						return yield(ev, nil)
					}
					if n == 1 {
						close(inner1Started)
						if c.innerProgress {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "inner"
							ev.Branch = ctx.Branch()
							ev.Content = textContent("model", "inner progress")
							if !yield(ev, nil) {
								return
							}
						}
						if !c.popThenPush {
							wait(inner2Started)
						}
					} else {
						close(inner2Started)
						// Content that a custom agent emits after it swallowed a
						// failure of a sub-agent recovers that failure, so the
						// second run answers before it runs par when it does.
						if c.bothFail && !answer() {
							return
						}
					}
					if c.holdRecoveredChild && c.recovers[n-1] {
						// The other child's event, which root delivers as it
						// comes, is created before this run's child starts,
						// so it is provably the other child's: only the held
						// event has no provable author.
						wait(childSeen[2-n])
					}
					if n == 1 || c.bothFail {
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								if !c.recovers[n-1] {
									swallowed()
								}
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
					}
					if n == 2 {
						if !c.bothFail && !answer() {
							return
						}
						if c.recovers[0] {
							// The first inner's recovery event falls in this
							// run.
							wait(ev1Received)
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(cctx agent.Context) (*genai.Content, error) {
					n := runOf(cctx)
					if n == 1 && c.popThenPush {
						// This plugin popped the first inner before ADK runs
						// its AfterAgentCallback; the second one starts now.
						close(inner1Popped)
						wait(inner2Started)
					}
					if !c.recovers[n-1] {
						return nil, nil
					}
					if c.bothFail {
						wait(child1Swallowed)
					}
					return textContent("model", "inner callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					type item struct {
						n   int
						ev  *session.Event
						err error
						ack chan struct{}
					}
					items := make(chan item)
					done := make(chan struct{})
					defer close(done)
					var wg sync.WaitGroup
					wg.Add(2)
					go func() {
						wg.Wait()
						close(items)
					}()
					start := func(n int) {
						go func() {
							defer wg.Done()
							runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, n))
							for ev, runErr := range inner.Run(runCtx) {
								ack := make(chan struct{})
								select {
								case items <- item{n: n, ev: ev, err: runErr, ack: ack}:
								case <-done:
									return
								}
								select {
								case <-ack:
								case <-done:
									return
								}
							}
						}()
					}
					start(1)
					if c.popThenPush {
						go func() {
							select {
							case <-inner1Popped:
								start(2)
							case <-done:
								wg.Done()
							}
						}()
					} else {
						wait(inner1Started)
						start(2)
					}
					// Root holds the inner activations' own content, and with
					// holdRecoveredChild the recovering run's child's event,
					// until both of them retired and delivers the rest as it
					// comes.
					var held [2][]*session.Event
					var heldChild []*session.Event
					for it := range items {
						if it.err == nil && it.ev != nil && it.ev.Author == "inner" && it.ev.Content != nil {
							held[it.n-1] = append(held[it.n-1], it.ev)
							if it.n == 1 && isTextContent(it.ev.Content, "inner callback recovery") {
								receivedEv1()
							}
							close(it.ack)
							continue
						}
						isChild := it.err == nil && it.ev != nil && it.ev.Author == "child" && it.ev.Content != nil
						if isChild && c.holdRecoveredChild && c.recovers[it.n-1] {
							heldChild = append(heldChild, it.ev)
							sawChild(it.n)
							close(it.ack)
							continue
						}
						if !yield(it.ev, it.err) {
							close(it.ack)
							return
						}
						if isChild {
							sawChild(it.n)
						}
						close(it.ack)
					}
					for _, ev := range heldChild {
						if !yield(ev, nil) {
							return
						}
					}
					order := [][]*session.Event{held[0], held[1]}
					if c.ev2First {
						order = [][]*session.Event{held[1], held[0]}
					}
					for _, evs := range order {
						for _, ev := range evs {
							if !yield(ev, nil) {
								return
							}
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		plugins := []*baseplugin.Plugin{p}
		if c.copyMaps {
			copier, err := baseplugin.New(baseplugin.Config{
				Name: "action_map_copier",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil {
						ev.Actions.StateDelta = maps.Clone(ev.Actions.StateDelta)
						ev.Actions.ArtifactDelta = maps.Clone(ev.Actions.ArtifactDelta)
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(action_map_copier): %v", err)
			}
			plugins = []*baseplugin.Plugin{copier, p}
		}
		for _, runErr := range newTestRunner(t, root, plugins...).Run(t.Context(), "u1", "s-concurrent-peer-retired", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	wantRecovered := func(t *testing.T, rows []map[string]any) {
		t.Helper()
		if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
			t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	}
	for _, order := range []string{"ev2_then_ev1", "ev1_then_ev2"} {
		for _, progress := range []bool{false, true} {
			t.Run(fmt.Sprintf("concurrent/%s/inner_progress_%v", order, progress), func(t *testing.T) {
				wantRecovered(t, run(t, config{ev2First: order == "ev2_then_ev1", innerProgress: progress, recovers: [2]bool{true, false}}))
			})
		}
	}
	for _, progress := range []bool{false, true} {
		t.Run(fmt.Sprintf("pop_then_push/inner_progress_%v", progress), func(t *testing.T) {
			wantRecovered(t, run(t, config{popThenPush: true, ev2First: true, innerProgress: progress, recovers: [2]bool{true, false}}))
		})
	}
	t.Run("concurrent/both_action_maps_copied_fails_closed", func(t *testing.T) {
		rows := run(t, config{ev2First: true, recovers: [2]bool{true, false}, copyMaps: true})
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
	for _, tc := range []struct {
		name     string
		recovers [2]bool
	}{
		{"concurrent/recovery_of_inner-1_keeps_child-1_that_inner-2_swallowed_earlier", [2]bool{true, false}},
		{"concurrent/recovery_of_inner-2_keeps_child-1_that_inner-1_swallowed_earlier", [2]bool{false, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := run(t, config{ev2First: true, recovers: tc.recovers, bothFail: true})
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			msg, _ := invErr["error_message"].(string)
			if !strings.Contains(msg, "child-1 failure") || strings.Contains(msg, "recovered child failure") {
				t.Errorf("INVOCATION_ERROR error_message = %q, want the unrecovered child-1 failure alone", msg)
			}
			for _, r := range rowsOfType(rows, EventAgentError) {
				if msg, _ := r["error_message"].(string); strings.Contains(msg, "recovered child failure") {
					t.Errorf("AGENT_ERROR row for the recovered failure: %q", msg)
				}
			}
		})
	}
	for _, tc := range []struct {
		name     string
		recovers [2]bool
	}{
		{"concurrent/event_of_inner-1's_child_held_until_both_retired_fails_closed", [2]bool{true, false}},
		{"concurrent/event_of_inner-2's_child_held_until_both_retired_fails_closed", [2]bool{false, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := run(t, config{ev2First: true, recovers: tc.recovers, bothFail: true, holdRecoveredChild: true})
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationError)
			if !slices.ContainsFunc(rowsOfType(rows, EventAgentError), func(r map[string]any) bool {
				msg, _ := r["error_message"].(string)
				return strings.Contains(msg, "recovered child failure")
			}) {
				t.Errorf("no AGENT_ERROR row for the failure of the child whose event has no provable author; event types=%v", eventTypes(rows))
			}
		})
	}
}

// retiredInnerActions reports, for the retired activations of the agent named
// name in the invocation of ctx that tm traces, whether one keeps the action
// maps of its after-agent callbacks, and whether one keeps them through a
// context.Context.
func retiredInnerActions(tm *traceManager, ctx agent.InvocationContext, name string) (kept, keptByContext bool) {
	tm.mu.Lock()
	inv := tm.invocations[ctx.InvocationID()]
	tm.mu.Unlock()
	if inv == nil {
		return false, false
	}
	inv.mu.Lock()
	defer inv.mu.Unlock()
	for _, ra := range inv.retiredActivations {
		if ra.swallower.name != name {
			continue
		}
		if ra.afterAgentActions != (actionMaps{}) && ra.afterAgentActionsRef != nil {
			kept = true
		}
		if _, ok := ra.afterAgentActionsRef.(context.Context); ok {
			keptByContext = true
		}
	}
	return kept, keptByContext
}

// TestAfterAgentRecoveryAfterEarlierProgressWithHeldChildEvent_Runner runs
// root -> [outer ->] inner -> parallel -> child, where inner first yields
// progress that root delivers at once, then drains its child's failure, and
// its AfterAgentCallback recovers. root holds the child's partial event and
// the recovery and delivers them in the order in which they were created once
// inner returned. inner responded before it retired and the child is still
// open then, so only the open child under it shows that the after-agent event
// may still recover a failure that moves up to root with inner among its
// swallowers. The after-agent maps are kept without the callback context.
func TestAfterAgentRecoveryAfterEarlierProgressWithHeldChildEvent_Runner(t *testing.T) {
	type config struct {
		// outer puts a custom agent that forwards events as they come
		// between root and inner, which retires while the child is open.
		outer        bool
		recovers     bool
		copyBothMaps bool
	}
	run := func(t *testing.T, c config) (rows []map[string]any, kept, keptByContext bool) {
		t.Helper()
		tm := newTraceManager()
		p, takeRows := newQueuedRowsPlugin(t, tm)
		tm.selfPlugin = weak.Make(p)
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("held child failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "inner progress")
					if !yield(ev, nil) {
						return
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{
				func(agent.Context) (*genai.Content, error) {
					if !c.recovers {
						return nil, nil
					}
					return textContent("model", "inner callback recovery"), nil
				},
			},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		top := inner
		if c.outer {
			top, err = agent.New(agent.Config{
				Name:      "outer",
				SubAgents: []agent.Agent{inner},
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						for ev, runErr := range inner.Run(ctx) {
							if !yield(ev, runErr) {
								return
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(outer): %v", err)
			}
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{top},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					var held []*session.Event
					for ev, runErr := range top.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if ev.Author == "inner" && isTextContent(ev.Content, "inner progress") {
							if !yield(ev, nil) {
								return
							}
							continue
						}
						held = append(held, ev)
					}
					kept, keptByContext = retiredInnerActions(tm, ctx, "inner")
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		plugins := []*baseplugin.Plugin{p}
		if c.copyBothMaps {
			copier, err := baseplugin.New(baseplugin.Config{
				Name: "action_map_copier",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil {
						ev.Actions.StateDelta = maps.Clone(ev.Actions.StateDelta)
						ev.Actions.ArtifactDelta = maps.Clone(ev.Actions.ArtifactDelta)
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(action_map_copier): %v", err)
			}
			plugins = []*baseplugin.Plugin{copier, p}
		}
		for _, runErr := range newTestRunner(t, root, plugins...).Run(t.Context(), "u1", "s-held-child-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return takeRows(), kept, keptByContext
	}
	for _, outer := range []bool{false, true} {
		t.Run(fmt.Sprintf("outer_%v/recovery_completes", outer), func(t *testing.T) {
			rows, kept, keptByContext := run(t, config{outer: outer, recovers: true})
			if !kept || keptByContext {
				t.Errorf("retired inner kept its after-agent maps = %v, through a context = %v; want true and false", kept, keptByContext)
			}
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
		t.Run(fmt.Sprintf("outer_%v/no_recovery_keeps_the_failure", outer), func(t *testing.T) {
			rows, _, _ := run(t, config{outer: outer})
			if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
				t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
			}
			invErr := onlyRowOfType(t, rows, EventInvocationError)
			if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "held child failure") {
				t.Errorf("INVOCATION_ERROR error_message = %q, want the child's failure", msg)
			}
		})
	}
	t.Run("both_action_maps_copied_fails_closed", func(t *testing.T) {
		rows, _, _ := run(t, config{recovers: true, copyBothMaps: true})
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationError)
	})
}

// TestOpenPeerRecoveryAfterConcurrentPeerFinished_Runner runs two concurrent
// activations of the custom agent inner under a custom root that delivers
// every event as it comes. The first answers and retires while the second is
// open; only then does the second run parallel -> child, drain the child's
// failure and yield its own recovery. No retired activation of inner can have
// created that recovery, so it is the open activation's, whatever
// retiredPeerCount and the failure that the open activation holds say. The
// controls keep the failure when nothing recovers, when the retired peer's
// after-agent callbacks ran but no after-agent event of it arrived (it may
// still come, with both of its maps replaced), and when the recovery was
// created while the retired peer still ran and root held it until that peer
// retired.
func TestOpenPeerRecoveryAfterConcurrentPeerFinished_Runner(t *testing.T) {
	type runKey struct{}
	type config struct {
		// peerCallback gives inner an AfterAgentCallback, whose first
		// activation returns content unless peerCallbackSilent is set, or
		// only sets state if peerCallbackState is set, for which ADK builds
		// an event without content.
		peerCallback       bool
		peerCallbackSilent bool
		peerCallbackState  bool
		// copyMaps makes an earlier plugin replace both action maps of
		// every event.
		copyMaps bool
		recovers bool
		// holdRecovery makes the second activation fail and recover before
		// the first one answers, and root hold the recovery until the first
		// one retired.
		holdRecovery bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		wait := func(ch <-chan struct{}) {
			select {
			case <-ch:
			case <-time.After(10 * time.Second):
			}
		}
		runOf := func(ctx context.Context) int {
			n, _ := ctx.Value(runKey{}).(int)
			return n
		}
		inner2Started := make(chan struct{})
		inner1Done := make(chan struct{})
		peerSettled := make(chan struct{})
		recoveryCreated := make(chan struct{})
		heldDelivered := make(chan struct{})
		var settleOnce, recoveryOnce, heldOnce sync.Once
		settle := func() { settleOnce.Do(func() { close(peerSettled) }) }
		created := func() { recoveryOnce.Do(func() { close(recoveryCreated) }) }
		delivered := func() { heldOnce.Do(func() { close(heldDelivered) }) }
		defer settle()
		defer created()
		defer delivered()
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("child-2 failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		cfg := agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if runOf(ctx) == 1 {
						wait(inner2Started)
						if c.holdRecovery {
							wait(recoveryCreated)
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "inner"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "inner-1 answer")
						yield(ev, nil)
						return
					}
					close(inner2Started)
					if !c.holdRecovery {
						// The first activation retired, and its after-agent
						// event, if any, was delivered.
						wait(inner1Done)
						wait(peerSettled)
					}
					for ev, runErr := range par.Run(ctx) {
						if runErr != nil {
							continue
						}
						if !yield(ev, nil) {
							return
						}
					}
					if !c.recovers {
						return
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "inner-2 recovery")
					created()
					if !yield(ev, nil) {
						return
					}
					if c.holdRecovery {
						// Still open when root delivers the held recovery.
						wait(heldDelivered)
					}
				}
			},
		}
		if c.peerCallback {
			cfg.AfterAgentCallbacks = []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runOf(cctx) != 1 || c.peerCallbackSilent {
					return nil, nil
				}
				if c.peerCallbackState {
					return nil, cctx.State().Set("inner_1_state", true)
				}
				return textContent("model", "inner-1 callback"), nil
			}}
		}
		inner, err := agent.New(cfg)
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					type item struct {
						ev  *session.Event
						err error
						ack chan struct{}
					}
					items := make(chan item)
					done := make(chan struct{})
					defer close(done)
					var wg sync.WaitGroup
					wg.Add(2)
					go func() {
						wg.Wait()
						close(items)
					}()
					for n := 1; n <= 2; n++ {
						go func() {
							defer wg.Done()
							if n == 1 {
								defer close(inner1Done)
							}
							runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, n))
							for ev, runErr := range inner.Run(runCtx) {
								ack := make(chan struct{})
								select {
								case items <- item{ev: ev, err: runErr, ack: ack}:
								case <-done:
									return
								}
								select {
								case <-ack:
								case <-done:
									return
								}
							}
						}()
					}
					if !c.peerCallback || c.peerCallbackSilent {
						settle()
					}
					var held []*session.Event
					peerDone := inner1Done
					for {
						select {
						case it, ok := <-items:
							if !ok {
								return
							}
							if c.holdRecovery && it.err == nil && it.ev != nil && isTextContent(it.ev.Content, "inner-2 recovery") {
								held = append(held, it.ev)
								close(it.ack)
								continue
							}
							if !yield(it.ev, it.err) {
								close(it.ack)
								return
							}
							if it.err == nil && it.ev != nil && (isTextContent(it.ev.Content, "inner-1 callback") ||
								(it.ev.Author == "inner" && it.ev.Content == nil && it.ev.Actions.StateDelta["inner_1_state"] != nil)) {
								settle()
							}
							close(it.ack)
						case <-peerDone:
							peerDone = nil
							for _, ev := range held {
								if !yield(ev, nil) {
									return
								}
							}
							delivered()
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		plugins := []*baseplugin.Plugin{p}
		if c.copyMaps {
			copier, err := baseplugin.New(baseplugin.Config{
				Name: "action_map_copier",
				OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil {
						ev.Actions.StateDelta = maps.Clone(ev.Actions.StateDelta)
						ev.Actions.ArtifactDelta = maps.Clone(ev.Actions.ArtifactDelta)
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(action_map_copier): %v", err)
			}
			plugins = []*baseplugin.Plugin{copier, p}
		}
		for _, runErr := range newTestRunner(t, root, plugins...).Run(t.Context(), "u1", "s-open-peer-recovery", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		return collectRows(t, requests, untilInvocationEnd)
	}
	wantFailed := func(t *testing.T, rows []map[string]any) {
		t.Helper()
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "child-2 failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want the child's failure", msg)
		}
	}
	for _, tc := range []struct {
		name string
		c    config
	}{
		{"peer_without_after-agent_callbacks", config{recovers: true}},
		{"peer_after-agent_event_delivered_first", config{peerCallback: true, recovers: true}},
		{"peer_after-agent_state_event_delivered_first", config{peerCallback: true, peerCallbackState: true, recovers: true}},
		{"peer_without_after-agent_callbacks_and_both_maps_copied", config{copyMaps: true, recovers: true}},
	} {
		t.Run(tc.name+"/recovery_completes", func(t *testing.T) {
			rows := run(t, tc.c)
			if n := len(rowsOfType(rows, EventAgentError)); n != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
		t.Run(tc.name+"/no_recovery_keeps_the_failure", func(t *testing.T) {
			c := tc.c
			c.recovers = false
			wantFailed(t, run(t, c))
		})
	}
	t.Run("peer_after-agent_callbacks_without_event_keep_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, config{peerCallback: true, peerCallbackSilent: true, recovers: true}))
	})
	t.Run("peer_after-agent_state_event_with_both_maps_copied_keeps_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, config{peerCallback: true, peerCallbackState: true, copyMaps: true, recovers: true}))
	})
	t.Run("recovery_created_while_the_peer_ran_and_held_keeps_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, config{recovers: true, holdRecovery: true}))
	})
}

// TestAfterAgentFunctionResponseSettlesOwnCredit_Runner runs two concurrent
// activations of the custom agent inner under a custom root that delivers
// every event as it comes. The first retires unresponded while the second is
// open, which leaves one unrespondedRetired credit for its response, and its
// AfterAgentCallback returns a function response. That event is the first
// activation's after-agent event and settles its credit, although it carries
// a call ID, and the second activation's later recovery from a failure that
// it drained completes the run. No credit is left at the end.
func TestAfterAgentFunctionResponseSettlesOwnCredit_Runner(t *testing.T) {
	type runKey struct{}
	tm := newTraceManager()
	p, takeRows := newQueuedRowsPlugin(t, tm)
	tm.selfPlugin = weak.Make(p)
	wait := func(ch <-chan struct{}) {
		select {
		case <-ch:
		case <-time.After(10 * time.Second):
		}
	}
	runOf := func(ctx context.Context) int {
		n, _ := ctx.Value(runKey{}).(int)
		return n
	}
	inner2Started := make(chan struct{})
	trailerDelivered := make(chan struct{})
	var trailerOnce sync.Once
	sawTrailer := func() { trailerOnce.Do(func() { close(trailerDelivered) }) }
	defer sawTrailer()
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "child"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "child partial")
				if !yield(ev, nil) {
					return
				}
				yield(nil, errors.New("child-2 failure"))
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(child): %v", err)
	}
	par := newParallelAgent(t, "par", child)
	inner, err := agent.New(agent.Config{
		Name:      "inner",
		SubAgents: []agent.Agent{par},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if runOf(ctx) == 1 {
					// Retires unresponded while the second activation runs.
					wait(inner2Started)
					return
				}
				close(inner2Started)
				wait(trailerDelivered)
				for ev, runErr := range par.Run(ctx) {
					if runErr != nil {
						continue
					}
					if !yield(ev, nil) {
						return
					}
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "inner"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "inner-2 recovery")
				yield(ev, nil)
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
			if runOf(cctx) != 1 {
				return nil, nil
			}
			return &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID:       "call-inner-1",
				Name:     "lookup",
				Response: map[string]any{"result": "ok"},
			}}}}, nil
		}},
	})
	if err != nil {
		t.Fatalf("agent.New(inner): %v", err)
	}
	credits, toolCredits := -1, -1
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{inner},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				type item struct {
					ev  *session.Event
					err error
					ack chan struct{}
				}
				items := make(chan item)
				done := make(chan struct{})
				defer close(done)
				var wg sync.WaitGroup
				wg.Add(2)
				go func() {
					wg.Wait()
					close(items)
				}()
				for n := 1; n <= 2; n++ {
					go func() {
						defer wg.Done()
						runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, n))
						for ev, runErr := range inner.Run(runCtx) {
							ack := make(chan struct{})
							select {
							case items <- item{ev: ev, err: runErr, ack: ack}:
							case <-done:
								return
							}
							select {
							case <-ack:
							case <-done:
								return
							}
						}
					}()
				}
				for it := range items {
					if !yield(it.ev, it.err) {
						close(it.ack)
						return
					}
					if it.err == nil && it.ev != nil && it.ev.Content != nil && len(it.ev.Content.Parts) == 1 && it.ev.Content.Parts[0].FunctionResponse != nil {
						sawTrailer()
					}
					close(it.ack)
				}
				tm.mu.Lock()
				inv := tm.invocations[ctx.InvocationID()]
				tm.mu.Unlock()
				if inv != nil {
					inv.mu.Lock()
					credits, toolCredits = 0, 0
					for _, n := range inv.unrespondedRetired {
						credits += n
					}
					for _, n := range inv.unrespondedRetiredTools {
						toolCredits += n
					}
					inv.mu.Unlock()
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-after-agent-function-response", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error = %v, want nil", runErr)
		}
	}
	rows := takeRows()
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
	}
	_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	if credits != 0 || toolCredits != 0 {
		t.Errorf("credits left once both activations retired: unrespondedRetired = %d, unrespondedRetiredTools = %d, want 0 and 0", credits, toolCredits)
	}
}

// TestLookalikeParentPrefersRunningGoroutine_Runner runs two activations of
// the custom agent inner under a custom root: the first on a goroutine that
// root starts, the second inline on root's own goroutine once the first
// started. The first then runs parallel -> child and drains the child's
// failure, and the second answers once the first returned. The parallel agent
// starts on the first activation's goroutine, which root's goroutine started:
// the first runs on that goroutine and the second, which started last, on the
// one that started it. The parallel agent runs under the first, so the first
// activation's AfterAgentCallback recovers the failure; placed under the
// second, the failure would not be recorded as one that the first swallowed,
// and that recovery would leave it reported. Without the recovery, the
// failure stays reported.
func TestLookalikeParentPrefersRunningGoroutine_Runner(t *testing.T) {
	type runKey struct{}
	run := func(t *testing.T, recovers bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		wait := func(ch <-chan struct{}) {
			select {
			case <-ch:
			case <-time.After(10 * time.Second):
			}
		}
		runOf := func(ctx context.Context) int {
			n, _ := ctx.Value(runKey{}).(int)
			return n
		}
		inner1Started := make(chan struct{})
		inner2Started := make(chan struct{})
		inner1Done := make(chan struct{})
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("child-1 failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		inner, err := agent.New(agent.Config{
			Name:      "inner",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					n := runOf(ctx)
					if n == 1 {
						close(inner1Started)
						wait(inner2Started)
						for ev, runErr := range par.Run(ctx) {
							if runErr != nil {
								continue
							}
							if !yield(ev, nil) {
								return
							}
						}
						return
					}
					close(inner2Started)
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "inner-2 progress")
					if !yield(ev, nil) {
						return
					}
					wait(inner1Done)
					ev = session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "inner"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "inner-2 answer")
					yield(ev, nil)
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runOf(cctx) != 1 || !recovers {
					return nil, nil
				}
				return textContent("model", "inner-1 callback recovery"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(inner): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{inner},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					// The first activation never waits for root to take its
					// events.
					first := make(chan *session.Event, 16)
					go func() {
						defer close(inner1Done)
						defer close(first)
						runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, 1))
						for ev, runErr := range inner.Run(runCtx) {
							if runErr == nil && ev != nil {
								first <- ev
							}
						}
					}()
					wait(inner1Started)
					runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, 2))
					for ev, runErr := range inner.Run(runCtx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						if isTextContent(ev.Content, "inner-2 progress") {
							// Deliver the child's event before the second
							// activation goes on.
							select {
							case cev, ok := <-first:
								if ok && !yield(cev, nil) {
									return
								}
							case <-time.After(10 * time.Second):
							}
						}
						if !yield(ev, nil) {
							return
						}
					}
					for ev := range first {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-lookalike-running-goroutine", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		// The first activation of inner starts first, so its AGENT_STARTING
		// row comes first, and the parallel agent runs under it.
		var innerSpans, parParents []string
		for _, r := range rowsOfType(rows, EventAgentStarting) {
			switch r["agent"] {
			case "inner":
				innerSpans = append(innerSpans, fmt.Sprint(r["span_id"]))
			case "par":
				parParents = append(parParents, fmt.Sprint(r["parent_span_id"]))
			}
		}
		if len(innerSpans) != 2 || len(parParents) != 1 || parParents[0] != innerSpans[0] {
			t.Errorf("par runs under %q, inner activations are %q; want the first inner", parParents, innerSpans)
		}
		return rows
	}
	t.Run("unrecovered_failure_of_the_goroutine's_activation_stays_reported", func(t *testing.T) {
		rows := run(t, false)
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "child-1 failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want the child's failure", msg)
		}
	})
	t.Run("recovery_by_the_goroutine's_activation_completes", func(t *testing.T) {
		rows := run(t, true)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})
}

// isTextContent reports whether c has a part with text.
func isTextContent(c *genai.Content, text string) bool {
	if c == nil {
		return false
	}
	for _, part := range c.Parts {
		if part.Text == text {
			return true
		}
	}
	return false
}

// TestCrossBranchRetiredPeerEventKeepsSiblingFailure_Runner runs one custom
// agent, worker, under both branches of the parallel agent top_par. On b2,
// worker drains the failure of child_b2 under b2_par, recovers only when the
// case says so, and stays open until b1 delivered its events. On b1, worker
// creates an event once that failure was recorded, either from its
// AfterAgentCallback or in its run, and b1 holds the event until worker
// retired, so the event arrives while only b2's worker is open. The event's
// branch is b1's, or empty, as session.NewEvent leaves it. It is not b2's
// worker's: a sibling branch's event never is, and an event without a branch
// could be either activation's. So it neither clears the failure that b2's
// worker swallowed nor is placed under b2's worker, as with another agent on
// b1, while a recovery that b2's worker creates on its own branch still
// clears it.
func TestCrossBranchRetiredPeerEventKeepsSiblingFailure_Runner(t *testing.T) {
	type config struct {
		// b1Worker is the agent that b1 runs: worker, or another one.
		b1Worker string
		// afterAgent makes b1's agent create its event from its
		// AfterAgentCallback rather than in its run.
		afterAgent bool
		// noBranch leaves the branch of b1's run event empty.
		noBranch bool
		// b2Recovers makes b2's worker yield a recovery on its own branch.
		b2Recovers bool
	}
	run := func(t *testing.T, c config) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		wait := func(ch <-chan struct{}) {
			select {
			case <-ch:
			case <-time.After(10 * time.Second):
			}
		}
		b2Failed := make(chan struct{})
		b1Delivered := make(chan struct{})
		var failOnce, deliverOnce sync.Once
		failed := func() { failOnce.Do(func() { close(b2Failed) }) }
		delivered := func() { deliverOnce.Do(func() { close(b1Delivered) }) }
		defer failed()
		defer delivered()
		childB2, err := agent.New(agent.Config{
			Name: "child_b2",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child_b2"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child_b2 partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("b2 child failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child_b2): %v", err)
		}
		b2Par := newParallelAgent(t, "b2_par", childB2)
		onB2 := func(ctx context.Context) bool {
			ic, _ := ctx.(interface{ Branch() string })
			return ic != nil && strings.HasPrefix(ic.Branch(), "top_par.b2")
		}
		newWorker := func(name string, subAgents ...agent.Agent) agent.Agent {
			w, err := agent.New(agent.Config{
				Name:      name,
				SubAgents: subAgents,
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						if onB2(ctx) {
							drained := false
							for ev, runErr := range b2Par.Run(ctx) {
								if runErr != nil {
									drained = true
									continue
								}
								if !yield(ev, nil) {
									return
								}
							}
							// b2_par retired, which recorded the failure.
							if drained {
								failed()
							}
							if c.b2Recovers {
								ev := session.NewEvent(ctx, ctx.InvocationID())
								ev.Author = name
								ev.Branch = ctx.Branch()
								ev.Content = textContent("model", "b2 worker recovery")
								if !yield(ev, nil) {
									return
								}
							}
							wait(b1Delivered)
							return
						}
						wait(b2Failed)
						if c.afterAgent {
							return
						}
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = name
						if !c.noBranch {
							ev.Branch = ctx.Branch()
						}
						ev.Content = textContent("model", "b1 worker answer")
						yield(ev, nil)
					}
				},
				AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
					if !c.afterAgent || onB2(cctx) {
						return nil, nil
					}
					return textContent("model", "b1 worker callback"), nil
				}},
			})
			if err != nil {
				t.Fatalf("agent.New(%s): %v", name, err)
			}
			return w
		}
		// Both branches run worker itself, so it is a sub-agent of neither
		// in the tree, whose names are unique.
		worker := newWorker("worker", b2Par)
		b1Worker := worker
		if c.b1Worker != "worker" {
			b1Worker = newWorker(c.b1Worker)
		}
		b1, err := agent.New(agent.Config{
			Name: "b1",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					defer delivered()
					var held []*session.Event
					for ev, runErr := range b1Worker.Run(ctx) {
						if runErr != nil {
							yield(nil, runErr)
							return
						}
						held = append(held, ev)
					}
					for _, ev := range held {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(b1): %v", err)
		}
		b2, err := agent.New(agent.Config{
			Name: "b2",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, runErr := range worker.Run(ctx) {
						if !yield(ev, runErr) || runErr != nil {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(b2): %v", err)
		}
		top := newParallelAgent(t, "top_par", b1, b2)
		for _, runErr := range newTestRunner(t, top, p).Run(t.Context(), "u1", "s-cross-branch-peer", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		var b2WorkerSpan string
		for _, r := range rowsOfType(rows, EventAgentStarting) {
			if r["agent"] == "b2_par" {
				b2WorkerSpan, _ = r["parent_span_id"].(string)
			}
		}
		if b2WorkerSpan == "" {
			t.Fatalf("no AGENT_STARTING row of b2_par; event types=%v", eventTypes(rows))
		}
		sawB1 := false
		for _, r := range rowsOfType(rows, EventAgentResponse) {
			if !strings.Contains(fmt.Sprint(r["content"]), "b1 worker") {
				continue
			}
			sawB1 = true
			if r["span_id"] == b2WorkerSpan {
				t.Errorf("b1's event is placed under b2's worker (span %q)", b2WorkerSpan)
			}
		}
		if !sawB1 {
			t.Errorf("no AGENT_RESPONSE row of b1's event; event types=%v", eventTypes(rows))
		}
		return rows
	}
	wantFailed := func(t *testing.T, rows []map[string]any) {
		t.Helper()
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "b2 child failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want b2's child failure", msg)
		}
	}
	for _, tc := range []struct {
		name string
		c    config
	}{
		{"after-agent_event", config{afterAgent: true}},
		{"held_run_event", config{}},
		{"held_run_event_without_branch", config{noBranch: true}},
	} {
		t.Run("other_agent_on_b1/"+tc.name+"_keeps_the_failure", func(t *testing.T) {
			c := tc.c
			c.b1Worker = "other_worker"
			wantFailed(t, run(t, c))
		})
		t.Run("worker_on_b1/"+tc.name+"_keeps_the_failure", func(t *testing.T) {
			c := tc.c
			c.b1Worker = "worker"
			wantFailed(t, run(t, c))
		})
		t.Run("worker_on_b1/"+tc.name+"_and_b2_recovery_completes", func(t *testing.T) {
			c := tc.c
			c.b1Worker = "worker"
			c.b2Recovers = true
			rows := run(t, c)
			if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
				t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
			}
			_ = onlyRowOfType(t, rows, EventInvocationCompleted)
		})
	}
}

// TestEventCreatedBeforeLaterActivationStarted_Runner runs two activations of
// worker one after the other under a custom root: a custom one, whose event
// root holds, and then an llmagent, whose AfterModelCallback waits until root
// delivered the held event. The held event was created before the second
// activation started, so it is not the second activation's although it is
// the only open activation of worker: it is not placed under it and does not
// take the model event that the second activation awaits. The second
// activation's own events are still its own, and the run completes.
func TestEventCreatedBeforeLaterActivationStarted_Runner(t *testing.T) {
	tm := newTraceManager()
	p, takeRows := newQueuedRowsPlugin(t, tm)
	tm.selfPlugin = weak.Make(p)
	paused := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	inner1, err := agent.New(agent.Config{
		Name: "worker",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "worker"
				ev.Branch = ctx.Branch()
				ev.Content = textContent("model", "worker-1 held answer")
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(worker): %v", err)
	}
	// The second activation is another agent named worker that root runs
	// itself, so it is not a sub-agent in the tree, whose names are unique.
	inner2, err := llmagent.New(llmagent.Config{
		Name: "worker",
		Model: &funcStreamLLM{fn: func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
			yield(&model.LLMResponse{Content: textContent("model", "worker-2 answer"), TurnComplete: true}, nil)
		}},
		AfterModelCallbacks: []llmagent.AfterModelCallback{func(ctx agent.Context, res *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			if res != nil && !res.Partial {
				close(paused)
				select {
				case <-release:
				case <-ctx.Done():
				case <-time.After(10 * time.Second):
				}
			}
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(worker): %v", err)
	}
	awaiting := -1
	root, err := agent.New(agent.Config{
		Name:      "root",
		SubAgents: []agent.Agent{inner1},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				var held []*session.Event
				for ev, runErr := range inner1.Run(ctx) {
					if runErr != nil {
						yield(nil, runErr)
						return
					}
					held = append(held, ev)
				}
				type item struct {
					ev  *session.Event
					err error
					ack chan struct{}
				}
				items := make(chan item)
				done := make(chan struct{})
				defer close(done)
				go func() {
					defer close(items)
					for ev, runErr := range inner2.Run(ctx) {
						ack := make(chan struct{})
						select {
						case items <- item{ev: ev, err: runErr, ack: ack}:
						case <-done:
							return
						}
						select {
						case <-ack:
						case <-done:
							return
						}
					}
				}()
				select {
				case <-paused:
				case <-time.After(10 * time.Second):
					yield(nil, errors.New("the second activation did not pause"))
					return
				}
				for _, ev := range held {
					if !yield(ev, nil) {
						return
					}
				}
				tm.mu.Lock()
				inv := tm.invocations[ctx.InvocationID()]
				tm.mu.Unlock()
				if inv != nil {
					inv.mu.Lock()
					for _, rec := range inv.agentStack[1:] {
						if rec.name == "worker" {
							awaiting = rec.awaitingModelEvent
						}
					}
					inv.mu.Unlock()
				}
				releaseOnce.Do(func() { close(release) })
				for it := range items {
					ok := yield(it.ev, it.err)
					close(it.ack)
					if !ok {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New(root): %v", err)
	}
	for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-event-before-later-activation", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error = %v, want nil", runErr)
		}
	}
	rows := takeRows()
	if awaiting != 1 {
		t.Errorf("awaitingModelEvent of the second activation once the held event was delivered = %d, want 1", awaiting)
	}
	var workerSpans []string
	for _, r := range rowsOfType(rows, EventAgentStarting) {
		if r["agent"] == "worker" {
			workerSpans = append(workerSpans, fmt.Sprint(r["span_id"]))
		}
	}
	if len(workerSpans) != 2 {
		t.Fatalf("got %d AGENT_STARTING rows of worker, want 2; event types=%v", len(workerSpans), eventTypes(rows))
	}
	placed := map[string]string{}
	for _, r := range rowsOfType(rows, EventAgentResponse) {
		for _, text := range []string{"worker-1 held answer", "worker-2 answer"} {
			if strings.Contains(fmt.Sprint(r["content"]), text) {
				placed[text] = fmt.Sprint(r["span_id"])
			}
		}
	}
	if got := placed["worker-1 held answer"]; got == "" || got == workerSpans[1] {
		t.Errorf("held event of the first activation placed at %q, want a place other than the second activation %q", got, workerSpans[1])
	}
	if got := placed["worker-2 answer"]; got != workerSpans[1] {
		t.Errorf("answer of the second activation placed at %q, want its span %q", got, workerSpans[1])
	}
	if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
		t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
	}
	_ = onlyRowOfType(t, rows, EventInvocationCompleted)
}

// TestLookalikeSubAgentOnHelperGoroutine_Runner runs two concurrent
// activations of the custom agent worker under a custom root. The first runs
// par -> child on a goroutine and drains the child's failure; it recovers in
// its AfterAgentCallback only when the case says so. The second answers once
// the first returned. When the first starts par on a goroutine of its own,
// par runs under it: its recovery clears the failure, and the second's answer
// does not. When a helper goroutine of the first starts par's goroutine, the
// runtime names only the helper as par's creator, which runs no callback,
// so par runs under root, which encloses both activations, rather than under
// the second activation, which started last. The second's answer then does
// not clear the failure, and the first one's recovery does not either, as
// Config states.
func TestLookalikeSubAgentOnHelperGoroutine_Runner(t *testing.T) {
	type runKey struct{}
	run := func(t *testing.T, hops int, recovers bool) []map[string]any {
		t.Helper()
		p, requests, _ := setupTestPlugin(t)
		wait := func(ch <-chan struct{}) {
			select {
			case <-ch:
			case <-time.After(10 * time.Second):
			}
		}
		runOf := func(ctx context.Context) int {
			n, _ := ctx.Value(runKey{}).(int)
			return n
		}
		worker1Started := make(chan struct{})
		worker2Started := make(chan struct{})
		worker1Done := make(chan struct{})
		child, err := agent.New(agent.Config{
			Name: "child",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "child"
					ev.Branch = ctx.Branch()
					ev.Content = textContent("model", "child partial")
					if !yield(ev, nil) {
						return
					}
					yield(nil, errors.New("child-1 failure"))
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(child): %v", err)
		}
		par := newParallelAgent(t, "par", child)
		worker, err := agent.New(agent.Config{
			Name:      "worker",
			SubAgents: []agent.Agent{par},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					if runOf(ctx) == 2 {
						close(worker2Started)
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "worker"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "worker-2 progress")
						if !yield(ev, nil) {
							return
						}
						wait(worker1Done)
						ev = session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "worker"
						ev.Branch = ctx.Branch()
						ev.Content = textContent("model", "worker-2 answer")
						yield(ev, nil)
						return
					}
					close(worker1Started)
					wait(worker2Started)
					type item struct {
						ev  *session.Event
						err error
					}
					items := make(chan item)
					runPar := func() {
						defer close(items)
						for ev, runErr := range par.Run(ctx) {
							items <- item{ev, runErr}
						}
					}
					if hops == 1 {
						go runPar()
					} else {
						go func() { go runPar() }()
					}
					// The failure is drained, and so is the rest once the
					// caller stops, so that par's goroutine ends.
					stopped := false
					for it := range items {
						if it.err == nil && !stopped && !yield(it.ev, nil) {
							stopped = true
						}
					}
				}
			},
			AfterAgentCallbacks: []agent.AfterAgentCallback{func(cctx agent.Context) (*genai.Content, error) {
				if runOf(cctx) != 1 || !recovers {
					return nil, nil
				}
				return textContent("model", "worker-1 callback recovery"), nil
			}},
		})
		if err != nil {
			t.Fatalf("agent.New(worker): %v", err)
		}
		root, err := agent.New(agent.Config{
			Name:      "root",
			SubAgents: []agent.Agent{worker},
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					type item struct {
						ev  *session.Event
						err error
						ack chan struct{}
					}
					items := make(chan item)
					done := make(chan struct{})
					defer close(done)
					var wg sync.WaitGroup
					wg.Add(2)
					go func() {
						wg.Wait()
						close(items)
					}()
					for n := 1; n <= 2; n++ {
						if n == 2 {
							// The second activation starts last.
							wait(worker1Started)
						}
						go func() {
							defer wg.Done()
							if n == 1 {
								defer close(worker1Done)
							}
							runCtx := ctx.WithContext(context.WithValue(ctx, runKey{}, n))
							for ev, runErr := range worker.Run(runCtx) {
								ack := make(chan struct{})
								select {
								case items <- item{ev: ev, err: runErr, ack: ack}:
								case <-done:
									return
								}
								select {
								case <-ack:
								case <-done:
									return
								}
							}
						}()
					}
					for it := range items {
						ok := yield(it.ev, it.err)
						close(it.ack)
						if !ok {
							return
						}
					}
				}
			},
		})
		if err != nil {
			t.Fatalf("agent.New(root): %v", err)
		}
		for _, runErr := range newTestRunner(t, root, p).Run(t.Context(), "u1", "s-lookalike-helper-goroutine", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error = %v, want nil", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		var rootSpan, parParent string
		var workerSpans []string
		for _, r := range rowsOfType(rows, EventAgentStarting) {
			switch r["agent"] {
			case "root":
				rootSpan = fmt.Sprint(r["span_id"])
			case "worker":
				workerSpans = append(workerSpans, fmt.Sprint(r["span_id"]))
			case "par":
				parParent = fmt.Sprint(r["parent_span_id"])
			}
		}
		want := rootSpan
		if hops == 1 {
			want = workerSpans[0]
		}
		if len(workerSpans) != 2 || parParent != want {
			t.Errorf("par runs under %q, root is %q and worker activations are %q; want %q", parParent, rootSpan, workerSpans, want)
		}
		return rows
	}
	wantFailed := func(t *testing.T, rows []map[string]any) {
		t.Helper()
		if n := len(rowsOfType(rows, EventInvocationCompleted)); n != 0 {
			t.Fatalf("got %d INVOCATION_COMPLETED rows, want 0; event types=%v", n, eventTypes(rows))
		}
		invErr := onlyRowOfType(t, rows, EventInvocationError)
		if msg, _ := invErr["error_message"].(string); !strings.Contains(msg, "child-1 failure") {
			t.Errorf("INVOCATION_ERROR error_message = %q, want the child's failure", msg)
		}
	}
	t.Run("own_goroutine/recovery_completes", func(t *testing.T) {
		rows := run(t, 1, true)
		if n := len(rowsOfType(rows, EventInvocationError)); n != 0 {
			t.Fatalf("got %d INVOCATION_ERROR rows, want 0; event types=%v", n, eventTypes(rows))
		}
		_ = onlyRowOfType(t, rows, EventInvocationCompleted)
	})
	t.Run("own_goroutine/answer_of_the_other_activation_keeps_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, 1, false))
	})
	t.Run("helper_goroutine/answer_of_the_other_activation_keeps_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, 2, false))
	})
	t.Run("helper_goroutine/recovery_keeps_the_failure", func(t *testing.T) {
		wantFailed(t, run(t, 2, true))
	})
}
