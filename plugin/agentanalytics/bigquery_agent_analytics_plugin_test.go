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
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/loggingplugin"
	"google.golang.org/adk/v2/plugin/retryandreflect"
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
	call2Ctx := withFunctionCall(mCtx, "call-2")
	errs.add(p.BeforeToolCallback()(call2Ctx, nil, map[string]any{"q": "2"}))
	errs.add(p.AfterToolCallback()(call2Ctx, nil, map[string]any{"q": "2"}, map[string]any{"ok": true}, nil))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	if diff := cmp.Diff([]string{EventToolStarting, EventToolError, EventToolStarting, EventToolCompleted, EventInvocationCompleted}, eventTypes(rows)); diff != "" {
		t.Fatalf("event types mismatch (-want +got):\n%s", diff)
	}
	startRow := rowsOfType(rows, EventToolStarting)[0]
	errRow := onlyRowOfType(t, rows, EventToolError)
	if errRow["span_id"] != startRow["span_id"] {
		t.Errorf("errRow span_id = %v, want equal to TOOL_STARTING span_id %v",
			errRow["span_id"], startRow["span_id"])
	}
	start2Row := rowsOfType(rows, EventToolStarting)[1]
	doneRow := onlyRowOfType(t, rows, EventToolCompleted)
	if doneRow["span_id"] != start2Row["span_id"] {
		t.Errorf("doneRow span_id = %v, want equal to second TOOL_STARTING span_id %v",
			doneRow["span_id"], start2Row["span_id"])
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

func runFailingToolWithPlugins(
	t *testing.T,
	toolName string,
	finalResponseTools []string,
	agentOnToolError llmagent.OnToolErrorCallback,
	pluginsFn func(analyticsPlugin *baseplugin.Plugin) []*baseplugin.Plugin,
) []map[string]any {
	t.Helper()
	p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
		c.FinalResponseToolNames = finalResponseTools
	})
	failingTool, err := functiontool.New(functiontool.Config{
		Name:        toolName,
		Description: "Always fails",
	}, func(_ agent.Context, _ map[string]any) (map[string]any, error) {
		return nil, errors.New("boom")
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	llm := &scriptedLLM{responses: []*model.LLMResponse{
		{Content: genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromFunctionCall(toolName, map[string]any{"x": "1"}),
		}, genai.RoleModel)},
		{Content: textContent("model", "done"), TurnComplete: true},
	}}
	var onToolErrs []llmagent.OnToolErrorCallback
	if agentOnToolError != nil {
		onToolErrs = append(onToolErrs, agentOnToolError)
	}
	bot, err := llmagent.New(llmagent.Config{
		Name:                 "bot",
		Model:                llm,
		Tools:                []tool.Tool{failingTool},
		OnToolErrorCallbacks: onToolErrs,
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	r := newTestRunner(t, bot, pluginsFn(p)...)
	for _, runErr := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("r.Run unexpected error: %v", runErr)
		}
	}
	return collectRows(t, requests, untilInvocationEnd)
}

func TestPlugin_RecoveredToolError_AgentCallback(t *testing.T) {
	rows := runFailingToolWithPlugins(t, "lookup", nil,
		func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
			return map[string]any{"answer": "recovered"}, nil
		},
		func(p *baseplugin.Plugin) []*baseplugin.Plugin {
			return []*baseplugin.Plugin{p}
		},
	)
	if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
		t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
	}
	startRow := onlyRowOfType(t, rows, EventToolStarting)
	errRow := onlyRowOfType(t, rows, EventToolError)
	if got, want := placeOf(errRow), placeOf(startRow); got != want {
		t.Errorf("TOOL_ERROR span = %+v, want %+v", got, want)
	}
	mustLatency(t, errRow)
	if errRow["status"] != "ERROR" {
		t.Errorf("TOOL_ERROR status = %v, want ERROR", errRow["status"])
	}
	if errRow["error_message"] != "boom" {
		t.Errorf("TOOL_ERROR error_message = %v, want %q", errRow["error_message"], "boom")
	}
}

func TestPlugin_RecoveredToolError_LaterPlugin(t *testing.T) {
	t.Run("custom plugin after analytics", func(t *testing.T) {
		recoverPlugin, err := baseplugin.New(baseplugin.Config{
			Name: "recover",
			OnToolErrorCallback: func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
				return map[string]any{"answer": "recovered"}, nil
			},
		})
		if err != nil {
			t.Fatalf("baseplugin.New: %v", err)
		}
		rows := runFailingToolWithPlugins(t, "lookup", nil, nil, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
			return []*baseplugin.Plugin{p, recoverPlugin}
		})
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
		}
	})

	t.Run("retryandreflect after analytics", func(t *testing.T) {
		rrPlugin, err := retryandreflect.New()
		if err != nil {
			t.Fatalf("retryandreflect.New: %v", err)
		}
		rows := runFailingToolWithPlugins(t, "lookup", nil, nil, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
			return []*baseplugin.Plugin{p, rrPlugin}
		})
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
		}
	})
}

func TestPlugin_RecoveredToolError_RetryAndReflectBeforeAnalytics(t *testing.T) {
	t.Run("normal retry guidance", func(t *testing.T) {
		rrPlugin, err := retryandreflect.New()
		if err != nil {
			t.Fatalf("retryandreflect.New: %v", err)
		}
		rows := runFailingToolWithPlugins(t, "lookup", nil, nil, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
			return []*baseplugin.Plugin{rrPlugin, p}
		})
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		startRow := onlyRowOfType(t, rows, EventToolStarting)
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(startRow); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want %+v", got, want)
		}
		mustLatency(t, errRow)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
		}
		checkJSON(t, "TOOL_ERROR content", jsonObjectColumn(t, errRow, "content"),
			`{"tool":"lookup","args":{"x":"1"},"tool_origin":"LOCAL"}`)
	})

	t.Run("WithMaxRetries(0) exceeded-retries guidance", func(t *testing.T) {
		rrPlugin, err := retryandreflect.New(retryandreflect.WithMaxRetries(0))
		if err != nil {
			t.Fatalf("retryandreflect.New: %v", err)
		}
		rows := runFailingToolWithPlugins(t, "lookup", nil, nil, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
			return []*baseplugin.Plugin{rrPlugin, p}
		})
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
		}
	})

	t.Run("WithErrorIfRetryExceeded(true) passthrough", func(t *testing.T) {
		rrPlugin, err := retryandreflect.New(
			retryandreflect.WithMaxRetries(0),
			retryandreflect.WithErrorIfRetryExceeded(true),
		)
		if err != nil {
			t.Fatalf("retryandreflect.New: %v", err)
		}
		p, requests, _ := setupTestPlugin(t)
		failingTool, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Always fails",
		}, func(_ agent.Context, _ map[string]any) (map[string]any, error) {
			return nil, errors.New("boom")
		})
		if err != nil {
			t.Fatalf("functiontool.New: %v", err)
		}
		llm := &scriptedLLM{responses: []*model.LLMResponse{
			{Content: genai.NewContentFromParts([]*genai.Part{
				genai.NewPartFromFunctionCall("lookup", map[string]any{"x": "1"}),
			}, genai.RoleModel)},
			{Content: textContent("model", "done"), TurnComplete: true},
		}}
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Model: llm,
			Tools: []tool.Tool{failingTool},
		})
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		r := newTestRunner(t, bot, rrPlugin, p)
		var fnResp *genai.FunctionResponse
		for ev, e := range r.Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
			if e != nil {
				t.Fatalf("r.Run yielded unexpected error: %v", e)
			}
			if ev != nil && ev.Content != nil {
				for _, part := range ev.Content.Parts {
					if part != nil && part.FunctionResponse != nil {
						fnResp = part.FunctionResponse
					}
				}
			}
		}
		if fnResp == nil {
			t.Fatal("missing FunctionResponse event")
		}
		if diff := cmp.Diff(map[string]any{"error": "boom"}, fnResp.Response); diff != "" {
			t.Fatalf("FunctionResponse.Response mismatch (-want +got):\n%s", diff)
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; event types: %v", n, eventTypes(rows))
		}
		startRow := onlyRowOfType(t, rows, EventAgentStarting)
		errRow := onlyRowOfType(t, rows, EventToolError)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
		}
		if errRow["trace_id"] != startRow["trace_id"] {
			t.Errorf("TOOL_ERROR trace_id = %v, want %v", errRow["trace_id"], startRow["trace_id"])
		}
		mustLatency(t, errRow)
	})

	t.Run("error_type fallback when error_details empty or non-string", func(t *testing.T) {
		for _, tc := range []struct {
			name         string
			errorDetails any
		}{
			{"empty string", ""},
			{"non-string int", 123},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				toolCtx := withFunctionCall(mCtx, "call-err-type")
				tl := &fakeTool{name: "lookup"}
				args := map[string]any{"x": "1"}
				ans := map[string]any{
					"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
					"error_details": tc.errorDetails,
					"error_type":    "*errors.errorString",
				}
				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
				errs.add(p.AfterToolCallback()(toolCtx, tl, args, ans, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
					t.Fatalf("got %d TOOL_COMPLETED rows, want 0", n)
				}
				errRow := onlyRowOfType(t, rows, EventToolError)
				if errRow["error_message"] != "*errors.errorString" {
					t.Errorf("TOOL_ERROR error_message = %v, want %q", errRow["error_message"], "*errors.errorString")
				}
			})
		}
	})

	t.Run("crafted non-string error_details and error_type uses fixed fallback", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		toolCtx := withFunctionCall(mCtx, "call-fallback")
		tl := &fakeTool{name: "lookup"}
		args := map[string]any{"x": "1"}
		ans := map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_details": 42,
			"error_type":    []string{"not-a-string"},
		}
		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
		errs.add(p.AfterToolCallback()(toolCtx, tl, args, ans, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0", n)
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		wantMsg := "Tool raised an error that ReflectAndRetryToolPlugin handled."
		if errRow["error_message"] != wantMsg {
			t.Errorf("TOOL_ERROR error_message = %v, want %q", errRow["error_message"], wantMsg)
		}
	})
}

func assertNoFinalResponseFromTool(t *testing.T, rows []map[string]any, toolName string) {
	t.Helper()
	for _, row := range rowsOfType(rows, EventAgentResponse) {
		attrs := jsonObjectColumn(t, row, "attributes")
		if attrs["source_tool"] == toolName {
			t.Errorf("got unexpected AGENT_RESPONSE with source_tool=%q: %+v", toolName, row)
		}
	}
}

func TestPlugin_RecoveredToolError_FinalResponseToolSuppressed(t *testing.T) {
	t.Run("agent OnToolErrorCallback", func(t *testing.T) {
		rows := runFailingToolWithPlugins(t, "final_answer", []string{"final_answer"},
			func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
				return map[string]any{"answer": "recovered"}, nil
			},
			func(p *baseplugin.Plugin) []*baseplugin.Plugin {
				return []*baseplugin.Plugin{p}
			},
		)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
		}
		assertNoFinalResponseFromTool(t, rows, "final_answer")
		onlyRowOfType(t, rows, EventToolError)
	})

	t.Run("retryandreflect before analytics", func(t *testing.T) {
		rrPlugin, err := retryandreflect.New()
		if err != nil {
			t.Fatalf("retryandreflect.New: %v", err)
		}
		rows := runFailingToolWithPlugins(t, "final_answer", []string{"final_answer"}, nil,
			func(p *baseplugin.Plugin) []*baseplugin.Plugin {
				return []*baseplugin.Plugin{rrPlugin, p}
			},
		)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
		}
		assertNoFinalResponseFromTool(t, rows, "final_answer")
		onlyRowOfType(t, rows, EventToolError)
	})
}

type taskActivationContext struct {
	*mockInvocationContext
	ic *runningInvocation
}

func (c *taskActivationContext) WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext {
	return c.ic
}

func TestPlugin_RecoveredToolError_FinishTaskDoesNotNoteTaskFinished(t *testing.T) {
	for _, tc := range []struct {
		name            string
		emitOnToolError bool
		afterToolResult map[string]any
	}{
		{
			name:            "closedByError via OnToolErrorCallback",
			emitOnToolError: true,
			afterToolResult: map[string]any{"ok": false},
		},
		{
			name:            "retryandreflect before analytics",
			emitOnToolError: false,
			afterToolResult: map[string]any{
				"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
				"error_details": "finish failed",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "test-agent",
				Mode:  llmagent.ModeTask,
				Model: &scriptedLLM{},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			ic1 := &runningInvocation{running: taskAgent}
			ic2 := &runningInvocation{running: taskAgent}
			act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
			act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
			toolCtx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish"), ic: ic1}
			finish := &fakeTool{name: finishTaskToolName}
			args := map[string]any{"summary": "draft"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(act1))
			errs.add(p.BeforeToolCallback()(toolCtx, finish, args))
			if tc.emitOnToolError {
				errs.add(p.OnToolErrorCallback()(toolCtx, finish, args, errors.New("finish failed")))
			}
			errs.add(p.AfterToolCallback()(toolCtx, finish, args, tc.afterToolResult, nil))
			// Push a second task activation before the runner-context finish_task
			// FunctionResponse event arrives. If noteTaskFinished had been called
			// during afterTool, finishedTasks[callID] would bind to act1 and close
			// act1 first; without noteTaskFinished, popTaskAgentSpan retires top-of-stack act2.
			errs.add(p.BeforeAgentCallback()(act2))
			runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
			errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
				Author: "test-agent",
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromParts([]*genai.Part{{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-finish",
							Name:     finishTaskToolName,
							Response: map[string]any{"result": "Task completed."},
						},
					}}, genai.RoleUser),
				},
			}))
			errs.add(p.AfterAgentCallback()(act1))
			errs.check(t)
			p.AfterRunCallback()(act1)

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
				t.Fatalf("got %d TOOL_COMPLETED rows, want 0", n)
			}
			onlyRowOfType(t, rows, EventToolError)
			starts := rowsOfType(rows, EventAgentStarting)
			completes := rowsOfType(rows, EventAgentCompleted)
			if len(starts) != 2 || len(completes) != 2 {
				t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED, want 2 each", len(starts), len(completes))
			}
			// First AGENT_COMPLETED must close act2 (top of stack), leaving act1 open until AfterAgentCallback(act1).
			if completes[0]["span_id"] != starts[1]["span_id"] {
				t.Errorf("first AGENT_COMPLETED span_id = %v, want act2 span_id %v (act1=%v)",
					completes[0]["span_id"], starts[1]["span_id"], starts[0]["span_id"])
			}
			if completes[1]["span_id"] != starts[0]["span_id"] {
				t.Errorf("second AGENT_COMPLETED span_id = %v, want act1 span_id %v",
					completes[1]["span_id"], starts[0]["span_id"])
			}
		})
	}

	t.Run("recovered finish_task with Task completed result does not pop later activation across runner delivery modes", func(t *testing.T) {
		for _, mode := range []struct {
			name        string
			usePluginRR bool
		}{
			{name: "OnToolErrorCallback", usePluginRR: false},
			{name: "retryandreflect", usePluginRR: true},
		} {
			for _, buffered := range []bool{false, true} {
				delivery := "unbuffered"
				if buffered {
					delivery = "buffered"
				}
				t.Run(mode.name+"/"+delivery, func(t *testing.T) {
					p, requests, _ := setupTestPlugin(t)
					var runCount atomic.Int32
					recoverPlugin, err := baseplugin.New(baseplugin.Config{
						Name: "recover_first_finish_task",
						AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
							if err == nil && tl != nil && tl.Name() == finishTaskToolName && runCount.Load() == 1 {
								if mode.usePluginRR {
									result["response_type"] = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
									result["error_details"] = "finish failed"
								} else {
									if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish failed")); cbErr != nil {
										return nil, cbErr
									}
								}
							}
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New: %v", err)
					}

					taskFinishResp := func() *model.LLMResponse {
						return &model.LLMResponse{
							Content: &genai.Content{
								Role: "model",
								Parts: []*genai.Part{{
									FunctionCall: &genai.FunctionCall{
										ID:   "call-0",
										Name: finishTaskToolName,
										Args: map[string]any{"result": "done"},
									},
								}},
							},
						}
					}
					taskModel := &scriptedLLM{
						responses: []*model.LLMResponse{taskFinishResp(), taskFinishResp()},
					}
					taskAgent, err := llmagent.New(llmagent.Config{
						Name:  "task",
						Mode:  llmagent.ModeTask,
						Model: taskModel,
					})
					if err != nil {
						t.Fatalf("llmagent.New(task): %v", err)
					}
					hasFinishTaskResponse := func(c *genai.Content) bool {
						if c == nil {
							return false
						}
						for _, part := range c.Parts {
							if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == finishTaskToolName {
								return true
							}
						}
						return false
					}
					root, err := agent.New(agent.Config{
						Name: "root",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								var held []*session.Event
								nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
								for range 2 {
									runCount.Add(1)
									for ev, err := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
										if err != nil {
											yield(nil, err)
											return
										}
										if buffered && hasFinishTaskResponse(ev.Content) {
											held = append(held, ev)
											continue
										}
										if !yield(ev, nil) {
											return
										}
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

					for _, runErr := range newTestRunner(t, root, recoverPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
						if runErr != nil {
							t.Fatalf("runner.Run error: %v", runErr)
						}
					}

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)
					if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
						t.Fatalf("got %d AGENT_ERROR rows, want 0; events: %v", got, eventTypes(rows))
					}

					starts := agentRowsOfType(rows, EventAgentStarting, "task")
					ends := agentRowsOfType(rows, EventAgentCompleted, "task")
					if len(starts) != 2 {
						t.Fatalf("got %d task AGENT_STARTING rows, want 2", len(starts))
					}
					if len(ends) != 1 {
						t.Fatalf("got %d task AGENT_COMPLETED rows, want 1; ends=%v", len(ends), ends)
					}
					if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
						t.Errorf("task AGENT_COMPLETED span_id = %v, want Activation 2 span_id %v (Activation 1 span_id = %v)", got, want, starts[0]["span_id"])
					}
					onlyRowOfType(t, rows, EventToolError)
					onlyRowOfType(t, rows, EventToolCompleted)
				})
			}
		}
	})
}

func TestPlugin_RecoveredToolError_ConfirmationOutcomesUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"required", tool.ErrConfirmationRequired},
		{"rejected", tool.ErrConfirmationRejected},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			toolCtx := withFunctionCall(mCtx, "call-confirm")
			tl := &fakeTool{name: "final_answer"}
			args := map[string]any{"answer": "draft"}
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
			errs.add(p.OnToolErrorCallback()(toolCtx, tl, args, tc.err))
			errs.add(p.AfterToolCallback()(toolCtx, tl, args, nil, tc.err))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolError)); n != 0 {
				t.Errorf("got %d TOOL_ERROR rows, want 0", n)
			}
			if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
				t.Errorf("got %d AGENT_RESPONSE rows, want 0", n)
			}
			compRow := onlyRowOfType(t, rows, EventToolCompleted)
			if compRow["status"] != "OK" {
				t.Errorf("TOOL_COMPLETED status = %v, want OK", compRow["status"])
			}
		})
	}

	t.Run("replacement confirmation error after onToolError emits no duplicate TOOL_COMPLETED", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		toolCtx := withFunctionCall(mCtx, "call-confirm-after-err")
		tl := &fakeTool{name: "lookup"}
		args := map[string]any{"q": "1"}
		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
		errs.add(p.OnToolErrorCallback()(toolCtx, tl, args, errors.New("boom")))
		errs.add(p.AfterToolCallback()(toolCtx, tl, args, nil, tool.ErrConfirmationRequired))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		want := []string{
			EventAgentStarting,
			EventToolStarting,
			EventToolError,
			EventAgentCompleted,
			EventInvocationCompleted,
		}
		if diff := cmp.Diff(want, eventTypes(rows)); diff != "" {
			t.Fatalf("event types mismatch (-want +got):\n%s", diff)
		}
	})
}

func drainAllRows(t *testing.T, ch <-chan *storagepb.AppendRowsRequest) []map[string]any {
	t.Helper()
	var rows []map[string]any
	for {
		select {
		case req := <-ch:
			rows = append(rows, decodeArrowRows(t, req)...)
		default:
			return rows
		}
	}
}

func TestPlugin_RecoveredToolError_LateStateParity(t *testing.T) {
	t.Run("closedByError via OnToolErrorCallback", func(t *testing.T) {
		p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		toolCtx := withFunctionCall(mCtx, "call-late")
		tl := &fakeTool{name: "final_answer"}
		args := map[string]any{"answer": "draft"}
		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
		errs.add(p.OnToolErrorCallback()(toolCtx, tl, args, errors.New("late boom")))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		// Late AfterToolCallback after AfterRun must not emit TOOL_COMPLETED or AGENT_RESPONSE.
		if _, err := p.AfterToolCallback()(toolCtx, tl, args, map[string]any{"answer": "recovered"}, nil); err != nil {
			t.Fatalf("late AfterToolCallback: %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close: %v", err)
		}

		rows := drainAllRows(t, requests)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0; events=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
			t.Errorf("got %d AGENT_RESPONSE rows, want 0; events=%v", n, eventTypes(rows))
		}
		onlyRowOfType(t, rows, EventToolError)
	})

	t.Run("retryandreflect before analytics", func(t *testing.T) {
		p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		toolCtx := withFunctionCall(mCtx, "call-late-rr")
		tl := &fakeTool{name: "final_answer"}
		args := map[string]any{"answer": "draft"}
		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(toolCtx, tl, args))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		// Late AfterToolCallback carrying retryandreflect payload after AfterRun must
		// emit TOOL_ERROR with TOOL_STARTING span + latency_ms and no TOOL_COMPLETED or AGENT_RESPONSE.
		ans := map[string]any{
			"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
			"error_details": "late retry boom",
		}
		if _, err := p.AfterToolCallback()(toolCtx, tl, args, ans, nil); err != nil {
			t.Fatalf("late AfterToolCallback: %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close: %v", err)
		}

		rows := drainAllRows(t, requests)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0; events=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
			t.Errorf("got %d AGENT_RESPONSE rows, want 0; events=%v", n, eventTypes(rows))
		}
		startRow := onlyRowOfType(t, rows, EventToolStarting)
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(startRow); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want %+v", got, want)
		}
		mustLatency(t, errRow)
		if errRow["error_message"] != "late retry boom" {
			t.Errorf("TOOL_ERROR error_message = %v, want %q", errRow["error_message"], "late retry boom")
		}
	})
}

func TestPlugin_RecoveredToolError_SequentialRetry(t *testing.T) {
	t.Run("distinct FunctionCall.IDs via runner", func(t *testing.T) {
		for _, mode := range []struct {
			name             string
			agentOnToolError llmagent.OnToolErrorCallback
			pluginsFn        func(analytics *baseplugin.Plugin) []*baseplugin.Plugin
		}{
			{
				name: "agent OnToolErrorCallback",
				agentOnToolError: func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
					return map[string]any{"retry": true}, nil
				},
				pluginsFn: func(p *baseplugin.Plugin) []*baseplugin.Plugin { return []*baseplugin.Plugin{p} },
			},
			{
				name: "retryandreflect after analytics",
				pluginsFn: func(p *baseplugin.Plugin) []*baseplugin.Plugin {
					rr, err := retryandreflect.New()
					if err != nil {
						t.Fatalf("retryandreflect.New: %v", err)
					}
					return []*baseplugin.Plugin{p, rr}
				},
			},
			{
				name: "retryandreflect before analytics",
				pluginsFn: func(p *baseplugin.Plugin) []*baseplugin.Plugin {
					rr, err := retryandreflect.New()
					if err != nil {
						t.Fatalf("retryandreflect.New: %v", err)
					}
					return []*baseplugin.Plugin{rr, p}
				},
			},
		} {
			t.Run(mode.name, func(t *testing.T) {
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				var calls atomic.Int32
				flakyTool, err := functiontool.New(functiontool.Config{
					Name:        "final_answer",
					Description: "Fails once then succeeds",
				}, func(_ agent.Context, args map[string]any) (map[string]any, error) {
					if calls.Add(1) == 1 {
						return nil, errors.New("boom")
					}
					return map[string]any{"accepted": true, "answer": args["answer"]}, nil
				})
				if err != nil {
					t.Fatalf("functiontool.New: %v", err)
				}
				llm := &scriptedLLM{responses: []*model.LLMResponse{
					{Content: genai.NewContentFromParts([]*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "call-1", Name: "final_answer", Args: map[string]any{"answer": "bad"}},
					}}, genai.RoleModel)},
					{Content: genai.NewContentFromParts([]*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "call-2", Name: "final_answer", Args: map[string]any{"answer": "good"}},
					}}, genai.RoleModel)},
					{Content: textContent("model", "done"), TurnComplete: true},
				}}
				var onToolErrs []llmagent.OnToolErrorCallback
				if mode.agentOnToolError != nil {
					onToolErrs = append(onToolErrs, mode.agentOnToolError)
				}
				bot, err := llmagent.New(llmagent.Config{
					Name:                 "bot",
					Model:                llm,
					Tools:                []tool.Tool{flakyTool},
					OnToolErrorCallbacks: onToolErrs,
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				for _, runErr := range newTestRunner(t, bot, mode.pluginsFn(p)...).Run(t.Context(), "u1", "s1", textContent("user", "hi"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("r.Run error: %v", runErr)
					}
				}

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)

				starts := rowsOfType(rows, EventToolStarting)
				if len(starts) != 2 {
					t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
				}
				if placeOf(starts[0]).span == placeOf(starts[1]).span {
					t.Errorf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
				}

				errRow := onlyRowOfType(t, rows, EventToolError)
				if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
					t.Errorf("TOOL_ERROR span = %+v, want Call 1 span %+v", got, want)
				}
				mustLatency(t, errRow)
				if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
					t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
				}
				checkJSONAt(t, jsonObjectColumn(t, errRow, "attributes"), `"call-1"`, "adk", "function_call_id")

				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
					t.Errorf("TOOL_COMPLETED span = %+v, want Call 2 span %+v", got, want)
				}
				mustLatency(t, compRow)
				if compRow["status"] != "OK" {
					t.Errorf("TOOL_COMPLETED status = %v, want OK", compRow["status"])
				}
				checkJSONAt(t, jsonObjectColumn(t, compRow, "attributes"), `"call-2"`, "adk", "function_call_id")
				checkJSON(t, "TOOL_COMPLETED content", jsonObjectColumn(t, compRow, "content"),
					`{"tool":"final_answer","result":{"accepted":true,"answer":"good"},"tool_origin":"LOCAL"}`)

				agentResps := rowsOfType(rows, EventAgentResponse)
				if len(agentResps) != 2 {
					t.Fatalf("got %d AGENT_RESPONSE rows, want 2 (Call 2 source_tool + final model text); events=%v", len(agentResps), eventTypes(rows))
				}
				checkJSONAt(t, jsonObjectColumn(t, agentResps[0], "attributes"), `"final_answer"`, "source_tool")
				checkJSONAt(t, jsonObjectColumn(t, agentResps[0], "attributes"), `"call-2"`, "adk", "function_call_id")
				checkJSON(t, "Call 2 final response content", jsonObjectColumn(t, agentResps[0], "content"), `{"response":{"answer":"good"}}`)
				checkJSON(t, "final model text content", jsonObjectColumn(t, agentResps[1], "content"), `{"response":"text: 'done'"}`)
			})
		}
	})

	t.Run("distinct FunctionCall.IDs finish_task recovery then retry notes second call", func(t *testing.T) {
		for _, tc := range []struct {
			name            string
			emitOnToolError bool
			call1Result     map[string]any
		}{
			{
				name:            "closedByError via OnToolErrorCallback",
				emitOnToolError: true,
				call1Result:     map[string]any{"ok": false},
			},
			{
				name:            "retryandreflect before analytics",
				emitOnToolError: false,
				call1Result: map[string]any{
					"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
					"error_details": "finish retryable",
				},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				ic1 := &runningInvocation{running: taskAgent}
				ic2 := &runningInvocation{running: taskAgent}
				act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
				act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
				call1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish-1"), ic: ic1}
				call2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish-2"), ic: ic1}
				finish := &fakeTool{name: finishTaskToolName}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(act1))
				errs.add(p.BeforeToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}))
				if tc.emitOnToolError {
					errs.add(p.OnToolErrorCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, errors.New("finish retryable")))
				}
				errs.add(p.AfterToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, tc.call1Result, nil))
				errs.add(p.BeforeToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}))
				errs.add(p.AfterToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}, map[string]any{"result": "Task completed."}, nil))
				errs.add(p.BeforeAgentCallback()(act2))
				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
				errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
					Author: "test-agent",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{{
							FunctionResponse: &genai.FunctionResponse{
								ID:       "call-finish-2",
								Name:     finishTaskToolName,
								Response: map[string]any{"result": "Task completed."},
							},
						}}, genai.RoleUser),
					},
				}))
				errs.add(p.AfterAgentCallback()(act2))
				errs.check(t)
				p.AfterRunCallback()(act2)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				errRow := onlyRowOfType(t, rows, EventToolError)
				checkJSONAt(t, jsonObjectColumn(t, errRow, "attributes"), `"call-finish-1"`, "adk", "function_call_id")
				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				checkJSONAt(t, jsonObjectColumn(t, compRow, "attributes"), `"call-finish-2"`, "adk", "function_call_id")
				starts := rowsOfType(rows, EventAgentStarting)
				completes := rowsOfType(rows, EventAgentCompleted)
				if len(starts) != 2 || len(completes) != 2 {
					t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED, want 2 each", len(starts), len(completes))
				}
				// Call 2 called noteTaskFinished("call-finish-2"), so FunctionResponse closes act1 before act2.
				if completes[0]["span_id"] != starts[0]["span_id"] {
					t.Errorf("first AGENT_COMPLETED span_id = %v, want act1 span_id %v (act2=%v)",
						completes[0]["span_id"], starts[0]["span_id"], starts[1]["span_id"])
				}
				if completes[1]["span_id"] != starts[1]["span_id"] {
					t.Errorf("second AGENT_COMPLETED span_id = %v, want act2 span_id %v",
						completes[1]["span_id"], starts[1]["span_id"])
				}
			})
		}
	})

	t.Run("empty FunctionCall.ID collides on tool name without stale closedByError bleed", func(t *testing.T) {
		for _, tc := range []struct {
			name            string
			emitOnToolError bool
			skipCall1After  bool
			call1Result     map[string]any
		}{
			{
				name:            "closedByError via OnToolErrorCallback",
				emitOnToolError: true,
				call1Result:     map[string]any{"answer": "recovered"},
			},
			{
				name:            "closedByError with Call 1 AfterTool skipped",
				emitOnToolError: true,
				skipCall1After:  true,
			},
			{
				name:            "retryandreflect before analytics",
				emitOnToolError: false,
				call1Result: map[string]any{
					"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
					"error_details": "boom",
				},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				if mCtx.FunctionCallID() != "" {
					t.Fatalf("expected empty FunctionCallID on base mock context, got %q", mCtx.FunctionCallID())
				}
				final := &fakeTool{name: "final_answer"}
				call1Args := map[string]any{"answer": "bad"}
				call2Args := map[string]any{"answer": "good"}
				call2Result := map[string]any{"accepted": true, "answer": "good"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(mCtx, final, call1Args))
				if tc.emitOnToolError {
					errs.add(p.OnToolErrorCallback()(mCtx, final, call1Args, errors.New("boom")))
				}
				if !tc.skipCall1After {
					errs.add(p.AfterToolCallback()(mCtx, final, call1Args, tc.call1Result, nil))
				}
				errs.add(p.BeforeToolCallback()(mCtx, final, call2Args))
				errs.add(p.AfterToolCallback()(mCtx, final, call2Args, call2Result, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)

				starts := rowsOfType(rows, EventToolStarting)
				if len(starts) != 2 {
					t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
				}
				if placeOf(starts[0]).span == placeOf(starts[1]).span {
					t.Errorf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
				}

				errRow := onlyRowOfType(t, rows, EventToolError)
				if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
					t.Errorf("TOOL_ERROR span = %+v, want Call 1 span %+v", got, want)
				}
				mustLatency(t, errRow)
				if errRow["status"] != "ERROR" || errRow["error_message"] != "boom" {
					t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom)", errRow["status"], errRow["error_message"])
				}

				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
					t.Errorf("TOOL_COMPLETED span = %+v, want Call 2 span %+v", got, want)
				}
				mustLatency(t, compRow)
				if compRow["status"] != "OK" {
					t.Errorf("TOOL_COMPLETED status = %v, want OK", compRow["status"])
				}
				checkJSON(t, "Call 2 TOOL_COMPLETED content", jsonObjectColumn(t, compRow, "content"),
					`{"tool":"final_answer","result":{"accepted":true,"answer":"good"},"tool_origin":"UNKNOWN"}`)

				respRow := onlyRowOfType(t, rows, EventAgentResponse)
				checkJSONAt(t, jsonObjectColumn(t, respRow, "attributes"), `"final_answer"`, "source_tool")
				checkJSON(t, "Call 2 AGENT_RESPONSE content", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
			})
		}
	})
}

func TestPlugin_RecoveredToolError_OverlappedEmptyIDRecovery(t *testing.T) {
	t.Run("error-first overlapping empty IDs", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}
		call1Result := map[string]any{"answer": "recovered"}
		call2Result := map[string]any{"accepted": true, "answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(call1Ctx, final, call1Args))
		errs.add(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("boom1")))
		errs.add(p.BeforeToolCallback()(call2Ctx, final, call2Args))
		errs.add(p.AfterToolCallback()(call1Ctx, final, call1Args, call1Result, nil))
		errs.add(p.AfterToolCallback()(call2Ctx, final, call2Args, call2Result, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)

		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
		}
		if placeOf(starts[0]).span == placeOf(starts[1]).span {
			t.Errorf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
		}

		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want Call 1 span %+v", got, want)
		}
		mustLatency(t, errRow)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom1" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom1)", errRow["status"], errRow["error_message"])
		}

		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("TOOL_COMPLETED span = %+v, want Call 2 span %+v", got, want)
		}
		mustLatency(t, compRow)
		checkJSON(t, "Call 2 TOOL_COMPLETED content", jsonObjectColumn(t, compRow, "content"),
			`{"tool":"final_answer","result":{"accepted":true,"answer":"good"},"tool_origin":"UNKNOWN"}`)

		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSONAt(t, jsonObjectColumn(t, respRow, "attributes"), `"final_answer"`, "source_tool")
		checkJSON(t, "Call 2 AGENT_RESPONSE content", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})

	t.Run("success-first overlapping empty IDs", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}
		call1Result := map[string]any{"answer": "recovered"}
		call2Result := map[string]any{"accepted": true, "answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(call1Ctx, final, call1Args))
		errs.add(p.BeforeToolCallback()(call2Ctx, final, call2Args))
		errs.add(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("boom1")))
		errs.add(p.AfterToolCallback()(call2Ctx, final, call2Args, call2Result, nil))
		errs.add(p.AfterToolCallback()(call1Ctx, final, call1Args, call1Result, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)

		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
		}
		if placeOf(starts[0]).span == placeOf(starts[1]).span {
			t.Errorf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
		}

		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want Call 1 span %+v", got, want)
		}
		mustLatency(t, errRow)
		if errRow["status"] != "ERROR" || errRow["error_message"] != "boom1" {
			t.Errorf("TOOL_ERROR = (status=%v, error_message=%v), want (ERROR, boom1)", errRow["status"], errRow["error_message"])
		}

		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("TOOL_COMPLETED span = %+v, want Call 2 span %+v", got, want)
		}
		mustLatency(t, compRow)
		checkJSON(t, "Call 2 TOOL_COMPLETED content", jsonObjectColumn(t, compRow, "content"),
			`{"tool":"final_answer","result":{"accepted":true,"answer":"good"},"tool_origin":"UNKNOWN"}`)

		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSONAt(t, jsonObjectColumn(t, respRow, "attributes"), `"final_answer"`, "source_tool")
		checkJSON(t, "Call 2 AGENT_RESPONSE content", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})

	t.Run("concurrent empty IDs under race detector", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}
		call1Result := map[string]any{"answer": "recovered"}
		call2Result := map[string]any{"accepted": true, "answer": "good"}

		var errs callbackErrors
		var errMu sync.Mutex
		addErr := func(res any, err error) {
			if err == nil && res == nil {
				return
			}
			errMu.Lock()
			errs.add(res, err)
			errMu.Unlock()
		}

		addErr(p.BeforeAgentCallback()(mCtx))
		bothStarted := make(chan struct{})
		call1Errored := make(chan struct{})
		call2Finished := make(chan struct{})
		var call1Before, call2Before sync.WaitGroup
		call1Before.Add(1)
		call2Before.Add(1)

		var wg sync.WaitGroup
		wg.Go(func() {
			addErr(p.BeforeToolCallback()(call1Ctx, final, call1Args))
			call1Before.Done()
			<-bothStarted
			addErr(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("boom1")))
			close(call1Errored)
			<-call2Finished
			addErr(p.AfterToolCallback()(call1Ctx, final, call1Args, call1Result, nil))
		})
		wg.Go(func() {
			addErr(p.BeforeToolCallback()(call2Ctx, final, call2Args))
			call2Before.Done()
			<-bothStarted
			<-call1Errored
			addErr(p.AfterToolCallback()(call2Ctx, final, call2Args, call2Result, nil))
			close(call2Finished)
		})
		call1Before.Wait()
		call2Before.Wait()
		close(bothStarted)
		wg.Wait()

		addErr(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)

		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
		}
		if placeOf(starts[0]).span == placeOf(starts[1]).span {
			t.Errorf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		mustLatency(t, errRow)
		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		mustLatency(t, compRow)
		if placeOf(errRow).span == placeOf(compRow).span {
			t.Errorf("expected distinct TOOL_ERROR and TOOL_COMPLETED spans, both got %+v", placeOf(errRow))
		}
		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSON(t, "Call 2 AGENT_RESPONSE content", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})
}

type nonComparableSliceToolCtx struct {
	*mockInvocationContext
	tags []string
}

type wrappedUnhashableToolCtx struct {
	*mockInvocationContext
	meta any
}

type comparableValueToolCtx struct {
	*mockInvocationContext
	tag string
}

func TestPlugin_RecoveredToolError_EmptyIDEdgeCases(t *testing.T) {
	t.Run("sequential reuse of same context pointer", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		sharedCtx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		args1 := map[string]any{"answer": "first-ok"}
		args2 := map[string]any{"answer": "second-err"}
		args3 := map[string]any{"answer": "third-ok"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		// Call 1 succeeds cleanly on sharedCtx.
		errs.add(p.BeforeToolCallback()(sharedCtx, final, args1))
		errs.add(p.AfterToolCallback()(sharedCtx, final, args1, map[string]any{"answer": "first-ok"}, nil))
		// Call 2 reuses sharedCtx, errors and recovers cleanly.
		errs.add(p.BeforeToolCallback()(sharedCtx, final, args2))
		errs.add(p.OnToolErrorCallback()(sharedCtx, final, args2, errors.New("boom-2")))
		errs.add(p.AfterToolCallback()(sharedCtx, final, args2, map[string]any{"answer": "recovered-2"}, nil))
		// Call 3 reuses sharedCtx again and succeeds without stale closedByError bleed.
		errs.add(p.BeforeToolCallback()(sharedCtx, final, args3))
		errs.add(p.AfterToolCallback()(sharedCtx, final, args3, map[string]any{"answer": "third-ok"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)

		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 3 {
			t.Fatalf("got %d TOOL_STARTING rows, want 3; events=%v", len(starts), eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[1]); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want Call 2 span %+v", got, want)
		}
		mustLatency(t, errRow)
		comps := rowsOfType(rows, EventToolCompleted)
		if len(comps) != 2 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 2; events=%v", len(comps), eventTypes(rows))
		}
		if got, want := placeOf(comps[0]), placeOf(starts[0]); got != want {
			t.Errorf("first TOOL_COMPLETED span = %+v, want Call 1 span %+v", got, want)
		}
		if got, want := placeOf(comps[1]), placeOf(starts[2]); got != want {
			t.Errorf("second TOOL_COMPLETED span = %+v, want Call 3 span %+v", got, want)
		}
		resps := rowsOfType(rows, EventAgentResponse)
		if len(resps) != 2 {
			t.Fatalf("got %d AGENT_RESPONSE rows, want 2; events=%v", len(resps), eventTypes(rows))
		}
		checkJSON(t, "first AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), `{"response":{"answer":"first-ok"}}`)
		checkJSON(t, "third AGENT_RESPONSE", jsonObjectColumn(t, resps[1], "content"), `{"response":{"answer":"third-ok"}}`)
	})

	t.Run("non-comparable value contexts do not panic", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			makeCtx func(*mockInvocationContext) agent.Context
		}{
			{
				name: "struct with slice field",
				makeCtx: func(m *mockInvocationContext) agent.Context {
					return nonComparableSliceToolCtx{mockInvocationContext: withFunctionCall(m, ""), tags: []string{"t1"}}
				},
			},
			{
				name: "comparable struct wrapping unhashable interface value",
				makeCtx: func(m *mockInvocationContext) agent.Context {
					return wrappedUnhashableToolCtx{mockInvocationContext: withFunctionCall(m, ""), meta: []string{"t1"}}
				},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				toolCtx := tc.makeCtx(mCtx)
				lookup := &fakeTool{name: "lookup"}
				args := map[string]any{"q": "ok"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(toolCtx, lookup, args))
				errs.add(p.OnToolErrorCallback()(toolCtx, lookup, args, errors.New("boom")))
				errs.add(p.AfterToolCallback()(toolCtx, lookup, args, map[string]any{"q": "recovered"}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				if n := len(rowsOfType(rows, EventToolCompleted)); n != 1 {
					t.Fatalf("got %d TOOL_COMPLETED rows, want 1; events=%v", n, eventTypes(rows))
				}
			})
		}
	})

	t.Run("comparable value struct contexts preserve valid completion on overlap", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		base := withFunctionCall(mCtx, "")
		call1Ctx := comparableValueToolCtx{mockInvocationContext: base, tag: "call-1"}
		call2Ctx := comparableValueToolCtx{mockInvocationContext: base, tag: "call-2"}
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(call1Ctx, final, call1Args))
		errs.add(p.BeforeToolCallback()(call2Ctx, final, call2Args))
		errs.add(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("boom-val")))
		errs.add(p.AfterToolCallback()(call2Ctx, final, call2Args, map[string]any{"answer": "good"}, nil))
		errs.add(p.AfterToolCallback()(call1Ctx, final, call1Args, map[string]any{"answer": "recovered"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2", len(starts))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		comps := rowsOfType(rows, EventToolCompleted)
		requireAmbiguousOverlapSpans(t, starts, comps, errRow)
		resps := rowsOfType(rows, EventAgentResponse)
		if len(resps) != 1 {
			t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
		}
		checkJSON(t, "Call 2 AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), `{"response":{"answer":"good"}}`)
	})

	t.Run("error without BeforeTool does not suppress overlapping empty-ID sibling", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		// Call 1 BeforeTool was skipped (short-circuited), OnToolError runs alone.
		errs.add(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("boom-no-before")))
		// Call 2 also skips BeforeTool (or completes after Call 1 errored) and succeeds first.
		errs.add(p.AfterToolCallback()(call2Ctx, final, call2Args, map[string]any{"answer": "good"}, nil))
		errs.add(p.AfterToolCallback()(call1Ctx, final, call1Args, map[string]any{"answer": "recovered"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		onlyRowOfType(t, rows, EventToolError)
		onlyRowOfType(t, rows, EventToolCompleted)
		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSON(t, "Call 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})

	t.Run("same context overlapping distinct tool names stay independent", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		sharedCtx := withFunctionCall(mCtx, "")
		outer := &fakeTool{name: "final_answer"}
		inner := &fakeTool{name: "helper"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(sharedCtx, outer, map[string]any{"answer": "bad"}))
		errs.add(p.BeforeToolCallback()(sharedCtx, inner, map[string]any{"q": "sub"}))
		errs.add(p.OnToolErrorCallback()(sharedCtx, outer, map[string]any{"answer": "bad"}, errors.New("outer-err")))
		errs.add(p.AfterToolCallback()(sharedCtx, inner, map[string]any{"q": "sub"}, map[string]any{"ok": true}, nil))
		errs.add(p.AfterToolCallback()(sharedCtx, outer, map[string]any{"answer": "bad"}, map[string]any{"answer": "recovered"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2", len(starts))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
			t.Errorf("outer TOOL_ERROR span = %+v, want %+v", got, want)
		}
		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("inner TOOL_COMPLETED span = %+v, want %+v", got, want)
		}
		if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
			t.Errorf("got %d AGENT_RESPONSE rows, want 0; events=%v", n, eventTypes(rows))
		}
	})

	t.Run("late AfterRun callbacks keep overlapping empty-IDs isolated", func(t *testing.T) {
		p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		call1Args := map[string]any{"answer": "bad"}
		call2Args := map[string]any{"answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(call1Ctx, final, call1Args))
		errs.add(p.OnToolErrorCallback()(call1Ctx, final, call1Args, errors.New("late-err")))
		errs.add(p.BeforeToolCallback()(call2Ctx, final, call2Args))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		if _, err := p.AfterToolCallback()(call1Ctx, final, call1Args, map[string]any{"answer": "recovered"}, nil); err != nil {
			t.Fatalf("late Call 1 AfterToolCallback: %v", err)
		}
		if _, err := p.AfterToolCallback()(call2Ctx, final, call2Args, map[string]any{"accepted": true, "answer": "good"}, nil); err != nil {
			t.Fatalf("late Call 2 AfterToolCallback: %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close: %v", err)
		}

		rows := drainAllRows(t, requests)
		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
			t.Errorf("TOOL_ERROR span = %+v, want Call 1 span %+v", got, want)
		}
		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("TOOL_COMPLETED span = %+v, want Call 2 span %+v", got, want)
		}
		mustLatency(t, compRow)
		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSON(t, "late Call 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})

	t.Run("overlapping empty-ID finish_task suppresses failed duplicate", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		call1Ctx := withFunctionCall(mCtx, "")
		call2Ctx := withFunctionCall(mCtx, "")
		finish := &fakeTool{name: finishTaskToolName}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}))
		errs.add(p.OnToolErrorCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, errors.New("finish-boom")))
		errs.add(p.BeforeToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}))
		errs.add(p.AfterToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, map[string]any{"ok": true}, nil))
		errs.add(p.AfterToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}, map[string]any{"ok": true}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2", len(starts))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
			t.Errorf("finish_task TOOL_ERROR span = %+v, want %+v", got, want)
		}
		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("finish_task TOOL_COMPLETED span = %+v, want %+v", got, want)
		}
	})
}

type nonComparableMapToolCtx struct {
	*mockInvocationContext
	labels map[string]string
}

type nonComparableFuncToolCtx struct {
	*mockInvocationContext
	fn func() int
}

// requireAmbiguousOverlapSpans checks the rows of two overlapping calls with
// an empty function call ID whose callback contexts cannot be told apart,
// one of which failed: one TOOL_ERROR row and one TOOL_COMPLETED row, since
// the failed call's completion is suppressed by count, each on a span of its
// own without latency, since neither can be told to belong to either
// TOOL_STARTING row.
func requireAmbiguousOverlapSpans(t *testing.T, starts, comps []map[string]any, errRow map[string]any) {
	t.Helper()
	if len(starts) != 2 {
		t.Fatalf("got %d TOOL_STARTING rows, want 2", len(starts))
	}
	if len(comps) != 1 {
		t.Fatalf("got %d TOOL_COMPLETED rows, want 1 (the failed call's completion is suppressed)", len(comps))
	}
	if errRow["latency_ms"] != nil {
		t.Errorf("ambiguous TOOL_ERROR latency_ms = %v, want nil", errRow["latency_ms"])
	}
	s0 := placeOf(starts[0]).span
	s1 := placeOf(starts[1]).span
	errPlace := placeOf(errRow)
	if errPlace.span == s0 || errPlace.span == s1 {
		t.Errorf("ambiguous TOOL_ERROR span = %+v, must be distinct from TOOL_STARTING spans %+v and %+v",
			errPlace, placeOf(starts[0]), placeOf(starts[1]))
	}
	if comps[0]["latency_ms"] != nil {
		t.Errorf("ambiguous TOOL_COMPLETED latency_ms = %v, want nil", comps[0]["latency_ms"])
	}
	if c := placeOf(comps[0]); c.span == s0 || c.span == s1 || c.span == errPlace.span {
		t.Errorf("ambiguous TOOL_COMPLETED span = %+v, must not borrow start (%s, %s) or error (%s) span",
			c, s0, s1, errPlace.span)
	}
}

func TestPlugin_RecoveredToolError_SameContextOverlap(t *testing.T) {
	ctxFactories := []struct {
		name  string
		build func(*mockInvocationContext) agent.Context
	}{
		{
			name: "pointer_context",
			build: func(m *mockInvocationContext) agent.Context {
				return withFunctionCall(m, "")
			},
		},
		{
			name: "comparable_value_context",
			build: func(m *mockInvocationContext) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, ""), tag: "shared-val"}
			},
		},
		{
			name: "non_comparable_slice_context",
			build: func(m *mockInvocationContext) agent.Context {
				return nonComparableSliceToolCtx{mockInvocationContext: withFunctionCall(m, ""), tags: []string{"shared-slice"}}
			},
		},
	}

	orders := []struct {
		name         string
		firstErrors  bool
		finishFirst  int // index (0 or 1) that calls AfterTool first
		freshLiteral bool
		nestedArgs   bool
		nilArgs      bool
		shareArgs    bool
	}{
		{name: "error_call_errors_first_reused_args", firstErrors: true, finishFirst: 1},
		{name: "error_call_recovered_first_reused_args", firstErrors: true, finishFirst: 0},
		{name: "success_call_finishes_before_error_reused_args", firstErrors: false, finishFirst: 0},
		{name: "error_call_fresh_map_literals", firstErrors: true, finishFirst: 1, freshLiteral: true},
		{name: "error_call_recovered_first_fresh_map_literals", firstErrors: true, finishFirst: 0, freshLiteral: true},
		{name: "error_call_recovered_first_fresh_nested_map_literals", firstErrors: true, finishFirst: 0, freshLiteral: true, nestedArgs: true},
		{name: "success_first_fresh_nested_map_literals", firstErrors: false, finishFirst: 0, freshLiteral: true, nestedArgs: true},
		{name: "error_first_nil_args", firstErrors: true, finishFirst: 1, nilArgs: true},
		{name: "error_first_recovered_first_nil_args", firstErrors: true, finishFirst: 0, nilArgs: true},
		{name: "success_first_nil_args", firstErrors: false, finishFirst: 0, nilArgs: true},
		{name: "error_first_shared_args", firstErrors: true, finishFirst: 1, shareArgs: true},
		{name: "error_first_recovered_first_shared_args", firstErrors: true, finishFirst: 0, shareArgs: true},
		{name: "success_first_shared_args", firstErrors: false, finishFirst: 0, shareArgs: true},
	}

	for _, cf := range ctxFactories {
		t.Run(cf.name, func(t *testing.T) {
			for _, ord := range orders {
				t.Run(ord.name, func(t *testing.T) {
					p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
						c.FinalResponseToolNames = []string{"final_answer"}
					})
					sharedCtx := cf.build(mCtx)
					final := &fakeTool{name: "final_answer"}

					makeArgs := func(ans string) map[string]any {
						if ord.nilArgs {
							return nil
						}
						if ord.nestedArgs {
							return map[string]any{
								"answer": ans,
								"tags":   []string{ans, "shared"},
								"meta":   map[string]any{"k": ans, "n": 1},
							}
						}
						return map[string]any{"answer": ans}
					}
					reusedBad := makeArgs("bad")
					reusedGood := makeArgs("good")
					if ord.shareArgs {
						reusedGood = reusedBad
					}
					getArgs := func(isErr bool) map[string]any {
						if ord.freshLiteral {
							if isErr {
								return makeArgs("bad")
							}
							return makeArgs("good")
						}
						if isErr {
							return reusedBad
						}
						return reusedGood
					}

					var errs callbackErrors
					errs.add(p.BeforeAgentCallback()(mCtx))
					errs.add(p.BeforeToolCallback()(sharedCtx, final, getArgs(ord.firstErrors)))
					errs.add(p.BeforeToolCallback()(sharedCtx, final, getArgs(!ord.firstErrors)))

					if ord.firstErrors {
						errs.add(p.OnToolErrorCallback()(sharedCtx, final, getArgs(true), errors.New("boom-bad")))
						if ord.finishFirst == 1 {
							errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(false), map[string]any{"answer": "good"}, nil))
							errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(true), map[string]any{"answer": "recovered"}, nil))
						} else {
							errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(true), map[string]any{"answer": "recovered"}, nil))
							errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(false), map[string]any{"answer": "good"}, nil))
						}
					} else {
						errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(false), map[string]any{"answer": "good"}, nil))
						errs.add(p.OnToolErrorCallback()(sharedCtx, final, getArgs(true), errors.New("boom-bad")))
						errs.add(p.AfterToolCallback()(sharedCtx, final, getArgs(true), map[string]any{"answer": "recovered"}, nil))
					}

					errs.add(p.AfterAgentCallback()(mCtx))
					errs.check(t)
					p.AfterRunCallback()(mCtx)

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)

					starts := rowsOfType(rows, EventToolStarting)
					if len(starts) != 2 {
						t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
					}
					if placeOf(starts[0]).span == placeOf(starts[1]).span {
						t.Fatalf("expected distinct TOOL_STARTING spans, got %+v and %+v", placeOf(starts[0]), placeOf(starts[1]))
					}
					errRow := onlyRowOfType(t, rows, EventToolError)
					comps := rowsOfType(rows, EventToolCompleted)
					requireAmbiguousOverlapSpans(t, starts, comps, errRow)
					resps := rowsOfType(rows, EventAgentResponse)
					if len(resps) != 1 {
						t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
					}
					foundGoodComp := false
					for _, c := range comps {
						gotComp := jsonObjectColumn(t, c, "content")
						resBody, _ := gotComp["result"].(map[string]any)
						if resBody["answer"] == "good" {
							foundGoodComp = true
						}
					}
					// With equal arguments, a recovered call that ends first
					// cannot be told apart from the healthy one, so its result
					// may stand in for the healthy call's.
					indistinct := ord.firstErrors && ord.finishFirst == 0 && (ord.nilArgs || ord.shareArgs)
					if !foundGoodComp && !indistinct {
						t.Errorf("expected TOOL_COMPLETED with result.answer=good, got %v", comps)
					}
					wantAnswer := any("good")
					if ord.shareArgs {
						wantAnswer = "bad"
					}
					foundResp := false
					for _, r := range resps {
						gotResp := jsonObjectColumn(t, r, "content")
						if ord.nilArgs {
							if val, ok := gotResp["response"]; ok && val == nil {
								foundResp = true
							}
							continue
						}
						respBody, _ := gotResp["response"].(map[string]any)
						if respBody["answer"] == wantAnswer {
							foundResp = true
						}
					}
					if !foundResp {
						t.Errorf("expected AGENT_RESPONSE matching args (%v) to be preserved, got %v", wantAnswer, resps)
					}
				})
			}
		})
	}
}

func TestPlugin_RecoveredToolError_NonComparableOverlap(t *testing.T) {
	variants := []struct {
		name  string
		build func(*mockInvocationContext) (agent.Context, agent.Context)
	}{
		{
			name: "slice_field",
			build: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				return nonComparableSliceToolCtx{mockInvocationContext: base, tags: []string{"c1"}},
					nonComparableSliceToolCtx{mockInvocationContext: base, tags: []string{"c2"}}
			},
		},
		{
			name: "map_field",
			build: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				return nonComparableMapToolCtx{mockInvocationContext: base, labels: map[string]string{"id": "1", "env": "dev"}},
					nonComparableMapToolCtx{mockInvocationContext: base, labels: map[string]string{"id": "2", "env": "prod"}}
			},
		},
		{
			name: "func_field",
			build: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				v1, v2 := 1, 2
				return nonComparableFuncToolCtx{mockInvocationContext: base, fn: func() int { return v1 }},
					nonComparableFuncToolCtx{mockInvocationContext: base, fn: func() int { return v2 }}
			},
		},
		{
			name: "wrapped_unhashable_interface",
			build: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				return wrappedUnhashableToolCtx{mockInvocationContext: base, meta: []string{"c1"}},
					wrappedUnhashableToolCtx{mockInvocationContext: base, meta: map[string]any{"id": 2}}
			},
		},
	}

	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			for _, tc := range []struct {
				name        string
				errorFirst  bool
				sameArgText bool
				shareArgs   bool
			}{
				{name: "error_first_distinct_args", errorFirst: true},
				{name: "success_first_distinct_args", errorFirst: false},
				{name: "error_first_identical_args", errorFirst: true, sameArgText: true},
				{name: "success_first_identical_args", errorFirst: false, sameArgText: true},
				{name: "success_first_shared_args", errorFirst: false, sameArgText: true, shareArgs: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
						c.FinalResponseToolNames = []string{"final_answer"}
					})
					ctx1, ctx2 := v.build(mCtx)
					final := &fakeTool{name: "final_answer"}
					args1 := map[string]any{"answer": "bad"}
					args2 := map[string]any{"answer": "good"}
					if tc.sameArgText {
						args1 = map[string]any{"answer": "same"}
						if tc.shareArgs {
							args2 = args1
						} else {
							args2 = map[string]any{"answer": "same"}
						}
					}

					var errs callbackErrors
					errs.add(p.BeforeAgentCallback()(mCtx))
					errs.add(p.BeforeToolCallback()(ctx1, final, args1))
					errs.add(p.BeforeToolCallback()(ctx2, final, args2))
					if tc.errorFirst {
						errs.add(p.OnToolErrorCallback()(ctx1, final, args1, errors.New("boom-nc")))
						errs.add(p.AfterToolCallback()(ctx2, final, args2, map[string]any{"answer": "good"}, nil))
						errs.add(p.AfterToolCallback()(ctx1, final, args1, map[string]any{"answer": "recovered"}, nil))
					} else {
						errs.add(p.AfterToolCallback()(ctx2, final, args2, map[string]any{"answer": "good"}, nil))
						errs.add(p.OnToolErrorCallback()(ctx1, final, args1, errors.New("boom-nc")))
						errs.add(p.AfterToolCallback()(ctx1, final, args1, map[string]any{"answer": "recovered"}, nil))
					}
					errs.add(p.AfterAgentCallback()(mCtx))
					errs.check(t)
					p.AfterRunCallback()(mCtx)

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)
					starts := rowsOfType(rows, EventToolStarting)
					if len(starts) != 2 {
						t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
					}
					errRow := onlyRowOfType(t, rows, EventToolError)
					comps := rowsOfType(rows, EventToolCompleted)
					requireAmbiguousOverlapSpans(t, starts, comps, errRow)
					resps := rowsOfType(rows, EventAgentResponse)
					if len(resps) != 1 {
						t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
					}
					wantResp := `{"response":{"answer":"good"}}`
					if tc.sameArgText {
						wantResp = `{"response":{"answer":"same"}}`
					}
					checkJSON(t, "Call 2 AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), wantResp)
				})
			}
		})
	}
}

func TestPlugin_RecoveredToolError_LateValueContextHandoff(t *testing.T) {
	ctxKinds := []struct {
		name  string
		build func(*mockInvocationContext, string) agent.Context
	}{
		{
			name: "comparable_value_context",
			build: func(m *mockInvocationContext, tag string) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, ""), tag: tag}
			},
		},
		{
			name: "non_comparable_slice_context",
			build: func(m *mockInvocationContext, tag string) agent.Context {
				return nonComparableSliceToolCtx{mockInvocationContext: withFunctionCall(m, ""), tags: []string{tag, "late"}}
			},
		},
		{
			name: "non_comparable_map_context",
			build: func(m *mockInvocationContext, tag string) agent.Context {
				return nonComparableMapToolCtx{mockInvocationContext: withFunctionCall(m, ""), labels: map[string]string{"tag": tag}}
			},
		},
	}

	for _, ck := range ctxKinds {
		t.Run(ck.name, func(t *testing.T) {
			t.Run("late_success_preserves_span_and_latency", func(t *testing.T) {
				p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				toolCtx := ck.build(mCtx, "late-ok")
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "good"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(toolCtx, final, args))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				if _, err := p.AfterToolCallback()(toolCtx, final, args, map[string]any{"answer": "good"}, nil); err != nil {
					t.Fatalf("late AfterToolCallback: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("p.Close: %v", err)
				}

				rows := drainAllRows(t, requests)
				startRow := onlyRowOfType(t, rows, EventToolStarting)
				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				if got, want := placeOf(compRow), placeOf(startRow); got != want {
					t.Errorf("late TOOL_COMPLETED span = %+v, want TOOL_STARTING span %+v", got, want)
				}
				mustLatency(t, compRow)
				respRow := onlyRowOfType(t, rows, EventAgentResponse)
				checkJSON(t, "late AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
			})

			t.Run("live_error_then_late_recovered_aftertool_preserves_completion", func(t *testing.T) {
				p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				toolCtx := ck.build(mCtx, "live-err")
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "bad"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(toolCtx, final, args))
				errs.add(p.OnToolErrorCallback()(toolCtx, final, args, errors.New("boom-live")))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				if _, err := p.AfterToolCallback()(toolCtx, final, args, map[string]any{"answer": "recovered"}, nil); err != nil {
					t.Fatalf("late AfterToolCallback: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("p.Close: %v", err)
				}

				rows := drainAllRows(t, requests)
				startRow := onlyRowOfType(t, rows, EventToolStarting)
				errRow := onlyRowOfType(t, rows, EventToolError)
				if got, want := placeOf(errRow), placeOf(startRow); got != want {
					t.Errorf("TOOL_ERROR span = %+v, want %+v", got, want)
				}
				mustLatency(t, errRow)
				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				if got := placeOf(compRow); got == placeOf(startRow) {
					t.Errorf("late value-context TOOL_COMPLETED reused starting span %+v", got)
				}
				respRow := onlyRowOfType(t, rows, EventAgentResponse)
				checkJSON(t, "late value-context AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"bad"}}`)
			})

			t.Run("late_error_then_late_recovered_aftertool_preserves_completion", func(t *testing.T) {
				p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				toolCtx := ck.build(mCtx, "late-err")
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "bad"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(toolCtx, final, args))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				if _, err := p.OnToolErrorCallback()(toolCtx, final, args, errors.New("boom-late")); err != nil {
					t.Fatalf("late OnToolErrorCallback: %v", err)
				}
				if _, err := p.AfterToolCallback()(toolCtx, final, args, map[string]any{"answer": "recovered"}, nil); err != nil {
					t.Fatalf("late AfterToolCallback: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("p.Close: %v", err)
				}

				rows := drainAllRows(t, requests)
				startRow := onlyRowOfType(t, rows, EventToolStarting)
				errRow := onlyRowOfType(t, rows, EventToolError)
				if got, want := placeOf(errRow), placeOf(startRow); got != want {
					t.Errorf("late TOOL_ERROR span = %+v, want %+v", got, want)
				}
				mustLatency(t, errRow)
				compRow := onlyRowOfType(t, rows, EventToolCompleted)
				if got := placeOf(compRow); got == placeOf(startRow) {
					t.Errorf("late value-context TOOL_COMPLETED reused starting span %+v", got)
				}
				respRow := onlyRowOfType(t, rows, EventAgentResponse)
				checkJSON(t, "late value-context AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"bad"}}`)
			})

			t.Run("overlapping_shared_context_across_AfterRun", func(t *testing.T) {
				p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				sharedCtx := ck.build(mCtx, "shared-late")
				final := &fakeTool{name: "final_answer"}
				call1Args := map[string]any{"answer": "bad"}
				call2Args := map[string]any{"answer": "good"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(sharedCtx, final, call1Args))
				errs.add(p.BeforeToolCallback()(sharedCtx, final, call2Args))
				errs.add(p.OnToolErrorCallback()(sharedCtx, final, call1Args, errors.New("boom-late-overlap")))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				if _, err := p.AfterToolCallback()(sharedCtx, final, call2Args, map[string]any{"answer": "good"}, nil); err != nil {
					t.Fatalf("late Call 2 AfterToolCallback: %v", err)
				}
				if _, err := p.AfterToolCallback()(sharedCtx, final, call1Args, map[string]any{"answer": "recovered"}, nil); err != nil {
					t.Fatalf("late Call 1 AfterToolCallback: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("p.Close: %v", err)
				}

				rows := drainAllRows(t, requests)
				starts := rowsOfType(rows, EventToolStarting)
				if len(starts) != 2 {
					t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
				}
				errRow := onlyRowOfType(t, rows, EventToolError)
				comps := rowsOfType(rows, EventToolCompleted)
				requireAmbiguousOverlapSpans(t, starts, comps, errRow)
				resps := rowsOfType(rows, EventAgentResponse)
				if len(resps) != 1 {
					t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
				}
				checkJSON(t, "late Call 2 AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), `{"response":{"answer":"good"}}`)
			})

			t.Run("overlapping_distinct_contexts_across_AfterRun", func(t *testing.T) {
				p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				ctx1 := ck.build(mCtx, "late-distinct-1")
				ctx2 := ck.build(mCtx, "late-distinct-2")
				final := &fakeTool{name: "final_answer"}
				call1Args := map[string]any{"answer": "bad"}
				call2Args := map[string]any{"answer": "good"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctx1, final, call1Args))
				errs.add(p.BeforeToolCallback()(ctx2, final, call2Args))
				errs.add(p.OnToolErrorCallback()(ctx1, final, call1Args, errors.New("boom-late-distinct")))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				if _, err := p.AfterToolCallback()(ctx2, final, call2Args, map[string]any{"answer": "good"}, nil); err != nil {
					t.Fatalf("late Call 2 AfterToolCallback: %v", err)
				}
				if _, err := p.AfterToolCallback()(ctx1, final, call1Args, map[string]any{"answer": "recovered"}, nil); err != nil {
					t.Fatalf("late Call 1 AfterToolCallback: %v", err)
				}
				if err := p.Close(); err != nil {
					t.Fatalf("p.Close: %v", err)
				}

				rows := drainAllRows(t, requests)
				starts := rowsOfType(rows, EventToolStarting)
				if len(starts) != 2 {
					t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
				}
				errRow := onlyRowOfType(t, rows, EventToolError)
				comps := rowsOfType(rows, EventToolCompleted)
				requireAmbiguousOverlapSpans(t, starts, comps, errRow)
				resps := rowsOfType(rows, EventAgentResponse)
				if len(resps) != 1 {
					t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
				}
				checkJSON(t, "late distinct Call 2 AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), `{"response":{"answer":"good"}}`)
			})
		})
	}
}

type lockedMapPointerToolCtx struct {
	*mockInvocationContext
	mu   sync.Mutex
	data map[string]string
}

func TestPlugin_RecoveredToolError_MutableLockedContextNoRace(t *testing.T) {
	t.Run("pointer context with mutex-guarded map mutated concurrently under race", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		ptrCtx := &lockedMapPointerToolCtx{
			mockInvocationContext: withFunctionCall(mCtx, ""),
			data:                  map[string]string{"k": "v0"},
		}
		lookup := &fakeTool{name: "lookup"}
		args := map[string]any{"q": "1"}

		stop := make(chan struct{})
		var wg sync.WaitGroup
		wg.Go(func() {
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				ptrCtx.mu.Lock()
				ptrCtx.data["k"] = fmt.Sprintf("v-%d", i)
				ptrCtx.mu.Unlock()
			}
		})

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		for range 20 {
			errs.add(p.BeforeToolCallback()(ptrCtx, lookup, args))
			errs.add(p.OnToolErrorCallback()(ptrCtx, lookup, args, errors.New("boom-locked")))
			errs.add(p.AfterToolCallback()(ptrCtx, lookup, args, map[string]any{"recovered": true}, nil))
		}
		errs.add(p.AfterAgentCallback()(mCtx))
		close(stop)
		wg.Wait()
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		if n := len(rowsOfType(rows, EventToolError)); n != 20 {
			t.Fatalf("got %d TOOL_ERROR rows, want 20", n)
		}
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0", n)
		}
	})

	t.Run("value context with backing map mutated between OnToolError and AfterTool", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		backing := map[string]string{"phase": "before"}
		valCtx := nonComparableMapToolCtx{
			mockInvocationContext: withFunctionCall(mCtx, ""),
			labels:                backing,
		}
		lookup := &fakeTool{name: "lookup"}
		args := map[string]any{"q": "1"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(valCtx, lookup, args))
		backing["phase"] = "errored"
		errs.add(p.OnToolErrorCallback()(valCtx, lookup, args, errors.New("boom-mutated")))
		backing["phase"] = "after"
		errs.add(p.AfterToolCallback()(valCtx, lookup, args, map[string]any{"recovered": true}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		errRow := onlyRowOfType(t, rows, EventToolError)
		mustLatency(t, errRow)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 1 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 1; events=%v", n, eventTypes(rows))
		}
	})
}

func TestPlugin_RecoveredToolError_SkippedAfterCallbackSameArgsRetry(t *testing.T) {
	for _, tc := range []struct {
		name string
		args map[string]any
	}{
		{name: "nil_args", args: nil},
		{name: "shared_args_map", args: map[string]any{"answer": "good"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			sharedCtx := withFunctionCall(mCtx, "")
			final := &fakeTool{name: "final_answer"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			// Attempt 1: BeforeTool -> OnToolError; an upstream plugin short-circuits AfterToolCallback.
			errs.add(p.BeforeToolCallback()(sharedCtx, final, tc.args))
			errs.add(p.OnToolErrorCallback()(sharedCtx, final, tc.args, errors.New("attempt1-boom")))
			// Attempt 2: Retry with identical context and identical/nil args succeeds cleanly.
			errs.add(p.BeforeToolCallback()(sharedCtx, final, tc.args))
			errs.add(p.AfterToolCallback()(sharedCtx, final, tc.args, map[string]any{"answer": "good"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			starts := rowsOfType(rows, EventToolStarting)
			if len(starts) != 2 {
				t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
			}
			errRow := onlyRowOfType(t, rows, EventToolError)
			if got, want := placeOf(errRow), placeOf(starts[0]); got != want {
				t.Errorf("TOOL_ERROR span = %+v, want Attempt 1 span %+v", got, want)
			}
			mustLatency(t, errRow)
			compRow := onlyRowOfType(t, rows, EventToolCompleted)
			if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
				t.Errorf("TOOL_COMPLETED span = %+v, want Attempt 2 span %+v", got, want)
			}
			mustLatency(t, compRow)
			checkJSON(t, "Attempt 2 TOOL_COMPLETED", jsonObjectColumn(t, compRow, "content"),
				`{"tool":"final_answer","result":{"answer":"good"},"tool_origin":"UNKNOWN"}`)
			respRow := onlyRowOfType(t, rows, EventAgentResponse)
			checkJSONAt(t, jsonObjectColumn(t, respRow, "attributes"), `"final_answer"`, "source_tool")
			if tc.args != nil {
				checkJSON(t, "Attempt 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
			}
		})
	}

	t.Run("late_retry_after_AfterRun", func(t *testing.T) {
		p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		sharedCtx := withFunctionCall(mCtx, "")
		final := &fakeTool{name: "final_answer"}
		sharedArgs := map[string]any{"answer": "good"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(sharedCtx, final, sharedArgs))
		errs.add(p.OnToolErrorCallback()(sharedCtx, final, sharedArgs, errors.New("attempt1-boom")))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		if _, err := p.BeforeToolCallback()(sharedCtx, final, sharedArgs); err != nil {
			t.Fatalf("late BeforeToolCallback: %v", err)
		}
		if _, err := p.AfterToolCallback()(sharedCtx, final, sharedArgs, map[string]any{"answer": "good"}, nil); err != nil {
			t.Fatalf("late AfterToolCallback: %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("p.Close: %v", err)
		}

		rows := drainAllRows(t, requests)
		starts := rowsOfType(rows, EventToolStarting)
		if len(starts) != 2 {
			t.Fatalf("got %d TOOL_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
		}
		errRow := onlyRowOfType(t, rows, EventToolError)
		mustLatency(t, errRow)
		compRow := onlyRowOfType(t, rows, EventToolCompleted)
		if got, want := placeOf(compRow), placeOf(starts[1]); got != want {
			t.Errorf("late TOOL_COMPLETED span = %+v, want Attempt 2 span %+v", got, want)
		}
		mustLatency(t, compRow)
		respRow := onlyRowOfType(t, rows, EventAgentResponse)
		checkJSON(t, "late Attempt 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
	})
}

func makeClosureFactoryToolCtx(base *mockInvocationContext, id int) nonComparableFuncToolCtx {
	return nonComparableFuncToolCtx{
		mockInvocationContext: base,
		fn:                    func() int { return id },
	}
}

func TestPlugin_RecoveredToolError_ClosureFactoryOverlapPreservesValidCompletion(t *testing.T) {
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.FinalResponseToolNames = []string{"final_answer"}
	})
	base := withFunctionCall(mCtx, "")
	ctx1 := makeClosureFactoryToolCtx(base, 1)
	ctx2 := makeClosureFactoryToolCtx(base, 2)
	final := &fakeTool{name: "final_answer"}
	sharedArgs := map[string]any{"answer": "same"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	errs.add(p.BeforeToolCallback()(ctx1, final, sharedArgs))
	errs.add(p.BeforeToolCallback()(ctx2, final, sharedArgs))
	errs.add(p.OnToolErrorCallback()(ctx1, final, sharedArgs, errors.New("boom-closure")))
	errs.add(p.AfterToolCallback()(ctx2, final, sharedArgs, map[string]any{"status": "good", "answer": "good"}, nil))
	errs.add(p.AfterToolCallback()(ctx1, final, sharedArgs, map[string]any{"status": "recovered", "answer": "recovered"}, nil))
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	starts := rowsOfType(rows, EventToolStarting)
	if len(starts) != 2 {
		t.Fatalf("got %d TOOL_STARTING rows, want 2", len(starts))
	}
	errRow := onlyRowOfType(t, rows, EventToolError)
	comps := rowsOfType(rows, EventToolCompleted)
	requireAmbiguousOverlapSpans(t, starts, comps, errRow)
	checkJSON(t, "healthy sibling TOOL_COMPLETED", jsonObjectColumn(t, comps[0], "content"),
		`{"tool":"final_answer","result":{"answer":"good","status":"good"},"tool_origin":"UNKNOWN"}`)
	resps := rowsOfType(rows, EventAgentResponse)
	if len(resps) != 1 {
		t.Fatalf("got %d AGENT_RESPONSE rows, want 1 (the failed call ends without one); events=%v", len(resps), eventTypes(rows))
	}
	checkJSON(t, "healthy sibling AGENT_RESPONSE", jsonObjectColumn(t, resps[0], "content"), `{"response":{"answer":"same"}}`)
}

func TestPlugin_RecoveredToolError_CyclicArgsMapConstantTime(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	valCtx := comparableValueToolCtx{
		mockInvocationContext: withFunctionCall(mCtx, ""),
		tag:                   "cyclic",
	}
	lookup := &fakeTool{name: "lookup"}
	cyclicArgs := make(map[string]any, 256)
	for i := range 256 {
		cyclicArgs[fmt.Sprintf("k-%03d", i)] = cyclicArgs
	}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(mCtx))
	start := time.Now()
	errs.add(p.BeforeToolCallback()(valCtx, lookup, cyclicArgs))
	errs.add(p.OnToolErrorCallback()(valCtx, lookup, cyclicArgs, errors.New("boom-cyclic")))
	errs.add(p.AfterToolCallback()(valCtx, lookup, cyclicArgs, map[string]any{"recovered": true}, nil))
	elapsed := time.Since(start)
	if elapsed > 100*time.Millisecond {
		t.Fatalf("cyclic args map took %v, want <100ms", elapsed)
	}
	errs.add(p.AfterAgentCallback()(mCtx))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	errRow := onlyRowOfType(t, rows, EventToolError)
	mustLatency(t, errRow)
	if n := len(rowsOfType(rows, EventToolCompleted)); n != 1 {
		t.Fatalf("got %d TOOL_COMPLETED rows, want 1", n)
	}
}

func TestPlugin_RecoveredToolError_FinishTaskCrossActivationDroppedOrSuppressed(t *testing.T) {
	hasFinishTaskFR := func(c *genai.Content) bool {
		if c == nil {
			return false
		}
		for _, part := range c.Parts {
			if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == finishTaskToolName {
				return true
			}
		}
		return false
	}

	runnerTaskFinishResp := func(callID string) *model.LLMResponse {
		return &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: finishTaskToolName,
						Args: map[string]any{"result": "done"},
					},
				}},
			},
		}
	}

	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	for _, mode := range []struct {
		name            string
		emitOnToolError bool
		call1Result     map[string]any
	}{
		{
			name:            "OnToolErrorCallback",
			emitOnToolError: true,
			call1Result:     map[string]any{"result": "Task completed."},
		},
		{
			name:            "retryandreflect",
			emitOnToolError: false,
			call1Result: map[string]any{
				"response_type": "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN",
				"error_details": "finish failed",
				"result":        "Task completed.",
			},
		},
	} {
		t.Run("runner/"+mode.name+"/root drops Attempt 1 events with reused callID", func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var runCount atomic.Int32
			recoverPlugin, err := baseplugin.New(baseplugin.Config{
				Name: "recover_first_finish_task",
				AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil && tl != nil && tl.Name() == finishTaskToolName && runCount.Load() == 1 {
						if mode.emitOnToolError {
							if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish failed")); cbErr != nil {
								return nil, cbErr
							}
						} else {
							result["response_type"] = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
							result["error_details"] = "finish failed"
						}
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(recover): %v", err)
			}

			taskModel := &scriptedLLM{
				responses: []*model.LLMResponse{
					runnerTaskFinishResp("call-finish-1"),
					runnerTaskFinishResp("call-finish-1"),
				},
			}
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "task",
				Mode:  llmagent.ModeTask,
				Model: taskModel,
			})
			if err != nil {
				t.Fatalf("llmagent.New(task): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name: "root",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
						for range 2 {
							n := runCount.Add(1)
							for ev, err := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
								if err != nil {
									yield(nil, err)
									return
								}
								if n == 1 {
									continue
								}
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

			for _, runErr := range newTestRunner(t, root, recoverPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
			}
			starts := agentRowsOfType(rows, EventAgentStarting, "task")
			ends := agentRowsOfType(rows, EventAgentCompleted, "task")
			allCompleted := rowsOfType(rows, EventAgentCompleted)
			if len(starts) != 2 {
				t.Fatalf("got %d task AGENT_STARTING rows, want 2", len(starts))
			}
			if len(ends) != 1 || len(allCompleted) != 2 {
				t.Fatalf("got %d task AGENT_COMPLETED and %d total AGENT_COMPLETED, want 1 and 2; events=%v",
					len(ends), len(allCompleted), eventTypes(rows))
			}
			if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
				t.Errorf("task AGENT_COMPLETED span_id = %v, want Attempt 2 span_id %v (Attempt 1=%v)", got, want, starts[0]["span_id"])
			}
			onlyRowOfType(t, rows, EventToolError)
			onlyRowOfType(t, rows, EventToolCompleted)
		})

		t.Run("runner/"+mode.name+"/earlier OnEventCallback suppresses Attempt 1 FR with reused callID", func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			var runCount atomic.Int32
			recoverPlugin, err := baseplugin.New(baseplugin.Config{
				Name: "recover_first_finish_task",
				AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil && tl != nil && tl.Name() == finishTaskToolName && runCount.Load() == 1 {
						if mode.emitOnToolError {
							if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish failed")); cbErr != nil {
								return nil, cbErr
							}
						} else {
							result["response_type"] = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
							result["error_details"] = "finish failed"
						}
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(recover): %v", err)
			}

			var suppressed atomic.Bool
			suppressFirstFR, err := baseplugin.New(baseplugin.Config{
				Name: "suppress_first_finish_task_fr",
				OnEventCallback: func(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
					if ev != nil && hasFinishTaskFR(ev.Content) && suppressed.CompareAndSwap(false, true) {
						return &session.Event{
							Author: ev.Author,
							Branch: ev.Branch,
							LLMResponse: model.LLMResponse{
								Content: genai.NewContentFromParts([]*genai.Part{{Text: "suppressed"}}, genai.RoleModel),
							},
						}, nil
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New(suppress): %v", err)
			}

			taskModel := &scriptedLLM{
				responses: []*model.LLMResponse{
					runnerTaskFinishResp("call-finish-1"),
					runnerTaskFinishResp("call-finish-1"),
				},
			}
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "task",
				Mode:  llmagent.ModeTask,
				Model: taskModel,
			})
			if err != nil {
				t.Fatalf("llmagent.New(task): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name: "root",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
						for range 2 {
							runCount.Add(1)
							for ev, err := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
								if err != nil {
									yield(nil, err)
									return
								}
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

			for _, runErr := range newTestRunner(t, root, recoverPlugin, suppressFirstFR, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}

			if !suppressed.Load() {
				t.Fatal("expected earlier OnEventCallback plugin to intercept Attempt 1 finish_task FR")
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
			}
			starts := agentRowsOfType(rows, EventAgentStarting, "task")
			ends := agentRowsOfType(rows, EventAgentCompleted, "task")
			allCompleted := rowsOfType(rows, EventAgentCompleted)
			if len(starts) != 2 {
				t.Fatalf("got %d task AGENT_STARTING rows, want 2", len(starts))
			}
			if len(ends) != 1 || len(allCompleted) != 2 {
				t.Fatalf("got %d task AGENT_COMPLETED and %d total AGENT_COMPLETED, want 1 and 2; events=%v",
					len(ends), len(allCompleted), eventTypes(rows))
			}
			if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
				t.Errorf("task AGENT_COMPLETED span_id = %v, want Attempt 2 span_id %v (Attempt 1=%v)", got, want, starts[0]["span_id"])
			}
			onlyRowOfType(t, rows, EventToolError)
			onlyRowOfType(t, rows, EventToolCompleted)
		})

		for _, callTC := range []struct {
			name   string
			callID string
		}{
			{name: "non-empty callID", callID: "call-finish-1"},
			{name: "empty callID", callID: ""},
		} {
			t.Run("callback/"+mode.name+"/cross-activation dropped FR "+callTC.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				ic1 := &runningInvocation{running: taskAgent}
				ic2 := &runningInvocation{running: taskAgent}
				act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
				act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
				call1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
				call2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic2}
				finish := &fakeTool{name: finishTaskToolName}
				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
				modelFC := &session.Event{
					Author: "test-agent",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   callTC.callID,
								Name: finishTaskToolName,
								Args: map[string]any{"summary": "attempt1"},
							},
						}}, genai.RoleModel),
					},
				}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(act1))
				errs.add(p.OnEventCallback()(runnerCtx, modelFC))
				errs.add(p.BeforeToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}))
				if mode.emitOnToolError {
					errs.add(p.OnToolErrorCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, errors.New("finish failed")))
				}
				errs.add(p.AfterToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, mode.call1Result, nil))
				// Attempt 1's FunctionResponse is suppressed/dropped before OnEventCallback.
				errs.add(p.BeforeAgentCallback()(act2))
				errs.add(p.BeforeToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}))
				errs.add(p.AfterToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}, map[string]any{"result": "Task completed."}, nil))

				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
				errs.check(t)
				p.AfterRunCallback()(act2)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
					t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
				}
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				if len(starts) != 2 {
					t.Fatalf("got %d AGENT_STARTING rows, want 2", len(starts))
				}
				if len(ends) != 1 {
					t.Fatalf("got %d AGENT_COMPLETED rows, want 1; events=%v", len(ends), eventTypes(rows))
				}
				if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
					t.Errorf("AGENT_COMPLETED span_id = %v, want act2 span_id %v (act1=%v)", got, want, starts[0]["span_id"])
				}
				onlyRowOfType(t, rows, EventToolError)
				onlyRowOfType(t, rows, EventToolCompleted)
			})

			t.Run("callback/"+mode.name+"/cross-activation short-circuited Attempt 2 AfterTool "+callTC.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				ic1 := &runningInvocation{running: taskAgent}
				ic2 := &runningInvocation{running: taskAgent}
				act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
				act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
				call1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
				finish := &fakeTool{name: finishTaskToolName}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(act1))
				errs.add(p.BeforeToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}))
				if mode.emitOnToolError {
					errs.add(p.OnToolErrorCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, errors.New("finish failed")))
				}
				errs.add(p.AfterToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, mode.call1Result, nil))
				// Attempt 1 FR is dropped; Attempt 2 starts on the same goroutine,
				// emits its finish_task FunctionCall, and an earlier AfterToolCallback
				// plugin short-circuits Attempt 2's BeforeTool and AfterTool callbacks
				// so noteTaskFinished never runs on act2.
				errs.add(p.BeforeAgentCallback()(act2))
				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
				modelFC2 := &session.Event{
					Author: "test-agent",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   callTC.callID,
								Name: finishTaskToolName,
								Args: map[string]any{"summary": "attempt2"},
							},
						}}, genai.RoleModel),
					},
				}
				errs.add(p.OnEventCallback()(runnerCtx, modelFC2))
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
				errs.check(t)
				p.AfterRunCallback()(act2)

				// Attempt 1's rejected response was dropped while Attempt 2's tool
				// callbacks were both short-circuited: Attempt 2's pending finish_task
				// FunctionCall records a deferred completion on the ambiguous response,
				// which settles as Attempt 2's AGENT_COMPLETED when the run ends
				// without further activity on Attempt 2.
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				if len(starts) != 2 {
					t.Fatalf("got %d AGENT_STARTING rows, want 2", len(starts))
				}
				if len(ends) != 1 {
					t.Fatalf("got %d AGENT_COMPLETED rows, want 1; events=%v", len(ends), eventTypes(rows))
				}
				if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
					t.Errorf("AGENT_COMPLETED span_id = %v, want act2 span_id %v (act1=%v)", got, want, starts[0]["span_id"])
				}
			})

			t.Run("callback/"+mode.name+"/single-activation retry suppressed FR "+callTC.name, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				ic1 := &runningInvocation{running: taskAgent}
				act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
				call1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
				call2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
				finish := &fakeTool{name: finishTaskToolName}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(act1))
				errs.add(p.BeforeToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}))
				if mode.emitOnToolError {
					errs.add(p.OnToolErrorCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, errors.New("finish failed")))
				}
				errs.add(p.AfterToolCallback()(call1Ctx, finish, map[string]any{"summary": "attempt1"}, mode.call1Result, nil))
				// First FR suppressed before OnEventCallback; same activation retries cleanly.
				errs.add(p.BeforeToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}))
				errs.add(p.AfterToolCallback()(call2Ctx, finish, map[string]any{"summary": "attempt2"}, map[string]any{"result": "Task completed."}, nil))

				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
				errs.check(t)
				p.AfterRunCallback()(act1)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				if len(starts) != 1 || len(ends) != 1 {
					t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED, want 1 each; events=%v", len(starts), len(ends), eventTypes(rows))
				}
				if got, want := ends[0]["span_id"], starts[0]["span_id"]; got != want {
					t.Errorf("AGENT_COMPLETED span_id = %v, want act1 span_id %v", got, want)
				}
				onlyRowOfType(t, rows, EventToolError)
				onlyRowOfType(t, rows, EventToolCompleted)
			})
		}
	}

	t.Run("concurrent activations rejected and succeeded with shared callID across both FR orders", func(t *testing.T) {
		for _, callTC := range []struct {
			name   string
			callID string
		}{
			{name: "non-empty callID", callID: "call-finish-shared"},
			{name: "empty callID", callID: ""},
		} {
			for _, branchMode := range []string{"shared-branch", "distinct-branch"} {
				for _, finishFirst := range []bool{false, true} {
					for _, frOrderFinishFirst := range []bool{false, true} {
						orderName := "rejectFR-then-finishFR"
						if frOrderFinishFirst {
							orderName = "finishFR-then-rejectFR"
						}
						pushOrder := "reject-pushed-first"
						if finishFirst {
							pushOrder = "finish-pushed-first"
						}
						t.Run(callTC.name+"/"+branchMode+"/"+pushOrder+"/"+orderName, func(t *testing.T) {
							p, requests, mCtx := setupTestPlugin(t)
							taskAgent, err := llmagent.New(llmagent.Config{
								Name:  "test-agent",
								Mode:  llmagent.ModeTask,
								Model: &scriptedLLM{},
							})
							if err != nil {
								t.Fatalf("llmagent.New: %v", err)
							}

							branchReject, branchFinish := "", ""
							if branchMode == "distinct-branch" {
								branchReject = "fan-reject"
								branchFinish = "fan-finish"
							}
							mReject := *mCtx
							mReject.branch = branchReject
							mFinish := *mCtx
							mFinish.branch = branchFinish

							actReject := &taskActivationContext{mockInvocationContext: &mReject, ic: &runningInvocation{running: taskAgent}}
							actFinish := &taskActivationContext{mockInvocationContext: &mFinish, ic: &runningInvocation{running: taskAgent}}
							toolRejectCtx := &taskActivationContext{mockInvocationContext: withFunctionCall(&mReject, callTC.callID), ic: actReject.ic}
							toolFinishCtx := &taskActivationContext{mockInvocationContext: withFunctionCall(&mFinish, callTC.callID), ic: actFinish.ic}
							finishTool := &fakeTool{name: finishTaskToolName}

							var mu sync.Mutex
							var errs callbackErrors
							addErr := func(v any, err error) {
								mu.Lock()
								errs.add(v, err)
								mu.Unlock()
							}
							runReject := func() {
								addErr(p.BeforeAgentCallback()(actReject))
								addErr(p.BeforeToolCallback()(toolRejectCtx, finishTool, map[string]any{"summary": "bad"}))
								addErr(p.OnToolErrorCallback()(toolRejectCtx, finishTool, map[string]any{"summary": "bad"}, errors.New("boom")))
								addErr(p.AfterToolCallback()(toolRejectCtx, finishTool, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))
							}
							runFinish := func() {
								addErr(p.BeforeAgentCallback()(actFinish))
								addErr(p.BeforeToolCallback()(toolFinishCtx, finishTool, map[string]any{"summary": "good"}))
								addErr(p.AfterToolCallback()(toolFinishCtx, finishTool, map[string]any{"summary": "good"}, map[string]any{"result": "Task completed."}, nil))
							}

							firstDone := make(chan struct{})
							secondDone := make(chan struct{})
							if finishFirst {
								go func() { runFinish(); close(firstDone) }()
								<-firstDone
								go func() { runReject(); close(secondDone) }()
								<-secondDone
							} else {
								go func() { runReject(); close(firstDone) }()
								<-firstDone
								go func() { runFinish(); close(secondDone) }()
								<-secondDone
							}

							runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
							if frOrderFinishFirst {
								errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", branchFinish, callTC.callID)))
								errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", branchReject, callTC.callID)))
							} else {
								errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", branchReject, callTC.callID)))
								errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", branchFinish, callTC.callID)))
							}
							errs.check(t)
							p.AfterRunCallback()(mCtx)

							rows := collectRows(t, requests, untilInvocationEnd)
							requireInvocationCompletedOK(t, rows)
							starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
							ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
							if len(starts) != 2 {
								t.Fatalf("got %d AGENT_STARTING rows, want 2", len(starts))
							}
							if len(ends) != 1 {
								t.Fatalf("got %d AGENT_COMPLETED rows, want 1; events=%v", len(ends), eventTypes(rows))
							}
							wantFinishSpan := starts[1]["span_id"]
							if finishFirst {
								wantFinishSpan = starts[0]["span_id"]
							}
							if got := ends[0]["span_id"]; got != wantFinishSpan {
								t.Errorf("AGENT_COMPLETED span_id = %v, want finished activation span_id %v (starts=%v/%v)",
									got, wantFinishSpan, starts[0]["span_id"], starts[1]["span_id"])
							}
							onlyRowOfType(t, rows, EventToolError)
							onlyRowOfType(t, rows, EventToolCompleted)
						})
					}
				}
			}
		}
	})
}

func TestPlugin_RecoveredToolError_SkippedCallbacksReusedID(t *testing.T) {
	for _, tc := range []struct {
		name           string
		skipBefore1    bool
		attempt2CallID string
	}{
		{name: "reused_id_skipped_after1_skipped_before2", attempt2CallID: "call_0"},
		{name: "reused_id_skipped_before1_after1_before2", skipBefore1: true, attempt2CallID: "call_0"},
		{name: "distinct_id_control", attempt2CallID: "call_1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			ctx1 := withFunctionCall(mCtx, "call_0")
			ctx2 := withFunctionCall(mCtx, tc.attempt2CallID)
			final := &fakeTool{name: "final_answer"}
			args1 := map[string]any{"answer": "bad"}
			args2 := map[string]any{"answer": "good"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			if !tc.skipBefore1 {
				errs.add(p.BeforeToolCallback()(ctx1, final, args1))
			}
			errs.add(p.OnToolErrorCallback()(ctx1, final, args1, errors.New("boom-attempt-1")))
			// Attempt 1 AfterToolCallback and Attempt 2 BeforeToolCallback are skipped.
			errs.add(p.AfterToolCallback()(ctx2, final, args2, map[string]any{"answer": "good"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			errRow := onlyRowOfType(t, rows, EventToolError)
			compRow := onlyRowOfType(t, rows, EventToolCompleted)
			if tc.skipBefore1 {
				if errRow["latency_ms"] != nil {
					t.Errorf("TOOL_ERROR latency_ms = %v, want nil when BeforeTool skipped", errRow["latency_ms"])
				}
			} else {
				startRow := onlyRowOfType(t, rows, EventToolStarting)
				if got, want := placeOf(errRow), placeOf(startRow); got != want {
					t.Errorf("TOOL_ERROR span = %+v, want %+v", got, want)
				}
				mustLatency(t, errRow)
			}
			if got := placeOf(compRow).span; got == placeOf(errRow).span {
				t.Errorf("Attempt 2 TOOL_COMPLETED reused Attempt 1 TOOL_ERROR span %v", got)
			}
			respRow := onlyRowOfType(t, rows, EventAgentResponse)
			checkJSON(t, "Attempt 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)
		})
	}

	t.Run("same_context_skipped_before_suppresses_recovered_completion", func(t *testing.T) {
		p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
			c.FinalResponseToolNames = []string{"final_answer"}
		})
		ctx1 := withFunctionCall(mCtx, "call_0")
		final := &fakeTool{name: "final_answer"}
		args := map[string]any{"answer": "bad"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.OnToolErrorCallback()(ctx1, final, args, errors.New("boom-same")))
		errs.add(p.AfterToolCallback()(ctx1, final, args, map[string]any{"answer": "recovered"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		onlyRowOfType(t, rows, EventToolError)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Fatalf("got %d TOOL_COMPLETED rows, want 0; events=%v", n, eventTypes(rows))
		}
		if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
			t.Fatalf("got %d AGENT_RESPONSE rows, want 0; events=%v", n, eventTypes(rows))
		}
	})
}

func TestPlugin_RecoveredToolError_ValueContextSkippedBefore(t *testing.T) {
	ctxKinds := []struct {
		name      string
		isPointer bool
		pair      func(*mockInvocationContext) (agent.Context, agent.Context)
	}{
		{
			name: "value",
			pair: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				return comparableValueToolCtx{mockInvocationContext: base, tag: "same"},
					comparableValueToolCtx{mockInvocationContext: base, tag: "same"}
			},
		},
		{
			name: "slice",
			pair: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				return nonComparableSliceToolCtx{mockInvocationContext: base, tags: []string{"c1"}},
					nonComparableSliceToolCtx{mockInvocationContext: base, tags: []string{"c2"}}
			},
		},
		{
			name: "closure",
			pair: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				base := withFunctionCall(m, "")
				v1, v2 := 1, 2
				return nonComparableFuncToolCtx{mockInvocationContext: base, fn: func() int { return v1 }},
					nonComparableFuncToolCtx{mockInvocationContext: base, fn: func() int { return v2 }}
			},
		},
		{
			name:      "pointer",
			isPointer: true,
			pair: func(m *mockInvocationContext) (agent.Context, agent.Context) {
				return withFunctionCall(m, ""), withFunctionCall(m, "")
			},
		},
	}

	for _, ck := range ctxKinds {
		t.Run(ck.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			ctx1, ctx2 := ck.pair(mCtx)
			final := &fakeTool{name: "final_answer"}
			args1 := map[string]any{"answer": "bad"}
			args2 := map[string]any{"answer": "good"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			// Call 1 skips both BeforeTool and AfterTool; Call 2 skips BeforeTool.
			errs.add(p.OnToolErrorCallback()(ctx1, final, args1, errors.New("boom-skipped-before")))
			errs.add(p.AfterToolCallback()(ctx2, final, args2, map[string]any{"answer": "good"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			errRow := onlyRowOfType(t, rows, EventToolError)
			compRow := onlyRowOfType(t, rows, EventToolCompleted)
			if got := placeOf(compRow).span; got == placeOf(errRow).span {
				t.Errorf("Call 2 TOOL_COMPLETED reused Call 1 TOOL_ERROR span %v", got)
			}
			respRow := onlyRowOfType(t, rows, EventAgentResponse)
			checkJSON(t, "Call 2 AGENT_RESPONSE", jsonObjectColumn(t, respRow, "content"), `{"response":{"answer":"good"}}`)

			if ck.isPointer {
				pSame, reqSame, mSame := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				samePtr, _ := ck.pair(mSame)
				var errsSame callbackErrors
				errsSame.add(pSame.BeforeAgentCallback()(mSame))
				errsSame.add(pSame.OnToolErrorCallback()(samePtr, final, args1, errors.New("boom-same-ptr")))
				errsSame.add(pSame.AfterToolCallback()(samePtr, final, args1, map[string]any{"answer": "recovered"}, nil))
				errsSame.add(pSame.AfterAgentCallback()(mSame))
				errsSame.check(t)
				pSame.AfterRunCallback()(mSame)

				sameRows := collectRows(t, reqSame, untilInvocationEnd)
				requireInvocationCompletedOK(t, sameRows)
				onlyRowOfType(t, sameRows, EventToolError)
				if n := len(rowsOfType(sameRows, EventToolCompleted)); n != 0 {
					t.Fatalf("same-pointer skipped BeforeTool: got %d TOOL_COMPLETED rows, want 0", n)
				}
			}
		})
	}
}

func TestPlugin_RecoveredToolError_RejectedFinishDifferentGoroutine(t *testing.T) {
	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	for _, callTC := range []struct {
		name   string
		callID string
	}{
		{name: "non-empty callID", callID: "call-finish-shared"},
		{name: "empty callID", callID: ""},
	} {
		t.Run(callTC.name, func(t *testing.T) {
			for _, mode := range []struct {
				name       string
				concurrent bool
				openFinish bool
			}{
				// With both of act2's callbacks skipped, act2's model FunctionCall
				// records pendingFinishCalls on act2; when the single finish_task
				// FunctionResponse arrives without StateDelta, act1's rejection is
				// consumed and act2's deferred completion settles cleanly at AfterRun.
				{name: "sequential_different_goroutine_both_callbacks_skipped"},
				{name: "sequential_different_goroutine_after_callback_skipped", openFinish: true},
				{name: "concurrent_different_goroutine_after_callback_skipped", concurrent: true, openFinish: true},
			} {
				t.Run(mode.name, func(t *testing.T) {
					p, requests, mCtx := setupTestPlugin(t)
					taskAgent, err := llmagent.New(llmagent.Config{
						Name:  "test-agent",
						Mode:  llmagent.ModeTask,
						Model: &scriptedLLM{},
					})
					if err != nil {
						t.Fatalf("llmagent.New: %v", err)
					}
					ic1 := &runningInvocation{running: taskAgent}
					ic2 := &runningInvocation{running: taskAgent}
					act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
					act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
					tool1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
					tool2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic2}
					finish := &fakeTool{name: finishTaskToolName}
					runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}

					var mu sync.Mutex
					var errs callbackErrors
					addErr := func(v any, err error) {
						mu.Lock()
						errs.add(v, err)
						mu.Unlock()
					}
					runOnGoroutine := func(fn func()) {
						done := make(chan struct{})
						go func() {
							defer close(done)
							fn()
						}()
						<-done
					}

					suppressedEv := &session.Event{
						Author: "test-agent",
						LLMResponse: model.LLMResponse{
							Content: textContent("model", "suppressed"),
						},
					}
					if mode.concurrent {
						runOnGoroutine(func() {
							addErr(p.BeforeAgentCallback()(act1))
						})
						runOnGoroutine(func() {
							addErr(p.BeforeAgentCallback()(act2))
						})
						runOnGoroutine(func() {
							addErr(p.BeforeToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}))
							addErr(p.OnToolErrorCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
							addErr(p.AfterToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))
						})
						if mode.openFinish {
							runOnGoroutine(func() {
								addErr(p.BeforeToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}))
							})
						}
					} else {
						runOnGoroutine(func() {
							addErr(p.BeforeAgentCallback()(act1))
							addErr(p.BeforeToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}))
							addErr(p.OnToolErrorCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
							addErr(p.AfterToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))
						})
						// Attempt 1's suppressed non-finish_task response reaches OnEvent
						// before Attempt 2 starts, leaving act1 on the stack with an
						// unconsumed rejectedTasks marker when Attempt 2 begins.
						errs.add(p.OnEventCallback()(runnerCtx, suppressedEv))
						runOnGoroutine(func() {
							addErr(p.BeforeAgentCallback()(act2))
							if mode.openFinish {
								addErr(p.BeforeToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}))
							}
						})
						if !mode.openFinish {
							errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
								Author: "test-agent",
								LLMResponse: model.LLMResponse{
									Content: genai.NewContentFromParts([]*genai.Part{{
										FunctionCall: &genai.FunctionCall{
											ID:   callTC.callID,
											Name: finishTaskToolName,
											Args: map[string]any{"summary": "good"},
										},
									}}, genai.RoleModel),
								},
							}))
						}
					}

					errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
					if mode.concurrent {
						// Complete act1 after act2 finished so act1's suppressed response
						// settles act1 cleanly without an AGENT_ERROR at AfterRun.
						errs.add(p.OnEventCallback()(runnerCtx, suppressedEv))
					}
					errs.check(t)
					p.AfterRunCallback()(mCtx)

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)
					if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
						t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
					}
					starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
					ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
					if len(starts) != 2 {
						t.Fatalf("got %d AGENT_STARTING rows, want 2", len(starts))
					}
					onlyRowOfType(t, rows, EventToolError)
					if len(ends) != 1 {
						t.Fatalf("got %d AGENT_COMPLETED rows, want 1; events=%v", len(ends), eventTypes(rows))
					}
					if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
						t.Errorf("AGENT_COMPLETED span_id = %v, want newer activation span_id %v (older=%v)",
							got, want, starts[0]["span_id"])
					}
				})
			}

			t.Run("concurrent_unstarted_peer_does_not_supersede_rejection", func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				ic1 := &runningInvocation{running: taskAgent}
				ic2 := &runningInvocation{running: taskAgent}
				act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
				act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
				tool1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
				finish := &fakeTool{name: finishTaskToolName}

				var mu sync.Mutex
				var errs callbackErrors
				addErr := func(v any, err error) {
					mu.Lock()
					errs.add(v, err)
					mu.Unlock()
				}
				runOnGoroutine := func(fn func()) {
					done := make(chan struct{})
					go func() {
						defer close(done)
						fn()
					}()
					<-done
				}

				runOnGoroutine(func() {
					addErr(p.BeforeAgentCallback()(act1))
				})
				runOnGoroutine(func() {
					addErr(p.BeforeAgentCallback()(act2))
				})
				runOnGoroutine(func() {
					addErr(p.BeforeToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}))
					addErr(p.OnToolErrorCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
					addErr(p.AfterToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))
				})

				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
				// Verify act2 remained open by completing act2's own finish_task
				// afterward and checking both TOOL_COMPLETED and AGENT_COMPLETED bind
				// to act2's span_id.
				tool2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic2}
				runOnGoroutine(func() {
					addErr(p.BeforeToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}))
					addErr(p.AfterToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}, map[string]any{"result": "Task completed."}, nil))
				})
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
					t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
				}
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				if len(starts) != 2 || len(ends) != 1 {
					t.Fatalf("got %d starts and %d ends, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
				}
				if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
					t.Errorf("AGENT_COMPLETED span_id = %v, want act2 span_id %v (act1=%v)", got, want, starts[0]["span_id"])
				}
				comp := onlyRowOfType(t, rows, EventToolCompleted)
				if got, want := comp["parent_span_id"], starts[1]["span_id"]; got != want {
					t.Errorf("TOOL_COMPLETED parent_span_id = %v, want act2 span_id %v", got, want)
				}
				onlyRowOfType(t, rows, EventToolError)
			})
		})
	}
}

func TestPlugin_RecoveredToolError_DelayedRejectedFinishLeavesNewPeerOpen(t *testing.T) {
	finishTaskFCEvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: finishTaskToolName,
						Args: map[string]any{"summary": "done"},
					},
				}}, genai.RoleModel),
			},
		}
	}
	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	for _, callTC := range []struct {
		name   string
		callID string
	}{
		{name: "non-empty callID", callID: "call-finish-shared"},
		{name: "empty callID", callID: ""},
	} {
		t.Run(callTC.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "test-agent",
				Mode:  llmagent.ModeTask,
				Model: &scriptedLLM{},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			ic1 := &runningInvocation{running: taskAgent}
			ic2 := &runningInvocation{running: taskAgent}
			act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
			act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
			tool1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic1}
			tool2Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callTC.callID), ic: ic2}
			finish := &fakeTool{name: finishTaskToolName}
			runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(act1))
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", callTC.callID)))
			errs.add(p.BeforeToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}))
			errs.add(p.OnToolErrorCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
			errs.add(p.AfterToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))

			// Peer B starts before A's buffered FunctionResponse arrives.
			errs.add(p.BeforeAgentCallback()(act2))
			// A's delayed rejected finish_task FunctionResponse must NOT retire B.
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))

			// B still owns an open finish_task call and finishes cleanly on B's own span.
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", callTC.callID)))
			errs.add(p.BeforeToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}))
			errs.add(p.AfterToolCallback()(tool2Ctx, finish, map[string]any{"summary": "good"}, map[string]any{"result": "Task completed."}, nil))
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", callTC.callID)))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
			}
			starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
			ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
			if len(starts) != 2 || len(ends) != 1 {
				t.Fatalf("got %d starts and %d ends, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
			}
			if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
				t.Errorf("AGENT_COMPLETED span_id = %v, want peer B span_id %v (A=%v)", got, want, starts[0]["span_id"])
			}
			comp := onlyRowOfType(t, rows, EventToolCompleted)
			if got, want := comp["parent_span_id"], starts[1]["span_id"]; got != want {
				t.Errorf("TOOL_COMPLETED parent_span_id = %v, want peer B span_id %v", got, want)
			}
			onlyRowOfType(t, rows, EventToolError)
			compIdx := slices.IndexFunc(rows, func(r map[string]any) bool { return r["event_type"] == EventToolCompleted })
			endIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
				return r["event_type"] == EventAgentCompleted && r["span_id"] == starts[1]["span_id"]
			})
			if compIdx < 0 || endIdx < compIdx {
				t.Errorf("expected TOOL_COMPLETED (idx=%d) before B AGENT_COMPLETED (idx=%d); events=%v", compIdx, endIdx, eventTypes(rows))
			}
		})
	}
}

func TestPlugin_RecoveredToolError_RunnerDelayedRejectedFinish(t *testing.T) {
	finishTaskResp := func(callID string) *model.LLMResponse {
		return &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: finishTaskToolName,
						Args: map[string]any{"result": "done"},
					},
				}},
			},
		}
	}
	firstBatchParts := func(order, callID string) []*genai.Part {
		finish := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   callID,
				Name: finishTaskToolName,
				Args: map[string]any{"result": "done"},
			},
		}
		echo := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   "call-echo-a",
				Name: "echo",
				Args: map[string]any{"msg": "a"},
			},
		}
		switch order {
		case "single_call":
			return []*genai.Part{finish}
		case "finish_first":
			return []*genai.Part{finish, echo}
		case "finish_second":
			return []*genai.Part{echo, finish}
		default:
			t.Fatalf("unknown batch order %q", order)
			return nil
		}
	}
	hasFinishTaskFC := func(c *genai.Content) bool {
		if c == nil {
			return false
		}
		for _, part := range c.Parts {
			if part != nil && part.FunctionCall != nil && part.FunctionCall.Name == finishTaskToolName {
				return true
			}
		}
		return false
	}
	hasFinishTaskFR := func(c *genai.Content) bool {
		if c == nil {
			return false
		}
		for _, part := range c.Parts {
			if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == finishTaskToolName {
				return true
			}
		}
		return false
	}

	for _, callTC := range []struct {
		name    string
		callID1 string
		callID2 string
	}{
		{name: "distinct_call_ids", callID1: "call-finish-a", callID2: "call-finish-b"},
		{name: "reused_call_id", callID1: "call-finish-shared", callID2: "call-finish-shared"},
	} {
		for _, batchOrder := range []string{"single_call", "finish_first", "finish_second"} {
			for _, flushAt := range []string{"after_peer_b_echo", "after_peer_b_finish_fc"} {
				name := callTC.name + "/" + batchOrder + "/" + flushAt
				t.Run(name, func(t *testing.T) {
					p, requests, _ := setupTestPlugin(t)
					var runCount atomic.Int32
					recoverPlugin, err := baseplugin.New(baseplugin.Config{
						Name: "recover_a_finish_task",
						AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
							if err == nil && tl != nil && tl.Name() == finishTaskToolName && runCount.Load() == 1 {
								if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish rejected")); cbErr != nil {
									return nil, cbErr
								}
							}
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New: %v", err)
					}

					echoTool, err := functiontool.New(functiontool.Config{
						Name:        "echo",
						Description: "Echo input",
					}, func(_ agent.Context, args map[string]any) (map[string]any, error) {
						return map[string]any{"echo": args["msg"]}, nil
					})
					if err != nil {
						t.Fatalf("functiontool.New(echo): %v", err)
					}

					taskModel := &scriptedLLM{
						responses: []*model.LLMResponse{
							{Content: &genai.Content{Role: "model", Parts: firstBatchParts(batchOrder, callTC.callID1)}},
							{
								Content: &genai.Content{
									Role: "model",
									Parts: []*genai.Part{{
										FunctionCall: &genai.FunctionCall{
											ID:   "call-echo-b",
											Name: "echo",
											Args: map[string]any{"msg": "working"},
										},
									}},
								},
							},
							finishTaskResp(callTC.callID2),
						},
					}
					taskAgent, err := llmagent.New(llmagent.Config{
						Name:  "task",
						Mode:  llmagent.ModeTask,
						Model: taskModel,
						Tools: []tool.Tool{echoTool},
					})
					if err != nil {
						t.Fatalf("llmagent.New(task): %v", err)
					}

					flushedA := false
					root, err := agent.New(agent.Config{
						Name: "root",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
								var bufferedA *session.Event
								for range 2 {
									n := runCount.Add(1)
									for ev, iterErr := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
										if iterErr != nil {
											yield(nil, iterErr)
											return
										}
										if n == 1 && ev != nil && hasFinishTaskFR(ev.Content) {
											bufferedA = ev
											continue
										}
										if !yield(ev, nil) {
											return
										}
										if n == 2 && !flushedA && bufferedA != nil && ev != nil && !ev.Partial && ev.Author == "task" && !hasFinishTaskFR(ev.Content) {
											trigger := false
											switch flushAt {
											case "after_peer_b_echo":
												trigger = !hasFinishTaskFC(ev.Content)
											case "after_peer_b_finish_fc":
												trigger = hasFinishTaskFC(ev.Content)
											}
											if trigger {
												flushedA = true
												if !yield(bufferedA, nil) {
													return
												}
											}
										}
									}
								}
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(root): %v", err)
					}

					for _, runErr := range newTestRunner(t, root, recoverPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
						if runErr != nil {
							t.Fatalf("runner.Run error: %v", runErr)
						}
					}
					if !flushedA {
						t.Fatal("buffered rejected finish_task response was never flushed")
					}

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)
					if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
						t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
					}
					starts := agentRowsOfType(rows, EventAgentStarting, "task")
					ends := agentRowsOfType(rows, EventAgentCompleted, "task")
					if len(starts) != 2 || len(ends) != 1 {
						t.Fatalf("got %d task starts and %d task ends, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
					}
					if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
						t.Errorf("task AGENT_COMPLETED span_id = %v, want peer B span_id %v", got, want)
					}
					finishCompIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
						return r["event_type"] == EventToolCompleted && r["parent_span_id"] == starts[1]["span_id"] &&
							jsonObjectColumn(t, r, "content")["tool"] == finishTaskToolName
					})
					endIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
						return r["event_type"] == EventAgentCompleted && r["span_id"] == starts[1]["span_id"]
					})
					if finishCompIdx < 0 || endIdx < finishCompIdx {
						t.Errorf("expected B finish_task TOOL_COMPLETED (idx=%d) before B AGENT_COMPLETED (idx=%d); events=%v",
							finishCompIdx, endIdx, eventTypes(rows))
					}
				})
			}
		}
	}
}

func TestPlugin_RecoveredToolError_RunnerRejectedFinishSingleRun(t *testing.T) {
	firstBatchParts := func(order string) []*genai.Part {
		finish := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   "call-finish-a",
				Name: finishTaskToolName,
				Args: map[string]any{"result": "done"},
			},
		}
		echo := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   "call-echo-a",
				Name: "echo",
				Args: map[string]any{"msg": "a"},
			},
		}
		switch order {
		case "single_call":
			return []*genai.Part{finish}
		case "finish_first":
			return []*genai.Part{finish, echo}
		case "finish_second":
			return []*genai.Part{echo, finish}
		default:
			t.Fatalf("unknown batch order %q", order)
			return nil
		}
	}

	for _, batchOrder := range []string{"single_call", "finish_first", "finish_second"} {
		t.Run(batchOrder, func(t *testing.T) {
			p, requests, _ := setupTestPlugin(t)
			recoverPlugin, err := baseplugin.New(baseplugin.Config{
				Name: "recover_a_finish_task",
				AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
					if err == nil && tl != nil && tl.Name() == finishTaskToolName {
						if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish rejected")); cbErr != nil {
							return nil, cbErr
						}
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("baseplugin.New: %v", err)
			}
			echoTool, err := functiontool.New(functiontool.Config{
				Name:        "echo",
				Description: "Echo input",
			}, func(_ agent.Context, args map[string]any) (map[string]any, error) {
				return map[string]any{"echo": args["msg"]}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New(echo): %v", err)
			}
			taskModel := &scriptedLLM{
				responses: []*model.LLMResponse{
					{Content: &genai.Content{Role: "model", Parts: firstBatchParts(batchOrder)}},
				},
			}
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "task",
				Mode:  llmagent.ModeTask,
				Model: taskModel,
				Tools: []tool.Tool{echoTool},
			})
			if err != nil {
				t.Fatalf("llmagent.New(task): %v", err)
			}
			root, err := agent.New(agent.Config{
				Name: "root",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
						for ev, iterErr := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
							if iterErr != nil {
								yield(nil, iterErr)
								return
							}
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
			for _, runErr := range newTestRunner(t, root, recoverPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}
			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(agentRowsOfType(rows, EventAgentCompleted, "task")); got != 0 {
				t.Errorf("%s: got %d task AGENT_COMPLETED rows, want 0; events=%v", batchOrder, got, eventTypes(rows))
			}
			if got := len(rowsOfType(rows, EventToolError)); got != 1 {
				t.Errorf("%s: got %d TOOL_ERROR rows, want 1; events=%v", batchOrder, got, eventTypes(rows))
			}
		})
	}
}

func TestPlugin_RecoveredToolError_RejectedFinishPeerStartedAfter(t *testing.T) {
	finishTaskFCEvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: finishTaskToolName,
						Args: map[string]any{"summary": "done"},
					},
				}}, genai.RoleModel),
			},
		}
	}
	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	p, requests, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	ic1 := &runningInvocation{running: taskAgent}
	ic2 := &runningInvocation{running: taskAgent}
	act1 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic1}
	act2 := &taskActivationContext{mockInvocationContext: mCtx, ic: ic2}
	tool1Ctx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish-shared"), ic: ic1}
	finish := &fakeTool{name: finishTaskToolName}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(act1))
	errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", "call-finish-shared")))
	errs.add(p.BeforeToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}))
	errs.add(p.OnToolErrorCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
	errs.add(p.AfterToolCallback()(tool1Ctx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))

	// Peer B starts after A's rejection and stays unfinished while A's buffered FR arrives.
	errs.add(p.BeforeAgentCallback()(act2))
	errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", "call-finish-shared")))

	// B then calls finish_task, earlier plugin skips BeforeTool+AfterTool, and B's FR arrives.
	errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", "call-finish-shared")))
	errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", "call-finish-shared")))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	requireInvocationCompletedOK(t, rows)
	if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
		t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
	}
	starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
	ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
	if len(starts) != 2 || len(ends) != 1 {
		t.Fatalf("got %d starts and %d ends, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
	}
	if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
		t.Errorf("AGENT_COMPLETED span_id = %v, want peer B span_id %v (A=%v)", got, want, starts[0]["span_id"])
	}
	onlyRowOfType(t, rows, EventToolError)
}

func TestPlugin_RecoveredToolError_RejectedFinishConcurrentBothSkipped(t *testing.T) {
	finishTaskFCEvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   callID,
						Name: finishTaskToolName,
						Args: map[string]any{"summary": "done"},
					},
				}}, genai.RoleModel),
			},
		}
	}
	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	for _, fcSource := range []string{"after_model", "before_model_short_circuit_event"} {
		t.Run(fcSource, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "test-agent",
				Mode:  llmagent.ModeTask,
				Model: &scriptedLLM{},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			icB := &runningInvocation{running: taskAgent}
			icA := &runningInvocation{running: taskAgent}
			actB := &taskActivationContext{mockInvocationContext: mCtx, ic: icB}
			actA := &taskActivationContext{mockInvocationContext: mCtx, ic: icA}
			toolACtx := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish-shared"), ic: icA}
			finish := &fakeTool{name: finishTaskToolName}
			runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
			suppressedEv := &session.Event{
				Author: "test-agent",
				LLMResponse: model.LLMResponse{
					Content: textContent("model", "suppressed"),
				},
			}

			var errs callbackErrors
			// Peer B starts BEFORE A!
			errs.add(p.BeforeAgentCallback()(actB))
			// A starts, emits finish_task, fails and gets recovered -> A holds rejectedTasks.
			errs.add(p.BeforeAgentCallback()(actA))
			errs.add(p.BeforeModelCallback()(actA, &model.LLMRequest{}))
			errs.add(p.AfterModelCallback()(actA, &model.LLMResponse{
				Content: finishTaskFCEvent("test-agent", "", "call-finish-shared").Content,
			}, nil))
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", "call-finish-shared")))
			errs.add(p.BeforeToolCallback()(toolACtx, finish, map[string]any{"summary": "bad"}))
			errs.add(p.OnToolErrorCallback()(toolACtx, finish, map[string]any{"summary": "bad"}, errors.New("boom")))
			errs.add(p.AfterToolCallback()(toolACtx, finish, map[string]any{"summary": "bad"}, map[string]any{"result": "Task completed."}, nil))

			// Peer B emits finish_task while both BeforeTool and AfterTool are skipped for B.
			if fcSource == "after_model" {
				errs.add(p.BeforeModelCallback()(actB, &model.LLMRequest{}))
				errs.add(p.AfterModelCallback()(actB, &model.LLMResponse{
					Content: finishTaskFCEvent("test-agent", "", "call-finish-shared").Content,
				}, nil))
			}
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFCEvent("test-agent", "", "call-finish-shared")))

			// Peer B's finish_task FunctionResponse arrives BEFORE A's buffered FR!
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", "call-finish-shared")))
			// A's delayed rejected FR arrives later; A finishes with suppressedEv so invocation closes OK.
			errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", "call-finish-shared")))
			errs.add(p.OnEventCallback()(runnerCtx, suppressedEv))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(rowsOfType(rows, EventAgentError)); got != 0 {
				t.Fatalf("got %d AGENT_ERROR rows, want 0; events=%v", got, eventTypes(rows))
			}
			starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
			ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
			if len(starts) != 2 || len(ends) != 1 {
				t.Fatalf("got %d starts and %d ends, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
			}
			// starts[0] is peer B, starts[1] is A.
			if got, want := ends[0]["span_id"], starts[0]["span_id"]; got != want {
				t.Errorf("AGENT_COMPLETED span_id = %v, want peer B span_id %v (A=%v)", got, want, starts[1]["span_id"])
			}
			onlyRowOfType(t, rows, EventToolError)
		})
	}
}

func TestPlugin_RecoveredToolError_ReusedContextSkippedBefore(t *testing.T) {
	for _, callTC := range []struct {
		name   string
		callID string
	}{
		{name: "non-empty callID", callID: "call-reused-ctx"},
		{name: "empty callID", callID: ""},
	} {
		t.Run(callTC.name, func(t *testing.T) {
			t.Run("changed_args_reusing_same_pointer_context", func(t *testing.T) {
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				ctx1 := withFunctionCall(mCtx, callTC.callID)
				final := &fakeTool{name: "final_answer"}
				args1 := map[string]any{"answer": "bad"}
				args2 := map[string]any{"answer": "good"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctx1, final, args1))
				errs.add(p.OnToolErrorCallback()(ctx1, final, args1, errors.New("boom")))
				// Attempt 1 AfterTool skipped; Attempt 2 BeforeTool skipped; Attempt 2 reuses ctx1 with args2.
				errs.add(p.AfterToolCallback()(ctx1, final, args2, map[string]any{"answer": "good"}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				comp := onlyRowOfType(t, rows, EventToolCompleted)
				checkJSON(t, "retry TOOL_COMPLETED", jsonObjectColumn(t, comp, "content"),
					`{"tool":"final_answer","result":{"answer":"good"},"tool_origin":"UNKNOWN"}`)
				resp := onlyRowOfType(t, rows, EventAgentResponse)
				checkJSON(t, "retry AGENT_RESPONSE", jsonObjectColumn(t, resp, "content"),
					`{"response":{"answer":"good"}}`)
			})

			for _, lateMode := range []bool{false, true} {
				subName := "turn_bump_reusing_same_context_same_args_intra_run"
				if lateMode {
					subName = "turn_bump_reusing_same_context_same_args_across_AfterRun"
				}
				t.Run(subName, func(t *testing.T) {
					p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
						c.FinalResponseToolNames = []string{"final_answer"}
					})
					ctx1 := withFunctionCall(mCtx, callTC.callID)
					final := &fakeTool{name: "final_answer"}
					args := map[string]any{"answer": "same"}

					var errs callbackErrors
					errs.add(p.BeforeAgentCallback()(mCtx))
					errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{}))
					errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "call tool")}, nil))
					errs.add(p.BeforeToolCallback()(ctx1, final, args))
					errs.add(p.OnToolErrorCallback()(ctx1, final, args, errors.New("boom")))
					// Next turn runs LLM step before skipped-Before retry executes AfterTool on ctx1 with same args.
					if !lateMode {
						errs.add(p.BeforeModelCallback()(mCtx, &model.LLMRequest{}))
						errs.add(p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "retry")}, nil))
						errs.add(p.AfterToolCallback()(ctx1, final, args, map[string]any{"answer": "good"}, nil))
					}
					errs.add(p.AfterAgentCallback()(mCtx))
					errs.check(t)
					p.AfterRunCallback()(mCtx)

					if lateMode {
						if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{}); err != nil {
							t.Fatalf("late BeforeModel: %v", err)
						}
						if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", "late retry")}, nil); err != nil {
							t.Fatalf("late AfterModel: %v", err)
						}
						if _, err := p.AfterToolCallback()(ctx1, final, args, map[string]any{"answer": "good"}, nil); err != nil {
							t.Fatalf("late AfterTool: %v", err)
						}
					}
					if err := p.Close(); err != nil {
						t.Fatalf("p.Close: %v", err)
					}

					rows := drainAllRows(t, requests)
					onlyRowOfType(t, rows, EventToolError)
					onlyRowOfType(t, rows, EventToolCompleted)
					onlyRowOfType(t, rows, EventAgentResponse)
				})
			}

			t.Run("observed_second_before_control", func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				ctx1 := withFunctionCall(mCtx, callTC.callID)
				tl := &fakeTool{name: "echo"}
				args := map[string]any{"x": 1}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctx1, tl, args))
				errs.add(p.OnToolErrorCallback()(ctx1, tl, args, errors.New("boom")))
				errs.add(p.BeforeToolCallback()(ctx1, tl, args))
				errs.add(p.AfterToolCallback()(ctx1, tl, args, map[string]any{"ok": true}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				onlyRowOfType(t, rows, EventToolCompleted)
			})

			t.Run("single_attempt_same_context_same_args_still_suppressed", func(t *testing.T) {
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				ctx1 := withFunctionCall(mCtx, callTC.callID)
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "same"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctx1, final, args))
				errs.add(p.OnToolErrorCallback()(ctx1, final, args, errors.New("boom")))
				// Direct callback invocation reusing the same pointer context without
				// EventActions and with identical JSON-shaped arguments in the same
				// turn (whether a single recovered attempt or a skipped-after /
				// skipped-before sequence) suppresses TOOL_COMPLETED and AGENT_RESPONSE.
				errs.add(p.AfterToolCallback()(ctx1, final, args, map[string]any{"answer": "recovered"}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
					t.Fatalf("got %d TOOL_COMPLETED rows on single recovered attempt, want 0", n)
				}
				if n := len(rowsOfType(rows, EventAgentResponse)); n != 0 {
					t.Fatalf("got %d AGENT_RESPONSE rows on suppressed final_answer, want 0", n)
				}
			})

			t.Run("opaque_args_reusing_same_pointer_context_without_actions_kept", func(t *testing.T) {
				type opaqueArg struct{ V int }
				p, requests, mCtx := setupTestPlugin(t)
				ctx1 := withFunctionCall(mCtx, callTC.callID)
				tl := &fakeTool{name: "echo"}
				args := map[string]any{"s": opaqueArg{V: 1}}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctx1, tl, args))
				errs.add(p.OnToolErrorCallback()(ctx1, tl, args, errors.New("boom")))
				errs.add(p.AfterToolCallback()(ctx1, tl, args, map[string]any{"ok": true}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				onlyRowOfType(t, rows, EventToolCompleted)
			})
		})
	}

	// Runner-backed proof that per-attempt EventActions suppresses recovered
	// attempts regardless of whether an earlier plugin registers AfterToolCallback
	// only or both BeforeToolCallback and AfterToolCallback.
	for _, hasEarlierBeforeAndAfter := range []bool{true, false} {
		name := "runner_after_only_earlier_plugin_single_attempt_suppressed"
		if hasEarlierBeforeAndAfter {
			name = "runner_earlier_before_and_after_plugin_single_attempt_suppressed"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, _ := newTestPluginWithConfig(t, func(c *Config) {
				c.FinalResponseToolNames = []string{"final_answer"}
			})
			var capturedAgentCtx, capturedToolCtx atomic.Value
			captureAgentCtx := func(ctx agent.Context) (*genai.Content, error) {
				capturedAgentCtx.Store(ctx)
				return nil, nil
			}
			var earlierCfg baseplugin.Config
			if hasEarlierBeforeAndAfter {
				earlierCfg = baseplugin.Config{
					Name:                "earlier_before_after",
					BeforeAgentCallback: captureAgentCtx,
					BeforeToolCallback: func(ctx agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
						capturedToolCtx.Store(ctx)
						return nil, nil
					},
					AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
						return nil, nil
					},
				}
			} else {
				earlierCfg = baseplugin.Config{
					Name:                "earlier_after_only",
					BeforeAgentCallback: captureAgentCtx,
					AfterToolCallback: func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
						capturedToolCtx.Store(ctx)
						return nil, nil
					},
				}
			}
			earlierPlugin, err := baseplugin.New(earlierCfg)
			if err != nil {
				t.Fatalf("baseplugin.New(earlier): %v", err)
			}

			probeTool, err := functiontool.New(functiontool.Config{
				Name:        "final_answer",
				Description: "Probe",
			}, func(_ agent.Context, _ map[string]any) (map[string]any, error) {
				return map[string]any{"answer": "seed"}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			probeAgent, err := llmagent.New(llmagent.Config{
				Name: "probe",
				Model: &scriptedLLM{responses: []*model.LLMResponse{
					{Content: genai.NewContentFromParts([]*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "call-pm", Name: "final_answer", Args: map[string]any{"answer": "same"}},
					}}, genai.RoleModel)},
					{Content: textContent("model", "done")},
				}},
				Tools: []tool.Tool{probeTool},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			for _, runErr := range newTestRunner(t, probeAgent, earlierPlugin, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}

			rawAgentCtx := capturedAgentCtx.Load()
			if rawAgentCtx == nil {
				t.Fatal("failed to capture runner agent context")
			}
			runnerAgentCtx, ok := rawAgentCtx.(agent.Context)
			if !ok {
				t.Fatalf("captured agent context type %T does not implement agent.Context", rawAgentCtx)
			}
			rawCtx := capturedToolCtx.Load()
			if rawCtx == nil {
				t.Fatal("failed to capture runner tool context")
			}
			runnerToolCtx, ok := rawCtx.(agent.Context)
			if !ok {
				t.Fatalf("captured context type %T does not implement agent.Context", rawCtx)
			}
			final := &fakeTool{name: "final_answer"}
			sameArgs := map[string]any{"answer": "same"}
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(runnerAgentCtx))
			errs.add(p.BeforeToolCallback()(runnerToolCtx, final, sameArgs))
			errs.add(p.OnToolErrorCallback()(runnerToolCtx, final, sameArgs, errors.New("boom")))
			errs.add(p.AfterToolCallback()(runnerToolCtx, final, sameArgs, map[string]any{"answer": "good"}, nil))
			errs.add(p.AfterAgentCallback()(runnerAgentCtx))
			errs.check(t)
			p.AfterRunCallback()(runnerAgentCtx)
			if err := p.Close(); err != nil {
				t.Fatalf("p.Close: %v", err)
			}

			rows := drainAllRows(t, requests)
			var probeComps int
			for _, r := range rowsOfType(rows, EventToolCompleted) {
				if res, _ := jsonObjectColumn(t, r, "content")["result"].(map[string]any); res != nil && res["answer"] == "good" {
					probeComps++
				}
			}
			if probeComps != 0 {
				t.Fatalf("got %d recovered TOOL_COMPLETED rows on runner tool context, want 0", probeComps)
			}
		})
	}
}

func TestPlugin_RecoveredToolError_UnrecoveredValueContext(t *testing.T) {
	factories := []struct {
		name  string
		build func(*mockInvocationContext, string) agent.Context
	}{
		{
			name: "comparable_value_ctx",
			build: func(m *mockInvocationContext, callID string) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, callID), tag: "val"}
			},
		},
		{
			name: "non_comparable_slice_ctx",
			build: func(m *mockInvocationContext, callID string) agent.Context {
				return nonComparableSliceToolCtx{mockInvocationContext: withFunctionCall(m, callID), tags: []string{"t"}}
			},
		},
		{
			name: "closure_factory_ctx",
			build: func(m *mockInvocationContext, callID string) agent.Context {
				v := 1
				return nonComparableFuncToolCtx{mockInvocationContext: withFunctionCall(m, callID), fn: func() int { return v }}
			},
		},
	}

	for _, f := range factories {
		for _, callID := range []string{"unique-call", ""} {
			callLabel := callID
			if callLabel == "" {
				callLabel = "empty_call_id"
			}
			for _, mode := range []string{"with_before", "skipped_before", "late_after_run", "mutated_args_in_place"} {
				t.Run(f.name+"/"+callLabel+"/"+mode, func(t *testing.T) {
					p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
						c.FinalResponseToolNames = []string{"final_answer"}
					})
					ctxVal := f.build(mCtx, callID)
					final := &fakeTool{name: "final_answer"}
					args := map[string]any{"answer": "bad"}
					boom := errors.New("still failed")

					var errs callbackErrors
					errs.add(p.BeforeAgentCallback()(mCtx))
					if mode != "skipped_before" {
						errs.add(p.BeforeToolCallback()(ctxVal, final, args))
					}
					errs.add(p.OnToolErrorCallback()(ctxVal, final, args, boom))
					if mode == "mutated_args_in_place" {
						args["answer"] = "mutated by earlier callback"
					}
					if mode != "late_after_run" {
						errs.add(p.AfterToolCallback()(ctxVal, final, args, nil, boom))
					}
					errs.add(p.AfterAgentCallback()(mCtx))
					errs.check(t)
					p.AfterRunCallback()(mCtx)
					if mode == "late_after_run" {
						if _, err := p.AfterToolCallback()(ctxVal, final, args, nil, boom); err != nil {
							t.Fatalf("late AfterToolCallback: %v", err)
						}
					}
					if err := p.Close(); err != nil {
						t.Fatalf("p.Close: %v", err)
					}

					rows := drainAllRows(t, requests)
					onlyRowOfType(t, rows, EventToolError)
					if got := len(rowsOfType(rows, EventToolCompleted)); got != 0 {
						t.Fatalf("got %d TOOL_COMPLETED rows on unrecovered failure, want 0", got)
					}
				})
			}

			t.Run(f.name+"/"+callLabel+"/two_distinct_300_key_maps", func(t *testing.T) {
				make300 := func(offset int) map[string]any {
					m := make(map[string]any, 300)
					for i := range 300 {
						m[fmt.Sprintf("k%03d", i)] = i + offset
					}
					return m
				}
				p, requests, mCtx := setupTestPlugin(t)
				final := &fakeTool{name: "final_answer"}
				boom1 := errors.New("fail-1")
				boom2 := errors.New("fail-2")
				ctx1 := f.build(mCtx, callID)
				ctx2 := f.build(mCtx, callID)
				args1 := make300(0)
				args2 := make300(1000)

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				if callID == "" {
					// Back-to-back OnToolErrorCallbacks on empty-ID value contexts
					// must keep independent markers so neither loses nor duplicates
					// a TOOL_ERROR row.
					errs.add(p.OnToolErrorCallback()(ctx1, final, args1, boom1))
					errs.add(p.OnToolErrorCallback()(ctx2, final, args2, boom2))
					args1["k001"] = "mutated"
					errs.add(p.AfterToolCallback()(ctx1, final, args1, nil, boom1))
					errs.add(p.AfterToolCallback()(ctx2, final, args2, nil, boom2))
				} else {
					// Attempt 1's AfterToolCallback is skipped after OnToolErrorCallback;
					// Attempt 2 runs BeforeToolCallback and its OnToolErrorCallback is
					// skipped before AfterToolCallback sees the error with a distinct
					// 300-key map.
					errs.add(p.BeforeToolCallback()(ctx1, final, args1))
					errs.add(p.OnToolErrorCallback()(ctx1, final, args1, boom1))
					errs.add(p.BeforeToolCallback()(ctx2, final, args2))
					errs.add(p.AfterToolCallback()(ctx2, final, args2, nil, boom2))
				}
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				if got := len(rowsOfType(rows, EventToolError)); got != 2 {
					t.Fatalf("got %d TOOL_ERROR rows for 2 distinct 300-key-map calls, want 2; events=%v", got, eventTypes(rows))
				}
			})

			t.Run(f.name+"/"+callLabel+"/back_to_back_errors_mutated_300_key_maps", func(t *testing.T) {
				make300 := func(offset int) map[string]any {
					m := make(map[string]any, 300)
					for i := range 300 {
						m[fmt.Sprintf("k%03d", i)] = i + offset
					}
					return m
				}
				p, requests, mCtx := setupTestPlugin(t)
				final := &fakeTool{name: "final_answer"}
				boom1 := errors.New("fail-1")
				boom2 := errors.New("fail-2")
				ctx1 := f.build(mCtx, callID)
				ctx2 := f.build(mCtx, callID)
				args1 := make300(0)
				args2 := make300(1000)

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.OnToolErrorCallback()(ctx1, final, args1, boom1))
				errs.add(p.OnToolErrorCallback()(ctx2, final, args2, boom2))
				args1["k001"] = "mutated-1"
				args2["k002"] = "mutated-2"
				errs.add(p.AfterToolCallback()(ctx1, final, args1, nil, boom1))
				errs.add(p.AfterToolCallback()(ctx2, final, args2, nil, boom2))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				if got := len(rowsOfType(rows, EventToolError)); got != 2 {
					t.Fatalf("got %d TOOL_ERROR rows for back-to-back errors, want 2; events=%v", got, eventTypes(rows))
				}
				if got := len(rowsOfType(rows, EventToolCompleted)); got != 0 {
					t.Fatalf("got %d TOOL_COMPLETED rows, want 0", got)
				}
			})

			t.Run(f.name+"/"+callLabel+"/recovered_control", func(t *testing.T) {
				p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				ctxVal := f.build(mCtx, callID)
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "ok"}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctxVal, final, args))
				errs.add(p.OnToolErrorCallback()(ctxVal, final, args, errors.New("recoverable")))
				errs.add(p.AfterToolCallback()(ctxVal, final, args, map[string]any{"answer": "recovered"}, nil))
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				onlyRowOfType(t, rows, EventToolCompleted)
			})
		}
	}
}

// TestPlugin_RecoveredToolError_StaleValueContextMarker drives the tool
// callbacks of two failed attempts of one call directly, on value contexts
// without attempt identity. Attempt 1 runs BeforeToolCallback and
// OnToolErrorCallback, and its AfterToolCallback is skipped or recovers the
// failure. Attempt 2 skips both BeforeToolCallback and OnToolErrorCallback and
// reaches AfterToolCallback with a new error. Attempt 2's failure gets a
// TOOL_ERROR row of its own whenever something tells it apart from attempt 1:
// attempt 1's recovered AfterToolCallback consumed attempt 1's marker, or
// attempt 2 sees another arguments map whose complete digest differs. When
// attempt 1's AfterToolCallback was skipped, equal arguments in a new map, or
// arguments past the digest bounds, do not tell the attempts apart, as
// EventToolError documents, and attempt 2 is taken for attempt 1's callback.
func TestPlugin_RecoveredToolError_StaleValueContextMarker(t *testing.T) {
	make300 := func(offset int) map[string]any {
		m := make(map[string]any, 300)
		for i := range 300 {
			m[fmt.Sprintf("k%03d", i)] = i + offset
		}
		return m
	}
	for _, callID := range []string{"call-val", ""} {
		callLabel := callID
		if callLabel == "" {
			callLabel = "empty_call_id"
		}
		for _, attempt1After := range []string{"skipped", "recovered"} {
			for _, tc := range []struct {
				name         string
				args1, args2 func() map[string]any
				// skippedErrors is the number of TOOL_ERROR rows when attempt
				// 1's AfterToolCallback is skipped.
				skippedErrors int
			}{
				{
					name:          "distinct_args",
					args1:         func() map[string]any { return map[string]any{"q": "first"} },
					args2:         func() map[string]any { return map[string]any{"q": "second"} },
					skippedErrors: 2,
				},
				{
					name:          "equal_args_in_new_map",
					args1:         func() map[string]any { return map[string]any{"q": "same"} },
					args2:         func() map[string]any { return map[string]any{"q": "same"} },
					skippedErrors: 1,
				},
				{
					name:          "distinct_args_past_digest_bounds",
					args1:         func() map[string]any { return make300(0) },
					args2:         func() map[string]any { return make300(1000) },
					skippedErrors: 1,
				},
			} {
				t.Run(callLabel+"/"+attempt1After+"_after/"+tc.name, func(t *testing.T) {
					p, requests, mCtx := setupTestPlugin(t)
					ctx1 := comparableValueToolCtx{mockInvocationContext: withFunctionCall(mCtx, callID), tag: "val"}
					ctx2 := comparableValueToolCtx{mockInvocationContext: withFunctionCall(mCtx, callID), tag: "val"}
					lookup := &fakeTool{name: "lookup"}
					args1, args2 := tc.args1(), tc.args2()

					var errs callbackErrors
					errs.add(p.BeforeAgentCallback()(mCtx))
					errs.add(p.BeforeToolCallback()(ctx1, lookup, args1))
					errs.add(p.OnToolErrorCallback()(ctx1, lookup, args1, errors.New("attempt 1 failed")))
					if attempt1After == "recovered" {
						errs.add(p.AfterToolCallback()(ctx1, lookup, args1, map[string]any{"recovered": true}, nil))
					}
					errs.add(p.AfterToolCallback()(ctx2, lookup, args2, nil, errors.New("attempt 2 failed")))
					errs.add(p.AfterAgentCallback()(mCtx))
					errs.check(t)
					p.AfterRunCallback()(mCtx)

					rows := collectRows(t, requests, untilInvocationEnd)
					wantErrors, wantCompleted := tc.skippedErrors, 0
					if attempt1After == "recovered" {
						// A recovered value-context attempt keeps its
						// TOOL_COMPLETED row (see EventToolCompleted).
						wantErrors, wantCompleted = 2, 1
					}
					if got := len(rowsOfType(rows, EventToolError)); got != wantErrors {
						t.Errorf("got %d TOOL_ERROR rows, want %d; events=%v", got, wantErrors, eventTypes(rows))
					}
					if got := len(rowsOfType(rows, EventToolCompleted)); got != wantCompleted {
						t.Errorf("got %d TOOL_COMPLETED rows, want %d; events=%v", got, wantCompleted, eventTypes(rows))
					}
				})
			}
		}
	}
}

func TestPlugin_RecoveredToolError_AmbiguousOverlapNoBorrowedSpan(t *testing.T) {
	ctxFactories := []struct {
		name  string
		build func(*mockInvocationContext) agent.Context
	}{
		{
			name: "pointer_context",
			build: func(m *mockInvocationContext) agent.Context {
				return withFunctionCall(m, "")
			},
		},
		{
			name: "comparable_value_context",
			build: func(m *mockInvocationContext) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, ""), tag: "shared-ambig"}
			},
		},
	}

	for _, cf := range ctxFactories {
		t.Run(cf.name+"/recovered_first", func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			ctxShared := cf.build(mCtx)
			tl := &fakeTool{name: "echo"}
			args := map[string]any{"x": "same"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.OnToolErrorCallback()(ctxShared, tl, args, errors.New("boom")))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"status": "recovered"}, nil))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"status": "healthy"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			requireAmbiguousOverlapSpans(t, rowsOfType(rows, EventToolStarting), rowsOfType(rows, EventToolCompleted), onlyRowOfType(t, rows, EventToolError))
		})

		t.Run(cf.name+"/healthy_first", func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			ctxShared := cf.build(mCtx)
			tl := &fakeTool{name: "echo"}
			args := map[string]any{"x": "same"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.OnToolErrorCallback()(ctxShared, tl, args, errors.New("boom")))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"status": "healthy"}, nil))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"status": "recovered"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			requireAmbiguousOverlapSpans(t, rowsOfType(rows, EventToolStarting), rowsOfType(rows, EventToolCompleted), onlyRowOfType(t, rows, EventToolError))
		})

		t.Run(cf.name+"/error_free_overlap_no_leaks_and_no_borrowed_latency", func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			ctxShared := cf.build(mCtx)
			tl := &fakeTool{name: "echo"}
			args := map[string]any{"x": "same"}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"n": 1}, nil))
			errs.add(p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"n": 2}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			if got := len(rowsOfType(rows, EventToolError)); got != 0 {
				t.Fatalf("got %d TOOL_ERROR rows, want 0 (unclosed tool span leak check)", got)
			}
			starts := rowsOfType(rows, EventToolStarting)
			comps := rowsOfType(rows, EventToolCompleted)
			if len(starts) != 2 || len(comps) != 2 {
				t.Fatalf("got %d starts and %d comps, want 2 and 2", len(starts), len(comps))
			}
			s0, s1 := placeOf(starts[0]).span, placeOf(starts[1]).span
			c0, c1 := placeOf(comps[0]).span, placeOf(comps[1]).span
			for i, row := range comps {
				if row["latency_ms"] != nil {
					t.Errorf("error-free ambiguous comps[%d] latency_ms = %v, want nil", i, row["latency_ms"])
				}
				cs := placeOf(row).span
				if cs == s0 || cs == s1 {
					t.Errorf("error-free ambiguous comps[%d] span = %s, must not borrow start (%s, %s)", i, cs, s0, s1)
				}
			}
			if c0 == c1 {
				t.Errorf("error-free ambiguous TOOL_COMPLETED rows reused same span %s", c0)
			}
		})
	}
}

// TestPlugin_RecoveredToolError_LoggingPluginOrder pins recovered-error
// suppression across plugin registration orders: analytics alone,
// loggingplugin (which registers both BeforeToolCallback and AfterToolCallback)
// before or after this plugin, and two split earlier plugins registering
// BeforeToolCallback and AfterToolCallback separately. Because stock ADK tool
// dispatch creates a fresh tool.Context and EventActions per attempt, all four
// orders suppress TOOL_COMPLETED and any tool-promoted AGENT_RESPONSE.
func TestPlugin_RecoveredToolError_LoggingPluginOrder(t *testing.T) {
	for _, kind := range []struct {
		name              string
		finalResponseTool bool
	}{
		{name: "regular_tool", finalResponseTool: false},
		{name: "final_response_tool", finalResponseTool: true},
	} {
		for _, mode := range []string{
			"analytics_only",
			"logging_plugin_first",
			"logging_plugin_after",
			"split_before_and_after_plugins_first",
		} {
			t.Run(kind.name+"/"+mode, func(t *testing.T) {
				const wantCompleted, wantAgentResp = 0, 1
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					if kind.finalResponseTool {
						c.FinalResponseToolNames = []string{"final_answer"}
					}
				})
				toolName := "flaky"
				if kind.finalResponseTool {
					toolName = "final_answer"
				}
				flaky, err := functiontool.New(functiontool.Config{
					Name:        toolName,
					Description: "always fails",
				}, func(_ agent.Context, _ map[string]any) (map[string]any, error) {
					return nil, errors.New("flaky failed")
				})
				if err != nil {
					t.Fatalf("functiontool.New: %v", err)
				}
				a, err := llmagent.New(llmagent.Config{
					Name: "probe",
					Model: &scriptedLLM{responses: []*model.LLMResponse{
						{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
							FunctionCall: &genai.FunctionCall{
								ID:   "call-1",
								Name: toolName,
								Args: map[string]any{"x": 1},
							},
						}}}},
						{Content: textContent("model", "done")},
					}},
					Tools: []tool.Tool{flaky},
					OnToolErrorCallbacks: []llmagent.OnToolErrorCallback{
						func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
							return map[string]any{"recovered": true}, nil
						},
					},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				plugins := []*baseplugin.Plugin{p}
				switch mode {
				case "logging_plugin_first":
					plugins = []*baseplugin.Plugin{loggingplugin.MustNew("log"), p}
				case "logging_plugin_after":
					plugins = []*baseplugin.Plugin{p, loggingplugin.MustNew("log")}
				case "split_before_and_after_plugins_first":
					beforeOnly, err := baseplugin.New(baseplugin.Config{
						Name: "before_only",
						BeforeToolCallback: func(_ agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New(before_only): %v", err)
					}
					afterOnly, err := baseplugin.New(baseplugin.Config{
						Name: "after_only",
						AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New(after_only): %v", err)
					}
					plugins = []*baseplugin.Plugin{beforeOnly, afterOnly, p}
				}
				for _, runErr := range newTestRunner(t, a, plugins...).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error: %v", runErr)
					}
				}
				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				if te := len(rowsOfType(rows, EventToolError)); te != 1 {
					t.Errorf("got TOOL_ERROR=%d, want 1; events=%v", te, eventTypes(rows))
				}
				if tc := len(rowsOfType(rows, EventToolCompleted)); tc != wantCompleted {
					t.Errorf("got TOOL_COMPLETED=%d, want %d; events=%v", tc, wantCompleted, eventTypes(rows))
				}
				if ar := len(rowsOfType(rows, EventAgentResponse)); ar != wantAgentResp {
					t.Errorf("got AGENT_RESPONSE=%d, want %d; events=%v", ar, wantAgentResp, eventTypes(rows))
				}
			})
		}
	}
}

// argumentMutatingAfterToolPlugin returns a plugin whose AfterToolCallback
// changes the tool call's arguments in place and passes the call on.
func argumentMutatingAfterToolPlugin(t *testing.T) *baseplugin.Plugin {
	t.Helper()
	pl, err := baseplugin.New(baseplugin.Config{
		Name: "mutate_args",
		AfterToolCallback: func(_ agent.Context, _ tool.Tool, args, _ map[string]any, _ error) (map[string]any, error) {
			args["x"] = "changed by an earlier AfterToolCallback"
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	return pl
}

// TestPlugin_RecoveredToolError_ArgumentsMutatedInPlace runs a failing tool
// through the runner while a callback that runs between this plugin's
// OnToolErrorCallback and AfterToolCallback changes the call's arguments in
// place: an earlier plugin's AfterToolCallback, or the agent's
// OnToolErrorCallback that counts retries in the arguments. The
// AfterToolCallback still belongs to the attempt whose failure
// OnToolErrorCallback recorded, so an unrecovered failure ends with one
// TOOL_ERROR row and a recovered one with no TOOL_COMPLETED row.
func TestPlugin_RecoveredToolError_ArgumentsMutatedInPlace(t *testing.T) {
	countRetries := func(recover bool) llmagent.OnToolErrorCallback {
		return func(_ agent.Context, _ tool.Tool, args map[string]any, _ error) (map[string]any, error) {
			n, _ := args["retries"].(int)
			args["retries"] = n + 1
			if recover {
				return map[string]any{"answer": "recovered"}, nil
			}
			return nil, nil
		}
	}
	recoverOnly := func(_ agent.Context, _ tool.Tool, _ map[string]any, _ error) (map[string]any, error) {
		return map[string]any{"answer": "recovered"}, nil
	}
	for _, tc := range []struct {
		name          string
		earlierPlugin bool
		agentCallback llmagent.OnToolErrorCallback
	}{
		{name: "earlier_after_tool_plugin/unrecovered", earlierPlugin: true},
		{name: "earlier_after_tool_plugin/recovered", earlierPlugin: true, agentCallback: recoverOnly},
		{name: "agent_on_tool_error/unrecovered", agentCallback: countRetries(false)},
		{name: "agent_on_tool_error/recovered", agentCallback: countRetries(true)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := runFailingToolWithPlugins(t, "lookup", nil, tc.agentCallback, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
				if tc.earlierPlugin {
					return []*baseplugin.Plugin{argumentMutatingAfterToolPlugin(t), p}
				}
				return []*baseplugin.Plugin{p}
			})
			if n := len(rowsOfType(rows, EventToolError)); n != 1 {
				t.Fatalf("got %d TOOL_ERROR rows, want 1; events=%v", n, eventTypes(rows))
			}
			if errRow := onlyRowOfType(t, rows, EventToolError); errRow["error_message"] != "boom" {
				t.Errorf("TOOL_ERROR error_message = %v, want boom", errRow["error_message"])
			}
			if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
				t.Errorf("got %d TOOL_COMPLETED rows, want 0; events=%v", n, eventTypes(rows))
			}
		})
	}
}

// sharedNestedArgs returns tool arguments whose map with int keys holds
// values that several of its entries share.
func sharedNestedArgs() map[string]any {
	sharedSlice := []any{"a", map[string]any{"b": 1}}
	sharedMap := map[string]any{"c": []int{1, 2}}
	nested := map[int]any{}
	for i := range 16 {
		if i%2 == 0 {
			nested[i] = sharedSlice
		} else {
			nested[i] = sharedMap
		}
	}
	return map[string]any{"q": "1", "nested": nested}
}

// TestPlugin_RecoveredToolError_SharedNestedArgsUnrecovered fails 16 tool
// calls whose unchanged arguments share values inside a map with int keys.
// Each failure ends with exactly one TOOL_ERROR row: through the runner, and
// on a custom context that is not a pointer, which only the digest of the
// arguments ties to its attempt, so the digest must not depend on the order
// in which the map is iterated.
func TestPlugin_RecoveredToolError_SharedNestedArgsUnrecovered(t *testing.T) {
	const calls = 16
	t.Run("runner", func(t *testing.T) {
		p, requests, _ := setupTestPlugin(t)
		failing, err := functiontool.New(functiontool.Config{
			Name:        "lookup",
			Description: "Always fails",
		}, func(_ agent.Context, _ map[string]any) (map[string]any, error) {
			return nil, errors.New("boom")
		})
		if err != nil {
			t.Fatalf("functiontool.New: %v", err)
		}
		var responses []*model.LLMResponse
		for i := range calls {
			responses = append(responses, &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%d", i), Name: "lookup", Args: sharedNestedArgs()},
			}}}})
		}
		responses = append(responses, &model.LLMResponse{Content: textContent("model", "done")})
		bot, err := llmagent.New(llmagent.Config{
			Name:  "bot",
			Model: &scriptedLLM{responses: responses},
			Tools: []tool.Tool{failing},
		})
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		for _, runErr := range newTestRunner(t, bot, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
			if runErr != nil {
				t.Fatalf("runner.Run error: %v", runErr)
			}
		}
		rows := collectRows(t, requests, untilInvocationEnd)
		if n := len(rowsOfType(rows, EventToolError)); n != calls {
			t.Errorf("got %d TOOL_ERROR rows, want %d", n, calls)
		}
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
		}
	})
	for _, tc := range []struct {
		name   string
		callID func(i int) string
	}{
		{name: "value_context/unique_call_ids", callID: func(i int) string { return fmt.Sprintf("call-%d", i) }},
		{name: "value_context/empty_call_ids", callID: func(int) string { return "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			tl := &fakeTool{name: "lookup"}
			boom := errors.New("boom")
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			for i := range calls {
				ctx := comparableValueToolCtx{mockInvocationContext: withFunctionCall(mCtx, tc.callID(i)), tag: "shared"}
				args := sharedNestedArgs()
				errs.add(p.BeforeToolCallback()(ctx, tl, args))
				errs.add(p.OnToolErrorCallback()(ctx, tl, args, boom))
				errs.add(p.AfterToolCallback()(ctx, tl, args, nil, boom))
			}
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolError)); n != calls {
				t.Errorf("got %d TOOL_ERROR rows, want %d", n, calls)
			}
			if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
				t.Errorf("got %d TOOL_COMPLETED rows, want 0", n)
			}
		})
	}
}

// TestPlugin_RecoveredToolError_HiddenBatchLeaderDelayedRejectedFinish runs a
// task-mode agent twice on one branch and node path under a parent that
// buffers the first run's function response event. The first run's model
// calls [echo, finish_task] at once: an earlier plugin answers both of echo's
// tool callbacks, so this plugin never sees the StateDelta map of echo's tool
// context, which the merged response event carries, and finish_task fails and
// is recovered. The parent delivers the buffered response, as it is or rebuilt
// without a StateDelta, after the second run's echo response or after its
// model called finish_task with the same call ID. That response must not
// retire the second run, whose own finish_task call later completes it.
func TestPlugin_RecoveredToolError_HiddenBatchLeaderDelayedRejectedFinish(t *testing.T) {
	const callID = "call-finish-shared"
	hasFinishTaskPart := func(c *genai.Content, response bool) bool {
		if c == nil {
			return false
		}
		for _, part := range c.Parts {
			if part == nil {
				continue
			}
			if response && part.FunctionResponse != nil && part.FunctionResponse.Name == finishTaskToolName {
				return true
			}
			if !response && part.FunctionCall != nil && part.FunctionCall.Name == finishTaskToolName {
				return true
			}
		}
		return false
	}
	for _, mode := range []string{"retry_and_reflect_payload", "recovered_by_error_callback"} {
		for _, delivery := range []string{"original_event", "rebuilt_without_state_delta"} {
			for _, flushAt := range []string{"after_peer_echo", "after_peer_finish_call"} {
				t.Run(mode+"/"+delivery+"/"+flushAt, func(t *testing.T) {
					p, requests, _ := setupTestPlugin(t)
					var runCount atomic.Int32
					earlier, err := baseplugin.New(baseplugin.Config{
						Name: "hide_echo_and_fail_finish",
						BeforeToolCallback: func(_ agent.Context, tl tool.Tool, _ map[string]any) (map[string]any, error) {
							if runCount.Load() == 1 && tl != nil && tl.Name() == "echo" {
								return map[string]any{"echo": "cached"}, nil
							}
							return nil, nil
						},
						AfterToolCallback: func(ctx agent.Context, tl tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
							if runCount.Load() != 1 || tl == nil || err != nil {
								return nil, nil
							}
							switch tl.Name() {
							case "echo":
								return result, nil
							case finishTaskToolName:
								if mode == "recovered_by_error_callback" {
									if _, cbErr := p.OnToolErrorCallback()(ctx, tl, args, errors.New("finish rejected")); cbErr != nil {
										return nil, cbErr
									}
								} else {
									result["response_type"] = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
									result["error_details"] = "finish rejected"
								}
							}
							return nil, nil
						},
					})
					if err != nil {
						t.Fatalf("baseplugin.New: %v", err)
					}
					echoTool, err := functiontool.New(functiontool.Config{
						Name:        "echo",
						Description: "Echo input",
					}, func(_ agent.Context, args map[string]any) (map[string]any, error) {
						return map[string]any{"echo": args["msg"]}, nil
					})
					if err != nil {
						t.Fatalf("functiontool.New(echo): %v", err)
					}
					finishCall := &genai.Part{FunctionCall: &genai.FunctionCall{ID: callID, Name: finishTaskToolName, Args: map[string]any{"result": "done"}}}
					taskAgent, err := llmagent.New(llmagent.Config{
						Name: "task",
						Mode: llmagent.ModeTask,
						Model: &scriptedLLM{responses: []*model.LLMResponse{
							{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
								{FunctionCall: &genai.FunctionCall{ID: "call-echo-a", Name: "echo", Args: map[string]any{"msg": "a"}}},
								finishCall,
							}}},
							{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
								{FunctionCall: &genai.FunctionCall{ID: "call-echo-b", Name: "echo", Args: map[string]any{"msg": "b"}}},
							}}},
							{Content: &genai.Content{Role: "model", Parts: []*genai.Part{finishCall}}},
						}},
						Tools: []tool.Tool{echoTool},
					})
					if err != nil {
						t.Fatalf("llmagent.New(task): %v", err)
					}
					flushed := false
					root, err := agent.New(agent.Config{
						Name: "root",
						Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(yield func(*session.Event, error) bool) {
								nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
								var buffered *session.Event
								for range 2 {
									n := runCount.Add(1)
									for ev, iterErr := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
										if iterErr != nil {
											yield(nil, iterErr)
											return
										}
										if n == 1 && ev != nil && hasFinishTaskPart(ev.Content, true) {
											buffered = ev
											if delivery == "rebuilt_without_state_delta" {
												rebuilt := *ev
												rebuilt.Actions = session.EventActions{}
												buffered = &rebuilt
											}
											continue
										}
										if !yield(ev, nil) {
											return
										}
										if n != 2 || flushed || buffered == nil || ev == nil || ev.Partial || ev.Author != "task" || hasFinishTaskPart(ev.Content, true) {
											continue
										}
										if (flushAt == "after_peer_finish_call") == hasFinishTaskPart(ev.Content, false) {
											flushed = true
											if !yield(buffered, nil) {
												return
											}
										}
									}
								}
							}
						},
					})
					if err != nil {
						t.Fatalf("agent.New(root): %v", err)
					}
					for _, runErr := range newTestRunner(t, root, earlier, p).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
						if runErr != nil {
							t.Fatalf("runner.Run error: %v", runErr)
						}
					}
					if !flushed {
						t.Fatal("the buffered finish_task response was never delivered")
					}

					rows := collectRows(t, requests, untilInvocationEnd)
					requireInvocationCompletedOK(t, rows)
					starts := agentRowsOfType(rows, EventAgentStarting, "task")
					if len(starts) != 2 {
						t.Fatalf("got %d task AGENT_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
					}
					spanA, spanB := starts[0]["span_id"], starts[1]["span_id"]
					finishCompIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
						return r["event_type"] == EventToolCompleted && r["parent_span_id"] == spanB &&
							jsonObjectColumn(t, r, "content")["tool"] == finishTaskToolName
					})
					if finishCompIdx < 0 {
						t.Fatalf("no finish_task TOOL_COMPLETED row under the second run's span %v; events=%v", spanB, eventTypes(rows))
					}
					var endsB, endsA []int
					for i, r := range rows {
						if r["event_type"] != EventAgentCompleted || r["agent"] != "task" {
							continue
						}
						switch r["span_id"] {
						case spanB:
							endsB = append(endsB, i)
						case spanA:
							endsA = append(endsA, i)
						}
					}
					if len(endsB) != 1 || endsB[0] < finishCompIdx {
						t.Errorf("second run AGENT_COMPLETED rows at %v, want one after its finish_task TOOL_COMPLETED at %d; events=%v",
							endsB, finishCompIdx, eventTypes(rows))
					}
					// Both retry-and-reflect payloads and OnToolErrorCallback recoveries
					// reject the first run's finish_task call even when an earlier plugin
					// registers both BeforeToolCallback and AfterToolCallback.
					if len(endsA) != 0 {
						t.Errorf("got %d first run AGENT_COMPLETED rows, want 0; events=%v", len(endsA), eventTypes(rows))
					}
				})
			}
		}
	}
}

// emptyCallIDToolCtx hides the function call ID of the tool context that it
// wraps, as a custom executor that dispatches calls without IDs does.
type emptyCallIDToolCtx struct{ agent.Context }

func (*emptyCallIDToolCtx) FunctionCallID() string { return "" }

// TestPlugin_RecoveredToolError_CustomExecutorReusedContextCachedRetry pins the
// per-attempt tool.Context / EventActions contract for custom tool executors:
// allocating a fresh agent.NewToolContext per attempt keeps a cached retry's
// healthy TOOL_COMPLETED and AGENT_RESPONSE rows even when an earlier plugin
// hides attempt 1's AfterToolCallback and serves attempt 2 from BeforeToolCallback,
// whereas reusing attempt 1's tool.Context / EventActions in the same turn is
// unsupported and suppresses the second AfterToolCallback as attempt 1's recovery.
func TestPlugin_RecoveredToolError_CustomExecutorReusedContextCachedRetry(t *testing.T) {
	for _, emptyID := range []bool{false, true} {
		idLabel := "non-empty_call_id"
		if emptyID {
			idLabel = "empty_call_id"
		}
		for _, reuseCtx := range []bool{false, true} {
			ctxLabel := "fresh_tool_context_per_attempt"
			wantCompleted := 1
			if reuseCtx {
				ctxLabel = "reused_tool_context_unsupported"
				wantCompleted = 0
			}
			t.Run(idLabel+"/"+ctxLabel, func(t *testing.T) {
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					c.FinalResponseToolNames = []string{"final_answer"}
				})
				var attempt atomic.Int32
				cache, err := baseplugin.New(baseplugin.Config{
					Name: "cache",
					BeforeToolCallback: func(_ agent.Context, _ tool.Tool, _ map[string]any) (map[string]any, error) {
						if attempt.Load() == 2 {
							return map[string]any{"answer": "cached"}, nil
						}
						return nil, nil
					},
					AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, result map[string]any, _ error) (map[string]any, error) {
						if attempt.Load() == 1 {
							return result, nil
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatalf("baseplugin.New: %v", err)
				}
				plugins := []*baseplugin.Plugin{cache, p}
				final := &fakeTool{name: "final_answer"}
				args := map[string]any{"answer": "same"}
				dispatch := func(toolCtx agent.Context, run func() (map[string]any, error), recover func(error) map[string]any) error {
					var result map[string]any
					var runErr error
					for _, pl := range plugins {
						if cb := pl.BeforeToolCallback(); cb != nil {
							if result, runErr = cb(toolCtx, final, args); result != nil || runErr != nil {
								break
							}
						}
					}
					if result == nil && runErr == nil {
						result, runErr = run()
					}
					if runErr != nil {
						var handled map[string]any
						var cbErr error
						for _, pl := range plugins {
							if cb := pl.OnToolErrorCallback(); cb != nil {
								if handled, cbErr = cb(toolCtx, final, args, runErr); handled != nil || cbErr != nil {
									break
								}
							}
						}
						if handled == nil && cbErr == nil {
							handled = recover(runErr)
						}
						result, runErr = handled, cbErr
					}
					for _, pl := range plugins {
						if cb := pl.AfterToolCallback(); cb != nil {
							if altered, cbErr := cb(toolCtx, final, args, result, runErr); altered != nil || cbErr != nil {
								return cbErr
							}
						}
					}
					return runErr
				}
				makeAttemptCtx := func(ctx agent.InvocationContext) agent.Context {
					tc := agent.NewToolContext(ctx, "call-reused", &session.EventActions{StateDelta: map[string]any{}}, nil)
					if emptyID {
						return &emptyCallIDToolCtx{Context: tc}
					}
					return tc
				}
				executor, err := agent.New(agent.Config{
					Name: "executor",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							toolCtx1 := makeAttemptCtx(ctx)
							toolCtx2 := toolCtx1
							if !reuseCtx {
								toolCtx2 = makeAttemptCtx(ctx)
							}
							attempt.Store(1)
							if err := dispatch(toolCtx1, func() (map[string]any, error) {
								return nil, errors.New("boom")
							}, func(error) map[string]any {
								return map[string]any{"answer": "recovered"}
							}); err != nil {
								yield(nil, err)
								return
							}
							attempt.Store(2)
							if err := dispatch(toolCtx2, func() (map[string]any, error) {
								return nil, errors.New("attempt 2 must be served from the cache")
							}, nil); err != nil {
								yield(nil, err)
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = ctx.Agent().Name()
							ev.LLMResponse = model.LLMResponse{Content: textContent("model", "cached answer")}
							yield(ev, nil)
						}
					},
				})
				if err != nil {
					t.Fatalf("agent.New: %v", err)
				}
				for _, runErr := range newTestRunner(t, executor, plugins...).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
					if runErr != nil {
						t.Fatalf("runner.Run error: %v", runErr)
					}
				}

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				onlyRowOfType(t, rows, EventToolError)
				comps := rowsOfType(rows, EventToolCompleted)
				if len(comps) != wantCompleted {
					t.Fatalf("got %d TOOL_COMPLETED rows, want %d; events=%v", len(comps), wantCompleted, eventTypes(rows))
				}
				if wantCompleted == 1 {
					checkJSON(t, "attempt 2 TOOL_COMPLETED", jsonObjectColumn(t, comps[0], "content"),
						`{"tool":"final_answer","result":{"answer":"cached"},"tool_origin":"UNKNOWN"}`)
				}
				var toolResps int
				for _, r := range rowsOfType(rows, EventAgentResponse) {
					if jsonObjectColumn(t, r, "attributes")["source_tool"] == "final_answer" {
						toolResps++
					}
				}
				if toolResps != wantCompleted {
					t.Errorf("got %d final_answer AGENT_RESPONSE rows, want %d; events=%v", toolResps, wantCompleted, eventTypes(rows))
				}
			})
		}
	}
}

// TestPlugin_RecoveredToolError_ChangedArgumentsOnReusedContext reuses one
// tool context for a retry whose callbacks a custom executor drives without
// the runner's plugins: attempt 1 fails and its AfterToolCallback is skipped,
// and attempt 2 skips BeforeToolCallback. Only the arguments tell the retry
// from a recovery of attempt 1, so they are compared in full: a long string
// that differs only in the middle, and a map with more than 32 keys changed
// in place, keep attempt 2's TOOL_COMPLETED row, while unchanged arguments
// suppress it.
func TestPlugin_RecoveredToolError_ChangedArgumentsOnReusedContext(t *testing.T) {
	longString := func(middle string) string {
		return strings.Repeat("a", 200) + middle + strings.Repeat("z", 200)
	}
	bigMap := func() map[string]any {
		m := make(map[string]any, 40)
		for i := range 40 {
			m[fmt.Sprintf("k%02d", i)] = i
		}
		return m
	}
	for _, tc := range []struct {
		name          string
		args          func() (attempt1, attempt2 map[string]any)
		wantCompleted int
	}{
		{
			name: "long_string_differs_in_the_middle",
			args: func() (map[string]any, map[string]any) {
				return map[string]any{"s": longString("first")}, map[string]any{"s": longString("other")}
			},
			wantCompleted: 1,
		},
		{
			name: "large_map_changed_in_place",
			args: func() (map[string]any, map[string]any) {
				m := bigMap()
				return m, m
			},
			wantCompleted: 1,
		},
		{
			name: "unchanged_arguments",
			args: func() (map[string]any, map[string]any) {
				m := bigMap()
				return m, m
			},
			wantCompleted: 0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			ctx := withFunctionCall(mCtx, "call-reused")
			tl := &fakeTool{name: "echo"}
			args1, args2 := tc.args()

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(ctx, tl, args1))
			errs.add(p.OnToolErrorCallback()(ctx, tl, args1, errors.New("boom")))
			if tc.name == "large_map_changed_in_place" {
				args2["k17"] = "changed"
			}
			errs.add(p.AfterToolCallback()(ctx, tl, args2, map[string]any{"ok": true}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			requireInvocationCompletedOK(t, rows)
			onlyRowOfType(t, rows, EventToolError)
			if n := len(rowsOfType(rows, EventToolCompleted)); n != tc.wantCompleted {
				t.Errorf("got %d TOOL_COMPLETED rows, want %d; events=%v", n, tc.wantCompleted, eventTypes(rows))
			}
		})
	}
}

// TestPlugin_RecoveredToolError_ConcurrentPeerTurn ends a failed tool call of
// one activation of an agent after a concurrent activation of the same agent
// on the same branch and node path took a model turn between the call's
// OnToolErrorCallback and AfterToolCallback. Turns are counted per activation,
// so the peer's turn does not make the AfterToolCallback a later turn: an
// unrecovered failure ends with one TOOL_ERROR row and a recovered one with no
// TOOL_COMPLETED row. A model turn of the call's own activation still does.
func TestPlugin_RecoveredToolError_ConcurrentPeerTurn(t *testing.T) {
	for _, tc := range []struct {
		name          string
		recovered     bool
		ownTurn       bool
		wantErrors    int
		wantCompleted int
	}{
		{name: "peer_turn/unrecovered", wantErrors: 1},
		{name: "peer_turn/recovered", recovered: true, wantErrors: 1},
		{name: "own_turn/recovered", recovered: true, ownTurn: true, wantErrors: 1, wantCompleted: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			bot, err := llmagent.New(llmagent.Config{Name: "test-agent", Model: &scriptedLLM{}})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			icA := &runningInvocation{running: bot}
			icB := &runningInvocation{running: bot}
			actA := &taskActivationContext{mockInvocationContext: mCtx, ic: icA}
			actB := &taskActivationContext{mockInvocationContext: mCtx, ic: icB}
			toolA := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-a"), ic: icA}
			tl := &fakeTool{name: "lookup"}
			args := map[string]any{"q": "1"}
			boom := errors.New("boom")
			turn := func(act agent.Context, text string) {
				t.Helper()
				var errs callbackErrors
				errs.add(p.BeforeModelCallback()(act, &model.LLMRequest{}))
				errs.add(p.AfterModelCallback()(act, &model.LLMResponse{Content: textContent("model", text)}, nil))
				errs.check(t)
			}

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(actA))
			errs.add(p.BeforeAgentCallback()(actB))
			turn(actA, "call lookup")
			errs.add(p.BeforeToolCallback()(toolA, tl, args))
			errs.add(p.OnToolErrorCallback()(toolA, tl, args, boom))
			if tc.ownTurn {
				turn(actA, "retry")
			} else {
				turn(actB, "peer turn")
			}
			if tc.recovered {
				errs.add(p.AfterToolCallback()(toolA, tl, args, map[string]any{"answer": "recovered"}, nil))
			} else {
				errs.add(p.AfterToolCallback()(toolA, tl, args, nil, boom))
			}
			errs.add(p.AfterAgentCallback()(actB))
			errs.add(p.AfterAgentCallback()(actA))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			if n := len(rowsOfType(rows, EventToolError)); n != tc.wantErrors {
				t.Errorf("got %d TOOL_ERROR rows, want %d; events=%v", n, tc.wantErrors, eventTypes(rows))
			}
			if n := len(rowsOfType(rows, EventToolCompleted)); n != tc.wantCompleted {
				t.Errorf("got %d TOOL_COMPLETED rows, want %d; events=%v", n, tc.wantCompleted, eventTypes(rows))
			}
		})
	}
}

// taskToolDeltas returns how many StateDelta references the open activations
// of the task-mode agent name keep in the invocation invocationID.
func taskToolDeltas(tm *traceManager, invocationID, name string) int {
	tm.mu.Lock()
	inv := tm.invocations[invocationID]
	tm.mu.Unlock()
	if inv == nil {
		return -1
	}
	inv.mu.Lock()
	defer inv.mu.Unlock()
	n := 0
	for _, rec := range inv.agentStack[1:] {
		if rec.kind == spanKindAgent && rec.taskMode && rec.name == name {
			n += len(rec.toolStateDeltas)
		}
	}
	return n
}

// TestPlugin_RecoveredToolError_TaskToolDeltasReleased runs 32 ordinary tool
// calls of a task-mode agent, one per model response or all in one. Once the
// function response events of the calls are delivered, the activation keeps
// none of their StateDelta maps; when a parent swallows the events, it keeps
// at most maxTaskToolDeltas of them.
func TestPlugin_RecoveredToolError_TaskToolDeltasReleased(t *testing.T) {
	const calls = 32
	for _, tc := range []struct {
		name      string
		batched   bool
		swallow   bool
		wantAfter func(n int) bool
		want      string
	}{
		{name: "one_call_per_response", wantAfter: func(n int) bool { return n == 0 }, want: "0"},
		{name: "all_calls_in_one_response", batched: true, wantAfter: func(n int) bool { return n == 0 }, want: "0"},
		{name: "responses_swallowed", swallow: true, wantAfter: func(n int) bool { return n > 0 && n <= maxTaskToolDeltas }, want: "1..maxTaskToolDeltas"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tm := newTraceManager()
			pl := newTraceOnlyPlugin(t, tm)
			echoTool, err := functiontool.New(functiontool.Config{
				Name:        "echo",
				Description: "Echo input",
			}, func(ctx agent.Context, args map[string]any) (map[string]any, error) {
				ctx.Actions().StateDelta["echoed"] = args["msg"]
				return map[string]any{"echo": args["msg"]}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New(echo): %v", err)
			}
			n := calls
			if tc.swallow {
				n = 2 * maxTaskToolDeltas
			}
			echoCall := func(i int) *genai.Part {
				return &genai.Part{FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%d", i), Name: "echo", Args: map[string]any{"msg": strconv.Itoa(i)}}}
			}
			var responses []*model.LLMResponse
			if tc.batched {
				var parts []*genai.Part
				for i := range n {
					parts = append(parts, echoCall(i))
				}
				responses = append(responses, &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: parts}})
			} else {
				for i := range n {
					responses = append(responses, &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{echoCall(i)}}})
				}
			}
			responses = append(responses, &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: "call-finish", Name: finishTaskToolName, Args: map[string]any{"result": "done"}},
			}}}})
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "task",
				Mode:  llmagent.ModeTask,
				Model: &scriptedLLM{responses: responses},
				Tools: []tool.Tool{echoTool},
			})
			if err != nil {
				t.Fatalf("llmagent.New(task): %v", err)
			}
			// echoResponses counts the echo function responses of ev.
			echoResponses := func(ev *session.Event) int {
				if ev == nil || ev.Content == nil {
					return 0
				}
				n := 0
				for _, part := range ev.Content.Parts {
					if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == "echo" {
						n++
					}
				}
				return n
			}
			seen, got := 0, -1
			root, err := agent.New(agent.Config{
				Name: "root",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						nodeCtx := agent.PromoteWithDelta(ctx, &agent.CommonContextDelta{})
						for ev, iterErr := range llmagent.RunLLMAgentAsNode(taskAgent, nodeCtx, "work") {
							if iterErr != nil {
								yield(nil, iterErr)
								return
							}
							count := echoResponses(ev)
							if count == 0 || !tc.swallow {
								if !yield(ev, nil) {
									return
								}
							}
							// Inspect once every echo call ended and, unless
							// swallowed, its response event was delivered, before
							// the finish_task call ends the activation.
							if seen += count; count > 0 && seen >= n && got < 0 {
								got = taskToolDeltas(tm, ctx.InvocationID(), "task")
							}
						}
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New(root): %v", err)
			}
			for _, runErr := range newTestRunner(t, root, pl).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
				if runErr != nil {
					t.Fatalf("runner.Run error: %v", runErr)
				}
			}
			if !tc.wantAfter(got) {
				t.Errorf("task activation keeps %d StateDelta references after %d echo calls, want %s", got, n, tc.want)
			}
		})
	}
}

// TestPlugin_RecoveredToolError_AmbiguousOverlapUnrecovered overlaps two calls
// with an empty function call ID on callback contexts that cannot be told
// apart, one of which fails without recovery, in either completion order. The
// AfterToolCallback that still sees the error settles the failure that
// OnToolErrorCallback recorded on the overlap and adds no second TOOL_ERROR
// row, so the overlap ends with one TOOL_ERROR row and one TOOL_COMPLETED
// row, neither borrowing a span.
func TestPlugin_RecoveredToolError_AmbiguousOverlapUnrecovered(t *testing.T) {
	for _, cf := range []struct {
		name  string
		build func(*mockInvocationContext) agent.Context
	}{
		{
			name: "pointer_context",
			build: func(m *mockInvocationContext) agent.Context {
				return withFunctionCall(m, "")
			},
		},
		{
			name: "comparable_value_context",
			build: func(m *mockInvocationContext) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, ""), tag: "shared-unrecovered"}
			},
		},
	} {
		for _, failedFirst := range []bool{true, false} {
			order := "healthy_first"
			if failedFirst {
				order = "failed_first"
			}
			t.Run(cf.name+"/"+order, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				ctxShared := cf.build(mCtx)
				tl := &fakeTool{name: "echo"}
				args := map[string]any{"x": "same"}
				boom := errors.New("boom")
				failed := func() (map[string]any, error) {
					return p.AfterToolCallback()(ctxShared, tl, args, nil, boom)
				}
				healthy := func() (map[string]any, error) {
					return p.AfterToolCallback()(ctxShared, tl, args, map[string]any{"status": "healthy"}, nil)
				}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(mCtx))
				errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
				errs.add(p.BeforeToolCallback()(ctxShared, tl, args))
				errs.add(p.OnToolErrorCallback()(ctxShared, tl, args, boom))
				if failedFirst {
					errs.add(failed())
					errs.add(healthy())
				} else {
					errs.add(healthy())
					errs.add(failed())
				}
				errs.add(p.AfterAgentCallback()(mCtx))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireAmbiguousOverlapSpans(t, rowsOfType(rows, EventToolStarting), rowsOfType(rows, EventToolCompleted), onlyRowOfType(t, rows, EventToolError))
				checkJSON(t, "TOOL_COMPLETED", jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolCompleted), "content"),
					`{"tool":"echo","result":{"status":"healthy"},"tool_origin":"UNKNOWN"}`)
			})
		}
	}
}

// TestPlugin_RecoveredToolError_UnfinishedFinishTaskClearsPendingCall runs two
// activations of a task-mode agent that call finish_task with the same call
// ID. The newer activation's call fails validation and ends without completing
// its task, so its model's function call no longer counts as evidence that a
// later finish_task response answers it: the older activation's response,
// whose tool callbacks an earlier plugin skipped, completes the older
// activation rather than the newer one.
func TestPlugin_RecoveredToolError_UnfinishedFinishTaskClearsPendingCall(t *testing.T) {
	const callID = "call-finish-shared"
	finishCall := genai.NewContentFromParts([]*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: callID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
	}}, genai.RoleModel)
	p, requests, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	icOld := &runningInvocation{running: taskAgent}
	icNew := &runningInvocation{running: taskAgent}
	actOld := &taskActivationContext{mockInvocationContext: mCtx, ic: icOld}
	actNew := &taskActivationContext{mockInvocationContext: mCtx, ic: icNew}
	toolNew := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icNew}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
	finish := &fakeTool{name: finishTaskToolName}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(actOld))
	errs.add(p.BeforeAgentCallback()(actNew))
	errs.add(p.BeforeModelCallback()(actNew, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actNew, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.BeforeToolCallback()(toolNew, finish, map[string]any{"summary": "done"}))
	errs.add(p.AfterToolCallback()(toolNew, finish, map[string]any{"summary": "done"}, map[string]any{"error": "summary is invalid"}, nil))
	// The older activation calls finish_task, and an earlier plugin answers
	// both of its tool callbacks.
	errs.add(p.BeforeModelCallback()(actOld, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actOld, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
		Author: "test-agent",
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{ID: callID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}},
		}}, genai.RoleUser)},
	}))
	errs.add(p.AfterAgentCallback()(actNew))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
	if len(starts) != 2 {
		t.Fatalf("got %d AGENT_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
	}
	endsOld, endsNew := agentEndIndexes(rows, "test-agent", starts[0]["span_id"]), agentEndIndexes(rows, "test-agent", starts[1]["span_id"])
	// The response ends the older activation when it arrives, before the
	// newer activation's AfterAgentCallback ends that one.
	if len(endsOld) != 1 || len(endsNew) != 1 || endsOld[0] > endsNew[0] {
		t.Errorf("AGENT_COMPLETED rows of the older activation at %v and of the newer one at %v, want one each with the older first; events=%v",
			endsOld, endsNew, eventTypes(rows))
	}
}

// agentEndIndexes returns the indexes in rows of the AGENT_COMPLETED rows of
// the activation of agentName whose span is spanID.
func agentEndIndexes(rows []map[string]any, agentName string, spanID any) []int {
	var ends []int
	for i, r := range rows {
		if r["event_type"] == EventAgentCompleted && r["agent"] == agentName && r["span_id"] == spanID {
			ends = append(ends, i)
		}
	}
	return ends
}

// TestPlugin_RecoveredToolError_DelayedRejectionConsumedBeforeHiddenPeerFinish
// runs two activations of a task-mode agent that call finish_task with the
// same call ID. The older activation's call fails and is answered with a
// retry-and-reflect payload, and its function response arrives late: after
// the newer activation's model made its own call, but before that call ran.
// The late response consumes the rejection and ends nothing. The newer
// activation's call then completes while an earlier plugin answers both of
// its tool callbacks, so the model's function call is the only evidence that
// the newer activation made it: its response must still end the newer
// activation when it arrives.
func TestPlugin_RecoveredToolError_DelayedRejectionConsumedBeforeHiddenPeerFinish(t *testing.T) {
	const callID = "call-finish-shared"
	finishCall := genai.NewContentFromParts([]*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: callID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
	}}, genai.RoleModel)
	finishResponse := func() *session.Event {
		return &session.Event{
			Author: "test-agent",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: callID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}},
			}}, genai.RoleUser)},
		}
	}
	p, requests, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	icOld := &runningInvocation{running: taskAgent}
	icNew := &runningInvocation{running: taskAgent}
	actOld := &taskActivationContext{mockInvocationContext: mCtx, ic: icOld}
	actNew := &taskActivationContext{mockInvocationContext: mCtx, ic: icNew}
	toolOld := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icOld}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
	finish := &fakeTool{name: finishTaskToolName}
	args := map[string]any{"summary": "done"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(actOld))
	errs.add(p.BeforeAgentCallback()(actNew))
	errs.add(p.BeforeModelCallback()(actOld, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actOld, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.BeforeToolCallback()(toolOld, finish, args))
	errs.add(p.AfterToolCallback()(toolOld, finish, args, map[string]any{
		"result":        "Task completed.",
		"response_type": reflectAndRetryResponseType,
		"error_details": "finish rejected",
	}, nil))
	errs.add(p.BeforeModelCallback()(actNew, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actNew, &model.LLMResponse{Content: finishCall}, nil))
	// The older activation's response arrives before the newer activation's
	// call runs.
	errs.add(p.OnEventCallback()(runnerCtx, finishResponse()))
	// An earlier plugin answers both tool callbacks of the newer activation's
	// call, which completes its task.
	errs.add(p.OnEventCallback()(runnerCtx, finishResponse()))
	errs.add(p.AfterAgentCallback()(actOld))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
	if len(starts) != 2 {
		t.Fatalf("got %d AGENT_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
	}
	endsOld, endsNew := agentEndIndexes(rows, "test-agent", starts[0]["span_id"]), agentEndIndexes(rows, "test-agent", starts[1]["span_id"])
	if len(endsOld) != 1 || len(endsNew) != 1 || endsNew[0] > endsOld[0] {
		t.Errorf("AGENT_COMPLETED rows of the newer activation at %v and of the older one at %v, want one each with the newer first; events=%v",
			endsNew, endsOld, eventTypes(rows))
	}
}

// TestPlugin_RecoveredToolError_DelayedRejectionThenPeerFinishFails is
// TestPlugin_RecoveredToolError_DelayedRejectionConsumedBeforeHiddenPeerFinish
// with a newer activation whose finish_task call fails validation, so that
// its response reports the failure instead of completion. That response
// consumes the newer activation's finish_task call and cancels its deferred
// completion, so the newer activation gets no AGENT_COMPLETED row: when an
// earlier plugin answers both of its tool callbacks and the older
// activation's delayed response arrives first, and when an earlier plugin
// answers only its AfterToolCallback and the older activation's delayed
// response arrives after the failure, also while a third activation holds a
// pending finish_task call with the same ID.
func TestPlugin_RecoveredToolError_DelayedRejectionThenPeerFinishFails(t *testing.T) {
	const callID = "call-finish-shared"
	finishCall := genai.NewContentFromParts([]*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: callID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
	}}, genai.RoleModel)
	finishResponse := func(resp map[string]any) *session.Event {
		return &session.Event{
			Author: "test-agent",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: callID, Name: finishTaskToolName, Response: resp},
			}}, genai.RoleUser)},
		}
	}
	completed := map[string]any{"result": "Task completed."}
	failed := map[string]any{"error": "summary is invalid"}
	for _, mode := range []string{"hidden_tool_callbacks", "hidden_after_tool_callback", "hidden_after_tool_callback_with_pending_peer"} {
		t.Run(mode, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			taskAgent, err := llmagent.New(llmagent.Config{
				Name:  "test-agent",
				Mode:  llmagent.ModeTask,
				Model: &scriptedLLM{},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			icOld := &runningInvocation{running: taskAgent}
			icNew := &runningInvocation{running: taskAgent}
			actOld := &taskActivationContext{mockInvocationContext: mCtx, ic: icOld}
			actNew := &taskActivationContext{mockInvocationContext: mCtx, ic: icNew}
			actPeer := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{running: taskAgent}}
			toolOld := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icOld}
			toolNew := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icNew}
			runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
			finish := &fakeTool{name: finishTaskToolName}
			args := map[string]any{"summary": "done"}
			withPeer := mode == "hidden_after_tool_callback_with_pending_peer"

			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(actOld))
			errs.add(p.BeforeAgentCallback()(actNew))
			if withPeer {
				errs.add(p.BeforeAgentCallback()(actPeer))
			}
			errs.add(p.BeforeModelCallback()(actOld, &model.LLMRequest{}))
			errs.add(p.AfterModelCallback()(actOld, &model.LLMResponse{Content: finishCall}, nil))
			errs.add(p.BeforeToolCallback()(toolOld, finish, args))
			errs.add(p.AfterToolCallback()(toolOld, finish, args, map[string]any{
				"result":        "Task completed.",
				"response_type": reflectAndRetryResponseType,
				"error_details": "finish rejected",
			}, nil))
			errs.add(p.BeforeModelCallback()(actNew, &model.LLMRequest{}))
			errs.add(p.AfterModelCallback()(actNew, &model.LLMResponse{Content: finishCall}, nil))
			if withPeer {
				// A third activation's model makes a finish_task call with the
				// same call ID, which has not run yet.
				errs.add(p.BeforeModelCallback()(actPeer, &model.LLMRequest{}))
				errs.add(p.AfterModelCallback()(actPeer, &model.LLMResponse{Content: finishCall}, nil))
			}
			if mode == "hidden_tool_callbacks" {
				// The older activation's response arrives before the newer
				// activation's call runs, and an earlier plugin answers both
				// tool callbacks of the newer activation's call.
				errs.add(p.OnEventCallback()(runnerCtx, finishResponse(completed)))
				errs.add(p.OnEventCallback()(runnerCtx, finishResponse(failed)))
			} else {
				// The newer activation's call runs, and an earlier plugin
				// answers its AfterToolCallback; the older activation's
				// response arrives after the newer one's. The failure answers
				// the newer activation's open call, which shows whose it is
				// even while another activation has a pending call with the
				// same ID.
				errs.add(p.BeforeToolCallback()(toolNew, finish, args))
				errs.add(p.OnEventCallback()(runnerCtx, finishResponse(failed)))
				errs.add(p.OnEventCallback()(runnerCtx, finishResponse(completed)))
			}
			errs.add(p.AfterAgentCallback()(actOld))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
			if wantStarts := map[bool]int{false: 2, true: 3}[withPeer]; len(starts) != wantStarts {
				t.Fatalf("got %d AGENT_STARTING rows, want %d; events=%v", len(starts), wantStarts, eventTypes(rows))
			}
			endsOld, endsNew := agentEndIndexes(rows, "test-agent", starts[0]["span_id"]), agentEndIndexes(rows, "test-agent", starts[1]["span_id"])
			if len(endsNew) != 0 {
				t.Errorf("newer activation whose finish_task call failed got AGENT_COMPLETED rows at %v, want none; events=%v", endsNew, eventTypes(rows))
			}
			if len(endsOld) != 1 {
				t.Errorf("older activation got %d AGENT_COMPLETED rows, want 1 (its own AfterAgentCallback); events=%v", len(endsOld), eventTypes(rows))
			}
		})
	}
}

// TestPlugin_RecoveredToolError_VisibleFailureKeepsHiddenPeerDeferral runs
// three activations of a task-mode agent whose finish_task calls share one
// call ID. The oldest activation's call is rejected, and its delayed response
// defers the completion of a newer activation whose tool callbacks an earlier
// plugin answers. A third activation's call then fails validation in plain
// sight, so its response, which reports the failure, may be its own rather
// than the hidden peer's: the response leaves the peer's deferred completion
// alone, and the peer still gets its AGENT_COMPLETED row when the run ends.
func TestPlugin_RecoveredToolError_VisibleFailureKeepsHiddenPeerDeferral(t *testing.T) {
	const callID = "call-finish-shared"
	finishCall := genai.NewContentFromParts([]*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: callID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
	}}, genai.RoleModel)
	finishResponse := func(resp map[string]any) *session.Event {
		return &session.Event{
			Author: "test-agent",
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: callID, Name: finishTaskToolName, Response: resp},
			}}, genai.RoleUser)},
		}
	}
	p, requests, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	icOld := &runningInvocation{running: taskAgent}
	icVisible := &runningInvocation{running: taskAgent}
	actOld := &taskActivationContext{mockInvocationContext: mCtx, ic: icOld}
	actHidden := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{running: taskAgent}}
	actVisible := &taskActivationContext{mockInvocationContext: mCtx, ic: icVisible}
	toolOld := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icOld}
	toolVisible := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, callID), ic: icVisible}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
	finish := &fakeTool{name: finishTaskToolName}
	args := map[string]any{"summary": "done"}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(actOld))
	errs.add(p.BeforeAgentCallback()(actHidden))
	errs.add(p.BeforeAgentCallback()(actVisible))
	errs.add(p.BeforeModelCallback()(actOld, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actOld, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.BeforeToolCallback()(toolOld, finish, args))
	errs.add(p.AfterToolCallback()(toolOld, finish, args, map[string]any{
		"result":        "Task completed.",
		"response_type": reflectAndRetryResponseType,
		"error_details": "finish rejected",
	}, nil))
	errs.add(p.BeforeModelCallback()(actHidden, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actHidden, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.BeforeModelCallback()(actVisible, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(actVisible, &model.LLMResponse{Content: finishCall}, nil))
	errs.add(p.BeforeToolCallback()(toolVisible, finish, args))
	errs.add(p.AfterToolCallback()(toolVisible, finish, args, map[string]any{"error": "summary is invalid"}, nil))
	// The oldest activation's delayed response defers the hidden peer's
	// completion; the visible activation's failure arrives next.
	errs.add(p.OnEventCallback()(runnerCtx, finishResponse(map[string]any{"result": "Task completed."})))
	errs.add(p.OnEventCallback()(runnerCtx, finishResponse(map[string]any{"error": "summary is invalid"})))
	errs.add(p.AfterAgentCallback()(actVisible))
	errs.add(p.AfterAgentCallback()(actOld))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
	if len(starts) != 3 {
		t.Fatalf("got %d AGENT_STARTING rows, want 3; events=%v", len(starts), eventTypes(rows))
	}
	for i, name := range []string{"oldest", "hidden peer", "visible"} {
		if ends := agentEndIndexes(rows, "test-agent", starts[i]["span_id"]); len(ends) != 1 {
			t.Errorf("%s activation got AGENT_COMPLETED rows at %v, want one; events=%v", name, ends, eventTypes(rows))
		}
	}
}

// TestPlugin_RecoveredToolError_RetryAndReflectAfterThisPlugin fails a tool
// call that plugin/retryandreflect, registered after this plugin, answers with
// a payload that reports the failure. This plugin's OnToolErrorCallback already
// wrote the TOOL_ERROR row, so the AfterToolCallback that sees the payload adds
// no second one: with loggingplugin registered first or after this plugin, on a
// custom context with arguments past the digest bounds, on a custom context
// that is not a pointer, and in an overlap of calls with an empty function call
// ID.
func TestPlugin_RecoveredToolError_RetryAndReflectAfterThisPlugin(t *testing.T) {
	retryPayload := func() map[string]any {
		return map[string]any{"response_type": reflectAndRetryResponseType, "error_details": "boom"}
	}
	requireCounts := func(t *testing.T, rows []map[string]any, wantErrors, wantCompleted int) {
		t.Helper()
		if n, m := len(rowsOfType(rows, EventToolError)), len(rowsOfType(rows, EventToolCompleted)); n != wantErrors || m != wantCompleted {
			t.Errorf("got TOOL_ERROR=%d TOOL_COMPLETED=%d, want %d and %d; events=%v", n, m, wantErrors, wantCompleted, eventTypes(rows))
		}
	}
	for _, order := range []string{"logging_plugin_first", "this_plugin_first"} {
		t.Run("runner/"+order, func(t *testing.T) {
			retry, err := retryandreflect.New()
			if err != nil {
				t.Fatalf("retryandreflect.New: %v", err)
			}
			rows := runFailingToolWithPlugins(t, "lookup", nil, nil, func(p *baseplugin.Plugin) []*baseplugin.Plugin {
				if order == "logging_plugin_first" {
					return []*baseplugin.Plugin{loggingplugin.MustNew("log"), p, retry}
				}
				return []*baseplugin.Plugin{p, retry}
			})
			requireCounts(t, rows, 1, 0)
		})
	}
	for _, tc := range []struct {
		name string
		ctx  func(*mockInvocationContext) agent.Context
		args map[string]any
	}{
		{
			name: "args_past_digest_bounds",
			ctx:  func(m *mockInvocationContext) agent.Context { return withFunctionCall(m, "call-big") },
			args: map[string]any{"big": strings.Repeat("x", maxDigestStringBytes+10)},
		},
		{
			name: "value_context",
			ctx: func(m *mockInvocationContext) agent.Context {
				return comparableValueToolCtx{mockInvocationContext: withFunctionCall(m, "call-val"), tag: "v1"}
			},
			args: map[string]any{"x": "1"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			ctx := tc.ctx(mCtx)
			lookup := &fakeTool{name: "lookup"}
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			errs.add(p.BeforeToolCallback()(ctx, lookup, tc.args))
			errs.add(p.OnToolErrorCallback()(ctx, lookup, tc.args, errors.New("boom")))
			errs.add(p.AfterToolCallback()(ctx, lookup, tc.args, retryPayload(), nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)
			requireCounts(t, collectRows(t, requests, untilInvocationEnd), 1, 0)
		})
	}
	t.Run("empty_id_overlap_payload_first", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		shared := withFunctionCall(mCtx, "")
		echo := &fakeTool{name: "echo"}
		args := map[string]any{"x": "same"}
		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(shared, echo, args))
		errs.add(p.BeforeToolCallback()(shared, echo, args))
		errs.add(p.OnToolErrorCallback()(shared, echo, args, errors.New("boom")))
		errs.add(p.AfterToolCallback()(shared, echo, args, retryPayload(), nil))
		errs.add(p.AfterToolCallback()(shared, echo, args, map[string]any{"status": "healthy"}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)
		requireCounts(t, collectRows(t, requests, untilInvocationEnd), 1, 1)
	})
}

// TestPlugin_RecoveredToolError_AmbiguousOverlapBothFail overlaps two calls
// with an empty function call ID on one callback context that both fail: the
// first is recovered and ends before the second fails without recovery. Each
// failure writes one TOOL_ERROR row, and no span of the overlap stays open
// after the run. With distinct arguments the recovered call is told apart and
// gets no TOOL_COMPLETED row; with equal arguments its TOOL_COMPLETED row is
// written before the second failure shows that both calls failed.
func TestPlugin_RecoveredToolError_AmbiguousOverlapBothFail(t *testing.T) {
	for _, tc := range []struct {
		name          string
		first, second map[string]any
		wantCompleted int
	}{
		{name: "equal_args", first: map[string]any{"x": "same"}, second: map[string]any{"x": "same"}, wantCompleted: 1},
		{name: "distinct_args", first: map[string]any{"x": "first"}, second: map[string]any{"x": "second"}, wantCompleted: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			shared := withFunctionCall(mCtx, "")
			echo := &fakeTool{name: "echo"}
			run := func(pl *baseplugin.Plugin) {
				var errs callbackErrors
				errs.add(pl.BeforeAgentCallback()(mCtx))
				errs.add(pl.BeforeToolCallback()(shared, echo, tc.first))
				errs.add(pl.BeforeToolCallback()(shared, echo, tc.second))
				errs.add(pl.OnToolErrorCallback()(shared, echo, tc.first, errors.New("boom1")))
				errs.add(pl.AfterToolCallback()(shared, echo, tc.first, map[string]any{"status": "recovered"}, nil))
				errs.add(pl.OnToolErrorCallback()(shared, echo, tc.second, errors.New("boom2")))
				errs.add(pl.AfterToolCallback()(shared, echo, tc.second, nil, errors.New("boom2")))
				errs.add(pl.AfterAgentCallback()(mCtx))
				errs.check(t)
				pl.AfterRunCallback()(mCtx)
			}
			run(p)
			rows := collectRows(t, requests, untilInvocationEnd)
			if n, m := len(rowsOfType(rows, EventToolError)), len(rowsOfType(rows, EventToolCompleted)); n != 2 || m != tc.wantCompleted {
				t.Errorf("got TOOL_ERROR=%d TOOL_COMPLETED=%d, want 2 and %d; events=%v", n, m, tc.wantCompleted, eventTypes(rows))
			}

			tm := newTraceManager()
			run(newTraceOnlyPlugin(t, tm))
			tm.mu.Lock()
			completed := tm.completed[mCtx.InvocationID()]
			waits := completed != nil && completed.waitsForLateCalls()
			open := completed != nil && completed.hasPendingOrOpenLateSpans()
			tm.mu.Unlock()
			if waits || open {
				t.Errorf("the run left a tool span of the overlap open: waitsForLateCalls=%v hasPendingOrOpenLateSpans=%v", waits, open)
			}
		})
	}
}

// TestPlugin_RecoveredToolError_AmbiguousOverlapDistinctArgs overlaps two
// calls with an empty function call ID and distinct arguments on one callback
// context, started in either order. One call fails and is recovered, while the
// other fails without its OnToolErrorCallback reaching this plugin, as when an
// earlier plugin answers it: the arguments tell the calls apart, so the call
// whose failure this plugin did not see writes its own TOOL_ERROR row instead
// of settling the other call's failure, and the recovered call writes no
// TOOL_COMPLETED row.
func TestPlugin_RecoveredToolError_AmbiguousOverlapDistinctArgs(t *testing.T) {
	for _, hiddenFirst := range []bool{true, false} {
		name := "recovered_call_starts_first"
		if hiddenFirst {
			name = "hidden_call_starts_first"
		}
		t.Run(name, func(t *testing.T) {
			p, requests, mCtx := setupTestPlugin(t)
			shared := withFunctionCall(mCtx, "")
			echo := &fakeTool{name: "echo"}
			hidden := map[string]any{"x": "hidden"}
			recovered := map[string]any{"x": "recovered"}
			starts := []map[string]any{recovered, hidden}
			if hiddenFirst {
				starts = []map[string]any{hidden, recovered}
			}
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			for _, args := range starts {
				errs.add(p.BeforeToolCallback()(shared, echo, args))
			}
			errs.add(p.OnToolErrorCallback()(shared, echo, recovered, errors.New("boom")))
			errs.add(p.AfterToolCallback()(shared, echo, hidden, nil, errors.New("hidden boom")))
			errs.add(p.AfterToolCallback()(shared, echo, recovered, map[string]any{"status": "recovered"}, nil))
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilInvocationEnd)
			if n, m := len(rowsOfType(rows, EventToolError)), len(rowsOfType(rows, EventToolCompleted)); n != 2 || m != 0 {
				t.Fatalf("got TOOL_ERROR=%d TOOL_COMPLETED=%d, want 2 and 0; events=%v", n, m, eventTypes(rows))
			}
			var messages []any
			for _, r := range rowsOfType(rows, EventToolError) {
				messages = append(messages, r["error_message"])
			}
			if !slices.Contains(messages, any("hidden boom")) || !slices.Contains(messages, any("boom")) {
				t.Errorf("TOOL_ERROR error_message values = %v, want boom and hidden boom", messages)
			}
		})
	}
}

// taskToolContext is the context of a tool call of a task-mode activation
// (see taskActivationContext) with EventActions of its own, as
// agent.NewToolContext gives each tool call of an activation.
type taskToolContext struct {
	*taskActivationContext
	actions *session.EventActions
}

func (c *taskToolContext) Actions() *session.EventActions { return c.actions }

// newTaskToolContext returns the context of a tool call with an empty
// function call ID of the activation ic, whose StateDelta is delta.
func newTaskToolContext(m *mockInvocationContext, ic *runningInvocation, delta map[string]any) *taskToolContext {
	return &taskToolContext{
		taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(m, ""), ic: ic},
		actions:               &session.EventActions{StateDelta: delta},
	}
}

// TestPlugin_RecoveredToolError_TaskToolDeltasKeepLaterCalls delivers the
// function response event of a task-mode activation's first echo call, with
// an empty call ID, only after the activation's next batch [echo, finish_task]
// ran and its finish_task call was rejected, and after a peer activation's
// finish_task call completed. The late event releases the StateDelta of its
// own call only, so the batch's merged event, which carries the second echo
// call's map, is still tied to the activation that made it: it ends neither
// activation, and the peer ends with its own event.
func TestPlugin_RecoveredToolError_TaskToolDeltasKeepLaterCalls(t *testing.T) {
	p, requests, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	echo := &fakeTool{name: "echo"}
	finish := &fakeTool{name: finishTaskToolName}
	icA := &runningInvocation{running: taskAgent}
	icB := &runningInvocation{running: taskAgent}
	actA := &taskActivationContext{mockInvocationContext: mCtx, ic: icA}
	actB := &taskActivationContext{mockInvocationContext: mCtx, ic: icB}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
	toolA1 := newTaskToolContext(mCtx, icA, map[string]any{"b": 1})
	toolA2Echo := newTaskToolContext(mCtx, icA, map[string]any{"b": 2})
	toolA2Finish := newTaskToolContext(mCtx, icA, map[string]any{"f": 2})
	toolBFinish := newTaskToolContext(mCtx, icB, map[string]any{"b_fin": 1})
	responseEvent := func(actions *session.EventActions, parts ...*genai.Part) *session.Event {
		return &session.Event{
			Author:      "test-agent",
			Actions:     *actions,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts(parts, genai.RoleUser)},
		}
	}
	echoResponse := func(v string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "echo", Response: map[string]any{"echo": v}}}
	}
	finishResponse := &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}}}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(actA))
	errs.add(p.BeforeAgentCallback()(actB))
	errs.add(p.BeforeToolCallback()(toolA1, echo, map[string]any{"m": "1"}))
	errs.add(p.AfterToolCallback()(toolA1, echo, map[string]any{"m": "1"}, map[string]any{"echo": "1"}, nil))
	errs.add(p.BeforeToolCallback()(toolA2Echo, echo, map[string]any{"m": "2"}))
	errs.add(p.AfterToolCallback()(toolA2Echo, echo, map[string]any{"m": "2"}, map[string]any{"echo": "2"}, nil))
	errs.add(p.BeforeToolCallback()(toolA2Finish, finish, map[string]any{"result": "done"}))
	errs.add(p.AfterToolCallback()(toolA2Finish, finish, map[string]any{"result": "done"}, map[string]any{
		"result":        "Task completed.",
		"response_type": reflectAndRetryResponseType,
		"error_details": "rejected",
	}, nil))
	// The first echo call's response arrives late.
	errs.add(p.OnEventCallback()(runnerCtx, responseEvent(toolA1.actions, echoResponse("1"))))
	errs.add(p.BeforeToolCallback()(toolBFinish, finish, map[string]any{"result": "B done"}))
	errs.add(p.AfterToolCallback()(toolBFinish, finish, map[string]any{"result": "B done"}, map[string]any{"result": "Task completed."}, nil))
	// The rejected batch's merged response carries the second echo call's map.
	errs.add(p.OnEventCallback()(runnerCtx, responseEvent(toolA2Echo.actions, echoResponse("2"), finishResponse)))
	errs.add(p.OnUserMessageCallback()(runnerCtx, textContent("user", "between the two responses")))
	errs.add(p.OnEventCallback()(runnerCtx, responseEvent(toolBFinish.actions, finishResponse)))
	errs.add(p.AfterAgentCallback()(actA))
	errs.check(t)
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilInvocationEnd)
	starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
	if len(starts) != 2 {
		t.Fatalf("got %d AGENT_STARTING rows, want 2; events=%v", len(starts), eventTypes(rows))
	}
	endsA, endsB := agentEndIndexes(rows, "test-agent", starts[0]["span_id"]), agentEndIndexes(rows, "test-agent", starts[1]["span_id"])
	marker := slices.IndexFunc(rows, func(r map[string]any) bool { return r["event_type"] == EventUserMessageReceived })
	if len(endsA) != 1 || len(endsB) != 1 || endsB[0] < marker || endsA[0] < endsB[0] {
		t.Errorf("AGENT_COMPLETED rows of the first activation at %v and of the peer at %v, marker at %d; want one each, the peer's after the marker and before the first's; events=%v",
			endsA, endsB, marker, eventTypes(rows))
	}
}

// TestPlugin_RecoveredToolError_TaskToolDeltasKeepLaterTurns delivers the
// merged function response event of a task-mode activation's first batch,
// whose second call an earlier plugin answered, after the activation's next
// model turn made a call of the same tool. The late event releases the
// StateDelta maps of its own turn only, so the later call keeps its map until
// its own event arrives.
func TestPlugin_RecoveredToolError_TaskToolDeltasKeepLaterTurns(t *testing.T) {
	tm := newTraceManager()
	p := newTraceOnlyPlugin(t, tm)
	_, _, mCtx := setupTestPlugin(t)
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Mode:  llmagent.ModeTask,
		Model: &scriptedLLM{},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	echo := &fakeTool{name: "echo"}
	lookup := &fakeTool{name: "lookup"}
	ic := &runningInvocation{running: taskAgent}
	act := &taskActivationContext{mockInvocationContext: mCtx, ic: ic}
	runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
	firstEcho := newTaskToolContext(mCtx, ic, map[string]any{"echo": 1})
	laterLookup := newTaskToolContext(mCtx, ic, map[string]any{"lookup": 2})
	call := func(name string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{Name: name}}
	}

	var errs callbackErrors
	errs.add(p.BeforeAgentCallback()(act))
	errs.add(p.BeforeModelCallback()(act, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(act, &model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{call("echo"), call("lookup")}, genai.RoleModel)}, nil))
	errs.add(p.BeforeToolCallback()(firstEcho, echo, map[string]any{"m": "1"}))
	errs.add(p.AfterToolCallback()(firstEcho, echo, map[string]any{"m": "1"}, map[string]any{"echo": "1"}, nil))
	// An earlier plugin answers both tool callbacks of the first lookup call.
	errs.add(p.BeforeModelCallback()(act, &model.LLMRequest{}))
	errs.add(p.AfterModelCallback()(act, &model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{call("lookup")}, genai.RoleModel)}, nil))
	errs.add(p.BeforeToolCallback()(laterLookup, lookup, map[string]any{"q": "2"}))
	errs.add(p.AfterToolCallback()(laterLookup, lookup, map[string]any{"q": "2"}, map[string]any{"found": true}, nil))
	if n := taskToolDeltas(tm, mCtx.InvocationID(), "test-agent"); n != 2 {
		t.Fatalf("task activation keeps %d StateDelta references before the first batch's response, want 2", n)
	}
	errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
		Author:  "test-agent",
		Actions: *firstEcho.actions,
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{Name: "echo", Response: map[string]any{"echo": "1"}}},
			{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"found": false}}},
		}, genai.RoleUser)},
	}))
	errs.check(t)
	if n := taskToolDeltas(tm, mCtx.InvocationID(), "test-agent"); n != 1 {
		t.Errorf("task activation keeps %d StateDelta references after the first batch's response, want 1 (the later lookup call's)", n)
	}
}

// TestPlugin_RecoveredToolError_OpaqueStructAndPointerArgsRaceSafe invokes
// production BeforeToolCallback, OnToolErrorCallback, and AfterToolCallback
// (both with all event types filtered out via EventAllowlist and with full
// logging enabled) while a background goroutine concurrently updates a
// privately synchronized atomic.Int64 and a mutex-guarded map inside struct
// and pointer values carried in args.
func TestPlugin_RecoveredToolError_OpaqueStructAndPointerArgsRaceSafe(t *testing.T) {
	type synchronizedHandle struct {
		PublicLabel string
		counter     atomic.Int64
		mu          sync.Mutex
		privateMap  map[string]int
	}
	type handleCarrier struct {
		Name   string
		Handle *synchronizedHandle
	}

	shared := &synchronizedHandle{
		PublicLabel: "stable",
		privateMap:  map[string]int{"init": 0},
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			shared.counter.Add(1)
			shared.mu.Lock()
			shared.privateMap[fmt.Sprintf("slot-%d", i%16)] = i
			if i%32 == 0 {
				delete(shared.privateMap, fmt.Sprintf("slot-%d", (i+7)%16))
			}
			shared.mu.Unlock()
		}
	})
	defer func() {
		close(stop)
		wg.Wait()
	}()

	for _, filteredOut := range []bool{true, false} {
		label := "all_tool_events_filtered_out"
		if !filteredOut {
			label = "logging_enabled"
		}
		t.Run(label, func(t *testing.T) {
			p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
				if filteredOut {
					c.EventAllowlist = []string{EventInvocationCompleted}
				}
			})
			lookup := &fakeTool{name: "lookup"}
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			for i := range 32 {
				callCtx := agent.NewToolContext(mCtx, fmt.Sprintf("call-%d", i), &session.EventActions{StateDelta: map[string]any{}}, nil)
				args := map[string]any{
					"ptr":   shared,
					"value": handleCarrier{Name: "carrier", Handle: shared},
				}
				errs.add(p.BeforeToolCallback()(callCtx, lookup, args))
				if i%2 == 0 {
					errs.add(p.AfterToolCallback()(callCtx, lookup, args, map[string]any{"ok": true}, nil))
				} else {
					errs.add(p.OnToolErrorCallback()(callCtx, lookup, args, errors.New("boom")))
					errs.add(p.AfterToolCallback()(callCtx, lookup, args, map[string]any{"recovered": true}, nil))
				}
			}
			errs.add(p.AfterAgentCallback()(mCtx))
			errs.check(t)
			p.AfterRunCallback()(mCtx)
			if err := p.Close(); err != nil {
				t.Fatalf("p.Close: %v", err)
			}

			rows := drainAllRows(t, requests)
			if filteredOut {
				if got := len(rows); got != 1 || rows[0]["event_type"] != EventInvocationCompleted {
					t.Fatalf("filtered-out run emitted %v, want [INVOCATION_COMPLETED]", eventTypes(rows))
				}
			} else {
				if got := len(rowsOfType(rows, EventToolError)); got != 16 {
					t.Errorf("got %d TOOL_ERROR rows, want 16", got)
				}
				if got := len(rowsOfType(rows, EventToolCompleted)); got != 16 {
					t.Errorf("got %d TOOL_COMPLETED rows (recovered ones suppressed via EventActions), want 16", got)
				}
			}
		})
	}
}

// privateActionsToolCtx is a custom tool context that keeps private
// bookkeeping in a field named actions behind its own mutex, and returns a
// separate, stable EventActions from Actions(), its interface contract.
type privateActionsToolCtx struct {
	*mockInvocationContext
	mu      sync.Mutex
	actions *session.EventActions
	stable  *session.EventActions
}

func (c *privateActionsToolCtx) Actions() *session.EventActions { return c.stable }

// replaceActions replaces the private bookkeeping under the context's mutex.
func (c *privateActionsToolCtx) replaceActions(i int) {
	c.mu.Lock()
	c.actions = &session.EventActions{StateDelta: map[string]any{"private": i}}
	c.mu.Unlock()
}

// TestPlugin_RecoveredToolError_CustomContextPrivateActionsNotRead drives the
// production tool callbacks with custom contexts whose private actions field
// changes under the context's own mutex while Actions() returns a stable
// EventActions. The plugin takes each attempt's EventActions from Actions()
// and never reads the private field: under -race while another goroutine
// replaces the field (with all tool events filtered out and with logging
// enabled), and when the field changes between the callbacks of one attempt,
// whose recovered AfterToolCallback is then still told to belong to the
// failed attempt.
func TestPlugin_RecoveredToolError_CustomContextPrivateActionsNotRead(t *testing.T) {
	newCtx := func(m *mockInvocationContext, callID string) *privateActionsToolCtx {
		c := &privateActionsToolCtx{
			mockInvocationContext: withFunctionCall(m, callID),
			stable:                &session.EventActions{StateDelta: map[string]any{}},
		}
		c.replaceActions(0)
		return c
	}
	for _, filteredOut := range []bool{true, false} {
		label := "concurrent_writer/all_tool_events_filtered_out"
		if !filteredOut {
			label = "concurrent_writer/logging_enabled"
		}
		t.Run(label, func(t *testing.T) {
			p, requests, mCtx := newTestPluginWithConfig(t, func(c *Config) {
				if filteredOut {
					c.EventAllowlist = []string{EventInvocationCompleted}
				}
			})
			lookup := &fakeTool{name: "lookup"}
			ctxs := make([]*privateActionsToolCtx, 32)
			for i := range ctxs {
				ctxs[i] = newCtx(mCtx, fmt.Sprintf("call-%d", i))
			}
			stop := make(chan struct{})
			var wg sync.WaitGroup
			wg.Go(func() {
				for i := 1; ; i++ {
					select {
					case <-stop:
						return
					default:
					}
					ctxs[i%len(ctxs)].replaceActions(i)
				}
			})
			var errs callbackErrors
			errs.add(p.BeforeAgentCallback()(mCtx))
			for i, callCtx := range ctxs {
				args := map[string]any{"q": i}
				errs.add(p.BeforeToolCallback()(callCtx, lookup, args))
				if i%2 == 0 {
					errs.add(p.AfterToolCallback()(callCtx, lookup, args, map[string]any{"ok": true}, nil))
				} else {
					errs.add(p.OnToolErrorCallback()(callCtx, lookup, args, errors.New("boom")))
					errs.add(p.AfterToolCallback()(callCtx, lookup, args, map[string]any{"recovered": true}, nil))
				}
			}
			errs.add(p.AfterAgentCallback()(mCtx))
			close(stop)
			wg.Wait()
			errs.check(t)
			p.AfterRunCallback()(mCtx)
			if err := p.Close(); err != nil {
				t.Fatalf("p.Close: %v", err)
			}

			rows := drainAllRows(t, requests)
			if filteredOut {
				if len(rows) != 1 || rows[0]["event_type"] != EventInvocationCompleted {
					t.Fatalf("filtered-out run emitted %v, want [INVOCATION_COMPLETED]", eventTypes(rows))
				}
				return
			}
			if got := len(rowsOfType(rows, EventToolError)); got != 16 {
				t.Errorf("got %d TOOL_ERROR rows, want 16", got)
			}
			if got := len(rowsOfType(rows, EventToolCompleted)); got != 16 {
				t.Errorf("got %d TOOL_COMPLETED rows (recovered ones suppressed by the EventActions of Actions()), want 16", got)
			}
		})
	}

	t.Run("private_actions_change_between_callbacks", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		callCtx := newCtx(mCtx, "call-swap")
		lookup := &fakeTool{name: "lookup"}
		args := map[string]any{"q": "1"}

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(mCtx))
		errs.add(p.BeforeToolCallback()(callCtx, lookup, args))
		callCtx.replaceActions(1)
		errs.add(p.OnToolErrorCallback()(callCtx, lookup, args, errors.New("boom")))
		callCtx.replaceActions(2)
		errs.add(p.AfterToolCallback()(callCtx, lookup, args, map[string]any{"recovered": true}, nil))
		errs.add(p.AfterAgentCallback()(mCtx))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		onlyRowOfType(t, rows, EventToolError)
		if n := len(rowsOfType(rows, EventToolCompleted)); n != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0: the attempt's EventActions come from Actions(), not from the private field; events=%v", n, eventTypes(rows))
		}
	})
}

// TestPlugin_RecoveredToolError_RejectedFinishThreeHoles pins the three
// rejected-finish_task supersede edge cases:
//  1. A peer activation's open or settled non-finish_task tool call reusing the
//     same call ID does not count as finish evidence and keeps its open span.
//  2. In the StateDelta-owner branch, a merely pending finish_task FunctionCall
//     in the next model turn does not override a recorded rejection when the
//     buffered merged [echo, finish_task] response arrives. Neither does the
//     open finish_task call of a retry whose BeforeToolCallback ran first: the
//     old batch's response neither ends that call nor counts it as evidence.
//  3. When an unnoted hidden-callback peer delivers its no-delta finish_task
//     response before an older activation's StateDelta-tagged rejected response
//     arrives (or inside an enclosing parent agent whose AfterAgentCallback
//     runs), the peer activation still emits AGENT_COMPLETED.
func TestPlugin_RecoveredToolError_RejectedFinishThreeHoles(t *testing.T) {
	const sharedCallID = "call-shared-0"
	finishTaskFREvent := func(agentName, branch, callID string) *session.Event {
		return &session.Event{
			Author: agentName,
			Branch: branch,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromParts([]*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     finishTaskToolName,
						Response: map[string]any{"result": "Task completed."},
					},
				}}, genai.RoleUser),
			},
		}
	}

	t.Run("reused_call_id_on_non_finish_tool_keeps_peer_and_tool_span_open", func(t *testing.T) {
		for _, settleEchoFirst := range []bool{false, true} {
			mode := "open_echo_span"
			if settleEchoFirst {
				mode = "settled_echo_span"
			}
			t.Run(mode, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				echo := &fakeTool{name: "echo"}
				finish := &fakeTool{name: finishTaskToolName}
				icA := &runningInvocation{running: taskAgent}
				icB := &runningInvocation{running: taskAgent}
				actA := &taskActivationContext{mockInvocationContext: mCtx, ic: icA}
				actB := &taskActivationContext{mockInvocationContext: mCtx, ic: icB}
				toolAFinish := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: icA}
				toolBEcho := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: icB}
				toolBFinish := &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-finish-b"), ic: icB}
				runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(actA))
				errs.add(p.BeforeToolCallback()(toolAFinish, finish, map[string]any{"summary": "a"}))
				errs.add(p.OnToolErrorCallback()(toolAFinish, finish, map[string]any{"summary": "a"}, errors.New("rejected")))
				errs.add(p.AfterToolCallback()(toolAFinish, finish, map[string]any{"summary": "a"}, map[string]any{"result": "Task completed."}, nil))

				errs.add(p.BeforeAgentCallback()(actB))
				errs.add(p.BeforeToolCallback()(toolBEcho, echo, map[string]any{"msg": "hi"}))
				if settleEchoFirst {
					// Earlier plugin skipped echo's AfterToolCallback, and echo's
					// FunctionResponse arrives before A's delayed finish_task response.
					errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
						Author: "test-agent",
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
							FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: "echo", Response: map[string]any{"echo": "hi"}},
						}}, genai.RoleUser)},
					}))
				}

				// Delivering A's delayed no-delta finish_task(sharedCallID) response
				// must neither retire B nor consume B's open echo(sharedCallID) span.
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", sharedCallID)))
				if !settleEchoFirst {
					errs.add(p.AfterToolCallback()(toolBEcho, echo, map[string]any{"msg": "hi"}, map[string]any{"echo": "hi"}, nil))
				}
				errs.add(p.BeforeToolCallback()(toolBFinish, finish, map[string]any{"summary": "b"}))
				errs.add(p.AfterToolCallback()(toolBFinish, finish, map[string]any{"summary": "b"}, map[string]any{"result": "Task completed."}, nil))
				errs.add(p.OnEventCallback()(runnerCtx, finishTaskFREvent("test-agent", "", "call-finish-b")))
				errs.check(t)
				p.AfterRunCallback()(mCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				if len(starts) != 2 || len(ends) != 1 {
					t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED rows, want 2 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
				}
				spanB := starts[1]["span_id"]
				if ends[0]["span_id"] != spanB {
					t.Errorf("AGENT_COMPLETED span_id = %v, want peer B span_id %v", ends[0]["span_id"], spanB)
				}
				if !settleEchoFirst {
					var echoStart, echoComp map[string]any
					for _, r := range rowsOfType(rows, EventToolStarting) {
						if jsonObjectColumn(t, r, "content")["tool"] == "echo" {
							echoStart = r
						}
					}
					for _, r := range rowsOfType(rows, EventToolCompleted) {
						if jsonObjectColumn(t, r, "content")["tool"] == "echo" {
							echoComp = r
						}
					}
					if echoStart == nil || echoComp == nil {
						t.Fatalf("missing echo start (%v) or completion (%v)", echoStart, echoComp)
					}
					if echoComp["span_id"] != echoStart["span_id"] || echoComp["parent_span_id"] != spanB {
						t.Errorf("echo TOOL_COMPLETED span=(%v, parent=%v), want (%v, parent=%v)",
							echoComp["span_id"], echoComp["parent_span_id"], echoStart["span_id"], spanB)
					}
					mustLatency(t, echoComp)
				}
				endBIdx := agentEndIndexes(rows, "test-agent", spanB)[0]
				finishCompIdx := slices.IndexFunc(rows, func(r map[string]any) bool {
					return r["event_type"] == EventToolCompleted && r["parent_span_id"] == spanB &&
						jsonObjectColumn(t, r, "content")["tool"] == finishTaskToolName
				})
				if endBIdx < finishCompIdx {
					t.Errorf("peer B AGENT_COMPLETED at %d before its finish_task TOOL_COMPLETED at %d", endBIdx, finishCompIdx)
				}
			})
		}
	})

	t.Run("merged_batch_state_delta_with_pending_retry_call_does_not_retire_early", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		taskAgent, err := llmagent.New(llmagent.Config{
			Name:  "test-agent",
			Mode:  llmagent.ModeTask,
			Model: &scriptedLLM{},
		})
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		echo := &fakeTool{name: "echo"}
		finish := &fakeTool{name: finishTaskToolName}
		ic := &runningInvocation{running: taskAgent}
		act := &taskActivationContext{mockInvocationContext: mCtx, ic: ic}
		runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
		echoActions := &session.EventActions{StateDelta: map[string]any{"echo": 1}}
		finishActions1 := &session.EventActions{StateDelta: map[string]any{"fin": 1}}
		finishActions2 := &session.EventActions{StateDelta: map[string]any{"fin": 2}}
		toolEcho := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-echo"), ic: ic},
			actions:               echoActions,
		}
		toolFinish1 := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: ic},
			actions:               finishActions1,
		}
		toolFinish2 := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: ic},
			actions:               finishActions2,
		}
		finishFC := genai.NewContentFromParts([]*genai.Part{{
			FunctionCall: &genai.FunctionCall{ID: sharedCallID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
		}}, genai.RoleModel)

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(act))
		errs.add(p.BeforeToolCallback()(toolEcho, echo, map[string]any{"m": "1"}))
		errs.add(p.AfterToolCallback()(toolEcho, echo, map[string]any{"m": "1"}, map[string]any{"echo": "1"}, nil))
		errs.add(p.BeforeToolCallback()(toolFinish1, finish, map[string]any{"summary": "1"}))
		errs.add(p.OnToolErrorCallback()(toolFinish1, finish, map[string]any{"summary": "1"}, errors.New("boom")))
		errs.add(p.AfterToolCallback()(toolFinish1, finish, map[string]any{"summary": "1"}, map[string]any{"result": "Task completed."}, nil))

		// Next model turn emits a new finish_task FunctionCall with the same ID
		// while the first batch's merged [echo, finish_task] response is still buffered.
		errs.add(p.BeforeModelCallback()(act, &model.LLMRequest{}))
		errs.add(p.AfterModelCallback()(act, &model.LLMResponse{Content: finishFC}, nil))

		// Delivering the buffered merged response (carrying echo's StateDelta)
		// must consume the rejection rather than retiring the activation early.
		errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
			Author:  "test-agent",
			Actions: *echoActions,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call-echo", Name: "echo", Response: map[string]any{"echo": "1"}}},
				{FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}}},
			}, genai.RoleUser)},
		}))
		errs.add(p.OnUserMessageCallback()(runnerCtx, textContent("user", "still-open-marker")))

		// Now the second finish_task call runs and completes cleanly.
		errs.add(p.BeforeToolCallback()(toolFinish2, finish, map[string]any{"summary": "2"}))
		errs.add(p.AfterToolCallback()(toolFinish2, finish, map[string]any{"summary": "2"}, map[string]any{"result": "Task completed."}, nil))
		errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
			Author:  "test-agent",
			Actions: *finishActions2,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}}},
			}, genai.RoleUser)},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
		ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
		marker := slices.IndexFunc(rows, func(r map[string]any) bool { return r["event_type"] == EventUserMessageReceived })
		if len(starts) != 1 || len(ends) != 1 {
			t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED rows, want 1 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
		}
		endIdx := agentEndIndexes(rows, "test-agent", starts[0]["span_id"])[0]
		if endIdx < marker {
			t.Errorf("AGENT_COMPLETED emitted at index %d before marker %d (buffered rejection retired activation early); events=%v",
				endIdx, marker, eventTypes(rows))
		}
	})

	t.Run("merged_batch_state_delta_after_retry_started_does_not_retire_early", func(t *testing.T) {
		p, requests, mCtx := setupTestPlugin(t)
		taskAgent, err := llmagent.New(llmagent.Config{
			Name:  "test-agent",
			Mode:  llmagent.ModeTask,
			Model: &scriptedLLM{},
		})
		if err != nil {
			t.Fatalf("llmagent.New: %v", err)
		}
		echo := &fakeTool{name: "echo"}
		finish := &fakeTool{name: finishTaskToolName}
		ic := &runningInvocation{running: taskAgent}
		act := &taskActivationContext{mockInvocationContext: mCtx, ic: ic}
		runnerCtx := &taskActivationContext{mockInvocationContext: mCtx, ic: &runningInvocation{}}
		echoActions := &session.EventActions{StateDelta: map[string]any{"echo": 1}}
		finishActions1 := &session.EventActions{StateDelta: map[string]any{"fin": 1}}
		finishActions2 := &session.EventActions{StateDelta: map[string]any{"fin": 2}}
		toolEcho := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, "call-echo"), ic: ic},
			actions:               echoActions,
		}
		toolFinish1 := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: ic},
			actions:               finishActions1,
		}
		toolFinish2 := &taskToolContext{
			taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: ic},
			actions:               finishActions2,
		}
		finishFC := genai.NewContentFromParts([]*genai.Part{{
			FunctionCall: &genai.FunctionCall{ID: sharedCallID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
		}}, genai.RoleModel)

		var errs callbackErrors
		errs.add(p.BeforeAgentCallback()(act))
		errs.add(p.BeforeToolCallback()(toolEcho, echo, map[string]any{"m": "1"}))
		errs.add(p.AfterToolCallback()(toolEcho, echo, map[string]any{"m": "1"}, map[string]any{"echo": "1"}, nil))
		errs.add(p.BeforeToolCallback()(toolFinish1, finish, map[string]any{"summary": "1"}))
		errs.add(p.OnToolErrorCallback()(toolFinish1, finish, map[string]any{"summary": "1"}, errors.New("boom")))
		errs.add(p.AfterToolCallback()(toolFinish1, finish, map[string]any{"summary": "1"}, map[string]any{"result": "Task completed."}, nil))
		errs.add(p.BeforeModelCallback()(act, &model.LLMRequest{}))
		errs.add(p.AfterModelCallback()(act, &model.LLMResponse{Content: finishFC}, nil))

		// The retry's BeforeToolCallback runs before the first batch's merged
		// [echo, finish_task] response, which carries echo's StateDelta, is
		// delivered. The old response neither ends the retry's open call nor
		// shows that the retry superseded the rejection.
		errs.add(p.BeforeToolCallback()(toolFinish2, finish, map[string]any{"summary": "2"}))
		errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
			Author:  "test-agent",
			Actions: *echoActions,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "call-echo", Name: "echo", Response: map[string]any{"echo": "1"}}},
				{FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}}},
			}, genai.RoleUser)},
		}))
		errs.add(p.OnUserMessageCallback()(runnerCtx, textContent("user", "still-open-marker")))
		errs.add(p.AfterToolCallback()(toolFinish2, finish, map[string]any{"summary": "2"}, map[string]any{"result": "Task completed."}, nil))
		errs.add(p.OnEventCallback()(runnerCtx, &session.Event{
			Author:  "test-agent",
			Actions: *finishActions2,
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}}},
			}, genai.RoleUser)},
		}))
		errs.check(t)
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
		ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
		if len(starts) != 1 || len(ends) != 1 {
			t.Fatalf("got %d AGENT_STARTING and %d AGENT_COMPLETED rows, want 1 and 1; events=%v", len(starts), len(ends), eventTypes(rows))
		}
		endIdx := agentEndIndexes(rows, "test-agent", starts[0]["span_id"])[0]
		marker := slices.IndexFunc(rows, func(r map[string]any) bool { return r["event_type"] == EventUserMessageReceived })
		isFinishRow := func(eventType string) func(map[string]any) bool {
			return func(r map[string]any) bool {
				return r["event_type"] == eventType && jsonObjectColumn(t, r, "content")["tool"] == finishTaskToolName
			}
		}
		var finishStarts []map[string]any
		for _, r := range rows {
			if isFinishRow(EventToolStarting)(r) {
				finishStarts = append(finishStarts, r)
			}
		}
		retryCompIdx := slices.IndexFunc(rows, isFinishRow(EventToolCompleted))
		if len(finishStarts) != 2 || retryCompIdx < 0 {
			t.Fatalf("got %d finish_task TOOL_STARTING rows and TOOL_COMPLETED at %d, want 2 and the retry's; events=%v", len(finishStarts), retryCompIdx, eventTypes(rows))
		}
		if endIdx < marker || endIdx < retryCompIdx {
			t.Errorf("AGENT_COMPLETED at %d, before the still-open marker at %d or the retry's finish_task TOOL_COMPLETED at %d (the old batch's response retired the activation early); events=%v",
				endIdx, marker, retryCompIdx, eventTypes(rows))
		}
		retryComp := rows[retryCompIdx]
		if retryComp["span_id"] != finishStarts[1]["span_id"] {
			t.Errorf("retry finish_task TOOL_COMPLETED span_id = %v, want the retry's TOOL_STARTING span %v (the old batch's response ended the retry's open call)",
				retryComp["span_id"], finishStarts[1]["span_id"])
		}
		mustLatency(t, retryComp)
	})

	t.Run("hidden_peer_response_before_state_delta_rejection_and_parent_after_agent", func(t *testing.T) {
		for _, deliverDelayedRejection := range []bool{false, true} {
			mode := "dropped_rejection_under_parent_after_agent"
			if deliverDelayedRejection {
				mode = "delayed_state_delta_rejection_arrives_second"
			}
			t.Run(mode, func(t *testing.T) {
				p, requests, mCtx := setupTestPlugin(t)
				taskAgent, err := llmagent.New(llmagent.Config{
					Name:  "test-agent",
					Mode:  llmagent.ModeTask,
					Model: &scriptedLLM{},
				})
				if err != nil {
					t.Fatalf("llmagent.New: %v", err)
				}
				rootAgent, err := agent.New(agent.Config{
					Name:      "root",
					SubAgents: []agent.Agent{taskAgent},
				})
				if err != nil {
					t.Fatalf("agent.New(root): %v", err)
				}
				finish := &fakeTool{name: finishTaskToolName}
				icRoot := &runningInvocation{running: rootAgent}
				icA := &runningInvocation{running: taskAgent}
				icB := &runningInvocation{running: taskAgent}
				rootCtx := &taskActivationContext{mockInvocationContext: &mockInvocationContext{ctx: context.Background(), invocationID: mCtx.InvocationID(), agentName: "root"}, ic: icRoot}
				actA := &taskActivationContext{mockInvocationContext: mCtx, ic: icA}
				actB := &taskActivationContext{mockInvocationContext: mCtx, ic: icB}
				rejActions := &session.EventActions{StateDelta: map[string]any{"rej": 1}}
				toolA := &taskToolContext{
					taskActivationContext: &taskActivationContext{mockInvocationContext: withFunctionCall(mCtx, sharedCallID), ic: icA},
					actions:               rejActions,
				}
				finishFC := genai.NewContentFromParts([]*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: sharedCallID, Name: finishTaskToolName, Args: map[string]any{"summary": "done"}},
				}}, genai.RoleModel)

				var errs callbackErrors
				errs.add(p.BeforeAgentCallback()(rootCtx))
				errs.add(p.BeforeAgentCallback()(actA))
				errs.add(p.BeforeToolCallback()(toolA, finish, map[string]any{"summary": "a"}))
				errs.add(p.OnToolErrorCallback()(toolA, finish, map[string]any{"summary": "a"}, errors.New("boom")))
				errs.add(p.AfterToolCallback()(toolA, finish, map[string]any{"summary": "a"}, map[string]any{"result": "Task completed."}, nil))

				errs.add(p.BeforeAgentCallback()(actB))
				errs.add(p.BeforeModelCallback()(actB, &model.LLMRequest{}))
				errs.add(p.AfterModelCallback()(actB, &model.LLMResponse{Content: finishFC}, nil))
				// Peer B's valid response (with hidden tool callbacks, no StateDelta) arrives first.
				errs.add(p.OnEventCallback()(rootCtx, finishTaskFREvent("test-agent", "", sharedCallID)))
				if deliverDelayedRejection {
					// A's delayed rejected response arrives second with A's StateDelta,
					// proving the first response belonged to Peer B.
					errs.add(p.OnEventCallback()(rootCtx, &session.Event{
						Author:  "test-agent",
						Actions: *rejActions,
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromParts([]*genai.Part{{
							FunctionResponse: &genai.FunctionResponse{ID: sharedCallID, Name: finishTaskToolName, Response: map[string]any{"result": "Task completed."}},
						}}, genai.RoleUser)},
					}))
				}
				errs.add(p.OnEventCallback()(rootCtx, &session.Event{
					Author:      "root",
					LLMResponse: model.LLMResponse{Content: textContent("model", "root done")},
				}))
				errs.add(p.AfterAgentCallback()(rootCtx))
				errs.check(t)
				p.AfterRunCallback()(rootCtx)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				starts := agentRowsOfType(rows, EventAgentStarting, "test-agent")
				ends := agentRowsOfType(rows, EventAgentCompleted, "test-agent")
				rootEnds := agentRowsOfType(rows, EventAgentCompleted, "root")
				if len(starts) != 2 || len(ends) != 1 || len(rootEnds) != 1 {
					t.Fatalf("got task starts=%d task ends=%d root ends=%d, want 2, 1, 1; events=%v",
						len(starts), len(ends), len(rootEnds), eventTypes(rows))
				}
				if got, want := ends[0]["span_id"], starts[1]["span_id"]; got != want {
					t.Errorf("task AGENT_COMPLETED span_id = %v, want peer B span_id %v (A=%v)", got, want, starts[0]["span_id"])
				}
				taskEndIdx := agentEndIndexes(rows, "test-agent", starts[1]["span_id"])[0]
				rootEndIdx := agentEndIndexes(rows, "root", rootEnds[0]["span_id"])[0]
				if taskEndIdx > rootEndIdx {
					t.Errorf("task AGENT_COMPLETED at %d came after root AGENT_COMPLETED at %d", taskEndIdx, rootEndIdx)
				}
			})
		}
	})
}
