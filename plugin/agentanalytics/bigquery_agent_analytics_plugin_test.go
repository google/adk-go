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
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
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
				tt.want, fmt.Sprintf(`{"api_key":%q}`, tt.want)), "args")
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
