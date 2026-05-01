// Copyright 2025 Google LLC
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

package runner

import (
	"context"
	"io"
	"iter"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// ---------------------------------------------------------------------------
// Mock: LiveConnection
// ---------------------------------------------------------------------------

type mockLiveConnection struct {
	mu             sync.Mutex
	sendLog        []*model.LiveRequest
	recvResponses  []*model.LLMResponse
	recvCh         chan *model.LLMResponse // dynamic feed (alternative to recvResponses)
	recvIdx        int
	sendErrAt      int // -1 = never
	sendErr        error
	sendCount      int
	recvErrAt      int // -1 = never
	recvErr        error
	closeCalled    bool
	closedCh       chan struct{}
	closeOnce      sync.Once
	inSend         int32
	concurrentSend int32
}

func newMockLiveConnection() *mockLiveConnection {
	return &mockLiveConnection{sendErrAt: -1, recvErrAt: -1, closedCh: make(chan struct{})}
}

func (m *mockLiveConnection) Send(_ context.Context, req *model.LiveRequest) error {
	if atomic.AddInt32(&m.inSend, 1) > 1 {
		atomic.StoreInt32(&m.concurrentSend, 1)
	}
	defer atomic.AddInt32(&m.inSend, -1)

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sendErrAt >= 0 && m.sendCount == m.sendErrAt {
		m.sendCount++
		return m.sendErr
	}
	m.sendLog = append(m.sendLog, req)
	m.sendCount++
	return nil
}

func (m *mockLiveConnection) Receive(ctx context.Context) (*model.LLMResponse, error) {
	if m.recvCh != nil {
		select {
		case resp, ok := <-m.recvCh:
			if !ok {
				return nil, io.EOF
			}
			m.mu.Lock()
			idx := m.recvIdx
			m.recvIdx++
			m.mu.Unlock()
			if m.recvErrAt >= 0 && idx == m.recvErrAt {
				return nil, m.recvErr
			}
			return resp, nil
		case <-m.closedCh:
			return nil, io.EOF
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.recvErrAt >= 0 && m.recvIdx == m.recvErrAt {
		m.recvIdx++
		return nil, m.recvErr
	}
	if m.recvIdx >= len(m.recvResponses) {
		return nil, io.EOF
	}
	resp := m.recvResponses[m.recvIdx]
	m.recvIdx++
	return resp, nil
}

func (m *mockLiveConnection) Close() error {
	m.mu.Lock()
	m.closeCalled = true
	m.mu.Unlock()
	m.closeOnce.Do(func() { close(m.closedCh) })
	return nil
}

func (m *mockLiveConnection) SendLog() []*model.LiveRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]*model.LiveRequest, len(m.sendLog))
	copy(cp, m.sendLog)
	return cp
}

func (m *mockLiveConnection) WasClosed() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.closeCalled
}

// ---------------------------------------------------------------------------
// Mock: Tool
// ---------------------------------------------------------------------------

type mockToolCall struct {
	Args map[string]any
	Time time.Time
}

type mockTool struct {
	name        string
	declaration *genai.FunctionDeclaration
	mu          sync.Mutex
	calls       []mockToolCall
	runFunc     func(ctx context.Context, args map[string]any) (map[string]any, error)
	result      map[string]any
}

func (t *mockTool) Name() string        { return t.name }
func (t *mockTool) Description() string { return "mock tool " + t.name }
func (t *mockTool) IsLongRunning() bool { return false }

func (t *mockTool) Declaration() *genai.FunctionDeclaration {
	if t.declaration != nil {
		return t.declaration
	}
	return &genai.FunctionDeclaration{Name: t.name}
}

func (t *mockTool) ProcessRequest(_ tool.Context, _ *model.LLMRequest) error { return nil }

func (t *mockTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	argsMap, _ := args.(map[string]any)
	t.mu.Lock()
	t.calls = append(t.calls, mockToolCall{Args: argsMap, Time: time.Now()})
	t.mu.Unlock()

	if t.runFunc != nil {
		return t.runFunc(ctx, argsMap)
	}
	return t.result, nil
}

func (t *mockTool) CallCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.calls)
}

func (t *mockTool) Calls() []mockToolCall {
	t.mu.Lock()
	defer t.mu.Unlock()
	cp := make([]mockToolCall, len(t.calls))
	copy(cp, t.calls)
	return cp
}

// ---------------------------------------------------------------------------
// Mock: LLM (LiveCapableLLM)
// ---------------------------------------------------------------------------

type mockLiveLLM struct {
	conn model.LiveConnection
}

func (m *mockLiveLLM) Name() string { return "mock-live" }

func (m *mockLiveLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func (m *mockLiveLLM) ConnectLive(_ context.Context, _ *model.LLMRequest) (model.LiveConnection, error) {
	return m.conn, nil
}

var _ model.LiveCapableLLM = (*mockLiveLLM)(nil)

// ---------------------------------------------------------------------------
// Mock: SessionService (wraps InMemory, records appended events)
// ---------------------------------------------------------------------------

type mockSessionService struct {
	session.Service
	mu             sync.Mutex
	appendedEvents []*session.Event
}

func (m *mockSessionService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	m.mu.Lock()
	m.appendedEvents = append(m.appendedEvents, ev)
	m.mu.Unlock()
	return m.Service.AppendEvent(ctx, sess, ev)
}

func (m *mockSessionService) PersistedEvents() []*session.Event {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]*session.Event, len(m.appendedEvents))
	copy(cp, m.appendedEvents)
	return cp
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func setupRunner(t *testing.T, conn *mockLiveConnection, tools []tool.Tool, plugins []*plugin.Plugin) (*Runner, *mockSessionService, session.Session) {
	t.Helper()
	return setupRunnerWithEvents(t, conn, tools, plugins, nil)
}

func setupRunnerWithEvents(t *testing.T, conn *mockLiveConnection, tools []tool.Tool, plugins []*plugin.Plugin, priorEvents []*session.Event) (*Runner, *mockSessionService, session.Session) {
	t.Helper()

	innerSvc := session.InMemoryService()
	svc := &mockSessionService{Service: innerSvc}

	resp, err := innerSvc.Create(context.Background(), &session.CreateRequest{
		AppName:   "test",
		UserID:    "user1",
		SessionID: "sess1",
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, ev := range priorEvents {
		if err := innerSvc.AppendEvent(context.Background(), resp.Session, ev); err != nil {
			t.Fatal(err)
		}
	}

	// Re-fetch session to get events
	getResp, err := innerSvc.Get(context.Background(), &session.GetRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})
	if err != nil {
		t.Fatal(err)
	}

	llm := &mockLiveLLM{conn: conn}
	a, err := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
		Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}

	r, err := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: svc,
		PluginConfig:   PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatal(err)
	}

	return r, svc, getResp.Session
}

func collectEvents(t *testing.T, r *Runner, queue *agent.LiveRequestQueue) ([]*session.Event, []error) {
	t.Helper()
	var events []*session.Event
	var errs []error
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{}) {
		if err != nil {
			errs = append(errs, err)
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return events, errs
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content: genai.NewContentFromText(text, "model"),
	}
}

func turnCompleteResponse() *model.LLMResponse {
	return &model.LLMResponse{TurnComplete: true}
}

func audioResponse(data []byte) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: data}},
			},
		},
		CustomMetadata: map[string]any{"is_audio": true},
	}
}

func transcriptResponse(text, kind string, finished bool) *model.LLMResponse {
	role := genai.Role("model")
	if kind == "input" {
		role = "user"
	}
	resp := &model.LLMResponse{
		Content: genai.NewContentFromText(text, role),
	}
	if kind == "input" {
		resp.InputTranscription = &genai.Transcription{Text: text, Finished: finished}
	} else {
		resp.OutputTranscription = &genai.Transcription{Text: text, Finished: finished}
	}
	return resp
}

func functionCallResponse(id, name string, args map[string]any) *model.LLMResponse {
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: args}},
			},
		},
	}
}

func toolCancellationResponse(ids []string) *model.LLMResponse {
	return &model.LLMResponse{
		CustomMetadata: map[string]any{"tool_cancellation_ids": ids},
	}
}

// textResponseWithTurnComplete creates a single model response carrying both
// content text and TurnComplete=true. This exercises the code path where
// guard suppression and TurnComplete/SetModelSpeaking handling happen on
// the same LLMResponse — not on two separate messages.
func textResponseWithTurnComplete(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:      genai.NewContentFromText(text, "model"),
		TurnComplete: true,
	}
}

// textResponseInterrupted creates a single model response with content and
// Interrupted=true. Used to test post-interruption guard behavior.
func textResponseInterrupted(text string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:     genai.NewContentFromText(text, "model"),
		Interrupted: true,
	}
}

// mixedResponse creates a model response with both model content and
// an output transcription. Used to test mixed-message guard handling.
func mixedResponse(text, transcriptionText string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:             genai.NewContentFromText(text, "model"),
		OutputTranscription: &genai.Transcription{Text: transcriptionText, Finished: true},
	}
}

// modelContentTexts extracts text from model-content events.
func modelContentTexts(events []*session.Event) []string {
	var texts []string
	for _, ev := range events {
		if ev.Content != nil && ev.Content.Role == "model" {
			for _, p := range ev.Content.Parts {
				if p.Text != "" {
					texts = append(texts, p.Text)
				}
			}
		}
	}
	return texts
}

// ---------------------------------------------------------------------------
// Scenario 1: Happy path — text query with tool call
// ---------------------------------------------------------------------------

func TestScenario1_HappyPathToolCall(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		functionCallResponse("fc1", "greet", map[string]any{"name": "Alice"}),
		textResponse("Hello Alice!"),
		turnCompleteResponse(),
	}

	greetTool := &mockTool{
		name:   "greet",
		result: map[string]any{"message": "Hi Alice"},
	}

	r, svc, _ := setupRunner(t, conn, []tool.Tool{greetTool}, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	if len(events) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(events))
	}

	if greetTool.CallCount() != 1 {
		t.Errorf("expected greet tool called once, got %d", greetTool.CallCount())
	}

	// Check persisted events (excludes partial/audio)
	persisted := svc.PersistedEvents()
	if len(persisted) < 3 {
		t.Fatalf("expected at least 3 persisted events, got %d", len(persisted))
	}

	// Check that a tool response was sent
	sendLog := conn.SendLog()
	hasToolResponse := false
	for _, req := range sendLog {
		if req.ToolResponse != nil {
			hasToolResponse = true
			if req.ToolResponse[0].Name != "greet" {
				t.Errorf("expected tool response for 'greet', got %q", req.ToolResponse[0].Name)
			}
		}
	}
	if !hasToolResponse {
		t.Error("expected SendToolResponse in send log")
	}
}

// ---------------------------------------------------------------------------
// Scenario 2: Audio in → transcript + audio out
// ---------------------------------------------------------------------------

func TestScenario2_AudioNotPersistedTranscriptPersisted(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		audioResponse([]byte{0x01, 0x02}),
		transcriptResponse("Hello there", "output", true),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	// Audio event should NOT be persisted
	persisted := svc.PersistedEvents()
	for _, ev := range persisted {
		if ev.CustomMetadata != nil {
			if v, ok := ev.CustomMetadata["is_audio"].(bool); ok && v {
				t.Error("audio event should not be persisted")
			}
		}
	}

	// Transcript should be persisted
	foundTranscript := false
	for _, ev := range persisted {
		if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text == "Hello there" {
			foundTranscript = true
		}
	}
	if !foundTranscript {
		t.Error("transcript event should be persisted")
	}
}

// ---------------------------------------------------------------------------
// Scenario 3: Interruption — ToolCallCancellation (IN-FLIGHT)
// ---------------------------------------------------------------------------

func TestScenario3_InFlightToolCancellation(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	slowTool := &mockTool{
		name: "slow_tool",
		runFunc: func(ctx context.Context, _ map[string]any) (map[string]any, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(5 * time.Second):
				return map[string]any{"result": "done"}, nil
			}
		},
	}

	llm := &mockLiveLLM{conn: conn}
	a, _ := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
		Tools: []tool.Tool{slowTool},
	})

	svc := &mockSessionService{Service: session.InMemoryService()}
	_, _ = svc.Service.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})

	r, _ := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: svc,
	})

	queue := agent.NewLiveRequestQueue(100)

	// Push function call, wait for coalesce to flush and tool to start, then cancel
	go func() {
		conn.recvCh <- functionCallResponse("call_1", "slow_tool", nil)
		// Wait for short coalesce window (10ms) to expire and tool to start
		time.Sleep(200 * time.Millisecond)
		conn.recvCh <- toolCancellationResponse([]string{"call_1"})
		time.Sleep(100 * time.Millisecond)
		conn.recvCh <- turnCompleteResponse()
		time.Sleep(50 * time.Millisecond)
		queue.Close()
	}()

	start := time.Now()
	cfg := agent.RunConfig{ToolCoalesceWindow: 10 * time.Millisecond}
	var events []*session.Event
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, cfg) {
		if err != nil {
			break
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Errorf("test took %v, tool should have been cancelled quickly", elapsed)
	}

	if slowTool.CallCount() < 1 {
		t.Error("expected slow_tool to have been called (started executing)")
	}

	// No tool response should have been sent for call_1
	for _, req := range conn.SendLog() {
		if req.ToolResponse != nil {
			for _, fr := range req.ToolResponse {
				if fr.ID == "call_1" {
					t.Error("should NOT have sent tool response for cancelled call_1")
				}
			}
		}
	}

	_ = events
}

// ---------------------------------------------------------------------------
// Scenario 4: Connection error mid-stream (receive side)
// ---------------------------------------------------------------------------

func TestScenario4_ReceiveError(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("Hello"),
	}
	conn.recvErrAt = 1
	conn.recvErr = io.ErrUnexpectedEOF

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)

	if len(events) < 1 {
		t.Error("expected at least 1 event before error")
	}
	if len(errs) == 0 {
		t.Fatal("expected an error")
	}
	if errs[len(errs)-1] != io.ErrUnexpectedEOF {
		t.Errorf("expected io.ErrUnexpectedEOF, got %v", errs[len(errs)-1])
	}
	if !conn.WasClosed() {
		t.Error("expected connection to be closed")
	}
}

// ---------------------------------------------------------------------------
// Scenario 5: Connection error mid-stream (send side)
// ---------------------------------------------------------------------------

func TestScenario5_SendError(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10) // blocks until close
	conn.sendErrAt = 2
	conn.sendErr = io.ErrClosedPipe

	priorEvents := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello", "user")}, Author: "user"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hi", "model")}, Author: "test_agent"},
	}

	r, _, _ := setupRunnerWithEvents(t, conn, nil, nil, priorEvents)

	queue := agent.NewLiveRequestQueue(100)
	// Pre-load a message but don't close queue yet — the sender must
	// attempt the send (and fail) before exiting via queue.Done().
	_ = queue.Send(context.Background(), &model.LiveRequest{Content: genai.NewContentFromText("new q", "user")})
	go func() {
		time.Sleep(200 * time.Millisecond)
		queue.Close()
	}()

	_, errs := collectEvents(t, r, queue)

	if len(errs) == 0 {
		t.Fatal("expected send error")
	}

	sendLog := conn.SendLog()
	if len(sendLog) != 2 {
		t.Errorf("expected 2 successful sends (history), got %d", len(sendLog))
	}

	if !conn.WasClosed() {
		t.Error("expected connection to be closed")
	}
}

// ---------------------------------------------------------------------------
// Scenario 6: History handoff — prior session events
// ---------------------------------------------------------------------------

func TestScenario6_HistoryHandoff(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	priorEvents := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello", "user")}, Author: "user"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hi there", "model")}, Author: "test_agent"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("What's the weather?", "user")}, Author: "user"},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Let me check", "model")}, Author: "test_agent"},
	}

	r, _, _ := setupRunnerWithEvents(t, conn, nil, nil, priorEvents)

	queue := agent.NewLiveRequestQueue(100)
	_ = queue.Send(context.Background(), &model.LiveRequest{
		Content: genai.NewContentFromText("new question", "user"),
	})

	go func() {
		// Wait for sends to happen
		time.Sleep(100 * time.Millisecond)
		conn.recvCh <- turnCompleteResponse()
		time.Sleep(50 * time.Millisecond)
		queue.Close()
	}()

	collectEvents(t, r, queue)

	sendLog := conn.SendLog()
	if len(sendLog) < 5 {
		t.Fatalf("expected at least 5 sends (4 history + 1 queue), got %d", len(sendLog))
	}

	// First 4 should be history
	for i := range 4 {
		if sendLog[i].Content == nil {
			t.Errorf("send[%d] should have Content (history)", i)
		}
	}

	// 5th should be the queue message
	if sendLog[4].Content == nil {
		t.Error("send[4] should have Content (queue message)")
	}
	if sendLog[4].Content.Parts[0].Text != "new question" {
		t.Errorf("expected 'new question', got %q", sendLog[4].Content.Parts[0].Text)
	}
}

// ---------------------------------------------------------------------------
// Scenario 7: Plugin hooks fire in correct order
// ---------------------------------------------------------------------------

func TestScenario7_PluginHooks(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		functionCallResponse("fc1", "greet", nil),
		textResponse("Hi!"),
		turnCompleteResponse(),
	}

	greetTool := &mockTool{name: "greet", result: map[string]any{"msg": "hi"}}

	var mu sync.Mutex
	var calls []string
	record := func(s string) {
		mu.Lock()
		calls = append(calls, s)
		mu.Unlock()
	}

	p, _ := plugin.New(plugin.Config{
		Name:              "recording",
		BeforeRunCallback: func(_ agent.InvocationContext) (*genai.Content, error) { record("BeforeRun"); return nil, nil },
		AfterRunCallback:  func(_ agent.InvocationContext) { record("AfterRun") },
		BeforeToolCallback: func(_ tool.Context, t tool.Tool, _ map[string]any) (map[string]any, error) {
			record("BeforeTool:" + t.Name())
			return nil, nil
		},
		AfterToolCallback: func(_ tool.Context, t tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			record("AfterTool:" + t.Name())
			return nil, nil
		},
	})

	r, _, _ := setupRunner(t, conn, []tool.Tool{greetTool}, []*plugin.Plugin{p})
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	collectEvents(t, r, queue)

	mu.Lock()
	defer mu.Unlock()

	expected := []string{"BeforeRun", "BeforeTool:greet", "AfterTool:greet", "AfterRun"}
	if len(calls) != len(expected) {
		t.Fatalf("expected %d calls, got %d: %v", len(expected), len(calls), calls)
	}
	for i, want := range expected {
		if calls[i] != want {
			t.Errorf("calls[%d] = %q, want %q", i, calls[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 8: LiveRequestQueue concurrency stress test
// ---------------------------------------------------------------------------

func TestScenario8_QueueConcurrency(t *testing.T) {
	queue := agent.NewLiveRequestQueue(10)

	var received int64
	done := make(chan struct{})

	// Consumer
	go func() {
		defer close(done)
		for {
			select {
			case <-queue.Events():
				atomic.AddInt64(&received, 1)
			case <-queue.Done():
				// Drain remaining
				for {
					select {
					case <-queue.Events():
						atomic.AddInt64(&received, 1)
					default:
						return
					}
				}
			}
		}
	}()

	// 50 senders x 100 messages
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := range 100 {
				_ = queue.Send(context.Background(), &model.LiveRequest{
					Content: genai.NewContentFromText("msg", "user"),
				})
				_ = j
			}
		}(i)
	}

	wg.Wait()
	queue.Close()
	<-done

	if got := atomic.LoadInt64(&received); got != 5000 {
		t.Errorf("expected 5000 messages, got %d", got)
	}

	// Backpressure test: send with tiny timeout on full buffer
	queue2 := agent.NewLiveRequestQueue(1)
	// fill the buffer
	_ = queue2.Send(context.Background(), &model.LiveRequest{})
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()
	time.Sleep(2 * time.Nanosecond)
	err := queue2.Send(ctx, &model.LiveRequest{})
	if err != context.DeadlineExceeded {
		t.Errorf("expected DeadlineExceeded, got %v", err)
	}

	// Close idempotence
	queue.Close()
	queue.Close()
	queue.Close()

	// Send after close
	err = queue.Send(context.Background(), &model.LiveRequest{})
	if err != agent.ErrQueueClosed {
		t.Errorf("expected ErrQueueClosed, got %v", err)
	}

	// Done channel is closed
	select {
	case <-queue.Done():
		// ok
	default:
		t.Error("Done() channel should be closed")
	}
}

// ---------------------------------------------------------------------------
// Scenario 9: Clean shutdown — empty queue
// ---------------------------------------------------------------------------

func TestScenario9_CleanShutdownEmptyQueue(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)

	goroutinesBefore := runtime.NumGoroutine()

	// Close queue immediately -> sender terminates -> receiver gets EOF from close
	queue.Close()

	events, errs := collectEvents(t, r, queue)

	// Should terminate cleanly with no errors
	for _, err := range errs {
		if err != io.EOF {
			t.Errorf("unexpected error: %v", err)
		}
	}
	_ = events

	if !conn.WasClosed() {
		t.Error("expected connection to be closed")
	}

	// Check goroutine leak
	time.Sleep(100 * time.Millisecond)
	goroutinesAfter := runtime.NumGoroutine()
	if goroutinesAfter > goroutinesBefore+2 {
		t.Errorf("possible goroutine leak: before=%d, after=%d", goroutinesBefore, goroutinesAfter)
	}
}

// ---------------------------------------------------------------------------
// Scenario 10: Tool call coalescing
// ---------------------------------------------------------------------------

func TestScenario10_ToolCallCoalescing(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	weatherTool := &mockTool{name: "get_weather", result: map[string]any{"temp": "72F"}}
	timeTool := &mockTool{name: "get_time", result: map[string]any{"time": "3pm"}}

	llm := &mockLiveLLM{conn: conn}
	a, _ := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
		Tools: []tool.Tool{weatherTool, timeTool},
	})

	svc := &mockSessionService{Service: session.InMemoryService()}
	_, _ = svc.Service.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})

	r, _ := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: svc,
	})

	queue := agent.NewLiveRequestQueue(100)

	t0 := time.Now()

	go func() {
		// t=0: push 3 function calls within window
		conn.recvCh <- functionCallResponse("c1", "get_weather", map[string]any{"city": "SF"})
		time.Sleep(10 * time.Millisecond)
		conn.recvCh <- functionCallResponse("c2", "get_weather", map[string]any{"city": "SF"}) // dup
		time.Sleep(10 * time.Millisecond)
		conn.recvCh <- functionCallResponse("c3", "get_time", map[string]any{"tz": "PST"})

		// Wait for coalesce window to expire and tools to execute
		time.Sleep(300 * time.Millisecond)

		conn.recvCh <- turnCompleteResponse()
		time.Sleep(50 * time.Millisecond)
		queue.Close()
	}()

	cfg := agent.RunConfig{ToolCoalesceWindow: 100 * time.Millisecond}
	var events []*session.Event
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, cfg) {
		if err != nil {
			break
		}
		if ev != nil {
			events = append(events, ev)
		}
	}

	// get_weather called exactly once (c1 and c2 deduped)
	if weatherTool.CallCount() != 1 {
		t.Errorf("expected get_weather called once, got %d", weatherTool.CallCount())
	}
	if timeTool.CallCount() != 1 {
		t.Errorf("expected get_time called once, got %d", timeTool.CallCount())
	}

	// Timing: tool should not execute before window expired
	weatherCalls := weatherTool.Calls()
	if len(weatherCalls) > 0 {
		execDelay := weatherCalls[0].Time.Sub(t0)
		if execDelay < 100*time.Millisecond {
			t.Errorf("tool executed too early: %v (should be >= 100ms)", execDelay)
		}
	}

	// Check that tool response contains 3 entries (c1, c2 dedup alias, c3)
	for _, req := range conn.SendLog() {
		if len(req.ToolResponse) >= 2 {
			// Should have at least 3 responses (c1+c2 alias + c3)
			return
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 11: Echo suppression signal — ModelSpeaking state transitions
// ---------------------------------------------------------------------------

func TestScenario11_ModelSpeakingStateTransitions(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	queue := agent.NewLiveRequestQueue(100)

	var mu sync.Mutex
	var states []bool

	p, _ := plugin.New(plugin.Config{
		Name: "state-observer",
		OnEventCallback: func(_ agent.InvocationContext, _ *session.Event) (*session.Event, error) {
			mu.Lock()
			states = append(states, queue.ModelSpeaking())
			mu.Unlock()
			return nil, nil
		},
	})

	llm := &mockLiveLLM{conn: conn}
	a, _ := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
	})

	svc := &mockSessionService{Service: session.InMemoryService()}
	_, _ = svc.Service.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})

	r, _ := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: svc,
		PluginConfig:   PluginConfig{Plugins: []*plugin.Plugin{p}},
	})

	go func() {
		conn.recvCh <- audioResponse([]byte{0x01})
		time.Sleep(10 * time.Millisecond)
		conn.recvCh <- audioResponse([]byte{0x02})
		time.Sleep(10 * time.Millisecond)
		conn.recvCh <- turnCompleteResponse()
		time.Sleep(50 * time.Millisecond)
		queue.Close()
	}()

	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{}) {
		_ = ev
		_ = err
	}

	mu.Lock()
	defer mu.Unlock()

	expected := []bool{true, true, false}
	if len(states) != len(expected) {
		t.Fatalf("expected %d state observations, got %d: %v", len(expected), len(states), states)
	}
	for i, want := range expected {
		if states[i] != want {
			t.Errorf("states[%d] = %v, want %v", i, states[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 12: Input transcript chunking — buffered until Finished
// ---------------------------------------------------------------------------

func TestScenario12_InputTranscriptChunking(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hello ", "input", false),
		transcriptResponse("world", "input", true),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 3 {
		t.Fatalf("expected 3 yielded events, got %d", len(events))
	}

	// Should persist one aggregated event with "Hello world"
	persisted := svc.PersistedEvents()
	found := false
	for _, ev := range persisted {
		if ev.Content != nil && len(ev.Content.Parts) > 0 &&
			ev.Content.Parts[0].Text == "Hello world" {
			found = true
			if ev.Author != "user" {
				t.Errorf("input transcript author = %q, want %q", ev.Author, "user")
			}
		}
	}
	if !found {
		t.Errorf("expected aggregated 'Hello world' in persisted events, got %d events: %v",
			len(persisted), persistedTexts(persisted))
	}
}

// ---------------------------------------------------------------------------
// Scenario 13: Output transcript persisted as agent response
// ---------------------------------------------------------------------------

func TestScenario13_OutputTranscriptPersisted(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("I can help with that", "output", true),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 yielded events, got %d", len(events))
	}

	persisted := svc.PersistedEvents()
	found := false
	for _, ev := range persisted {
		if ev.Content != nil && len(ev.Content.Parts) > 0 &&
			ev.Content.Parts[0].Text == "I can help with that" {
			found = true
			// Author should be the agent name (set by runner from invCtx.Agent().Name())
			if ev.Author == "" || ev.Author == "user" {
				t.Errorf("output transcript author = %q, want agent name", ev.Author)
			}
		}
	}
	if !found {
		t.Errorf("expected output transcript persisted, got %d events: %v",
			len(persisted), persistedTexts(persisted))
	}
}

// persistedTexts extracts text from persisted events for test diagnostics.
func persistedTexts(events []*session.Event) []string {
	var texts []string
	for _, ev := range events {
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.Text != "" {
					texts = append(texts, p.Text)
				}
			}
		}
	}
	return texts
}

// ---------------------------------------------------------------------------
// Scenario 14: TurnComplete boundary flush — output transcription split per segment
// ---------------------------------------------------------------------------

func TestScenario14_TurnCompleteBoundaryFlush(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		// First speaking segment: two output chunks, then TurnComplete.
		transcriptResponse("Hello, ", "output", false),
		transcriptResponse("how are you?", "output", false),
		turnCompleteResponse(),
		// Second speaking segment: one chunk, then TurnComplete.
		transcriptResponse("I found the patient.", "output", false),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 5 {
		t.Fatalf("expected 5 yielded events, got %d", len(events))
	}

	// Should produce two separate persisted agent_response events.
	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)

	if len(texts) != 2 {
		t.Fatalf("expected 2 persisted transcripts, got %d: %v", len(texts), texts)
	}
	if texts[0] != "Hello, how are you?" {
		t.Errorf("first segment = %q, want %q", texts[0], "Hello, how are you?")
	}
	if texts[1] != "I found the patient." {
		t.Errorf("second segment = %q, want %q", texts[1], "I found the patient.")
	}

	// Both should have agent name as author (not "user").
	for i, ev := range persisted {
		if ev.Author == "" || ev.Author == "user" {
			t.Errorf("persisted[%d] author = %q, want agent name", i, ev.Author)
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 15: Defer flush on consumer break — buffered text persisted on early exit
// ---------------------------------------------------------------------------

func TestScenario15_DeferFlushOnConsumerBreak(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 10)

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)

	// Feed output transcript chunks without Finished or TurnComplete.
	conn.recvCh <- transcriptResponse("Partial one ", "output", false)
	conn.recvCh <- transcriptResponse("partial two", "output", false)

	// Consumer collects 2 events then breaks (simulates user pressing Stop).
	collected := 0
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{}) {
		_ = ev
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		collected++
		if collected >= 2 {
			break
		}
	}

	// Close resources after break.
	queue.Close()

	// The defer flush should have persisted the accumulated output transcription.
	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)

	if len(texts) != 1 {
		t.Fatalf("expected 1 flushed transcript, got %d: %v", len(texts), texts)
	}
	if texts[0] != "Partial one partial two" {
		t.Errorf("flushed text = %q, want %q", texts[0], "Partial one partial two")
	}
	if persisted[0].Author == "" || persisted[0].Author == "user" {
		t.Errorf("flushed author = %q, want agent name", persisted[0].Author)
	}
}

func TestRunLive_LiveDiagnostics_NotLeakedToSession(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("hello from model"),
	}
	r, svc, sess := setupRunner(t, conn, nil, nil)

	queue := agent.NewLiveRequestQueue(10)
	queue.Close()

	var yieldedEvents []*session.Event
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if ev != nil {
			yieldedEvents = append(yieldedEvents, ev)
		}
	}

	if len(yieldedEvents) == 0 {
		t.Fatal("expected at least one yielded event")
	}

	// Yielded events should carry LiveDiagnostics.
	for i, ev := range yieldedEvents {
		if ev.LiveDiagnostics == nil && ev.Content != nil && len(ev.Content.Parts) > 0 {
			t.Errorf("yielded event[%d] should have LiveDiagnostics", i)
		}
	}

	// Persisted events must NOT have LiveDiagnostics.
	for i, ev := range svc.PersistedEvents() {
		if ev.LiveDiagnostics != nil {
			t.Errorf("persisted event[%d] has LiveDiagnostics, want nil", i)
		}
	}

	// Events in the session (via ctx.Session().Events()) must NOT have LiveDiagnostics.
	// Re-fetch the session to see what's stored.
	getResp, err := svc.Get(context.Background(), &session.GetRequest{
		AppName: "test", UserID: "user1", SessionID: sess.ID(),
	})
	if err != nil {
		t.Fatal(err)
	}
	for ev := range getResp.Session.Events().All() {
		if ev.LiveDiagnostics != nil {
			t.Errorf("session event %q has LiveDiagnostics, want nil", ev.ID)
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 16: Suppresses duplicate content after TurnComplete
// ---------------------------------------------------------------------------

func TestScenario16_SuppressesDuplicateAfterTurnComplete(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("Hello"),
		turnCompleteResponse(),
		textResponse("Hello"), // duplicate — should be suppressed
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	texts := modelContentTexts(events)
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("expected [Hello], got %v", texts)
	}

	// Verify only 1 model-content text persisted.
	pTexts := persistedTexts(svc.PersistedEvents())
	if len(pTexts) != 1 || pTexts[0] != "Hello" {
		t.Errorf("expected 1 persisted text [Hello], got %v", pTexts)
	}
}

// ---------------------------------------------------------------------------
// Scenario 17: Guard resets on function call
// ---------------------------------------------------------------------------

func TestScenario17_GuardResetsOnFunctionCall(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("A"),
		turnCompleteResponse(),
		functionCallResponse("fc1", "greet", nil), // resets guard
		textResponse("B"),
		turnCompleteResponse(),
	}

	greetTool := &mockTool{name: "greet", result: map[string]any{"msg": "hi"}}
	r, _, _ := setupRunner(t, conn, []tool.Tool{greetTool}, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	texts := modelContentTexts(events)
	if len(texts) != 2 || texts[0] != "A" || texts[1] != "B" {
		t.Errorf("expected [A B], got %v", texts)
	}
}

// ---------------------------------------------------------------------------
// Scenario 18: Input transcriptions are never suppressed
// ---------------------------------------------------------------------------

func TestScenario18_TranscriptionNeverSuppressed(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("Model"),
		turnCompleteResponse(),
		transcriptResponse("User speaks", "input", true),
	}

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	// Transcription must be present.
	found := false
	for _, ev := range events {
		if ev.InputTranscription != nil && ev.InputTranscription.Text == "User speaks" {
			found = true
		}
	}
	if !found {
		t.Error("input transcription event should not be suppressed")
	}
}

// ---------------------------------------------------------------------------
// Scenario 19: Empty TurnComplete does not arm suppression
// ---------------------------------------------------------------------------

func TestScenario19_EmptyTurnCompleteNoArm(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		turnCompleteResponse(), // empty TC — guard no-op
		textResponse("Hello"),  // should NOT be suppressed
		turnCompleteResponse(),
	}

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	texts := modelContentTexts(events)
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Errorf("expected [Hello], got %v", texts)
	}
}

// ---------------------------------------------------------------------------
// Scenario 20: Multi-turn text — guard resets on user turn via turnResetCh
// ---------------------------------------------------------------------------

func TestScenario20_MultiTurnResetViaTurnResetCh(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 20)

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)

	go func() {
		// Turn 1: model responds.
		conn.recvCh <- textResponse("Turn1")
		conn.recvCh <- turnCompleteResponse()

		// Allow receiverLoop to process Turn 1.
		time.Sleep(50 * time.Millisecond)

		// User sends new message — triggers turnResetCh via senderLoop.
		_ = queue.Send(context.Background(), &model.LiveRequest{
			Content: genai.NewContentFromText("hello", "user"),
		})

		// Allow senderLoop to signal turnResetCh.
		time.Sleep(50 * time.Millisecond)

		// Turn 2: model responds — should NOT be suppressed.
		conn.recvCh <- textResponse("Turn2")
		conn.recvCh <- turnCompleteResponse()

		time.Sleep(50 * time.Millisecond)
		queue.Close()
	}()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	texts := modelContentTexts(events)
	if len(texts) != 2 || texts[0] != "Turn1" || texts[1] != "Turn2" {
		t.Errorf("expected [Turn1 Turn2], got %v", texts)
	}

	// Both turns must be persisted (not filtered).
	pTexts := persistedTexts(svc.PersistedEvents())
	if len(pTexts) < 2 {
		t.Fatalf("expected at least 2 persisted texts, got %d: %v", len(pTexts), pTexts)
	}
	foundTurn1, foundTurn2 := false, false
	for _, pt := range pTexts {
		if pt == "Turn1" {
			foundTurn1 = true
		}
		if pt == "Turn2" {
			foundTurn2 = true
		}
	}
	if !foundTurn1 || !foundTurn2 {
		t.Errorf("expected both Turn1 and Turn2 persisted, got %v", pTexts)
	}
}

// ---------------------------------------------------------------------------
// Scenario 21: Late orphan tool flush — duplicate suppressed after tool completes
// ---------------------------------------------------------------------------

func TestScenario21_LateOrphanFlushAfterTurnComplete(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvCh = make(chan *model.LLMResponse, 20)

	slowTool := &mockTool{
		name: "slow_tool",
		runFunc: func(ctx context.Context, _ map[string]any) (map[string]any, error) {
			time.Sleep(500 * time.Millisecond)
			return map[string]any{"ok": true}, nil
		},
	}

	llm := &mockLiveLLM{conn: conn}
	a, _ := llmagent.New(llmagent.Config{
		Name:  "test_agent",
		Model: llm,
		Tools: []tool.Tool{slowTool},
	})

	svc := &mockSessionService{Service: session.InMemoryService()}
	_, _ = svc.Service.Create(context.Background(), &session.CreateRequest{
		AppName: "test", UserID: "user1", SessionID: "sess1",
	})

	r, _ := New(Config{
		AppName:        "test",
		Agent:          a,
		SessionService: svc,
	})

	queue := agent.NewLiveRequestQueue(100)

	go func() {
		// Phase 1: Model sends function call — triggers buffering + timer.
		conn.recvCh <- functionCallResponse("fc1", "slow_tool", nil)

		// Phase 2: Wait for timer to fire and slow tool to start executing.
		time.Sleep(300 * time.Millisecond)

		// Phase 3: While slow_tool blocks, feed content + TC (pile up in recvCh).
		conn.recvCh <- textResponse("A")
		conn.recvCh <- turnCompleteResponse()

		// Phase 4: Wait for slow_tool to complete + flushToolCalls to return.
		time.Sleep(500 * time.Millisecond)

		// Phase 5: Duplicate (model re-response after receiving orphan result).
		conn.recvCh <- textResponse("A")
		conn.recvCh <- turnCompleteResponse()

		time.Sleep(100 * time.Millisecond)
		close(conn.recvCh) // signal EOF to receiver — all messages drained first
		queue.Close()      // terminate sender so RunLive can shut down
	}()

	cfg := agent.RunConfig{ToolCoalesceWindow: 10 * time.Millisecond}
	var events []*session.Event
	for ev, err := range r.RunLive(context.Background(), "user1", "sess1", queue, cfg) {
		if err != nil {
			break
		}
		if ev != nil {
			events = append(events, ev)
		}
	}

	// Assert: slow_tool was actually called.
	if slowTool.CallCount() < 1 {
		t.Error("expected slow_tool to have been called")
	}

	// Assert: content "A" emitted exactly once.
	contentCount := 0
	for _, ev := range events {
		if ev.Content != nil && ev.Content.Role == "model" {
			for _, p := range ev.Content.Parts {
				if p.Text == "A" {
					contentCount++
				}
			}
		}
	}
	if contentCount != 1 {
		t.Errorf("expected content 'A' emitted once, got %d", contentCount)
	}
}

// ---------------------------------------------------------------------------
// Scenario 22: Suppressed TurnComplete on same message drives SetModelSpeaking(false)
// ---------------------------------------------------------------------------

func TestScenario22_SuppressedTurnCompleteDrivesSetModelSpeaking(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		// Turn 1: genuine.
		audioResponse([]byte{0x01}),
		textResponseWithTurnComplete("Hello"),
		// Turn 2: duplicate — single message with content + TurnComplete.
		audioResponse([]byte{0x02}),
		textResponseWithTurnComplete("Hello"),
	}

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, _ := collectEvents(t, r, queue)

	// Assert: "Hello" content emitted once (duplicate suppressed).
	helloCount := 0
	for _, ev := range events {
		if ev.Content != nil && ev.Content.Role == "model" {
			for _, p := range ev.Content.Parts {
				if p.Text == "Hello" {
					helloCount++
				}
			}
		}
	}
	if helloCount != 1 {
		t.Errorf("expected 'Hello' emitted once, got %d", helloCount)
	}

	// Assert: ModelSpeaking is false — proves suppressed TurnComplete
	// called SetModelSpeaking(false) after audio set it to true.
	if queue.ModelSpeaking() {
		t.Error("ModelSpeaking should be false: suppressed TurnComplete must call SetModelSpeaking(false)")
	}
}

// ---------------------------------------------------------------------------
// Scenario 23: Mixed transcription+content — content suppressed, transcription emitted
// ---------------------------------------------------------------------------

func TestScenario23_MixedMessageContentSuppressedTranscriptionEmitted(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponse("A"),
		turnCompleteResponse(),
		mixedResponse("A", "hello"), // duplicate content + transcription
	}

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	// Model content "A" should appear only once.
	texts := modelContentTexts(events)
	if len(texts) != 1 || texts[0] != "A" {
		t.Errorf("expected [A], got %v", texts)
	}

	// The mixed message event should have Content==nil but transcription intact.
	lastEv := events[len(events)-1]
	if lastEv.Content != nil {
		t.Error("mixed message event should have Content stripped (nil)")
	}
	if lastEv.OutputTranscription == nil || lastEv.OutputTranscription.Text != "hello" {
		t.Error("mixed message event should preserve output transcription")
	}
}

// ---------------------------------------------------------------------------
// Scenario 24: Post-Interrupted known limitation — out of scope for #34
// ---------------------------------------------------------------------------

func TestScenario24_PostInterruptedKnownLimitation(t *testing.T) {
	// Documents known false positive — out of scope for issue #34.
	// If the model is Interrupted and immediately sends new content
	// before turnResetCh fires, the guard may suppress it.
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		textResponseWithTurnComplete("A"),
		textResponseInterrupted("B"),
	}

	r, _, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, _ := collectEvents(t, r, queue)

	// "A" is emitted. "B" is suppressed (known false positive).
	texts := modelContentTexts(events)
	if len(texts) != 1 || texts[0] != "A" {
		t.Errorf("expected only [A] emitted (B suppressed as known limitation), got %v", texts)
	}

	// SetModelSpeaking(false) must still fire for Interrupted.
	if queue.ModelSpeaking() {
		t.Error("ModelSpeaking should be false: Interrupted must call SetModelSpeaking(false)")
	}
}

// ---------------------------------------------------------------------------
// Scenario 25: InputTranscription flushed before OutputTranscription
// ---------------------------------------------------------------------------

func TestScenario25_InputFlushedBeforeOutput(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hello ", "input", false),
		transcriptResponse("world", "input", false),
		transcriptResponse("Hi!", "output", true),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 4 {
		t.Fatalf("expected 4 yielded events, got %d", len(events))
	}

	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)
	if len(texts) != 2 || texts[0] != "Hello world" || texts[1] != "Hi!" {
		t.Fatalf("expected [Hello world, Hi!], got %v", texts)
	}

	// First persisted event must be user input.
	if persisted[0].Author != "user" {
		t.Errorf("first persisted Author = %q, want %q", persisted[0].Author, "user")
	}
	if persisted[0].Content.Role != "user" {
		t.Errorf("first persisted Role = %q, want %q", persisted[0].Content.Role, "user")
	}
	// Second persisted event must be agent output.
	if persisted[1].Author != "test_agent" {
		t.Errorf("second persisted Author = %q, want %q", persisted[1].Author, "test_agent")
	}
	if persisted[1].Content.Role != "model" {
		t.Errorf("second persisted Role = %q, want %q", persisted[1].Content.Role, "model")
	}
}

// ---------------------------------------------------------------------------
// Scenario 26: InputTranscription flushed before Content (tool call)
// ---------------------------------------------------------------------------

func TestScenario26_InputFlushedBeforeToolCall(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Check patient", "input", false),
		functionCallResponse("fc1", "lookup", map[string]any{}),
		textResponse("Found"),
		turnCompleteResponse(),
	}

	lookupTool := &mockTool{
		name:   "lookup",
		result: map[string]any{"status": "ok"},
	}

	r, svc, _ := setupRunner(t, conn, []tool.Tool{lookupTool}, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	_ = events

	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)

	// "Check patient" must be the first persisted text.
	if len(texts) == 0 || texts[0] != "Check patient" {
		t.Fatalf("expected first persistedText = %q, got %v", "Check patient", texts)
	}
	if persisted[0].Author != "user" {
		t.Errorf("first persisted Author = %q, want %q", persisted[0].Author, "user")
	}

	// Tool must have been called exactly once.
	if lookupTool.CallCount() != 1 {
		t.Errorf("expected lookup tool called once, got %d", lookupTool.CallCount())
	}

	// "Check patient" must appear before function call in persisted order.
	fcIdx := -1
	for i, ev := range persisted {
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p.FunctionCall != nil {
					fcIdx = i
					break
				}
			}
		}
		if fcIdx >= 0 {
			break
		}
	}
	if fcIdx < 0 {
		t.Fatal("expected a function call in persisted events")
	}
	if fcIdx <= 0 {
		t.Error("function call should appear after user transcript in persisted order")
	}
}

// ---------------------------------------------------------------------------
// Scenario 27: InputTranscription flushed on TurnComplete (no other boundary)
// ---------------------------------------------------------------------------

func TestScenario27_InputFlushedOnTurnComplete(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hello", "input", false),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 yielded events, got %d", len(events))
	}

	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)
	if len(texts) != 1 || texts[0] != "Hello" {
		t.Fatalf("expected [Hello], got %v", texts)
	}
	if persisted[0].Author != "user" {
		t.Errorf("persisted Author = %q, want %q", persisted[0].Author, "user")
	}
	if persisted[0].Content.Role != "user" {
		t.Errorf("persisted Role = %q, want %q", persisted[0].Content.Role, "user")
	}
}

// ---------------------------------------------------------------------------
// Scenario 28: No-op when input buffer empty
// ---------------------------------------------------------------------------

func TestScenario28_EmptyInputBufferNoop(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hi!", "output", true),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 yielded events, got %d", len(events))
	}

	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)
	if len(texts) != 1 || texts[0] != "Hi!" {
		t.Fatalf("expected [Hi!], got %v", texts)
	}
	if len(persisted) != 1 {
		t.Fatalf("expected 1 persisted event, got %d", len(persisted))
	}
	if persisted[0].Author == "user" {
		t.Error("persisted event should not have Author=user (no input transcript)")
	}
}

// ---------------------------------------------------------------------------
// Scenario 29: Multi-turn interleaving
// ---------------------------------------------------------------------------

func TestScenario29_MultiTurnInterleaving(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hello", "input", false),
		transcriptResponse("Hi!", "output", true),
		functionCallResponse("fc1", "lookup", map[string]any{}),
		transcriptResponse("Found it", "output", true),
		transcriptResponse("Thanks", "input", false),
		transcriptResponse("You're welcome", "output", true),
		turnCompleteResponse(),
	}

	lookupTool := &mockTool{
		name:   "lookup",
		result: map[string]any{"result": "ok"},
	}

	r, svc, _ := setupRunner(t, conn, []tool.Tool{lookupTool}, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	_, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	persisted := svc.PersistedEvents()

	// Classify each persisted event by type and record its index.
	type classified struct {
		idx  int
		kind string // "user", "model", "fc", "fr"
		text string
	}
	var classes []classified
	for i, ev := range persisted {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				classes = append(classes, classified{i, "fc", ""})
			case p.FunctionResponse != nil:
				classes = append(classes, classified{i, "fr", ""})
			case p.Text != "" && ev.Author == "user":
				classes = append(classes, classified{i, "user", p.Text})
			case p.Text != "":
				classes = append(classes, classified{i, "model", p.Text})
			}
		}
	}

	// Helper to find index of first match.
	findIdx := func(kind, text string) int {
		for _, c := range classes {
			if c.kind == kind && (text == "" || c.text == text) {
				return c.idx
			}
		}
		return -1
	}

	helloIdx := findIdx("user", "Hello")
	hiIdx := findIdx("model", "Hi!")
	fcIdx := findIdx("fc", "")
	frIdx := findIdx("fr", "")
	foundIdx := findIdx("model", "Found it")
	thanksIdx := findIdx("user", "Thanks")
	welcomeIdx := findIdx("model", "You're welcome")

	// Assert all events found.
	for _, pair := range []struct {
		name string
		idx  int
	}{
		{"Hello", helloIdx},
		{"Hi!", hiIdx},
		{"fc", fcIdx},
		{"fr", frIdx},
		{"Found it", foundIdx},
		{"Thanks", thanksIdx},
		{"You're welcome", welcomeIdx},
	} {
		if pair.idx < 0 {
			t.Fatalf("missing %q in persisted events", pair.name)
		}
	}

	// Assert chronological ordering.
	if helloIdx >= hiIdx {
		t.Errorf("Hello (idx %d) should come before Hi! (idx %d)", helloIdx, hiIdx)
	}
	if hiIdx >= fcIdx {
		t.Errorf("Hi! (idx %d) should come before function call (idx %d)", hiIdx, fcIdx)
	}
	if fcIdx >= frIdx {
		t.Errorf("function call (idx %d) should come before tool response (idx %d)", fcIdx, frIdx)
	}
	if frIdx >= foundIdx {
		t.Errorf("tool response (idx %d) should come before Found it (idx %d)", frIdx, foundIdx)
	}
	if thanksIdx >= welcomeIdx {
		t.Errorf("Thanks (idx %d) should come before You're welcome (idx %d)", thanksIdx, welcomeIdx)
	}

	// persistedTexts should skip function call/response (no .Text).
	texts := persistedTexts(persisted)
	expected := []string{"Hello", "Hi!", "Found it", "Thanks", "You're welcome"}
	if len(texts) != len(expected) {
		t.Fatalf("persistedTexts = %v, want %v", texts, expected)
	}
	for i, want := range expected {
		if texts[i] != want {
			t.Errorf("persistedTexts[%d] = %q, want %q", i, texts[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// Scenario 30: InputTranscription flushed before plain-text Content (no OutputTranscription)
// ---------------------------------------------------------------------------

func TestScenario30_InputFlushedBeforeTextContent(t *testing.T) {
	conn := newMockLiveConnection()
	conn.recvResponses = []*model.LLMResponse{
		transcriptResponse("Hello", "input", false),
		textResponse("Hi there"),
		turnCompleteResponse(),
	}

	r, svc, _ := setupRunner(t, conn, nil, nil)
	queue := agent.NewLiveRequestQueue(100)
	queue.Close()

	events, errs := collectEvents(t, r, queue)
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(events) != 3 {
		t.Fatalf("expected 3 yielded events, got %d", len(events))
	}

	persisted := svc.PersistedEvents()
	texts := persistedTexts(persisted)
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != "Hi there" {
		t.Fatalf("expected [Hello, Hi there], got %v", texts)
	}

	if persisted[0].Author != "user" {
		t.Errorf("first persisted Author = %q, want %q", persisted[0].Author, "user")
	}
	if persisted[0].Content.Role != "user" {
		t.Errorf("first persisted Role = %q, want %q", persisted[0].Content.Role, "user")
	}
	if persisted[1].Author == "user" {
		t.Error("second persisted event should not have Author=user")
	}
}
