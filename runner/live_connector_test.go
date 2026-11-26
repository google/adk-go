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

package runner

import (
	"context"
	"errors"
	"io"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type liveConnectorModel struct {
	noopModel
	connect func(context.Context, *model.LLMRequest) (model.LiveConnection, error)
}

func (m liveConnectorModel) Connect(ctx context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
	return m.connect(ctx, req)
}

type liveConnectorInput struct {
	value    any
	realtime bool
}

type scriptedLiveConnection struct {
	history [][]*genai.Content
	inputs  chan liveConnectorInput
	closed  chan struct{}
	receive func(context.Context, func(*model.LLMResponse, error) bool)
}

func (c *scriptedLiveConnection) SendHistory(_ context.Context, history []*genai.Content) error {
	c.history = append(c.history, history)
	return nil
}

func (c *scriptedLiveConnection) SendContent(ctx context.Context, content *genai.Content) error {
	return c.send(ctx, liveConnectorInput{value: content})
}

func (c *scriptedLiveConnection) SendRealtime(ctx context.Context, input any) error {
	return c.send(ctx, liveConnectorInput{value: input, realtime: true})
}

func (c *scriptedLiveConnection) send(ctx context.Context, input liveConnectorInput) error {
	select {
	case c.inputs <- input:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-c.closed:
		return io.ErrClosedPipe
	}
}

func (c *scriptedLiveConnection) Receive(ctx context.Context) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) { c.receive(ctx, yield) }
}

func (c *scriptedLiveConnection) Close() error {
	close(c.closed)
	return nil
}

func TestRunner_RunLive_LiveConnector(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	const appName, userID, sessionID = "testApp", "testUser", "testSession"
	const toolName, callID = "lookup", "call-1"
	const handle = "resume-1"
	history := genai.NewContentFromText("earlier turn", genai.RoleUser)
	userTurn := genai.NewContentFromText("next turn", genai.RoleUser)
	reply := genai.NewContentFromText("finished", genai.RoleModel)
	realtime := &genai.Blob{MIMEType: "audio/pcm", Data: []byte{1, 2}}
	result := map[string]any{"status": "ready"}
	lookup, err := functiontool.New(functiontool.Config{Name: toolName}, func(agent.Context, struct{}) (map[string]any, error) {
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	first := &scriptedLiveConnection{inputs: make(chan liveConnectorInput), closed: make(chan struct{})}
	second := &scriptedLiveConnection{closed: make(chan struct{})}
	first.receive = func(ctx context.Context, yield func(*model.LLMResponse, error) bool) {
		checkInput := func(want any, realtime bool) bool {
			select {
			case got := <-first.inputs:
				if got.realtime != realtime || !cmp.Equal(got.value, want) {
					t.Errorf("live input of type %T does not match expected %T or was sent through the wrong method", got.value, want)
				}
				return true
			case <-ctx.Done():
				return false
			case <-first.closed:
				return false
			}
		}
		if !checkInput(userTurn, false) || !checkInput(realtime, true) {
			return
		}
		call := genai.NewContentFromParts([]*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: callID, Name: toolName,
		}}}, genai.RoleModel)
		if !yield(&model.LLMResponse{Content: call}, nil) {
			return
		}
		toolReply := genai.NewContentFromParts([]*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID: callID, Name: toolName, Response: result,
		}}}, genai.RoleUser)
		if checkInput(toolReply, false) && yield(&model.LLMResponse{SessionResumptionHandle: handle}, nil) {
			yield(nil, io.EOF)
		}
	}
	second.receive = func(_ context.Context, yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: reply, TurnComplete: true}, nil)
	}

	var dials int
	var initialConfig *genai.LiveConnectConfig
	m := liveConnectorModel{connect: func(_ context.Context, req *model.LLMRequest) (model.LiveConnection, error) {
		dials++
		if req.Model != "noop" || req.Config == nil || len(req.Config.Tools) == 0 ||
			req.Config.SystemInstruction == nil || len(req.Config.SystemInstruction.Parts) == 0 ||
			!strings.Contains(req.Config.SystemInstruction.Parts[0].Text, "Answer briefly.") {
			t.Error("Connect did not receive the preprocessed model request")
		}
		cfg := req.LiveConnectConfig
		if cfg == nil || !cmp.Equal(cfg.ResponseModalities, []genai.Modality{genai.ModalityText}) ||
			cfg.SessionResumption == nil {
			return nil, errors.New("Connect did not receive the live run config")
		}
		switch dials {
		case 1:
			initialConfig = cfg
			if cfg.SessionResumption.Handle != "" {
				t.Error("initial connection received a resumption handle")
			}
			return first, nil
		case 2:
			if cfg.SessionResumption.Handle != handle {
				t.Error("reconnect did not receive the latest resumption handle")
			}
			return second, nil
		default:
			return nil, errors.New("unexpected live reconnect")
		}
	}}
	a := must(llmagent.New(llmagent.Config{
		Name: "live_agent", Model: m, Instruction: "Answer briefly.", Tools: []tool.Tool{lookup},
	}))
	service := session.InMemoryService()
	stored, err := service.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatal(err)
	}
	event := session.NewEvent(ctx, "earlier-invocation")
	event.Author = "user"
	event.Content = history
	if err := service.AppendEvent(ctx, stored.Session, event); err != nil {
		t.Fatal(err)
	}
	r, err := New(Config{AppName: appName, Agent: a, SessionService: service})
	if err != nil {
		t.Fatal(err)
	}
	cfg := agent.LiveRunConfig{
		ResponseModalities: []genai.Modality{genai.ModalityText},
		SessionResumption:  &genai.SessionResumptionConfig{},
	}
	live, events, err := r.RunLive(ctx, userID, sessionID, cfg)
	if err != nil {
		t.Fatalf("RunLive failed for a model with Connect: %v", err)
	}
	defer func() { _ = live.Close() }()
	sent := make(chan error, 1)
	go func() {
		sent <- live.Send(agent.LiveRequest{Content: userTurn, RealtimeInput: realtime})
	}()
	gotReply := false
	for ev, err := range events {
		if err != nil {
			t.Fatalf("RunLive stream failed: %v", err)
		}
		gotReply = gotReply || (ev.TurnComplete && cmp.Equal(ev.Content, reply))
	}
	if err := <-sent; err != nil {
		t.Fatalf("Send failed: %v", err)
	}
	if !gotReply {
		t.Error("RunLive did not emit the completed response after reconnect")
	}
	for i, conn := range []*scriptedLiveConnection{first, second} {
		select {
		case <-conn.closed:
		default:
			t.Fatalf("connection %d was not closed at teardown", i+1)
		}
	}
	if !cmp.Equal(first.history, [][]*genai.Content{{history}}) || len(second.history) != 0 {
		t.Error("RunLive should send stored history once and skip it on resume")
	}
	if cfg.SessionResumption.Handle != "" || initialConfig.SessionResumption.Handle != "" {
		t.Error("RunLive modified the caller config or the initial connection config")
	}
}

func TestRunner_RunLive_UnsupportedModel(t *testing.T) {
	a := must(llmagent.New(llmagent.Config{Name: "live_agent", Model: noopModel{}}))
	r, err := New(Config{
		AppName: "testApp", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	live, events, err := r.RunLive(t.Context(), "testUser", "testSession", agent.LiveRunConfig{})
	if err == nil || err.Error() != "model does not support live connection" || live != nil || events != nil {
		t.Error("RunLive did not reject a model without a live connection")
	}
}

type blockedHistoryConnection struct {
	scriptedLiveConnection
	started  chan struct{}
	testDone <-chan struct{}
}

func (c *blockedHistoryConnection) SendHistory(context.Context, []*genai.Content) error {
	close(c.started)
	select {
	case <-c.closed:
	case <-c.testDone:
	}
	return io.ErrClosedPipe
}

func TestRunner_RunLive_CloseDuringConnectorSetup(t *testing.T) {
	for _, stage := range []string{"connect", "history"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			const appName, userID, sessionID = "testApp", "testUser", "testSession"
			started := make(chan struct{})
			released := make(chan struct{})
			conn := &blockedHistoryConnection{
				scriptedLiveConnection: scriptedLiveConnection{closed: released},
				started:                started,
				testDone:               ctx.Done(),
			}
			m := liveConnectorModel{connect: func(ctx context.Context, _ *model.LLMRequest) (model.LiveConnection, error) {
				if stage == "history" {
					return conn, nil
				}
				close(started)
				<-ctx.Done()
				close(released)
				return nil, ctx.Err()
			}}
			a := must(llmagent.New(llmagent.Config{Name: "live_agent", Model: m}))
			service := session.InMemoryService()
			stored, err := service.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
			if err != nil {
				t.Fatal(err)
			}
			event := session.NewEvent(ctx, "earlier-invocation")
			event.Author = "user"
			event.Content = genai.NewContentFromText("earlier turn", genai.RoleUser)
			if err := service.AppendEvent(ctx, stored.Session, event); err != nil {
				t.Fatal(err)
			}
			r, err := New(Config{AppName: appName, Agent: a, SessionService: service})
			if err != nil {
				t.Fatal(err)
			}
			live, events, err := r.RunLive(ctx, userID, sessionID, agent.LiveRunConfig{})
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}
			defer func() { _ = live.Close() }()
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatal("connector setup did not start")
			}
			if err := live.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-released:
				if ctx.Err() != nil {
					t.Fatal("connector setup was released only when the invocation expired")
				}
			case <-ctx.Done():
				t.Fatal("Close did not release connector setup")
			}
			for ev, err := range events {
				t.Errorf("closed session yielded event %t or error %t", ev != nil, err != nil)
			}
		})
	}
}
