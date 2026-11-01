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

package controllers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/agentengine/controllers/method"
	"google.golang.org/adk/v2/server/agentengine/internal/models"
	"google.golang.org/adk/v2/session"
)

func BenchmarkFromSession(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			_, sess := seedBenchAESession(b, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sd := models.FromSession(sess)
				if len(sd.Events) != n {
					b.Fatalf("len(Events) = %d, want %d", len(sd.Events), n)
				}
			}
		})
	}
}

func BenchmarkQueryGetSession(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc, _ := seedBenchAESession(b, n)
			ctrl, err := NewAgentEngineAPIController(svc, -1, 1<<20, []method.MethodHandler{
				method.NewGetSessionHandler(svc, "app", "async_get_session", "async"),
			})
			if err != nil {
				b.Fatal(err)
			}
			body := []byte(`{"class_method":"async_get_session","input":{"user_id":"user","session_id":"sess"}}`)
			rw := &discardResponseWriter{header: make(http.Header)}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				req := &http.Request{
					Method: http.MethodPost,
					Body:   io.NopCloser(bytes.NewReader(body)),
				}
				rw.code = http.StatusOK
				ctrl.Query(rw, req)
				if rw.code != http.StatusOK {
					b.Fatalf("status = %d", rw.code)
				}
			}
		})
	}
}

func BenchmarkQueryDecode(b *testing.B) {
	b.Run("StreamQuery/content=4KB", func(b *testing.B) {
		h := &decodeOnlyHandler{
			name: "async_stream_query",
			decode: func(payload []byte) error {
				var req models.StreamQueryRequest
				return json.Unmarshal(payload, &req)
			},
		}
		ctrl, err := NewAgentEngineAPIController(nil, -1, 1<<20, []method.MethodHandler{h})
		if err != nil {
			b.Fatal(err)
		}
		msg := genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: strings.Repeat("Prompt text segment for stream_query benchmark. ", 80)},
			},
		}
		body, err := json.Marshal(models.StreamQueryRequest{
			ClassMethod: "async_stream_query",
			Input: models.StreamQueryInput{
				UserID:    "user-123",
				SessionID: "sess-456",
				Message:   msg,
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		rw := &discardResponseWriter{header: make(http.Header)}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := &http.Request{
				Method: http.MethodPost,
				Body:   io.NopCloser(bytes.NewReader(body)),
			}
			rw.code = http.StatusOK
			ctrl.Query(rw, req)
			if rw.code != http.StatusOK {
				b.Fatalf("status = %d", rw.code)
			}
		}
	})

	b.Run("StreamingAgentRunWithEvents/requestJSON=4KB", func(b *testing.B) {
		h := &decodeOnlyHandler{
			name: "streaming_agent_run_with_events",
			decode: func(payload []byte) error {
				var req models.StreamingAgentRunWithEventsRequest
				if err := json.Unmarshal(payload, &req); err != nil {
					return err
				}
				var runReq models.StreamingAgentRunWithEventsRunRequest
				return json.Unmarshal([]byte(req.Input.RequestJSON), &runReq)
			},
		}
		ctrl, err := NewAgentEngineAPIController(nil, -1, 1<<20, []method.MethodHandler{h})
		if err != nil {
			b.Fatal(err)
		}
		innerBytes, err := json.Marshal(models.StreamingAgentRunWithEventsRunRequest{
			UserID:    "user-123",
			SessionID: "projects/111111/locations/global/collections/default_collection/engines/test-engine/sessions/999999",
			Message: genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: strings.Repeat("AgentSpace request_json message payload segment. ", 80)},
				},
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		body, err := json.Marshal(models.StreamingAgentRunWithEventsRequest{
			ClassMethod: "streaming_agent_run_with_events",
			Input: models.StreamingAgentRunWithEventsInput{
				RequestJSON: string(innerBytes),
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		rw := &discardResponseWriter{header: make(http.Header)}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := &http.Request{
				Method: http.MethodPost,
				Body:   io.NopCloser(bytes.NewReader(body)),
			}
			rw.code = http.StatusOK
			ctrl.Query(rw, req)
			if rw.code != http.StatusOK {
				b.Fatalf("status = %d", rw.code)
			}
		}
	})
}

type decodeOnlyHandler struct {
	name   string
	decode func([]byte) error
}

func (h *decodeOnlyHandler) Name() string { return h.name }

func (h *decodeOnlyHandler) Handle(_ context.Context, _ http.ResponseWriter, payload []byte) error {
	return h.decode(payload)
}

func (h *decodeOnlyHandler) Metadata() (*structpb.Struct, error) { return &structpb.Struct{}, nil }

type discardResponseWriter struct {
	header http.Header
	code   int
}

func (w *discardResponseWriter) Header() http.Header         { return w.header }
func (w *discardResponseWriter) Write(p []byte) (int, error) { return len(p), nil }
func (w *discardResponseWriter) WriteHeader(statusCode int)  { w.code = statusCode }
func (w *discardResponseWriter) SetWriteDeadline(time.Time) error {
	return nil
}

var aeBenchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

func seedBenchAESession(b *testing.B, n int) (session.Service, session.Session) {
	b.Helper()
	ctx := b.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sess",
		State:     map[string]any{"k1": "v1", "k2": 42},
	})
	if err != nil {
		b.Fatal(err)
	}
	start := time.Unix(1_700_000_000, 0)
	for i := 0; i < n; i++ {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%06d", i),
			InvocationID: fmt.Sprintf("inv-%06d", i/2),
			Timestamp:    start.Add(time.Duration(i) * time.Second),
		}
		switch {
		case i%2 == 0:
			ev.Author = "user"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(aeBenchText, genai.RoleUser)}
		case i%8 == 1:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}}
		case i%8 == 3:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": aeBenchText}},
			}}}}
		default:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(aeBenchText, genai.RoleModel)}
		}
		if i%10 == 9 {
			ev.Actions.StateDelta = map[string]any{"turn": i}
		}
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			b.Fatal(err)
		}
	}
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
	if err != nil {
		b.Fatal(err)
	}
	return svc, resp.Session
}
