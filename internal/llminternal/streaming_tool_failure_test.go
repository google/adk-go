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

package llminternal

import (
	"iter"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func TestStreamingToolFailureReporting(t *testing.T) {
	type args struct {
		Value string `json:"value"`
	}
	lookup, err := functiontool.NewStreaming(functiontool.Config{Name: "lookup"},
		func(_ agent.Context, a args) iter.Seq2[string, error] {
			return func(func(string, error) bool) { panic("lookup failed for " + a.Value) }
		})
	if err != nil {
		t.Fatal(err)
	}
	tools := map[string]tool.Tool{"lookup": lookup}
	resp := &model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "lookup", Args: map[string]any{"value": "user-secret-123"}},
	}}}}
	newCtx := func(t *testing.T) agent.InvocationContext {
		return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{InvocationID: "inv", Agent: &mockAgent{name: "a"}})
	}
	wantErr := streamingToolInternalError("lookup")

	t.Run("non-live", func(t *testing.T) {
		ev, err := (&Flow{}).handleFunctionCalls(newCtx(t), tools, resp, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		msg, _ := ev.Content.Parts[0].FunctionResponse.Response["error"].(string)
		if msg != wantErr {
			t.Errorf("error = %q, want %q", msg, wantErr)
		}
		if strings.Contains(msg, "user-secret-123") || strings.Contains(msg, "runtime/debug.Stack") {
			t.Errorf("function response sent to the model (%d bytes) contains the panic value or a stack trace", len(msg))
		}
	})

	t.Run("live", func(t *testing.T) {
		sent := make(chan agent.LiveRequest, 1)
		sess := &mockLiveSession{sendFunc: func(r agent.LiveRequest) error { sent <- r; return nil }}
		if _, err := (&Flow{}).handleFunctionCalls(newCtx(t), tools, resp, nil, sess); err != nil {
			t.Fatal(err)
		}
		select {
		case req := <-sent:
			if req.Content == nil || len(req.Content.Parts) == 0 || req.Content.Parts[0].FunctionResponse == nil {
				t.Fatalf("follow-up missing FunctionResponse: %#v", req.Content)
			}
			fr := req.Content.Parts[0].FunctionResponse
			if fr.ID != "call_1" || fr.Name != "lookup" {
				t.Errorf("FunctionResponse id/name = %q/%q, want call_1/lookup", fr.ID, fr.Name)
			}
			msg, _ := fr.Response["error"].(string)
			if msg != wantErr {
				t.Errorf("error = %q, want %q", msg, wantErr)
			}
			if strings.Contains(msg, "user-secret-123") || strings.Contains(msg, "runtime/debug.Stack") {
				t.Errorf("live follow-up contains panic value or stack trace")
			}
		case <-time.After(2 * time.Second):
			t.Error("the model got the pending response but no follow-up after the tool failed")
		}
	})
}
