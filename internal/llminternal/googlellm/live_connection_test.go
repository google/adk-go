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

package googlellm_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/llminternal/googlellm"
	"google.golang.org/adk/model"
)

// connectFakeLiveServing returns a LiveConnection on a fake Live API server
// that sends messages, in order, right after the setup exchange. The
// setupComplete response has already been received.
func connectFakeLiveServing(t *testing.T, messages ...string) *googlellm.LiveConnection {
	t.Helper()
	var upgrader websocket.Upgrader
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			return
		}
		for _, m := range messages {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		// Keep the socket open until the client goes away.
		_, _, _ = conn.ReadMessage()
	}))
	t.Cleanup(ts.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	session, err := client.Live.Connect(t.Context(), "gemini-live-2.5-flash-preview", &genai.LiveConnectConfig{})
	if err != nil {
		t.Fatalf("Live.Connect failed: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	conn := googlellm.NewLiveConnection(session, "gemini-live-2.5-flash-preview", genai.BackendGeminiAPI)
	// Connect does not consume setupComplete, so Recv returns it first as an
	// empty response.
	if _, err := conn.Recv(t.Context()); err != nil {
		t.Fatalf("Recv of setupComplete failed: %v", err)
	}
	return conn
}

const liveUsageJSON = `"usageMetadata":{` +
	`"promptTokenCount":174,"cachedContentTokenCount":20,"responseTokenCount":146,` +
	`"toolUsePromptTokenCount":7,"thoughtsTokenCount":31,"totalTokenCount":358,` +
	`"promptTokensDetails":[{"modality":"AUDIO","tokenCount":174}],` +
	`"cacheTokensDetails":[{"modality":"TEXT","tokenCount":20}],` +
	`"responseTokensDetails":[{"modality":"AUDIO","tokenCount":146}],` +
	`"toolUsePromptTokensDetails":[{"modality":"TEXT","tokenCount":7}],` +
	`"trafficType":"ON_DEMAND"}`

// wantLiveUsage is liveUsageJSON as GenerateContent usage: the Live API's
// response* fields become candidates*.
var wantLiveUsage = &genai.GenerateContentResponseUsageMetadata{
	PromptTokenCount:           174,
	CachedContentTokenCount:    20,
	CandidatesTokenCount:       146,
	ToolUsePromptTokenCount:    7,
	ThoughtsTokenCount:         31,
	TotalTokenCount:            358,
	PromptTokensDetails:        []*genai.ModalityTokenCount{{Modality: genai.MediaModalityAudio, TokenCount: 174}},
	CacheTokensDetails:         []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 20}},
	CandidatesTokensDetails:    []*genai.ModalityTokenCount{{Modality: genai.MediaModalityAudio, TokenCount: 146}},
	ToolUsePromptTokensDetails: []*genai.ModalityTokenCount{{Modality: genai.MediaModalityText, TokenCount: 7}},
	TrafficType:                genai.TrafficTypeOnDemand,
}

func TestLiveConnectionRecvCarriesUsageMetadata(t *testing.T) {
	tests := []struct {
		name    string
		message string
		// want is the response Recv returns for message, UsageMetadata aside.
		want *model.LLMResponse
	}{
		{
			name:    "usage only",
			message: `{` + liveUsageJSON + `}`,
			want:    &model.LLMResponse{},
		},
		{
			name:    "usage with turn complete",
			message: `{"serverContent":{"turnComplete":true},` + liveUsageJSON + `}`,
			want:    &model.LLMResponse{TurnComplete: true},
		},
		{
			name:    "usage with session resumption update",
			message: `{"sessionResumptionUpdate":{"newHandle":"h1","resumable":true},` + liveUsageJSON + `}`,
			want:    &model.LLMResponse{SessionResumptionHandle: "h1"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			conn := connectFakeLiveServing(t, tc.message)
			got, err := conn.Recv(t.Context())
			if err != nil {
				t.Fatalf("Recv failed: %v", err)
			}
			want := *tc.want
			want.UsageMetadata = wantLiveUsage
			if diff := cmp.Diff(&want, got); diff != "" {
				t.Errorf("Recv mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// When a turn ends with buffered transcriptions, the usage stays on the
// turn-complete response, which Recv delivers after the transcriptions.
func TestLiveConnectionRecvKeepsUsageOnBufferedTurnComplete(t *testing.T) {
	conn := connectFakeLiveServing(t,
		`{"serverContent":{"outputTranscription":{"text":"hello"}}}`,
		`{"serverContent":{"turnComplete":true},`+liveUsageJSON+`}`,
	)
	var got []*model.LLMResponse
	for range 3 {
		resp, err := conn.Recv(t.Context())
		if err != nil {
			t.Fatalf("Recv failed: %v", err)
		}
		got = append(got, resp)
	}
	want := []*model.LLMResponse{
		{OutputTranscription: &genai.Transcription{Text: "hello"}, Partial: true},
		{OutputTranscription: &genai.Transcription{Text: "hello", Finished: true}},
		{TurnComplete: true, UsageMetadata: wantLiveUsage},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Recv sequence mismatch (-want +got):\n%s", diff)
	}
}
