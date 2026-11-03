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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/testutil"
)

// connectFakeLive returns a LiveConnection for model on a fake Live API server,
// and a channel that yields the first frame the server reads after the setup
// exchange.
func connectFakeLive(t *testing.T, backend genai.Backend, model string) (*googlellm.LiveConnection, <-chan []byte) {
	t.Helper()
	frames := make(chan []byte, 1)
	var upgrader websocket.Upgrader
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(frames)
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
		if _, frame, err := conn.ReadMessage(); err == nil {
			frames <- frame
		}
	}))
	t.Cleanup(ts.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     backend,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	session, err := client.Live.Connect(t.Context(), model, &genai.LiveConnectConfig{})
	if err != nil {
		t.Fatalf("Live.Connect failed: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return googlellm.NewLiveConnection(session, model), frames
}

func TestLiveConnectionRoutesInputByModel(t *testing.T) {
	audio := &genai.Blob{MIMEType: "audio/pcm;rate=16000", Data: []byte{1}}
	image := &genai.Blob{MIMEType: "image/jpeg", Data: []byte{2}}
	text := genai.NewContentFromText("hi", genai.RoleUser)
	twoTexts := genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("a"), genai.NewPartFromText("b")}, genai.RoleUser)
	inlineImage := genai.NewContentFromParts([]*genai.Part{genai.NewPartFromBytes([]byte{2}, "image/jpeg")}, genai.RoleUser)
	gemini, vertex := genai.BackendGeminiAPI, genai.BackendVertexAI
	testCases := []struct {
		name    string
		backend genai.Backend
		model   string
		input   any
		want    string
	}{
		{"3.x audio", gemini, "gemini-3.8-live", audio, `{"realtimeInput":{"audio":{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}}}`},
		{"3.x image", gemini, "gemini-3.8-live", image, `{"realtimeInput":{"video":{"data":"Ag==","mimeType":"image/jpeg"}}}`},
		{"3.x text", gemini, "gemini-3.8-live", text, `{"realtimeInput":{"text":"hi"}}`},
		{"live translate audio", gemini, "gemini-3.5-live-translate-preview", audio, `{"realtimeInput":{"audio":{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}}}`},
		{"live translate text", gemini, "gemini-3.5-live-translate-preview", text, `{"clientContent":{"turns":[{"parts":[{"text":"hi"}],"role":"user"}],"turnComplete":true}}`},
		{"2.5 audio", gemini, "gemini-2.5-flash-native-audio-latest", audio, `{"realtimeInput":{"mediaChunks":[{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}]}}`},
		{"3.x two text parts", gemini, "gemini-3.8-live", twoTexts, `{"clientContent":{"turns":[{"parts":[{"text":"a"},{"text":"b"}],"role":"user"}],"turnComplete":true}}`},
		{"3.x inline image part", gemini, "gemini-3.8-live", inlineImage, `{"clientContent":{"turns":[{"parts":[{"inlineData":{"data":"Ag==","mimeType":"image/jpeg"}}],"role":"user"}],"turnComplete":true}}`},
		{"3.1 audio", gemini, "gemini-3.1-flash-live-preview", audio, `{"realtimeInput":{"audio":{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}}}`},
		{"3.1 text", gemini, "gemini-3.1-flash-live-preview", text, `{"realtimeInput":{"text":"hi"}}`},
		{"3.1 audio on Vertex AI", vertex, "gemini-3.1-flash-live-preview", audio, `{"realtimeInput":{"audio":{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}}}`},
		{"3.1 text on Vertex AI", vertex, "gemini-3.1-flash-live-preview", text, `{"realtimeInput":{"text":"hi"}}`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			conn, frames := connectFakeLive(t, tc.backend, tc.model)
			var err error
			if content, ok := tc.input.(*genai.Content); ok {
				err = conn.SendContent(t.Context(), content)
			} else {
				err = conn.SendRealtime(t.Context(), tc.input)
			}
			if err != nil {
				t.Fatalf("send failed: %v", err)
			}

			var got, want any
			if err := json.Unmarshal(testutil.AwaitValue(t, frames, "input frame"), &got); err != nil {
				t.Fatalf("decoding the input frame failed: %v", err)
			}
			if err := json.Unmarshal([]byte(tc.want), &want); err != nil {
				t.Fatalf("decoding the wanted frame failed: %v", err)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("sent frame mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Gemini 3.x Live takes realtime blobs on the audio and video fields, which
// carry audio/* and image/* only, so a blob of another type is an error rather
// than a send on the media field.
func TestLiveConnectionRejectsOtherMIMETypesOn3xLive(t *testing.T) {
	conn, frames := connectFakeLive(t, genai.BackendGeminiAPI, "gemini-3.8-live")
	if err := conn.SendRealtime(t.Context(), &genai.Blob{MIMEType: "video/mp4", Data: []byte{3}}); err == nil {
		t.Error("SendRealtime() of a video/mp4 blob succeeded, want an error")
	}
	// The audio blob is the first frame the server reads only if the rejected
	// blob sent nothing and left the connection usable.
	if err := conn.SendRealtime(t.Context(), &genai.Blob{MIMEType: "audio/pcm;rate=16000", Data: []byte{1}}); err != nil {
		t.Fatalf("SendRealtime() of an audio blob after the rejected one failed: %v", err)
	}

	var got, want any
	if err := json.Unmarshal(testutil.AwaitValue(t, frames, "input frame"), &got); err != nil {
		t.Fatalf("decoding the input frame failed: %v", err)
	}
	if err := json.Unmarshal([]byte(`{"realtimeInput":{"audio":{"data":"AQ==","mimeType":"audio/pcm;rate=16000"}}}`), &want); err != nil {
		t.Fatalf("decoding the wanted frame failed: %v", err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("first frame after the rejected blob mismatch (-want +got):\n%s", diff)
	}
}
