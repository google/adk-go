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

package googlellm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

// startFakeLiveServer runs a Gemini Live wire-protocol fake: it upgrades the
// websocket, consumes the client's setup frame, replies {"setupComplete":{}},
// then forwards every later client frame on the returned channel.
func startFakeLiveServer(t *testing.T) (*genai.Client, <-chan []byte) {
	t.Helper()
	var upgrader websocket.Upgrader
	frames := make(chan []byte, 4)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("websocket upgrade failed: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			t.Errorf("reading setup frame failed: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			t.Errorf("writing setupComplete failed: %v", err)
			return
		}
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			frames <- msg
		}
	}))
	t.Cleanup(ts.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-api-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    strings.Replace(ts.URL, "http", "ws", 1),
			APIVersion: "v1alpha",
		},
	})
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	return client, frames
}

// newTestLiveConnection connects a LiveConnection to a fake live server and
// returns it alongside the channel of frames the server received.
func newTestLiveConnection(t *testing.T, modelName string) (*LiveConnection, <-chan []byte) {
	t.Helper()
	client, frames := startFakeLiveServer(t)
	sdkSession, err := client.Live.Connect(t.Context(), modelName, &genai.LiveConnectConfig{})
	if err != nil {
		t.Fatalf("Live.Connect failed: %v", err)
	}
	t.Cleanup(func() { _ = sdkSession.Close() })
	return NewLiveConnection(sdkSession, modelName, genai.BackendGeminiAPI), frames
}

// nextFrame returns the next frame the fake server received, unmarshaled, and
// fails the test rather than hanging if none arrives.
func nextFrame(t *testing.T, frames <-chan []byte) *genai.LiveClientMessage {
	t.Helper()
	select {
	case raw := <-frames:
		msg := &genai.LiveClientMessage{}
		if err := json.Unmarshal(raw, msg); err != nil {
			t.Fatalf("unmarshaling client frame %s failed: %v", raw, err)
		}
		return msg
	case <-time.After(10 * time.Second):
		t.Fatal("server received no frame")
		return nil
	}
}

// TestLiveConnectionSendHistoryFiltersInternalParts checks that SendHistory
// strips the parts the live API must not be primed with - internal _adk_live
// artifact references (written by the audio cache manager) and cached audio
// blobs - while leaving text and ordinary file references untouched.
func TestLiveConnectionSendHistoryFiltersInternalParts(t *testing.T) {
	testCases := []struct {
		name string
		// history is what the contents processor hands SendHistory.
		history []*genai.Content
		// want is the turns the live API should be primed with.
		want []*genai.Content
		// wantTurnComplete mirrors whether want ends in a user turn.
		wantTurnComplete bool
	}{
		{
			name: "DropsInternalArtifactRefsAndKeepsRest",
			history: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "what is in this file?"},
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/report.pdf#1", MIMEType: "application/pdf"}},
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/_adk_live/audio_1.pcm#1", MIMEType: "audio/pcm"}},
					},
				},
				{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "a quarterly report"},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "what is in this file?"},
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/report.pdf#1", MIMEType: "application/pdf"}},
					},
				},
				{
					Role: "model",
					Parts: []*genai.Part{
						{Text: "a quarterly report"},
					},
				},
			},
			wantTurnComplete: false,
		},
		{
			name: "DropsContentLeftEmptyByFiltering",
			history: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/_adk_live/audio_1.pcm#1", MIMEType: "audio/pcm"}},
					},
				},
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "hello"},
					},
				},
			},
			want: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "hello"}},
				},
			},
			wantTurnComplete: true,
		},
		{
			name: "DropsInlineAudioButKeepsNonLiveArtifactAudio",
			history: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{0x01, 0x02}}},
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/uploaded.wav#1", MIMEType: "audio/wav"}},
						{Text: "transcribe this"},
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/uploaded.wav#1", MIMEType: "audio/wav"}},
						{Text: "transcribe this"},
					},
				},
			},
			wantTurnComplete: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			conn, frames := newTestLiveConnection(t, "gemini-2.0-flash-live-001")
			if err := conn.SendHistory(t.Context(), tc.history); err != nil {
				t.Fatalf("SendHistory failed: %v", err)
			}
			msg := nextFrame(t, frames)
			if msg.ClientContent == nil {
				t.Fatalf("SendHistory sent %+v, want a clientContent message", msg)
			}
			if diff := cmp.Diff(tc.want, msg.ClientContent.Turns); diff != "" {
				t.Errorf("SendHistory sent unexpected turns (-want +got):\n%s", diff)
			}
			if got := msg.ClientContent.TurnComplete; got != tc.wantTurnComplete {
				t.Errorf("SendHistory sent turnComplete=%t, want %t", got, tc.wantTurnComplete)
			}
		})
	}
}

// TestLiveConnectionSendHistoryAllFiltered checks that a history made up
// entirely of internal parts sends nothing at all, rather than an empty turn.
func TestLiveConnectionSendHistoryAllFiltered(t *testing.T) {
	conn, frames := newTestLiveConnection(t, "gemini-2.0-flash-live-001")
	history := []*genai.Content{
		nil,
		{
			Role: "user",
			Parts: []*genai.Part{
				nil,
				{FileData: &genai.FileData{FileURI: "artifact://app/user/session/_adk_live/audio_1.pcm#1", MIMEType: "audio/pcm"}},
				{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{0x01}}},
			},
		},
	}
	if err := conn.SendHistory(t.Context(), history); err != nil {
		t.Fatalf("SendHistory failed: %v", err)
	}
	select {
	case raw := <-frames:
		t.Errorf("SendHistory sent %s, want no frame", raw)
	case <-time.After(500 * time.Millisecond):
	}
}
