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
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
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
		{"2.5 image", gemini, "gemini-2.5-flash-native-audio-latest", image, `{"realtimeInput":{"mediaChunks":[{"data":"Ag==","mimeType":"image/jpeg"}]}}`},
		{"2.5 text", gemini, "gemini-2.5-flash-native-audio-latest", text, `{"clientContent":{"turns":[{"parts":[{"text":"hi"}],"role":"user"}],"turnComplete":true}}`},
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
			// turnComplete has to follow the last surviving turn, not the last
			// turn of the history as it arrived: here the history ends in a
			// model turn that is filtered away, leaving a user turn.
			name: "TurnCompleteFollowsLastSurvivingTurn",
			history: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "hello"}},
				},
				{
					Role: "model",
					Parts: []*genai.Part{
						{FileData: &genai.FileData{FileURI: "artifact://app/user/session/_adk_live/adk_live_audio_storage_output_audio_1.pcm#0", MIMEType: "audio/pcm"}},
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
			conn, frames := connectFakeLive(t, genai.BackendGeminiAPI, "gemini-2.0-flash-live-001")
			if err := conn.SendHistory(t.Context(), tc.history); err != nil {
				t.Fatalf("SendHistory failed: %v", err)
			}
			msg := &genai.LiveClientMessage{}
			if err := json.Unmarshal(testutil.AwaitValue(t, frames, "history frame"), msg); err != nil {
				t.Fatalf("decoding the history frame failed: %v", err)
			}
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
	conn, frames := connectFakeLive(t, genai.BackendGeminiAPI, "gemini-2.0-flash-live-001")
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
	case raw, ok := <-frames:
		if ok {
			t.Errorf("SendHistory sent %s, want no frame", raw)
		}
	case <-time.After(500 * time.Millisecond):
	}
}

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
	conn := googlellm.NewLiveConnection(session, "gemini-live-2.5-flash-preview")
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
