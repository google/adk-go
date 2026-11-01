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
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"
)

func TestLiveConnectionRecvTranscription_Equivalence(t *testing.T) {
	frames := []string{
		`{"serverContent":{"inputTranscription":{"text":"hello "}}}`,
		`{"serverContent":{"inputTranscription":{"text":""}}}`,
		`{"serverContent":{"inputTranscription":{"text":"world 🌍"}}}`,
		`{"serverContent":{"outputTranscription":{"text":"answer "}}}`,
		`{"serverContent":{"outputTranscription":{"text":"part 2"}}}`,
		`{"serverContent":{"turnComplete":true}}`,
		`{"serverContent":{"outputTranscription":{"text":"interrupted "}}}`,
		`{"serverContent":{"outputTranscription":{"text":"stream"}}}`,
		`{"serverContent":{"interrupted":true}}`,
	}

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
		for _, f := range frames {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(f)); err != nil {
				return
			}
		}
	}))
	defer ts.Close()

	ctx := t.Context()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("genai.NewClient() error = %v", err)
	}
	sess, err := client.Live.Connect(ctx, "gemini-2.0-flash-exp", &genai.LiveConnectConfig{})
	if err != nil {
		t.Fatalf("client.Live.Connect() error = %v", err)
	}
	lc := NewLiveConnection(sess, "gemini-2.0-flash-exp", genai.BackendGeminiAPI)
	defer func() { _ = lc.Close() }()

	// Drain the buffered setupComplete message from Connect.
	if _, err := lc.Recv(ctx); err != nil {
		t.Fatalf("Recv() setupComplete error = %v", err)
	}

	// 3 input chunks + 2 output chunks -> 5 partials, then aggregated input, aggregated output, then turnComplete.
	for i := range 5 {
		resp, err := lc.Recv(ctx)
		if err != nil || resp == nil || !resp.Partial {
			t.Fatalf("step %d: Recv() = (%v, %v), want partial", i, resp, err)
		}
	}
	inFinal, err := lc.Recv(ctx)
	if err != nil || inFinal == nil || inFinal.InputTranscription == nil || inFinal.InputTranscription.Text != "hello world 🌍" || !inFinal.InputTranscription.Finished {
		t.Fatalf("Recv() input final = (%+v, %v), want finished 'hello world 🌍'", inFinal, err)
	}
	outFinal, err := lc.Recv(ctx)
	if err != nil || outFinal == nil || outFinal.OutputTranscription == nil || outFinal.OutputTranscription.Text != "answer part 2" || !outFinal.OutputTranscription.Finished {
		t.Fatalf("Recv() output final = (%+v, %v), want finished 'answer part 2'", outFinal, err)
	}
	turnDone, err := lc.Recv(ctx)
	if err != nil || turnDone == nil || !turnDone.TurnComplete {
		t.Fatalf("Recv() turnDone = (%+v, %v), want TurnComplete", turnDone, err)
	}

	// Second turn: 2 output chunks then interrupted.
	for i := range 2 {
		resp, err := lc.Recv(ctx)
		if err != nil || resp == nil || !resp.Partial {
			t.Fatalf("turn 2 step %d: Recv() = (%v, %v), want partial", i, resp, err)
		}
	}
	intOutFinal, err := lc.Recv(ctx)
	if err != nil || intOutFinal == nil || intOutFinal.OutputTranscription == nil || intOutFinal.OutputTranscription.Text != "interrupted stream" || !intOutFinal.OutputTranscription.Finished {
		t.Fatalf("Recv() interrupted output final = (%+v, %v), want finished 'interrupted stream'", intOutFinal, err)
	}
	intDone, err := lc.Recv(ctx)
	if err != nil || intDone == nil || !intDone.Interrupted {
		t.Fatalf("Recv() intDone = (%+v, %v), want Interrupted", intDone, err)
	}
}

func BenchmarkLiveConnectionRecvTranscription(b *testing.B) {
	chunkFrame := []byte(`{"serverContent":{"outputTranscription":{"text":"0123456789abcdefghij"}}}`)
	turnCompleteFrame := []byte(`{"serverContent":{"turnComplete":true}}`)

	for _, chunks := range []int{200, 2000} {
		b.Run(fmt.Sprintf("chunks=%d", chunks), func(b *testing.B) {
			turnCh := make(chan struct{})
			defer close(turnCh)

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
				for range turnCh {
					for range chunks {
						if err := conn.WriteMessage(websocket.TextMessage, chunkFrame); err != nil {
							return
						}
					}
					if err := conn.WriteMessage(websocket.TextMessage, turnCompleteFrame); err != nil {
						return
					}
				}
			}))
			defer ts.Close()

			ctx := b.Context()
			client, err := genai.NewClient(ctx, &genai.ClientConfig{
				Backend:     genai.BackendGeminiAPI,
				APIKey:      "test-api-key",
				HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
			})
			if err != nil {
				b.Fatalf("genai.NewClient() error = %v", err)
			}
			sess, err := client.Live.Connect(ctx, "gemini-2.0-flash-exp", &genai.LiveConnectConfig{})
			if err != nil {
				b.Fatalf("client.Live.Connect() error = %v", err)
			}
			lc := NewLiveConnection(sess, "gemini-2.0-flash-exp", genai.BackendGeminiAPI)
			defer func() { _ = lc.Close() }()
			if _, err := lc.Recv(ctx); err != nil {
				b.Fatalf("Recv() setupComplete error = %v", err)
			}

			wantLen := chunks * 20
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				turnCh <- struct{}{}
				for range chunks {
					resp, err := lc.Recv(ctx)
					if err != nil || resp == nil || !resp.Partial {
						b.Fatalf("Recv() partial = (%v, %v)", resp, err)
					}
				}
				finalResp, err := lc.Recv(ctx)
				if err != nil || finalResp == nil || finalResp.OutputTranscription == nil || len(finalResp.OutputTranscription.Text) != wantLen {
					b.Fatalf("Recv() final transcription = (%v, %v), want len %d", finalResp, err, wantLen)
				}
				doneResp, err := lc.Recv(ctx)
				if err != nil || doneResp == nil || !doneResp.TurnComplete {
					b.Fatalf("Recv() turnComplete = (%v, %v)", doneResp, err)
				}
			}
			b.StopTimer()
		})
	}
}
