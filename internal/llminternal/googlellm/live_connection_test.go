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
