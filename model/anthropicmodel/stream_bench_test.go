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

package anthropicmodel_test

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
)

// streamingBenchBody returns a Messages API SSE stream with one content block
// of blockType ("text" or "thinking") streamed as n deltas of size bytes.
func streamingBenchBody(blockType string, n, size int) []byte {
	chunk := strings.Repeat("lorem ipsum ", size/12+1)[:size]
	var b bytes.Buffer
	b.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_bench\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5-20250929\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n")
	switch blockType {
	case "text":
		b.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		for range n {
			fmt.Fprintf(&b, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":%q}}\n\n", chunk)
		}
	case "thinking":
		b.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n")
		for range n {
			fmt.Fprintf(&b, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":%q}}\n\n", chunk)
		}
		b.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_bench\"}}\n\n")
	default:
		panic("unknown block type " + blockType)
	}
	b.WriteString("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
	fmt.Fprintf(&b, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":%d}}\n\n", n)
	b.WriteString("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	return b.Bytes()
}

// BenchmarkGenerateContentStreaming measures a streamed GenerateContent call
// whose single text or thinking block arrives as 200 or 2,000 deltas of 50 B,
// served from memory so that only client-side work is measured.
func BenchmarkGenerateContentStreaming(b *testing.B) {
	const deltaSize = 50
	for _, kind := range []string{"text", "thinking"} {
		for _, n := range []int{200, 2000} {
			b.Run(fmt.Sprintf("%s/deltas=%d", kind, n), func(b *testing.B) {
				body := streamingBenchBody(kind, n, deltaSize)
				m, err := anthropicmodel.NewModel(b.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
					APIKey: "test-key",
					HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
							Body:       io.NopCloser(bytes.NewReader(body)),
							Request:    r,
						}, nil
					})},
				})
				if err != nil {
					b.Fatalf("NewModel() error = %v", err)
				}
				req := &model.LLMRequest{Contents: genai.Text("hi")}

				var partials int
				var final *model.LLMResponse
				for resp, err := range m.GenerateContent(b.Context(), req, true) {
					if err != nil {
						b.Fatalf("GenerateContent() error = %v", err)
					}
					if resp.Partial {
						partials++
					} else {
						final = resp
					}
				}
				if partials != n || final == nil || len(final.Content.Parts) != 1 || len(final.Content.Parts[0].Text) != n*deltaSize {
					b.Fatalf("got %d partials and final %+v, want %d partials and one %d-byte part", partials, final, n, n*deltaSize)
				}

				b.ReportAllocs()
				for b.Loop() {
					for _, err := range m.GenerateContent(b.Context(), req, true) {
						if err != nil {
							b.Fatalf("GenerateContent() error = %v", err)
						}
					}
				}
			})
		}
	}
}
