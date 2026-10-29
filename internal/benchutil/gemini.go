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

package benchutil

import (
	"bytes"
	"io"
	"net/http"
	"sync/atomic"
)

// GeminiTransport answers Gemini API requests with Bodies in turn, without
// opening a connection, so a benchmark through the real Gemini model
// measures the client side of each call: building and encoding the request
// and decoding the response.
type GeminiTransport struct {
	Bodies [][]byte
	calls  atomic.Uint64
}

func (t *GeminiTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		if _, err := io.Copy(io.Discard, req.Body); err != nil {
			return nil, err
		}
		_ = req.Body.Close()
	}
	body := t.Bodies[(t.calls.Add(1)-1)%uint64(len(t.Bodies))]
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(body)),
		Request:    req,
	}, nil
}

// GeminiText is a generateContent response holding a final text answer.
var GeminiText = []byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"It is 22.5°C and sunny in Paris."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1200,"candidatesTokenCount":12,"totalTokenCount":1212}}`)

// GeminiToolTurn is ToolCallModel's turn on the wire: a get_weather call,
// then GeminiText. Sequential turns that each make two calls stay in step.
var GeminiToolTurn = [][]byte{
	[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris","unit":"celsius"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1200,"candidatesTokenCount":8,"totalTokenCount":1208}}`),
	GeminiText,
}
