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

package agentengine

import (
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

const (
	lineA = `{"author":"root","content":{"role":"model","parts":[{"text":"A"}]},"partial":false}` + "\n"
	lineB = `{"author":"root","content":{"role":"model","parts":[{"text":"B"}]}}` + "\n"
	lineC = `{"author":"root","content":{"role":"model","parts":[{"text":"C"}]},"partial":true}` + "\n"
)

func TestDecodeJSONL(t *testing.T) {
	tests := []struct {
		name      string
		chunks    iter.Seq2[[]byte, error]
		wantTexts []string
		wantErr   string // substring; "" means no error
	}{
		{
			name:      "three events in one chunk",
			chunks:    chunksOf(lineA + lineB + lineC),
			wantTexts: []string{"A", "B", "C"},
		},
		{
			name:      "line split across two chunks",
			chunks:    chunksOf(lineA[:20], lineA[20:]+lineB),
			wantTexts: []string{"A", "B"},
		},
		{
			name:      "split exactly at the newline",
			chunks:    chunksOf(lineA, lineB),
			wantTexts: []string{"A", "B"},
		},
		{
			name:      "trailing line without newline",
			chunks:    chunksOf(lineA + strings.TrimSuffix(lineB, "\n")),
			wantTexts: []string{"A", "B"},
		},
		{
			name:      "event without content is kept with nil Content",
			chunks:    chunksOf(`{"author":"root","actions":{"state_delta":{}}}` + "\n" + lineA),
			wantTexts: []string{"<nil>", "A"},
		},
		{
			name:      "blank lines are skipped",
			chunks:    chunksOf("\n" + lineA + "\n\n" + lineB),
			wantTexts: []string{"A", "B"},
		},
		{
			name:      "whitespace-only line is skipped",
			chunks:    chunksOf(lineA + " \t\n" + lineB),
			wantTexts: []string{"A", "B"},
		},
		{
			name:      "error line ends the stream",
			chunks:    chunksOf(lineA + `{"error":"boom"}` + "\n" + lineB),
			wantTexts: []string{"A"},
			wantErr:   "boom",
		},
		{
			name:      "malformed line is an error",
			chunks:    chunksOf(lineA + "{not json\n"),
			wantTexts: []string{"A"},
			wantErr:   "decoding",
		},
		{
			name:      "null line is an error",
			chunks:    chunksOf(lineA + "null\n" + lineB),
			wantTexts: []string{"A"},
			wantErr:   "not a JSON object",
		},
		{
			name:      "lines split into many small chunks",
			chunks:    chunksOf(splitEvery(lineA+lineB+lineC, 7)...),
			wantTexts: []string{"A", "B", "C"},
		},
		{
			name:      "empty stream",
			chunks:    chunksOf(),
			wantTexts: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := collect(t, decodeJSONL(tc.chunks))
			if tc.wantErr == "" && err != nil {
				t.Fatalf("decodeJSONL() error = %v, want nil", err)
			}
			if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Fatalf("decodeJSONL() error = %v, want containing %q", err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.wantTexts, texts(got)); diff != "" {
				t.Errorf("texts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecodeJSONLFields(t *testing.T) {
	got, err := collect(t, decodeJSONL(chunksOf(lineC)))
	if err != nil || len(got) != 1 {
		t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
	}
	ev := got[0]
	if ev.Author != "root" || !ev.Partial {
		t.Errorf("Author=%q Partial=%v, want root/true", ev.Author, ev.Partial)
	}
	wantContent := &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "C"}}}
	if diff := cmp.Diff(wantContent, ev.Content); diff != "" {
		t.Errorf("Content mismatch (-want +got):\n%s", diff)
	}
	if ev.Raw["author"] != "root" {
		t.Errorf("Raw[author] = %v, want root", ev.Raw["author"])
	}
}

func TestDecodeJSONLNullContent(t *testing.T) {
	got, err := collect(t, decodeJSONL(chunksOf(`{"author":"root","content":null}`+"\n")))
	if err != nil || len(got) != 1 {
		t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
	}
	if got[0].Content != nil {
		t.Errorf("Content = %+v, want nil for \"content\": null", got[0].Content)
	}
}

func TestDecodeJSONLErrorFields(t *testing.T) {
	// Both runtimes report a failed model call as an ordinary event with
	// snake_case error fields and no content, not as an error line.
	line := `{"author":"root","error_code":"SAFETY","error_message":"blocked"}` + "\n"
	got, err := collect(t, decodeJSONL(chunksOf(line)))
	if err != nil || len(got) != 1 {
		t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
	}
	if got[0].ErrorCode != "SAFETY" || got[0].ErrorMessage != "blocked" || got[0].Content != nil {
		t.Errorf("Event = %+v, want ErrorCode SAFETY, ErrorMessage blocked, nil Content", got[0])
	}
}

func TestDecodeJSONLRemoteErrorIsErrRemote(t *testing.T) {
	_, err := collect(t, decodeJSONL(chunksOf(`{"error":"model quota"}`+"\n")))
	if !errors.Is(err, ErrRemote) || !strings.Contains(err.Error(), "model quota") {
		t.Fatalf("error = %v, want ErrRemote carrying the server message", err)
	}
}

func TestDecodeJSONLPropagatesChunkError(t *testing.T) {
	boom := errors.New("transport broke")
	chunks := func(yield func([]byte, error) bool) {
		if !yield([]byte(lineA), nil) {
			return
		}
		if !yield(nil, boom) {
			return
		}
		// Reached only if decodeJSONL keeps reading after the error.
		yield([]byte(lineB), nil)
	}
	got, err := collect(t, decodeJSONL(chunks))
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want %v", err, boom)
	}
	if len(got) != 1 {
		t.Errorf("got %d events before the error, want 1", len(got))
	}
}

func TestDecodeJSONLStopsWhenConsumerStops(t *testing.T) {
	yielded := 0
	chunks := func(yield func([]byte, error) bool) {
		for {
			yielded++
			if !yield([]byte(lineA), nil) {
				return
			}
			if yielded > 10 {
				t.Fatal("producer was not stopped")
			}
		}
	}
	for range decodeJSONL(chunks) {
		break
	}
	if yielded != 1 {
		t.Errorf("producer yielded %d chunks after consumer stopped, want 1", yielded)
	}
}

func TestDecodeJSONLPartFieldCasing(t *testing.T) {
	// The adk-go server and Python agents both write snake_case keys, and
	// camelCase input must decode to the same parts. Keys inside args,
	// response and partMetadata are user data that must stay as written.
	tests := []struct {
		name  string
		parts string
	}{
		{
			name: "snake_case from adk-go and Python servers",
			parts: `[{"function_call":{"id":"c1","name":"lookup","args":{"user_id":"U1","nested":{"some_key":1}}}},` +
				`{"function_response":{"name":"lookup","response":{"status_code":200}}},` +
				`{"inline_data":{"mime_type":"image/png","data":"AQID"}},` +
				`{"text":"hi","thought_signature":"AQ==","part_metadata":{"source_file":"x"}}]`,
		},
		{
			name: "camelCase input passes through unchanged",
			parts: `[{"functionCall":{"id":"c1","name":"lookup","args":{"user_id":"U1","nested":{"some_key":1}}}},` +
				`{"functionResponse":{"name":"lookup","response":{"status_code":200}}},` +
				`{"inlineData":{"mimeType":"image/png","data":"AQID"}},` +
				`{"text":"hi","thoughtSignature":"AQ==","partMetadata":{"source_file":"x"}}]`,
		},
	}
	want := []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "lookup", Args: map[string]any{"user_id": "U1", "nested": map[string]any{"some_key": float64(1)}}}},
		{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"status_code": float64(200)}}},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
		{Text: "hi", ThoughtSignature: []byte{1}, PartMetadata: map[string]any{"source_file": "x"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			line := `{"author":"root","content":{"role":"model","parts":` + tc.parts + `}}` + "\n"
			got, err := collect(t, decodeJSONL(chunksOf(line)))
			if err != nil || len(got) != 1 {
				t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
			}
			if diff := cmp.Diff(want, got[0].Content.Parts); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecodeJSONLURLSafeBase64(t *testing.T) {
	// Pydantic writes bytes as padded URL-safe base64 (vertexai 2.6.0 emits
	// "-_8=" for {0xfb, 0xff}); unpadded input is accepted as well.
	tests := []struct {
		name string
		part string
		want *genai.Part
	}{
		{
			name: "URL-safe inline data",
			part: `{"inline_data":{"mime_type":"application/octet-stream","data":"-__-"}}`,
			want: &genai.Part{InlineData: &genai.Blob{MIMEType: "application/octet-stream", Data: []byte{0xfb, 0xff, 0xfe}}},
		},
		{
			name: "padded URL-safe thought signature",
			part: `{"text":"x","thought_signature":"-_8="}`,
			want: &genai.Part{Text: "x", ThoughtSignature: []byte{0xfb, 0xff}},
		},
		{
			name: "unpadded URL-safe thought signature",
			part: `{"text":"x","thought_signature":"-_8"}`,
			want: &genai.Part{Text: "x", ThoughtSignature: []byte{0xfb, 0xff}},
		},
		{
			name: "unpadded without URL-safe characters",
			part: `{"text":"x","thought_signature":"AQ"}`,
			want: &genai.Part{Text: "x", ThoughtSignature: []byte{1}},
		},
		{
			name: "padded standard base64 is unchanged",
			part: `{"text":"x","thought_signature":"+/8="}`,
			want: &genai.Part{Text: "x", ThoughtSignature: []byte{0xfb, 0xff}},
		},
		{
			// Captured from vertexai 2.6.0 AdkApp.async_stream_query for a
			// FunctionResponse carrying a media part.
			name: "media part inside a function response",
			part: `{"function_response":{"parts":[{"inline_data":{"mime_type":"image/png","data":"-__-"}}],"name":"lookup","response":{"status_code":200}}}`,
			want: &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name: "lookup", Response: map[string]any{"status_code": float64(200)},
				Parts: []*genai.FunctionResponsePart{{InlineData: &genai.FunctionResponseBlob{MIMEType: "image/png", Data: []byte{0xfb, 0xff, 0xfe}}}},
			}},
		},
		{
			name: "URL-safe-looking strings inside user data are left alone",
			part: `{"function_response":{"name":"lookup","response":{"inlineData":{"data":"-__-"},"thoughtSignature":"-_8"}}}`,
			want: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"inlineData": map[string]any{"data": "-__-"}, "thoughtSignature": "-_8"}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			line := `{"author":"root","content":{"role":"model","parts":[` + tc.part + `]}}` + "\n"
			got, err := collect(t, decodeJSONL(chunksOf(line)))
			if err != nil || len(got) != 1 {
				t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
			}
			if diff := cmp.Diff([]*genai.Part{tc.want}, got[0].Content.Parts); diff != "" {
				t.Errorf("Parts mismatch (-want +got):\n%s", diff)
			}
			// Raw shares the user-data maps with the decoded content and
			// must still hold the line as written.
			var want map[string]any
			if err := json.Unmarshal([]byte(line), &want); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(want, got[0].Raw); diff != "" {
				t.Errorf("Raw mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDecodeJSONLURLSafeBase64EveryBytesField(t *testing.T) {
	paths := bytesPaths(reflect.TypeOf(genai.Content{}), nil, map[reflect.Type]bool{})
	if len(paths) < 3 {
		t.Fatalf("found %d bytes paths under genai.Content, want at least 3: %v", len(paths), paths)
	}
	for _, path := range paths {
		t.Run(strings.Join(path, "."), func(t *testing.T) {
			content, err := json.Marshal(nest(path, "-__-"))
			if err != nil {
				t.Fatal(err)
			}
			line := `{"author":"root","content":` + string(content) + `}` + "\n"
			got, err := collect(t, decodeJSONL(chunksOf(line)))
			if err != nil || len(got) != 1 {
				t.Fatalf("decodeJSONL() = %d events, %v; want 1, nil", len(got), err)
			}
			back, err := json.Marshal(got[0].Content)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(back), `"+//+"`) {
				t.Errorf("decoded content %s does not carry the bytes fb ff fe", back)
			}
		})
	}
}

// chunksOf yields the given byte slices in order, then stops.
func chunksOf(parts ...string) iter.Seq2[[]byte, error] {
	return func(yield func([]byte, error) bool) {
		for _, p := range parts {
			if !yield([]byte(p), nil) {
				return
			}
		}
	}
}

// collect drains events and returns what arrived before the first error.
func collect(t *testing.T, events iter.Seq2[*Event, error]) ([]*Event, error) {
	t.Helper()
	var got []*Event
	for ev, err := range events {
		if err != nil {
			return got, err
		}
		got = append(got, ev)
	}
	return got, nil
}

// splitEvery cuts s into pieces of at most n bytes.
func splitEvery(s string, n int) []string {
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	return append(out, s)
}

// texts returns each event's first part text, or "<nil>" without content.
func texts(events []*Event) []string {
	var out []string
	for _, ev := range events {
		if ev.Content == nil {
			out = append(out, "<nil>")
			continue
		}
		out = append(out, ev.Content.Parts[0].Text)
	}
	return out
}

// bytesPaths lists every []byte field reachable from genai.Content as a JSON
// path, so TestDecodeJSONLURLSafeBase64EveryBytesField cannot fall behind the
// type.
func bytesPaths(t reflect.Type, prefix []string, seen map[reflect.Type]bool) [][]string {
	for t.Kind() == reflect.Pointer || t.Kind() == reflect.Slice {
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return [][]string{prefix}
		}
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct || seen[t] {
		return nil
	}
	seen[t] = true
	defer delete(seen, t)
	var out [][]string
	for i := range t.NumField() {
		f := t.Field(i)
		tag := strings.Split(f.Tag.Get("json"), ",")[0]
		if tag == "" || tag == "-" {
			continue
		}
		path := append(append([]string(nil), prefix...), tag)
		if f.Type.Kind() == reflect.Slice && f.Type.Elem().Kind() != reflect.Uint8 {
			path = append(path, "[]")
		}
		out = append(out, bytesPaths(f.Type, path, seen)...)
	}
	return out
}

// nest builds the JSON value that places leaf at path.
func nest(path []string, leaf any) any {
	if len(path) == 0 {
		return leaf
	}
	if path[0] == "[]" {
		return []any{nest(path[1:], leaf)}
	}
	return map[string]any{path[0]: nest(path[1:], leaf)}
}
