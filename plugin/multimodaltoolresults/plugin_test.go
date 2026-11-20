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

package multimodaltoolresults_test

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/adkcontext"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/multimodaltoolresults"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// The plugin's private invocation-scoped keys, written as literals so the
// tests pin them.
const (
	currentTurnKey = "temp:multimodal_tool_results_plugin:current_turn_parts"
	updatedKey     = "temp:multimodal_tool_results_plugin:updated_in_invocation"
)

func newPlugin(t *testing.T, r multimodaltoolresults.Retention) *plugin.Plugin {
	t.Helper()
	p, err := multimodaltoolresults.New(multimodaltoolresults.Config{Retention: r})
	if err != nil {
		t.Fatalf("New(%q) failed: %v", r, err)
	}
	return p
}

func newSession(t *testing.T) session.Session {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create session failed: %v", err)
	}
	return resp.Session
}

// newCallbackContext returns a callback context over a fresh session.
func newCallbackContext(t *testing.T) agent.Context {
	t.Helper()
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: newSession(t)})
	return icontext.NewCallbackContextWithDelta(invCtx, nil, nil)
}

func callAfterTool(t *testing.T, p *plugin.Plugin, ctx agent.Context, result map[string]any) {
	t.Helper()
	got, err := p.AfterToolCallback()(ctx, nil, map[string]any{}, result, nil)
	if got != nil || err != nil {
		t.Fatalf("AfterToolCallback() = (%v, %v), want (nil, nil)", got, err)
	}
}

func callBeforeModel(t *testing.T, p *plugin.Plugin, ctx agent.Context, req *model.LLMRequest) {
	t.Helper()
	got, err := p.BeforeModelCallback()(ctx, req)
	if got != nil || err != nil {
		t.Fatalf("BeforeModelCallback() = (%v, %v), want (nil, nil)", got, err)
	}
}

func setState(t *testing.T, ctx agent.Context, key string, v any) {
	t.Helper()
	if err := ctx.State().Set(key, v); err != nil {
		t.Fatalf("State().Set(%q) failed: %v", key, err)
	}
}

func getState(t *testing.T, ctx agent.Context, key string) any {
	t.Helper()
	v, err := ctx.State().Get(key)
	if err != nil {
		t.Fatalf("State().Get(%q) failed: %v", key, err)
	}
	return v
}

func checkAbsent(t *testing.T, ctx agent.Context, key string) {
	t.Helper()
	if v, err := ctx.State().Get(key); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Errorf("State().Get(%q) = (%v, %v), want ErrStateKeyNotExist", key, v, err)
	}
}

// oneEmptyContent returns a request with one user content without parts.
func oneEmptyContent() *model.LLMRequest {
	return &model.LLMRequest{Contents: []*genai.Content{{Role: "user"}}}
}

func tail(req *model.LLMRequest) []*genai.Part {
	return req.Contents[len(req.Contents)-1].Parts
}

// py returns the 17-key map that google-genai 2.34.0's
// Part.model_dump(mode="json") writes: the keys in set hold the given values
// and every other key is nil.
func py(set map[string]any) map[string]any {
	m := map[string]any{
		"media_resolution":      nil,
		"code_execution_result": nil,
		"executable_code":       nil,
		"file_data":             nil,
		"function_call":         nil,
		"function_response":     nil,
		"inline_data":           nil,
		"text":                  nil,
		"thought":               nil,
		"thought_signature":     nil,
		"video_metadata":        nil,
		"tool_call":             nil,
		"tool_response":         nil,
		"part_metadata":         nil,
		"audio_transcription":   nil,
		"media_processing":      nil,
		"speech_metadata":       nil,
	}
	maps.Copy(m, set)
	return m
}

// Fails if New accepts any retention string.
func TestNew_InvalidRetentionFails(t *testing.T) {
	p, err := multimodaltoolresults.New(multimodaltoolresults.Config{Retention: "invalid"})
	if p != nil {
		t.Errorf("New() plugin = %v, want nil", p)
	}
	const want = "retention must be 'next_model_call' or 'session', got invalid"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("New() error = %v, want it to contain %q", err, want)
	}
}

// Fails if New ignores Config.Name or does not apply the default name.
func TestNew_Name(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"", "multimodal_tool_results_plugin"},
		{"mine", "mine"},
	}
	for _, tc := range tests {
		p, err := multimodaltoolresults.New(multimodaltoolresults.Config{Name: tc.name})
		if err != nil {
			t.Fatalf("New(Name: %q) failed: %v", tc.name, err)
		}
		if got := p.Name(); got != tc.want {
			t.Errorf("New(Name: %q).Name() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// Fails if the one-key rule is dropped (row "two keys"), if a map is taken
// for a part (rows "map entry" and "map first"), if empty lists are accepted
// (rows "empty" and "nil slice"), or if a nil first part is accepted.
func TestAfterToolIgnoresResultsThatAreNotParts(t *testing.T) {
	results := []struct {
		name   string
		result map[string]any
	}{
		{"not a result key", map[string]any{"a": 1}},
		{"map entry", map[string]any{"result": []any{map[string]any{"text": "x"}}}},
		{"map first", map[string]any{"result": []any{map[string]any{"a": 1}, &genai.Part{Text: "late"}}}},
		{"empty", map[string]any{"result": []*genai.Part{}}},
		{"nil slice", map[string]any{"result": []*genai.Part(nil)}},
		{"typed nil first", map[string]any{"result": []*genai.Part{nil, {Text: "x"}}}},
		{"string", map[string]any{"result": "s"}},
		{"nil map", nil},
		{"two keys", map[string]any{"result": []*genai.Part{{Text: "x"}}, "status": "ok"}},
	}
	for _, r := range []multimodaltoolresults.Retention{multimodaltoolresults.RetentionNextModelCall, multimodaltoolresults.RetentionSession} {
		for _, tc := range results {
			t.Run(string(r)+"/"+tc.name, func(t *testing.T) {
				p := newPlugin(t, r)
				ctx := newCallbackContext(t)

				callAfterTool(t, p, ctx, tc.result)

				checkAbsent(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)
				checkAbsent(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)
				checkAbsent(t, ctx, currentTurnKey)
				checkAbsent(t, ctx, updatedKey)
			})
		}
	}
}

// adk-python stores a non-part entry after the first part unchanged
// (multimodal_tool_results_plugin.py:130) and never attaches it. Fails if a
// nil *genai.Part after the first entry panics, is dropped from the stored
// list, or reaches the request.
func TestAfterTool_NilPartAfterFirstIsKeptAsNonPart(t *testing.T) {
	tests := []struct {
		retention  multimodaltoolresults.Retention
		key        string
		wantStored []any
	}{
		{multimodaltoolresults.RetentionNextModelCall, multimodaltoolresults.PartsReturnedByToolsKey, []any{}},
		{multimodaltoolresults.RetentionSession, multimodaltoolresults.SessionPartsReturnedByToolsKey, []any{py(map[string]any{"text": "a"}), (*genai.Part)(nil)}},
	}
	for _, tc := range tests {
		t.Run(string(tc.retention), func(t *testing.T) {
			p := newPlugin(t, tc.retention)
			ctx := newCallbackContext(t)
			req := oneEmptyContent()

			callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "a"}, nil}})
			callBeforeModel(t, p, ctx, req)

			if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(req)); diff != "" {
				t.Errorf("request tail mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantStored, getState(t, ctx, tc.key)); diff != "" {
				t.Errorf("state %q after attach mismatch (-want +got):\n%s", tc.key, diff)
			}
		})
	}
}

// An on-tool-error callback can return a response together with an error, and
// the flow passes both to the after-tool callbacks. Fails if the plugin saves
// the parts of a call that failed.
func TestAfterToolIgnoresFailedCall(t *testing.T) {
	for _, r := range []multimodaltoolresults.Retention{multimodaltoolresults.RetentionNextModelCall, multimodaltoolresults.RetentionSession} {
		t.Run(string(r), func(t *testing.T) {
			p := newPlugin(t, r)
			ctx := newCallbackContext(t)

			got, err := p.AfterToolCallback()(ctx, nil, map[string]any{}, map[string]any{"result": &genai.Part{Text: "x"}}, errors.New("tool failed"))

			if got != nil || err != nil {
				t.Errorf("AfterToolCallback() = (%v, %v), want (nil, nil)", got, err)
			}
			checkAbsent(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)
			checkAbsent(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)
			checkAbsent(t, ctx, currentTurnKey)
		})
	}
}

// Fails if a part that cannot be stored is skipped silently, or if the
// failure leaves part of the state written.
func TestAfterTool_SessionSerializeErrorChangesNoState(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionSession)
	ctx := newCallbackContext(t)
	result := map[string]any{"result": []*genai.Part{{Text: "ok"}, {Text: "bad", PartMetadata: map[string]any{"c": make(chan int)}}}}

	got, err := p.AfterToolCallback()(ctx, nil, map[string]any{}, result, nil)

	if got != nil || err == nil || !strings.Contains(err.Error(), "multimodaltoolresults: serialize part") {
		t.Errorf("AfterToolCallback() = (%v, %v), want an error containing %q", got, err, "multimodaltoolresults: serialize part")
	}
	checkAbsent(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)
	checkAbsent(t, ctx, currentTurnKey)
	checkAbsent(t, ctx, updatedKey)
}

// Fails if a single part result, not wrapped in a list, is ignored (row
// "pointer"), or if a part returned by value is not taken for a part (rows
// "value" and "value slice").
func TestAfterToolSavesSinglePartResult(t *testing.T) {
	tests := []struct {
		name   string
		result any
	}{
		{"pointer", &genai.Part{Text: "hi"}},
		{"value", genai.Part{Text: "hi"}},
		{"value slice", []genai.Part{{Text: "hi"}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, "")
			ctx := newCallbackContext(t)

			callAfterTool(t, p, ctx, map[string]any{"result": tc.result})
			saved := getState(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)
			req := oneEmptyContent()
			callBeforeModel(t, p, ctx, req)

			if diff := cmp.Diff([]any{&genai.Part{Text: "hi"}}, saved); diff != "" {
				t.Errorf("saved parts mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]*genai.Part{{Text: "hi"}}, tail(req)); diff != "" {
				t.Errorf("request tail mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails if the after-tool callback returns the result it was given for a
// result that is not parts, as adk-python does: the plugin manager then stops
// at this plugin and the flow skips the agent's callbacks.
func TestAfterTool_NonPartResultRunsLaterCallbacks(t *testing.T) {
	var (
		mu  sync.Mutex
		log []string
	)
	record := func(who string) {
		mu.Lock()
		defer mu.Unlock()
		log = append(log, who)
	}
	mtr := newPlugin(t, "")
	plugin2, err := plugin.New(plugin.Config{
		Name: "plugin2",
		AfterToolCallback: func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
			record("plugin2")
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New failed: %v", err)
	}
	mockTool, err := functiontool.New(functiontool.Config{Name: "mock_tool", Description: "Returns a map."},
		func(agent.Context, struct{}) (map[string]any, error) {
			return map[string]any{"a": 1}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}
	mockModel := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("mock_tool", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: mockModel,
		Tools: []tool.Tool{mockTool},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(agent.Context, tool.Tool, map[string]any, map[string]any, error) (map[string]any, error) {
				record("agent")
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New failed: %v", err)
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create session failed: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: sessions,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{mtr, plugin2}},
	})
	if err != nil {
		t.Fatalf("runner.New failed: %v", err)
	}

	events, err := testutil.CollectEvents(r.Run(t.Context(), "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}))
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	var responses []map[string]any
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part.FunctionResponse != nil {
				responses = append(responses, part.FunctionResponse.Response)
			}
		}
	}
	if diff := cmp.Diff([]map[string]any{{"a": float64(1)}}, responses); diff != "" {
		t.Errorf("function responses mismatch (-want +got):\n%s", diff)
	}
	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{"plugin2", "agent"}, log); diff != "" {
		t.Errorf("after-tool callbacks run mismatch (-want +got):\n%s", diff)
	}
}

// The framework moves media parts into FunctionResponse.Parts. Fails if
// default retention saves the media parts the framework carries (row
// "mixed"), drops a file part without a MIME type, which the framework does
// not carry (row "mixed"), or writes the key when no part is left (row "media
// only").
func TestDefaultRetentionSkipsPartsTheFrameworkCarries(t *testing.T) {
	inline := &genai.Part{InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}}
	file := &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/f", MIMEType: "application/pdf"}}
	noMIME := &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/g"}}

	t.Run("mixed", func(t *testing.T) {
		p := newPlugin(t, multimodaltoolresults.RetentionNextModelCall)
		ctx := newCallbackContext(t)

		callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{inline, {Text: "a"}, file, noMIME}})

		want := []any{&genai.Part{Text: "a"}, &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/g"}}}
		if diff := cmp.Diff(want, getState(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)); diff != "" {
			t.Errorf("saved parts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("media only", func(t *testing.T) {
		p := newPlugin(t, multimodaltoolresults.RetentionNextModelCall)
		ctx := newCallbackContext(t)

		callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{inline, file}})

		checkAbsent(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)
	})
}

// Fails if the saved parts are not reset after the model request that
// attaches them.
func TestDefaultRetentionAttachesPartsOnce(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionNextModelCall)
	ctx := newCallbackContext(t)
	callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "a"}}})

	r1, r2 := oneEmptyContent(), oneEmptyContent()
	callBeforeModel(t, p, ctx, r1)
	callBeforeModel(t, p, ctx, r2)

	if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(r1)); diff != "" {
		t.Errorf("first request tail mismatch (-want +got):\n%s", diff)
	}
	if got := tail(r2); len(got) != 0 {
		t.Errorf("second request tail = %v, want empty", got)
	}
	if diff := cmp.Diff([]any{}, getState(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey)); diff != "" {
		t.Errorf("saved parts after attach mismatch (-want +got):\n%s", diff)
	}
}

// Fails if a saved entry that is not a part reaches the request.
func TestBeforeModel_SkipsNonPartEntries(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionNextModelCall)
	ctx := newCallbackContext(t)
	setState(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey, []any{genai.NewPartFromText("part1"), map[string]any{"a": 1}})
	req := oneEmptyContent()

	callBeforeModel(t, p, ctx, req)

	if diff := cmp.Diff([]*genai.Part{genai.NewPartFromText("part1")}, tail(req)); diff != "" {
		t.Errorf("request tail mismatch (-want +got):\n%s", diff)
	}
}

// Fails if the stored map differs from what adk-python stores: a missing or
// extra key, a key without its nil value, bytes in standard base64, a list of
// nested messages not expanded (row "list of nested messages"), or an empty
// part not stored as Part(text="") is (row "empty text").
func TestAfterTool_SessionStoresModelDumpMaps(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want map[string]any
	}{
		{
			name: "text",
			part: genai.NewPartFromText("part1"),
			want: map[string]any{
				"media_resolution":      nil,
				"code_execution_result": nil,
				"executable_code":       nil,
				"file_data":             nil,
				"function_call":         nil,
				"function_response":     nil,
				"inline_data":           nil,
				"text":                  "part1",
				"thought":               nil,
				"thought_signature":     nil,
				"video_metadata":        nil,
				"tool_call":             nil,
				"tool_response":         nil,
				"part_metadata":         nil,
				"audio_transcription":   nil,
				"media_processing":      nil,
				"speech_metadata":       nil,
			},
		},
		{
			name: "file data",
			part: &genai.Part{FileData: &genai.FileData{FileURI: "gs://bucket/doc.pdf", MIMEType: "application/pdf"}},
			want: map[string]any{
				"media_resolution":      nil,
				"code_execution_result": nil,
				"executable_code":       nil,
				"file_data": map[string]any{
					"display_name": nil,
					"file_uri":     "gs://bucket/doc.pdf",
					"mime_type":    "application/pdf",
				},
				"function_call":       nil,
				"function_response":   nil,
				"inline_data":         nil,
				"text":                nil,
				"thought":             nil,
				"thought_signature":   nil,
				"video_metadata":      nil,
				"tool_call":           nil,
				"tool_response":       nil,
				"part_metadata":       nil,
				"audio_transcription": nil,
				"media_processing":    nil,
				"speech_metadata":     nil,
			},
		},
		{
			name: "thought signature",
			part: &genai.Part{Text: "t", Thought: true, ThoughtSignature: []byte{0xfb, 0xff, 0x00, 'a', 'b', 'c', 'd'}},
			want: map[string]any{
				"media_resolution":      nil,
				"code_execution_result": nil,
				"executable_code":       nil,
				"file_data":             nil,
				"function_call":         nil,
				"function_response":     nil,
				"inline_data":           nil,
				"text":                  "t",
				"thought":               true,
				"thought_signature":     "-_8AYWJjZA==",
				"video_metadata":        nil,
				"tool_call":             nil,
				"tool_response":         nil,
				"part_metadata":         nil,
				"audio_transcription":   nil,
				"media_processing":      nil,
				"speech_metadata":       nil,
			},
		},
		{
			name: "list of nested messages",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name:  "f",
				Parts: []*genai.FunctionResponsePart{{FileData: &genai.FunctionResponseFileData{FileURI: "gs://b/f", MIMEType: "image/png"}}},
			}},
			want: map[string]any{
				"media_resolution":      nil,
				"code_execution_result": nil,
				"executable_code":       nil,
				"file_data":             nil,
				"function_call":         nil,
				"function_response": map[string]any{
					"id":   nil,
					"name": "f",
					"parts": []any{
						map[string]any{
							"file_data": map[string]any{
								"display_name": nil,
								"file_uri":     "gs://b/f",
								"mime_type":    "image/png",
							},
							"inline_data":      nil,
							"media_processing": nil,
						},
					},
					"response":      nil,
					"scheduling":    nil,
					"will_continue": nil,
				},
				"inline_data":         nil,
				"text":                nil,
				"thought":             nil,
				"thought_signature":   nil,
				"video_metadata":      nil,
				"tool_call":           nil,
				"tool_response":       nil,
				"part_metadata":       nil,
				"audio_transcription": nil,
				"media_processing":    nil,
				"speech_metadata":     nil,
			},
		},
		{
			name: "empty text",
			part: genai.NewPartFromText(""),
			want: map[string]any{
				"media_resolution":      nil,
				"code_execution_result": nil,
				"executable_code":       nil,
				"file_data":             nil,
				"function_call":         nil,
				"function_response":     nil,
				"inline_data":           nil,
				"text":                  "",
				"thought":               nil,
				"thought_signature":     nil,
				"video_metadata":        nil,
				"tool_call":             nil,
				"tool_response":         nil,
				"part_metadata":         nil,
				"audio_transcription":   nil,
				"media_processing":      nil,
				"speech_metadata":       nil,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, multimodaltoolresults.RetentionSession)
			ctx := newCallbackContext(t)

			callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{tc.part}})

			got := getState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)
			if diff := cmp.Diff([]any{tc.want}, got); diff != "" {
				t.Errorf("stored parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails if the before-model callback clears the flag that marks the first
// save of an invocation: the second save then replaces the first.
func TestSessionRetentionKeepsUpdatedFlagAcrossModelCall(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionSession)
	ctx := newCallbackContext(t)

	callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "part1"}}})
	callBeforeModel(t, p, ctx, oneEmptyContent())
	callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "part2"}}})

	want := []any{py(map[string]any{"text": "part1"}), py(map[string]any{"text": "part2"})}
	if diff := cmp.Diff(want, getState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)); diff != "" {
		t.Errorf("stored parts mismatch (-want +got):\n%s", diff)
	}
}

// Fails if a request without contents clears the current-turn parts in
// session retention.
func TestSessionRetentionEmptyContentsLeavesPartsPending(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionSession)
	ctx := newCallbackContext(t)
	callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "a"}}})

	empty := &model.LLMRequest{}
	callBeforeModel(t, p, ctx, empty)
	pending := getState(t, ctx, currentTurnKey)
	req := oneEmptyContent()
	callBeforeModel(t, p, ctx, req)

	if len(empty.Contents) != 0 {
		t.Errorf("empty request contents = %v, want none", empty.Contents)
	}
	if diff := cmp.Diff([]any{&genai.Part{Text: "a"}}, pending); diff != "" {
		t.Errorf("current-turn parts after empty request mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(req)); diff != "" {
		t.Errorf("request tail mismatch (-want +got):\n%s", diff)
	}
}

// A request without a last content to append to is left alone instead of
// failing the model call, and the saved parts stay pending for the next one.
// Fails if a nil last content panics or clears the saved parts (rows "nil
// last content"), or if a request without contents clears them (rows "no
// contents").
func TestBeforeModelWithoutLastContentLeavesPartsPending(t *testing.T) {
	tests := []struct {
		name       string
		retention  multimodaltoolresults.Retention
		contents   []*genai.Content
		pendingKey string
	}{
		{"session/no contents", multimodaltoolresults.RetentionSession, nil, currentTurnKey},
		{"session/nil last content", multimodaltoolresults.RetentionSession, []*genai.Content{nil}, currentTurnKey},
		{"next_model_call/no contents", multimodaltoolresults.RetentionNextModelCall, nil, multimodaltoolresults.PartsReturnedByToolsKey},
		{"next_model_call/nil last content", multimodaltoolresults.RetentionNextModelCall, []*genai.Content{nil}, multimodaltoolresults.PartsReturnedByToolsKey},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, tc.retention)
			ctx := newCallbackContext(t)
			callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "a"}}})

			unusable := &model.LLMRequest{Contents: slices.Clone(tc.contents)}
			callBeforeModel(t, p, ctx, unusable)
			pending := getState(t, ctx, tc.pendingKey)
			req := oneEmptyContent()
			callBeforeModel(t, p, ctx, req)

			if diff := cmp.Diff(tc.contents, unusable.Contents); diff != "" {
				t.Errorf("request without a last content changed (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]any{&genai.Part{Text: "a"}}, pending); diff != "" {
				t.Errorf("pending parts after unusable request mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(req)); diff != "" {
				t.Errorf("request tail mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails if session retention stores inline-data parts, which adk-python
// attaches to the next model request only.
func TestSessionRetentionAttachesInlineDataOnce(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionSession)
	ctx := newCallbackContext(t)
	image := &genai.Part{InlineData: &genai.Blob{Data: []byte("img"), MIMEType: "image/png"}}
	callAfterTool(t, p, ctx, map[string]any{"result": []*genai.Part{{Text: "a"}, image}})

	r1, r2 := oneEmptyContent(), oneEmptyContent()
	callBeforeModel(t, p, ctx, r1)
	callBeforeModel(t, p, ctx, r2)

	if diff := cmp.Diff([]any{py(map[string]any{"text": "a"})}, getState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)); diff != "" {
		t.Errorf("stored parts mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "a"}, {InlineData: &genai.Blob{Data: []byte("img"), MIMEType: "image/png"}}}, tail(r1)); diff != "" {
		t.Errorf("first request tail mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(r2)); diff != "" {
		t.Errorf("second request tail mismatch (-want +got):\n%s", diff)
	}
}

// Inline-data parts are attached once through the current turn and never
// stored for later turns. Fails if a result of inline-data parts only
// replaces the stored parts or sets the updated flag (row "inline only"), or
// if a result with other parts does not replace the parts an earlier
// invocation stored (row "text and inline").
func TestSessionRetentionDoesNotStoreInlineParts(t *testing.T) {
	inline := &genai.Part{InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}}
	old := []any{py(map[string]any{"text": "old"})}
	tests := []struct {
		name        string
		result      []*genai.Part
		wantStored  []any
		wantCurrent []any
		wantFlag    bool
	}{
		{
			name:        "text and inline",
			result:      []*genai.Part{{Text: "a"}, inline},
			wantStored:  []any{py(map[string]any{"text": "a"})},
			wantCurrent: []any{&genai.Part{Text: "a"}, &genai.Part{InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}}},
			wantFlag:    true,
		},
		{
			name:        "inline only",
			result:      []*genai.Part{inline},
			wantStored:  []any{py(map[string]any{"text": "old"})},
			wantCurrent: []any{&genai.Part{InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}}},
			wantFlag:    false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, multimodaltoolresults.RetentionSession)
			ctx := newCallbackContext(t)
			setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, old)

			callAfterTool(t, p, ctx, map[string]any{"result": tc.result})

			if diff := cmp.Diff(tc.wantStored, getState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey)); diff != "" {
				t.Errorf("stored parts mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantCurrent, getState(t, ctx, currentTurnKey)); diff != "" {
				t.Errorf("current-turn parts mismatch (-want +got):\n%s", diff)
			}
			_, err := ctx.State().Get(updatedKey)
			if got := err == nil; got != tc.wantFlag {
				t.Errorf("updated flag present = %t (Get error %v), want %t", got, err, tc.wantFlag)
			}
		})
	}
}

// Fails if the dedupe is removed or never matches (rows "same text",
// "signature", "empty signature", "metadata"), if it compares only some
// fields (rows "function call", "thought"), or if it compares parts without
// the JSON round trip that a stored part went through (rows "empty
// signature", "metadata").
func TestSessionRetentionDedupe(t *testing.T) {
	tests := []struct {
		name    string
		stored  map[string]any
		current *genai.Part
		want    []*genai.Part
	}{
		{
			name:    "same text",
			stored:  py(map[string]any{"text": "hello"}),
			current: &genai.Part{Text: "hello"},
			want:    []*genai.Part{{Text: "hello"}},
		},
		{
			name:    "function call",
			stored:  py(map[string]any{"function_call": map[string]any{"id": nil, "args": nil, "name": "a", "partial_args": nil, "will_continue": nil}}),
			current: &genai.Part{FunctionCall: &genai.FunctionCall{Name: "b"}},
			want:    []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "a"}}, {FunctionCall: &genai.FunctionCall{Name: "b"}}},
		},
		{
			name:    "thought",
			stored:  py(map[string]any{"text": "x"}),
			current: &genai.Part{Text: "x", Thought: true},
			want:    []*genai.Part{{Text: "x"}, {Text: "x", Thought: true}},
		},
		{
			name:    "display name",
			stored:  py(map[string]any{"file_data": map[string]any{"display_name": "d", "file_uri": "gs://b/f", "mime_type": "application/pdf"}}),
			current: &genai.Part{FileData: &genai.FileData{FileURI: "gs://b/f", MIMEType: "application/pdf"}},
			want: []*genai.Part{
				{FileData: &genai.FileData{DisplayName: "d", FileURI: "gs://b/f", MIMEType: "application/pdf"}},
				{FileData: &genai.FileData{FileURI: "gs://b/f", MIMEType: "application/pdf"}},
			},
		},
		{
			name:    "signature",
			stored:  py(map[string]any{"text": "s", "thought_signature": "c2ln"}),
			current: &genai.Part{Text: "s", ThoughtSignature: []byte("sig")},
			want:    []*genai.Part{{Text: "s", ThoughtSignature: []byte("sig")}},
		},
		{
			name:    "empty signature",
			stored:  py(map[string]any{"text": "e"}),
			current: &genai.Part{Text: "e", ThoughtSignature: []byte{}},
			want:    []*genai.Part{{Text: "e", ThoughtSignature: []byte{}}},
		},
		{
			name:    "inline data",
			stored:  py(map[string]any{"text": "x"}),
			current: &genai.Part{InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}},
			want:    []*genai.Part{{Text: "x"}, {InlineData: &genai.Blob{Data: []byte("d"), MIMEType: "image/png"}}},
		},
		{
			name:    "metadata",
			stored:  py(map[string]any{"text": "x", "part_metadata": map[string]any{"n": float64(1)}}),
			current: &genai.Part{Text: "x", PartMetadata: map[string]any{"n": 1}},
			want:    []*genai.Part{{Text: "x", PartMetadata: map[string]any{"n": 1}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, multimodaltoolresults.RetentionSession)
			ctx := newCallbackContext(t)
			setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, []any{tc.stored})
			setState(t, ctx, currentTurnKey, []any{tc.current})
			req := oneEmptyContent()

			callBeforeModel(t, p, ctx, req)

			if diff := cmp.Diff(tc.want, tail(req)); diff != "" {
				t.Errorf("request tail mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails if the read path does not convert Python's snake_case keys or
// URL-safe base64, or reads only one of the two list types.
func TestSessionRetentionDecodesPythonStoredMaps(t *testing.T) {
	stored := []map[string]any{
		py(map[string]any{"text": "hello"}),
		py(map[string]any{"file_data": map[string]any{"display_name": nil, "file_uri": "gs://b/x", "mime_type": "image/jpeg"}}),
		py(map[string]any{"text": "t", "thought_signature": "-_8="}),
	}
	want := []*genai.Part{
		{Text: "hello"},
		{FileData: &genai.FileData{FileURI: "gs://b/x", MIMEType: "image/jpeg"}},
		{Text: "t", ThoughtSignature: []byte{0xfb, 0xff}},
	}
	tests := []struct {
		name  string
		value any
	}{
		{"any list", []any{stored[0], stored[1], stored[2]}},
		{"map list", stored},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, multimodaltoolresults.RetentionSession)
			ctx := newCallbackContext(t)
			setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, tc.value)
			req := oneEmptyContent()

			callBeforeModel(t, p, ctx, req)

			if diff := cmp.Diff(want, tail(req)); diff != "" {
				t.Errorf("request tail mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails if a stored map that is not a part is decoded as an empty part.
func TestSessionRetentionSkipsStoredNonPartMaps(t *testing.T) {
	p := newPlugin(t, multimodaltoolresults.RetentionSession)
	ctx := newCallbackContext(t)
	setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, []any{py(map[string]any{"text": "caption"}), map[string]any{"a": 1}})
	req := oneEmptyContent()

	callBeforeModel(t, p, ctx, req)

	if diff := cmp.Diff([]*genai.Part{{Text: "caption"}}, tail(req)); diff != "" {
		t.Errorf("request tail mismatch (-want +got):\n%s", diff)
	}
}

// Fails if the lock around the state update is removed or narrowed: entries
// are lost, the flag is written twice, or the race detector reports.
func TestConcurrentAfterToolCallbacksKeepEveryPart(t *testing.T) {
	const n = 32
	wantTexts := make([]string, n)
	for i := range n {
		wantTexts[i] = fmt.Sprint(i)
	}
	slices.Sort(wantTexts)

	// run calls the after-tool callback from n goroutines, each with its own
	// tool context, and returns a callback context over the same session and
	// the tool contexts' state deltas.
	run := func(t *testing.T, p *plugin.Plugin) (agent.Context, []map[string]any) {
		t.Helper()
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: newSession(t)})
		deltas := make([]map[string]any, n)
		var wg sync.WaitGroup
		for i := range n {
			deltas[i] = map[string]any{}
			toolCtx := agent.NewToolContext(invCtx, fmt.Sprint(i), &session.EventActions{StateDelta: deltas[i]}, nil)
			wg.Go(func() {
				got, err := p.AfterToolCallback()(toolCtx, nil, map[string]any{}, map[string]any{"result": []*genai.Part{{Text: fmt.Sprint(i)}}}, nil)
				if got != nil || err != nil {
					t.Errorf("AfterToolCallback() = (%v, %v), want (nil, nil)", got, err)
				}
			})
		}
		wg.Wait()
		return icontext.NewCallbackContextWithDelta(invCtx, nil, nil), deltas
	}
	partTexts := func(t *testing.T, v any) []string {
		t.Helper()
		list, ok := v.([]any)
		if !ok {
			t.Fatalf("state value is %T, want []any", v)
		}
		var texts []string
		for _, e := range list {
			switch e := e.(type) {
			case *genai.Part:
				texts = append(texts, e.Text)
			case map[string]any:
				texts = append(texts, fmt.Sprint(e["text"]))
			default:
				t.Fatalf("state entry is %T, want *genai.Part or map[string]any", e)
			}
		}
		slices.Sort(texts)
		return texts
	}

	t.Run("next_model_call", func(t *testing.T) {
		ctx, _ := run(t, newPlugin(t, multimodaltoolresults.RetentionNextModelCall))

		if diff := cmp.Diff(wantTexts, partTexts(t, getState(t, ctx, multimodaltoolresults.PartsReturnedByToolsKey))); diff != "" {
			t.Errorf("saved part texts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("session", func(t *testing.T) {
		ctx, deltas := run(t, newPlugin(t, multimodaltoolresults.RetentionSession))

		if diff := cmp.Diff(wantTexts, partTexts(t, getState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey))); diff != "" {
			t.Errorf("stored part texts mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(wantTexts, partTexts(t, getState(t, ctx, currentTurnKey))); diff != "" {
			t.Errorf("current-turn part texts mismatch (-want +got):\n%s", diff)
		}
		flagWrites := 0
		for _, d := range deltas {
			if _, ok := d[updatedKey]; ok {
				flagWrites++
			}
		}
		if flagWrites != 1 {
			t.Errorf("tool contexts that wrote the flag = %d, want 1", flagWrites)
		}
	})
}

// pausingState is a session.State whose first Get of key reads the value and
// then waits until resume is closed or limit has passed.
type pausingState struct {
	session.State
	key    string
	read   chan struct{} // closed once the first Get of key has read its value
	resume <-chan struct{}
	limit  time.Duration
	once   sync.Once
}

func (s *pausingState) Get(key string) (any, error) {
	v, err := s.State.Get(key)
	if key == s.key {
		s.once.Do(func() {
			close(s.read)
			select {
			case <-s.resume:
			case <-time.After(s.limit):
			}
		})
	}
	return v, err
}

// stateContext is an agent.Context whose State is state.
type stateContext struct {
	agent.Context
	state session.State
}

func (c stateContext) State() session.State { return c.state }

// A part a tool saves while a model request is reading the saved parts is
// attached by that request or the next one, never lost. Fails if the lock
// in the before-model callback is removed or narrowed, so that the request's
// read and clear of the saved parts and the tool's read and append no longer
// exclude each other.
func TestBeforeModelAttachesEveryConcurrentlySavedPartOnce(t *testing.T) {
	// The model request waits at most pauseLimit for the save. When the two
	// exclude each other the save cannot finish during the wait, so the wait
	// runs to the limit; when they do not, the save finishes at once.
	const pauseLimit = time.Second
	p := newPlugin(t, multimodaltoolresults.RetentionNextModelCall)
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: newSession(t)})
	toolCtx := func(id string) agent.Context {
		return agent.NewToolContext(invCtx, id, &session.EventActions{StateDelta: map[string]any{}}, nil)
	}
	// Each model request has its own callback context, as in the flow.
	modelCtx := func() agent.Context { return icontext.NewCallbackContextWithDelta(invCtx, nil, nil) }
	callAfterTool(t, p, toolCtx("x"), map[string]any{"result": []*genai.Part{{Text: "x"}}})

	saved := make(chan struct{})
	firstCtx := modelCtx()
	paused := &pausingState{State: firstCtx.State(), key: multimodaltoolresults.PartsReturnedByToolsKey, read: make(chan struct{}), resume: saved, limit: pauseLimit}
	first := oneEmptyContent()
	var wg sync.WaitGroup
	wg.Go(func() {
		if _, err := p.BeforeModelCallback()(stateContext{firstCtx, paused}, first); err != nil {
			t.Errorf("BeforeModelCallback failed: %v", err)
		}
	})
	<-paused.read
	wg.Go(func() {
		defer close(saved)
		if _, err := p.AfterToolCallback()(toolCtx("y"), nil, map[string]any{}, map[string]any{"result": []*genai.Part{{Text: "y"}}}, nil); err != nil {
			t.Errorf("AfterToolCallback failed: %v", err)
		}
	})
	wg.Wait()
	second := oneEmptyContent()
	callBeforeModel(t, p, modelCtx(), second)

	if diff := cmp.Diff([]*genai.Part{{Text: "x"}}, tail(first)); diff != "" {
		t.Errorf("first request tail mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]*genai.Part{{Text: "y"}}, tail(second)); diff != "" {
		t.Errorf("second request tail mismatch (-want +got):\n%s", diff)
	}
}

// adk-python iterates whatever the session key holds: the characters of a
// string are attached as non-part entries (which Go skips) and the
// current-turn parts still follow, while a number raises before anything
// changes. Fails if a string value fails the model call, or if a number is
// treated as no stored parts.
func TestSessionRetentionNonListStoredValue(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		p := newPlugin(t, multimodaltoolresults.RetentionSession)
		ctx := newCallbackContext(t)
		setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, "abc")
		setState(t, ctx, currentTurnKey, []any{&genai.Part{Text: "a"}})
		req := oneEmptyContent()

		callBeforeModel(t, p, ctx, req)

		if diff := cmp.Diff([]*genai.Part{{Text: "a"}}, tail(req)); diff != "" {
			t.Errorf("request tail mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]any{}, getState(t, ctx, currentTurnKey)); diff != "" {
			t.Errorf("current-turn parts mismatch (-want +got):\n%s", diff)
		}
	})
	t.Run("number", func(t *testing.T) {
		p := newPlugin(t, multimodaltoolresults.RetentionSession)
		ctx := newCallbackContext(t)
		setState(t, ctx, multimodaltoolresults.SessionPartsReturnedByToolsKey, 5)
		setState(t, ctx, currentTurnKey, []any{&genai.Part{Text: "a"}})
		req := oneEmptyContent()

		got, err := p.BeforeModelCallback()(ctx, req)

		if got != nil || err == nil || !strings.Contains(err.Error(), multimodaltoolresults.SessionPartsReturnedByToolsKey) {
			t.Errorf("BeforeModelCallback() = (%v, %v), want an error naming %q", got, err, multimodaltoolresults.SessionPartsReturnedByToolsKey)
		}
		if got := tail(req); len(got) != 0 {
			t.Errorf("request tail = %v, want empty", got)
		}
		if diff := cmp.Diff([]any{&genai.Part{Text: "a"}}, getState(t, ctx, currentTurnKey)); diff != "" {
			t.Errorf("current-turn parts mismatch (-want +got):\n%s", diff)
		}
	})
}

// Two function calls of one model response whose tools both return parts end
// the run, and the next model request carries both parts, when the later call
// reaches the plugin while the earlier call's tool still runs. Fails if an
// after-tool callback holds the plugin's lock while it waits for the earlier
// calls of the response: the earlier call then waits for the lock and the run
// hangs until its context is done.
func TestParallelCallsThatSavePartsEndTheRun(t *testing.T) {
	// runLimit bounds the run, so that a hang fails the test in seconds.
	// headStart is how long the earlier call's tool keeps running after the
	// later call has reached the after-tool callbacks.
	const (
		runLimit  = 3 * time.Second
		headStart = 100 * time.Millisecond
	)
	for _, retention := range []multimodaltoolresults.Retention{multimodaltoolresults.RetentionNextModelCall, multimodaltoolresults.RetentionSession} {
		t.Run(string(retention), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), runLimit)
			defer cancel()
			laterAtCallbacks := make(chan struct{})
			// Registered before the plugin under test, so it runs first.
			probe, err := plugin.New(plugin.Config{
				Name: "probe",
				AfterToolCallback: func(_ agent.Context, tl tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
					if tl.Name() == "later" {
						close(laterAtCallbacks)
					}
					return nil, nil
				},
			})
			if err != nil {
				t.Fatalf("plugin.New failed: %v", err)
			}
			earlier, err := functiontool.New(functiontool.Config{Name: "earlier", Description: "Returns a part late."},
				func(tc agent.Context, _ struct{}) (*genai.Part, error) {
					select {
					case <-laterAtCallbacks:
						time.Sleep(headStart)
					case <-tc.Done():
					}
					return genai.NewPartFromText("from earlier"), nil
				})
			if err != nil {
				t.Fatalf("functiontool.New failed: %v", err)
			}
			later, err := functiontool.New(functiontool.Config{Name: "later", Description: "Returns a part."},
				func(agent.Context, struct{}) (*genai.Part, error) {
					return genai.NewPartFromText("from later"), nil
				})
			if err != nil {
				t.Fatalf("functiontool.New failed: %v", err)
			}
			mockModel := &testutil.MockModel{Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{
					genai.NewPartFromFunctionCall("earlier", map[string]any{}),
					genai.NewPartFromFunctionCall("later", map[string]any{}),
				}},
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			a, err := llmagent.New(llmagent.Config{Name: "agent", Model: mockModel, Tools: []tool.Tool{earlier, later}})
			if err != nil {
				t.Fatalf("llmagent.New failed: %v", err)
			}
			sessions := session.InMemoryService()
			created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
			if err != nil {
				t.Fatalf("Create session failed: %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName:        "app",
				Agent:          a,
				SessionService: sessions,
				PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{probe, newPlugin(t, retention)}},
			})
			if err != nil {
				t.Fatalf("runner.New failed: %v", err)
			}

			_, err = testutil.CollectEvents(r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}))

			if ctx.Err() != nil {
				t.Fatalf("Run did not end within %v (err %v)", runLimit, err)
			}
			if err != nil {
				t.Fatalf("Run failed: %v", err)
			}
			if len(mockModel.Requests) != 2 {
				t.Fatalf("model requests = %d, want 2", len(mockModel.Requests))
			}
			var texts []string
			for _, p := range tail(mockModel.Requests[1]) {
				if p.Text != "" {
					texts = append(texts, p.Text)
				}
			}
			slices.Sort(texts)
			if diff := cmp.Diff([]string{"from earlier", "from later"}, texts); diff != "" {
				t.Errorf("text parts of the second request's last content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A call's after-tool callback saves its parts at once while an earlier call
// of its model response still runs. Fails if the callback waits for the
// earlier calls of its model response before it saves.
func TestAfterToolSavesWhileAnEarlierCallRuns(t *testing.T) {
	for _, tc := range []struct {
		retention multimodaltoolresults.Retention
		key       string
	}{
		{multimodaltoolresults.RetentionNextModelCall, multimodaltoolresults.PartsReturnedByToolsKey},
		{multimodaltoolresults.RetentionSession, currentTurnKey},
	} {
		t.Run(string(tc.retention), func(t *testing.T) {
			p := newPlugin(t, tc.retention)
			// Call 1 of 2; call 0 never returns.
			callCtx, cancel := context.WithCancel(adkcontext.WithCallPlace(t.Context(), adkcontext.NewCallOrder(2), 1))
			defer cancel()
			invCtx := icontext.NewInvocationContext(callCtx, icontext.InvocationContextParams{Session: newSession(t)})
			toolCtx := agent.NewToolContext(invCtx, "1", &session.EventActions{StateDelta: map[string]any{}}, nil)
			type callbackResult struct {
				got map[string]any
				err error
			}
			returned := make(chan callbackResult, 1)

			go func() {
				got, err := p.AfterToolCallback()(toolCtx, nil, map[string]any{}, map[string]any{"result": []*genai.Part{{Text: "a"}}}, nil)
				returned <- callbackResult{got, err}
			}()

			var r callbackResult
			select {
			case r = <-returned:
			case <-time.After(2 * time.Second):
				cancel()
				r = <-returned
				t.Fatalf("AfterToolCallback() did not return within 2s; after its context was canceled it returned (%v, %v)", r.got, r.err)
			}
			if r.got != nil || r.err != nil {
				t.Errorf("AfterToolCallback() = (%v, %v), want (nil, nil)", r.got, r.err)
			}
			if diff := cmp.Diff([]any{&genai.Part{Text: "a"}}, getState(t, toolCtx, tc.key)); diff != "" {
				t.Errorf("saved parts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
