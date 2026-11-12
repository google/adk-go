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

package llminternal

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// namedFakeLiveModel is fakeLiveModel with a configurable model name, for
// behavior that depends on the Live model family.
type namedFakeLiveModel struct {
	fakeLiveModel
	name string
}

func (m *namedFakeLiveModel) Name() string { return m.name }

// liveSetupFields runs RunLive for modelName with live, captures the setup
// frame the client sends to a fake Live server, and returns every
// realtimeInputConfig, proactivity and enableAffectiveDialog value in it,
// wherever the backend nests them. If requestModel is set, a request
// processor writes it to the request's Model before the live config is built.
func liveSetupFields(t *testing.T, modelName, requestModel string, live *agent.LiveRunConfig) map[string]any {
	t.Helper()
	setup := make(chan []byte, 1)
	var upgrader websocket.Upgrader
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		_, frame, err := conn.ReadMessage()
		if err != nil {
			return
		}
		select {
		case setup <- frame:
		default:
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			return
		}
		blockUntilClientCloses(conn)
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

	testAgent, err := agent.New(agent.Config{Name: "test-live-agent"})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	baseCtx, cancel := context.WithCancel(t.Context())
	defer cancel()
	ctx := icontext.NewInvocationContext(
		runconfig.ToContext(baseCtx, &runconfig.RunConfig{
			StreamingMode: runconfig.StreamingModeBidi,
			Live:          live,
		}),
		icontext.InvocationContextParams{Agent: testAgent},
	)
	processors := []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor}
	if requestModel != "" {
		processors = append(processors, func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
			req.Model = requestModel
			return func(yield func(*session.Event, error) bool) {}
		})
	}
	f := &Flow{
		Model:             &namedFakeLiveModel{fakeLiveModel: fakeLiveModel{client: client}, name: modelName},
		RequestProcessors: processors,
	}
	sess, _, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	var frame []byte
	select {
	case frame = <-setup:
	case <-time.After(10 * time.Second):
		t.Fatal("no setup frame within 10s")
	}
	var msg any
	if err := json.Unmarshal(frame, &msg); err != nil {
		t.Fatalf("setup frame is not JSON: %v\n%s", err, frame)
	}
	got := map[string]any{}
	collectLiveSetupFields(msg, got)
	return got
}

func collectLiveSetupFields(v any, out map[string]any) {
	switch v := v.(type) {
	case map[string]any:
		for k, child := range v {
			switch k {
			case "realtimeInputConfig", "proactivity", "enableAffectiveDialog":
				out[k] = child
			}
			collectLiveSetupFields(child, out)
		}
	case []any:
		for _, child := range v {
			collectLiveSetupFields(child, out)
		}
	}
}

// TestRunLiveSendsLiveRunConfigFields checks that RealtimeInputConfig,
// EnableAffectiveDialog and Proactivity from LiveRunConfig reach the setup
// message, and that the last two are left out for the models IsGemini3xLive
// matches.
func TestRunLiveSendsLiveRunConfigFields(t *testing.T) {
	live := func() *agent.LiveRunConfig {
		return &agent.LiveRunConfig{
			RealtimeInputConfig: &genai.RealtimeInputConfig{
				AutomaticActivityDetection: &genai.AutomaticActivityDetection{
					Disabled:               true,
					EndOfSpeechSensitivity: genai.EndSensitivityHigh,
					SilenceDurationMs:      genai.Ptr[int32](300),
				},
			},
			EnableAffectiveDialog: true,
			Proactivity:           &genai.ProactivityConfig{ProactiveAudio: new(true)},
		}
	}
	wantRealtime := map[string]any{
		"automaticActivityDetection": map[string]any{
			"disabled":               true,
			"endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
			"silenceDurationMs":      float64(300),
		},
	}
	tests := []struct {
		name         string
		model        string
		requestModel string
		live         *agent.LiveRunConfig
		want         map[string]any
	}{
		{
			name:  "gemini 2.5 live",
			model: "gemini-live-2.5-flash-preview",
			live:  live(),
			want: map[string]any{
				"realtimeInputConfig":   wantRealtime,
				"enableAffectiveDialog": true,
				"proactivity":           map[string]any{"proactiveAudio": true},
			},
		},
		{
			name:  "gemini 3.x live",
			model: "gemini-3.1-flash-live-preview",
			live:  live(),
			want:  map[string]any{"realtimeInputConfig": wantRealtime},
		},
		{
			name:  "gemini 3.x live by resource name",
			model: "projects/p/locations/us-central1/publishers/google/models/gemini-3.1-flash-live-preview",
			live:  live(),
			want:  map[string]any{"realtimeInputConfig": wantRealtime},
		},
		{
			// Live Translate is not treated as a 3.x Live model.
			name:  "gemini 3.5 live translate",
			model: "gemini-3.5-live-translate-preview",
			live:  live(),
			want: map[string]any{
				"realtimeInputConfig":   wantRealtime,
				"enableAffectiveDialog": true,
				"proactivity":           map[string]any{"proactiveAudio": true},
			},
		},
		{
			// The connection is opened for the Flow's model, so the guard
			// follows it even when a request processor rewrites req.Model.
			name:         "request model rewritten to gemini 3.x live",
			model:        "gemini-live-2.5-flash-preview",
			requestModel: "gemini-3.1-flash-live-preview",
			live:         live(),
			want: map[string]any{
				"realtimeInputConfig":   wantRealtime,
				"enableAffectiveDialog": true,
				"proactivity":           map[string]any{"proactiveAudio": true},
			},
		},
		{
			name:         "request model rewritten away from gemini 3.x live",
			model:        "gemini-3.1-flash-live-preview",
			requestModel: "gemini-live-2.5-flash-preview",
			live:         live(),
			want:         map[string]any{"realtimeInputConfig": wantRealtime},
		},
		// Each field is set on its own as well, so that dropping one of them
		// cannot hide behind another being set.
		{
			name:  "realtime input config only",
			model: "gemini-live-2.5-flash-preview",
			live:  &agent.LiveRunConfig{RealtimeInputConfig: live().RealtimeInputConfig},
			want:  map[string]any{"realtimeInputConfig": wantRealtime},
		},
		{
			name:  "affective dialog only",
			model: "gemini-live-2.5-flash-preview",
			live:  &agent.LiveRunConfig{EnableAffectiveDialog: true},
			want:  map[string]any{"enableAffectiveDialog": true},
		},
		{
			name:  "proactivity only",
			model: "gemini-live-2.5-flash-preview",
			live: &agent.LiveRunConfig{
				Proactivity: &genai.ProactivityConfig{ProactiveAudio: new(true)},
			},
			want: map[string]any{"proactivity": map[string]any{"proactiveAudio": true}},
		},
		{
			name:  "unset",
			model: "gemini-live-2.5-flash-preview",
			live:  &agent.LiveRunConfig{},
			want:  map[string]any{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			got := liveSetupFields(t, tc.model, tc.requestModel, tc.live)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("setup fields mismatch (-want +got):\n%s", diff)
			}
			assertNoRunLiveLeak(t, baseline)
		})
	}
}
