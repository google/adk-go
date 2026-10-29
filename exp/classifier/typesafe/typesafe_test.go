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

package typesafe

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/exp/classifier"
)

var (
	routeQuestion = classifier.Choice{
		Instructions: "Which team should handle this?",
		Criteria: map[string]any{
			"billing":   "Invoices and refunds",
			"technical": nil,
		},
	}
	urgentQuestion = classifier.Binary{
		Instructions: "Is this urgent?",
		Criteria:     "Time-sensitive",
	}
	frustrationQuestion = classifier.Score{
		Instructions: "How frustrated is the customer?",
		Criteria:     []any{"Calm", "Frustrated", "Very angry"},
	}
)

type queuedReply struct {
	resp *http.Response
	err  error
	wait time.Duration
}

type mockTransport struct {
	mu       sync.Mutex
	replies  []queuedReply
	requests []*http.Request
	bodies   [][]byte
}

func newMockTransport(replies ...queuedReply) *mockTransport {
	return &mockTransport{replies: replies}
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		var err error
		body, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
	}

	m.mu.Lock()
	m.requests = append(m.requests, req.Clone(req.Context()))
	m.bodies = append(m.bodies, body)
	if len(m.replies) == 0 {
		m.mu.Unlock()
		return nil, errors.New("unexpected request: no queued replies left")
	}
	reply := m.replies[0]
	m.replies = m.replies[1:]
	m.mu.Unlock()

	if reply.wait > 0 {
		timer := time.NewTimer(reply.wait)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}
	if reply.err != nil {
		return nil, reply.err
	}
	return reply.resp, nil
}

func (m *mockTransport) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

func (m *mockTransport) bodyMap(t *testing.T, idx int) map[string]any {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	var got map[string]any
	if err := json.Unmarshal(m.bodies[idx], &got); err != nil {
		t.Fatalf("unmarshaling request body[%d]: %v", idx, err)
	}
	return got
}

func okReply(t *testing.T, answers map[string]any, model string) queuedReply {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"model":   model,
		"answers": answers,
		"usage":   map[string]any{"input_tokens": 10, "output_tokens": 2},
	})
	if err != nil {
		t.Fatalf("marshaling mock response: %v", err)
	}
	return textReply(http.StatusOK, string(b), nil)
}

func textReply(status int, body string, headers map[string]string) queuedReply {
	hdr := make(http.Header)
	for k, v := range headers {
		hdr.Set(k, v)
	}
	return queuedReply{
		resp: &http.Response{
			StatusCode: status,
			Header:     hdr,
			Body:       io.NopCloser(strings.NewReader(body)),
		},
	}
}

type failingReader struct {
	err error
}

func (f failingReader) Read([]byte) (int, error) { return 0, f.err }

type customTimeoutNetError struct{}

func (customTimeoutNetError) Error() string   { return "custom network timeout" }
func (customTimeoutNetError) Timeout() bool   { return true }
func (customTimeoutNetError) Temporary() bool { return false }

func newTestClassifier(t *testing.T, tr *mockTransport, cfg Config) *Classifier {
	t.Helper()
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Transport: tr}
	}
	clf, err := New(cfg)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	clf.sleep = func(ctx context.Context, _ time.Duration) error {
		return ctx.Err()
	}
	return clf
}

func TestSendsEveryQuestionTypeInOneRequest(t *testing.T) {
	tr := newMockTransport(okReply(t, map[string]any{
		"route": map[string]any{
			"type":          "choice",
			"choice":        "billing",
			"probabilities": map[string]any{"billing": 0.9, "technical": 0.1},
			"confidence":    0.8,
		},
		"urgent": map[string]any{
			"type": "noul",
			"noul": 0.95,
		},
		"frustration": map[string]any{
			"type":          "score",
			"score":         1.05,
			"legend":        map[string]any{"0": "Calm", "1": "Frustrated", "2": "Very angry"},
			"probabilities": map[string]any{"0": 0.0, "1": 0.95, "2": 0.05},
			"confidence":    0.92,
		},
	}, "jev-1.13.0"))

	clf := newTestClassifier(t, tr, Config{})
	resp, err := clf.Classify(t.Context(), map[string]any{"message": "I was charged twice!"}, map[string]classifier.Question{
		"route":       routeQuestion,
		"urgent":      urgentQuestion,
		"frustration": frustrationQuestion,
	})
	if err != nil {
		t.Fatalf("Classify() unexpected error: %v", err)
	}

	if got := tr.count(); got != 1 {
		t.Fatalf("request count = %d, want 1", got)
	}
	req := tr.requests[0]
	if got, want := req.URL.String(), "https://api.typesafe.ai/v1/systemone"; got != want {
		t.Errorf("request URL = %q, want %q", got, want)
	}
	if got, want := req.Header.Get("Authorization"), "Bearer test-key"; got != want {
		t.Errorf("Authorization = %q, want %q", got, want)
	}
	if got, want := req.Header.Get("Content-Type"), "application/json"; got != want {
		t.Errorf("Content-Type = %q, want %q", got, want)
	}

	wantBody := map[string]any{
		"model": "jev-latest",
		"state": map[string]any{"message": "I was charged twice!"},
		"questions": map[string]any{
			"route": map[string]any{
				"type":         "choice",
				"instructions": "Which team should handle this?",
				"criteria": map[string]any{
					"billing":   "Invoices and refunds",
					"technical": nil,
				},
			},
			"urgent": map[string]any{
				"type":         "noul",
				"instructions": "Is this urgent?",
				"criteria":     map[string]any{"true": "Time-sensitive"},
			},
			"frustration": map[string]any{
				"type":         "score",
				"instructions": "How frustrated is the customer?",
				"criteria":     []any{"Calm", "Frustrated", "Very angry"},
			},
		},
	}
	if diff := cmp.Diff(wantBody, tr.bodyMap(t, 0)); diff != "" {
		t.Errorf("request body mismatch (-want +got):\n%s", diff)
	}

	confRoute := 0.8
	confScore := 0.92
	wantResp := &classifier.Response{
		ModelVersion: "jev-1.13.0",
		Answers: map[string]classifier.Answer{
			"route": classifier.ChoiceAnswer{
				Choice:        "billing",
				Probabilities: map[string]float64{"billing": 0.9, "technical": 0.1},
				Confidence:    &confRoute,
			},
			"urgent": classifier.BinaryAnswer{
				Probability: 0.95,
			},
			"frustration": classifier.ScoreAnswer{
				Score:         1.05,
				Probabilities: map[int]float64{0: 0.0, 1: 0.95, 2: 0.05},
				Confidence:    &confScore,
			},
		},
	}
	if diff := cmp.Diff(wantResp, resp); diff != "" {
		t.Errorf("Classify() response mismatch (-want +got):\n%s", diff)
	}
}

func TestBinaryWithoutCriteriaOmitsCriteriaField(t *testing.T) {
	tr := newMockTransport(okReply(t, map[string]any{
		"urgent": map[string]any{"type": "noul", "noul": 0.1, "confidence": nil},
	}, "jev-1.13.0"))

	clf := newTestClassifier(t, tr, Config{})
	resp, err := clf.Classify(t.Context(), "hi", map[string]classifier.Question{
		"urgent": classifier.Binary{Instructions: "Is this urgent?"},
	})
	if err != nil {
		t.Fatalf("Classify() unexpected error: %v", err)
	}

	gotQ := tr.bodyMap(t, 0)["questions"].(map[string]any)["urgent"].(map[string]any)
	if _, hasCriteria := gotQ["criteria"]; hasCriteria {
		t.Errorf("expected criteria to be omitted when nil, got %v", gotQ["criteria"])
	}
	if ans := resp.Answers["urgent"].(classifier.BinaryAnswer); ans.Confidence != nil {
		t.Errorf("Confidence = %v, want nil when response confidence is null", *ans.Confidence)
	}
}

func TestContentInputHandling(t *testing.T) {
	t.Run("concatenates_text_and_signed_streamed_parts", func(t *testing.T) {
		tr := newMockTransport(okReply(t, map[string]any{
			"urgent": map[string]any{"type": "noul", "noul": 0.1},
		}, "jev-1.13.0"))
		content := &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: "Help, "},
				{Text: "now", ThoughtSignature: []byte("gemini-sig"), PartMetadata: map[string]any{"k": "v"}},
				{Text: "", ThoughtSignature: []byte("trailing-stream-sig")},
			},
		}
		_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), content, map[string]classifier.Question{
			"urgent": urgentQuestion,
		})
		if err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		if got := tr.bodyMap(t, 0)["state"]; got != "Help, now" {
			t.Errorf("state = %v, want %q", got, "Help, now")
		}
	})

	nonTextParts := []struct {
		name string
		part *genai.Part
	}{
		{name: "nil_part", part: nil},
		{name: "inline_data", part: genai.NewPartFromBytes([]byte("png"), "image/png")},
		{name: "file_data", part: genai.NewPartFromURI("gs://bucket/img.png", "image/png")},
		{name: "thought", part: &genai.Part{Text: "hmm", Thought: true}},
		{name: "thought_empty_text_with_signature", part: &genai.Part{Thought: true, ThoughtSignature: []byte("sig")}},
		{name: "function_call", part: genai.NewPartFromFunctionCall("fn", map[string]any{"x": 1})},
		{name: "function_response", part: genai.NewPartFromFunctionResponse("fn", map[string]any{"ok": true})},
		{name: "executable_code", part: genai.NewPartFromExecutableCode("print(1)", genai.LanguagePython)},
		{name: "code_execution_result", part: genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "1")},
		{name: "tool_call", part: &genai.Part{ToolCall: &genai.ToolCall{}}},
		{name: "tool_response", part: &genai.Part{ToolResponse: &genai.ToolResponse{}}},
		{name: "video_metadata", part: &genai.Part{VideoMetadata: &genai.VideoMetadata{}}},
		{name: "media_resolution", part: &genai.Part{MediaResolution: &genai.PartMediaResolution{}}},
		{name: "audio_transcription", part: &genai.Part{AudioTranscription: &genai.Transcription{}}},
		{name: "speech_metadata", part: &genai.Part{Text: "hi", SpeechMetadata: &genai.SpeechMetadata{}}},
	}

	for _, tc := range nonTextParts {
		t.Run("rejects_"+tc.name, func(t *testing.T) {
			tr := newMockTransport()
			content := genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{tc.part},
			}
			_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), content, map[string]classifier.Question{
				"urgent": urgentQuestion,
			})
			if !errors.Is(err, classifier.ErrInvalidInput) {
				t.Fatalf("Classify() error = %v, want errors.Is(ErrInvalidInput)", err)
			}
			if tr.count() != 0 {
				t.Errorf("request count = %d, want 0", tr.count())
			}
		})
	}

	// genai values outside a single top-level Content marshal as plain JSON and
	// would otherwise be sent as "state" with their inline data and thought parts.
	imageContent := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{genai.NewPartFromBytes([]byte("png-bytes"), "image/png")}}
	thoughtContent := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "secret reasoning", Thought: true}}}
	nestedGenaiInputs := []struct {
		name  string
		input any
	}{
		{name: "content_history", input: []*genai.Content{imageContent, thoughtContent}},
		{name: "content_value_history", input: []genai.Content{*imageContent, *thoughtContent}},
		{name: "genai_text_helper", input: genai.Text("secret reasoning")},
		{name: "empty_content_history", input: []*genai.Content{}},
		{name: "part_value", input: genai.Part{Text: "secret reasoning", Thought: true}},
		{name: "part_pointer", input: &genai.Part{InlineData: &genai.Blob{Data: []byte("png-bytes"), MIMEType: "image/png"}}},
		{name: "history_in_map", input: map[string]any{"ticket": "hi", "history": []*genai.Content{imageContent, thoughtContent}}},
		{name: "content_in_struct_field", input: struct {
			Ticket  string
			Context *genai.Content
		}{Ticket: "hi", Context: thoughtContent}},
	}
	for _, tc := range nestedGenaiInputs {
		t.Run("rejects_nested_"+tc.name, func(t *testing.T) {
			tr := newMockTransport()
			_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), tc.input, map[string]classifier.Question{
				"urgent": urgentQuestion,
			})
			if !errors.Is(err, classifier.ErrInvalidInput) {
				t.Fatalf("Classify(%T) error = %v, want errors.Is(ErrInvalidInput)", tc.input, err)
			}
			if msg := err.Error(); strings.Contains(msg, "secret reasoning") || strings.Contains(msg, "png-bytes") {
				t.Errorf("Classify(%T) error leaked input content: %q", tc.input, msg)
			}
			if tr.count() != 0 {
				t.Errorf("request count = %d, want 0", tr.count())
			}
		})
	}
}

func TestPreRequestValidationErrorsSendNoRequest(t *testing.T) {
	questionCases := []struct {
		name      string
		questions map[string]classifier.Question
	}{
		{
			name:      "empty_questions",
			questions: map[string]classifier.Question{},
		},
		{
			name: "binary_without_instructions",
			questions: map[string]classifier.Question{
				"urgent": classifier.Binary{},
			},
		},
		{
			name: "choice_without_instructions",
			questions: map[string]classifier.Question{
				"route": classifier.Choice{Criteria: map[string]any{"a": nil, "b": nil}},
			},
		},
		{
			name: "score_without_instructions",
			questions: map[string]classifier.Question{
				"score": classifier.Score{Criteria: []any{"low", "high"}},
			},
		},
		{
			name: "invalid_choice_single_label",
			questions: map[string]classifier.Question{
				"route": classifier.Choice{Instructions: "pick", Criteria: map[string]any{"only": nil}},
			},
		},
	}

	for _, tc := range questionCases {
		t.Run(tc.name, func(t *testing.T) {
			tr := newMockTransport()
			_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", tc.questions)
			if !errors.Is(err, classifier.ErrInvalidQuestion) {
				t.Fatalf("Classify() error = %v, want errors.Is(ErrInvalidQuestion)", err)
			}
			if tr.count() != 0 {
				t.Errorf("request count = %d, want 0", tr.count())
			}
		})
	}

	inputCases := []struct {
		name  string
		input any
	}{
		{name: "nil_input", input: nil},
		{name: "bare_int_input", input: 42},
		{name: "bare_bool_input", input: true},
	}
	for _, tc := range inputCases {
		t.Run(tc.name, func(t *testing.T) {
			tr := newMockTransport()
			_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), tc.input, map[string]classifier.Question{
				"urgent": urgentQuestion,
			})
			if !errors.Is(err, classifier.ErrInvalidInput) {
				t.Fatalf("Classify() error = %v, want errors.Is(ErrInvalidInput)", err)
			}
			if tr.count() != 0 {
				t.Errorf("request count = %d, want 0", tr.count())
			}
		})
	}

	t.Run("zero_value_classifier_rejected", func(t *testing.T) {
		var zero Classifier
		_, err := zero.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("zero Classifier.Classify() error = %v, want errors.Is(ErrInvalidConfig)", err)
		}
	})
}

func TestConfigAndEnvironmentResolution(t *testing.T) {
	t.Run("reads_api_key_and_base_url_from_env", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "  env-key\n")
		t.Setenv("TYPESAFE_BASE_URL", "http://localhost:8088/")
		tr := newMockTransport(okReply(t, map[string]any{
			"urgent": map[string]any{"type": "noul", "noul": 0.1},
		}, "jev-1.13.0"))

		clf, err := New(Config{
			Model:      "jev-1.13",
			HTTPClient: &http.Client{Transport: tr},
		})
		if err != nil {
			t.Fatalf("New() unexpected error: %v", err)
		}
		if got := clf.Name(); got != "jev-1.13" {
			t.Errorf("Name() = %q, want %q", got, "jev-1.13")
		}

		_, err = clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		req := tr.requests[0]
		if got, want := req.URL.String(), "http://localhost:8088/v1/systemone"; got != want {
			t.Errorf("URL = %q, want %q", got, want)
		}
		if got, want := req.Header.Get("Authorization"), "Bearer env-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
		if got, want := tr.bodyMap(t, 0)["model"], "jev-1.13"; got != want {
			t.Errorf("model = %v, want %q", got, want)
		}
	})

	t.Run("longer_config_timeout_overrides_shorter_client_timeout", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "env-key")
		t.Setenv("TYPESAFE_BASE_URL", "http://env-host")

		tr := newMockTransport(queuedReply{
			resp: okReply(t, map[string]any{"urgent": map[string]any{"type": "noul", "noul": 0.1}}, "jev-1.13.0").resp,
			wait: 50 * time.Millisecond,
		})
		callerClient := &http.Client{
			Transport: tr,
			Timeout:   15 * time.Millisecond,
		}
		clf := newTestClassifier(t, tr, Config{
			APIKey:     "explicit-key",
			BaseURL:    "http://explicit/",
			Timeout:    200 * time.Millisecond,
			HTTPClient: callerClient,
		})

		_, err := clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		if callerClient.Timeout != 15*time.Millisecond {
			t.Errorf("caller HTTPClient.Timeout mutated to %v", callerClient.Timeout)
		}
		req := tr.requests[0]
		if got, want := req.URL.String(), "http://explicit/v1/systemone"; got != want {
			t.Errorf("URL = %q, want %q", got, want)
		}
		if got, want := req.Header.Get("Authorization"), "Bearer explicit-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
	})

	t.Run("shorter_config_timeout_overrides_longer_client_timeout_and_retries", func(t *testing.T) {
		tr := newMockTransport(
			queuedReply{wait: 150 * time.Millisecond},
			queuedReply{wait: 150 * time.Millisecond},
		)
		callerClient := &http.Client{
			Transport: tr,
			Timeout:   5 * time.Second,
		}
		clf := newTestClassifier(t, tr, Config{
			Timeout:     20 * time.Millisecond,
			MaxAttempts: 2,
			HTTPClient:  callerClient,
		})

		_, err := clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Classify() error = %v, want errors.Is(context.DeadlineExceeded)", err)
		}
		if got := tr.count(); got != 2 {
			t.Errorf("request count = %d, want 2", got)
		}
	})

	t.Run("uses_supplied_client_timeout_when_config_timeout_unset", func(t *testing.T) {
		tr := newMockTransport(
			queuedReply{wait: 150 * time.Millisecond},
			okReply(t, map[string]any{"urgent": map[string]any{"type": "noul", "noul": 0.3}}, "jev-1.13.0"),
		)
		callerClient := &http.Client{
			Transport: tr,
			Timeout:   20 * time.Millisecond,
		}
		clf := newTestClassifier(t, tr, Config{
			MaxAttempts: 2,
			HTTPClient:  callerClient,
		})

		resp, err := clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		if got := tr.count(); got != 2 {
			t.Errorf("request count = %d, want 2", got)
		}
		if got := resp.Answers["urgent"].(classifier.BinaryAnswer).Probability; got != 0.3 {
			t.Errorf("Probability = %v, want 0.3", got)
		}
	})

	t.Run("missing_or_whitespace_api_key_returns_err_missing_api_key", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "   ")
		tr := newMockTransport()
		clf, err := New(Config{HTTPClient: &http.Client{Transport: tr}})
		if err != nil {
			t.Fatalf("New() unexpected error: %v", err)
		}

		_, err = clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, ErrMissingAPIKey) {
			t.Fatalf("Classify() error = %v, want errors.Is(ErrMissingAPIKey)", err)
		}
		if tr.count() != 0 {
			t.Errorf("request count = %d, want 0", tr.count())
		}
	})

	t.Run("invalid_config_rejected", func(t *testing.T) {
		badConfigs := []struct {
			name string
			cfg  Config
		}{
			{name: "whitespace_model", cfg: Config{Model: "   "}},
			{name: "negative_timeout", cfg: Config{Timeout: -time.Second}},
			{name: "negative_max_attempts", cfg: Config{MaxAttempts: -1}},
			{name: "bad_scheme_base_url", cfg: Config{BaseURL: "ftp://api.typesafe.ai"}},
			{name: "hostless_base_url", cfg: Config{BaseURL: "https://"}},
			{name: "whitespace_base_url", cfg: Config{BaseURL: "   "}},
			{name: "query_in_base_url", cfg: Config{BaseURL: "https://api.typesafe.ai?token=secret-url-tok"}},
			{name: "trailing_question_mark_in_base_url", cfg: Config{BaseURL: "https://api.typesafe.ai?"}},
			{name: "fragment_in_base_url", cfg: Config{BaseURL: "https://api.typesafe.ai#frag"}},
			{name: "trailing_hash_in_base_url", cfg: Config{BaseURL: "https://api.typesafe.ai#"}},
			{name: "userinfo_in_base_url", cfg: Config{BaseURL: "https://user:secret-pass@api.typesafe.ai"}},
		}
		for _, tc := range badConfigs {
			t.Run(tc.name, func(t *testing.T) {
				_, err := New(tc.cfg)
				if !errors.Is(err, ErrInvalidConfig) {
					t.Fatalf("New() error = %v, want errors.Is(ErrInvalidConfig)", err)
				}
				if strings.Contains(err.Error(), "secret-") {
					t.Errorf("error leaked URL credential material: %v", err)
				}
			})
		}
	})

	t.Run("invalid_env_base_url_rejected_without_leaking_url", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "k")
		t.Setenv("TYPESAFE_BASE_URL", "https://evil.example.com?token=secret-env-tok")
		tr := newMockTransport()
		clf, err := New(Config{HTTPClient: &http.Client{Transport: tr}})
		if err != nil {
			t.Fatalf("New() unexpected error: %v", err)
		}
		_, err = clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("Classify() error = %v, want errors.Is(ErrInvalidConfig)", err)
		}
		if strings.Contains(err.Error(), "secret-env-tok") || strings.Contains(err.Error(), "evil.example.com") {
			t.Errorf("error leaked TYPESAFE_BASE_URL value: %v", err)
		}
	})
}

func TestSecretAndPayloadHygiene(t *testing.T) {
	cfg := Config{APIKey: "super-secret-key", Model: "jev-latest"}
	clf, err := New(cfg)
	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}
	// A Classifier value, not only a *Classifier, must redact: fmt only uses
	// String and GoString on a value when they have value receivers.
	type holder struct {
		Config     Config
		Classifier Classifier
		Ptr        *Classifier
	}
	held := holder{Config: cfg, Classifier: *clf, Ptr: clf}
	for _, formatted := range []string{
		fmt.Sprint(cfg),
		fmt.Sprintf("%+v", cfg),
		fmt.Sprintf("%#v", cfg),
		fmt.Sprint(clf),
		fmt.Sprintf("%+v", clf),
		fmt.Sprintf("%#v", clf),
		fmt.Sprint(*clf),
		fmt.Sprintf("%+v", *clf),
		fmt.Sprintf("%#v", *clf),
		fmt.Sprint(held),
		fmt.Sprintf("%+v", held),
		fmt.Sprintf("%#v", held),
		fmt.Sprintf("%+v", []Classifier{*clf}),
		fmt.Sprintf("%+v", map[string]Classifier{"clf": *clf}),
	} {
		if strings.Contains(formatted, "super-secret-key") {
			t.Errorf("formatted representation leaked API key: %s", formatted)
		}
	}
	if got, want := fmt.Sprint(*clf), `typesafe.Classifier{Model:"jev-latest"}`; got != want {
		t.Errorf("fmt.Sprint(Classifier value) = %q, want %q", got, want)
	}

	// Non-2xx response body and transport URL must never appear in error strings.
	tr := newMockTransport(
		textReply(http.StatusUnprocessableEntity, `{"detail":"sensitive user prompt echo"}`, map[string]string{
			"X-Request-Id": "req_12345",
		}),
	)
	clf = newTestClassifier(t, tr, Config{BaseURL: "https://internal-proxy.example.com"})
	_, err = clf.Classify(t.Context(), "sensitive user prompt echo", map[string]classifier.Question{"urgent": urgentQuestion})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, ErrBadRequest) {
		t.Fatalf("Classify() error = %v, want *APIError wrapping ErrBadRequest", err)
	}
	if apiErr.StatusCode != http.StatusUnprocessableEntity || apiErr.RequestID != "req_12345" {
		t.Errorf("APIError = %+v, want status 422 and request_id req_12345", apiErr)
	}
	if got, want := apiErr.Error(), "typesafe: request failed with status 422 [request_id=req_12345]: bad request"; got != want {
		t.Errorf("APIError.Error() = %q, want %q", got, want)
	}
	if strings.Contains(err.Error(), "sensitive user prompt") || strings.Contains(err.Error(), "internal-proxy.example.com") {
		t.Errorf("APIError leaked sensitive body or URL: %v", err)
	}

	// Transport *url.Error (both nested and nil-Err) must have its URL stripped.
	for _, tc := range []struct {
		err  error
		want string
	}{
		{
			err: &url.Error{
				Op:  "Post",
				URL: "https://internal-proxy.example.com/v1/systemone?secret=tok123",
				Err: errors.New("unsupported protocol scheme"),
			},
			want: "typesafe: transport error (Post): unsupported protocol scheme",
		},
		{
			err: &url.Error{
				Op:  "Post",
				URL: "https://internal-proxy.example.com/v1/systemone?secret=tok123",
				Err: nil,
			},
			want: "typesafe: transport error (Post): unknown transport error",
		},
	} {
		urlErrTransport := newMockTransport(queuedReply{err: tc.err})
		clf = newTestClassifier(t, urlErrTransport, Config{MaxAttempts: 1})
		_, err = clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if err == nil {
			t.Fatal("Classify() succeeded, want transport error")
		}
		if got := err.Error(); got != tc.want {
			t.Errorf("transport error = %q, want %q", got, tc.want)
		}
	}
}

func TestRetriesAndFailureClassification(t *testing.T) {
	t.Run("retries_429_and_529_then_succeeds", func(t *testing.T) {
		for _, status := range []int{http.StatusTooManyRequests, statusOverloaded} {
			t.Run(fmt.Sprintf("status_%d", status), func(t *testing.T) {
				tr := newMockTransport(
					textReply(status, `{"detail":"slow down"}`, nil),
					okReply(t, map[string]any{"urgent": map[string]any{"type": "noul", "noul": 0.7}}, "jev-1.13.0"),
				)
				resp, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"urgent": urgentQuestion,
				})
				if err != nil {
					t.Fatalf("Classify() unexpected error: %v", err)
				}
				if got := tr.count(); got != 2 {
					t.Errorf("request count = %d, want 2", got)
				}
				if diff := cmp.Diff(classifier.BinaryAnswer{Probability: 0.7}, resp.Answers["urgent"]); diff != "" {
					t.Errorf("answer mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("gives_up_after_max_attempts_on_429_and_529", func(t *testing.T) {
		cases := []struct {
			status  int
			wantErr error
		}{
			{status: http.StatusTooManyRequests, wantErr: ErrRateLimited},
			{status: statusOverloaded, wantErr: ErrOverloaded},
		}
		for _, tc := range cases {
			t.Run(fmt.Sprintf("status_%d", tc.status), func(t *testing.T) {
				tr := newMockTransport(
					textReply(tc.status, "busy", nil),
					textReply(tc.status, "busy", nil),
				)
				_, err := newTestClassifier(t, tr, Config{MaxAttempts: 2}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"urgent": urgentQuestion,
				})
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("Classify() error = %v, want errors.Is(%v)", err, tc.wantErr)
				}
				if got := tr.count(); got != 2 {
					t.Errorf("request count = %d, want 2", got)
				}
			})
		}
	})

	t.Run("retries_transient_transport_and_body_read_errors", func(t *testing.T) {
		transientReplies := []struct {
			name  string
			reply queuedReply
		}{
			{
				name:  "net_op_dial_error",
				reply: queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}},
			},
			{
				name:  "per_attempt_deadline_exceeded",
				reply: queuedReply{err: context.DeadlineExceeded},
			},
			{
				name:  "eof_disconnect",
				reply: queuedReply{err: io.EOF},
			},
			{
				name:  "unexpected_eof_disconnect",
				reply: queuedReply{err: io.ErrUnexpectedEOF},
			},
			{
				name:  "dns_timeout",
				reply: queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: &net.DNSError{Err: "i/o timeout", IsTimeout: true}}},
			},
			{
				name:  "dns_temporary",
				reply: queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: &net.DNSError{Err: "temporary failure", IsTemporary: true}}},
			},
			{
				name:  "custom_net_error_timeout",
				reply: queuedReply{err: customTimeoutNetError{}},
			},
			{
				name:  "http2_rst_stream",
				reply: queuedReply{err: errors.New("stream error: stream ID 3; INTERNAL_ERROR; received from peer")},
			},
			{
				name:  "http2_goaway",
				reply: queuedReply{err: errors.New("http2: server sent GOAWAY and closed the connection; LastStreamID=1")},
			},
			{
				name: "response_body_mid_stream_disconnect",
				reply: queuedReply{
					resp: &http.Response{
						StatusCode: http.StatusOK,
						Header:     make(http.Header),
						Body:       io.NopCloser(failingReader{err: io.ErrUnexpectedEOF}),
					},
				},
			},
		}

		for _, tc := range transientReplies {
			t.Run(tc.name, func(t *testing.T) {
				tr := newMockTransport(
					tc.reply,
					okReply(t, map[string]any{"urgent": map[string]any{"type": "noul", "noul": 0.7}}, "jev-1.13.0"),
				)
				resp, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"urgent": urgentQuestion,
				})
				if err != nil {
					t.Fatalf("Classify() unexpected error: %v", err)
				}
				if got := tr.count(); got != 2 {
					t.Errorf("request count = %d, want 2", got)
				}
				if got := resp.Answers["urgent"].(classifier.BinaryAnswer).Probability; got != 0.7 {
					t.Errorf("Probability = %v, want 0.7", got)
				}
			})
		}
	})

	t.Run("permanent_transport_and_http_errors_not_retried", func(t *testing.T) {
		permanentCases := []struct {
			name    string
			reply   queuedReply
			wantErr error
		}{
			{
				name:  "unsupported_protocol",
				reply: queuedReply{err: &url.Error{Op: "Post", URL: "ftp://bad", Err: errors.New("unsupported protocol scheme")}},
			},
			{
				name:  "proxyconnect_error",
				reply: queuedReply{err: &net.OpError{Op: "proxyconnect", Net: "tcp", Err: errors.New("bad proxy")}},
			},
			{
				name:  "x509_unknown_authority",
				reply: queuedReply{err: x509.UnknownAuthorityError{}},
			},
			{
				name:  "op_wrapped_dns_not_found",
				reply: queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: &net.DNSError{Err: "no such host", IsNotFound: true}}},
			},
			{
				name:  "op_wrapped_net_addr_error",
				reply: queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: &net.AddrError{Err: "missing port in address", Addr: "bad"}}},
			},
			{
				name:    "op_wrapped_context_canceled",
				reply:   queuedReply{err: &net.OpError{Op: "dial", Net: "tcp", Err: context.Canceled}},
				wantErr: context.Canceled,
			},
			{
				name:    "http_400_bad_request",
				reply:   textReply(http.StatusBadRequest, "bad json", nil),
				wantErr: ErrBadRequest,
			},
			{
				name:    "http_401_unauthorized",
				reply:   textReply(http.StatusUnauthorized, "unauthorized", nil),
				wantErr: ErrAuthentication,
			},
			{
				name:    "http_403_forbidden",
				reply:   textReply(http.StatusForbidden, "forbidden", nil),
				wantErr: ErrAuthentication,
			},
			{
				name:    "http_422_bad_question",
				reply:   textReply(http.StatusUnprocessableEntity, "bad question", nil),
				wantErr: ErrBadRequest,
			},
			{
				name:    "http_500_internal_error",
				reply:   textReply(http.StatusInternalServerError, "boom", nil),
				wantErr: ErrAPIStatus,
			},
		}

		for _, tc := range permanentCases {
			t.Run(tc.name, func(t *testing.T) {
				tr := newMockTransport(tc.reply)
				_, err := newTestClassifier(t, tr, Config{MaxAttempts: 3}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"urgent": urgentQuestion,
				})
				if err == nil {
					t.Fatal("Classify() succeeded, want error")
				}
				if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
					t.Errorf("Classify() error = %v, want errors.Is(%v)", err, tc.wantErr)
				}
				if got := tr.count(); got != 1 {
					t.Errorf("request count = %d, want 1 (no retry)", got)
				}
			})
		}
	})

	t.Run("caller_context_cancellation_before_and_during_request_and_backoff", func(t *testing.T) {
		// Pre-canceled context sends 0 requests.
		preCtx, preCancel := context.WithCancel(t.Context())
		preCancel()
		trPre := newMockTransport()
		_, err := newTestClassifier(t, trPre, Config{MaxAttempts: 3}).Classify(preCtx, "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-canceled Classify() error = %v, want context.Canceled", err)
		}
		if got := trPre.count(); got != 0 {
			t.Errorf("pre-canceled request count = %d, want 0", got)
		}

		// Canceled while request is in flight returns context.Canceled without retry.
		midCtx, midCancel := context.WithCancel(t.Context())
		go func() {
			time.Sleep(15 * time.Millisecond)
			midCancel()
		}()
		trMid := newMockTransport(queuedReply{wait: time.Second})
		_, err = newTestClassifier(t, trMid, Config{MaxAttempts: 3}).Classify(midCtx, "hi", map[string]classifier.Question{"urgent": urgentQuestion})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("mid-flight Classify() error = %v, want context.Canceled", err)
		}
		if got := trMid.count(); got != 1 {
			t.Errorf("mid-flight request count = %d, want 1", got)
		}

		// Canceled during retry sleep stops retries and keeps the error that
		// triggered the retry, whether the sleep reports the cancellation itself
		// or returns nil because its timer fired in the same instant.
		dialErr := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
		for _, tc := range []struct {
			name     string
			sleepErr func(ctx context.Context) error
			reply    queuedReply
			wantErr  error
			check    func(error) bool
		}{
			{
				name:     "sleep_reports_cancellation",
				sleepErr: func(ctx context.Context) error { return ctx.Err() },
				reply:    textReply(http.StatusTooManyRequests, "busy", nil),
				wantErr:  ErrRateLimited,
				check: func(err error) bool {
					var apiErr *APIError
					return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests
				},
			},
			{
				name:     "sleep_returns_nil_after_cancellation",
				sleepErr: func(context.Context) error { return nil },
				reply:    textReply(statusOverloaded, "busy", nil),
				wantErr:  ErrOverloaded,
				check: func(err error) bool {
					var apiErr *APIError
					return errors.As(err, &apiErr) && apiErr.StatusCode == statusOverloaded
				},
			},
			{
				name:     "transport_error_kept",
				sleepErr: func(ctx context.Context) error { return ctx.Err() },
				reply:    queuedReply{err: dialErr},
				wantErr:  dialErr,
				check: func(err error) bool {
					var opErr *net.OpError
					return errors.As(err, &opErr) && opErr.Op == "dial"
				},
			},
		} {
			sleepCtx, sleepCancel := context.WithCancel(t.Context())
			trSleep := newMockTransport(tc.reply)
			clf := newTestClassifier(t, trSleep, Config{MaxAttempts: 3})
			clf.sleep = func(ctx context.Context, _ time.Duration) error {
				sleepCancel()
				return tc.sleepErr(ctx)
			}
			_, err = clf.Classify(sleepCtx, "hi", map[string]classifier.Question{"urgent": urgentQuestion})
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("%s: Classify() error = %v, want context.Canceled", tc.name, err)
			}
			if !errors.Is(err, tc.wantErr) || !tc.check(err) {
				t.Errorf("%s: Classify() error = %v, want it to also wrap the error that triggered the retry (%v)", tc.name, err, tc.wantErr)
			}
			if got := trSleep.count(); got != 1 {
				t.Errorf("%s: request count = %d, want 1", tc.name, got)
			}
		}
	})

	t.Run("retry_backoff_grows_exponentially_up_to_cap", func(t *testing.T) {
		for attempt := range 10 {
			minExpected := min(initialBackoff*(1<<attempt), maxBackoff)
			d := retryBackoff(attempt)
			if d < minExpected || d > maxBackoff {
				t.Fatalf("retryBackoff(%d) = %v, want within [%v, %v]", attempt, d, minExpected, maxBackoff)
			}
		}

		// Verify sendWithRetry passes advancing attempt indices (0, 1) to sleep.
		var sleeps []time.Duration
		tr := newMockTransport(
			textReply(statusOverloaded, "busy", nil),
			textReply(statusOverloaded, "busy", nil),
			okReply(t, map[string]any{"urgent": map[string]any{"type": "noul", "noul": 0.5}}, "jev-1.13.0"),
		)
		clf := newTestClassifier(t, tr, Config{MaxAttempts: 3})
		clf.sleep = func(_ context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		}
		if _, err := clf.Classify(t.Context(), "hi", map[string]classifier.Question{"urgent": urgentQuestion}); err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		if len(sleeps) != 2 || sleeps[0] < initialBackoff || sleeps[1] < 2*initialBackoff {
			t.Errorf("recorded sleep durations = %v, want 2 sleeps with sleeps[1] >= 1s", sleeps)
		}
	})
}

func TestProbabilityClampingAndResponseValidation(t *testing.T) {
	t.Run("clamps_rounding_drift_within_tolerance", func(t *testing.T) {
		confZero := 0.0
		confOne := 1.0
		cases := []struct {
			name string
			q    classifier.Question
			raw  map[string]any
			want classifier.Answer
		}{
			{
				name: "binary",
				q:    urgentQuestion,
				raw:  map[string]any{"type": "noul", "noul": 1.0000001, "confidence": -1e-9},
				want: classifier.BinaryAnswer{Probability: 1.0, Confidence: &confZero},
			},
			{
				name: "choice",
				q:    routeQuestion,
				raw: map[string]any{
					"type":          "choice",
					"choice":        "billing",
					"probabilities": map[string]any{"billing": 1.0000001, "technical": -1e-9},
					"confidence":    1.0000001,
				},
				want: classifier.ChoiceAnswer{
					Choice:        "billing",
					Probabilities: map[string]float64{"billing": 1.0, "technical": 0.0},
					Confidence:    &confOne,
				},
			},
			{
				name: "score",
				q:    frustrationQuestion,
				raw: map[string]any{
					"type":          "score",
					"score":         2.0000001,
					"probabilities": map[string]any{"0": 1.0000001, "1": -1e-9, "2": 0.0},
				},
				want: classifier.ScoreAnswer{
					Score:         2.0,
					Probabilities: map[int]float64{0: 1.0, 1: 0.0, 2: 0.0},
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				tr := newMockTransport(okReply(t, map[string]any{"q": tc.raw}, "jev-1.13.0"))
				resp, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"q": tc.q,
				})
				if err != nil {
					t.Fatalf("Classify() unexpected error: %v", err)
				}
				if diff := cmp.Diff(tc.want, resp.Answers["q"]); diff != "" {
					t.Errorf("answer mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("rejects_malformed_responses", func(t *testing.T) {
		malformedBodies := []struct {
			name string
			q    classifier.Question
			body string
		}{
			{name: "top_level_array", q: urgentQuestion, body: `["not","an","object"]`},
			{name: "empty_body", q: urgentQuestion, body: ``},
			{name: "trailing_json_tokens", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":0.5}}}{}`},
			{name: "missing_answers_field", q: urgentQuestion, body: `{"model":"jev-1.13.0"}`},
			{name: "non_object_answers_field", q: urgentQuestion, body: `{"answers":[]}`},
			{name: "non_string_model_field", q: urgentQuestion, body: `{"model":123,"answers":{"q":{"type":"noul","noul":0.5}}}`},
			{name: "missing_question_answer", q: urgentQuestion, body: `{"answers":{}}`},
			{name: "wrong_answer_type", q: urgentQuestion, body: `{"answers":{"q":{"type":"choice","choice":"x"}}}`},
			{name: "binary_missing_noul", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul"}}}`},
			{name: "choice_missing_choice", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","probabilities":{"billing":1.0,"technical":0.0}}}}`},
			{name: "choice_non_string_choice", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":123,"probabilities":{"billing":1.0,"technical":0.0}}}}`},
			{name: "choice_missing_probabilities", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing"}}}`},
			{name: "choice_non_object_probabilities", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing","probabilities":1}}}`},
			{name: "score_missing_score", q: frustrationQuestion, body: `{"answers":{"q":{"type":"score","probabilities":{"0":1.0,"1":0.0,"2":0.0}}}}`},
			{name: "score_missing_probabilities", q: frustrationQuestion, body: `{"answers":{"q":{"type":"score","score":0.0}}}`},
			{name: "bool_probability", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":true}}}`},
			{name: "string_probability", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":"0.5"}}}`},
			{name: "null_probability", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":null}}}`},
			{name: "nan_syntax", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":NaN}}}`},
			{name: "float_overflow_infinity", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":1e400}}}`},
			{name: "binary_out_of_range", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":70}}}`},
			{name: "confidence_out_of_range", q: urgentQuestion, body: `{"answers":{"q":{"type":"noul","noul":0.5,"confidence":1.5}}}`},
			{name: "choice_prob_negative", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing","probabilities":{"billing":-0.2,"technical":1.0}}}}`},
			{name: "score_above_top_level", q: frustrationQuestion, body: `{"answers":{"q":{"type":"score","score":3.0,"probabilities":{"0":0.0,"1":0.0,"2":1.0}}}}`},
			{name: "choice_label_outside_criteria", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"sales","probabilities":{"billing":0.5,"technical":0.5}}}}`},
			{name: "choice_missing_label_prob", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing","probabilities":{"billing":1.0}}}}`},
			{name: "choice_extra_label_prob", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing","probabilities":{"billing":0.8,"technical":0.1,"sales":0.1}}}}`},
			{name: "choice_wrong_label_same_count", q: routeQuestion, body: `{"answers":{"q":{"type":"choice","choice":"billing","probabilities":{"billing":0.8,"sales":0.2}}}}`},
			{name: "score_missing_level_prob", q: frustrationQuestion, body: `{"answers":{"q":{"type":"score","score":1.0,"probabilities":{"0":0.0,"1":1.0}}}}`},
		}

		for _, tc := range malformedBodies {
			t.Run(tc.name, func(t *testing.T) {
				tr := newMockTransport(textReply(http.StatusOK, tc.body, nil))
				_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", map[string]classifier.Question{
					"q": tc.q,
				})
				if !errors.Is(err, ErrMalformedResponse) {
					t.Fatalf("Classify() error = %v, want errors.Is(ErrMalformedResponse)", err)
				}
			})
		}
	})

	t.Run("rejects_oversize_response_body", func(t *testing.T) {
		// Pad a syntactically valid 200 JSON payload past 16 MiB so removing the
		// size cap would let Classify succeed rather than fail JSON parsing.
		prefix := []byte(`{"model":"jev-1.13.0","answers":{"urgent":{"type":"noul","noul":0.5}}}`)
		oversize := append(prefix, bytes.Repeat([]byte(" "), (16<<20)-len(prefix)+1)...)
		tr := newMockTransport(queuedReply{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader(oversize)),
			},
		})
		_, err := newTestClassifier(t, tr, Config{}).Classify(t.Context(), "hi", map[string]classifier.Question{
			"urgent": urgentQuestion,
		})
		if !errors.Is(err, ErrMalformedResponse) {
			t.Fatalf("Classify() error = %v, want errors.Is(ErrMalformedResponse)", err)
		}
		if got := tr.count(); got != 1 {
			t.Errorf("request count = %d, want 1", got)
		}
	})
}

func TestDefaultHTTPClientTimeoutRedirectsAndPointerQuestions(t *testing.T) {
	t.Run("applies_default_timeout_and_accepts_pointer_questions", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"model": "jev-1.13.0",
				"answers": {
					"route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.2}},
					"urgent": {"type": "noul", "noul": 0.4},
					"frustration": {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0}}
				}
			}`)
		}))
		defer srv.Close()

		clf, err := New(Config{
			APIKey:  "test-key",
			BaseURL: srv.URL,
		})
		if err != nil {
			t.Fatalf("New() unexpected error: %v", err)
		}
		if _, defaultTimeout := clf.clientForAttempt(); defaultTimeout != DefaultTimeout {
			t.Errorf("default per-attempt timeout = %v, want %v", defaultTimeout, DefaultTimeout)
		}

		routeCopy := routeQuestion
		urgentCopy := urgentQuestion
		frustrationCopy := frustrationQuestion
		resp, err := clf.Classify(t.Context(), "ticket", map[string]classifier.Question{
			"route":       &routeCopy,
			"urgent":      &urgentCopy,
			"frustration": &frustrationCopy,
		})
		if err != nil {
			t.Fatalf("Classify() unexpected error: %v", err)
		}
		if got := resp.Answers["route"].(classifier.ChoiceAnswer).Choice; got != "billing" {
			t.Errorf("Choice = %q, want %q", got, "billing")
		}
	})

	t.Run("default_client_does_not_follow_3xx_redirects", func(t *testing.T) {
		var redirectTargetHits atomic.Int32
		targetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			redirectTargetHits.Add(1)
			w.WriteHeader(http.StatusOK)
		}))
		defer targetSrv.Close()

		redirectSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, targetSrv.URL+"/steal", http.StatusTemporaryRedirect)
		}))
		defer redirectSrv.Close()

		clf, err := New(Config{
			APIKey:      "secret-key",
			BaseURL:     redirectSrv.URL,
			MaxAttempts: 1,
		})
		if err != nil {
			t.Fatalf("New() unexpected error: %v", err)
		}

		_, err = clf.Classify(t.Context(), "sensitive ticket", map[string]classifier.Question{
			"urgent": urgentQuestion,
		})
		if !errors.Is(err, ErrAPIStatus) {
			t.Fatalf("Classify() error = %v, want errors.Is(ErrAPIStatus) on 307 redirect", err)
		}
		if hits := redirectTargetHits.Load(); hits != 0 {
			t.Errorf("redirect target hits = %d, want 0", hits)
		}
	})
}

func TestSleepWithContextAndHeaderSanitization(t *testing.T) {
	const wait = 20 * time.Millisecond
	start := time.Now()
	if err := sleepWithContext(t.Context(), wait); err != nil {
		t.Fatalf("sleepWithContext(%v) error = %v", wait, err)
	}
	if elapsed := time.Since(start); elapsed < wait {
		t.Fatalf("sleepWithContext(%v) returned after %v, want at least %v", wait, elapsed, wait)
	}
	start = time.Now()
	if err := sleepWithContext(t.Context(), 0); err != nil {
		t.Fatalf("sleepWithContext(0) error = %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("sleepWithContext(0) took %v, want an immediate return", elapsed)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := sleepWithContext(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("sleepWithContext(canceled) error = %v, want context.Canceled", err)
	}
	if err := sleepWithContext(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("sleepWithContext(canceled, 0) error = %v, want context.Canceled", err)
	}

	// Unsafe characters or overlong (>128-byte) X-Request-Id values are dropped.
	tr := newMockTransport(
		textReply(http.StatusBadRequest, "bad", map[string]string{
			"X-Request-Id": "req id with spaces\nand\rnewlines",
			"Request-Id":   "fallback_req_42",
		}),
		textReply(http.StatusBadRequest, "bad", map[string]string{
			"X-Request-Id": strings.Repeat("a", 129),
		}),
	)
	_, err := newTestClassifier(t, tr, Config{MaxAttempts: 1}).Classify(t.Context(), "hi", map[string]classifier.Question{
		"urgent": urgentQuestion,
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || !errors.Is(err, ErrBadRequest) || apiErr.RequestID != "fallback_req_42" {
		t.Fatalf("APIError = %+v, want ErrBadRequest with fallback RequestID fallback_req_42", apiErr)
	}

	_, err = newTestClassifier(t, tr, Config{MaxAttempts: 1}).Classify(t.Context(), "hi", map[string]classifier.Question{
		"urgent": urgentQuestion,
	})
	apiErr = nil
	if !errors.As(err, &apiErr) || apiErr.RequestID != "" {
		t.Fatalf("overlong X-Request-Id APIError = %+v, want empty RequestID", apiErr)
	}
}
