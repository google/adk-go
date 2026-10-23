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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
	"unsafe"

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/go-cmp/cmp"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/httprr"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// To re-record the httprr cassettes against live endpoints:
//
//   - For TestGenerate_Anthropic and TestAgentRunner_MultiTurnImageAndTool:
//     export ANTHROPIC_API_KEY="sk-ant-..."
//   - For TestGenerate_Vertex:
//     export GOOGLE_CLOUD_PROJECT="your-project"
//     export GOOGLE_CLOUD_LOCATION="us-east5"
//     gcloud auth application-default login
//
// Then run:
//
//   go generate ./model/anthropicmodel/...
//
//go:generate go test -httprecord=testdata/TestGenerate_Anthropic\.httprr -run=^TestGenerate_Anthropic$
//go:generate go test -httprecord=testdata/TestGenerate_Vertex\.httprr -run=^TestGenerate_Vertex$
//go:generate go test -httprecord=testdata/TestAgentRunner_MultiTurnImageAndTool\.httprr -run=^TestAgentRunner_MultiTurnImageAndTool$

// samplePNG1x1 is a valid 1x1 blue PNG image used across multimodal tests.
var samplePNG1x1 = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
	0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
	0xde, 0x00, 0x00, 0x00, 0x10, 0x49, 0x44, 0x41,
	0x54, 0x78, 0x9c, 0x62, 0x62, 0x60, 0xf8, 0x0f,
	0x08, 0x00, 0x00, 0xff, 0xff, 0x01, 0x0b, 0x01,
	0x02, 0xd6, 0x95, 0x43, 0xa8, 0x00, 0x00, 0x00,
	0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60,
	0x82,
}

type staticTokenSource struct {
	token string
}

func (s staticTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{
		AccessToken: s.token,
		TokenType:   "Bearer",
		Expiry:      time.Now().Add(time.Hour),
	}, nil
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestSamplePNG1x1_DecodesAsValidPNG(t *testing.T) {
	img, err := png.Decode(bytes.NewReader(samplePNG1x1))
	if err != nil {
		t.Fatalf("png.Decode(samplePNG1x1) error = %v", err)
	}
	if img.Bounds().Dx() != 1 || img.Bounds().Dy() != 1 {
		t.Fatalf("samplePNG1x1 bounds = %v, want 1x1", img.Bounds())
	}
}

func scrubAnthropicRequest(req *http.Request) error {
	for k := range req.Header {
		lower := strings.ToLower(k)
		if lower == "authorization" ||
			lower == "x-api-key" ||
			lower == "user-agent" ||
			lower == "x-goog-api-client" ||
			strings.HasPrefix(lower, "x-stainless-") {
			delete(req.Header, k)
		}
	}

	// Canonicalize any Vertex project ID in the request path to "test-project".
	if idx := strings.Index(req.URL.Path, "/projects/"); idx >= 0 {
		rest := req.URL.Path[idx+len("/projects/"):]
		if slash := strings.IndexByte(rest, '/'); slash > 0 {
			req.URL.Path = req.URL.Path[:idx] + "/projects/test-project" + rest[slash:]
			req.RequestURI = ""
		}
	}

	if req.Body != nil {
		if b, ok := req.Body.(*httprr.Body); ok && len(b.Data) > 0 {
			var buf bytes.Buffer
			if err := json.Compact(&buf, b.Data); err == nil {
				b.Data = buf.Bytes()
			}
		}
	}
	return nil
}

func scrubAnthropicResponse(buf *bytes.Buffer) error {
	lines := strings.Split(buf.String(), "\n")
	out := make([]string, 0, len(lines))
	inHeaders := true
	for _, line := range lines {
		if inHeaders {
			trimmed := strings.TrimRight(line, "\r")
			if trimmed == "" {
				inHeaders = false
				out = append(out, line)
				continue
			}
			lower := strings.ToLower(trimmed)
			if strings.HasPrefix(lower, "anthropic-organization-id:") ||
				strings.HasPrefix(lower, "anthropic-workspace-id:") ||
				strings.HasPrefix(lower, "set-cookie:") ||
				strings.HasPrefix(lower, "request-id:") ||
				strings.HasPrefix(lower, "x-request-id:") {
				continue
			}
		}
		out = append(out, line)
	}
	buf.Reset()
	buf.WriteString(strings.Join(out, "\n"))
	return nil
}

func openHTTPRR(t *testing.T, rrfile string) *httprr.RecordReplay {
	t.Helper()
	rr, err := httprr.Open(rrfile, http.DefaultTransport)
	if err != nil {
		t.Fatalf("httprr.Open(%q) failed: %v", rrfile, err)
	}
	rr.ScrubReq(scrubAnthropicRequest)
	rr.ScrubResp(scrubAnthropicResponse)
	t.Cleanup(func() {
		_ = rr.Close()
	})
	return rr
}

func TestGenerate_Anthropic(t *testing.T) {
	rrfile := filepath.Join("testdata", "TestGenerate_Anthropic.httprr")
	rr := openHTTPRR(t, rrfile)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if !rr.Recording() || apiKey == "" {
		apiKey = "test-anthropic-key"
	}

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		Backend:    anthropicmodel.BackendAnthropic,
		APIKey:     apiKey,
		HTTPClient: rr.Client(),
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("What is the capital of France? Reply with one word."),
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("Reply with the city name only.", genai.RoleUser),
			Temperature:       new(float32),
		},
	}

	t.Run("non_streaming", func(t *testing.T) {
		var responses []*model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent(stream=false) error = %v", err)
			}
			responses = append(responses, resp)
		}
		if len(responses) != 1 {
			t.Fatalf("got %d responses, want 1", len(responses))
		}
		got := responses[0]
		if got.Partial || !got.TurnComplete {
			t.Errorf("got Partial=%v TurnComplete=%v, want Partial=false TurnComplete=true", got.Partial, got.TurnComplete)
		}
		if got.FinishReason != genai.FinishReasonStop {
			t.Errorf("FinishReason = %v, want %v", got.FinishReason, genai.FinishReasonStop)
		}
		if got.Content == nil || len(got.Content.Parts) != 1 || !strings.Contains(got.Content.Parts[0].Text, "Paris") {
			t.Errorf("unexpected Content = %+v", got.Content)
		}
		if got.UsageMetadata == nil || got.UsageMetadata.PromptTokenCount <= 0 || got.UsageMetadata.CandidatesTokenCount <= 0 {
			t.Errorf("unexpected UsageMetadata = %+v", got.UsageMetadata)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var partialTexts []string
		var finalResp *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), req, true) {
			if err != nil {
				t.Fatalf("GenerateContent(stream=true) error = %v", err)
			}
			if resp.Partial {
				if resp.TurnComplete {
					t.Errorf("partial response had TurnComplete=true")
				}
				for _, p := range resp.Content.Parts {
					partialTexts = append(partialTexts, p.Text)
				}
			} else {
				if finalResp != nil {
					t.Fatalf("received multiple final responses")
				}
				finalResp = resp
			}
		}
		if len(partialTexts) == 0 {
			t.Fatalf("expected at least one partial streaming chunk")
		}
		if finalResp == nil {
			t.Fatalf("expected a final aggregated response")
		}
		if !finalResp.TurnComplete || finalResp.Partial {
			t.Errorf("finalResp Partial=%v TurnComplete=%v, want false/true", finalResp.Partial, finalResp.TurnComplete)
		}
		joined := strings.Join(partialTexts, "")
		if finalResp.Content == nil || len(finalResp.Content.Parts) != 1 || finalResp.Content.Parts[0].Text != joined {
			t.Errorf("final text %q does not match concatenated partial text %q", finalResp.Content.Parts[0].Text, joined)
		}
	})
}

func TestGenerate_Vertex(t *testing.T) {
	rrfile := filepath.Join("testdata", "TestGenerate_Vertex.httprr")
	rr := openHTTPRR(t, rrfile)

	cfg := anthropicmodel.ClientConfig{
		Backend:    anthropicmodel.BackendVertex,
		Project:    "test-project",
		Location:   "us-east5",
		HTTPClient: rr.Client(),
	}
	if !rr.Recording() {
		cfg.TokenSource = staticTokenSource{token: "test-vertex-oauth-token"}
	} else {
		if p := os.Getenv("GOOGLE_CLOUD_PROJECT"); p != "" {
			cfg.Project = p
		}
		if l := os.Getenv("GOOGLE_CLOUD_LOCATION"); l != "" {
			cfg.Location = l
		}
	}

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", cfg)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("What is 2 + 2? Reply with the digit only."),
		Config: &genai.GenerateContentConfig{
			Temperature: new(float32),
		},
	}

	var got *model.LLMResponse
	for resp, err := range m.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		got = resp
	}
	if got == nil || got.Content == nil || len(got.Content.Parts) != 1 {
		t.Fatalf("unexpected response: %+v", got)
	}
	if !strings.Contains(got.Content.Parts[0].Text, "4") {
		t.Errorf("got text %q, want string containing \"4\"", got.Content.Parts[0].Text)
	}
	if got.FinishReason != genai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want %v", got.FinishReason, genai.FinishReasonStop)
	}
}

func TestAgentRunner_MultiTurnImageAndTool(t *testing.T) {
	rrfile := filepath.Join("testdata", "TestAgentRunner_MultiTurnImageAndTool.httprr")
	rr := openHTTPRR(t, rrfile)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if !rr.Recording() || apiKey == "" {
		apiKey = "test-anthropic-key"
	}

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		Backend:    anthropicmodel.BackendAnthropic,
		APIKey:     apiKey,
		HTTPClient: rr.Client(),
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	type paletteArgs struct {
		Color string `json:"color" jsonschema:"Dominant color name"`
	}
	type paletteResult struct {
		Color   string `json:"color"`
		HexCode string `json:"hex_code"`
	}

	toolCalls := 0
	paletteTool, err := functiontool.New(functiontool.Config{
		Name:        "get_palette_info",
		Description: "Returns the canonical hex code for a color name.",
	}, func(_ agent.Context, args paletteArgs) (paletteResult, error) {
		toolCalls++
		return paletteResult{
			Color:   args.Color,
			HexCode: "#0055FF",
		}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "vision_palette_agent",
		Description: "Inspects an image and looks up its palette info.",
		Model:       m,
		Instruction: "Identify the color of the image and call get_palette_info to report its hex code.",
		Tools:       []tool.Tool{paletteTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	tr := testutil.NewTestAgentRunner(t, a)
	const sessionID = "multi_turn_session"

	// Turn 1: send an inline PNG image + text prompt; agent calls get_palette_info and answers.
	turn1Content := &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			genai.NewPartFromBytes(samplePNG1x1, "image/png"),
			genai.NewPartFromText("What color is this image and what does get_palette_info return for it?"),
		},
	}

	turn1Events, err := testutil.CollectEvents(tr.RunContent(t, sessionID, turn1Content))
	if err != nil {
		t.Fatalf("Turn 1 RunContent() error = %v", err)
	}
	if toolCalls != 1 {
		t.Fatalf("Turn 1 toolCalls = %d, want 1", toolCalls)
	}
	if len(turn1Events) < 3 {
		t.Fatalf("Turn 1 emitted %d events, want at least 3 (tool call, tool response, final answer)", len(turn1Events))
	}
	lastTurn1 := turn1Events[len(turn1Events)-1]
	if lastTurn1.LLMResponse.Content == nil || len(lastTurn1.LLMResponse.Content.Parts) == 0 ||
		!strings.Contains(lastTurn1.LLMResponse.Content.Parts[0].Text, "#0055FF") {
		t.Errorf("Turn 1 final event content = %+v, want text containing #0055FF", lastTurn1.LLMResponse.Content)
	}

	// Verify stored session events after Turn 1.
	sessResp, err := tr.SessionService().Get(t.Context(), &session.GetRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("SessionService.Get() error = %v", err)
	}
	if sessResp.Session.Events().Len() < 4 {
		t.Fatalf("stored session has %d events after Turn 1, want at least 4", sessResp.Session.Events().Len())
	}

	// Turn 2: run a follow-up turn in the same session built from the stored events of Turn 1.
	turn2Texts, err := testutil.CollectTextParts(tr.Run(t, sessionID, "Summarize what you found in one short sentence."))
	if err != nil {
		t.Fatalf("Turn 2 Run() error = %v", err)
	}
	joinedTurn2 := strings.Join(turn2Texts, "")
	if !strings.Contains(joinedTurn2, "#0055FF") {
		t.Errorf("Turn 2 text = %q, want summary referencing #0055FF", joinedTurn2)
	}
}

// slogCaptureHandler records slog.Record entries for assertions in tests.
type slogCaptureHandler struct {
	mu      sync.Mutex
	records []capturedLogRecord
}

type capturedLogRecord struct {
	Level   slog.Level
	Message string
	Attrs   map[string]string
}

func (h *slogCaptureHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *slogCaptureHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	attrs := make(map[string]string)
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.String()
		return true
	})
	h.records = append(h.records, capturedLogRecord{
		Level:   r.Level,
		Message: r.Message,
		Attrs:   attrs,
	})
	return nil
}

func (h *slogCaptureHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *slogCaptureHandler) WithGroup(string) slog.Handler      { return h }

func (h *slogCaptureHandler) Snapshot() []capturedLogRecord {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]capturedLogRecord, len(h.records))
	copy(out, h.records)
	return out
}

func TestNewModel_AnthropicCredentialsAndConfig(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	t.Run("missing_credentials_fails", func(t *testing.T) {
		_, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !errors.Is(err, anthropicmodel.ErrAuthentication) {
			t.Fatalf("NewModel() error = %v, want wrapping ErrInvalidConfig and ErrAuthentication", err)
		}
	})

	t.Run("canceled_context_fails", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		_, err := anthropicmodel.NewModel(ctx, "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{APIKey: "k"})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("NewModel(canceled ctx) error = %v, want context.Canceled", err)
		}
	})

	t.Run("env_api_key_and_precedence", func(t *testing.T) {
		var gotAPIKey, gotAuthHeader string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAPIKey = r.Header.Get("X-Api-Key")
			gotAuthHeader = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
		}))
		defer srv.Close()

		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")

		// Custom BaseURL WITHOUT explicit credential must be rejected even when env vars are set.
		_, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			BaseURL: srv.URL,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "custom BaseURL on BackendAnthropic") {
			t.Fatalf("NewModel(BaseURL without explicit cred) error = %v, want ErrInvalidConfig mentioning custom BaseURL on BackendAnthropic", err)
		}

		// Custom BaseURL in RequestOptions WITHOUT explicit credential must also be rejected.
		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			RequestOptions: []option.RequestOption{option.WithBaseURL(srv.URL)},
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "custom BaseURL on BackendAnthropic") {
			t.Fatalf("NewModel(RequestOptions WithBaseURL without explicit cred) error = %v, want ErrInvalidConfig mentioning custom BaseURL on BackendAnthropic", err)
		}

		// Explicit APIKey with custom BaseURL succeeds and uses explicit APIKey over env.
		m, err := anthropicmodel.NewModel(t.Context(), "anthropic/claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "explicit-api-key",
			BaseURL: srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}
		if m.Name() != "anthropic/claude-sonnet-4-5-20250929" {
			t.Errorf("Name() = %q, want %q", m.Name(), "anthropic/claude-sonnet-4-5-20250929")
		}
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if gotAPIKey != "explicit-api-key" {
			t.Errorf("X-Api-Key = %q, want %q", gotAPIKey, "explicit-api-key")
		}
		if gotAuthHeader != "" {
			t.Errorf("Authorization = %q, want empty", gotAuthHeader)
		}
	})

	t.Run("env_credentials_without_base_url_and_anthropic_base_url_env", func(t *testing.T) {
		var gotAPIKey, gotAuthHeader, gotHost string
		fakeClient := &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				gotAPIKey = r.Header.Get("X-Api-Key")
				gotAuthHeader = r.Header.Get("Authorization")
				gotHost = r.URL.Host
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)),
				}, nil
			}),
		}

		// 1. Both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN set -> ANTHROPIC_API_KEY wins.
		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")
		t.Setenv("ANTHROPIC_BASE_URL", "")
		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			HTTPClient: fakeClient,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if gotAPIKey != "env-api-key" || gotAuthHeader != "" {
			t.Errorf("got X-Api-Key=%q Authorization=%q, want env-api-key and empty", gotAPIKey, gotAuthHeader)
		}

		// 2. Only ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL set -> Bearer header and env base URL used.
		t.Setenv("ANTHROPIC_API_KEY", "")
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")
		t.Setenv("ANTHROPIC_BASE_URL", "https://gateway.internal.example.com")
		m2, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			HTTPClient: fakeClient,
		})
		if err != nil {
			t.Fatalf("NewModel(ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL) error = %v", err)
		}
		for _, err := range m2.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if gotAPIKey != "" || gotAuthHeader != "Bearer env-auth-token" {
			t.Errorf("got X-Api-Key=%q Authorization=%q, want empty and Bearer env-auth-token", gotAPIKey, gotAuthHeader)
		}
		if gotHost != "gateway.internal.example.com" {
			t.Errorf("got Host=%q, want gateway.internal.example.com", gotHost)
		}
	})

	t.Run("explicit_auth_token", func(t *testing.T) {
		var gotAuthHeader string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotAuthHeader = r.Header.Get("Authorization")
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
		}))
		defer srv.Close()

		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			AuthToken: "my-bearer-token",
			BaseURL:   srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if gotAuthHeader != "Bearer my-bearer-token" {
			t.Errorf("Authorization = %q, want %q", gotAuthHeader, "Bearer my-bearer-token")
		}
	})

	t.Run("invalid_client_configs", func(t *testing.T) {
		negOne := -1
		cases := []struct {
			name      string
			modelName string
			cfg       anthropicmodel.ClientConfig
		}{
			{
				name:      "empty_model_name",
				modelName: "   ",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k"},
			},
			{
				name:      "invalid_model_slash",
				modelName: "foo/bar/baz",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k"},
			},
			{
				name:      "vertex_resource_on_anthropic_backend",
				modelName: "projects/p/locations/l/publishers/anthropic/models/claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{Backend: anthropicmodel.BackendAnthropic, APIKey: "k"},
			},
			{
				name:      "unknown_backend",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{Backend: "bedrock", APIKey: "k"},
			},
			{
				name:      "negative_max_tokens",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", MaxTokens: -5},
			},
			{
				name:      "negative_max_retries",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", MaxRetries: &negOne},
			},
			{
				name:      "invalid_effort",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", Effort: "extreme"},
			},
			{
				name:      "invalid_prompt_cache_ttl",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", PromptCache: &anthropicmodel.PromptCacheConfig{TTL: "10m"}},
			},
			{
				name:      "invalid_base_url_scheme",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", BaseURL: "ftp://example.com"},
			},
			{
				name:      "whitespace_base_url",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", BaseURL: "   "},
			},
			{
				name:      "nil_request_option",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", RequestOptions: []option.RequestOption{nil}},
			},
			{
				name:      "invalid_request_option_base_url",
				modelName: "claude-sonnet-4-5-20250929",
				cfg:       anthropicmodel.ClientConfig{APIKey: "k", RequestOptions: []option.RequestOption{option.WithBaseURL("ftp://example.com")}},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := anthropicmodel.NewModel(t.Context(), tc.modelName, tc.cfg)
				if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
					t.Errorf("NewModel(%q) error = %v, want ErrInvalidConfig", tc.modelName, err)
				}
			})
		}

		t.Run("unparsable_request_option_base_url_does_not_retry_or_leak_userinfo", func(t *testing.T) {
			// 1. Valid WithBaseURL combined with WithMaxRetries(3) must not cause the probe middleware
			// to sleep through SDK retry backoffs.
			probeCtx, cancelProbe := context.WithTimeout(t.Context(), 200*time.Millisecond)
			defer cancelProbe()
			if _, err := anthropicmodel.NewModel(probeCtx, "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
				APIKey: "k",
				RequestOptions: []option.RequestOption{
					option.WithBaseURL("https://gateway.example.com"),
					option.WithMaxRetries(3),
				},
			}); err != nil {
				t.Fatalf("NewModel(valid WithBaseURL + WithMaxRetries(3)) error = %v, want nil without retry backoff", err)
			}

			// 2. Unparsable WithBaseURL must fail immediately with a fixed message without leaking raw URL userinfo.
			start := time.Now()
			_, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
				APIKey: "k",
				RequestOptions: []option.RequestOption{
					option.WithBaseURL("https://user:s3cret@gateway.example:badport/"),
					option.WithMaxRetries(3),
				},
			})
			elapsed := time.Since(start)
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
				t.Fatalf("NewModel() error = %v, want ErrInvalidConfig", err)
			}
			if elapsed > 500*time.Millisecond {
				t.Errorf("NewModel() took %v, want < 500ms (probe must not sleep through SDK retry backoffs)", elapsed)
			}
			errMsg := err.Error()
			if !strings.Contains(errMsg, "BaseURL in RequestOptions must be a valid http or https URL with a host") {
				t.Errorf("error = %q, want fixed BaseURL validation message", errMsg)
			}
			if strings.Contains(errMsg, "s3cret") || strings.Contains(errMsg, "gateway.example") {
				t.Errorf("error = %q leaked raw URL or userinfo credentials", errMsg)
			}
		})

		t.Run("default_http_client_sets_response_header_timeout", func(t *testing.T) {
			m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
				APIKey: "test-key",
			})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}
			optsVal := reflect.ValueOf(m).Elem().FieldByName("client").FieldByName("Options")
			opts := *(*[]option.RequestOption)(unsafe.Pointer(optsVal.UnsafeAddr()))
			if len(opts) == 0 {
				t.Fatal("anthropicModel.client.Options is empty")
			}
			rcType := reflect.TypeOf(opts).Elem().Method(0).Type.In(0).Elem()
			rcPtr := reflect.New(rcType)
			rcPtr.Elem().FieldByName("Request").Set(reflect.ValueOf(&http.Request{Header: make(http.Header)}))
			for i, opt := range opts {
				res := reflect.ValueOf(opt).MethodByName("Apply").Call([]reflect.Value{rcPtr})
				if !res[0].IsNil() {
					t.Fatalf("Options[%d].Apply error = %v", i, res[0].Interface())
				}
			}
			hcVal := rcPtr.Elem().FieldByName("HTTPClient")
			if hcVal.IsNil() {
				t.Fatal("RequestConfig.HTTPClient is nil after applying client options")
			}
			hc, ok := hcVal.Interface().(*http.Client)
			if !ok || hc == nil {
				t.Fatalf("RequestConfig.HTTPClient = %T, want non-nil *http.Client", hcVal.Interface())
			}
			tr, ok := hc.Transport.(*http.Transport)
			if !ok || tr == nil {
				t.Fatalf("default HTTPClient.Transport = %T, want non-nil *http.Transport", hc.Transport)
			}
			if tr == http.DefaultTransport {
				t.Error("default HTTPClient.Transport must clone http.DefaultTransport rather than mutating it in place")
			}
			if tr.ResponseHeaderTimeout != 10*time.Minute {
				t.Errorf("ResponseHeaderTimeout = %v, want %v", tr.ResponseHeaderTimeout, 10*time.Minute)
			}
			if defTr, ok := http.DefaultTransport.(*http.Transport); ok && defTr.ResponseHeaderTimeout != 0 {
				t.Errorf("http.DefaultTransport.ResponseHeaderTimeout = %v, want 0 (unmutated)", defTr.ResponseHeaderTimeout)
			}
		})
	})
}

func TestNewModel_VertexAndModelNames(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("CLOUD_ML_PROJECT_ID", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("CLOUD_ML_REGION", "")

	ts := staticTokenSource{token: "vertex-token"}

	t.Run("resource_name_resolution_and_headers", func(t *testing.T) {
		var gotPath, gotAPIClient, gotUserAgent, gotAuth string
		var gotBody map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotPath = r.URL.Path
			gotAPIClient = r.Header.Get("X-Goog-Api-Client")
			gotUserAgent = r.Header.Get("User-Agent")
			gotAuth = r.Header.Get("Authorization")
			data, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(data, &gotBody)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_v","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":4,"output_tokens":1}}`)
		}))
		defer srv.Close()

		resourceName := "projects/res-proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5-20250929"
		m, err := anthropicmodel.NewModel(t.Context(), resourceName, anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			TokenSource: ts,
			BaseURL:     srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("ping")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}

		wantPath := "/v1/projects/res-proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5-20250929:rawPredict"
		if gotPath != wantPath {
			t.Errorf("request path = %q, want %q", gotPath, wantPath)
		}
		if gotBody["anthropic_version"] != "vertex-2023-10-16" {
			t.Errorf("anthropic_version = %v, want vertex-2023-10-16", gotBody["anthropic_version"])
		}
		if !strings.Contains(gotAPIClient, "google-adk/") || !strings.Contains(gotAPIClient, "gl-go/") {
			t.Errorf("X-Goog-Api-Client = %q, want google-adk/... gl-go/...", gotAPIClient)
		}
		if !strings.Contains(gotUserAgent, "google-adk/") {
			t.Errorf("User-Agent = %q, want google-adk/...", gotUserAgent)
		}
		if gotAuth != "Bearer vertex-token" {
			t.Errorf("Authorization = %q, want Bearer vertex-token", gotAuth)
		}

		// Request-level resource name with mismatched project or location must fail before network call.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Model:    "projects/other-proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5-20250929",
			Contents: genai.Text("ping"),
		}, false) {
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
				t.Errorf("mismatched request project error = %v, want ErrInvalidConfig", err)
			}
		}
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Model:    "projects/res-proj/locations/europe-west1/publishers/anthropic/models/claude-sonnet-4-5-20250929",
			Contents: genai.Text("ping"),
		}, false) {
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
				t.Errorf("mismatched request location error = %v, want ErrInvalidConfig", err)
			}
		}
	})

	t.Run("vertex_validation_and_base_url_guard", func(t *testing.T) {
		// Mismatch between ClientConfig.Project and resource name in NewModel.
		_, err := anthropicmodel.NewModel(
			t.Context(),
			"projects/proj-a/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5-20250929",
			anthropicmodel.ClientConfig{
				Backend:     anthropicmodel.BackendVertex,
				Project:     "proj-b",
				Location:    "us-east5",
				TokenSource: ts,
			},
		)
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("mismatched Project error = %v, want ErrInvalidConfig", err)
		}

		// Mismatch between ClientConfig.Location and resource name in NewModel.
		_, err = anthropicmodel.NewModel(
			t.Context(),
			"projects/proj-a/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5-20250929",
			anthropicmodel.ClientConfig{
				Backend:     anthropicmodel.BackendVertex,
				Project:     "proj-a",
				Location:    "global",
				TokenSource: ts,
			},
		)
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("mismatched Location error = %v, want ErrInvalidConfig", err)
		}

		// Missing project or location.
		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			Location:    "us-east5",
			TokenSource: ts,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("missing Project error = %v, want ErrInvalidConfig", err)
		}

		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			Project:     "proj-a",
			TokenSource: ts,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("missing Location error = %v, want ErrInvalidConfig", err)
		}

		// Custom BaseURL on BackendVertex without explicit Credentials or TokenSource must fail
		// even when ambient Application Default Credentials are present in the environment.
		credFile := filepath.Join(t.TempDir(), "adc.json")
		const adcJSON = `{"type":"authorized_user","client_id":"test-client","client_secret":"test-secret","refresh_token":"test-refresh"}`
		if err := os.WriteFile(credFile, []byte(adcJSON), 0o600); err != nil {
			t.Fatalf("os.WriteFile() error = %v", err)
		}
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credFile)

		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:  anthropicmodel.BackendVertex,
			Project:  "proj-a",
			Location: "us-east5",
			BaseURL:  "https://custom-proxy.example.com",
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "custom BaseURL on BackendVertex") {
			t.Errorf("BaseURL without explicit Vertex creds error = %v, want ErrInvalidConfig mentioning custom BaseURL on BackendVertex", err)
		}

		// Custom BaseURL via RequestOptions on BackendVertex without explicit Credentials or TokenSource must fail.
		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:        anthropicmodel.BackendVertex,
			Project:        "proj-a",
			Location:       "us-east5",
			RequestOptions: []option.RequestOption{option.WithBaseURL("https://custom-proxy.example.com")},
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "custom BaseURL on BackendVertex") {
			t.Errorf("RequestOptions WithBaseURL without explicit Vertex creds error = %v, want ErrInvalidConfig mentioning custom BaseURL on BackendVertex", err)
		}

		// Invalid Vertex Location or Project (e.g. URL injection characters) must fail.
		_, err = anthropicmodel.NewModel(
			t.Context(),
			"projects/proj-a/locations/attacker.example#/publishers/anthropic/models/claude-sonnet-4-5-20250929",
			anthropicmodel.ClientConfig{
				Backend:     anthropicmodel.BackendVertex,
				TokenSource: ts,
			},
		)
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("malicious location in resource name error = %v, want ErrInvalidConfig", err)
		}

		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			Project:     "proj-a",
			Location:    "attacker.example#",
			TokenSource: ts,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("malicious ClientConfig.Location error = %v, want ErrInvalidConfig", err)
		}

		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			Project:     "bad#project",
			Location:    "us-east5",
			TokenSource: ts,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("malicious ClientConfig.Project error = %v, want ErrInvalidConfig", err)
		}

		// Credentials with nil TokenSource must fail.
		_, err = anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			Project:     "proj-a",
			Location:    "us-east5",
			Credentials: &google.Credentials{},
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("Credentials with nil TokenSource error = %v, want ErrInvalidConfig", err)
		}

		// Malformed resource name.
		_, err = anthropicmodel.NewModel(t.Context(), "projects/proj-a/locations/us-east5/models/claude", anthropicmodel.ClientConfig{
			Backend:     anthropicmodel.BackendVertex,
			TokenSource: ts,
		})
		if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
			t.Errorf("malformed resource name error = %v, want ErrInvalidConfig", err)
		}
	})

	t.Run("adc_credentials_detach_init_context", func(t *testing.T) {
		tokenSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"adc-refreshed-token","token_type":"Bearer","expires_in":3600}`)
		}))
		defer tokenSrv.Close()

		credFile := filepath.Join(t.TempDir(), "adc.json")
		adcJSON := fmt.Sprintf(`{"type":"authorized_user","client_id":"test-client","client_secret":"test-secret","refresh_token":"test-refresh","token_uri":%q}`, tokenSrv.URL)
		if err := os.WriteFile(credFile, []byte(adcJSON), 0o600); err != nil {
			t.Fatalf("os.WriteFile() error = %v", err)
		}
		t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credFile)

		var gotAuthHeader string
		apiHTTPClient := &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				gotAuthHeader = r.Header.Get("Authorization")
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Request:    r,
					Body:       io.NopCloser(strings.NewReader(`{"id":"msg_adc","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":4,"output_tokens":1}}`)),
				}, nil
			}),
		}

		initCtx, cancelInit := context.WithCancel(t.Context())
		m, err := anthropicmodel.NewModel(initCtx, "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			Backend:    anthropicmodel.BackendVertex,
			Project:    "proj-a",
			Location:   "us-east5",
			HTTPClient: apiHTTPClient,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}
		// Cancel the startup context before the first token refresh occurs.
		cancelInit()

		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("ping")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() after initCtx canceled error = %v", err)
			}
		}
		if gotAuthHeader != "Bearer adc-refreshed-token" {
			t.Errorf("Authorization = %q, want %q", gotAuthHeader, "Bearer adc-refreshed-token")
		}
	})
}

func TestToolsAndToolConfig(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		capturedBody = nil
		_ = json.Unmarshal(data, &capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_t","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"tool_use","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}},{"type":"tool_use","id":"toolu_2","name":"get_time","input":{"tz":"UTC"}}],"usage":{"input_tokens":15,"output_tokens":12}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	weatherTools := []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "get_weather",
					Description: "Look up weather.",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"city": {Type: genai.TypeString, Description: "City name"},
							"days": {
								Type:  genai.TypeArray,
								Items: &genai.Schema{Type: genai.TypeInteger},
							},
						},
						Required: []string{"city"},
					},
				},
				{
					Name:        "get_time",
					Description: "Look up time.",
					ParametersJsonSchema: map[string]any{
						"type": "OBJECT",
						"properties": map[string]any{
							"tz": map[string]any{"type": []any{"STRING", "NULL"}},
						},
					},
				},
			},
		},
	}

	t.Run("schema_normalization_and_parallel_tool_calls_and_id_sanitization", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Weather and time in Paris?", genai.RoleUser),
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{ID: "bad id#1", Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
						{FunctionCall: &genai.FunctionCall{ID: "", Name: "get_time", Args: map[string]any{"tz": "UTC"}}},
					},
				},
				{
					Role: genai.RoleUser,
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{ID: "bad id#1", Name: "get_weather", Response: map[string]any{"result": "Sunny 22C"}}},
						{FunctionResponse: &genai.FunctionResponse{ID: "", Name: "get_time", Response: map[string]any{"content": []any{map[string]any{"type": "text", "text": "12:00 UTC"}}}}},
					},
				},
			},
			Config: &genai.GenerateContentConfig{
				Tools: weatherTools,
				ToolConfig: &genai.ToolConfig{
					FunctionCallingConfig: &genai.FunctionCallingConfig{
						Mode:                 genai.FunctionCallingConfigModeAny,
						AllowedFunctionNames: []string{"get_weather"},
					},
				},
			},
		}

		var got *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
			got = resp
		}

		// Check normalized tool schemas in captured request body.
		toolsSlice, _ := capturedBody["tools"].([]any)
		if len(toolsSlice) != 2 {
			t.Fatalf("sent %d tools, want 2", len(toolsSlice))
		}
		firstTool, _ := toolsSlice[0].(map[string]any)
		inputSchema, _ := firstTool["input_schema"].(map[string]any)
		if inputSchema["type"] != "object" {
			t.Errorf("input_schema.type = %v, want \"object\"", inputSchema["type"])
		}
		props, _ := inputSchema["properties"].(map[string]any)
		cityProp, _ := props["city"].(map[string]any)
		if cityProp["type"] != "string" {
			t.Errorf("city.type = %v, want \"string\"", cityProp["type"])
		}
		daysProp, _ := props["days"].(map[string]any)
		if daysProp["type"] != "array" {
			t.Errorf("days.type = %v, want \"array\"", daysProp["type"])
		}

		// Check tool_choice mapped to {"type":"tool","name":"get_weather"}.
		tcMap, _ := capturedBody["tool_choice"].(map[string]any)
		if tcMap["type"] != "tool" || tcMap["name"] != "get_weather" {
			t.Errorf("tool_choice = %+v, want type=tool name=get_weather", tcMap)
		}

		// Check deterministic tool_use_id sanitization paired across assistant and user turns.
		msgs, _ := capturedBody["messages"].([]any)
		if len(msgs) != 3 {
			t.Fatalf("len(messages) = %d, want 3", len(msgs))
		}
		asstContent, _ := msgs[1].(map[string]any)["content"].([]any)
		userContent, _ := msgs[2].(map[string]any)["content"].([]any)
		tu0, _ := asstContent[0].(map[string]any)
		tu1, _ := asstContent[1].(map[string]any)
		tr0, _ := userContent[0].(map[string]any)
		tr1, _ := userContent[1].(map[string]any)
		if tu0["id"] != "toolu_fallback_0" || tr0["tool_use_id"] != "toolu_fallback_0" {
			t.Errorf("first pair IDs = (%v, %v), want toolu_fallback_0", tu0["id"], tr0["tool_use_id"])
		}
		if tu1["id"] != "toolu_fallback_1" || tr1["tool_use_id"] != "toolu_fallback_1" {
			t.Errorf("second pair IDs = (%v, %v), want toolu_fallback_1", tu1["id"], tr1["tool_use_id"])
		}

		// Check response returned both parallel FunctionCalls.
		if got == nil || got.Content == nil || len(got.Content.Parts) != 2 {
			t.Fatalf("got %+v, want 2 FunctionCall parts", got)
		}
		if got.Content.Parts[0].FunctionCall == nil || got.Content.Parts[0].FunctionCall.Name != "get_weather" {
			t.Errorf("part[0] = %+v, want get_weather FunctionCall", got.Content.Parts[0])
		}
		if got.Content.Parts[1].FunctionCall == nil || got.Content.Parts[1].FunctionCall.Name != "get_time" {
			t.Errorf("part[1] = %+v, want get_time FunctionCall", got.Content.Parts[1])
		}
	})

	t.Run("tool_config_modes_auto_any_none", func(t *testing.T) {
		modes := []struct {
			mode     genai.FunctionCallingConfigMode
			wantType string
		}{
			{mode: genai.FunctionCallingConfigModeAuto, wantType: "auto"},
			{mode: genai.FunctionCallingConfigModeAny, wantType: "any"},
			{mode: genai.FunctionCallingConfigModeNone, wantType: "none"},
		}
		for _, tc := range modes {
			req := &model.LLMRequest{
				Contents: genai.Text("hi"),
				Config: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: tc.mode},
					},
				},
			}
			for _, err := range m.GenerateContent(t.Context(), req, false) {
				if err != nil {
					t.Fatalf("mode %s error = %v", tc.mode, err)
				}
			}
			tcMap, _ := capturedBody["tool_choice"].(map[string]any)
			if tcMap["type"] != tc.wantType {
				t.Errorf("mode %s: tool_choice.type = %v, want %q", tc.mode, tcMap["type"], tc.wantType)
			}
		}
	})

	t.Run("rejected_tools_and_tool_configs", func(t *testing.T) {
		streamArgs := true
		budget := int32(1024)
		cases := []struct {
			name    string
			cfg     *genai.GenerateContentConfig
			wantSub string
		}{
			{
				name:    "google_search",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}}},
				wantSub: "GoogleSearch",
			},
			{
				name:    "google_search_retrieval",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{GoogleSearchRetrieval: &genai.GoogleSearchRetrieval{}}}},
				wantSub: "GoogleSearchRetrieval",
			},
			{
				name:    "retrieval",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{Retrieval: &genai.Retrieval{}}}},
				wantSub: "Retrieval",
			},
			{
				name:    "code_execution",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}},
				wantSub: "CodeExecution",
			},
			{
				name:    "enterprise_web_search",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{EnterpriseWebSearch: &genai.EnterpriseWebSearch{}}}},
				wantSub: "EnterpriseWebSearch",
			},
			{
				name:    "google_maps",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{GoogleMaps: &genai.GoogleMaps{}}}},
				wantSub: "GoogleMaps",
			},
			{
				name:    "computer_use",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{ComputerUse: &genai.ComputerUse{}}}},
				wantSub: "ComputerUse",
			},
			{
				name:    "url_context",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{URLContext: &genai.URLContext{}}}},
				wantSub: "URLContext",
			},
			{
				name:    "file_search",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{FileSearch: &genai.FileSearch{}}}},
				wantSub: "FileSearch",
			},
			{
				name: "parallel_ai_search_with_function_declarations",
				cfg: &genai.GenerateContentConfig{Tools: []*genai.Tool{{
					FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn"}},
					ParallelAISearch:     &genai.ToolParallelAISearch{},
				}}},
				wantSub: "ParallelAISearch",
			},
			{
				name: "mcp_servers_with_function_declarations",
				cfg: &genai.GenerateContentConfig{Tools: []*genai.Tool{{
					FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn"}},
					MCPServers:           []*genai.MCPServer{{Name: "srv"}},
				}}},
				wantSub: "MCPServers",
			},
			{
				name: "exa_ai_search_with_function_declarations",
				cfg: &genai.GenerateContentConfig{Tools: []*genai.Tool{{
					FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn"}},
					ExaAISearch:          &genai.ToolExaAISearch{},
				}}},
				wantSub: "ExaAISearch",
			},
			{
				name:    "empty_tool",
				cfg:     &genai.GenerateContentConfig{Tools: []*genai.Tool{{}}},
				wantSub: "FunctionDeclarations",
			},
			{
				name: "include_server_side_tool_invocations",
				cfg: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						IncludeServerSideToolInvocations: &streamArgs,
					},
				},
				wantSub: "IncludeServerSideToolInvocations",
			},
			{
				name: "validated_mode",
				cfg: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeValidated},
					},
				},
				wantSub: "VALIDATED",
			},
			{
				name: "any_with_multiple_allowed_names",
				cfg: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{
							Mode:                 genai.FunctionCallingConfigModeAny,
							AllowedFunctionNames: []string{"get_weather", "get_time"},
						},
					},
				},
				wantSub: "AllowedFunctionNames",
			},
			{
				name: "any_without_tools",
				cfg: &genai.GenerateContentConfig{
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
					},
				},
				wantSub: "ANY",
			},
			{
				name: "any_with_thinking_enabled",
				cfg: &genai.GenerateContentConfig{
					Tools:          weatherTools,
					ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &budget},
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
					},
				},
				wantSub: "ThinkingConfig",
			},
			{
				name: "auto_with_allowed_names",
				cfg: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{
							Mode:                 genai.FunctionCallingConfigModeAuto,
							AllowedFunctionNames: []string{"get_weather"},
						},
					},
				},
				wantSub: "AllowedFunctionNames",
			},
			{
				name: "stream_function_call_arguments",
				cfg: &genai.GenerateContentConfig{
					Tools: weatherTools,
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{
							StreamFunctionCallArguments: &streamArgs,
						},
					},
				},
				wantSub: "StreamFunctionCallArguments",
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				req := &model.LLMRequest{Contents: genai.Text("hi"), Config: tc.cfg}
				for _, err := range m.GenerateContent(t.Context(), req, false) {
					if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
						t.Errorf("error = %v, want ErrInvalidConfig", err)
					}
					if err != nil && !strings.Contains(err.Error(), tc.wantSub) {
						t.Errorf("error = %q, want substring %q", err.Error(), tc.wantSub)
					}
				}
			})
		}
	})
}

func TestMultimodalSupport(t *testing.T) {
	var requestCount int
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_m","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":1}}`)
	}))
	defer srv.Close()

	newBackendModel := func(t *testing.T, backend anthropicmodel.Backend, handler *slogCaptureHandler) model.LLM {
		t.Helper()
		cfg := anthropicmodel.ClientConfig{
			Backend: backend,
			BaseURL: srv.URL,
			Logger:  slog.New(handler),
		}
		if backend == anthropicmodel.BackendAnthropic {
			cfg.APIKey = "test-key"
		} else {
			cfg.Project = "test-project"
			cfg.Location = "us-east5"
			cfg.TokenSource = staticTokenSource{token: "test-token"}
		}
		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", cfg)
		if err != nil {
			t.Fatalf("NewModel(%s) error = %v", backend, err)
		}
		return m
	}

	cases := []struct {
		name              string
		part              *genai.Part
		wantAnthropicSent bool
		wantVertexSent    bool
	}{
		{
			name:              "inline_image_png",
			part:              genai.NewPartFromBytes(samplePNG1x1, "image/png; charset=binary"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "inline_image_jpeg",
			part:              genai.NewPartFromBytes([]byte("jpeg-bytes"), "image/jpeg"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "inline_image_gif",
			part:              genai.NewPartFromBytes([]byte("gif-bytes"), "image/gif"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "inline_image_webp",
			part:              genai.NewPartFromBytes([]byte("webp-bytes"), "image/webp"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "inline_pdf",
			part:              genai.NewPartFromBytes([]byte("%PDF-1.4"), "application/pdf"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "inline_plain_text",
			part:              genai.NewPartFromBytes([]byte("plain document text"), "text/plain; charset=utf-8"),
			wantAnthropicSent: true,
			wantVertexSent:    true,
		},
		{
			name:              "https_url_image",
			part:              genai.NewPartFromURI("https://example.com/secret-image.png", "image/png"),
			wantAnthropicSent: true,
			wantVertexSent:    false,
		},
		{
			name:              "https_url_pdf",
			part:              genai.NewPartFromURI("https://example.com/secret-doc.pdf", "application/pdf"),
			wantAnthropicSent: true,
			wantVertexSent:    false,
		},
		{
			name:              "https_url_plain_text_unsupported",
			part:              genai.NewPartFromURI("https://example.com/notes.txt", "text/plain"),
			wantAnthropicSent: false,
			wantVertexSent:    false,
		},
		{
			name:              "gs_uri_unsupported",
			part:              genai.NewPartFromURI("gs://my-secret-bucket/doc.pdf", "application/pdf"),
			wantAnthropicSent: false,
			wantVertexSent:    false,
		},
		{
			name:              "inline_audio_unsupported",
			part:              genai.NewPartFromBytes([]byte("SECRET_AUDIO_BYTES"), "audio/mp3"),
			wantAnthropicSent: false,
			wantVertexSent:    false,
		},
		{
			name:              "inline_video_unsupported",
			part:              genai.NewPartFromBytes([]byte("SECRET_VIDEO_BYTES"), "video/mp4"),
			wantAnthropicSent: false,
			wantVertexSent:    false,
		},
		{
			name:              "inline_octet_stream_unsupported",
			part:              genai.NewPartFromBytes([]byte("SECRET_BINARY"), "application/octet-stream"),
			wantAnthropicSent: false,
			wantVertexSent:    false,
		},
	}

	for _, tc := range cases {
		for _, backend := range []anthropicmodel.Backend{anthropicmodel.BackendAnthropic, anthropicmodel.BackendVertex} {
			t.Run(fmt.Sprintf("%s_%s", tc.name, backend), func(t *testing.T) {
				wantSent := tc.wantAnthropicSent
				if backend == anthropicmodel.BackendVertex {
					wantSent = tc.wantVertexSent
				}

				logHandler := &slogCaptureHandler{}
				m := newBackendModel(t, backend, logHandler)

				// Send alongside a text part so we can verify whether the media block was included or skipped with WARN.
				req := &model.LLMRequest{
					Contents: []*genai.Content{
						{
							Role:  genai.RoleUser,
							Parts: []*genai.Part{tc.part, genai.NewPartFromText("Describe this.")},
						},
					},
				}
				for _, err := range m.GenerateContent(t.Context(), req, false) {
					if err != nil {
						t.Fatalf("GenerateContent() error = %v", err)
					}
				}

				msgs, _ := lastBody["messages"].([]any)
				firstMsg, _ := msgs[0].(map[string]any)
				contentBlocks, _ := firstMsg["content"].([]any)
				records := logHandler.Snapshot()

				if wantSent {
					if len(contentBlocks) != 2 {
						t.Errorf("sent %d content blocks, want 2", len(contentBlocks))
					}
					if len(records) != 0 {
						t.Errorf("logged %d warnings, want 0", len(records))
					}
				} else {
					if len(contentBlocks) != 1 {
						t.Errorf("sent %d content blocks, want 1 (only text)", len(contentBlocks))
					}
					if len(records) != 1 || records[0].Level != slog.LevelWarn {
						t.Fatalf("logged records = %+v, want 1 WARN record", records)
					}
					// Ensure no secret URL or payload leaked into log message or attributes.
					for k, v := range records[0].Attrs {
						if strings.Contains(v, "example.com") || strings.Contains(v, "my-secret-bucket") || strings.Contains(v, "SECRET_") {
							t.Errorf("log attr %s=%q leaked sensitive data", k, v)
						}
					}
				}
			})
		}
	}

	t.Run("newest_message_empty_after_skipping_unsupported_fails_without_network_call", func(t *testing.T) {
		logHandler := &slogCaptureHandler{}
		m := newBackendModel(t, anthropicmodel.BackendAnthropic, logHandler)
		beforeCount := requestCount

		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Earlier valid message", genai.RoleUser),
				genai.NewContentFromText("Earlier assistant reply", genai.RoleModel),
				{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{genai.NewPartFromBytes([]byte("audio"), "audio/wav")},
				},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if !errors.Is(err, anthropicmodel.ErrNoContent) || !errors.Is(err, anthropicmodel.ErrUnsupportedContent) {
				t.Fatalf("GenerateContent() error = %v, want wrapping ErrNoContent and ErrUnsupportedContent", err)
			}
			if !strings.Contains(err.Error(), "mime_type=audio/wav") {
				t.Errorf("error = %q, want diagnostic containing mime_type=audio/wav", err.Error())
			}
		}
		if requestCount != beforeCount {
			t.Errorf("network was called (%d -> %d) when newest message was empty", beforeCount, requestCount)
		}
		if len(logHandler.Snapshot()) != 1 {
			t.Errorf("expected 1 WARN log, got %d", len(logHandler.Snapshot()))
		}
	})

	t.Run("earlier_unsupported_message_omitted_and_session_continues", func(t *testing.T) {
		logHandler := &slogCaptureHandler{}
		m := newBackendModel(t, anthropicmodel.BackendAnthropic, logHandler)

		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{genai.NewPartFromBytes([]byte("audio"), "audio/wav")},
				},
				genai.NewContentFromText("Valid follow-up question", genai.RoleUser),
			},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if len(logHandler.Snapshot()) != 1 {
			t.Errorf("expected 1 WARN log, got %d", len(logHandler.Snapshot()))
		}
	})

	t.Run("function_response_media_parts", func(t *testing.T) {
		logHandler := &slogCaptureHandler{}
		m := newBackendModel(t, anthropicmodel.BackendAnthropic, logHandler)

		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Render chart", genai.RoleUser),
				{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "toolu_chart", Name: "render_chart"}}},
				},
				{
					Role: genai.RoleUser,
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								ID:       "toolu_chart",
								Name:     "render_chart",
								Response: map[string]any{"status": "done"},
								Parts: []*genai.FunctionResponsePart{
									{InlineData: &genai.FunctionResponseBlob{MIMEType: "image/png", Data: samplePNG1x1}},
									{InlineData: &genai.FunctionResponseBlob{MIMEType: "application/pdf", Data: []byte("%PDF-1.4")}},
									{FileData: &genai.FunctionResponseFileData{MIMEType: "image/png", FileURI: "https://example.com/chart.png"}},
									{FileData: &genai.FunctionResponseFileData{MIMEType: "application/pdf", FileURI: "gs://bucket/chart.pdf"}},
									{InlineData: &genai.FunctionResponseBlob{MIMEType: "audio/ogg", Data: []byte("audio")}},
								},
							},
						},
					},
				},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if len(logHandler.Snapshot()) != 2 {
			t.Errorf("expected 2 WARN logs for unsupported gs:// URI and audio in FunctionResponse.Parts, got %d", len(logHandler.Snapshot()))
		}
		msgs, _ := lastBody["messages"].([]any)
		userBlocks, _ := msgs[2].(map[string]any)["content"].([]any)
		trBlock, _ := userBlocks[0].(map[string]any)
		trContent, _ := trBlock["content"].([]any)
		if len(trContent) != 4 {
			t.Fatalf("tool_result content has %d blocks, want 4 (text, inline image, inline pdf, https image URL)", len(trContent))
		}
		urlImgBlock, _ := trContent[3].(map[string]any)
		urlImgSrc, _ := urlImgBlock["source"].(map[string]any)
		if urlImgBlock["type"] != "image" || urlImgSrc["type"] != "url" || urlImgSrc["url"] != "https://example.com/chart.png" {
			t.Errorf("tool_result block[3] = %+v, want url image block", urlImgBlock)
		}
	})
}

func TestExtendedThinkingAndEffort(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)

		if isStream, _ := lastBody["stream"].(bool); isStream {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			events := []string{
				`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"usage":{"input_tokens":20,"output_tokens":1}}}` + "\n\n",
				`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}` + "\n\n",
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Step 1. "}}` + "\n\n",
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Step 2."}}` + "\n\n",
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_stream_123"}}` + "\n\n",
				`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n",
				`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"enc_blob_456"}}` + "\n\n",
				`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":1}` + "\n\n",
				`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_calc","name":"calc","input":{}}}` + "\n\n",
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"expr\":"}}` + "\n\n",
				`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"6*7\"}"}}` + "\n\n",
				`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":2}` + "\n\n",
				`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30,"output_tokens_details":{"thinking_tokens":18}}}` + "\n\n",
				`event: message_stop` + "\n" + `data: {"type":"message_stop"}` + "\n\n",
			}
			for _, ev := range events {
				_, _ = io.WriteString(w, ev)
				if flusher != nil {
					flusher.Flush()
				}
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_th","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"Deep thought","signature":"sig_abc"},{"type":"redacted_thinking","data":"redacted_xyz"},{"type":"text","text":"42"}],"usage":{"input_tokens":10,"output_tokens":25,"output_tokens_details":{"thinking_tokens":20}}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Effort:  anthropicmodel.EffortHigh,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	t.Run("thinking_config_table", func(t *testing.T) {
		zero := int32(0)
		pos := int32(512) // intentionally < 1024 to verify we do NOT clamp
		neg := int32(-1)

		// 1. nil ThinkingConfig -> thinking omitted, effort sent.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("q")}, false) {
			if err != nil {
				t.Fatalf("nil ThinkingConfig error = %v", err)
			}
		}
		if _, exists := lastBody["thinking"]; exists {
			t.Errorf("expected thinking to be omitted when ThinkingConfig is nil, got %v", lastBody["thinking"])
		}
		outCfg, _ := lastBody["output_config"].(map[string]any)
		if outCfg["effort"] != "high" {
			t.Errorf("output_config.effort = %v, want \"high\"", outCfg["effort"])
		}

		// 2. ThinkingBudget == 0 -> disabled.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Contents: genai.Text("q"),
			Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &zero}},
		}, false) {
			if err != nil {
				t.Fatalf("ThinkingBudget=0 error = %v", err)
			}
		}
		thMap, _ := lastBody["thinking"].(map[string]any)
		if thMap["type"] != "disabled" {
			t.Errorf("thinking = %+v, want type=disabled", thMap)
		}

		// 3. ThinkingBudget > 0 -> enabled with unclamped budget_tokens.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Contents: genai.Text("q"),
			Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &pos, IncludeThoughts: false}},
		}, false) {
			if err != nil {
				t.Fatalf("ThinkingBudget>0 error = %v", err)
			}
		}
		thMap, _ = lastBody["thinking"].(map[string]any)
		if thMap["type"] != "enabled" || thMap["budget_tokens"] != float64(512) {
			t.Errorf("thinking = %+v, want type=enabled budget_tokens=512", thMap)
		}

		// 4. ThinkingBudget < 0 -> adaptive.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Contents: genai.Text("q"),
			Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &neg}},
		}, false) {
			if err != nil {
				t.Fatalf("ThinkingBudget<0 error = %v", err)
			}
		}
		thMap, _ = lastBody["thinking"].(map[string]any)
		if thMap["type"] != "adaptive" {
			t.Errorf("thinking = %+v, want type=adaptive", thMap)
		}

		// 5. ThinkingBudget == nil -> error naming ThinkingBudget.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Contents: genai.Text("q"),
			Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true}},
		}, false) {
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "ThinkingBudget") {
				t.Errorf("nil ThinkingBudget error = %v, want ErrInvalidConfig naming ThinkingBudget", err)
			}
		}

		// 6. ThinkingLevel set -> error naming ThinkingLevel.
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{
			Contents: genai.Text("q"),
			Config:   &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &pos, ThinkingLevel: genai.ThinkingLevelHigh}},
		}, false) {
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) || !strings.Contains(err.Error(), "ThinkingLevel") {
				t.Errorf("ThinkingLevel error = %v, want ErrInvalidConfig naming ThinkingLevel", err)
			}
		}
	})

	t.Run("streaming_thinking_redacted_and_tool_loop_roundtrip", func(t *testing.T) {
		var partialThoughts []string
		var finalResp *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("Compute 6*7")}, true) {
			if err != nil {
				t.Fatalf("streaming error = %v", err)
			}
			if resp.Partial {
				for _, p := range resp.Content.Parts {
					if p.FunctionCall != nil {
						t.Errorf("partial event must not carry FunctionCall: %+v", p)
					}
					if p.Thought {
						partialThoughts = append(partialThoughts, p.Text)
					}
				}
			} else {
				finalResp = resp
			}
		}
		if strings.Join(partialThoughts, "") != "Step 1. Step 2." {
			t.Errorf("partialThoughts = %q, want \"Step 1. Step 2.\"", strings.Join(partialThoughts, ""))
		}
		if finalResp == nil || len(finalResp.Content.Parts) != 3 {
			t.Fatalf("finalResp parts = %+v, want 3 parts (thinking, redacted_thinking, tool_use)", finalResp)
		}
		if !finalResp.Content.Parts[0].Thought || string(finalResp.Content.Parts[0].ThoughtSignature) != "sig_stream_123" {
			t.Errorf("part[0] = %+v, want signed thinking part", finalResp.Content.Parts[0])
		}
		if !finalResp.Content.Parts[1].Thought || finalResp.Content.Parts[1].Text != "" || string(finalResp.Content.Parts[1].ThoughtSignature) != "enc_blob_456" {
			t.Errorf("part[1] = %+v, want redacted_thinking part", finalResp.Content.Parts[1])
		}
		if finalResp.Content.Parts[2].FunctionCall == nil || finalResp.Content.Parts[2].FunctionCall.Args["expr"] != "6*7" {
			t.Errorf("part[2] = %+v, want FunctionCall expr=6*7", finalResp.Content.Parts[2])
		}
		if finalResp.UsageMetadata.ThoughtsTokenCount != 18 || finalResp.UsageMetadata.CandidatesTokenCount != 12 {
			t.Errorf("UsageMetadata = %+v, want ThoughtsTokenCount=18 CandidatesTokenCount=12", finalResp.UsageMetadata)
		}

		// Replay the turn in history alongside an unsigned thinking block and a foreign signature,
		// plus text placed before thinking in the input Content, to verify ordering and stripping.
		replayedAssistant := &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				genai.NewPartFromText("I will calculate that."),
				{Thought: true, Text: "unsigned thought that must be stripped"},
				{ThoughtSignature: []byte("foreign_gemini_sig_without_thought_flag")},
				finalResp.Content.Parts[0], // signed thinking
				finalResp.Content.Parts[1], // redacted_thinking
				finalResp.Content.Parts[2], // tool_use
			},
		}
		followUpReq := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Compute 6*7", genai.RoleUser),
				replayedAssistant,
				{
					Role: genai.RoleUser,
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{ID: "toolu_calc", Name: "calc", Response: map[string]any{"result": 42}}},
					},
				},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), followUpReq, false) {
			if err != nil {
				t.Fatalf("followUp GenerateContent() error = %v", err)
			}
		}

		msgs, _ := lastBody["messages"].([]any)
		asstBlocks, _ := msgs[1].(map[string]any)["content"].([]any)
		if len(asstBlocks) != 4 {
			t.Fatalf("replayed assistant blocks = %d (%+v), want 4 (thinking, redacted_thinking, text, tool_use)", len(asstBlocks), asstBlocks)
		}
		if asstBlocks[0].(map[string]any)["type"] != "thinking" || asstBlocks[1].(map[string]any)["type"] != "redacted_thinking" {
			t.Errorf("thinking blocks were not ordered first on assistant turn: %+v", asstBlocks)
		}
	})
}

func TestPromptCaching(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_c","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"cached reply"}],"usage":{"input_tokens":50,"cache_read_input_tokens":1024,"cache_creation_input_tokens":256,"output_tokens":10}}`)
	}))
	defer srv.Close()

	for _, ttl := range []anthropicmodel.PromptCacheTTL{"", anthropicmodel.PromptCacheTTL5m, anthropicmodel.PromptCacheTTL1h} {
		t.Run("ttl_"+string(ttl), func(t *testing.T) {
			m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
				APIKey:      "test-key",
				BaseURL:     srv.URL,
				PromptCache: &anthropicmodel.PromptCacheConfig{TTL: ttl},
			})
			if err != nil {
				t.Fatalf("NewModel() error = %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("First user message", genai.RoleUser),
					{
						Role: genai.RoleModel,
						Parts: []*genai.Part{
							genai.NewPartFromText("Assistant text before thinking"),
							{Thought: true, Text: "Trailing thinking", ThoughtSignature: []byte("sig_1")},
						},
					},
				},
				Config: &genai.GenerateContentConfig{
					SystemInstruction: genai.NewContentFromText("System prompt", genai.RoleUser),
					Tools: []*genai.Tool{
						{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "tool_a"}, {Name: "tool_b"}}},
					},
				},
			}

			var got *model.LLMResponse
			for resp, err := range m.GenerateContent(t.Context(), req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() error = %v", err)
				}
				got = resp
			}

			wantTTL := "5m"
			if ttl == anthropicmodel.PromptCacheTTL1h {
				wantTTL = "1h"
			}

			// 1. Breakpoint on last tool only.
			toolsSlice, _ := lastBody["tools"].([]any)
			if _, hasCC := toolsSlice[0].(map[string]any)["cache_control"]; hasCC {
				t.Errorf("first tool should not have cache_control")
			}
			lastToolCC, _ := toolsSlice[1].(map[string]any)["cache_control"].(map[string]any)
			if lastToolCC["type"] != "ephemeral" || lastToolCC["ttl"] != wantTTL {
				t.Errorf("last tool cache_control = %+v, want ephemeral/%s", lastToolCC, wantTTL)
			}

			// 2. Breakpoint on system block.
			sysSlice, _ := lastBody["system"].([]any)
			sysCC, _ := sysSlice[0].(map[string]any)["cache_control"].(map[string]any)
			if sysCC["type"] != "ephemeral" || sysCC["ttl"] != wantTTL {
				t.Errorf("system cache_control = %+v, want ephemeral/%s", sysCC, wantTTL)
			}

			// 3. Breakpoint on last non-thinking block in messages (assistant text block, skipping thinking block).
			msgs, _ := lastBody["messages"].([]any)
			asstBlocks, _ := msgs[1].(map[string]any)["content"].([]any)
			// asstBlocks[0] is thinking, asstBlocks[1] is text.
			if _, hasCC := asstBlocks[0].(map[string]any)["cache_control"]; hasCC {
				t.Errorf("thinking block must never carry cache_control")
			}
			textCC, _ := asstBlocks[1].(map[string]any)["cache_control"].(map[string]any)
			if textCC["type"] != "ephemeral" || textCC["ttl"] != wantTTL {
				t.Errorf("last non-thinking message block cache_control = %+v, want ephemeral/%s", textCC, wantTTL)
			}

			// 4. Verify UsageMetadata and CustomMetadata.
			wantUsage := &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        50 + 1024 + 256,
				CandidatesTokenCount:    10,
				TotalTokenCount:         50 + 1024 + 256 + 10,
				CachedContentTokenCount: 1024,
				ThoughtsTokenCount:      0,
			}
			if diff := cmp.Diff(wantUsage, got.UsageMetadata); diff != "" {
				t.Errorf("UsageMetadata diff (-want +got):\n%s", diff)
			}
			if got.CustomMetadata[anthropicmodel.CacheCreationInputTokensMetadataKey] != int64(256) {
				t.Errorf("CustomMetadata[%q] = %v, want 256", anthropicmodel.CacheCreationInputTokensMetadataKey, got.CustomMetadata[anthropicmodel.CacheCreationInputTokensMetadataKey])
			}
		})
	}
}

func TestFinishReasonAndErrorClassification(t *testing.T) {
	t.Run("stop_reason_mapping_without_error_code", func(t *testing.T) {
		cases := []struct {
			stopReason string
			wantFinish genai.FinishReason
			wantMeta   string
		}{
			{stopReason: "end_turn", wantFinish: genai.FinishReasonStop, wantMeta: ""},
			{stopReason: "stop_sequence", wantFinish: genai.FinishReasonStop, wantMeta: ""},
			{stopReason: "tool_use", wantFinish: genai.FinishReasonStop, wantMeta: ""},
			{stopReason: "max_tokens", wantFinish: genai.FinishReasonMaxTokens, wantMeta: ""},
			{stopReason: "model_context_window_exceeded", wantFinish: genai.FinishReasonMaxTokens, wantMeta: "model_context_window_exceeded"},
			{stopReason: "refusal", wantFinish: genai.FinishReasonSafety, wantMeta: "refusal"},
			{stopReason: "pause_turn", wantFinish: genai.FinishReasonStop, wantMeta: "pause_turn"},
			{stopReason: "future_unknown_reason", wantFinish: genai.FinishReasonOther, wantMeta: "future_unknown_reason"},
		}

		for _, tc := range cases {
			t.Run(tc.stopReason, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprintf(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":%q,"content":[{"type":"text","text":"content present"}],"usage":{"input_tokens":5,"output_tokens":2}}`, tc.stopReason)
				}))
				defer srv.Close()

				m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
					APIKey:  "test-key",
					BaseURL: srv.URL,
				})
				if err != nil {
					t.Fatalf("NewModel() error = %v", err)
				}
				var got *model.LLMResponse
				for resp, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
					if err != nil {
						t.Fatalf("GenerateContent() error = %v", err)
					}
					got = resp
				}
				if got.FinishReason != tc.wantFinish {
					t.Errorf("FinishReason = %v, want %v", got.FinishReason, tc.wantFinish)
				}
				if got.ErrorCode != "" || got.ErrorMessage != "" {
					t.Errorf("ErrorCode=%q ErrorMessage=%q must not be set on content-bearing response", got.ErrorCode, got.ErrorMessage)
				}
				gotMeta, _ := got.CustomMetadata[anthropicmodel.StopReasonMetadataKey].(string)
				if gotMeta != tc.wantMeta {
					t.Errorf("CustomMetadata[%q] = %q, want %q", anthropicmodel.StopReasonMetadataKey, gotMeta, tc.wantMeta)
				}
			})
		}
	})

	t.Run("http_status_and_midstream_sse_errors", func(t *testing.T) {
		zeroRetries := 0
		errorCases := []struct {
			name         string
			status       int
			sse          bool
			body         string
			wantSentinel error
			wantCode     int
			wantType     string
		}{
			{
				name:         "http_429_rate_limit",
				status:       http.StatusTooManyRequests,
				body:         `{"type":"error","request_id":"req_429","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}`,
				wantSentinel: anthropicmodel.ErrRateLimited,
				wantCode:     http.StatusTooManyRequests,
				wantType:     "rate_limit_error",
			},
			{
				name:         "http_529_overloaded",
				status:       anthropicmodel.StatusOverloaded,
				body:         `{"type":"error","request_id":"req_529","error":{"type":"overloaded_error","message":"Overloaded"}}`,
				wantSentinel: anthropicmodel.ErrOverloaded,
				wantCode:     anthropicmodel.StatusOverloaded,
				wantType:     "overloaded_error",
			},
			{
				name:         "http_401_auth",
				status:       http.StatusUnauthorized,
				body:         `{"type":"error","request_id":"req_401","error":{"type":"authentication_error","message":"Invalid key sk-ant-secret"}}`,
				wantSentinel: anthropicmodel.ErrAuthentication,
				wantCode:     http.StatusUnauthorized,
				wantType:     "authentication_error",
			},
			{
				name:         "http_400_bad_request",
				status:       http.StatusBadRequest,
				body:         `{"type":"error","request_id":"req_400","error":{"type":"invalid_request_error","message":"thinking.budget_tokens: Input should be >= 1024"}}`,
				wantSentinel: anthropicmodel.ErrBadRequest,
				wantCode:     http.StatusBadRequest,
				wantType:     "invalid_request_error",
			},
			{
				name:         "midstream_sse_overloaded",
				status:       http.StatusOK,
				sse:          true,
				body:         "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5-20250929\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\nevent: error\ndata: {\"type\":\"error\",\"request_id\":\"req_sse_529\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded mid-stream\"}}\n\n",
				wantSentinel: anthropicmodel.ErrOverloaded,
				wantCode:     anthropicmodel.StatusOverloaded,
				wantType:     "overloaded_error",
			},
			{
				name:         "midstream_sse_rate_limit",
				status:       http.StatusOK,
				sse:          true,
				body:         "event: error\ndata: {\"type\":\"error\",\"request_id\":\"req_sse_429\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Rate limited mid-stream\"}}\n\n",
				wantSentinel: anthropicmodel.ErrRateLimited,
				wantCode:     http.StatusTooManyRequests,
				wantType:     "rate_limit_error",
			},
			{
				name:         "midstream_sse_api_error_maps_to_500_not_200",
				status:       http.StatusOK,
				sse:          true,
				body:         "event: error\ndata: {\"type\":\"error\",\"request_id\":\"req_sse_500\",\"error\":{\"type\":\"api_error\",\"message\":\"Internal server error mid-stream\"}}\n\n",
				wantSentinel: anthropicmodel.ErrAPIStatus,
				wantCode:     http.StatusInternalServerError,
				wantType:     "api_error",
			},
			{
				name:         "multibyte_utf8_message_truncation_preserves_valid_utf8",
				status:       http.StatusInternalServerError,
				body:         fmt.Sprintf(`{"type":"error","request_id":"req_utf8","error":{"type":"api_error","message":%q}}`, "a"+strings.Repeat("é", 200)),
				wantSentinel: anthropicmodel.ErrAPIStatus,
				wantCode:     http.StatusInternalServerError,
				wantType:     "api_error",
			},
		}

		for _, tc := range errorCases {
			t.Run(tc.name, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if tc.sse {
						w.Header().Set("Content-Type", "text/event-stream")
						w.WriteHeader(tc.status)
						_, _ = io.WriteString(w, tc.body)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("Request-Id", "req_hdr")
					w.WriteHeader(tc.status)
					_, _ = io.WriteString(w, tc.body)
				}))
				defer srv.Close()

				m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
					APIKey:     "sk-ant-secret",
					BaseURL:    srv.URL,
					MaxRetries: &zeroRetries,
				})
				if err != nil {
					t.Fatalf("NewModel() error = %v", err)
				}

				var gotErr error
				for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("SECRET_PROMPT_TEXT")}, tc.sse) {
					if err != nil {
						gotErr = err
					}
				}
				if !errors.Is(gotErr, tc.wantSentinel) {
					t.Fatalf("GenerateContent() error = %v, want errors.Is(%v)", gotErr, tc.wantSentinel)
				}
				var apiErr *anthropicmodel.APIError
				if !errors.As(gotErr, &apiErr) {
					t.Fatalf("expected errors.As(*APIError), got %T (%v)", gotErr, gotErr)
				}
				if apiErr.StatusCode != tc.wantCode {
					t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, tc.wantCode)
				}
				if apiErr.Type != tc.wantType {
					t.Errorf("Type = %q, want %q", apiErr.Type, tc.wantType)
				}
				if !utf8.ValidString(apiErr.Message) || !utf8.ValidString(gotErr.Error()) {
					t.Errorf("error message is not valid UTF-8: %q", apiErr.Message)
				}
				errStr := gotErr.Error()
				if strings.Contains(errStr, "SECRET_PROMPT_TEXT") || strings.Contains(errStr, "sk-ant-secret") || strings.Contains(errStr, srv.URL) {
					t.Errorf("error string %q leaked prompt, key, or URL", errStr)
				}
			})
		}
	})
}

func TestGenerateContentConfigValidation(t *testing.T) {
	var gotCustomHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotCustomHeader = r.Header.Get("X-Gemini-Only")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_v","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	temp := float32(0.5)
	topP := float32(0.9)
	negFloat := float32(-1)
	posInt := int32(1)
	boolTrue := true
	negTimeout := -time.Second

	rejectedCases := []struct {
		name      string
		cfg       *genai.GenerateContentConfig
		wantField string
	}{
		{
			name:      "temperature_and_top_p_together",
			cfg:       &genai.GenerateContentConfig{Temperature: &temp, TopP: &topP},
			wantField: "Temperature",
		},
		{
			name:      "negative_top_k",
			cfg:       &genai.GenerateContentConfig{TopK: &negFloat},
			wantField: "TopK",
		},
		{
			name:      "candidate_count_gt_1",
			cfg:       &genai.GenerateContentConfig{CandidateCount: 2},
			wantField: "CandidateCount",
		},
		{
			name:      "negative_max_output_tokens",
			cfg:       &genai.GenerateContentConfig{MaxOutputTokens: -10},
			wantField: "MaxOutputTokens",
		},
		{
			name: "non_text_system_instruction",
			cfg: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{Parts: []*genai.Part{genai.NewPartFromBytes(samplePNG1x1, "image/png")}},
			},
			wantField: "SystemInstruction",
		},
		{
			name:      "non_text_response_modality",
			cfg:       &genai.GenerateContentConfig{ResponseModalities: []string{"AUDIO"}},
			wantField: "ResponseModalities",
		},
		{
			name:      "http_options_base_url",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{BaseURL: "https://other.example.com"}},
			wantField: "HTTPOptions.BaseURL",
		},
		{
			name:      "http_options_base_url_resource_scope",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{BaseURLResourceScope: genai.ResourceScopeCollection}},
			wantField: "HTTPOptions.BaseURLResourceScope",
		},
		{
			name:      "http_options_api_version",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{APIVersion: "v1beta"}},
			wantField: "HTTPOptions.APIVersion",
		},
		{
			name:      "http_options_extra_body",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{ExtraBody: map[string]any{"k": "v"}}},
			wantField: "HTTPOptions.ExtraBody",
		},
		{
			name: "http_options_extras_request_provider",
			cfg: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{
				ExtrasRequestProvider: func(b map[string]any) map[string]any {
					return b
				},
			}},
			wantField: "HTTPOptions.ExtrasRequestProvider",
		},
		{
			name:      "http_options_retry_options",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{}}},
			wantField: "HTTPOptions.RetryOptions",
		},
		{
			name:      "http_options_negative_timeout",
			cfg:       &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: &negTimeout}},
			wantField: "HTTPOptions.Timeout",
		},
		{
			name:      "response_mime_type_without_schema",
			cfg:       &genai.GenerateContentConfig{ResponseMIMEType: "application/json"},
			wantField: "ResponseMIMEType",
		},
		{
			name: "response_mime_type_non_json_with_schema",
			cfg: &genai.GenerateContentConfig{
				ResponseMIMEType: "text/plain",
				ResponseSchema:   &genai.Schema{Type: genai.TypeObject},
			},
			wantField: "ResponseMIMEType",
		},
		{
			name: "response_schema_and_response_json_schema_both_set",
			cfg: &genai.GenerateContentConfig{
				ResponseSchema:     &genai.Schema{Type: genai.TypeObject},
				ResponseJsonSchema: map[string]any{"type": "object"},
			},
			wantField: "ResponseSchema",
		},
		{
			name:      "response_logprobs",
			cfg:       &genai.GenerateContentConfig{ResponseLogprobs: true},
			wantField: "ResponseLogprobs",
		},
		{
			name:      "logprobs",
			cfg:       &genai.GenerateContentConfig{Logprobs: &posInt},
			wantField: "Logprobs",
		},
		{
			name:      "presence_penalty",
			cfg:       &genai.GenerateContentConfig{PresencePenalty: &temp},
			wantField: "PresencePenalty",
		},
		{
			name:      "frequency_penalty",
			cfg:       &genai.GenerateContentConfig{FrequencyPenalty: &temp},
			wantField: "FrequencyPenalty",
		},
		{
			name:      "seed",
			cfg:       &genai.GenerateContentConfig{Seed: &posInt},
			wantField: "Seed",
		},
		{
			name:      "safety_settings",
			cfg:       &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{}}},
			wantField: "SafetySettings",
		},
		{
			name:      "cached_content",
			cfg:       &genai.GenerateContentConfig{CachedContent: "caches/123"},
			wantField: "CachedContent",
		},
		{
			name:      "labels",
			cfg:       &genai.GenerateContentConfig{Labels: map[string]string{"env": "prod"}},
			wantField: "Labels",
		},
		{
			name:      "media_resolution",
			cfg:       &genai.GenerateContentConfig{MediaResolution: genai.MediaResolutionHigh},
			wantField: "MediaResolution",
		},
		{
			name:      "speech_config",
			cfg:       &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}},
			wantField: "SpeechConfig",
		},
		{
			name:      "audio_timestamp",
			cfg:       &genai.GenerateContentConfig{AudioTimestamp: true},
			wantField: "AudioTimestamp",
		},
		{
			name:      "image_config",
			cfg:       &genai.GenerateContentConfig{ImageConfig: &genai.ImageConfig{}},
			wantField: "ImageConfig",
		},
		{
			name:      "routing_config",
			cfg:       &genai.GenerateContentConfig{RoutingConfig: &genai.GenerationConfigRoutingConfig{}},
			wantField: "RoutingConfig",
		},
		{
			name:      "model_selection_config",
			cfg:       &genai.GenerateContentConfig{ModelSelectionConfig: &genai.ModelSelectionConfig{}},
			wantField: "ModelSelectionConfig",
		},
		{
			name:      "model_armor_config",
			cfg:       &genai.GenerateContentConfig{ModelArmorConfig: &genai.ModelArmorConfig{}},
			wantField: "ModelArmorConfig",
		},
		{
			name:      "enable_enhanced_civic_answers",
			cfg:       &genai.GenerateContentConfig{EnableEnhancedCivicAnswers: &boolTrue},
			wantField: "EnableEnhancedCivicAnswers",
		},
		{
			name:      "service_tier",
			cfg:       &genai.GenerateContentConfig{ServiceTier: "flex"},
			wantField: "ServiceTier",
		},
		{
			name:      "audio_transcription_config",
			cfg:       &genai.GenerateContentConfig{AudioTranscriptionConfig: &genai.AudioTranscriptionConfig{}},
			wantField: "AudioTranscriptionConfig",
		},
		{
			name:      "continuation_token",
			cfg:       &genai.GenerateContentConfig{ContinuationToken: []byte("tok")},
			wantField: "ContinuationToken",
		},
	}

	for _, tc := range rejectedCases {
		t.Run(tc.name, func(t *testing.T) {
			for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi"), Config: tc.cfg}, false) {
				if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
					t.Fatalf("error = %v, want ErrInvalidConfig", err)
				}
				if !strings.Contains(err.Error(), tc.wantField) {
					t.Errorf("error = %q, want substring %q", err.Error(), tc.wantField)
				}
			}
		})
	}

	t.Run("ignored_fields_do_not_error_or_leak_headers", func(t *testing.T) {
		timeout := 5 * time.Second
		cfg := &genai.GenerateContentConfig{
			ResponseModalities:    []string{"TEXT"},
			AutomaticContinuation: &boolTrue,
			HTTPOptions: &genai.HTTPOptions{
				Timeout: &timeout,
				Headers: http.Header{"X-Gemini-Only": []string{"should-not-be-sent"}},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi"), Config: cfg}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if gotCustomHeader != "" {
			t.Errorf("HTTPOptions.Headers leaked X-Gemini-Only=%q to server", gotCustomHeader)
		}
	})
}

func TestConcurrencyAndNonMutation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_conc","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"concurrent ok"}],"usage":{"input_tokens":8,"output_tokens":2}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:      "test-key",
		BaseURL:     srv.URL,
		PromptCache: &anthropicmodel.PromptCacheConfig{TTL: anthropicmodel.PromptCacheTTL5m},
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	temp := float32(0.2)
	sharedReq := &model.LLMRequest{
		Model: "claude-sonnet-4-5-20250929",
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", genai.RoleUser),
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "invalid id", Name: "lookup", Args: map[string]any{"k": "v"}}},
				},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "invalid id", Name: "lookup", Response: map[string]any{"result": "found"}}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("Be helpful.", genai.RoleUser),
			Temperature:       &temp,
			StopSequences:     []string{"STOP"},
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name: "lookup",
							ParametersJsonSchema: map[string]any{
								"type": "OBJECT",
								"properties": map[string]any{
									"k": map[string]any{"type": "STRING"},
								},
							},
						},
					},
				},
			},
		},
	}

	snapshotBefore, err := json.Marshal(sharedReq)
	if err != nil {
		t.Fatalf("json.Marshal(sharedReq) error = %v", err)
	}

	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for _, err := range m.GenerateContent(t.Context(), sharedReq, false) {
				if err != nil {
					errs <- fmt.Errorf("worker %d: %w", idx, err)
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent GenerateContent failed: %v", err)
	}

	snapshotAfter, err := json.Marshal(sharedReq)
	if err != nil {
		t.Fatalf("json.Marshal(sharedReq) error = %v", err)
	}
	if !bytes.Equal(snapshotBefore, snapshotAfter) {
		t.Errorf("sharedReq was mutated during concurrent GenerateContent:\nbefore: %s\nafter:  %s", snapshotBefore, snapshotAfter)
	}
}

func TestContextCancellation(t *testing.T) {
	blockCh := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5-20250929\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n")
		_, _ = io.WriteString(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"first\"}}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-blockCh
	}))
	defer srv.Close()
	defer close(blockCh)

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	// 1. Pre-canceled context.
	canceledCtx, cancel := context.WithCancel(t.Context())
	cancel()
	for _, err := range m.GenerateContent(canceledCtx, &model.LLMRequest{Contents: genai.Text("hi")}, false) {
		if !errors.Is(err, context.Canceled) {
			t.Errorf("pre-canceled context error = %v, want context.Canceled", err)
		}
	}

	// 2. Mid-stream cancellation.
	streamCtx, streamCancel := context.WithCancel(t.Context())
	defer streamCancel()
	var gotErr error
	for resp, err := range m.GenerateContent(streamCtx, &model.LLMRequest{Contents: genai.Text("hi")}, true) {
		if err != nil {
			gotErr = err
			break
		}
		if resp != nil && resp.Partial {
			streamCancel()
		}
	}
	if !errors.Is(gotErr, context.Canceled) {
		t.Errorf("mid-stream cancellation error = %v, want context.Canceled", gotErr)
	}
}

func TestOutputSchemaAndStructuredOutput(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		if lastBody["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, strings.Join([]string{
				`event: message_start`,
				`data: {"type":"message_start","message":{"id":"msg_os_s","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`,
				``,
				`event: content_block_start`,
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"{\"city\":"}}`,
				``,
				`event: content_block_delta`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\"Warsaw\",\"temp_c\":18}"}}`,
				``,
				`event: content_block_stop`,
				`data: {"type":"content_block_stop","index":0}`,
				``,
				`event: message_delta`,
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":10}}`,
				``,
				`event: message_stop`,
				`data: {"type":"message_stop"}`,
				``,
			}, "\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_os","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"{\"city\":\"Warsaw\",\"temp_c\":18}"}],"usage":{"input_tokens":10,"output_tokens":10}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	t.Run("output_schema_processor_request_shape", func(t *testing.T) {
		// Exact shape produced by internal/llminternal/basic_processor.go:53-56.
		req := &model.LLMRequest{
			Contents: genai.Text("Return JSON"),
			Config: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
				ResponseSchema: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"city":   {Type: genai.TypeString},
						"temp_c": {Type: genai.TypeInteger},
					},
					Required: []string{"city", "temp_c"},
				},
			},
		}

		var got *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
			got = resp
		}
		if got == nil || got.Content == nil || len(got.Content.Parts) != 1 {
			t.Fatalf("unexpected response: %+v", got)
		}
		outCfg, _ := lastBody["output_config"].(map[string]any)
		formatMap, _ := outCfg["format"].(map[string]any)
		if formatMap["type"] != "json_schema" {
			t.Fatalf("output_config.format.type = %v, want \"json_schema\"", formatMap["type"])
		}
		schemaMap, _ := formatMap["schema"].(map[string]any)
		if schemaMap["type"] != "object" || schemaMap["additionalProperties"] != false {
			t.Errorf("output_config.format.schema = %+v, want type=object additionalProperties=false", schemaMap)
		}

		// Object root schema (unwrapped) streams partial text deltas as usual.
		var partialTexts []string
		var finalStreamResp *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), req, true) {
			if err != nil {
				t.Fatalf("GenerateContent(stream=true) error = %v", err)
			}
			if resp.Partial {
				for _, p := range resp.Content.Parts {
					partialTexts = append(partialTexts, p.Text)
				}
			} else {
				finalStreamResp = resp
			}
		}
		if len(partialTexts) != 2 || strings.Join(partialTexts, "") != `{"city":"Warsaw","temp_c":18}` {
			t.Errorf("object schema partialTexts = %v, want 2 chunks joining to %q", partialTexts, `{"city":"Warsaw","temp_c":18}`)
		}
		if finalStreamResp == nil || finalStreamResp.Content == nil || len(finalStreamResp.Content.Parts) != 1 || finalStreamResp.Content.Parts[0].Text != `{"city":"Warsaw","temp_c":18}` {
			t.Errorf("object schema finalStreamResp = %+v, want %q", finalStreamResp, `{"city":"Warsaw","temp_c":18}`)
		}
	})

	t.Run("response_json_schema_with_nested_objects_and_effort", func(t *testing.T) {
		mEffort, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
			Effort:  anthropicmodel.EffortXHigh,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		req := &model.LLMRequest{
			Contents: genai.Text("Return nested JSON"),
			Config: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{
					"type": "OBJECT",
					"properties": map[string]any{
						"items": map[string]any{
							"type": "ARRAY",
							"items": map[string]any{
								"type": "OBJECT",
								"properties": map[string]any{
									"name": map[string]any{"type": "STRING"},
								},
							},
						},
					},
				},
			},
		}
		for _, err := range mEffort.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		outCfg, _ := lastBody["output_config"].(map[string]any)
		if outCfg["effort"] != "xhigh" {
			t.Errorf("output_config.effort = %v, want \"xhigh\"", outCfg["effort"])
		}
		formatMap, _ := outCfg["format"].(map[string]any)
		schemaMap, _ := formatMap["schema"].(map[string]any)
		props, _ := schemaMap["properties"].(map[string]any)
		itemsProp, _ := props["items"].(map[string]any)
		itemSchema, _ := itemsProp["items"].(map[string]any)
		if itemSchema["type"] != "object" || itemSchema["additionalProperties"] != false {
			t.Errorf("nested item schema = %+v, want type=object additionalProperties=false", itemSchema)
		}
	})

	t.Run("non_object_root_schema_wrapped_and_unwrapped", func(t *testing.T) {
		var wrappedReqBody map[string]any
		arraySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			data, _ := io.ReadAll(r.Body)
			wrappedReqBody = nil
			_ = json.Unmarshal(data, &wrappedReqBody)
			if wrappedReqBody["stream"] == true {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, strings.Join([]string{
					`event: message_start`,
					`data: {"type":"message_start","message":{"id":"msg_arr_s","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":6,"output_tokens":1}}}`,
					``,
					`event: content_block_start`,
					`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"{\"result\":"}}`,
					``,
					`event: content_block_delta`,
					`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"[\"Warsaw\",\"Krakow\"]}"}}`,
					``,
					`event: content_block_stop`,
					`data: {"type":"content_block_stop","index":0}`,
					``,
					`event: message_delta`,
					`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":8}}`,
					``,
					`event: message_stop`,
					`data: {"type":"message_stop"}`,
					``,
				}, "\n"))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_arr","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"{\"result\":[\"Warsaw\",\"Krakow\"]}"}],"usage":{"input_tokens":6,"output_tokens":8}}`)
		}))
		defer arraySrv.Close()

		mArray, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: arraySrv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		req := &model.LLMRequest{
			Contents: genai.Text("List two Polish cities"),
			Config: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
				ResponseSchema: &genai.Schema{
					Type:  genai.TypeArray,
					Items: &genai.Schema{Type: genai.TypeString},
				},
			},
		}

		for _, streamMode := range []bool{false, true} {
			var partialResps []*model.LLMResponse
			var finalResp *model.LLMResponse
			for resp, err := range mArray.GenerateContent(t.Context(), req, streamMode) {
				if err != nil {
					t.Fatalf("GenerateContent(stream=%v) error = %v", streamMode, err)
				}
				if resp.Partial {
					partialResps = append(partialResps, resp)
				} else {
					finalResp = resp
				}
			}
			if len(partialResps) != 0 {
				t.Errorf("stream=%v emitted %d partial responses for wrapped schema (%+v), want 0", streamMode, len(partialResps), partialResps)
			}
			outCfg, _ := wrappedReqBody["output_config"].(map[string]any)
			formatMap, _ := outCfg["format"].(map[string]any)
			schemaMap, _ := formatMap["schema"].(map[string]any)
			if schemaMap["type"] != "object" || schemaMap["additionalProperties"] != false {
				t.Fatalf("stream=%v wrapped schema = %+v, want root type=object additionalProperties=false", streamMode, schemaMap)
			}
			props, _ := schemaMap["properties"].(map[string]any)
			resultSchema, _ := props["result"].(map[string]any)
			if resultSchema["type"] != "array" {
				t.Errorf("stream=%v wrapped properties.result = %+v, want type=array", streamMode, resultSchema)
			}
			if finalResp == nil || finalResp.Content == nil || len(finalResp.Content.Parts) != 1 {
				t.Fatalf("stream=%v unexpected final response: %+v", streamMode, finalResp)
			}
			if gotText := finalResp.Content.Parts[0].Text; gotText != `["Warsaw","Krakow"]` {
				t.Errorf("stream=%v unwrapped text = %q, want %q", streamMode, gotText, `["Warsaw","Krakow"]`)
			}
		}
	})

	t.Run("empty_response_json_schema_rejected", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: genai.Text("Return JSON"),
			Config: &genai.GenerateContentConfig{
				ResponseJsonSchema: map[string]any{},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if !errors.Is(err, anthropicmodel.ErrInvalidConfig) {
				t.Errorf("empty ResponseJsonSchema error = %v, want ErrInvalidConfig", err)
			}
		}
	})

	t.Run("llmagent_with_output_schema_end_to_end", func(t *testing.T) {
		a, err := llmagent.New(llmagent.Config{
			Name:        "structured_weather_agent",
			Description: "Returns structured weather JSON.",
			Model:       m,
			OutputSchema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"city":   {Type: genai.TypeString},
					"temp_c": {Type: genai.TypeInteger},
				},
				Required: []string{"city", "temp_c"},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New() error = %v", err)
		}
		tr := testutil.NewTestAgentRunner(t, a)
		texts, err := testutil.CollectTextParts(tr.Run(t, "structured_session", "Weather in Warsaw?"))
		if err != nil {
			t.Fatalf("llmagent Run() with OutputSchema error = %v", err)
		}
		if len(texts) == 0 || !strings.Contains(strings.Join(texts, ""), "Warsaw") {
			t.Errorf("unexpected texts = %v, want JSON containing Warsaw", texts)
		}

		// Non-object (array) OutputSchema on llmagent end-to-end.
		arraySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_arr_agent","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"{\"result\":[\"Warsaw\",\"Krakow\"]}"}],"usage":{"input_tokens":6,"output_tokens":8}}`)
		}))
		defer arraySrv.Close()

		mArray, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: arraySrv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}
		arrayAgent, err := llmagent.New(llmagent.Config{
			Name:        "city_list_agent",
			Description: "Returns a JSON array of city names.",
			Model:       mArray,
			OutputSchema: &genai.Schema{
				Type:  genai.TypeArray,
				Items: &genai.Schema{Type: genai.TypeString},
			},
		})
		if err != nil {
			t.Fatalf("llmagent.New(array OutputSchema) error = %v", err)
		}
		trArray := testutil.NewTestAgentRunner(t, arrayAgent)
		arrayTexts, err := testutil.CollectTextParts(trArray.Run(t, "array_session", "List two Polish cities"))
		if err != nil {
			t.Fatalf("llmagent Run() with array OutputSchema error = %v", err)
		}
		if got := strings.Join(arrayTexts, ""); got != `["Warsaw","Krakow"]` {
			t.Errorf("array OutputSchema agent output = %q, want %q", got, `["Warsaw","Krakow"]`)
		}
	})
}

func TestTruncatedStreamedToolCall(t *testing.T) {
	t.Run("streaming_truncated_tool_call_returns_max_tokens_without_error_or_function_call", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, strings.Join([]string{
				`event: message_start`,
				`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`,
				``,
				`event: content_block_start`,
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
				``,
				`event: content_block_delta`,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\": \"Warsa"}}`,
				``,
				`event: content_block_stop`,
				`data: {"type":"content_block_stop","index":0}`,
				``,
				`event: message_delta`,
				`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":20}}`,
				``,
				`event: message_stop`,
				`data: {"type":"message_stop"}`,
				``,
			}, "\n"))
		}))
		defer srv.Close()

		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		var resps []*model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, true) {
			if err != nil {
				t.Fatalf("GenerateContent(stream=true) returned error %v; want LLMResponse with FinishReasonMaxTokens", err)
			}
			resps = append(resps, resp)
		}
		if len(resps) != 1 || resps[0].FinishReason != genai.FinishReasonMaxTokens {
			t.Fatalf("got %+v, want 1 final response with FinishReasonMaxTokens", resps)
		}
		if resps[0].Content != nil {
			for _, p := range resps[0].Content.Parts {
				if p.FunctionCall != nil {
					t.Errorf("truncated tool_use block must not emit incomplete FunctionCall: %+v", p.FunctionCall)
				}
			}
		}
	})

	t.Run("non_streaming_max_tokens_omits_incomplete_tool_use_and_keeps_text", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_trunc","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"max_tokens","content":[{"type":"text","text":"Let me check the weather."},{"type":"tool_use","id":"toolu_trunc","name":"get_weather","input":{}}],"usage":{"input_tokens":10,"output_tokens":20}}`)
		}))
		defer srv.Close()

		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		var got *model.LLMResponse
		for resp, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
			got = resp
		}
		if got == nil || got.FinishReason != genai.FinishReasonMaxTokens {
			t.Fatalf("got = %+v, want FinishReasonMaxTokens", got)
		}
		if got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Parts[0].Text != "Let me check the weather." {
			t.Errorf("got Content.Parts = %+v, want only the text part without truncated FunctionCall", got.Content)
		}
	})
}

func TestFunctionResponseStructSerializationAndEmptyResult(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_fr","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":1}}`)
	}))
	defer srv.Close()

	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	type catalogItem struct {
		Name  string `json:"name"`
		Price int    `json:"price"`
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("List items and ping", genai.RoleUser),
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "toolu_items", Name: "list_items"}},
					{FunctionCall: &genai.FunctionCall{ID: "toolu_empty", Name: "noop"}},
				},
			},
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "toolu_items",
							Name: "list_items",
							Response: map[string]any{
								"result": []catalogItem{
									{Name: "apple", Price: 3},
									{Name: "pear", Price: 4},
								},
							},
						},
					},
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "toolu_empty",
							Name:     "noop",
							Response: map[string]any{},
						},
					},
				},
			},
		},
	}

	for _, err := range m.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
	}

	msgs, _ := lastBody["messages"].([]any)
	userBlocks, _ := msgs[2].(map[string]any)["content"].([]any)
	if len(userBlocks) != 2 {
		t.Fatalf("len(userBlocks) = %d, want 2", len(userBlocks))
	}

	// 1. Typed slice of structs under "result" must serialize as JSON, not Go fmt.Sprint syntax.
	trItems, _ := userBlocks[0].(map[string]any)
	trItemsContent, _ := trItems["content"].([]any)
	if len(trItemsContent) != 1 {
		t.Fatalf("trItems content = %+v, want 1 text block", trItemsContent)
	}
	gotText, _ := trItemsContent[0].(map[string]any)["text"].(string)
	wantJSON := `[{"name":"apple","price":3},{"name":"pear","price":4}]`
	if gotText != wantJSON {
		t.Errorf("tool_result text = %q, want %q", gotText, wantJSON)
	}

	// 2. Empty FunctionResponse must omit content on tool_result rather than sending [{"type":"text","text":""}].
	trEmpty, _ := userBlocks[1].(map[string]any)
	if contentVal, exists := trEmpty["content"]; exists {
		t.Errorf("empty FunctionResponse tool_result had content = %+v, want content omitted", contentVal)
	}
}

func TestThinkingAndEffortOmitSamplingParamsAndDefaultMaxTokens(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_s","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
	}))
	defer srv.Close()

	t.Run("default_max_tokens_is_8192_and_all_effort_levels_accepted", func(t *testing.T) {
		if anthropicmodel.DefaultMaxTokens != 8192 {
			t.Fatalf("DefaultMaxTokens = %d, want 8192", anthropicmodel.DefaultMaxTokens)
		}
		efforts := []anthropicmodel.Effort{
			anthropicmodel.EffortLow,
			anthropicmodel.EffortMedium,
			anthropicmodel.EffortHigh,
			anthropicmodel.EffortXHigh,
			anthropicmodel.EffortMax,
		}
		for _, eff := range efforts {
			m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
				APIKey:  "test-key",
				BaseURL: srv.URL,
				Effort:  eff,
			})
			if err != nil {
				t.Fatalf("NewModel(Effort=%q) error = %v", eff, err)
			}
			for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
				if err != nil {
					t.Fatalf("GenerateContent(Effort=%q) error = %v", eff, err)
				}
			}
			if lastBody["max_tokens"] != float64(8192) {
				t.Errorf("max_tokens = %v, want 8192", lastBody["max_tokens"])
			}
			outCfg, _ := lastBody["output_config"].(map[string]any)
			if outCfg["effort"] != string(eff) {
				t.Errorf("output_config.effort = %v, want %q", outCfg["effort"], eff)
			}
		}
	})

	t.Run("sampling_params_omitted_with_warn_when_thinking_or_effort_active", func(t *testing.T) {
		logHandler := &slogCaptureHandler{}
		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
			Logger:  slog.New(logHandler),
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		budget := int32(2048)
		temp := float32(0.0)
		topK := float32(40)
		req := &model.LLMRequest{
			Contents: genai.Text("solve"),
			Config: &genai.GenerateContentConfig{
				Temperature:    &temp,
				TopK:           &topK,
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &budget},
			},
		}
		for _, err := range m.GenerateContent(t.Context(), req, false) {
			if err != nil {
				t.Fatalf("GenerateContent() error = %v", err)
			}
		}
		if _, hasTemp := lastBody["temperature"]; hasTemp {
			t.Errorf("temperature was sent when thinking is enabled: %v", lastBody["temperature"])
		}
		if _, hasTopK := lastBody["top_k"]; hasTopK {
			t.Errorf("top_k was sent when thinking is enabled: %v", lastBody["top_k"])
		}
		records := logHandler.Snapshot()
		if len(records) != 1 || records[0].Level != slog.LevelWarn {
			t.Fatalf("logged records = %+v, want 1 WARN record for omitted sampling params", records)
		}

		// Effort active without ThinkingConfig must also omit sampling parameters and log a warning.
		effortLogHandler := &slogCaptureHandler{}
		mEffort, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
			Effort:  anthropicmodel.EffortHigh,
			Logger:  slog.New(effortLogHandler),
		})
		if err != nil {
			t.Fatalf("NewModel(Effort) error = %v", err)
		}
		reqEffortOnly := &model.LLMRequest{
			Contents: genai.Text("solve with effort"),
			Config: &genai.GenerateContentConfig{
				Temperature: &temp,
				TopK:        &topK,
			},
		}
		for _, err := range mEffort.GenerateContent(t.Context(), reqEffortOnly, false) {
			if err != nil {
				t.Fatalf("GenerateContent(Effort) error = %v", err)
			}
		}
		if _, hasTemp := lastBody["temperature"]; hasTemp {
			t.Errorf("temperature was sent when effort is active: %v", lastBody["temperature"])
		}
		if _, hasTopK := lastBody["top_k"]; hasTopK {
			t.Errorf("top_k was sent when effort is active: %v", lastBody["top_k"])
		}
		effortRecords := effortLogHandler.Snapshot()
		if len(effortRecords) != 1 || effortRecords[0].Level != slog.LevelWarn {
			t.Fatalf("logged records for effort = %+v, want 1 WARN record for omitted sampling params", effortRecords)
		}
	})
}

func TestExecutableCodeAndURISchemePrivacy(t *testing.T) {
	var lastBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		lastBody = nil
		_ = json.Unmarshal(data, &lastBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_ec","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":5,"output_tokens":1}}`)
	}))
	defer srv.Close()

	logHandler := &slogCaptureHandler{}
	m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
		Logger:  slog.New(logHandler),
	})
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					genai.NewPartFromURI("reports/q3-acquisition-plan.pdf?next=https://x.example/y", "application/pdf"),
					{ExecutableCode: &genai.ExecutableCode{Language: genai.LanguagePython, Code: "print(1)"}},
					{ExecutableCode: &genai.ExecutableCode{Language: genai.LanguageUnspecified, Code: "echo hi"}},
					{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: genai.OutcomeOK, Output: "1"}},
				},
			},
		},
	}

	for _, err := range m.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
	}

	// Verify relative path with query string containing :// logs uri_scheme="unknown" without leaking path.
	records := logHandler.Snapshot()
	if len(records) != 1 {
		t.Fatalf("logged %d records, want 1", len(records))
	}
	if records[0].Attrs["uri_scheme"] != "unknown" {
		t.Errorf("uri_scheme = %q, want \"unknown\"", records[0].Attrs["uri_scheme"])
	}
	for k, v := range records[0].Attrs {
		if strings.Contains(v, "reports") || strings.Contains(v, "q3-acquisition") {
			t.Errorf("log attr %s=%q leaked file path", k, v)
		}
	}

	// Verify ExecutableCode and CodeExecutionResult fenced blocks place opening fence on a new line.
	msgs, _ := lastBody["messages"].([]any)
	blocks, _ := msgs[0].(map[string]any)["content"].([]any)
	if len(blocks) != 3 {
		t.Fatalf("sent %d blocks, want 3", len(blocks))
	}
	b0, _ := blocks[0].(map[string]any)["text"].(string)
	b1, _ := blocks[1].(map[string]any)["text"].(string)
	b2, _ := blocks[2].(map[string]any)["text"].(string)
	if b0 != "Code:\n```python\nprint(1)\n```" {
		t.Errorf("block[0] = %q, want %q", b0, "Code:\n```python\nprint(1)\n```")
	}
	if b1 != "Code:\n```\necho hi\n```" {
		t.Errorf("block[1] = %q, want %q", b1, "Code:\n```\necho hi\n```")
	}
	if b2 != "Execution Result:\n```code_output\n1\n```" {
		t.Errorf("block[2] = %q, want %q", b2, "Execution Result:\n```code_output\n1\n```")
	}
}

func TestMaxRetriesAndInFlightNonStreamingCancellation(t *testing.T) {
	t.Run("max_retries_honored", func(t *testing.T) {
		zero := 0
		one := 1
		cases := []struct {
			name       string
			maxRetries *int
			wantCalls  int
		}{
			{name: "nil_default_2_retries", maxRetries: nil, wantCalls: 3},
			{name: "zero_retries", maxRetries: &zero, wantCalls: 1},
			{name: "one_retry", maxRetries: &one, wantCalls: 2},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				var calls int
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls++
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("Retry-After-Ms", "1")
					w.WriteHeader(anthropicmodel.StatusOverloaded)
					_, _ = io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
				}))
				defer srv.Close()

				m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
					APIKey:     "test-key",
					BaseURL:    srv.URL,
					MaxRetries: tc.maxRetries,
				})
				if err != nil {
					t.Fatalf("NewModel() error = %v", err)
				}
				for _, err := range m.GenerateContent(t.Context(), &model.LLMRequest{Contents: genai.Text("hi")}, false) {
					if !errors.Is(err, anthropicmodel.ErrOverloaded) {
						t.Fatalf("GenerateContent() error = %v, want ErrOverloaded", err)
					}
				}
				if calls != tc.wantCalls {
					t.Errorf("server received %d calls, want %d", calls, tc.wantCalls)
				}
			})
		}
	})

	t.Run("in_flight_non_streaming_request_cancellation", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			close(started)
			<-release
		}))
		defer srv.Close()
		defer close(release)

		m, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
			APIKey:  "test-key",
			BaseURL: srv.URL,
		})
		if err != nil {
			t.Fatalf("NewModel() error = %v", err)
		}

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		go func() {
			<-started
			cancel()
		}()

		var gotErr error
		for _, err := range m.GenerateContent(ctx, &model.LLMRequest{Contents: genai.Text("hi")}, false) {
			gotErr = err
		}
		if !errors.Is(gotErr, context.Canceled) {
			t.Errorf("in-flight non-streaming cancellation error = %v, want context.Canceled", gotErr)
		}
	})
}

func TestGenaiConfigReflectionCompleteness(t *testing.T) {
	assertFieldsMatch := func(t *testing.T, typ reflect.Type, known map[string]bool) {
		t.Helper()
		exported := make(map[string]bool, typ.NumField())
		for i := range typ.NumField() {
			f := typ.Field(i)
			if !f.IsExported() {
				continue
			}
			exported[f.Name] = true
			if !known[f.Name] {
				t.Errorf("%s has unhandled exported field %q; classify it in config/tool validation and update this test", typ.Name(), f.Name)
			}
		}
		for name := range known {
			if !exported[name] {
				t.Errorf("%s allowlist contains unknown field %q that does not exist on struct", typ.Name(), name)
			}
		}
	}

	assertFieldsMatch(t, reflect.TypeFor[genai.GenerateContentConfig](), map[string]bool{
		"SystemInstruction":          true,
		"Temperature":                true,
		"TopP":                       true,
		"TopK":                       true,
		"CandidateCount":             true,
		"MaxOutputTokens":            true,
		"StopSequences":              true,
		"ResponseLogprobs":           true,
		"Logprobs":                   true,
		"PresencePenalty":            true,
		"FrequencyPenalty":           true,
		"Seed":                       true,
		"ResponseMIMEType":           true,
		"ResponseSchema":             true,
		"ResponseJsonSchema":         true,
		"RoutingConfig":              true,
		"ModelSelectionConfig":       true,
		"SafetySettings":             true,
		"Tools":                      true,
		"ToolConfig":                 true,
		"Labels":                     true,
		"CachedContent":              true,
		"ResponseModalities":         true,
		"MediaResolution":            true,
		"SpeechConfig":               true,
		"AudioTimestamp":             true,
		"ThinkingConfig":             true,
		"ImageConfig":                true,
		"HTTPOptions":                true,
		"ModelArmorConfig":           true,
		"EnableEnhancedCivicAnswers": true,
		"ServiceTier":                true,
		"AudioTranscriptionConfig":   true,
		"AutomaticContinuation":      true,
		"ContinuationToken":          true,
	})

	assertFieldsMatch(t, reflect.TypeFor[genai.HTTPOptions](), map[string]bool{
		"BaseURL":               true,
		"BaseURLResourceScope":  true,
		"APIVersion":            true,
		"Headers":               true,
		"Timeout":               true,
		"ExtraBody":             true,
		"ExtrasRequestProvider": true,
		"RetryOptions":          true,
	})

	assertFieldsMatch(t, reflect.TypeFor[genai.ThinkingConfig](), map[string]bool{
		"IncludeThoughts": true,
		"ThinkingBudget":  true,
		"ThinkingLevel":   true,
	})

	assertFieldsMatch(t, reflect.TypeFor[genai.ToolConfig](), map[string]bool{
		"FunctionCallingConfig":            true,
		"RetrievalConfig":                  true,
		"IncludeServerSideToolInvocations": true,
	})

	assertFieldsMatch(t, reflect.TypeFor[genai.FunctionCallingConfig](), map[string]bool{
		"Mode":                        true,
		"AllowedFunctionNames":        true,
		"StreamFunctionCallArguments": true,
	})

	assertFieldsMatch(t, reflect.TypeFor[genai.Tool](), map[string]bool{
		"FunctionDeclarations":  true,
		"Retrieval":             true,
		"GoogleSearch":          true,
		"GoogleSearchRetrieval": true,
		"EnterpriseWebSearch":   true,
		"GoogleMaps":            true,
		"URLContext":            true,
		"ComputerUse":           true,
		"CodeExecution":         true,
		"FileSearch":            true,
		"ParallelAISearch":      true,
		"MCPServers":            true,
		"ExaAISearch":           true,
	})
}
