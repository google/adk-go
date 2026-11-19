// Copyright 2025 Google LLC
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

package openaimodel

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/responses"
	"google.golang.org/adk/v2/model/openaimodel/internal/shared"
)

// TestNewModel_SelectsAPI is the facade's whole job: turn the config field into
// the right endpoint. It asserts on the path the request reached, because that
// is the only thing a caller can observe about the choice.
func TestNewModel_SelectsAPI(t *testing.T) {
	tests := []struct {
		name     string
		api      API
		wantPath string
	}{
		{name: "zero value keeps Responses", api: "", wantPath: "/v1/responses"},
		{name: "explicit Responses", api: APIResponses, wantPath: "/v1/responses"},
		{name: "chat completions", api: APIChatCompletions, wantPath: "/v1/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got string
			server := newLoopbackServer(t, func(w http.ResponseWriter, r *http.Request) {
				got = r.URL.Path
				w.WriteHeader(http.StatusInternalServerError)
			})

			llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
				APIKey:     "test",
				BaseURL:    server.URL + "/v1",
				HTTPClient: server.Client(),
				API:        tt.api,
			})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			req := &model.LLMRequest{Contents: []*genai.Content{
				genai.NewContentFromText("hi", genai.RoleUser),
			}}
			for range llm.GenerateContent(t.Context(), req, false) { //nolint:errcheck // the server answers 500; the test asserts the request path
			}
			if got != tt.wantPath {
				t.Errorf("path = %q, want %q", got, tt.wantPath)
			}
		})
	}
}

func TestNewModel_RejectsUnknownAPI(t *testing.T) {
	_, err := NewModel(t.Context(), "m", &ClientConfig{APIKey: "k", API: "grpc"})
	if !errors.Is(err, ErrUnsupportedAPI) {
		t.Fatalf("err = %v, want %v", err, ErrUnsupportedAPI)
	}
	if !strings.Contains(err.Error(), "grpc") {
		t.Errorf("err = %v, want it to name the value", err)
	}
}

func TestNewModel_RequiresModelName(t *testing.T) {
	if _, err := NewModel(t.Context(), "", &ClientConfig{APIKey: "k"}); !errors.Is(err, ErrModelNameRequired) {
		t.Fatalf("err = %v, want %v", err, ErrModelNameRequired)
	}
}

// TestSentinelsAliasTheInternalOnes pins the aliasing: errors.Is must still
// match an error the endpoint packages return, so every exported sentinel has
// to be the internal one rather than a lookalike with the same message.
func TestSentinelsAliasTheInternalOnes(t *testing.T) {
	for name, pair := range map[string][2]error{
		"ErrModelNameRequired":              {ErrModelNameRequired, shared.ErrModelNameRequired},
		"ErrUnsupportedAPI":                 {ErrUnsupportedAPI, shared.ErrUnsupportedAPI},
		"ErrNoChoices":                      {ErrNoChoices, shared.ErrNoChoices},
		"ErrRequestNil":                     {ErrRequestNil, shared.ErrRequestNil},
		"ErrNoContents":                     {ErrNoContents, shared.ErrNoContents},
		"ErrFunctionCallMissingName":        {ErrFunctionCallMissingName, shared.ErrFunctionCallMissingName},
		"ErrTopKNotSupported":               {ErrTopKNotSupported, shared.ErrTopKNotSupported},
		"ErrStopSequencesNotSupported":      {ErrStopSequencesNotSupported, shared.ErrStopSequencesNotSupported},
		"ErrMultipleCandidatesNotSupported": {ErrMultipleCandidatesNotSupported, shared.ErrMultipleCandidatesNotSupported},
		"ErrPenaltiesNotSupported":          {ErrPenaltiesNotSupported, shared.ErrPenaltiesNotSupported},
		"ErrLabelsNotSupported":             {ErrLabelsNotSupported, shared.ErrLabelsNotSupported},
		"ErrSafetySettingsNotSupported":     {ErrSafetySettingsNotSupported, shared.ErrSafetySettingsNotSupported},
		"ErrUnsupportedMIMEType":            {ErrUnsupportedMIMEType, shared.ErrUnsupportedMIMEType},
		"ErrMediaMIMETypeRequired":          {ErrMediaMIMETypeRequired, shared.ErrMediaMIMETypeRequired},
		"ErrMediaDataRequired":              {ErrMediaDataRequired, shared.ErrMediaDataRequired},
		"ErrMediaURIRequired":               {ErrMediaURIRequired, shared.ErrMediaURIRequired},
		"ErrMediaOnAssistantTurn":           {ErrMediaOnAssistantTurn, shared.ErrMediaOnAssistantTurn},
		"ErrUnsupportedConfigField":         {ErrUnsupportedConfigField, shared.ErrUnsupportedConfigField},
		"ErrEmptyJSONSchema":                {ErrEmptyJSONSchema, shared.ErrEmptyJSONSchema},
		"ErrEmptyResponse":                  {ErrEmptyResponse, shared.ErrEmptyResponse},
		"ErrNoOutputItems":                  {ErrNoOutputItems, shared.ErrNoOutputItems},
		"ErrUnsupportedMessageContentType":  {ErrUnsupportedMessageContentType, shared.ErrUnsupportedMessageContentType},
		"ErrUnsupportedOutputItemType":      {ErrUnsupportedOutputItemType, shared.ErrUnsupportedOutputItemType},
		"ErrFunctionCallArgs":               {ErrFunctionCallArgs, shared.ErrFunctionCallArgs},
		"ErrNoTextOrToolContent":            {ErrNoTextOrToolContent, shared.ErrNoTextOrToolContent},
		"ErrResponseFailed":                 {ErrResponseFailed, shared.ErrResponseFailed},
	} {
		if pair[0] != pair[1] {
			t.Errorf("%s is not the sentinel the endpoint packages return", name)
		}
	}

	// And end to end, through an error an endpoint package actually built.
	llm, err := NewModel(t.Context(), "m", &ClientConfig{APIKey: "k", API: APIChatCompletions})
	if err != nil {
		t.Fatalf("NewModel() err = %v", err)
	}
	var got error
	for _, err := range llm.GenerateContent(t.Context(), &model.LLMRequest{}, false) {
		got = err
	}
	if !errors.Is(got, ErrNoContents) {
		t.Fatalf("err = %v, want the aliased %v to match", got, ErrNoContents)
	}
}

// TestFinishMessageKeyMatchesTheInternalOne pins the literal the facade
// publishes to the key the Responses path writes, since the two are separate
// constants so that the published docs show the value.
func TestFinishMessageKeyMatchesTheInternalOne(t *testing.T) {
	if FinishMessageKey != responses.FinishMessageKey {
		t.Fatalf("FinishMessageKey = %q, but the Responses path writes %q", FinishMessageKey, responses.FinishMessageKey)
	}
}

// TestHTTPOptionsHeadersNeverReachTheWire checks, through the constructor
// callers use, that no header a caller put in HTTPOptions reaches the provider
// on either API: a Gemini credential must not leave for OpenAI, and a caller's
// Authorization must not displace the configured key.
func TestHTTPOptionsHeadersNeverReachTheWire(t *testing.T) {
	for _, api := range []API{APIResponses, APIChatCompletions} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", api, stream), func(t *testing.T) {
				var got http.Header
				server := newLoopbackServer(t, func(w http.ResponseWriter, r *http.Request) {
					got = r.Header.Clone()
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprint(w, completedBodies[api])
				})
				llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
					APIKey:     "real-key",
					BaseURL:    server.URL + "/v1",
					HTTPClient: server.Client(),
					API:        api,
				})
				if err != nil {
					t.Fatalf("NewModel() err = %v", err)
				}
				// A timeout is set deliberately. Without one the translation
				// returns early, so header forwarding reintroduced after that
				// point would never run here and the test would pass vacuously.
				timeout := 30 * time.Second
				req := &model.LLMRequest{
					Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
					Config: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{
						Timeout: &timeout,
						Headers: http.Header{
							"Authorization":  []string{"Bearer caller"},
							"X-Goog-Api-Key": []string{"gemini-key"},
							"X-Trace-Id":     []string{"harmless"},
						},
					}},
				}
				// A streamed call fails on the JSON body; only its headers matter.
				for _, err := range llm.GenerateContent(t.Context(), req, stream) {
					if err != nil && !stream {
						t.Fatalf("GenerateContent() err = %v", err)
					}
				}
				if got == nil {
					t.Fatal("no request reached the server")
				}
				if v := got.Get("Authorization"); v != "Bearer real-key" {
					t.Errorf("Authorization = %s, want the configured key: a caller header displaced it", redact(v))
				}
				if v := got.Get("X-Goog-Api-Key"); v != "" {
					t.Errorf("X-Goog-Api-Key = %s, want absent: a Gemini credential reached the provider", redact(v))
				}
				if v := got.Get("X-Trace-Id"); v != "" {
					t.Errorf("X-Trace-Id = %s, want absent: headers are not forwarded", redact(v))
				}
			})
		}
	}
}

// completedBodies holds a minimal completed blocking response per API.
var completedBodies = map[API]string{
	APIResponses: `{"id":"r","object":"response","status":"completed",` +
		`"output":[{"type":"message","id":"m","role":"assistant","status":"completed",` +
		`"content":[{"type":"output_text","text":"hi","annotations":[]}]}]}`,
	APIChatCompletions: `{"id":"c","object":"chat.completion","model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`,
}

// TestNewModel_UsesTheConfiguredClient pins that NewModel hands
// ClientConfig.HTTPClient and ClientConfig.Options to the SDK. The configured
// client answers in memory, while BaseURL points at a server that fails every
// request, so a model built on the default client gets the failure instead.
func TestNewModel_UsesTheConfiguredClient(t *testing.T) {
	for _, api := range []API{APIResponses, APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			server := newLoopbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
				// Not a status the SDK retries.
				w.WriteHeader(http.StatusTeapot)
			})
			used := false
			var optionHeader string
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				used = true
				optionHeader = r.Header.Get("X-Adk-Option")
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(completedBodies[api])),
					Request:    r,
				}, nil
			})}
			llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
				APIKey:     "test",
				BaseURL:    server.URL + "/v1",
				HTTPClient: client,
				Options:    []option.RequestOption{option.WithHeader("X-Adk-Option", "set")},
				API:        api,
			})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)}}
			for _, err := range llm.GenerateContent(t.Context(), req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() err = %v", err)
				}
			}
			if !used {
				t.Error("the request did not go through ClientConfig.HTTPClient")
			}
			if optionHeader != "set" {
				t.Errorf("X-Adk-Option = %q, want the header ClientConfig.Options set", optionHeader)
			}
		})
	}
}

// TestNewModel_DeclaresNoOutputSchemaWithTools checks, on both APIs, that an
// OpenAI model declares it cannot take an output schema next to tools, as
// adk-python's OpenAILlm and OpenAIResponsesLlm do through the base fallback
// for a non-Gemini model (models/base_llm.py:92-124).
func TestNewModel_DeclaresNoOutputSchemaWithTools(t *testing.T) {
	for _, api := range []API{APIResponses, APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			llm, err := NewModel(t.Context(), "gpt-4o", &ClientConfig{APIKey: "test", API: api})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			cp, ok := llm.(model.CapabilitiesProvider)
			if !ok {
				t.Fatalf("NewModel() returned %T, which does not implement model.CapabilitiesProvider", llm)
			}
			if cp.Capabilities().OutputSchemaAndTools {
				t.Error("Capabilities().OutputSchemaAndTools = true, want false")
			}
		})
	}
}

// TestNewModel_ClientFunctionCallIDsByAPI checks that the Responses API, which
// is also the default, declares that it pairs tool calls with their results by
// ID, so the flow keeps ADK's adk- function call IDs in the request, and that
// Chat Completions does not, so the flow removes them. adk-python keeps the
// IDs for OpenAIResponsesLlm only (flows/llm_flows/context/_contents.py:96).
func TestNewModel_ClientFunctionCallIDsByAPI(t *testing.T) {
	tests := []struct {
		api  API
		want bool
	}{
		{api: "", want: true},
		{api: APIResponses, want: true},
		{api: APIChatCompletions, want: false},
	}
	for _, tt := range tests {
		t.Run("api="+string(tt.api), func(t *testing.T) {
			llm, err := NewModel(t.Context(), "gpt-4o", &ClientConfig{APIKey: "test", API: tt.api})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			cp, ok := llm.(model.CapabilitiesProvider)
			if !ok {
				t.Fatalf("NewModel() returned %T, which does not implement model.CapabilitiesProvider", llm)
			}
			if got := cp.Capabilities().ClientFunctionCallIDs; got != tt.want {
				t.Errorf("Capabilities().ClientFunctionCallIDs = %t, want %t", got, tt.want)
			}
		})
	}
}

// TestOpenAIAndAnthropic_IgnoreADKAgentNameLabelOnly checks, on both APIs, that
// the adk_agent_name label the flow adds to every request is accepted and never
// sent, while a label the caller set, or an empty labels map, is still refused
// before any request leaves.
func TestOpenAIAndAnthropic_IgnoreADKAgentNameLabelOnly(t *testing.T) {
	tests := []struct {
		name    string
		labels  map[string]string
		wantErr error
	}{
		{name: "adk_label_only", labels: map[string]string{"adk_agent_name": "my_agent"}},
		{name: "adk_and_user_label", labels: map[string]string{"adk_agent_name": "my_agent", "env": "prod"}, wantErr: ErrLabelsNotSupported},
		{name: "empty_map", labels: map[string]string{}, wantErr: ErrLabelsNotSupported},
	}
	for _, api := range []API{APIResponses, APIChatCompletions} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s/%s", api, tt.name), func(t *testing.T) {
				var requestBody []byte
				server := newLoopbackServer(t, func(w http.ResponseWriter, r *http.Request) {
					var err error
					if requestBody, err = io.ReadAll(r.Body); err != nil {
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprint(w, completedBodies[api])
				})
				llm, err := NewModel(t.Context(), "gpt-4o-mini", &ClientConfig{
					APIKey:     "test",
					BaseURL:    server.URL + "/v1",
					HTTPClient: server.Client(),
					API:        api,
				})
				if err != nil {
					t.Fatalf("NewModel() err = %v", err)
				}
				req := &model.LLMRequest{
					Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
					Config:   &genai.GenerateContentConfig{Labels: tt.labels},
				}
				var gotErr error
				for _, err := range llm.GenerateContent(t.Context(), req, false) {
					if err != nil {
						gotErr = err
					}
				}
				if tt.wantErr != nil {
					if !errors.Is(gotErr, tt.wantErr) {
						t.Errorf("GenerateContent() err = %v, want %v", gotErr, tt.wantErr)
					}
					if requestBody != nil {
						t.Error("a request reached the server, want it refused before sending")
					}
					return
				}
				if gotErr != nil {
					t.Fatalf("GenerateContent() err = %v", gotErr)
				}
				if requestBody == nil {
					t.Fatal("no request reached the server")
				}
				var payload map[string]any
				if err := json.Unmarshal(requestBody, &payload); err != nil {
					t.Fatalf("json.Unmarshal(request body) err = %v", err)
				}
				if v, ok := payload["labels"]; ok {
					t.Errorf("request body carries labels %v, want none", v)
				}
			})
		}
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// newLoopbackServer starts handler on a loopback port and closes it when the
// test ends. httptest binds 127.0.0.1 before trying IPv6, so a sandbox refusing
// IPv6 is served as well.
func newLoopbackServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

// redact describes a header value without printing it.
func redact(v string) string {
	if v == "" {
		return "empty"
	}
	return fmt.Sprintf("<%d-byte value>", len(v))
}

// TestRunHTTPOptions_KeepsTimeoutAndHeaders checks, on both APIs, that the
// agent flow is given only the fields of a run's HTTP options these models
// accept, so that RetryOptions or ExtraBody set for every agent of a run do
// not make an OpenAI request fail, as adk-python's OpenAI models ignore HTTP
// options.
func TestRunHTTPOptions_KeepsTimeoutAndHeaders(t *testing.T) {
	opts := &genai.HTTPOptions{
		Timeout:      genai.Ptr(2 * time.Second),
		Headers:      http.Header{"X-Run": {"run"}},
		BaseURL:      "https://proxy.example",
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3)},
		ExtraBody:    map[string]any{"k": "v"},
	}
	for _, api := range []API{APIResponses, APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			llm, err := NewModel(t.Context(), "gpt-4o", &ClientConfig{APIKey: "test", API: api})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			filter, ok := llm.(llminternal.RunHTTPOptionsFilter)
			if !ok {
				t.Fatalf("NewModel() returned %T, which does not implement llminternal.RunHTTPOptionsFilter", llm)
			}
			want := &genai.HTTPOptions{Timeout: genai.Ptr(2 * time.Second), Headers: http.Header{"X-Run": {"run"}}}
			if diff := cmp.Diff(want, filter.RunHTTPOptions(opts)); diff != "" {
				t.Errorf("RunHTTPOptions() mismatch (-want +got):\n%s", diff)
			}
			if got := filter.RunHTTPOptions(&genai.HTTPOptions{ExtraBody: map[string]any{"k": "v"}}); got != nil {
				t.Errorf("RunHTTPOptions(ExtraBody only) = %+v, want nil", got)
			}
		})
	}
}

// TestRunLabels_DropsAll checks, on both APIs, that the agent flow is given
// none of a run's labels for an OpenAI agent, so labels set for every agent of
// a run do not make an OpenAI request fail, as adk-python's OpenAI models
// ignore labels.
func TestRunLabels_DropsAll(t *testing.T) {
	for _, api := range []API{APIResponses, APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			llm, err := NewModel(t.Context(), "gpt-4o", &ClientConfig{APIKey: "test", API: api})
			if err != nil {
				t.Fatalf("NewModel() err = %v", err)
			}
			filter, ok := llm.(llminternal.RunLabelsFilter)
			if !ok {
				t.Fatalf("NewModel() returned %T, which does not implement llminternal.RunLabelsFilter", llm)
			}
			if got := filter.RunLabels(map[string]string{"cost_center": "x"}); got != nil {
				t.Errorf("RunLabels() = %v, want nil", got)
			}
		})
	}
}
