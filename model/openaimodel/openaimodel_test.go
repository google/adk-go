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
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"

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
			for range llm.GenerateContent(t.Context(), req, false) {
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
