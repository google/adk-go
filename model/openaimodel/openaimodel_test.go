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
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel/internal/openaicommon"
	"google.golang.org/adk/v2/model/openaimodel/internal/responses"
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
		"ErrModelNameRequired":              {ErrModelNameRequired, openaicommon.ErrModelNameRequired},
		"ErrUnsupportedAPI":                 {ErrUnsupportedAPI, openaicommon.ErrUnsupportedAPI},
		"ErrNoChoices":                      {ErrNoChoices, openaicommon.ErrNoChoices},
		"ErrRequestNil":                     {ErrRequestNil, openaicommon.ErrRequestNil},
		"ErrNoContents":                     {ErrNoContents, openaicommon.ErrNoContents},
		"ErrFunctionCallMissingName":        {ErrFunctionCallMissingName, openaicommon.ErrFunctionCallMissingName},
		"ErrTopKNotSupported":               {ErrTopKNotSupported, openaicommon.ErrTopKNotSupported},
		"ErrStopSequencesNotSupported":      {ErrStopSequencesNotSupported, openaicommon.ErrStopSequencesNotSupported},
		"ErrMultipleCandidatesNotSupported": {ErrMultipleCandidatesNotSupported, openaicommon.ErrMultipleCandidatesNotSupported},
		"ErrPenaltiesNotSupported":          {ErrPenaltiesNotSupported, openaicommon.ErrPenaltiesNotSupported},
		"ErrLabelsNotSupported":             {ErrLabelsNotSupported, openaicommon.ErrLabelsNotSupported},
		"ErrSafetySettingsNotSupported":     {ErrSafetySettingsNotSupported, openaicommon.ErrSafetySettingsNotSupported},
		"ErrUnsupportedMIMEType":            {ErrUnsupportedMIMEType, openaicommon.ErrUnsupportedMIMEType},
		"ErrUnsupportedConfigField":         {ErrUnsupportedConfigField, openaicommon.ErrUnsupportedConfigField},
		"ErrEmptyJSONSchema":                {ErrEmptyJSONSchema, openaicommon.ErrEmptyJSONSchema},
		"ErrEmptyResponse":                  {ErrEmptyResponse, openaicommon.ErrEmptyResponse},
		"ErrNoOutputItems":                  {ErrNoOutputItems, openaicommon.ErrNoOutputItems},
		"ErrUnsupportedMessageContentType":  {ErrUnsupportedMessageContentType, openaicommon.ErrUnsupportedMessageContentType},
		"ErrUnsupportedOutputItemType":      {ErrUnsupportedOutputItemType, openaicommon.ErrUnsupportedOutputItemType},
		"ErrFunctionCallArgs":               {ErrFunctionCallArgs, openaicommon.ErrFunctionCallArgs},
		"ErrNoTextOrToolContent":            {ErrNoTextOrToolContent, openaicommon.ErrNoTextOrToolContent},
		"ErrResponseFailed":                 {ErrResponseFailed, openaicommon.ErrResponseFailed},
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
					// A 400 is not retried, so the one request is the one checked.
					w.WriteHeader(http.StatusBadRequest)
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
				// The call fails by design; only the headers the server saw matter.
				for range llm.GenerateContent(t.Context(), req, stream) {
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

// newLoopbackServer serves handler on IPv4 loopback, which the sandboxes these
// tests run in allow where the default listener's IPv6 may be refused.
func newLoopbackServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(handler)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	server.Listener = ln
	server.Start()
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
