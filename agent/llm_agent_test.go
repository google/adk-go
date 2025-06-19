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

package agent_test

import (
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/adk-go/agent"
	"github.com/google/adk-go/internal/httprr"
	"github.com/google/adk-go/model"
	"google.golang.org/genai"
)

//go:generate go test -httprecord=TestLLMAgent

func TestLLMAgent(t *testing.T) {
	ctx := t.Context()
	modelName := "gemini-2.0-flash"
	errNoNetwork := errors.New("no network")

	for _, tc := range []struct {
		name      string
		transport http.RoundTripper
		wantErr   error
	}{
		{
			name:      "healthy_backend",
			transport: nil, // httprr + http.DefaultTransport
		},
		{
			name:      "broken_backed",
			transport: roundTripperFunc(func(*http.Request) (*http.Response, error) { return nil, errNoNetwork }),
			wantErr:   errNoNetwork,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newGeminiModel(t, modelName, tc.transport)
			a := &agent.LLMAgent{
				AgentName:         "hello_world_agent",
				AgentDescription:  "hello world agent",
				Model:             model,
				Instruction:       "Roll the dice and report only the result.",
				GlobalInstruction: "Answer as precisely as possible.",

				// TODO: set tools, planner.

				DisallowTransferToParent: true,
				DisallowTransferToPeers:  true,
			}
			stream, err := a.Run(ctx, &adk.InvocationContext{
				InvocationID: "12345",
				Agent:        a,
			})
			// TODO: do we want to make a.Run return just adk.EventStream?
			if err != nil {
				t.Fatalf("failed to run agent: %v", err)
			}
			texts, err := collectTextParts(stream)
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("stream = (%q, %v), want (_, %v)", texts, err, tc.wantErr)
			}
			if tc.wantErr == nil && (err != nil || len(texts) != 1) {
				t.Fatalf("stream = (%q, %v), want exactly one text response", texts, err)
			}
		})
	}
}

func newGeminiModel(t *testing.T, modelName string, transport http.RoundTripper) *model.GeminiModel {
	apiKey := "fakekey"
	if transport == nil { // use httprr
		trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
		recording := false
		transport, recording = newGeminiTestClientConfig(t, trace)
		if recording { // if we are recording httprr trace, don't use the fakkey.
			apiKey = ""
		}
	}
	model, err := model.NewGeminiModel(t.Context(), modelName, &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: transport},
		APIKey:     apiKey,
	})
	if err != nil {
		t.Fatalf("failedto create model: %v", err)
	}
	return model
}

// collectTextParts collects all text parts from the llm response until encountering an error.
// It returns all collected text parts and the last error.
func collectTextParts(stream adk.EventStream) ([]string, error) {
	var texts []string
	for ev, err := range stream {
		if err != nil {
			return texts, err
		}
		if ev == nil || ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
			return texts, fmt.Errorf("unexpected empty event: %v", ev)
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
	}
	return texts, nil
}

func newGeminiTestClientConfig(t *testing.T, rrfile string) (http.RoundTripper, bool) {
	t.Helper()
	rr, err := httprr.NewGeminiTransportForTesting(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	recording, _ := httprr.Recording(rrfile)
	return rr, recording
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

// RoundTrip implements http.RoundTripper.
func (fn roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}
