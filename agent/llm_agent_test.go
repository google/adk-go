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
	"context"
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
	"github.com/google/adk-go/session"
	"github.com/google/adk-go/tool"
	"google.golang.org/genai"
)

//go:generate go test -httprecord=Test

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

func TestFunctionTool(t *testing.T) {
	modelName := "gemini-2.0-flash"
	model := newGeminiModel(t, modelName, nil)

	type Args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	type Result struct {
		Sum int `json:"sum"`
	}

	prompt := "what is the sum of 1 + 2?"
	handler := func(_ context.Context, input Args) Result {
		if input.A != 1 || input.B != 2 {
			t.Errorf("handler received %+v, want {a: 1, b: 2}", input)
		}
		return Result{Sum: input.A + input.B}
	}
	rand, _ := tool.NewFunctionTool(tool.FunctionToolConfig{
		Name:        "sum",
		Description: "computes the sum of two numbers",
	}, handler)
	agent := &agent.LLMAgent{
		AgentName:        "agent",
		AgentDescription: "math agent",
		Model:            model,
		Instruction:      "output ONLY the result computed by the provided function",
		Tools:            []adk.Tool{rand},
		// TODO(hakim): set to false when autoflow is implemented.
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
	}

	runner := newTestAgentRunner(t, agent)
	stream := runner.Run(t, "session1", prompt)
	ans, err := collectTextParts(stream)
	if err != nil || len(ans) == 0 {
		t.Fatalf("agent returned (%v, %v), want result", ans, err)
	}
	if got, want := strings.TrimSpace(ans[len(ans)-1]), "3"; got != want {
		t.Errorf("unexpected result from agent = (%v, %v), want ([%q], nil)", ans, err, want)
	}
}

type testAgentRunner struct {
	agent          adk.Agent
	sessionService adk.SessionService
}

func (r *testAgentRunner) Run(t *testing.T, sessionID, newMessage string) adk.EventStream {
	t.Helper()
	ctx := t.Context()
	session, _ := r.sessionService.Create(ctx, &adk.SessionCreateRequest{
		AppName:   "test",
		UserID:    "user",
		SessionID: sessionID,
	})

	return func(yield func(*adk.Event, error) bool) {
		ctx, inv := adk.NewInvocationContext(ctx, r.agent)
		inv.SessionService = r.sessionService
		inv.Session = session
		defer inv.End(nil)

		userMessageEvent := adk.NewEvent(inv.InvocationID)
		userMessageEvent.Author = "user"
		userMessageEvent.LLMResponse = &adk.LLMResponse{
			Content: genai.NewContentFromText(newMessage, "user"),
		}
		r.sessionService.AppendEvent(ctx, session, userMessageEvent)

		stream, err := r.agent.Run(ctx, inv)
		if err != nil {
			t.Fatal(err)
		}
		for ev, err := range stream {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := r.sessionService.AppendEvent(ctx, session, ev); err != nil {
				t.Fatalf("failed to record event %v: %v", ev, err)
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

func newTestAgentRunner(t *testing.T, agent adk.Agent) *testAgentRunner {
	return &testAgentRunner{
		agent:          agent,
		sessionService: &session.InMemorySessionService{},
	}
}

func newGeminiModel(t *testing.T, modelName string, transport http.RoundTripper) *model.GeminiModel {
	apiKey := "fakeKey"
	if transport == nil { // use httprr
		trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
		recording := false
		transport, recording = newGeminiTestClientConfig(t, trace)
		if recording { // if we are recording httprr trace, don't use the fakeKey.
			apiKey = ""
		}
	}
	model, err := model.NewGeminiModel(t.Context(), modelName, &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: transport},
		APIKey:     apiKey,
	})
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
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
