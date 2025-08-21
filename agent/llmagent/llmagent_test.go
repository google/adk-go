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

package llmagent_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"

	"google.golang.org/adk/llm"
	"google.golang.org/adk/llm/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

const modelName = "gemini-2.0-flash"

//go:generate go test -httprecord=Test

func TestLLMAgent(t *testing.T) {
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
			name:      "broken_backend",
			transport: roundTripperFunc(func(*http.Request) (*http.Response, error) { return nil, errNoNetwork }),
			wantErr:   errNoNetwork,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newGeminiModel(t, modelName, tc.transport)
			a, err := llmagent.New(llmagent.Config{
				Name:                     "hello_world_agent",
				Description:              "hello world agent",
				Model:                    model,
				Instruction:              "Roll the dice and report only the result.",
				GlobalInstruction:        "Answer as precisely as possible.",
				DisallowTransferToParent: true,
				DisallowTransferToPeers:  true,
			})
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}
			// TODO: set tools, planner.
			runner := newTestAgentRunner(t, a)
			stream := runner.Run(t, "test_session", "")
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

func TestModelCallbacks(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name                 string
		llmResponses         []*genai.Content
		beforeModelCallbacks []llmagent.BeforeModelCallback
		afterModelCallbacks  []llmagent.AfterModelCallback
		wantTexts            []string
		wantErr              error
	}{
		{
			name: "before model callback doesn't modify anything",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return nil, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "before model callback returns an error",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "before model callback returns new LLMResponse",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantTexts: []string{
				"hello from before_model_callback",
			},
		},
		{
			name: "before model callback returns both new LLMResponse and error",
			beforeModelCallbacks: []llmagent.BeforeModelCallback{
				func(ctx agent.Context, llmRequest *llm.Request) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "after model callback doesn't modify anything",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return nil, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "after model callback returns new LLMResponse",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantTexts: []string{
				"hello from after_model_callback",
			},
		},
		{
			name: "after model callback returns error",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "after model callback returns both new LLMResponse and error",
			afterModelCallbacks: []llmagent.AfterModelCallback{
				func(ctx agent.Context, llmResponse *llm.Response, llmResponseError error) (*llm.Response, error) {
					return &llm.Response{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []*genai.Content{
				genai.NewContentFromText("hello from model", genai.RoleModel),
			},
			wantErr: http.ErrNoCookie,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &mockModel{
				responses: tc.llmResponses,
			}
			a, err := llmagent.New(llmagent.Config{
				Name:        "hello_world_agent",
				Model:       model,
				BeforeModel: tc.beforeModelCallbacks,
				AfterModel:  tc.afterModelCallbacks,
			})
			if err != nil {
				t.Fatalf("failed to create llm agent: %v", err)
			}
			runner := newTestAgentRunner(t, a)
			stream := runner.Run(t, "test_session", "")
			texts, err := collectTextParts(stream)
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("stream = (%q, %v), want (_, %v)", texts, err, tc.wantErr)
			}
			if (err != nil) != (tc.wantErr != nil) {
				t.Fatalf("unexpected result from agent, got error: %v, want error: %v", err, tc.wantErr)
			}

			if diff := cmp.Diff(tc.wantTexts, texts); diff != "" {
				t.Fatalf("unexpected result from agent, want: %v, got: %v, diff: %v", tc.wantTexts, texts, diff)
			}
		})
	}
}

func TestFunctionTool(t *testing.T) {
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

	agent, err := llmagent.New(llmagent.Config{
		Name:        "agent",
		Description: "math agent",
		Model:       model,
		Instruction: "output ONLY the result computed by the provided function",
		// TODO(hakim): set to false when autoflow is implemented.
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
		Tools:                    []tool.Tool{rand},
	})
	if err != nil {
		t.Fatalf("failed to create LLM Agent: %v", err)
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

func TestAgentTransfer(t *testing.T) {
	// Helpers to create genai.Content conveniently.
	transferCall := func(agentName string) *genai.Content {
		return genai.NewContentFromFunctionCall(
			"transfer_to_agent",
			map[string]any{"agent_name": agentName},
			"model",
		)
	}
	transferResponse := func() *genai.Content {
		return genai.NewContentFromFunctionResponse(
			"transfer_to_agent", map[string]any{}, "user")
	}
	text := func(text string) *genai.Content {
		return genai.NewContentFromText(
			text,
			"model",
		)
	}
	// returns a model that returns the prepopulated resp one by one.
	testModel := func(resp ...*genai.Content) llm.Model {
		return &mockModel{responses: resp}
	}

	type content struct {
		Author string
		Parts  []*genai.Part
	}
	// contents returns (Author, Parts) stream extracted from the event stream.
	contents := func(stream iter.Seq2[*session.Event, error]) ([]content, error) {
		var ret []content
		for ev, err := range stream {
			if err != nil {
				return nil, err
			}
			if ev.LLMResponse == nil || ev.LLMResponse.Content == nil {
				return nil, fmt.Errorf("unexpected event: %v", ev)
			}
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.FunctionCall != nil {
					p.FunctionCall.ID = ""
				}
				if p.FunctionResponse != nil {
					p.FunctionResponse.ID = ""
				}
			}
			ret = append(ret, content{Author: ev.Author, Parts: ev.LLMResponse.Content.Parts})
		}
		return ret, nil
	}

	check := func(t *testing.T, rootAgent agent.Agent, wants [][]content) {
		runner := newTestAgentRunner(t, rootAgent)
		for i := range len(wants) {
			got, err := contents(runner.Run(t, "session_id", fmt.Sprintf("round %d", i)))
			if err != nil {
				t.Fatalf("[round $d]: stream ended with an error: %v", err)
			}
			if diff := cmp.Diff(wants[i], got); diff != "" {
				t.Errorf("[round %d] events diff (-want, +got) = %v", i, diff)
			}
		}
	}

	t.Run("auto_to_auto", func(t *testing.T) {
		// root_agent -- sub_agent_1
		model := testModel(
			transferCall("sub_agent_1"),
			text("response1"),
			text("response2"))

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:  "sub_agent_1",
			Model: model,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", text("response1").Parts},
			},
			1: { // rootAgent should still be the current agent.
				{"sub_agent_1", text("response2").Parts},
			},
		})
	})

	t.Run("auto_to_single", func(t *testing.T) {
		// root_agent -- sub_agent_1 (single)
		model := testModel(
			transferCall("sub_agent_1"),
			text("response1"),
			text("response2"))

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:                     "sub_agent_1",
			Model:                    model,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", text("response1").Parts},
			},
			1: { // rootAgent should still be the current agent.
				{"root_agent", text("response2").Parts},
			},
		})
	})

	t.Run("auto_to_auto_to_single", func(t *testing.T) {
		// root_agent -- sub_agent_1 -- sub_agent_1_1
		model := testModel(
			transferCall("sub_agent_1"),
			transferCall("sub_agent_1_1"),
			text("response1"),
			text("response2"))

		subAgent1_1, err := llmagent.New(llmagent.Config{
			Name:                     "sub_agent_1_1",
			Model:                    model,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1_1: %v", err)
		}

		subAgent1, err := llmagent.New(llmagent.Config{
			Name:      "sub_agent_1",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1_1},
		})
		if err != nil {
			t.Fatalf("failed to create subAgent1: %v", err)
		}

		rootAgent, err := llmagent.New(llmagent.Config{
			Name:      "root_agent",
			Model:     model,
			SubAgents: []agent.Agent{subAgent1},
		})
		if err != nil {
			t.Fatalf("failed to create rootAgent: %v", err)
		}

		check(t, rootAgent, [][]content{
			0: {
				{"root_agent", transferCall("sub_agent_1").Parts},
				{"root_agent", transferResponse().Parts},
				{"sub_agent_1", transferCall("sub_agent_1_1").Parts},
				{"sub_agent_1", transferResponse().Parts},
				{"sub_agent_1_1", text("response1").Parts},
			},
			1: {
				// sub_agent_1 should still be the current agent.
				// sub_agent_1_1 is single, so it should not be the current agent.
				// Otherwise, the conversation will be tied to sub_agent_1_1 forever.
				{"sub_agent_1", text("response2").Parts},
			},
		})
	})

	// TODO: cover cases similar to adk-python's
	// tests/unittests/flows/llm_flows/test_agent_transfer.py
	//   - test_auto_to_sequential
	//   - test_auto_to_sequential_to_auto
	//   - test_auto_to_loop
}

// TODO(hakim): move testAgentRunner to an internal test utility package.
// See adk-python's tests/unittests/testing_utils.py.
type testAgentRunner struct {
	agent          agent.Agent
	sessionService sessionservice.Service
	lastSession    sessionservice.StoredSession
	appName        string
	// TODO: move runner definition to the adk package and it's a part of public api, but the logic to the internal runner
	runner *runner.Runner
}

func (r *testAgentRunner) session(t *testing.T, appName, userID, sessionID string) (sessionservice.StoredSession, error) {
	ctx := t.Context()
	if last := r.lastSession; last != nil && last.ID().SessionID == sessionID {
		session, err := r.sessionService.Get(ctx, &sessionservice.GetRequest{
			ID: session.ID{
				AppName:   "test_app",
				UserID:    "test_user",
				SessionID: sessionID,
			},
		})
		r.lastSession = session
		return session, err
	}
	session, err := r.sessionService.Create(ctx, &sessionservice.CreateRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: sessionID,
	})
	r.lastSession = session
	return session, err
}

func (r *testAgentRunner) Run(t *testing.T, sessionID, newMessage string) iter.Seq2[*session.Event, error] {
	t.Helper()
	ctx := t.Context()

	userID := "test_user"

	session, err := r.session(t, r.appName, userID, sessionID)
	if err != nil {
		t.Fatalf("failed to get/create session: %v", err)
	}

	var content *genai.Content
	if newMessage != "" {
		content = genai.NewContentFromText(newMessage, genai.RoleUser)
	}

	return r.runner.Run(ctx, userID, session.ID().SessionID, content, &types.AgentRunConfig{})
}

func newTestAgentRunner(t *testing.T, agent agent.Agent) *testAgentRunner {
	appName := "test_app"
	sessionService := sessionservice.Mem()

	runner, err := runner.New(appName, agent, sessionService)
	if err != nil {
		t.Fatal(err)
	}

	return &testAgentRunner{
		agent:          agent,
		sessionService: sessionService,
		appName:        appName,
		runner:         runner,
	}
}

type mockModel struct {
	responses []*genai.Content
}

var errNoModelData = errors.New("no data")

// GenerateContent implements llm.Model.
func (m *mockModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if len(m.responses) == 0 {
		return nil, errNoModelData
	}

	resp := &llm.Response{
		Content: m.responses[0],
	}

	m.responses = m.responses[1:]

	return resp, nil
}

func (m *mockModel) GenerateStream(ctx context.Context, req *llm.Request) iter.Seq2[*llm.Response, error] {
	return func(yield func(*llm.Response, error) bool) {
		if len(m.responses) > 0 {
			resp := &llm.Response{Content: m.responses[0]}
			m.responses = m.responses[1:]
			yield(resp, nil)
			return
		}
		yield(nil, fmt.Errorf("no more data"))
	}
}

// Name implements llm.Model.
func (m *mockModel) Name() string {
	return "mock"
}

var _ llm.Model = (*mockModel)(nil)

func newGeminiModel(t *testing.T, modelName string, transport http.RoundTripper) *gemini.Model {
	apiKey := "fakeKey"
	if transport == nil { // use httprr
		trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
		recording := false
		transport, recording = newGeminiTestClientConfig(t, trace)
		if recording { // if we are recording httprr trace, don't use the fakeKey.
			apiKey = ""
		}
	}
	model, err := gemini.NewModel(t.Context(), modelName, &genai.ClientConfig{
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
func collectTextParts(stream iter.Seq2[*session.Event, error]) ([]string, error) {
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
	rr, err := testutil.NewGeminiTransport(rrfile)
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
