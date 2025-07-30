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
	"iter"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/adk-go/agent"
	"github.com/google/adk-go/internal/httprr"
	"github.com/google/adk-go/model"
	"github.com/google/adk-go/session"
	"github.com/google/adk-go/tool"
	"github.com/google/go-cmp/cmp"
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
			a, err := agent.NewLLMAgent("hello_world_agent", model,
				agent.WithDescription("hello world agent"))
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}
			a.Instruction = "Roll the dice and report only the result."
			a.GlobalInstruction = "Answer as precisely as possible."
			a.DisallowTransferToParent = true
			a.DisallowTransferToPeers = true
			/* TODO: alternative
			   a := agent.NewLLMAgent("hello_world_agent",
			        agent.WithDescription("hello world agent"),
					agent.WithModel(m),
					agent.WithInstruction("Roll the dice and report only the result.")
					agent.WithGlobalInstruction("Answer as precisely as possible."),
					agent.WithDisallowTransferToParent(true),
					agent.WithDisallowTransferToPeers(true),
				}
			*/
			// TODO: set tools, planner.
			ctx, invCtx := adk.NewInvocationContext(t.Context(), a, nil, nil, nil, nil)
			stream := a.Run(ctx, invCtx)
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
		llmResponses         []llmResponse
		beforeModelCallbacks []agent.BeforeModelCallback
		afterModelCallbacks  []agent.AfterModelCallback
		wantTexts            []string
		wantErr              error
	}{
		{
			name: "before model callback doesn't modify anything",
			beforeModelCallbacks: []agent.BeforeModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return nil, nil
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "before model callback returns an error",
			beforeModelCallbacks: []agent.BeforeModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return nil, fmt.Errorf("before_model_callback_error: %w", http.ErrHijacked)
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "before model callback returns new LLMResponse",
			beforeModelCallbacks: []agent.BeforeModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantTexts: []string{
				"hello from before_model_callback",
			},
		},
		{
			name: "before model callback returns both new LLMResponse and error",
			beforeModelCallbacks: []agent.BeforeModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmRequest *adk.LLMRequest) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from before_model_callback", genai.RoleModel),
					}, fmt.Errorf("before_model_callback_error: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "after model callback doesn't modify anything",
			afterModelCallbacks: []agent.AfterModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return nil, nil
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantTexts: []string{
				"hello from model",
			},
		},
		{
			name: "after model callback returns new LLMResponse",
			afterModelCallbacks: []agent.AfterModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, nil
				},
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("unexpected text", genai.RoleModel),
					}, nil
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantTexts: []string{
				"hello from after_model_callback",
			},
		},
		{
			name: "after model callback returns error",
			afterModelCallbacks: []agent.AfterModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return nil, fmt.Errorf("error from after_model_callback: %w", http.ErrHijacked)
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantErr: http.ErrNoCookie,
		},
		{
			name: "after model callback returns both new LLMResponse and error",
			afterModelCallbacks: []agent.AfterModelCallback{
				func(ctx context.Context, callbackCtx *adk.CallbackContext, llmResponse *adk.LLMResponse, llmResponseError error) (*adk.LLMResponse, error) {
					return &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from after_model_callback", genai.RoleModel),
					}, fmt.Errorf("error from after_model_callback: %w", http.ErrNoCookie)
				},
			},
			llmResponses: []llmResponse{
				{
					resp: &adk.LLMResponse{
						Content: genai.NewContentFromText("hello from model", genai.RoleModel),
					},
				},
			},
			wantErr: http.ErrNoCookie,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeModel{
				records: tc.llmResponses,
			}
			a, err := agent.NewLLMAgent("hello_world_agent", model,
				agent.WithBeforeModelCallbacks(tc.beforeModelCallbacks...),
				agent.WithAfterModelCallbacks(tc.afterModelCallbacks...),
			)
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}
			ctx, invCtx := adk.NewInvocationContext(t.Context(), a, nil, nil, nil, nil)
			stream := a.Run(ctx, invCtx)
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
	agent, err := agent.NewLLMAgent("agent", model, agent.WithDescription("math agent"))
	if err != nil {
		t.Fatalf("NewLLMAgent failed: %v", err)
	}
	agent.Instruction = "output ONLY the result computed by the provided function"
	agent.Tools = []adk.Tool{rand}
	// TODO(hakim): set to false when autoflow is implemented.
	agent.DisallowTransferToParent = true
	agent.DisallowTransferToPeers = true

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
	testModel := func(resp ...*genai.Content) adk.Model {
		return &mockModel{responses: resp}
	}
	// creates an LLM model with the name and the model.
	llmAgentFn := func(t *testing.T) func(name string, model adk.Model, opts ...agent.AgentOption) *agent.LLMAgent {
		return func(name string, model adk.Model, opts ...agent.AgentOption) *agent.LLMAgent {
			a, err := agent.NewLLMAgent(name, model, opts...)
			if err != nil {
				t.Fatalf("NewLLMAgent failed: %v", err)
			}
			return a
		}
	}

	type content struct {
		Author string
		Parts  []*genai.Part
	}
	// contents returns (Author, Parts) stream extracted from the event stream.
	contents := func(stream iter.Seq2[*adk.Event, error]) ([]content, error) {
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

	check := func(t *testing.T, rootAgent adk.Agent, wants [][]content) {
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
		llmAgent := llmAgentFn(t)

		subAgent1 := llmAgent("sub_agent_1", model)

		rootAgent := llmAgent("root_agent", model, agent.WithSubAgents(subAgent1))

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
		llmAgent := llmAgentFn(t)

		subAgent1 := llmAgent("sub_agent_1", model)
		subAgent1.DisallowTransferToParent = true
		subAgent1.DisallowTransferToPeers = true

		rootAgent := llmAgent("root_agent", model, agent.WithSubAgents(subAgent1))

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
		llmAgent := llmAgentFn(t)

		subAgent1_1 := llmAgent("sub_agent_1_1", model)
		subAgent1_1.DisallowTransferToParent = true
		subAgent1_1.DisallowTransferToPeers = true

		subAgent1 := llmAgent("sub_agent_1", model, agent.WithSubAgents(subAgent1_1))

		rootAgent := llmAgent("root_agent", model, agent.WithSubAgents(subAgent1))

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
	agent          adk.Agent
	sessionService adk.SessionService
	lastSession    *adk.Session
}

func (r *testAgentRunner) session(t *testing.T, sessionID string) (*adk.Session, error) {
	ctx := t.Context()
	if last := r.lastSession; last != nil && last.ID == sessionID {
		session, err := r.sessionService.Get(ctx, &adk.SessionGetRequest{
			AppName:   "test_app",
			UserID:    "test_user",
			SessionID: sessionID,
		})
		r.lastSession = session
		return session, err
	}
	session, err := r.sessionService.Create(ctx, &adk.SessionCreateRequest{
		AppName:   "test_app",
		UserID:    "test_user",
		SessionID: sessionID,
	})
	r.lastSession = session
	return session, err
}

func (r *testAgentRunner) Run(t *testing.T, sessionID, newMessage string) iter.Seq2[*adk.Event, error] {
	t.Helper()
	ctx := t.Context()
	session, err := r.session(t, sessionID)
	if err != nil {
		t.Fatalf("failed to get/create session: %v", err)
	}

	// TODO: replace this with the real runner.
	return func(yield func(*adk.Event, error) bool) {
		ctx, inv := adk.NewInvocationContext(ctx, r.agent, nil, nil, nil, nil)
		inv.SessionService = r.sessionService
		inv.Session = session
		defer inv.End(nil)

		userMessageEvent := adk.NewEvent(inv.InvocationID)
		userMessageEvent.Author = "user"
		userMessageEvent.LLMResponse = &adk.LLMResponse{
			Content: genai.NewContentFromText(newMessage, "user"),
		}
		r.sessionService.AppendEvent(ctx, session, userMessageEvent)

		agentToRun := r.findAgentToRun(session, r.agent)
		for ev, err := range agentToRun.Run(ctx, inv) {
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

func (r *testAgentRunner) findAgentToRun(s *adk.Session, rootAgent adk.Agent) adk.Agent {
	// runner.py Runner's _find_agent_to_run.

	// TODO: findMatchingFunctionCall.

	for _, ev := range slices.Backward(s.Events) {
		if ev.Author == rootAgent.Spec().Name {
			// Found root agent.
			return rootAgent
		}
		matching := findSubAgent(rootAgent, ev.Author)
		if matching == nil {
			// event from an unknown agent.
			// TODO: log.
			continue
		}
		if r.isTransferableAcrossAgentTree(matching) {
			return matching
		}
	}
	return rootAgent
}

// isTransferableAcrossAgentTree returns whether the agent
// to run can transfer to any other agent in the agent tree.
// This typicall means all agentToRun's parents through root agent
// can transfer to their parent agents.
func (r *testAgentRunner) isTransferableAcrossAgentTree(agentToRun adk.Agent) bool {
	for {
		if agentToRun == nil {
			return true
		}
		agent, ok := agentToRun.(*agent.LLMAgent)
		if !ok {
			return false // only LLMAgent can provide agent transfer capability.
		}
		if agent.DisallowTransferToParent {
			return false
		}
		agentToRun = agent.Spec().Parent()
	}
}

func findAgent(agent adk.Agent, name string) adk.Agent {
	if agent.Spec().Name == name {
		return agent
	}
	return findSubAgent(agent, name)
}

func findSubAgent(a adk.Agent, name string) adk.Agent {
	llmAgent, ok := a.(*agent.LLMAgent)
	if !ok {
		return nil
	}

	for _, sub := range llmAgent.Spec().SubAgents {
		return findAgent(sub, name)
	}
	return nil
}

type mockModel struct {
	responses []*genai.Content
}

// GenerateContent implements adk.Model.
func (m *mockModel) GenerateContent(ctx context.Context, req *adk.LLMRequest, stream bool) adk.LLMResponseStream {
	return func(yield func(*adk.LLMResponse, error) bool) {
		if len(m.responses) > 0 {
			resp := &adk.LLMResponse{Content: m.responses[0]}
			m.responses = m.responses[1:]
			yield(resp, nil)
			return
		}
		yield(nil, fmt.Errorf("no more data"))
	}
}

// Name implements adk.Model.
func (m *mockModel) Name() string {
	return "mock"
}

var _ adk.Model = (*mockModel)(nil)

func newTestAgentRunner(_ *testing.T, agent adk.Agent) *testAgentRunner {
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
func collectTextParts(stream iter.Seq2[*adk.Event, error]) ([]string, error) {
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

type llmResponse struct {
	resp *adk.LLMResponse
	err  error
}

type fakeModel struct {
	records []llmResponse
}

func (m *fakeModel) Name() string { return "fakeModel" }

func (m *fakeModel) GenerateContent(ctx context.Context, req *adk.LLMRequest, stream bool) adk.LLMResponseStream {
	return func(yield func(*adk.LLMResponse, error) bool) {
		for _, r := range m.records {
			if !yield(r.resp, r.err) {
				return
			}
		}
	}
}
