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

package remoteagent

//go:generate go test -httprecord=.*

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	"github.com/sjzsdu/adk-go/agent/llmagent"
	"github.com/sjzsdu/adk-go/agent/workflowagents/sequentialagent"
	"github.com/sjzsdu/adk-go/internal/converters"
	"github.com/sjzsdu/adk-go/internal/httprr"
	"github.com/sjzsdu/adk-go/internal/testutil"
	"github.com/sjzsdu/adk-go/internal/utils"
	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/model/gemini"
	"github.com/sjzsdu/adk-go/runner"
	"github.com/sjzsdu/adk-go/server/adka2a"
	"github.com/sjzsdu/adk-go/session"
	"github.com/sjzsdu/adk-go/tool"
	"github.com/sjzsdu/adk-go/tool/functiontool"
)

const (
	approvalToolName            = "request_approval"
	modelTextRequiresApproval   = "need to request approval first!"
	modelTextWaitingForApproval = "waiting for user's approval..."
	modelTextTaskComplete       = "Task complete!"

	transferToolName      = "transfer_to_agent"
	modelTextRootTransfer = "transfering... please hold... beepboop..."
)

type approvalStatus string

var (
	approvalStatusPending  approvalStatus = "pending"
	approvalStatusApproved approvalStatus = "approved"
	approvalStatusVerified approvalStatus = "verified"
)

type approval struct {
	Status   approvalStatus `json:"status"`
	TicketID string         `json:"ticket_id"`
}

/**
 * a2aclient -> a2aserver -> adka2a.Executor -> llmagent with a long running tool
 */
func TestA2AInputRequired(t *testing.T) {
	testCases := []struct {
		name                    string
		tool                    tool.Tool
		createApproval          func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part
		wantFirstArtifactParts  a2a.ContentParts
		wantSecondArtifactParts a2a.ContentParts
	}{
		{
			name: "long-running",
			tool: newLongRunningTool(t),
			createApproval: func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part {
				return createLongRunningToolApproval(t, pendingResponse)
			},
			wantFirstArtifactParts: a2a.ContentParts{
				a2a.TextPart{Text: modelTextRequiresApproval},
				a2a.TextPart{Text: modelTextWaitingForApproval},
			},
			wantSecondArtifactParts: a2a.ContentParts{a2a.TextPart{Text: modelTextTaskComplete}},
		},
		{
			name: "tool confirmation",
			tool: newToolConfirmation(t),
			createApproval: func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part {
				return createToolConfirmationApproval(t, toolCall)
			},
			wantFirstArtifactParts: a2a.ContentParts{
				a2a.TextPart{Text: modelTextRequiresApproval},
				a2a.DataPart{
					Data:     map[string]any{"name": approvalToolName},
					Metadata: map[string]any{"adk_is_long_running": false, "adk_type": "function_call"},
				},
				a2a.DataPart{
					Data: map[string]any{
						"name":     approvalToolName,
						"response": map[string]any{"status": string(approvalStatusPending)},
					},
					Metadata: map[string]any{"adk_type": "function_response"},
				},
			},
			wantSecondArtifactParts: a2a.ContentParts{
				a2a.DataPart{
					Data: map[string]any{
						"name":     approvalToolName,
						"response": map[string]any{"status": string(approvalStatusVerified)},
					},
					Metadata: map[string]any{"adk_type": "function_response"},
				},
				a2a.TextPart{Text: modelTextTaskComplete},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Server
			inputRequestingAgent := newInputRequestingAgent(t, "agent-b", tc.tool)
			executor := newAgentExecutor(inputRequestingAgent, nil)
			server := startA2AServer(executor)
			defer server.Close()

			// Client
			client := newA2AClient(t, server)

			// Initial message triggers input required
			taskContent := "Perform important task!"
			msg1 := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: taskContent})
			task1 := mustSendMessage(t, client, msg1)
			if task1.Status.State != a2a.TaskStateInputRequired {
				t.Fatalf("client.SendMessage(Initial) result state = %q, want %q", task1.Status.State, a2a.TaskStateInputRequired)
			}
			if len(task1.Artifacts) != 1 {
				t.Fatalf("len(task.Artifacts) = %d, want 1", len(task1.Artifacts))
			}

			// Incomplete followup keeps the task in input-required
			incompleteFollowupText := "Is it really necessary?"
			msg2 := a2a.NewMessageForTask(a2a.MessageRoleUser, task1, a2a.TextPart{Text: incompleteFollowupText})
			task2 := mustSendMessage(t, client, msg2)
			if task2.Status.State != a2a.TaskStateInputRequired {
				t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task2.Status.State, a2a.TaskStateInputRequired)
			}
			if len(task2.Artifacts) != 1 {
				t.Fatalf("len(task.Artifacts) = %d, want 1", len(task2.Artifacts))
			}

			// Required input gets delivered

			// Verify that error message is present
			if len(task2.Status.Message.Parts) < 2 {
				t.Fatalf("task2.Status.Message.Parts len = %d; want >= 2", len(task2.Status.Message.Parts))
			}
			// The last part should be the error message
			lastPart := task2.Status.Message.Parts[len(task2.Status.Message.Parts)-1]
			tp, ok := lastPart.(a2a.TextPart)
			if !ok {
				t.Fatalf("last part is not TextPart")
			}
			if !strings.Contains(tp.Text, "no input provided") {
				t.Errorf("last part text = %q; want it to contain 'no input provided'", tp.Text)
			}

			// Another incomplete followup should not accumulate error messages
			msg2a := a2a.NewMessageForTask(a2a.MessageRoleUser, task1, a2a.TextPart{Text: "Still debating?"})
			task2a := mustSendMessage(t, client, msg2a)
			if task2a.Status.State != a2a.TaskStateInputRequired {
				t.Fatalf("client.SendMessage(IncompleteInput 2) result state = %q, want %q", task2a.Status.State, a2a.TaskStateInputRequired)
			}

			// Count validation errors in parts
			validationErrors := 0
			for _, p := range task2a.Status.Message.Parts {
				if tp, ok := p.(a2a.TextPart); ok && strings.Contains(tp.Text, "no input provided") {
					validationErrors++
				}
			}
			if validationErrors != 1 {
				t.Errorf("validationErrors count = %d; want 1", validationErrors)
			}

			// Check for adk_request_confirmation
			toolCall, pendingResponse := findLongRunningCall(t, toGenaiParts(t, task2.Status.Message.Parts))
			approvedResponse := tc.createApproval(t, toolCall, pendingResponse)

			msg3 := a2a.NewMessageForTask(a2a.MessageRoleUser, task2,
				a2a.TextPart{Text: "LGTM"},
				toA2AParts(t, []*genai.Part{approvedResponse}, []string{toolCall.ID})[0],
			)
			task3 := mustSendMessage(t, client, msg3)
			if task3.Status.State != a2a.TaskStateCompleted {
				t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task3.Status.State, a2a.TaskStateCompleted)
			}

			// Verify the final task state
			opts := []cmp.Option{
				cmpopts.EquateEmpty(),
				cmpopts.IgnoreMapEntries(func(k string, v any) bool { return strings.HasSuffix(k, "id") }),
				cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			}
			if len(task3.Artifacts) != 2 {
				t.Fatalf("len(task.Artifacts) = %d, want 2", len(task3.Artifacts))
			}

			gotHistory := task3.History
			wantHistory := []*a2a.Message{msg1, msg2, task1.Status.Message, msg2a, task2a.Status.Message, msg3, task2a.Status.Message}
			if diff := cmp.Diff(wantHistory, gotHistory, opts...); diff != "" {
				t.Fatalf("unexpected history (+got,-want) diff:\n%s", diff)
			}

			gotFirstArtifactParts := adka2a.WithoutPartialArtifacts(task3.Artifacts)[0].Parts
			if diff := cmp.Diff(tc.wantFirstArtifactParts, gotFirstArtifactParts, opts...); diff != "" {
				t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
			}

			gotSecondArtifactParts := task3.Artifacts[1].Parts
			if diff := cmp.Diff(tc.wantSecondArtifactParts, gotSecondArtifactParts, opts...); diff != "" {
				t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
			}
		})
	}
}

/**
 * a2aclient -> server A -> adka2a.Executor A ->-> llmagent with remote subagent ->
 * 		remotesubagent -> server B -> adka2a.Executor B -> llmagent with a long running tool
 */
func TestA2AMultiHopInputRequired(t *testing.T) {
	remoteAgentName := "remote-agent-B"
	testCases := []struct {
		name                    string
		tool                    tool.Tool
		createApproval          func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part
		wantFirstArtifactParts  a2a.ContentParts
		wantSecondArtifactParts a2a.ContentParts
	}{
		{
			name: "long-running",
			tool: newLongRunningTool(t),
			createApproval: func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part {
				return createLongRunningToolApproval(t, pendingResponse)
			},
			wantFirstArtifactParts: toA2AParts(t, []*genai.Part{
				genai.NewPartFromText(modelTextRootTransfer),
				genai.NewPartFromFunctionCall(transferToolName, map[string]any{"agent_name": remoteAgentName}),
				genai.NewPartFromFunctionResponse(transferToolName, nil),
				genai.NewPartFromText(modelTextRequiresApproval),
				genai.NewPartFromText(modelTextWaitingForApproval),
			}, []string{}),
			wantSecondArtifactParts: a2a.ContentParts{
				a2a.TextPart{Text: modelTextTaskComplete},
			},
		},
		{
			name: "tool confirmation",
			tool: newToolConfirmation(t),
			createApproval: func(t *testing.T, toolCall *genai.FunctionCall, pendingResponse *genai.FunctionResponse) *genai.Part {
				return createToolConfirmationApproval(t, toolCall)
			},
			wantFirstArtifactParts: toA2AParts(t, []*genai.Part{
				genai.NewPartFromText(modelTextRootTransfer),
				genai.NewPartFromFunctionCall(transferToolName, map[string]any{"agent_name": remoteAgentName}),
				genai.NewPartFromFunctionResponse(transferToolName, nil),
				genai.NewPartFromText(modelTextRequiresApproval),
				genai.NewPartFromFunctionCall(approvalToolName, nil),
				genai.NewPartFromFunctionResponse(approvalToolName, map[string]any{"status": string(approvalStatusPending)}),
			}, []string{}),
			wantSecondArtifactParts: a2a.ContentParts{
				a2a.DataPart{
					Data: map[string]any{
						"name":     approvalToolName,
						"response": map[string]any{"status": string(approvalStatusVerified)},
					},
					Metadata: map[string]any{"adk_type": "function_response"},
				},
				a2a.TextPart{Text: modelTextTaskComplete},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Server B
			inputRequestingAgent := newInputRequestingAgent(t, "agent-b", tc.tool)
			executorB := newAgentExecutor(inputRequestingAgent, nil)
			serverB := startA2AServer(executorB)
			defer serverB.Close()

			// Server A
			remoteAgent := newA2ARemoteAgent(t, remoteAgentName, serverB)
			rootAgent := newRootAgent("root", remoteAgent)
			executorA := newAgentExecutor(rootAgent, nil)
			serverA := startA2AServer(executorA)
			defer serverA.Close()

			// Client for Server A
			client := newA2AClient(t, serverA)

			// Initial message triggers input required
			msg1 := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Hello, perform important task!"})
			task1 := mustSendMessage(t, client, msg1)
			if task1.Status.State != a2a.TaskStateInputRequired {
				t.Fatalf("client.SendMessage(Initial) result state = %q, want %q", task1.Status.State, a2a.TaskStateInputRequired)
			}

			// Incomplete followup keeps the task in input-required
			msg2 := a2a.NewMessageForTask(a2a.MessageRoleUser, task1, a2a.TextPart{Text: "Is it really necessary?"})
			task2 := mustSendMessage(t, client, msg2)
			if task2.Status.State != a2a.TaskStateInputRequired {
				t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task2.Status.State, a2a.TaskStateInputRequired)
			}

			// Required input gets delivered
			toolCall, pendingResponse := findLongRunningCall(t, toGenaiParts(t, filterPartial(task2.Status.Message.Parts)))
			approvedResponse := tc.createApproval(t, toolCall, pendingResponse)
			msg3 := a2a.NewMessageForTask(a2a.MessageRoleUser, task2,
				a2a.TextPart{Text: "LGTM"},
				toA2AParts(t, []*genai.Part{approvedResponse}, nil)[0],
			)
			task3 := mustSendMessage(t, client, msg3)
			if task3.Status.State != a2a.TaskStateCompleted {
				t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task3.Status.State, a2a.TaskStateCompleted)
			}

			// Verify task on server A
			opts := []cmp.Option{
				cmpopts.EquateEmpty(),
				cmpopts.IgnoreMapEntries(func(k string, v any) bool {
					return strings.HasSuffix(k, "id")
				}),
			}
			gotHistory := task3.History
			wantHistory := []*a2a.Message{msg1, msg2, task1.Status.Message, msg3, task2.Status.Message}
			if diff := cmp.Diff(wantHistory, gotHistory, opts...); diff != "" {
				t.Fatalf("unexpected history (+got,-want) diff:\n%s", diff)
			}

			gotFirstArtifactParts := a2a.ContentParts(filterPartial(task3.Artifacts[0].Parts))
			if diff := cmp.Diff(tc.wantFirstArtifactParts, gotFirstArtifactParts, opts...); diff != "" {
				t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
			}

			gotSecondArtifactParts := a2a.ContentParts(filterPartial(adka2a.WithoutPartialArtifacts(task3.Artifacts)[1].Parts))
			if diff := cmp.Diff(tc.wantSecondArtifactParts, gotSecondArtifactParts, opts...); diff != "" {
				t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
			}
		})
	}
}

func TestA2ASingleHopFinalResponse(t *testing.T) {
	testCases := []struct {
		name              string
		agentFn           func(*testing.T) agent.Agent
		wantArtifactParts a2a.ContentParts
		wantState         a2a.TaskState
		wantStatusContain string
		wantPartial       bool
	}{
		{
			name: "streaming",
			agentFn: func(t *testing.T) agent.Agent {
				beep := newADKEventReplay(t, "beep", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello,", genai.RoleModel), Partial: true}},
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(" I am beep!", genai.RoleModel), Partial: true, TurnComplete: true}},
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello, I am beep!", genai.RoleModel)}},
				})
				boop := newADKEventReplay(t, "boop", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("I am boop", genai.RoleModel), Partial: true}},
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(". We are", genai.RoleModel), Partial: true}},
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("here to help!", genai.RoleModel), Partial: true, TurnComplete: true}},
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("I am boop. We are here to help!", genai.RoleModel)}},
				})
				return utils.Must(sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{beep, boop}},
				}))
			},
			wantState: a2a.TaskStateCompleted,
			wantArtifactParts: a2a.ContentParts{
				a2a.TextPart{Text: "Hello, I am beep!"},
				a2a.TextPart{Text: "I am boop. We are here to help!"},
			},
			wantPartial: true,
		},
		{
			name: "non-streaming",
			agentFn: func(t *testing.T) agent.Agent {
				beep := newADKEventReplay(t, "beep", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello, I am beep!", genai.RoleModel)}},
				})
				boop := newADKEventReplay(t, "boop", []*session.Event{
					{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("I am boop. We are here to help!", genai.RoleModel)}},
				})
				return utils.Must(sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{beep, boop}},
				}))
			},
			wantState: a2a.TaskStateCompleted,
			wantArtifactParts: a2a.ContentParts{
				a2a.TextPart{Text: "Hello, I am beep!"},
				a2a.TextPart{Text: "I am boop. We are here to help!"},
			},
		},
		{
			name: "internal error",
			agentFn: func(t *testing.T) agent.Agent {
				return utils.Must(agent.New(agent.Config{
					Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {}
					},
				}))
			},
			wantStatusContain: "app_name and user_id are required, got app_name: ",
			wantState:         a2a.TaskStateFailed,
			wantArtifactParts: a2a.ContentParts{},
		},
		{
			name: "llm mid-response error response",
			agentFn: func(t *testing.T) agent.Agent {
				event := 0
				llmModel := newGeminiModel(t, "gemini-2.5-flash")
				return utils.Must(llmagent.New(llmagent.Config{
					Name:  "model-agent",
					Model: llmModel,
					AfterModelCallbacks: []llmagent.AfterModelCallback{
						func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
							if event < 2 {
								event++
								return nil, nil
							}
							return &model.LLMResponse{ErrorCode: "500", ErrorMessage: "Model Failed!"}, nil
						},
					},
					Instruction: "You are a helpful assistant.",
				}))
			},
			wantStatusContain: "Model Failed!",
			wantState:         a2a.TaskStateFailed,
			wantArtifactParts: a2a.ContentParts{},
			wantPartial:       true,
		},
		{
			name: "llm mid-response error",
			agentFn: func(t *testing.T) agent.Agent {
				event := 0
				llmModel := newGeminiModel(t, "gemini-2.5-flash")
				return utils.Must(llmagent.New(llmagent.Config{
					Name:  "model-agent",
					Model: llmModel,
					AfterModelCallbacks: []llmagent.AfterModelCallback{
						func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
							if event < 2 {
								event++
								return nil, nil
							}
							return nil, fmt.Errorf("connection error!")
						},
					},
					Instruction: "You are a helpful assistant.",
				}))
			},
			wantStatusContain: "connection error!",
			wantState:         a2a.TaskStateFailed,
			wantArtifactParts: a2a.ContentParts{},
			wantPartial:       true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executor := newAgentExecutor(tc.agentFn(t), nil)
			server := startA2AServer(executor)
			defer server.Close()

			client := newA2AClient(t, server)
			msg := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Tell me about the current weather"})
			task := mustSendMessage(t, client, msg)
			if task.Status.State != tc.wantState {
				t.Fatalf("client.SendMessage(Initial) result state = %q, want %q", task.Status.State, tc.wantState)
			}

			nonPartialArtifacts := adka2a.WithoutPartialArtifacts(task.Artifacts)
			wantResponse := len(tc.wantArtifactParts) > 0
			if wantResponse {
				if len(nonPartialArtifacts) != 1 {
					t.Fatalf("len(artifacts) = %d, want 1", len(nonPartialArtifacts))
				}
				if diff := cmp.Diff(tc.wantArtifactParts, nonPartialArtifacts[0].Parts); diff != "" {
					t.Fatalf("task wrong artifact parts (+got,-want) diff = %s", diff)
				}
			}

			if tc.wantStatusContain != "" {
				if task.Status.Message == nil || len(task.Status.Message.Parts) != 1 {
					t.Fatalf("got status message = %v, want message with one part", task.Status.Message)
				}
				if tp, ok := task.Status.Message.Parts[0].(a2a.TextPart); !ok || !strings.Contains(tp.Text, tc.wantStatusContain) {
					t.Fatalf("got status message = %v, want text containing %q", task.Status.Message.Parts[0], tc.wantStatusContain)
				}
			}

			if !tc.wantPartial {
				if wantResponse && len(task.Artifacts) != 1 {
					t.Fatalf("len(artifacts) = %d, want 1", len(task.Artifacts))
				}
				return
			}

			wantArtifactCount := 1
			if wantResponse {
				wantArtifactCount++
			}
			if wantResponse && len(task.Artifacts) != wantArtifactCount {
				t.Fatalf("len(artifacts) = %d, want %d", len(task.Artifacts), wantArtifactCount)
			}
			var partialArtifact *a2a.Artifact
			if adka2a.IsPartial(task.Artifacts[0].Metadata) {
				partialArtifact = task.Artifacts[0]
			} else {
				partialArtifact = task.Artifacts[1]
			}
			wantPartialParts := a2a.ContentParts{a2a.DataPart{Data: map[string]any{}, Metadata: map[string]any{"adk_partial": true}}}
			if diff := cmp.Diff(wantPartialParts, partialArtifact.Parts); diff != "" {
				t.Fatalf("task wrong artifact parts (+got,-want) diff = %s", diff)
			}
		})
	}
}

func TestA2ARemoteAgentStreamingGeminiSuccess(t *testing.T) {
	// Server B with replayable LLMAgent
	llmModel := newGeminiModel(t, "gemini-2.5-flash")
	modelAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:        "model-agent",
		Model:       llmModel,
		Instruction: "You are a helpful assistant.",
	}))
	executorB := newAgentExecutor(modelAgent, nil)
	serverB := startA2AServer(executorB)
	defer serverB.Close()

	// Server A with RemoteAgent
	remoteAgent := newA2ARemoteAgent(t, "remote-agent", serverB)
	serviceA := session.InMemoryService()
	executorA := newAgentExecutor(remoteAgent, serviceA)
	serverA := startA2AServer(executorA)
	defer serverA.Close()

	ctx := t.Context()
	client := newA2AClient(t, serverA)
	msg := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "tell me about the capital of Poland"})
	msg.ContextID = a2a.NewContextID()

	// Make streaming request and aggregate results
	var taskID a2a.TaskID
	partialText, finalText := "", ""
	for event, err := range client.SendStreamingMessage(t.Context(), &a2a.MessageSendParams{Message: msg}) {
		if err != nil {
			t.Fatalf("client.SendStreamingMessage() error = %v", err)
		}
		if tau, ok := event.(*a2a.TaskArtifactUpdateEvent); ok {
			if adka2a.IsPartial(tau.Metadata) && tau.LastChunk {
				if len(tau.Artifact.Parts) != 1 {
					t.Fatalf("got %d parts in final partial artifact update, want 1", len(tau.Artifact.Parts))
				}
				if dp, ok := tau.Artifact.Parts[0].(a2a.DataPart); !ok || len(dp.Data) > 0 {
					t.Fatalf("got %v part in final partial artifact update, want empty data part", tau.Artifact.Parts[0])
				}
				continue
			}

			if adka2a.IsPartial(tau.Metadata) {
				for _, p := range tau.Artifact.Parts {
					partialText += p.(a2a.TextPart).Text
				}
				continue
			}

			if len(finalText) > 0 {
				t.Fatal("got multiple non-partial updates, want 1")
			}
			finalText = tau.Artifact.Parts[0].(a2a.TextPart).Text
		}
		taskID = event.TaskInfo().TaskID
	}

	// Check streaming contents
	if len(finalText) == 0 {
		t.Fatal("got empty final text")
	}
	if diff := cmp.Diff(partialText, finalText); diff != "" {
		t.Fatalf("got final event text different from streaming (+got, -want), diff = %s", diff)
	}

	// Check A2A Task state
	task, err := client.GetTask(ctx, &a2a.TaskQueryParams{ID: taskID})
	if err != nil {
		t.Fatalf("client.GetTask() error = %v", err)
	}
	if task.Status.State != a2a.TaskStateCompleted {
		t.Fatalf("task state = %q, want %q", task.Status.State, a2a.TaskStateCompleted)
	}

	// Check Session Store state
	fullSessionResp, err := serviceA.Get(ctx, &session.GetRequest{
		AppName:   remoteAgent.Name(),
		UserID:    "A2A_USER_" + msg.ContextID,
		SessionID: msg.ContextID,
	})
	if err != nil {
		t.Fatalf("serviceA.GetSession() error = %v", err)
	}
	events := fullSessionResp.Session.Events()
	if events.Len() != 3 {
		t.Fatalf("got event count = %d, want [user-msg, response, turn-complete]", events.Len())
	}
	if events.At(0).Author != "user" {
		t.Fatalf("got first event author = %s, want user", events.At(0).Author)
	}
	if !events.At(2).TurnComplete || events.At(2).Content != nil {
		t.Fatalf("got last event turn complete = true with no content, got turn complete = %v, content = %v", events.At(2).TurnComplete, events.At(2).Content)
	}
	if len(events.At(1).Content.Parts) != 1 {
		t.Fatalf("got content event with %d parts, want 1", len(events.At(1).Content.Parts))
	}
	if diff := cmp.Diff(finalText, events.At(1).Content.Parts[0].Text); diff != "" {
		t.Fatalf("got content event text different from A2A response (+got, -want), diff = %s", diff)
	}
}

func TestA2ARemoteAgentStreamingGeminiError(t *testing.T) {
	// Server B with replayable LLMAgent which fails after emitting some events
	eventCount := 0
	const errorMessage = "connection error!"
	llmModel := newGeminiModel(t, "gemini-2.5-flash")
	modelAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:        "model-agent",
		Model:       llmModel,
		Instruction: "You are a helpful assistant.",
		AfterModelCallbacks: []llmagent.AfterModelCallback{
			func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error) {
				if eventCount < 3 {
					eventCount++
					return nil, nil
				}
				return nil, errors.New(errorMessage)
			},
		},
	}))
	executorB := newAgentExecutor(modelAgent, nil)
	serverB := startA2AServer(executorB)
	defer serverB.Close()

	// Server A with RemoteAgent
	remoteAgent := newA2ARemoteAgent(t, "remote-agent", serverB)
	serviceA := session.InMemoryService()
	executorA := newAgentExecutor(remoteAgent, serviceA)
	serverA := startA2AServer(executorA)
	defer serverA.Close()

	ctx := t.Context()
	client := newA2AClient(t, serverA)
	msg := a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "tell me about the capital of Poland"})
	msg.ContextID = a2a.NewContextID()

	// Make streaming request and aggregate results
	var taskID a2a.TaskID
	for event, err := range client.SendStreamingMessage(t.Context(), &a2a.MessageSendParams{Message: msg}) {
		if err != nil {
			t.Fatalf("client.SendStreamingMessage() error = %v", err)
		}
		taskID = event.TaskInfo().TaskID
	}

	// Check A2A Task state
	task, err := client.GetTask(ctx, &a2a.TaskQueryParams{ID: taskID})
	if err != nil {
		t.Fatalf("client.GetTask() error = %v", err)
	}
	if task.Status.State != a2a.TaskStateFailed {
		t.Fatalf("task state = %q, want %q", task.Status.State, a2a.TaskStateFailed)
	}
	if task.Status.Message == nil || len(task.Status.Message.Parts) != 1 {
		t.Fatalf("task status message = %v, want 1 part", task.Status.Message)
	}
	if tp, ok := task.Status.Message.Parts[0].(a2a.TextPart); !ok || !strings.Contains(tp.Text, errorMessage) {
		t.Fatalf("task status message = %v, want containing %q", task.Status.Message.Parts[0], errorMessage)
	}
	if len(task.Artifacts) != 1 || len(adka2a.WithoutPartialArtifacts(task.Artifacts)) != 0 {
		t.Fatalf("task artifacts = %v, want single partial artifact", task.Artifacts)
	}
	if dp, ok := task.Artifacts[0].Parts[0].(a2a.DataPart); !ok || len(dp.Data) != 0 {
		t.Fatalf("task artifact = %v, want reset partial artifact", task.Artifacts[0])
	}

	// Check Session Store state
	fullSessionResp, err := serviceA.Get(ctx, &session.GetRequest{
		AppName:   remoteAgent.Name(),
		UserID:    "A2A_USER_" + msg.ContextID,
		SessionID: msg.ContextID,
	})
	if err != nil {
		t.Fatalf("serviceA.GetSession() error = %v", err)
	}
	events := fullSessionResp.Session.Events()
	if events.Len() != 2 {
		t.Fatalf("got event count = %d, want 2", events.Len())
	}
	if !strings.Contains(events.At(1).ErrorMessage, errorMessage) {
		t.Fatalf("got event error message = %q, want containing %q", events.At(1).ErrorMessage, errorMessage)
	}
}

type llmStub struct {
	name            string
	generateContent func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error]
}

func (d *llmStub) Name() string {
	return d.name
}

func (d *llmStub) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return d.generateContent(ctx, req, stream)
}

func newLongRunningTool(t *testing.T) tool.Tool {
	t.Helper()
	requestApproval, err := functiontool.New(functiontool.Config{
		Name:          approvalToolName,
		Description:   "Request approval before proceeding.",
		IsLongRunning: true,
	}, func(ctx tool.Context, x map[string]any) (approval, error) {
		return approval{Status: approvalStatusPending, TicketID: a2a.NewContextID()}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return requestApproval
}

func newToolConfirmation(t *testing.T) tool.Tool {
	t.Helper()

	requestApproval, err := functiontool.New(functiontool.Config{
		Name:        approvalToolName,
		Description: "Request approval before proceeding.",
	}, func(ctx tool.Context, x map[string]any) (approval, error) {
		confirmation := ctx.ToolConfirmation()
		if confirmation == nil {
			ticketID := a2a.NewContextID()
			if err := ctx.RequestConfirmation("I need approval", map[string]string{"ticket_id": ticketID}); err != nil {
				return approval{}, err
			}
			return approval{Status: approvalStatusPending, TicketID: ticketID}, nil
		}
		if !confirmation.Confirmed {
			return approval{}, fmt.Errorf("confirmation was rejected")
		}
		jsonBytes, err := json.Marshal(confirmation.Payload)
		if err != nil {
			return approval{}, fmt.Errorf("error marshalling payload %s: %w", confirmation.Payload, err)
		}
		var payload approval
		if err := json.Unmarshal(jsonBytes, &payload); err != nil {
			return approval{}, fmt.Errorf("error unmarshalling payload %s: %w", confirmation.Payload, err)
		}
		return approval{Status: approvalStatusVerified, TicketID: payload.TicketID}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return requestApproval
}

func newInputRequestingAgent(t *testing.T, name string, requestApproval tool.Tool) agent.Agent {
	t.Helper()
	return utils.Must(llmagent.New(llmagent.Config{
		Name:  name,
		Tools: []tool.Tool{requestApproval},
		Model: &llmStub{
			generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					lastMessage := req.Contents[len(req.Contents)-1]
					approvalResult := utils.FunctionResponses(lastMessage)
					var content *genai.Content
					switch {
					case len(approvalResult) == 0: // the first model invocation - invoke a long running tool
						content = genai.NewContentFromParts([]*genai.Part{
							genai.NewPartFromText(modelTextRequiresApproval),
							genai.NewPartFromFunctionCall(approvalToolName, map[string]any{}),
						}, genai.RoleModel)
					case len(approvalResult) == 1 && approvalResult[0].Response["status"] == string(approvalStatusPending): // the tool returned a pending result
						content = genai.NewContentFromText(modelTextWaitingForApproval, genai.RoleModel)
					default: // user approval is in the session
						content = genai.NewContentFromText(modelTextTaskComplete, genai.RoleModel)
					}
					yield(&model.LLMResponse{Content: content}, nil)
				}
			},
		},
	}))
}

func newRootAgent(name string, subAgent agent.Agent) agent.Agent {
	return utils.Must(llmagent.New(llmagent.Config{
		Name:      name,
		SubAgents: []agent.Agent{subAgent},
		Model: &llmStub{
			name: name + "-model",
			generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					yield(&model.LLMResponse{
						Content: genai.NewContentFromParts([]*genai.Part{
							genai.NewPartFromText(modelTextRootTransfer),
							genai.NewPartFromFunctionCall(transferToolName, map[string]any{"agent_name": subAgent.Name()}),
						}, genai.RoleModel),
					}, nil)
				}
			},
		},
	}))
}

func newAgentExecutor(agnt agent.Agent, service session.Service) a2asrv.AgentExecutor {
	if service == nil {
		service = session.InMemoryService()
	}
	return adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        agnt.Name(),
			SessionService: service,
			Agent:          agnt,
		},
		RunConfig: agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
	})
}

func mustSendMessage(t *testing.T, client *a2aclient.Client, msg *a2a.Message) *a2a.Task {
	t.Helper()
	sendParams := &a2a.MessageSendParams{Message: msg}
	result, err := client.SendMessage(t.Context(), sendParams)
	if err != nil {
		t.Fatalf("client.SendMessage() error = %v", err)
	}
	task, ok := result.(*a2a.Task)
	if !ok {
		t.Fatalf("client.SendMessage() result is %T, want *a2a.Task", result)
	}
	return task
}

func filterPartial(parts []a2a.Part) []a2a.Part {
	var result []a2a.Part
	for _, p := range parts {
		if b, _ := p.Meta()[adka2a.ToA2AMetaKey("partial")].(bool); b {
			continue
		}
		result = append(result, p)
	}
	return result
}

func findLongRunningCall(t *testing.T, parts []*genai.Part) (*genai.FunctionCall, *genai.FunctionResponse) {
	t.Helper()
	content := genai.NewContentFromParts(parts, genai.RoleModel)
	calls := utils.FunctionCalls(content)
	responses := utils.FunctionResponses(content)
	if len(calls) > 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if len(responses) > 1 {
		t.Fatalf("got %d responses, want 1", len(responses))
	}
	var call *genai.FunctionCall
	if len(calls) == 1 {
		call = calls[0]
	}
	var response *genai.FunctionResponse
	if len(responses) == 1 {
		response = responses[0]
	}
	return call, response
}

func toA2AParts(t *testing.T, parts []*genai.Part, callIDs []string) []a2a.Part {
	t.Helper()
	a2aParts, err := adka2a.ToA2AParts(parts, callIDs)
	if err != nil {
		t.Fatalf("adka2a.ToA2AParts() error = %v", err)
	}
	return a2aParts
}

func toGenaiParts(t *testing.T, a2aParts []a2a.Part) []*genai.Part {
	t.Helper()
	parts, err := adka2a.ToGenAIParts(a2aParts)
	if err != nil {
		t.Fatalf("adka2a.ToGenAIParts() error = %v", err)
	}
	return parts
}

func toMap(t *testing.T, v any) map[string]any {
	t.Helper()
	result, err := converters.ToMapStructure(v)
	if err != nil {
		t.Fatalf("converters.ToMapStructure error = %v", err)
	}
	return result
}

func fromMap[T any](t *testing.T, m map[string]any) *T {
	t.Helper()
	result, err := converters.FromMapStructure[T](m)
	if err != nil {
		t.Fatalf("converters.FromMapStructure() error = %v", err)
	}
	return result
}

func newA2AClient(t *testing.T, server *testA2AServer) *a2aclient.Client {
	t.Helper()

	result, err := a2aclient.NewFromCard(t.Context(), &a2a.AgentCard{
		PreferredTransport: a2a.TransportProtocolJSONRPC,
		URL:                server.URL, Capabilities: a2a.AgentCapabilities{Streaming: true},
	})
	if err != nil {
		t.Fatalf("a2aclient.NewFromEndpoints() error = %v", err)
	}
	return result
}

func createLongRunningToolApproval(t *testing.T, pendingResponse *genai.FunctionResponse) *genai.Part {
	t.Helper()
	pendingApproval := fromMap[approval](t, pendingResponse.Response)
	response := genai.NewPartFromFunctionResponse(approvalToolName, toMap(t, approval{
		Status:   approvalStatusApproved,
		TicketID: pendingApproval.TicketID,
	}))
	response.FunctionResponse.ID = pendingResponse.ID
	return response
}

func createToolConfirmationApproval(t *testing.T, toolCall *genai.FunctionCall) *genai.Part {
	t.Helper()
	tcMap, ok := toolCall.Args["toolConfirmation"].(map[string]any)
	if !ok {
		t.Fatalf("toolCall = %v, want toolConfirmation", toolCall)
	}
	payloadMap, ok := tcMap["payload"].(map[string]any)
	if !ok {
		t.Fatalf("toolCall = %v, want payload", toolCall)
	}
	ticketID, ok := payloadMap["ticket_id"].(string)
	if !ok {
		t.Fatalf("toolCall = %v, want ticket_id", toolCall)
	}
	return &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:   toolCall.ID,
			Name: toolCall.Name,
			Response: map[string]any{
				"confirmed": true,
				"payload":   map[string]string{"ticket_id": ticketID},
			},
		},
	}
}

func newGeminiTestClientConfig(t *testing.T, rrfile string) (http.RoundTripper, bool) {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}

	// Ensure the transport is closed to flush data and release locks
	if c, ok := rr.(io.Closer); ok {
		t.Cleanup(func() {
			if err := c.Close(); err != nil {
				t.Errorf("failed to close transport: %v", err)
			}
		})
	}

	recording, _ := httprr.Recording(rrfile)
	return rr, recording
}

func newGeminiModel(t *testing.T, modelName string) model.LLM {
	apiKey := "fakeKey"
	trace := filepath.Join("testdata", strings.ReplaceAll(t.Name()+".httprr", "/", "_"))
	recording := false
	transport, recording := newGeminiTestClientConfig(t, trace)
	if recording { // if we are recording httprr trace, don't use the fakeKey.
		apiKey = ""
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
