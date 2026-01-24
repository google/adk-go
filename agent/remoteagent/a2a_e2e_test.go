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

import (
	"context"
	"iter"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2aclient"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const (
	approvalToolName            = "request_approval"
	modelTextRequiresApproval   = "need to request approval first!"
	modelTextWaitingForApproval = "waiting for user's approval..."
	modelTextTaskComplete       = "Task complete!"

	transferToolName      = "transfer_to_agent"
	modelTextRootTransfer = "transfering... please hold... beepboop..."

	ticketStatusApproved = "approved"
)

type approvalStatus string

var (
	approvalStatusPending  approvalStatus = "pending"
	approvalStatusApproved approvalStatus = "approved"
)

type approval struct {
	Status   approvalStatus `json:"status"`
	TicketID string         `json:"ticket_id"`
}

/**
 * a2aclient -> a2aserver -> adka2a.Executor -> llmagent with a long running tool
 */
func TestA2AInputRequired(t *testing.T) {
	// Server
	inputRequestingAgent := newInputRequestingAgent(t, "agent-b")
	executor := newAgentExecutor(inputRequestingAgent)
	server := startA2AServer(executor)
	defer server.Close()

	// Client
	ctx := t.Context()
	client, err := a2aclient.NewFromEndpoints(ctx, []a2a.AgentInterface{
		{URL: server.URL, Transport: a2a.TransportProtocolJSONRPC},
	})
	if err != nil {
		t.Fatalf("a2aclient.NewFromCard() error = %v", err)
	}

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
	toolCall, pendingResponse := findLongRunningCall(t, toGenaiParts(t, task2.Status.Message.Parts))
	approvedResponse := pendingToApproved(t, pendingResponse)
	msg3 := a2a.NewMessageForTask(a2a.MessageRoleUser, task2,
		a2a.TextPart{Text: "LGTM"},
		toA2AParts(t, []*genai.Part{approvedResponse}, []string{toolCall.ID})[0],
	)
	task3 := mustSendMessage(t, client, msg3)
	if task3.Status.State != a2a.TaskStateCompleted {
		t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task3.Status.State, a2a.TaskStateCompleted)
	}

	// Verify the final task state
	opts := []cmp.Option{cmpopts.EquateEmpty()}
	if len(task3.Artifacts) != 2 {
		t.Fatalf("len(task.Artifacts) = %d, want 2", len(task3.Artifacts))
	}

	gotHistory := task3.History
	wantHistory := []*a2a.Message{msg1, msg2, task1.Status.Message, msg3, task2.Status.Message}
	if diff := cmp.Diff(wantHistory, gotHistory, opts...); diff != "" {
		t.Fatalf("unexpected history (+got,-want) diff:\n%s", diff)
	}

	gotFirstArtifactParts := task3.Artifacts[0].Parts
	wantFirstAftifactParts := a2a.ContentParts{
		a2a.TextPart{Text: modelTextRequiresApproval},
		toA2AParts(t, []*genai.Part{{FunctionCall: toolCall}}, []string{toolCall.ID})[0],
		toA2AParts(t, []*genai.Part{{FunctionResponse: pendingResponse}}, nil)[0],
		a2a.TextPart{Text: modelTextWaitingForApproval},
	}
	if diff := cmp.Diff(wantFirstAftifactParts, gotFirstArtifactParts, opts...); diff != "" {
		t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
	}

	gotSecondArtifactParts := task3.Artifacts[1].Parts
	wantSecondArtifactParts := a2a.ContentParts{a2a.TextPart{Text: modelTextTaskComplete}}
	if diff := cmp.Diff(wantSecondArtifactParts, gotSecondArtifactParts, opts...); diff != "" {
		t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
	}
}

/**
 * a2aclient -> server A -> adka2a.Executor A ->-> llmagent with remote subagent ->
 * 		remotesubagent -> server B -> adka2a.Executor B -> llmagent with a long running tool
 */
func TestA2AMultiHopInputRequired(t *testing.T) {
	// Server B
	inputRequestingAgent := newInputRequestingAgent(t, "agent-b")
	executorB := newAgentExecutor(inputRequestingAgent)
	serverB := startA2AServer(executorB)
	defer serverB.Close()

	// Server A
	remoteAgent := newA2ARemoteAgent(t, "remote-"+inputRequestingAgent.Name(), serverB)
	rootAgent := newRootAgent("root", remoteAgent)
	executorA := newAgentExecutor(rootAgent)
	serverA := startA2AServer(executorA)
	defer serverA.Close()

	// Client for Server A
	ctx := t.Context()
	client, err := a2aclient.NewFromEndpoints(ctx, []a2a.AgentInterface{
		{URL: serverA.URL, Transport: a2a.TransportProtocolJSONRPC},
	})
	if err != nil {
		t.Fatalf("a2aclient.NewFromCard() error = %v", err)
	}

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
	approvedResponse := pendingToApproved(t, pendingResponse)
	msg3 := a2a.NewMessageForTask(a2a.MessageRoleUser, task2,
		a2a.TextPart{Text: "LGTM"},
		toA2AParts(t, []*genai.Part{approvedResponse}, nil)[0],
	)
	task3 := mustSendMessage(t, client, msg3)
	if task3.Status.State != a2a.TaskStateCompleted {
		t.Fatalf("client.SendMessage(IncompleteInput) result state = %q, want %q", task3.Status.State, a2a.TaskStateCompleted)
	}

	// Verify task on server A
	opts := []cmp.Option{cmpopts.EquateEmpty(), cmpopts.IgnoreMapEntries(func(k string, v any) bool {
		return k == "id"
	})}
	gotHistory := task3.History
	wantHistory := []*a2a.Message{msg1, msg2, task1.Status.Message, msg3, task2.Status.Message}
	if diff := cmp.Diff(wantHistory, gotHistory, opts...); diff != "" {
		t.Fatalf("unexpected history (+got,-want) diff:\n%s", diff)
	}

	gotFirstArtifactParts := filterPartial(task3.Artifacts[0].Parts)
	wantFirstAftifactParts := toA2AParts(t, []*genai.Part{
		genai.NewPartFromText(modelTextRootTransfer),
		genai.NewPartFromFunctionCall(transferToolName, map[string]any{"agent_name": remoteAgent.Name()}),
		genai.NewPartFromFunctionResponse(transferToolName, nil),
		genai.NewPartFromText(modelTextRequiresApproval),
		genai.NewPartFromText(modelTextWaitingForApproval),
		{FunctionCall: &genai.FunctionCall{Name: toolCall.Name, ID: toolCall.ID}},
		{FunctionResponse: pendingResponse},
	}, []string{toolCall.ID})
	if diff := cmp.Diff(wantFirstAftifactParts, gotFirstArtifactParts, opts...); diff != "" {
		t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
	}

	gotSecondArtifactParts := filterPartial(task3.Artifacts[1].Parts)
	wantSecondArtifactParts := []a2a.Part{a2a.TextPart{Text: modelTextTaskComplete}}
	if diff := cmp.Diff(wantSecondArtifactParts, gotSecondArtifactParts, opts...); diff != "" {
		t.Fatalf("unexpected artifact parts (+got,-want) diff:\n%s", diff)
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

func newInputRequestingAgent(t *testing.T, name string) agent.Agent {
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
	return utils.Must(llmagent.New(llmagent.Config{
		Name:  name,
		Tools: []tool.Tool{requestApproval},
		Model: &llmStub{
			name: name + "-model",
			generateContent: func(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					lastMessage := req.Contents[len(req.Contents)-1]
					_, approvalResult := findLongRunningCall(t, lastMessage.Parts)
					var content *genai.Content
					switch {
					case approvalResult == nil: // the first model invocation - invoke a long running tool
						content = genai.NewContentFromParts([]*genai.Part{
							genai.NewPartFromText(modelTextRequiresApproval),
							genai.NewPartFromFunctionCall(approvalToolName, map[string]any{}),
						}, genai.RoleModel)
					case approvalResult.Response["status"] != ticketStatusApproved: // the tool returned a pending result
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

func newAgentExecutor(agent agent.Agent) a2asrv.AgentExecutor {
	return adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        agent.Name(),
			SessionService: session.InMemoryService(),
			Agent:          agent,
		},
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

func pendingToApproved(t *testing.T, pendingResponse *genai.FunctionResponse) *genai.Part {
	t.Helper()
	pendingApproval := fromMap[approval](t, pendingResponse.Response)
	response := genai.NewPartFromFunctionResponse(approvalToolName, toMap(t, approval{
		Status:   approvalStatusApproved,
		TicketID: pendingApproval.TicketID,
	}))
	response.FunctionResponse.ID = pendingResponse.ID
	return response
}
