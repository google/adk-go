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

package replayplugin

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/plugin/replayplugin/recording"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// New creates an instance of the logging plugin.
//
// This plugin helps users track the invocation status by logging:
// - User messages and invocation context
// - Agent execution flow
// - LLM requests and responses
// - Tool calls with arguments and results
// - Events and final responses
// - Errors during model and tool execution
func New() (*plugin.Plugin, error) {
	p := &replayPlugin{
		invocationStates: make(map[string]*invocationReplayState),
	}
	return plugin.New(plugin.Config{
		Name:                "replay_plugin",
		BeforeRunCallback:   p.beforeRun,
		AfterRunCallback:    p.afterRun,
		BeforeModelCallback: p.beforeModel,
		BeforeToolCallback:  p.beforeTool,
	})
}

// MustNew is like New but panics if there is an error.
func MustNew() *plugin.Plugin {
	p, err := New()
	if err != nil {
		panic(err)
	}
	return p
}

type replayPlugin struct {
	mu               sync.Mutex // Mutex to protect the map
	invocationStates map[string]*invocationReplayState
}

func (p *replayPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	if ctx.Session() == nil {
		return nil, nil
	}
	if !p.isReplayModeOn(ctx.Session().State()) {
		return nil, nil
	}

	_, err := p.loadInvocationState(ctx)
	if err != nil {
		return nil, err
	}
	return nil, nil
}

func (p *replayPlugin) beforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	if !p.isReplayModeOn(ctx.State()) {
		return nil, nil
	}

	invocationState, err := p.getInvocationState(ctx)
	if err != nil {
		return nil, err
	}

	agentName := ctx.AgentName()
	recording, err := p.verifyAndGetNextLLMRecordingForAgent(invocationState, agentName, req)
	if err != nil {
		return nil, err
	}

	return recording.LlmResponse.ToLLMResponse(), nil
}

func (p *replayPlugin) beforeTool(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	if !p.isReplayModeOn(ctx.State()) {
		return nil, nil
	}

	invocationState, err := p.getInvocationState(ctx)
	if err != nil {
		return nil, err
	}

	agentName := ctx.AgentName()
	recording, err := p.verifyAndGetNextToolRecordingForAgent(invocationState, agentName, t, args)
	if err != nil {
		return nil, err
	}
	typeName := fmt.Sprintf("%T", t)
	if !strings.HasSuffix(typeName, "agentTool") {
		// TODO: support replay requests and responses from AgentTool.
		if ft, ok := t.(toolinternal.FunctionTool); ok {
			_, err := ft.Run(ctx, args)
			if err != nil {
				return nil, err
			}
		}
	}

	return recording.ToolResponse.Response, nil
}

func (p *replayPlugin) afterRun(ctx agent.InvocationContext) {
	if ctx.Session() == nil {
		return
	}
	sessionState := ctx.Session().State()
	if !p.isReplayModeOn(sessionState) {
		return
	}
	delete(p.invocationStates, ctx.InvocationID())
}

func (p *replayPlugin) isReplayModeOn(sessionState session.State) bool {
	if sessionState == nil {
		return false
	}
	configVal, err := sessionState.Get("_adk_replay_config")
	// If the key doesn't exist or there's an error, we treat it as disabled.
	if err != nil {
		return false
	}

	config, ok := configVal.(map[string]any)
	if !ok {
		return false
	}

	caseDirVal, ok := config["dir"]
	if !ok {
		return false
	}
	caseDir, ok := caseDirVal.(string)
	if !ok || caseDir == "" {
		return false
	}

	msgIndexVal, ok := config["user_message_index"]
	if !ok || msgIndexVal == nil {
		return false
	}

	return true
}

func (p *replayPlugin) getInvocationState(ctx agent.CallbackContext) (*invocationReplayState, error) {
	invocationID := ctx.InvocationID()
	state, ok := p.invocationStates[invocationID]
	if !ok {
		return nil, fmt.Errorf("replay state not initialized. ensure before_run created it")
	}
	return state, nil
}

func (p *replayPlugin) loadInvocationState(ctx agent.InvocationContext) (*invocationReplayState, error) {
	invocationID := ctx.InvocationID()

	// 1. Extract Configuration
	// We assume ctx.State is map[string]any
	configVal, err := ctx.Session().State().Get("_adk_replay_config")
	if err != nil {
		return nil, fmt.Errorf("replay config error: %w", err)
	}

	config, ok := configVal.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("replay config error: '_adk_replay_config' is not a valid map")
	}

	// 2. Validate Parameters
	// Safely extract 'dir'
	caseDir, ok := config["dir"].(string)
	if !ok || caseDir == "" {
		return nil, fmt.Errorf("replay config error: 'dir' parameter is missing or empty")
	}

	// Safely extract 'user_message_index'
	// Note: JSON/YAML unmarshaling into 'any' often results in float64,
	// so we check for both int and float64 to be robust.
	var msgIndex int
	switch v := config["user_message_index"].(type) {
	case int:
		msgIndex = v
	case float64:
		msgIndex = int(v)
	default:
		return nil, fmt.Errorf("replay config error: 'user_message_index' is missing or not a number")
	}

	// 3. Load Recordings File
	recordingsPath := filepath.Join(caseDir, "generated-recordings.yaml")

	// Check if file exists
	if _, err := os.Stat(recordingsPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("replay config error: recordings file not found: %s", recordingsPath)
	}

	// Read file
	data, err := os.ReadFile(recordingsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read recordings file: %w", err)
	}

	// Parse YAML
	var recordings recording.Recordings
	if err := yaml.Unmarshal(data, &recordings); err != nil {
		return nil, fmt.Errorf("failed to parse recordings from %s: %w", recordingsPath, err)
	}

	// Add index to each recording, based on user message index. Used for parallel execution sync.
	index := 0
	prevMessageId := 0
	for i := range recordings.Recordings {
		if prevMessageId != recordings.Recordings[i].UserMessageIndex {
			prevMessageId = recordings.Recordings[i].UserMessageIndex
			index = 0
		}
		recordings.Recordings[i].Index = index
		index++
	}

	// 4. Create and Store State
	state := newInvocationReplayState(caseDir, msgIndex, &recordings)

	p.mu.Lock()
	p.invocationStates[invocationID] = state
	p.mu.Unlock()

	return state, nil
}

func getNextRecordingForAgent(state *invocationReplayState, agentName string) (*recording.Recording, error) {
	// Get current agent index
	currentAgentIndex, ok := state.GetAgentReplayIndex(agentName)
	if !ok {
		currentAgentIndex = 0
	}

	// Filter ALL recordings for this agent and user message index (strict order)
	agentRecordings := make([]*recording.Recording, 0)
	for _, recording := range state.recordings.Recordings {
		if recording.AgentName == agentName && recording.UserMessageIndex == state.userMessageIndex {
			agentRecordings = append(agentRecordings, &recording)
		}
	}

	// Check if we have enough recordings for this agent
	if currentAgentIndex >= len(agentRecordings) {
		return nil, fmt.Errorf("runtime sent more requests than expected for agent '%s' at user_message_index %d. Expected %d, but got request at index %d",
			agentName, state.userMessageIndex, len(agentRecordings), currentAgentIndex)
	}

	// Get the expected recording
	expectedRecording := agentRecordings[currentAgentIndex]

	// Wait for the current index to match the expected index
	// This ensures that we process recordings in the expected order, even if agents are executing in parallel
	state.mu.Lock()
	for state.curIndex != expectedRecording.Index {
		state.cond.Wait()
		time.Sleep(1 * time.Second)
	}

	state.agentReplayIndices[agentName]++
	state.curIndex++

	state.mu.Unlock()
	state.cond.Broadcast()

	return expectedRecording, nil
}

func (p *replayPlugin) verifyAndGetNextLLMRecordingForAgent(state *invocationReplayState, agentName string, llmRequest *model.LLMRequest) (*recording.LLMRecording, error) {
	currentAgentIndex, ok := state.GetAgentReplayIndex(agentName)
	if !ok {
		currentAgentIndex = 0
	}
	expectedRecording, err := getNextRecordingForAgent(state, agentName)
	if err != nil {
		return nil, err
	}

	if expectedRecording.LLMRecording == nil {
		return nil, fmt.Errorf("expected LLM recording for agent '%s' at index %d, but found tool recording", agentName, currentAgentIndex)
	}

	// Strict verification of LLM request
	err = verifyLLMRequestMatch(expectedRecording.LLMRecording.LlmRequest.ToLLMRequest(), llmRequest, agentName, currentAgentIndex)
	if err != nil {
		return nil, err
	}

	return expectedRecording.LLMRecording, nil
}

func verifyLLMRequestMatch(expectedLLMRequest, actualLLMRequest *model.LLMRequest, agentName string, agentIndex int) error {
	// Define options to ignore specific fields.
	opts := []cmp.Option{
		cmpopts.IgnoreFields(genai.FunctionDeclaration{}, "ParametersJsonSchema", "ResponseJsonSchema", "Parameters", "Response"),
		cmpopts.IgnoreFields(model.LLMRequest{}, "Tools"),
		cmpopts.IgnoreFields(genai.GenerateContentConfig{}, "Labels"),
		cmpopts.EquateEmpty(),
	}

	// Compare!
	// cmp.Diff returns an empty string if they are equal, otherwise a human-readable diff.
	if diff := cmp.Diff(expectedLLMRequest, actualLLMRequest, opts...); diff != "" {
		return fmt.Errorf("llm request mismatch for agent '%s' (index %d):\n%s",
			agentName, agentIndex, diff)
	}

	return nil
}

func (p *replayPlugin) verifyAndGetNextToolRecordingForAgent(state *invocationReplayState, agentName string, t tool.Tool, args map[string]any) (*recording.ToolRecording, error) {
	currentAgentIndex, ok := state.GetAgentReplayIndex(agentName)
	if !ok {
		currentAgentIndex = 0
	}
	expectedRecording, err := getNextRecordingForAgent(state, agentName)
	if err != nil {
		return nil, err
	}

	if expectedRecording.ToolRecording == nil {
		return nil, fmt.Errorf("expected tool recording for agent '%s' at index %d, but found LLM recording", agentName, currentAgentIndex)
	}

	// Strict verification of tool call
	err = verifyToolCallMatch(expectedRecording.ToolRecording.ToolCall.ToGenAI(), t.Name(), args, agentName, currentAgentIndex)
	if err != nil {
		return nil, err
	}

	return expectedRecording.ToolRecording, nil
}

func verifyToolCallMatch(expectedToolCall *genai.FunctionCall, toolName string, toolArgs map[string]any, agentName string, agentIndex int) error {
	if expectedToolCall.Name != toolName {
		return fmt.Errorf("tool name mismatch for agent '%s' (index %d): expected '%s', got '%s'",
			agentName, agentIndex, expectedToolCall.Name, toolName)
	}

	if diff := cmp.Diff(expectedToolCall.Args, toolArgs); diff != "" {
		return fmt.Errorf("tool args mismatch for agent '%s' (index %d):\n%s",
			agentName, agentIndex, diff)
	}

	return nil
}
