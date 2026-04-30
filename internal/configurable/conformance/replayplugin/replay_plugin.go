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

// Package replayplugin provides an ADK plugin for replaying recorded LLM and tool interactions.
//
// This plugin is primarily used for conformance testing, allowing deterministic execution
// of agents by mocking LLM responses and tool outputs based on a pre-recorded session.
//
// The plugin operates by intercepting:
//   - BeforeRun: To initialize replay state from configuration.
//   - BeforeModel: To match LLM requests against recordings and return mock responses.
//   - BeforeTool: To match tool calls against recordings and return mock outputs.
//   - AfterRun: To clean up invocation state.
//
// Replay configuration is expected in the session state under the key "_adk_replay_config",
// containing:
//   - "dir": Path to the directory containing "generated-recordings.yaml".
//   - "user_message_index": The index of the user message to replay.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable/conformance/replayplugin/recording"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// New creates an instance of the replay plugin.
//
// allowedBaseDir specifies the root directory from which recordings can be loaded.
// Attempts to load recordings from outside this directory will result in an error.
func New(allowedBaseDir string) (*plugin.Plugin, error) {
	p := &replayPlugin{
		invocationStates: make(map[string]*invocationReplayState),
		allowedBaseDir:   allowedBaseDir,
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
func MustNew(allowedBaseDir string) *plugin.Plugin {
	p, err := New(allowedBaseDir)
	if err != nil {
		panic(err)
	}
	return p
}

type replayPlugin struct {
	mu               sync.Mutex // Mutex to protect the map
	invocationStates map[string]*invocationReplayState
	allowedBaseDir   string
}

// beforeRun initializes the replay state for the current invocation if replay mode is enabled.
func (p *replayPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	if ctx.Session() == nil {
		return nil, nil
	}

	on, err := p.isReplayModeOn(ctx.Session().State())
	if err != nil {
		return nil, err
	}
	if !on {
		return nil, nil
	}

	_, err = p.loadInvocationState(ctx)
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// beforeModel intercepts LLM requests, verifies them against the recording, and returns the recorded response.
func (p *replayPlugin) beforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	on, err := p.isReplayModeOn(ctx.State())
	if err != nil {
		return nil, err
	}
	if !on {
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

	return recording.LLMResponses[0], nil
}

// beforeTool intercepts tool calls, verifies them against the recording, and returns the recorded response.
func (p *replayPlugin) beforeTool(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	on, err := p.isReplayModeOn(ctx.State())
	if err != nil {
		return nil, err
	}
	if !on {
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
				fmt.Println("Error calling tool:", err)
			}
		}
	}

	return recording.ToolResponse.Response, nil
}

// afterRun cleans up the invocation state.
func (p *replayPlugin) afterRun(ctx agent.InvocationContext) {
	if ctx.Session() == nil {
		return
	}
	sessionState := ctx.Session().State()
	on, err := p.isReplayModeOn(sessionState)
	if err != nil || !on {
		return
	}
	p.mu.Lock()
	delete(p.invocationStates, ctx.InvocationID())
	p.mu.Unlock()
}

// isReplayModeOn checks if replay mode is enabled in the session state and validates the configuration.
func (p *replayPlugin) isReplayModeOn(sessionState session.State) (bool, error) {
	if sessionState == nil {
		return false, nil
	}
	configVal, err := sessionState.Get("_adk_replay_config")
	// If the key doesn't exist or there's an error, we treat it as disabled.
	if err != nil {
		return false, nil
	}

	config, ok := configVal.(map[string]any)
	if !ok {
		return false, nil
	}

	caseDirVal, ok := config["dir"]
	if !ok {
		return false, nil
	}
	caseDir, ok := caseDirVal.(string)
	if !ok || caseDir == "" {
		return false, nil
	}

	basePath, err := filepath.Abs(p.allowedBaseDir)
	if err != nil {
		return false, fmt.Errorf("invalid path format: %v", err)
	}
	requestedAbsPath, err := filepath.Abs(caseDir)
	if err != nil {
		return false, fmt.Errorf("invalid path format: %v", err)
	}
	rel, err := filepath.Rel(basePath, requestedAbsPath)
	if err != nil {
		return false, fmt.Errorf("invalid path format: %v", err)
	}
	if strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return false, fmt.Errorf("replay config error: 'dir' is not within the allowed base directory")
	}

	msgIndexVal, ok := config["user_message_index"]
	if !ok || msgIndexVal == nil {
		return false, nil
	}

	return true, nil
}

// getInvocationState retrieves the replay state for the current invocation.
func (p *replayPlugin) getInvocationState(ctx agent.CallbackContext) (*invocationReplayState, error) {
	invocationID := ctx.InvocationID()
	state, ok := p.invocationStates[invocationID]
	if !ok {
		return nil, fmt.Errorf("replay state not initialized. ensure before_run created it")
	}
	return state, nil
}

// loadInvocationState loads the recordings and initializes the replay state for the invocation.
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

	basePath, err := filepath.Abs(p.allowedBaseDir)
	if err != nil {
		return nil, fmt.Errorf("invalid path format: %v", err)
	}
	requestedAbsPath, err := filepath.Abs(caseDir)
	if err != nil {
		return nil, fmt.Errorf("invalid path format: %v", err)
	}
	rel, err := filepath.Rel(basePath, requestedAbsPath)
	if err != nil {
		return nil, fmt.Errorf("invalid path format: %v", err)
	}
	if strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return nil, fmt.Errorf("replay config error: 'dir' is not within the allowed base directory")
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
	recordingsPath := filepath.Join(requestedAbsPath, "generated-recordings.yaml")

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
	var root yaml.Node
	if err := yaml.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("failed to parse recordings from %s: %w", recordingsPath, err)
	}

	removeUnderscores(&root)
	fixTypeMismatches(&root)

	var recordings recording.Recordings
	if err := root.Decode(&recordings); err != nil {
		return nil, fmt.Errorf("failed to decode recordings: %w", err)
	}

	// Add index to each recording, based on user message index. Used for parallel execution sync.
	index := 0
	prevMessageId := 0
	for i := range recordings.Recordings {
		if prevMessageId != recordings.Recordings[i].UserMessageIndex {
			prevMessageId = recordings.Recordings[i].UserMessageIndex
			index = 0
		}
		if recordings.Recordings[i].LLMRecording != nil {
			recordings.Recordings[i].Index = index
			index++
		} else {
			recordings.Recordings[i].Index = -1 // Not used for sync
		}
	}

	// 4. Create and Store State
	state := newInvocationReplayState(caseDir, msgIndex, &recordings)

	p.mu.Lock()
	p.invocationStates[invocationID] = state
	p.mu.Unlock()

	return state, nil
}

// getNextRecordingForAgent retrieves the next expected recording for the given agent.
// It enforces ordering of events within the user message turn to ensure deterministic replay.
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
	}
	// FIXME: remove this sleep, move curIndex++ and state cond.Broadcast() to onEvent callback.
	// This sleep is here to make the replay deterministic, but it's not ideal.
	time.Sleep(time.Duration(expectedRecording.Index) * time.Millisecond * 10)

	state.agentReplayIndices[agentName]++
	state.curIndex++

	state.mu.Unlock()
	state.cond.Broadcast()

	return expectedRecording, nil
}

// verifyAndGetNextLLMRecordingForAgent ensures the next recording is an LLM request and matches the actual request.
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
	err = verifyLLMRequestMatch(expectedRecording.LLMRecording.LLMRequest, llmRequest, agentName, currentAgentIndex)
	if err != nil {
		return nil, err
	}

	return expectedRecording.LLMRecording, nil
}

// verifyLLMRequestMatch compares the expected LLM request from recording with the actual request.
func verifyLLMRequestMatch(expectedLLMRequest, actualLLMRequest *model.LLMRequest, agentName string, agentIndex int) error {
	// Define options to ignore specific fields.
	opts := []cmp.Option{
		cmpopts.IgnoreFields(genai.FunctionDeclaration{}, "ParametersJsonSchema", "ResponseJsonSchema", "Parameters", "Response"),
		cmpopts.IgnoreFields(model.LLMRequest{}, "Tools"),
		cmpopts.IgnoreFields(genai.GenerateContentConfig{}, "Labels"),
		cmpopts.EquateEmpty(),
	}

	if transferToolAny, ok := expectedLLMRequest.Tools["transfer_to_agent"]; ok {
		transferTool := transferToolAny.(*genai.FunctionDeclaration)
		transferTool.Description = `Transfer the question to another agent.
This tool hands off control to another agent when it's more suitable to answer the user's question according to the agent's description.`
	}

	if expectedLLMRequest.Config != nil {
		for _, tool := range expectedLLMRequest.Config.Tools {
			for _, funcDecl := range tool.FunctionDeclarations {
				if funcDecl.Name == "transfer_to_agent" {
					funcDecl.Description = `Transfer the question to another agent.
This tool hands off control to another agent when it's more suitable to answer the user's question according to the agent's description.`
				}
			}
		}
	}

	// Compare!
	// cmp.Diff returns an empty string if they are equal, otherwise a human-readable diff.
	if diff := cmp.Diff(expectedLLMRequest, actualLLMRequest, opts...); diff != "" {
		for _, content := range expectedLLMRequest.Contents {
			for _, part := range content.Parts {
				if part.Text != "" {
					part.Text = modifyString(part.Text)
				}
			}
		}

		if diff := cmp.Diff(expectedLLMRequest, actualLLMRequest, opts...); diff != "" {
			return fmt.Errorf("LLM request mismatch for agent '%s' (index %d):\n%s",
				agentName, agentIndex, diff)
		}
	}

	return nil
}

var (
	// Matches either "parameters: " or "result: " followed by a JSON-like object/array
	dataBlockRegex = regexp.MustCompile(`(?i)(parameters|result):\s*([\{\[].*[\}\]])`)
	// Matches 'key' or 'value' but ignores apostrophes inside words like O'Malley
	quoteRegex = regexp.MustCompile(`'([^']*)'`)
	// Matches Python/Pseudo-JSON constants specifically as values
	nullRegex = regexp.MustCompile(`\bNone\b`)
	boolRegex = regexp.MustCompile(`\b(True|False)\b`)
)

func modifyString(input string) string {
	// We use ReplaceAllStringFunc to process ONLY the captured data parts
	return dataBlockRegex.ReplaceAllStringFunc(input, func(fullMatch string) string {
		// Split label (e.g., "parameters:") from the data (e.g., "{'a': 1}")
		parts := dataBlockRegex.FindStringSubmatch(fullMatch)
		if len(parts) < 3 {
			return fullMatch
		}

		label := parts[1]
		rawData := parts[2]

		// Normalize Python-isms to JSON-isms
		// Replace single quotes with double quotes
		normalized := quoteRegex.ReplaceAllString(rawData, `"$1"`)
		// Replace None -> null
		normalized = nullRegex.ReplaceAllString(normalized, "null")
		// Replace True/False -> true/false
		normalized = boolRegex.ReplaceAllStringFunc(normalized, func(m string) string {
			return strings.ToLower(m)
		})

		// Round-trip through JSON to validate and clean up
		var parsed any
		if err := json.Unmarshal([]byte(normalized), &parsed); err != nil {
			// If it's still not valid JSON, return the original match to avoid corruption
			return fullMatch
		}

		// Marshal back to a clean, standard JSON string
		fixedJSON, err := json.Marshal(parsed)
		if err != nil {
			return fullMatch
		}

		return fmt.Sprintf("%s: %s", label, string(fixedJSON))
	})
}

// getNextToolRecordingForAgent retrieves the next unconsumed tool recording that matches the given function.
func getNextToolRecordingForAgent(state *invocationReplayState, agentName string, matchFn func(*recording.Recording) (bool, error)) (*recording.Recording, error) {
	state.mu.Lock()
	defer state.mu.Unlock()

	var firstError error

	for i := range state.recordings.Recordings {
		rec := &state.recordings.Recordings[i]
		if rec.UserMessageIndex != state.userMessageIndex || rec.AgentName != agentName {
			continue
		}
		if state.consumedRecordings[i] {
			continue
		}

		matched, err := matchFn(rec)
		if matched {
			state.consumedRecordings[i] = true
			return rec, nil
		}
		if firstError == nil && err != nil {
			firstError = err
		}
	}

	if firstError != nil {
		return nil, firstError
	}

	return nil, fmt.Errorf("no matching tool recording found for agent '%s' at user_message_index %d", agentName, state.userMessageIndex)
}

// verifyAndGetNextToolRecordingForAgent ensures the next recording is a tool call and matches the actual call.
func (p *replayPlugin) verifyAndGetNextToolRecordingForAgent(state *invocationReplayState, agentName string, t tool.Tool, args map[string]any) (*recording.ToolRecording, error) {
	matchFn := func(rec *recording.Recording) (bool, error) {
		if rec.ToolRecording == nil {
			return false, fmt.Errorf("expected tool recording for agent '%s', but found LLM recording", agentName)
		}
		err := verifyToolCallMatch(rec.ToolRecording.ToolCall, t.Name(), args, agentName, state.agentReplayIndices[agentName])
		return err == nil, err
	}

	expectedRecording, err := getNextToolRecordingForAgent(state, agentName, matchFn)
	if err != nil {
		return nil, err
	}

	state.mu.Lock()
	state.agentReplayIndices[agentName]++
	state.mu.Unlock()

	return expectedRecording.ToolRecording, nil
}

// verifyToolCallMatch compares the expected tool call from recording with the actual tool call.
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
