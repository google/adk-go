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

package recordplugin

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable/conformance/replayplugin/recording"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type recordPlugin struct {
	mu               sync.Mutex
	invocationStates map[string]*invocationRecordState
	allowedBaseDir   string
}

// New creates a new conformance record plugin.
func New(allowedBaseDir string) (*plugin.Plugin, error) {
	p := &recordPlugin{
		invocationStates: make(map[string]*invocationRecordState),
		allowedBaseDir:   allowedBaseDir,
	}
	return plugin.New(plugin.Config{
		Name:                "record_plugin",
		BeforeRunCallback:   p.beforeRun,
		AfterRunCallback:    p.afterRun,
		BeforeModelCallback: p.beforeModel,
		AfterModelCallback:  p.afterModel,
		BeforeToolCallback:  p.beforeTool,
		AfterToolCallback:   p.afterTool,
	})
}

// MustNew is like New but panics on error.
func MustNew(allowedBaseDir string) *plugin.Plugin {
	p, err := New(allowedBaseDir)
	if err != nil {
		panic(err)
	}
	return p
}

func (p *recordPlugin) beforeRun(ctx agent.InvocationContext) (*genai.Content, error) {
	if ctx.Session() == nil {
		return nil, nil
	}

	on, err := p.isRecordModeOn(ctx.Session().State())
	if err != nil {
		return nil, err
	}
	if !on {
		return nil, nil
	}

	// Create fresh record state for this invocation
	_, err = p.createInvocationState(ctx)
	return nil, err
}

func (p *recordPlugin) beforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	on, err := p.isRecordModeOn(ctx.State())
	if err != nil || !on {
		return nil, nil
	}

	state, err := p.getInvocationState(ctx.InvocationID())
	if err != nil {
		return nil, err
	}

	// Sanitize and clone the request to prevent YAML serialization panics caused by HTTPOptions.ExtrasRequestProvider
	sanitizedReq := sanitizeLLMRequest(req)

	state.StartLLMRecording(sanitizedReq)
	return nil, nil
}

func (p *recordPlugin) afterModel(ctx agent.CallbackContext, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
	on, recordErr := p.isRecordModeOn(ctx.State())
	if recordErr != nil || !on {
		return nil, nil
	}

	if err != nil || resp == nil {
		return nil, nil
	}

	state, stateErr := p.getInvocationState(ctx.InvocationID())
	if stateErr != nil {
		return nil, stateErr
	}

	state.AppendLLMResponse(resp)

	// If it is a final non-partial response, complete the recording
	if !resp.Partial {
		state.CompleteLLMRecording(ctx.AgentName())
	}

	return nil, nil
}

func (p *recordPlugin) beforeTool(ctx agent.ToolContext, t tool.Tool, args map[string]any) (map[string]any, error) {
	on, err := p.isRecordModeOn(ctx.State())
	if err != nil || !on {
		return nil, nil
	}

	state, err := p.getInvocationState(ctx.InvocationID())
	if err != nil {
		return nil, err
	}

	fc := &genai.FunctionCall{
		ID:   ctx.FunctionCallID(),
		Name: t.Name(),
		Args: args,
	}

	state.StartToolRecording(ctx.FunctionCallID(), fc)
	return nil, nil
}

func (p *recordPlugin) afterTool(ctx agent.ToolContext, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	on, recordErr := p.isRecordModeOn(ctx.State())
	if recordErr != nil || !on {
		return nil, nil
	}

	state, stateErr := p.getInvocationState(ctx.InvocationID())
	if stateErr != nil {
		return nil, stateErr
	}

	resp := &genai.FunctionResponse{
		ID:       ctx.FunctionCallID(),
		Name:     t.Name(),
		Response: result,
	}

	state.CompleteToolRecording(ctx.FunctionCallID(), ctx.AgentName(), resp)
	return nil, nil
}

func (p *recordPlugin) afterRun(ctx agent.InvocationContext) {
	if ctx.Session() == nil {
		return
	}

	on, err := p.isRecordModeOn(ctx.Session().State())
	if err != nil || !on {
		return
	}

	state, err := p.getInvocationState(ctx.InvocationID())
	if err != nil {
		return
	}

	defer func() {
		p.mu.Lock()
		delete(p.invocationStates, ctx.InvocationID())
		p.mu.Unlock()
	}()

	// Serialize and save
	p.saveRecordings(state)
}

func (p *recordPlugin) parseRecordConfig(sessionState session.State) (string, int, string, error) {
	if sessionState == nil {
		return "", 0, "", nil
	}

	configVal, err := sessionState.Get("_adk_recordings_config")
	if err != nil {
		return "", 0, "", nil
	}

	config, ok := configVal.(map[string]any)
	if !ok {
		return "", 0, "", nil
	}

	caseDir, ok := config["dir"].(string)
	if !ok || caseDir == "" {
		return "", 0, "", nil
	}

	basePath, err := filepath.Abs(p.allowedBaseDir)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid path format: %v", err)
	}
	requestedAbsPath, err := filepath.Abs(caseDir)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid path format: %v", err)
	}
	rel, err := filepath.Rel(basePath, requestedAbsPath)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid path format: %v", err)
	}
	if strings.HasPrefix(rel, "..") || filepath.IsAbs(rel) {
		return "", 0, "", fmt.Errorf("record config error: 'dir' is not within the allowed base directory")
	}

	msgIndexVal, ok := config["user_message_index"]
	if !ok || msgIndexVal == nil {
		return "", 0, "", nil
	}

	var msgIndex int
	switch v := msgIndexVal.(type) {
	case int:
		msgIndex = v
	case float64:
		msgIndex = int(v)
	default:
		return "", 0, "", fmt.Errorf("record config 'user_message_index' is not a number")
	}

	streamingMode, _ := config["streaming_mode"].(string)
	if streamingMode == "" {
		streamingMode = "none"
	}

	return caseDir, msgIndex, streamingMode, nil
}

func (p *recordPlugin) isRecordModeOn(sessionState session.State) (bool, error) {
	caseDir, _, _, err := p.parseRecordConfig(sessionState)
	if err != nil {
		return false, err
	}
	return caseDir != "", nil
}

func (p *recordPlugin) getInvocationState(id string) (*invocationRecordState, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	state, ok := p.invocationStates[id]
	if !ok {
		return nil, fmt.Errorf("record state not initialized for invocation: %s", id)
	}
	return state, nil
}

func (p *recordPlugin) createInvocationState(ctx agent.InvocationContext) (*invocationRecordState, error) {
	caseDir, msgIndex, streamingMode, err := p.parseRecordConfig(ctx.Session().State())
	if err != nil {
		return nil, err
	}
	if caseDir == "" {
		return nil, fmt.Errorf("record state not configured")
	}

	state := newInvocationRecordState(caseDir, msgIndex, streamingMode)

	p.mu.Lock()
	p.invocationStates[ctx.InvocationID()] = state
	p.mu.Unlock()

	return state, nil
}

func (p *recordPlugin) saveRecordings(state *invocationRecordState) {
	var filename string
	if state.streamingMode == "sse" {
		filename = "generated-recordings-sse.yaml"
	} else {
		filename = "generated-recordings.yaml"
	}

	filePath := filepath.Join(state.caseDir, filename)

	// Load existing recordings if the file exists
	var existingRecordings recording.Recordings
	if _, err := os.Stat(filePath); err == nil {
		data, err := os.ReadFile(filePath)
		if err == nil {
			_ = yaml.Unmarshal(data, &existingRecordings)
		}
	}

	// Filter out any existing recordings for the CURRENT user_message_index
	var filtered []recording.Recording
	for _, r := range existingRecordings.Recordings {
		if r.UserMessageIndex != state.userMessageIndex {
			filtered = append(filtered, r)
		}
	}

	// Append newly completed recordings in this turn
	state.mu.Lock()
	filtered = append(filtered, state.recordings...)
	state.mu.Unlock()

	outputRecordings := recording.Recordings{
		Recordings: filtered,
	}

	// Write YAML back to disk using AST traversal to prune nulls/defaults
	var node yaml.Node
	if err := node.Encode(&outputRecordings); err != nil {
		return
	}

	cleanYAMLNode(&node, false)

	out, err := yaml.Marshal(&node)
	if err != nil {
		return
	}

	_ = os.WriteFile(filePath, out, 0o644)
}

func shouldPruneYAMLField(key string, value *yaml.Node, inToolArgs bool) bool {
	if key == "usermessageindex" || key == "user_message_index" {
		return false
	}
	if !inToolArgs {
		if key == "property_order" || key == "response_json_schema" {
			return true
		}
	}
	if value.Kind == yaml.ScalarNode {
		if value.Tag == "!!null" || value.Value == "null" || value.Value == "~" {
			return true
		}
		if !inToolArgs {
			if (value.Tag == "!!int" || value.Tag == "") && value.Value == "0" {
				return true
			}
			if (value.Tag == "!!bool" || value.Tag == "") && value.Value == "false" {
				return true
			}
			if value.Value == "" {
				return true
			}
		}
	}
	if value.Kind == yaml.SequenceNode && len(value.Content) == 0 {
		return true
	}
	if value.Kind == yaml.MappingNode && len(value.Content) == 0 {
		if inToolArgs {
			return false
		}
		switch key {
		case "google_search", "code_execution", "retrieval":
			return false
		default:
			return true
		}
	}
	return false
}

var snakeCaseFieldMap = make(map[string]string)

func init() {
	snakeCaseKeys := []string{
		// Top-level and Recording structs
		"user_message_index", "agent_name", "llm_recording", "llm_request", "llm_responses",
		"tool_recording", "tool_call", "tool_response",

		// LLMResponse fields
		"citation_metadata", "grounding_metadata", "usage_metadata", "custom_metadata",
		"logprobs_result", "input_transcription", "output_transcription", "model_version",
		"session_resumption_handle", "error_code", "error_message", "finish_reason", "avg_logprobs",

		// GenerateContentConfig fields
		"http_options", "system_instruction", "candidate_count", "max_output_tokens",
		"stop_sequences", "response_logprobs", "presence_penalty", "frequency_penalty",
		"response_mime_type", "response_schema", "response_json_schema", "routing_config",
		"model_selection_config", "safety_settings", "tool_config", "cached_content",
		"response_modalities", "media_resolution", "speech_config", "audio_timestamp",
		"thinking_config", "image_config", "enable_enhanced_civic_answers", "model_armor_config",
		"service_tier",

		// Tool fields
		"google_search", "blocking_confidence", "function_declarations", "parameters_json_schema",
		"additional_properties", "property_order",

		// Part / Content fields
		"part_metadata", "video_metadata", "code_execution_result", "executable_code",
		"file_data", "function_call", "function_call_id", "function_response", "inline_data",
		"thought_signature",

		// GroundingMetadata fields
		"grounding_chunks", "grounding_supports", "retrieval_metadata", "search_entry_point",
		"web_search_queries",

		// GroundingChunk / Support / Segment / SearchEntryPoint fields
		"grounding_chunk_indices", "confidence_score", "start_index", "end_index",
		"rendered_content", "sdk_active_queries",

		// Token / usage details
		"google_maps_widget_context_token", "candidates_token_count", "prompt_token_count",
		"prompt_tokens_details", "token_count", "thoughts_token_count", "tool_use_prompt_token_count",
		"tool_use_prompt_tokens_details", "total_token_count", "traffic_type",
	}

	for _, key := range snakeCaseKeys {
		cleaned := strings.ReplaceAll(key, "_", "")
		snakeCaseFieldMap[cleaned] = key
	}

	// Manual edge cases
	snakeCaseFieldMap["propertyordering"] = "property_order"
}

// findKeyNode searches a MappingNode's children to locate a value node associated
// with the specified key string. Returns nil if the node is not a mapping or if the key is not found.
func findKeyNode(node *yaml.Node, key string) *yaml.Node {
	if node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

// cleanYAMLNode is the a post-processor for the recorded YAML representation.
// It performs a deep recursive AST traversal to:
//
//	Normalize Go-serialized lowercase keys back to their strict canonical snake_case.
//	Auto-extract and simplify standard system_instruction structures to single scalar strings.
//	Coalesce raw integer-sequence thought_signatures back to compact Base64 strings.
//	Prune all default empty fields, empty sequences, and null properties outside user tool arguments.
func cleanYAMLNode(node *yaml.Node, inToolArgs bool) {
	switch node.Kind {
	case yaml.DocumentNode, yaml.SequenceNode:
		for _, child := range node.Content {
			cleanYAMLNode(child, inToolArgs)
		}
	case yaml.MappingNode:
		// If this mapping represents an OpenAPI function declaration, recursively inject parameter schema titles.
		// This ensures the generated schemas exactly match Pydantic reference standards.
		var toolName string
		if nameNode := findKeyNode(node, "name"); nameNode != nil && nameNode.Kind == yaml.ScalarNode {
			toolName = nameNode.Value
		}
		var paramSchemaNode *yaml.Node
		for _, key := range []string{"parameters", "parameters_json_schema", "parametersjsonschema"} {
			if p := findKeyNode(node, key); p != nil && p.Kind == yaml.MappingNode {
				paramSchemaNode = p
				break
			}
		}
		if toolName != "" && paramSchemaNode != nil {
			injectSchemaTitles(paramSchemaNode, toolName)
		}

		var newContent []*yaml.Node
		for i := 0; i < len(node.Content); i += 2 {
			k := node.Content[i]
			v := node.Content[i+1]

			// Translate lowercase serialized struct fields back to canonical snake_case
			if mappedKey, exists := snakeCaseFieldMap[k.Value]; exists {
				k.Value = mappedKey
			}

			// Set insideToolArgs context flag so we avoid pruning user data and parameter payloads
			nextInToolArgs := inToolArgs
			if k.Value == "args" || k.Value == "response" || k.Value == "tool_call" || k.Value == "tool_response" {
				nextInToolArgs = true
			}

			// Extract structured system instructions (Content with Parts) into simple string scalars
			if k.Value == "system_instruction" && v.Kind == yaml.MappingNode {
				if parts := findKeyNode(v, "parts"); parts != nil && parts.Kind == yaml.SequenceNode && len(parts.Content) > 0 {
					part := parts.Content[0]
					if text := findKeyNode(part, "text"); text != nil && text.Kind == yaml.ScalarNode {
						v.Kind = yaml.ScalarNode
						v.Tag = "!!str"
						v.Style = text.Style
						v.Value = text.Value
						v.Content = nil
					}
				}
			}

			// Coalesce raw integer sequences representing byte signatures back to standard Base64 string representation
			if k.Value == "thought_signature" && v.Kind == yaml.SequenceNode && len(v.Content) > 0 {
				var bytes []byte
				for _, child := range v.Content {
					if child.Kind == yaml.ScalarNode && (child.Tag == "!!int" || child.Tag == "") {
						var b int
						_, err := fmt.Sscan(child.Value, &b)
						if err == nil {
							bytes = append(bytes, byte(b))
						}
					}
				}
				v.Kind = yaml.ScalarNode
				v.Style = yaml.DoubleQuotedStyle
				v.Tag = "!!str"
				v.Value = base64.StdEncoding.EncodeToString(bytes)
				v.Content = nil
			}

			cleanYAMLNode(v, nextInToolArgs)

			// Drop empty, null, or redundant fields outside tool parameters
			if shouldPruneYAMLField(k.Value, v, nextInToolArgs) {
				continue
			}

			newContent = append(newContent, k, v)
		}
		node.Content = newContent
	}
}

// injectSchemaTitles recursively verifies and inserts correct parameter schema titles
// for OpenAPI declarations, such as mapping "validate_email" to "validate_emailParams"
// and camelCasing individual property titles (e.g. "email_address" -> "Email Address").
func injectSchemaTitles(schemaNode *yaml.Node, toolName string) {
	if schemaNode.Kind != yaml.MappingNode {
		return
	}

	// Check or inject schema top-level title
	titleNode := findKeyNode(schemaNode, "title")
	if titleNode != nil {
		if titleNode.Value == "" || titleNode.Tag == "!!null" || titleNode.Value == "null" {
			titleNode.Kind = yaml.ScalarNode
			titleNode.Tag = "!!str"
			titleNode.Style = yaml.DoubleQuotedStyle
			titleNode.Value = toolName + "Params"
			titleNode.Content = nil
		}
	} else {
		titleKey := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "title"}
		titleValue := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: toolName + "Params", Style: yaml.DoubleQuotedStyle}
		schemaNode.Content = append([]*yaml.Node{titleKey, titleValue}, schemaNode.Content...)
	}

	// Recursively check and inject titles for individual properties
	if propertiesNode := findKeyNode(schemaNode, "properties"); propertiesNode != nil && propertiesNode.Kind == yaml.MappingNode {
		for i := 0; i < len(propertiesNode.Content); i += 2 {
			propNameNode := propertiesNode.Content[i]
			propDefNode := propertiesNode.Content[i+1]
			if propDefNode.Kind == yaml.MappingNode {
				propTitle := findKeyNode(propDefNode, "title")
				if propTitle != nil {
					if propTitle.Value == "" || propTitle.Tag == "!!null" || propTitle.Value == "null" {
						propTitle.Kind = yaml.ScalarNode
						propTitle.Tag = "!!str"
						propTitle.Style = yaml.DoubleQuotedStyle
						propTitle.Value = toCamelCaseTitle(propNameNode.Value)
						propTitle.Content = nil
					}
				} else {
					titleKey := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: "title"}
					titleValue := &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: toCamelCaseTitle(propNameNode.Value), Style: yaml.DoubleQuotedStyle}
					propDefNode.Content = append([]*yaml.Node{titleKey, titleValue}, propDefNode.Content...)
				}
			}
		}
	}
}

// toCamelCaseTitle translates a snake_case property key (e.g. "phone_number")
// into a human-readable Title Case string (e.g. "Phone Number") for schema metadata.
func toCamelCaseTitle(s string) string {
	parts := strings.Split(s, "_")
	for i, part := range parts {
		if len(part) > 0 {
			parts[i] = strings.ToUpper(part[:1]) + part[1:]
		}
	}
	return strings.Join(parts, " ")
}

// sanitizeLLMRequest creates a shallow copy of the LLMRequest and strips HTTPOptions
// and internal Tool registries to prevent YAML serialization panics and leaked internal state.
func sanitizeLLMRequest(req *model.LLMRequest) *model.LLMRequest {
	if req == nil {
		return nil
	}

	reqCopy := *req
	reqCopy.Tools = nil

	if req.Config != nil {
		configCopy := *req.Config
		configCopy.HTTPOptions = nil
		reqCopy.Config = &configCopy
	}

	return &reqCopy
}
