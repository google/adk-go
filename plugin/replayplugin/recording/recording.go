/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package recording

import (
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// LLMRecording represents a paired LLM request and response.
type LLMRecording struct {
	// Required. The LLM request.
	LlmRequest *llmRequestRecording `yaml:"llm_request,omitempty"`

	// Required. The LLM response.
	LlmResponse *llmResponseRecording `yaml:"llm_response,omitempty"`
}

type llmRequestRecording struct {
	Model    string                      `yaml:"model,omitempty"`
	Contents []*localContent             `yaml:"contents,omitempty"`
	Config   *localGenerateContentConfig `yaml:"config,omitempty"`
	Tools    map[string]any              `yaml:"tools,omitempty"`
}

func (l *llmRequestRecording) ToLLMRequest() *model.LLMRequest {
	return &model.LLMRequest{
		Model:    l.Model,
		Contents: transformContents(l.Contents),
		Config:   l.Config.ToGenAI(),
		Tools:    l.Tools,
	}
}

type localGenerateContentConfig struct {
	*genai.GenerateContentConfig
	SystemInstruction string       `yaml:"system_instruction,omitempty"`
	Temperature       *float32     `yaml:"temperature,omitempty"`
	Tools             []*localTool `yaml:"tools,omitempty"`
}

func (l *localGenerateContentConfig) ToGenAI() *genai.GenerateContentConfig {
	if l == nil {
		return nil
	}
	out := l.GenerateContentConfig
	if out == nil {
		out = &genai.GenerateContentConfig{}
	}
	out.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: l.SystemInstruction}}, Role: genai.RoleUser}
	out.Temperature = l.Temperature
	tools := make([]*genai.Tool, len(l.Tools))
	for i, t := range l.Tools {
		tools[i] = t.ToGenAI()
	}
	out.Tools = tools
	return out
}

type localTool struct {
	*genai.Tool
	FunctionDeclarations []localFunctionDeclaration `yaml:"function_declarations,omitempty"`
	GoogleSearch         *genai.GoogleSearch        `yaml:"google_search,omitempty"`
	GoogleMaps           *genai.GoogleMaps          `yaml:"google_maps,omitempty"`
	URLContext           *genai.URLContext          `yaml:"url_context,omitempty"`
}

func (l *localTool) ToGenAI() *genai.Tool {
	if l == nil {
		return nil
	}
	functionDeclarations := make([]*genai.FunctionDeclaration, len(l.FunctionDeclarations))
	for i, fd := range l.FunctionDeclarations {
		functionDeclarations[i] = fd.ToGenAI()
	}
	tool := l.Tool
	if tool == nil {
		tool = &genai.Tool{}
	}
	tool.FunctionDeclarations = functionDeclarations
	tool.GoogleSearch = l.GoogleSearch
	tool.GoogleMaps = l.GoogleMaps
	tool.URLContext = l.URLContext
	return tool
}

type localFunctionDeclaration struct {
	Name        string `yaml:"name,omitempty"`
	Description string `yaml:"description,omitempty"`
}

func (l *localFunctionDeclaration) ToGenAI() *genai.FunctionDeclaration {
	if l == nil {
		return nil
	}
	return &genai.FunctionDeclaration{
		Name:        l.Name,
		Description: l.Description,
	}
}

type llmResponseRecording struct {
	Content           *localContent           `yaml:"content,omitempty"`
	GroundingMetadata *localGroundingMetadata `yaml:"grounding_metadata,omitempty"`
	UsageMetadata     *localUsageMetadata     `yaml:"usage_metadata,omitempty"`
	LogprobsResult    *genai.LogprobsResult   `yaml:"logprobs_result,omitempty"`
	Partial           bool                    `yaml:"partial,omitempty"`
	TurnComplete      bool                    `yaml:"turn_complete,omitempty"`
	Interrupted       bool                    `yaml:"interrupted,omitempty"`
	ErrorCode         string                  `yaml:"error_code,omitempty"`
	ErrorMessage      string                  `yaml:"error_message,omitempty"`
	FinishReason      genai.FinishReason      `yaml:"finish_reason,omitempty"`
	AvgLogprobs       float64                 `yaml:"avg_logprobs,omitempty"`
	ModelVersion      string                  `yaml:"model_version,omitempty"`
}

func (l *llmResponseRecording) ToLLMResponse() *model.LLMResponse {
	return &model.LLMResponse{
		Content:           l.Content.ToGenAI(),
		GroundingMetadata: l.GroundingMetadata.ToGenAI(),
		UsageMetadata:     l.UsageMetadata.ToGenAI(),
		LogprobsResult:    l.LogprobsResult,
		Partial:           l.Partial,
		TurnComplete:      l.TurnComplete,
		Interrupted:       l.Interrupted,
		ErrorCode:         l.ErrorCode,
		ErrorMessage:      l.ErrorMessage,
		FinishReason:      l.FinishReason,
		AvgLogprobs:       l.AvgLogprobs,
		// TODO: ModelVersion:      l.ModelVersion,
	}
}

type localUsageMetadata struct {
	CacheTokensDetails         []*localModalityTokenCount `yaml:"cache_tokens_details,omitempty"`
	CachedContentTokenCount    int32                      `yaml:"cached_content_token_count,omitempty"`
	CandidatesTokenCount       int32                      `yaml:"candidates_token_count,omitempty"`
	CandidatesTokensDetails    []*localModalityTokenCount `yaml:"candidates_tokens_details,omitempty"`
	PromptTokenCount           int32                      `yaml:"prompt_token_count,omitempty"`
	PromptTokensDetails        []*localModalityTokenCount `yaml:"prompt_tokens_details,omitempty"`
	ThoughtsTokenCount         int32                      `yaml:"thoughts_token_count,omitempty"`
	ToolUsePromptTokenCount    int32                      `yaml:"tool_use_prompt_token_count,omitempty"`
	ToolUsePromptTokensDetails []*localModalityTokenCount `yaml:"tool_use_prompt_tokens_details,omitempty"`
	TotalTokenCount            int32                      `yaml:"total_token_count,omitempty"`
	TrafficType                string                     `yaml:"traffic_type,omitempty"`
}

func (l *localUsageMetadata) ToGenAI() *genai.GenerateContentResponseUsageMetadata {
	if l == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		CacheTokensDetails:         transformModalityTokenCount(l.CacheTokensDetails),
		CachedContentTokenCount:    l.CachedContentTokenCount,
		CandidatesTokenCount:       l.CandidatesTokenCount,
		CandidatesTokensDetails:    transformModalityTokenCount(l.CandidatesTokensDetails),
		PromptTokenCount:           l.PromptTokenCount,
		PromptTokensDetails:        transformModalityTokenCount(l.PromptTokensDetails),
		ThoughtsTokenCount:         l.ThoughtsTokenCount,
		ToolUsePromptTokenCount:    l.ToolUsePromptTokenCount,
		ToolUsePromptTokensDetails: transformModalityTokenCount(l.ToolUsePromptTokensDetails),
		TotalTokenCount:            l.TotalTokenCount,
		TrafficType:                genai.TrafficType(l.TrafficType),
	}
}

func transformModalityTokenCount(l []*localModalityTokenCount) []*genai.ModalityTokenCount {
	if l == nil {
		return nil
	}
	var result []*genai.ModalityTokenCount
	for _, item := range l {
		result = append(result, item.ToGenAI())
	}
	return result
}

func (l *localModalityTokenCount) ToGenAI() *genai.ModalityTokenCount {
	if l == nil {
		return nil
	}
	return &genai.ModalityTokenCount{
		Modality:   l.Modality,
		TokenCount: l.TokenCount,
	}
}

type localModalityTokenCount struct {
	Modality   genai.MediaModality `yaml:"modality,omitempty"`
	TokenCount int32               `yaml:"token_count,omitempty"`
}

type localGroundingMetadata struct {
	GoogleMapsWidgetContextToken string                   `yaml:"google_maps_widget_context_token,omitempty"`
	GroundingChunks              []*localGroundingChunk   `yaml:"grounding_chunks,omitempty"`
	GroundingSupports            []*localGroundingSupport `yaml:"grounding_supports,omitempty"`
	RetrievalMetadata            *localRetrievalMetadata  `yaml:"retrieval_metadata,omitempty"`
	RetrievalQueries             []string                 `yaml:"retrieval_queries,omitempty"`
	SearchEntryPoint             *localSearchEntryPoint   `yaml:"search_entry_point,omitempty"`
	WebSearchQueries             []string                 `yaml:"web_search_queries,omitempty"`
}

func (l *localGroundingMetadata) ToGenAI() *genai.GroundingMetadata {
	if l == nil {
		return nil
	}
	return &genai.GroundingMetadata{
		GoogleMapsWidgetContextToken: l.GoogleMapsWidgetContextToken,
		GroundingChunks:              transformGroundingChunks(l.GroundingChunks),
		GroundingSupports:            transformGroundingSupports(l.GroundingSupports),
		RetrievalMetadata:            l.RetrievalMetadata.ToGenAI(),
		RetrievalQueries:             l.RetrievalQueries,
		SearchEntryPoint:             l.SearchEntryPoint.ToGenAI(),
		WebSearchQueries:             l.WebSearchQueries,
	}
}

type localGroundingChunk struct {
	Web        *localGroundingChunkWeb        `yaml:"web,omitempty"`
	GoogleMaps *localGroundingChunkGoogleMaps `yaml:"maps,omitempty"`
}

func transformGroundingChunks(l []*localGroundingChunk) []*genai.GroundingChunk {
	if l == nil {
		return nil
	}
	var result []*genai.GroundingChunk
	for _, item := range l {
		result = append(result, item.ToGenAI())
	}
	return result
}

func (l *localGroundingChunk) ToGenAI() *genai.GroundingChunk {
	if l == nil {
		return nil
	}
	return &genai.GroundingChunk{
		Web:  l.Web.ToGenAI(),
		Maps: l.GoogleMaps.ToGenAI(),
	}
}

type localGroundingChunkWeb struct {
	Domain string `yaml:"domain,omitempty"`
	Title  string `yaml:"title,omitempty"`
	URI    string `yaml:"uri,omitempty"`
}

func (l *localGroundingChunkWeb) ToGenAI() *genai.GroundingChunkWeb {
	if l == nil {
		return nil
	}
	return &genai.GroundingChunkWeb{
		Domain: l.Domain,
		Title:  l.Title,
		URI:    l.URI,
	}
}

type localGroundingChunkGoogleMaps struct {
	PlaceID string `yaml:"place_id,omitempty"`
	Text    string `yaml:"text,omitempty"`
	Title   string `yaml:"title,omitempty"`
	URI     string `yaml:"uri,omitempty"`
}

func (l *localGroundingChunkGoogleMaps) ToGenAI() *genai.GroundingChunkMaps {
	if l == nil {
		return nil
	}
	return &genai.GroundingChunkMaps{
		PlaceID: l.PlaceID,
		Text:    l.Text,
		Title:   l.Title,
		URI:     l.URI,
	}
}

type localGroundingSupport struct {
	ConfidenceScores      []float32     `yaml:"confidence_scores,omitempty"`
	GroundingChunkIndices []int32       `yaml:"grounding_chunk_indices,omitempty"`
	Segment               *localSegment `yaml:"segment,omitempty"`
}

func transformGroundingSupports(l []*localGroundingSupport) []*genai.GroundingSupport {
	if l == nil {
		return nil
	}
	var result []*genai.GroundingSupport
	for _, item := range l {
		result = append(result, item.ToGenAI())
	}
	return result
}

func (l *localGroundingSupport) ToGenAI() *genai.GroundingSupport {
	if l == nil {
		return nil
	}
	return &genai.GroundingSupport{
		ConfidenceScores:      l.ConfidenceScores,
		GroundingChunkIndices: l.GroundingChunkIndices,
		Segment:               l.Segment.ToGenAI(),
	}
}

type localSegment struct {
	EndIndex   int32  `yaml:"end_index,omitempty"`
	PartIndex  int32  `yaml:"part_index,omitempty"`
	StartIndex int32  `yaml:"start_index,omitempty"`
	Text       string `yaml:"text,omitempty"`
}

func (l *localSegment) ToGenAI() *genai.Segment {
	if l == nil {
		return nil
	}
	return &genai.Segment{
		EndIndex:   l.EndIndex,
		PartIndex:  l.PartIndex,
		StartIndex: l.StartIndex,
		Text:       l.Text,
	}
}

type localRetrievalMetadata struct {
	GoogleSearchDynamicRetrievalScore float32 `yaml:"google_search_dynamic_retrieval_score,omitempty"`
}

func (l *localRetrievalMetadata) ToGenAI() *genai.RetrievalMetadata {
	if l == nil {
		return nil
	}
	return &genai.RetrievalMetadata{
		GoogleSearchDynamicRetrievalScore: l.GoogleSearchDynamicRetrievalScore,
	}
}

type localSearchEntryPoint struct {
	RenderedContent string `yaml:"rendered_content,omitempty"`
	SDKBlob         []byte `yaml:"sdk_blob,omitempty"`
}

func (l *localSearchEntryPoint) ToGenAI() *genai.SearchEntryPoint {
	if l == nil {
		return nil
	}
	return &genai.SearchEntryPoint{
		RenderedContent: l.RenderedContent,
		SDKBlob:         l.SDKBlob,
	}
}

// ToolRecording represents a paired tool call and response.
type ToolRecording struct {
	// Required. The tool call.
	ToolCall *localFunctionCall `yaml:"tool_call,omitempty"`

	// Required. The tool response.
	ToolResponse *localFunctionResponse `yaml:"tool_response,omitempty"`
}

type localContent struct {
	Parts []*localPart `yaml:"parts,omitempty"`
	Role  string       `yaml:"role,omitempty"`
}

func transformContents(l []*localContent) []*genai.Content {
	if l == nil {
		return nil
	}
	var result []*genai.Content
	for _, item := range l {
		result = append(result, item.ToGenAI())
	}
	return result
}

func (l *localContent) ToGenAI() *genai.Content {
	if l == nil {
		return nil
	}
	return &genai.Content{
		Parts: transformParts(l.Parts),
		Role:  l.Role,
	}
}

func transformParts(l []*localPart) []*genai.Part {
	if l == nil {
		return nil
	}
	var result []*genai.Part
	for _, item := range l {
		result = append(result, item.ToGenAI())
	}
	return result
}

type localPart struct {
	*genai.Part
	Text             string                 `yaml:"text,omitempty"`
	FunctionCall     *localFunctionCall     `yaml:"function_call,omitempty"`
	FunctionResponse *localFunctionResponse `yaml:"function_response,omitempty"`
}

func (l *localPart) ToGenAI() *genai.Part {
	if l == nil {
		return nil
	}
	out := l.Part
	if out == nil {
		out = &genai.Part{}
	}
	out.Text = l.Text
	out.FunctionCall = l.FunctionCall.ToGenAI()
	out.FunctionResponse = l.FunctionResponse.ToGenAI()
	return out
}

type localFunctionCall struct {
	*genai.FunctionCall
	ID   string         `yaml:"id,omitempty"`
	Args map[string]any `yaml:"args,omitempty"`
	Name string         `yaml:"name,omitempty"`
}

func (l *localFunctionCall) ToGenAI() *genai.FunctionCall {
	if l == nil {
		return nil
	}
	out := l.FunctionCall
	if out == nil {
		out = &genai.FunctionCall{}
	}
	out.ID = l.ID
	out.Args = l.Args
	out.Name = l.Name
	return out
}

type localFunctionResponse struct {
	*genai.FunctionResponse
	ID       string         `yaml:"id,omitempty"`
	Name     string         `yaml:"name,omitempty"`
	Response map[string]any `yaml:"response,omitempty"`
}

func (l *localFunctionResponse) ToGenAI() *genai.FunctionResponse {
	if l == nil {
		return nil
	}
	out := l.FunctionResponse
	if out == nil {
		out = &genai.FunctionResponse{}
	}
	out.ID = l.ID
	out.Name = l.Name
	out.Response = l.Response
	return out
}

// Recording represents a single interaction recording, ordered by request timestamp.
type Recording struct {
	// Index of the user message this recording belongs to (0-based).
	UserMessageIndex int `yaml:"user_message_index"`

	// Name of the agent.
	AgentName string `yaml:"agent_name"`

	// oneof fields - start

	// LLM request-response pair.
	LLMRecording *LLMRecording `yaml:"llm_recording,omitempty"`

	// Tool call-response pair.
	ToolRecording *ToolRecording `yaml:"tool_recording,omitempty"`

	// oneof fields - end

	// Index of the recording in the recordings list (0-based).
	Index int `yaml:"-"`
}

// Recordings represents all recordings in chronological order.
type Recordings struct {
	// Chronological list of all recordings.
	Recordings []Recording `yaml:"recordings"`
}
