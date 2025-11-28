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

// Package openai implements the [model.LLM] interface for OpenAI-compatible APIs.
// This includes OpenAI API, LM Studio, Ollama, LocalAI and other compatible endpoints.
//
// # Basic Usage
//
//	model, err := openai.NewModel("gpt-4", &openai.Config{
//	    BaseURL: "https://api.openai.com/v1",
//	    APIKey:  os.Getenv("OPENAI_API_KEY"),
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.(*openaiModel).Close() // Important: prevents goroutine leak
//
// # Session Management
//
// The adapter maintains conversation history per session. To enable multi-turn
// conversations, you must pass a session ID in the context:
//
//	ctx := openai.WithSessionID(context.Background(), "user-123")
//	model.GenerateContent(ctx, req, false)
//
// Without a session ID, each request starts a new conversation and history is lost.
// If no session ID is provided, a UUID is auto-generated (with a warning logged).
//
// # Resource Cleanup
//
// The model starts a background goroutine for session cleanup. Call Close() when
// the model is no longer needed to prevent goroutine leaks:
//
//	if closer, ok := model.(interface{ Close() error }); ok {
//	    defer closer.Close()
//	}
//
// # Compatibility Notes
//
//   - Works with any OpenAI-compatible endpoint (just set BaseURL)
//   - API key is optional (for local servers like Ollama/LM Studio)
//   - Only sends standard headers (no cloud-specific headers)
//   - Tested with: OpenAI API, LM Studio, Google Gemma 3
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/model"
)

const (
	defaultMaxHistoryLength = 50
	defaultSessionTTL       = 1 * time.Hour
	defaultMaxRetries       = 3
	defaultTimeout          = 120 * time.Second
)

// OpenAIMessage represents a message in OpenAI chat format.
type OpenAIMessage struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content,omitempty"`
	Name       string      `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// ToolCall represents an OpenAI tool call.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Index    int          `json:"index,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call within a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool represents an OpenAI tool definition.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition for OpenAI tools.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ResponseFormat specifies the format of the model's output
type ResponseFormat struct {
	Type       string         `json:"type"`                  // "text", "json_object", or "json_schema"
	JSONSchema *JSONSchemaRef `json:"json_schema,omitempty"` // For structured outputs
}

// JSONSchemaRef references a JSON schema for structured outputs
type JSONSchemaRef struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict,omitempty"`
}

// ContentPart represents a part of multimodal content (text or image)
type ContentPartText struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

type ContentPartImage struct {
	Type     string `json:"type"` // "image_url"
	ImageURL struct {
		URL string `json:"url"`
	} `json:"image_url"`
}

// ChatCompletionRequest represents an OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model            string          `json:"model"`
	Messages         []OpenAIMessage `json:"messages"`
	Temperature      *float32        `json:"temperature,omitempty"`
	TopP             *float32        `json:"top_p,omitempty"`
	N                int32           `json:"n,omitempty"`
	MaxTokens        *int32          `json:"max_tokens,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	PresencePenalty  *float32        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float32        `json:"frequency_penalty,omitempty"`
	Seed             *int32          `json:"seed,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       interface{}     `json:"tool_choice,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	Logprobs         bool            `json:"logprobs,omitempty"`
	TopLogprobs      *int32          `json:"top_logprobs,omitempty"`
}

// ChatCompletionResponse represents an OpenAI chat completion response.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int              `json:"index"`
	Message      OpenAIMessage    `json:"message"`
	Delta        OpenAIMessage    `json:"delta,omitempty"`
	FinishReason string           `json:"finish_reason,omitempty"`
	Logprobs     *ChoiceLogprobs  `json:"logprobs,omitempty"`
}

// ChoiceLogprobs contains log probability information for the choice.
type ChoiceLogprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
}

// TokenLogprob represents log probability information for a single token.
type TokenLogprob struct {
	Token       string            `json:"token"`
	Logprob     float64           `json:"logprob"`
	Bytes       []int             `json:"bytes,omitempty"`
	TopLogprobs []TopTokenLogprob `json:"top_logprobs,omitempty"`
}

// TopTokenLogprob represents a top candidate token with its log probability.
type TopTokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// validateMessage validates an OpenAI message according to API rules.
// Returns an error if the message violates OpenAI API constraints:
//   - "tool" role messages must have ToolCallID
//   - "assistant" role messages with ToolCalls should have empty or minimal Content
//   - All messages must have a valid role
func validateMessage(msg *OpenAIMessage) error {
	if msg == nil {
		return fmt.Errorf("message cannot be nil")
	}

	// Validate role is not empty
	if msg.Role == "" {
		return fmt.Errorf("message role cannot be empty")
	}

	// Validate "tool" role messages
	if msg.Role == "tool" {
		if msg.ToolCallID == "" {
			return fmt.Errorf("tool role message must have ToolCallID")
		}
		// Tool messages should have content (the tool result)
		if msg.Content == nil || msg.Content == "" {
			return fmt.Errorf("tool role message must have content (tool result)")
		}
	}

	// Validate "assistant" role messages with tool calls
	if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
		// When assistant makes tool calls, content should be empty or null
		// Some models allow a brief message, but typically it's empty
		// We'll allow empty string, nil, or empty content
		if msg.Content != nil {
			if str, ok := msg.Content.(string); ok && str != "" {
				// Non-empty string content with tool calls is allowed but not typical
				// Log a debug message if logger is available, but don't error
			}
		}
	}

	// Validate tool calls if present
	for i, tc := range msg.ToolCalls {
		if tc.ID == "" {
			return fmt.Errorf("tool call at index %d must have an ID", i)
		}
		if tc.Type == "" {
			return fmt.Errorf("tool call at index %d must have a type", i)
		}
		if tc.Function.Name == "" {
			return fmt.Errorf("tool call at index %d must have a function name", i)
		}
		// Arguments can be empty for functions with no parameters
	}

	return nil
}

// validateMessageSequence validates the order and structure of a message sequence.
// According to OpenAI API rules:
//   - After an "assistant" message with tool_calls, there must be "tool" messages
//   - Each tool_call must have a corresponding tool message with matching tool_call_id
//   - No "user" messages should appear between tool_calls and their responses
func validateMessageSequence(messages []OpenAIMessage) error {
	if len(messages) == 0 {
		return nil
	}

	// Track pending tool calls that need responses
	pendingToolCalls := make(map[string]bool) // tool_call_id -> waiting for response

	for i, msg := range messages {
		// Validate individual message first
		if err := validateMessage(&msg); err != nil {
			return fmt.Errorf("invalid message at index %d: %w", i, err)
		}

		switch msg.Role {
		case "assistant":
			// If assistant has tool calls, track them as pending
			if len(msg.ToolCalls) > 0 {
				// Clear previous pending calls (starting a new tool call cycle)
				pendingToolCalls = make(map[string]bool)
				for _, tc := range msg.ToolCalls {
					pendingToolCalls[tc.ID] = true
				}
			}

		case "tool":
			// Tool message should resolve a pending tool call
			if msg.ToolCallID == "" {
				return fmt.Errorf("tool message at index %d has empty tool_call_id", i)
			}

			// Check if this tool call was expected
			if len(pendingToolCalls) == 0 {
				return fmt.Errorf("tool message at index %d has no corresponding tool_calls", i)
			}

			// Mark this tool call as resolved
			if pendingToolCalls[msg.ToolCallID] {
				delete(pendingToolCalls, msg.ToolCallID)
			} else {
				// Warning: tool_call_id doesn't match any pending call
				// This might happen in some edge cases, so we log but don't fail
				// In strict mode, you could return an error here
			}

		case "user":
			// User message should not appear while tool calls are pending
			if len(pendingToolCalls) > 0 {
				return fmt.Errorf("user message at index %d appears before tool responses are provided (pending: %d tool calls)", i, len(pendingToolCalls))
			}

		case "system":
			// System messages are typically at the start, but can appear anywhere
			// No special validation needed
		}
	}

	// After processing all messages, check if there are unresolved tool calls
	if len(pendingToolCalls) > 0 {
		return fmt.Errorf("message sequence ends with %d unresolved tool calls", len(pendingToolCalls))
	}

	return nil
}

// conversationState holds the history for a session.
type conversationState struct {
	history    []*OpenAIMessage
	lastAccess time.Time
}

// Config holds configuration for the OpenAI adapter.
type Config struct {
	// BaseURL is the API endpoint (e.g., "https://api.openai.com/v1" or "http://localhost:1234/v1")
	BaseURL string
	// APIKey for authentication (optional for local LLMs)
	APIKey string
	// HTTPClient for making requests (optional, will use default if nil)
	HTTPClient *http.Client
	// MaxHistoryLength is the maximum number of messages to keep in history
	MaxHistoryLength int
	// SessionTTL is how long to keep session history before cleanup
	SessionTTL time.Duration
	// MaxRetries for failed requests
	MaxRetries int
	// Timeout for HTTP requests
	Timeout time.Duration
	// Logger for session and request logging (optional)
	Logger *log.Logger
	// DebugLogging enables raw request/response JSON dumps (useful for debugging 400 errors)
	DebugLogging bool
}

type openaiModel struct {
	name             string
	baseURL          string
	apiKey           string
	httpClient       *http.Client
	maxHistoryLength int
	sessionTTL       time.Duration
	maxRetries       int
	timeout          time.Duration
	logger           *log.Logger
	debugLogging     bool

	// Conversation state management
	conversations map[string]*conversationState
	mu            sync.RWMutex

	// JSON pool for performance
	jsonPool sync.Pool

	// Cleanup goroutine control
	stopCleanup chan struct{}
	cleanupDone chan struct{}
}

// NewModel creates a new OpenAI-compatible model adapter.
//
// modelName specifies which model to use (e.g., "gpt-4", "gemma-3-12b-it").
// cfg provides the configuration including API endpoint and authentication.
func NewModel(modelName string, cfg *Config) (model.LLM, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("BaseURL must be specified")
	}

	// Set defaults
	maxHistoryLength := cfg.MaxHistoryLength
	if maxHistoryLength == 0 {
		maxHistoryLength = defaultMaxHistoryLength
	}

	sessionTTL := cfg.SessionTTL
	if sessionTTL == 0 {
		sessionTTL = defaultSessionTTL
	}

	maxRetries := cfg.MaxRetries
	if maxRetries == 0 {
		maxRetries = defaultMaxRetries
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: timeout,
		}
	}

	m := &openaiModel{
		name:             modelName,
		baseURL:          strings.TrimSuffix(cfg.BaseURL, "/"),
		apiKey:           cfg.APIKey,
		httpClient:       httpClient,
		maxHistoryLength: maxHistoryLength,
		sessionTTL:       sessionTTL,
		maxRetries:       maxRetries,
		timeout:          timeout,
		logger:           cfg.Logger,
		debugLogging:     cfg.DebugLogging,
		conversations:    make(map[string]*conversationState),
		jsonPool: sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
		stopCleanup: make(chan struct{}),
		cleanupDone: make(chan struct{}),
	}

	// Start cleanup goroutine for expired sessions
	go m.cleanupExpiredSessions()

	return m, nil
}

func (m *openaiModel) Name() string {
	return m.name
}

// GenerateContent calls the OpenAI-compatible API with conversation history management.
func (m *openaiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// cleanupExpiredSessions removes old conversation states periodically.
// This goroutine runs until Close() is called.
func (m *openaiModel) cleanupExpiredSessions() {
	defer close(m.cleanupDone)

	ticker := time.NewTicker(m.sessionTTL / 2)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCleanup:
			return
		case <-ticker.C:
			m.mu.Lock()
			now := time.Now()
			for sessionID, state := range m.conversations {
				if now.Sub(state.lastAccess) > m.sessionTTL {
					delete(m.conversations, sessionID)
				}
			}
			m.mu.Unlock()
		}
	}
}

// Close stops the background cleanup goroutine and releases resources.
// This should be called when the model is no longer needed to prevent goroutine leaks.
func (m *openaiModel) Close() error {
	close(m.stopCleanup)
	<-m.cleanupDone // Wait for cleanup goroutine to finish
	return nil
}

// getConversationHistory retrieves the conversation history for a session.
// Returns nil if no history exists or if the session has expired.
func (m *openaiModel) getConversationHistory(sessionID string) []*OpenAIMessage {
	if sessionID == "" {
		return nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	state, exists := m.conversations[sessionID]
	if !exists {
		return nil
	}

	// Check TTL
	if time.Since(state.lastAccess) > m.sessionTTL {
		return nil
	}

	// Return a copy to prevent concurrent modification
	history := make([]*OpenAIMessage, len(state.history))
	copy(history, state.history)

	return history
}

// addToHistory adds messages to the conversation history for a session.
// Validates each message before adding. If validation fails, logs error and skips the message.
func (m *openaiModel) addToHistory(sessionID string, msgs ...*OpenAIMessage) {
	if sessionID == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	state, exists := m.conversations[sessionID]
	if !exists {
		state = &conversationState{
			history: make([]*OpenAIMessage, 0, m.maxHistoryLength),
		}
		m.conversations[sessionID] = state
	}

	// Validate and add messages
	for _, msg := range msgs {
		if err := validateMessage(msg); err != nil {
			// Log validation error if logger is available
			if m.logger != nil {
				m.logger.Printf("WARNING: Invalid message skipped: %v", err)
			}
			// Skip invalid message
			continue
		}
		state.history = append(state.history, msg)
	}

	state.lastAccess = time.Now()

	// Trim if exceeds max length (keep system message if present)
	if len(state.history) > m.maxHistoryLength {
		systemMsg := []*OpenAIMessage{}
		if len(state.history) > 0 && state.history[0].Role == "system" {
			systemMsg = append(systemMsg, state.history[0])
		}

		// Keep the most recent messages
		startIdx := len(state.history) - m.maxHistoryLength + len(systemMsg)
		state.history = append(systemMsg, state.history[startIdx:]...)
	}
}

// clearHistory removes all conversation history for a session.
func (m *openaiModel) clearHistory(sessionID string) {
	if sessionID == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.conversations, sessionID)
}

// generate calls the model synchronously.
func (m *openaiModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	// Convert genai.Content to OpenAI messages
	messages, err := m.convertToOpenAIMessages(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Validate message sequence before sending to API
	if err := validateMessageSequence(messages); err != nil {
		return nil, fmt.Errorf("invalid message sequence: %w", err)
	}

	// Build OpenAI request
	chatReq := ChatCompletionRequest{
		Model:    m.name,
		Messages: messages,
		Stream:   false,
	}

	// Add configuration from req.Config
	if req.Config != nil {
		if req.Config.Temperature != nil {
			chatReq.Temperature = req.Config.Temperature
		}
		if req.Config.TopP != nil {
			chatReq.TopP = req.Config.TopP
		}
		if req.Config.MaxOutputTokens > 0 {
			tokens := req.Config.MaxOutputTokens
			chatReq.MaxTokens = &tokens
		}
		if len(req.Config.StopSequences) > 0 {
			chatReq.Stop = req.Config.StopSequences
		}
		if req.Config.PresencePenalty != nil {
			chatReq.PresencePenalty = req.Config.PresencePenalty
		}
		if req.Config.FrequencyPenalty != nil {
			chatReq.FrequencyPenalty = req.Config.FrequencyPenalty
		}
		if req.Config.Seed != nil {
			chatReq.Seed = req.Config.Seed
		}
		if req.Config.CandidateCount > 0 {
			chatReq.N = req.Config.CandidateCount
		}
		// Logprobs support
		if req.Config.ResponseLogprobs {
			chatReq.Logprobs = true
			if req.Config.Logprobs != nil {
				chatReq.TopLogprobs = req.Config.Logprobs
			}
		}
		// Map ResponseMIMEType to OpenAI response_format
		if req.Config.ResponseMIMEType != "" {
			if req.Config.ResponseMIMEType == "application/json" {
				chatReq.ResponseFormat = &ResponseFormat{Type: "json_object"}
			}
			// "text/plain" maps to default (no response_format)
		}
		// ResponseSchema for structured outputs
		if req.Config.ResponseSchema != nil {
			schema := convertGenaiSchemaToMap(req.Config.ResponseSchema)
			chatReq.ResponseFormat = &ResponseFormat{
				Type: "json_schema",
				JSONSchema: &JSONSchemaRef{
					Name:   "response_schema",
					Schema: schema,
					Strict: true,
				},
			}
		}
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		chatReq.Tools = m.convertTools(req.Tools)
		chatReq.ToolChoice = "auto"
	}

	// Make API call
	respData, err := m.makeRequest(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(respData, &chatResp); err != nil {
		return nil, &OpenAIError{
			Type:    ErrorTypeInvalidJSON,
			Message: fmt.Sprintf("failed to parse response: %v", err),
			Details: truncateString(string(respData), 200),
		}
	}

	if len(chatResp.Choices) == 0 {
		return nil, &OpenAIError{
			Type:    ErrorTypeUnknown,
			Message: "no choices in response",
			Details: chatResp,
		}
	}

	// Get the response message and logprobs
	choice := &chatResp.Choices[0]
	responseMsg := &choice.Message

	// Add response to history (even if no tools were used)
	sessionID := extractSessionIDWithLogging(ctx, m.logger)
	if sessionID != "" {
		m.addToHistory(sessionID, responseMsg)
	}

	// Convert back to genai format (including logprobs if present)
	return m.convertToLLMResponse(responseMsg, &chatResp.Usage, choice.Logprobs)
}


// makeRequest makes an HTTP request to the OpenAI API.
func (m *openaiModel) makeRequest(ctx context.Context, req ChatCompletionRequest) ([]byte, error) {
	buf := m.jsonPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		m.jsonPool.Put(buf)
	}()

	if err := json.NewEncoder(buf).Encode(req); err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}

	url := m.baseURL + "/chat/completions"

	var lastErr error
	initialBackoff := 1 * time.Second
	maxBackoff := 30 * time.Second

	// Store request body bytes for retries
	reqBody := buf.Bytes()

	// Debug logging: dump raw request
	if m.debugLogging && m.logger != nil {
		var prettyReq bytes.Buffer
		if err := json.Indent(&prettyReq, reqBody, "", "  "); err == nil {
			m.logger.Printf("=== RAW REQUEST to %s ===\n%s", url, prettyReq.String())
		} else {
			m.logger.Printf("=== RAW REQUEST to %s ===\n%s", url, string(reqBody))
		}
	}

	for attempt := 0; attempt < m.maxRetries; attempt++ {
		// Create a new request for each attempt (required for retries)
		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if m.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)
		}

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			lastErr = &OpenAIError{
				Type:    ErrorTypeNetwork,
				Message: fmt.Sprintf("network error: %v", err),
			}

			// Retry with backoff on network errors
			if attempt < m.maxRetries-1 {
				backoff := calculateBackoff(attempt, initialBackoff, maxBackoff, 2.0)
				if m.logger != nil {
					m.logger.Printf("Network error, retrying in %v (attempt %d/%d): %v", backoff, attempt+1, m.maxRetries, err)
				}
				time.Sleep(backoff)
			}
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if err != nil {
			lastErr = &OpenAIError{
				Type:    ErrorTypeNetwork,
				Message: fmt.Sprintf("failed to read response: %v", err),
			}
			continue
		}

		// Debug logging: dump raw response
		if m.debugLogging && m.logger != nil {
			var prettyResp bytes.Buffer
			if err := json.Indent(&prettyResp, body, "", "  "); err == nil {
				m.logger.Printf("=== RAW RESPONSE (status %d) ===\n%s", resp.StatusCode, prettyResp.String())
			} else {
				m.logger.Printf("=== RAW RESPONSE (status %d) ===\n%s", resp.StatusCode, string(body))
			}
		}

		// Handle success
		if resp.StatusCode == http.StatusOK {
			return body, nil
		}

		// Handle rate limiting
		if resp.StatusCode == http.StatusTooManyRequests {
			lastErr = &OpenAIError{
				Type:    ErrorTypeRateLimit,
				Message: "rate limit exceeded",
				Code:    "rate_limit_exceeded",
				Details: string(body),
			}

			// Check for retry-after header
			if attempt < m.maxRetries-1 {
				retryAfter := extractRetryAfter(resp)
				if retryAfter == 0 {
					retryAfter = calculateBackoff(attempt, initialBackoff, maxBackoff, 2.0)
				}

				if m.logger != nil {
					m.logger.Printf("Rate limit hit, retrying in %v (attempt %d/%d)", retryAfter, attempt+1, m.maxRetries)
				}

				// Wait before retrying
				select {
				case <-time.After(retryAfter):
					continue
				case <-ctx.Done():
					return nil, &OpenAIError{
						Type:    ErrorTypeTimeout,
						Message: "request cancelled during rate limit backoff",
					}
				}
			}
			continue
		}

		// Handle other HTTP errors
		// Try to parse OpenAI error format
		var apiError struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			} `json:"error"`
		}

		if json.Unmarshal(body, &apiError) == nil && apiError.Error.Message != "" {
			// Map OpenAI error codes to our error types
			errorType := ErrorTypeUnknown
			switch apiError.Error.Code {
			case "context_length_exceeded":
				errorType = ErrorTypeValidation
			case "invalid_request_error":
				errorType = ErrorTypeValidation
			case "authentication_error":
				errorType = ErrorTypeValidation
			case "rate_limit_exceeded":
				errorType = ErrorTypeRateLimit
			}

			lastErr = &OpenAIError{
				Type:    errorType,
				Message: apiError.Error.Message,
				Code:    apiError.Error.Code,
				Details: map[string]any{
					"status": resp.StatusCode,
					"type":   apiError.Error.Type,
				},
			}
		} else {
			// Fallback to generic HTTP error
			lastErr = &HTTPError{
				StatusCode: resp.StatusCode,
				Status:     resp.Status,
				Body:       string(body),
			}
		}

		// Don't retry 4xx errors (except 429) - they're client errors
		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			break
		}

		// Retry 5xx errors with backoff
		if attempt < m.maxRetries-1 {
			backoff := calculateBackoff(attempt, initialBackoff, maxBackoff, 2.0)
			if m.logger != nil {
				m.logger.Printf("Server error %d, retrying in %v (attempt %d/%d)", resp.StatusCode, backoff, attempt+1, m.maxRetries)
			}
			time.Sleep(backoff)
		}
	}

	return nil, fmt.Errorf("request failed after %d retries: %w", m.maxRetries, lastErr)
}
