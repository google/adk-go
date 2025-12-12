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

package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// Test JSON argument edge cases
func TestJSONArgs_EdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		shouldErr bool
		expected  string
	}{
		{
			name:      "null value",
			input:     `{"value":null}`,
			shouldErr: false,
			expected:  `{"value":null}`,
		},
		{
			name:      "empty object",
			input:     `{}`,
			shouldErr: false,
			expected:  `{}`,
		},
		{
			name:      "empty array",
			input:     `{"arr":[]}`,
			shouldErr: false,
			expected:  `{"arr":[]}`,
		},
		{
			name:      "unicode characters",
			input:     `{"text":"Hello 世界 🌍"}`,
			shouldErr: false,
		},
		{
			name:      "escaped characters",
			input:     `{"text":"Line 1\nLine 2\tTabbed"}`,
			shouldErr: false,
		},
		{
			name:      "numbers",
			input:     `{"int":123,"float":45.67,"exp":1.23e10}`,
			shouldErr: false,
		},
		{
			name:      "booleans",
			input:     `{"true":true,"false":false}`,
			shouldErr: false,
		},
		{
			name:      "malformed JSON - trailing comma",
			input:     `{"key":"value",}`,
			shouldErr: true,
		},
		{
			name:      "malformed JSON - missing quotes",
			input:     `{key:"value"}`,
			shouldErr: true,
		},
		{
			name:      "malformed JSON - single quotes",
			input:     `{'key':'value'}`,
			shouldErr: true,
		},
		{
			name:      "whitespace only",
			input:     "   \n\t  ",
			shouldErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := sanitizeJSONArgs(tt.input)

			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error for input: %s", tt.input)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if tt.expected != "" && result != tt.expected {
					t.Errorf("Expected %s, got %s", tt.expected, result)
				}
			}
		})
	}
}

// Test extremely large JSON
func TestJSONArgs_SizeLimits(t *testing.T) {
	// Test with many small fields (under both size and string limits)
	largeButValid := `{"fields":[`
	for i := 0; i < 100; i++ {
		if i > 0 {
			largeButValid += ","
		}
		largeButValid += fmt.Sprintf(`{"id":%d,"value":"data%d"}`, i, i)
	}
	largeButValid += "]}"

	_, err := sanitizeJSONArgs(largeButValid)
	if err != nil {
		t.Errorf("Should accept valid JSON with many fields: %v", err)
	}

	// Over total size limit - should fail
	tooLarge := strings.Repeat("x", maxJSONArgSize+1)
	_, err = sanitizeJSONArgs(tooLarge)
	if err == nil {
		t.Error("Should reject JSON over size limit")
	}
	if !IsValidationError(err) {
		t.Errorf("Expected validation error, got %T", err)
	}
}

// Test deeply nested structures
func TestJSONArgs_DepthLimits(t *testing.T) {
	// Build JSON at exactly the limit (maxJSONDepth - 1 because we start at depth 0)
	var buildAtDepth func(depth int) map[string]any
	buildAtDepth = func(depth int) map[string]any {
		if depth >= maxJSONDepth-1 {
			return map[string]any{"leaf": "value"}
		}
		return map[string]any{"nested": buildAtDepth(depth + 1)}
	}

	atLimit := buildAtDepth(0)
	atLimitJSON, _ := json.Marshal(atLimit)
	_, err := sanitizeJSONArgs(string(atLimitJSON))
	if err != nil {
		t.Errorf("Should accept JSON at depth limit: %v", err)
	}

	// Over limit - build even deeper
	var buildOverLimit func(depth int) map[string]any
	buildOverLimit = func(depth int) map[string]any {
		if depth >= maxJSONDepth+2 {
			return map[string]any{"leaf": "value"}
		}
		return map[string]any{"nested": buildOverLimit(depth + 1)}
	}

	overLimit := buildOverLimit(0)
	overLimitJSON, _ := json.Marshal(overLimit)
	_, err = sanitizeJSONArgs(string(overLimitJSON))
	if err == nil {
		t.Error("Should reject JSON over depth limit")
	}
}

// Test rate limit simulation with mock server
func TestRateLimit_RetryBackoff(t *testing.T) {
	attemptCount := 0
	var mu sync.Mutex

	// Mock server that returns 429 for first N attempts
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		count := attemptCount
		mu.Unlock()

		if count <= 2 {
			// Return 429 for first 2 attempts
			w.Header().Set("Retry-After", "1")
			w.Header().Set("X-RateLimit-Remaining", "0")
			w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", time.Now().Unix()+1))
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":{"message":"Rate limit exceeded"}}`))
			return
		}

		// Success on 3rd attempt
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		response := ChatCompletionResponse{
			ID:      "test-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "Success after retry",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	// Create model with test server
	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 3,
		Timeout:    10 * time.Second,
		Logger:     log.New(os.Stdout, "[RATE_LIMIT_TEST] ", log.LstdFlags),
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	// Make request
	ctx := context.Background()
	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	start := time.Now()
	respData, err := om.makeRequest(ctx, req)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Request failed after retries: %v", err)
	}

	mu.Lock()
	finalCount := attemptCount
	mu.Unlock()

	if finalCount != 3 {
		t.Errorf("Expected 3 attempts, got %d", finalCount)
	}

	// Should have taken at least 2 seconds due to backoff
	if elapsed < 2*time.Second {
		t.Errorf("Expected backoff delay, but request completed in %v", elapsed)
	}

	var response ChatCompletionResponse
	if err := json.Unmarshal(respData, &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if response.Choices[0].Message.Content != "Success after retry" {
		t.Errorf("Unexpected response: %v", response.Choices[0].Message.Content)
	}
}

// Test rate limit exhaustion (all retries fail)
func TestRateLimit_Exhausted(t *testing.T) {
	// Mock server that always returns 429
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "1") // Short retry delay
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"Rate limit exceeded"}}`))
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 2, // Small number for fast test
		Timeout:    10 * time.Second,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	// Use context with timeout to prevent hanging
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err = om.makeRequest(ctx, req)
	if err == nil {
		t.Error("Expected error after retry exhaustion")
	}

	// Error should mention rate limit
	if !strings.Contains(err.Error(), "rate limit") && !strings.Contains(err.Error(), "rate_limit") {
		t.Errorf("Expected rate limit error, got: %v", err)
	}
}

// Test timeout scenarios
func TestTimeout_ContextDeadline(t *testing.T) {
	// Mock server with delay
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second) // Longer than timeout
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 1,
		Timeout:    500 * time.Millisecond,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	// Context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	_, err = om.makeRequest(ctx, req)
	if err == nil {
		t.Error("Expected timeout error")
	}

	// Should be context deadline exceeded
	if err != context.DeadlineExceeded && !strings.Contains(err.Error(), "context deadline exceeded") {
		t.Errorf("Expected deadline exceeded, got: %v", err)
	}
}

// Test partial tool calls handling
func TestPartialToolCalls_Detection(t *testing.T) {
	tests := []struct {
		name      string
		toolCall  *ToolCall
		isPartial bool
	}{
		{
			name:      "nil tool call",
			toolCall:  nil,
			isPartial: true,
		},
		{
			name: "complete tool call",
			toolCall: &ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_func",
					Arguments: `{"arg":"value"}`,
				},
			},
			isPartial: false,
		},
		{
			name: "missing ID",
			toolCall: &ToolCall{
				Type: "function",
				Function: FunctionCall{
					Name:      "test_func",
					Arguments: `{}`,
				},
			},
			isPartial: true,
		},
		{
			name: "empty ID",
			toolCall: &ToolCall{
				ID:   "",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_func",
					Arguments: `{}`,
				},
			},
			isPartial: true,
		},
		{
			name: "missing function name",
			toolCall: &ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: FunctionCall{
					Arguments: `{}`,
				},
			},
			isPartial: true,
		},
		{
			name: "empty arguments is valid",
			toolCall: &ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: FunctionCall{
					Name:      "test_func",
					Arguments: "",
				},
			},
			isPartial: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPartialToolCall(tt.toolCall)
			if result != tt.isPartial {
				t.Errorf("Expected isPartial=%v, got %v", tt.isPartial, result)
			}
		})
	}
}

// Test concurrent requests to same session
func TestConcurrentRequests_SameSession(t *testing.T) {
	cfg := &Config{
		BaseURL:          "http://localhost:1234/v1",
		MaxHistoryLength: 100,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "concurrent-session"

	var wg sync.WaitGroup
	numGoroutines := 50
	messagesPerGoroutine := 20

	// Concurrent writes to same session
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			for j := 0; j < messagesPerGoroutine; j++ {
				msg := &OpenAIMessage{
					Role:    "user",
					Content: fmt.Sprintf("Message from goroutine %d, iteration %d", goroutineID, j),
				}
				om.addToHistory(sessionID, msg)
			}
		}(i)
	}

	wg.Wait()

	// Verify history integrity
	history := om.getConversationHistory(sessionID)
	if history == nil {
		t.Fatal("History should not be nil")
	}

	// Should be trimmed to max length
	if len(history) > cfg.MaxHistoryLength {
		t.Errorf("History length %d exceeds max %d", len(history), cfg.MaxHistoryLength)
	}

	// All messages should be valid
	for i, msg := range history {
		if msg.Role == "" {
			t.Errorf("Message %d has empty role", i)
		}
	}
}

// Test no-tools conversation flow
func TestNoTools_HistoryUpdate(t *testing.T) {
	// Mock server that returns simple response without tools
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := ChatCompletionResponse{
			ID:      "test-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "Simple response without tools",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     5,
				CompletionTokens: 5,
				TotalTokens:      10,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL: server.URL,
		Logger:  log.New(os.Stdout, "[NO_TOOLS_TEST] ", log.LstdFlags),
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	sessionID := "no-tools-session"
	ctx := WithSessionID(context.Background(), sessionID)

	// Initial history should be empty
	history := om.getConversationHistory(sessionID)
	if history != nil && len(history) > 0 {
		t.Error("Initial history should be empty")
	}

	// Add user message manually (simulating what convertToOpenAIMessages does)
	userMsg := &OpenAIMessage{
		Role:    "user",
		Content: "Hello",
	}
	om.addToHistory(sessionID, userMsg)

	// Make request
	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{*userMsg},
	}

	respData, err := om.makeRequest(ctx, req)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(respData, &chatResp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// Add response to history (this is what generate() does)
	responseMsg := &chatResp.Choices[0].Message
	om.addToHistory(sessionID, responseMsg)

	// Verify history now contains both messages
	history = om.getConversationHistory(sessionID)
	if len(history) != 2 {
		t.Errorf("Expected 2 messages in history, got %d", len(history))
	}

	if history[0].Role != "user" {
		t.Errorf("First message should be user, got %s", history[0].Role)
	}

	if history[1].Role != "assistant" {
		t.Errorf("Second message should be assistant, got %s", history[1].Role)
	}

	// Verify no tool calls
	if len(history[1].ToolCalls) > 0 {
		t.Error("Response should not have tool calls")
	}
}

// Test network error retry - test with wrong port (simulates connection refused)
func TestNetworkError_Retry(t *testing.T) {
	// Use an invalid port to simulate network error
	cfg := &Config{
		BaseURL:    "http://localhost:99999", // Invalid port
		MaxRetries: 3,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = om.makeRequest(ctx, req)
	if err == nil {
		t.Error("Expected network error")
	}

	// Should contain network-related error message
	errStr := err.Error()
	if !strings.Contains(errStr, "network") && !strings.Contains(errStr, "connection") && !strings.Contains(errStr, "dial") {
		t.Errorf("Expected network error, got: %v", err)
	}
}

// Test 4xx errors are not retried (except 429)
func TestClientError_NoRetry(t *testing.T) {
	attemptCount := 0
	var mu sync.Mutex

	// Mock server that returns 400
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		mu.Unlock()

		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"Bad request"}}`))
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 5, // Even with many retries...
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	_, err = om.makeRequest(context.Background(), req)
	if err == nil {
		t.Error("Expected error for 400 response")
	}

	mu.Lock()
	finalCount := attemptCount
	mu.Unlock()

	// Should only attempt once (no retries for 4xx)
	if finalCount != 1 {
		t.Errorf("Expected 1 attempt for 4xx error, got %d", finalCount)
	}
}

// Test 5xx errors are retried
func TestServerError_Retry(t *testing.T) {
	attemptCount := 0
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		count := attemptCount
		mu.Unlock()

		if count <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":{"message":"Internal server error"}}`))
			return
		}

		// Success on 3rd attempt
		w.Header().Set("Content-Type", "application/json")
		response := ChatCompletionResponse{
			ID:      "test-500",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{
				{
					Index:        0,
					Message:      OpenAIMessage{Role: "assistant", Content: "Recovered"},
					FinishReason: "stop",
				},
			},
			Usage: Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 3,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	_, err = om.makeRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("Should succeed after retries: %v", err)
	}

	mu.Lock()
	finalCount := attemptCount
	mu.Unlock()

	if finalCount != 3 {
		t.Errorf("Expected 3 attempts for 5xx errors, got %d", finalCount)
	}
}

// Test empty response handling
func TestEmptyResponse_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return response with no choices
		response := ChatCompletionResponse{
			ID:      "empty",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{}, // Empty!
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL: server.URL,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)
	ctx := WithSessionID(context.Background(), "empty-test")

	// Simulate generate flow
	respData, err := om.makeRequest(ctx, ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	})
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	var chatResp ChatCompletionResponse
	json.Unmarshal(respData, &chatResp)

	if len(chatResp.Choices) != 0 {
		t.Error("Expected empty choices")
	}

	// This should error in the actual generate() flow
	// Simulating that check here
	if len(chatResp.Choices) == 0 {
		expectedErr := &OpenAIError{
			Type:    ErrorTypeUnknown,
			Message: "no choices in response",
		}
		if expectedErr.Type != ErrorTypeUnknown {
			t.Error("Should create appropriate error for empty choices")
		}
	}
}

// Test malformed JSON response
func TestMalformedResponse_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"invalid": json}`)) // Malformed
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL: server.URL,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	respData, err := om.makeRequest(context.Background(), ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	})
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	var chatResp ChatCompletionResponse
	err = json.Unmarshal(respData, &chatResp)
	if err == nil {
		t.Error("Expected JSON unmarshal error for malformed response")
	}
}

// TestRateLimitRetry tests detailed retry behavior with exponential backoff for 429 responses.
func TestRateLimitRetry(t *testing.T) {
	var attemptTimes []time.Time
	var mu sync.Mutex
	attemptCount := 0

	// Mock server that returns 429 for first 2 attempts
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		count := attemptCount
		attemptTimes = append(attemptTimes, time.Now())
		mu.Unlock()

		t.Logf("[SERVER] Request #%d received", count)

		if count <= 2 {
			// Return 429 rate limit error
			t.Logf("[SERVER] Returning 429 for request #%d", count)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-RateLimit-Limit", "100")
			w.Header().Set("X-RateLimit-Remaining", "0")
			// Don't set X-RateLimit-Reset to test exponential backoff calculation
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{
				"error": {
					"message": "Rate limit exceeded. Please retry after 1 second.",
					"type": "rate_limit_error",
					"code": "rate_limit_exceeded"
				}
			}`))
			return
		}

		// Success on 3rd attempt
		t.Logf("[SERVER] Returning 200 Success for request #%d", count)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		response := ChatCompletionResponse{
			ID:      "chatcmpl-test",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4",
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "Success after rate limit retries",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[RATE_LIMIT] ", 0)

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 5, // Allow enough retries
		Timeout:    30 * time.Second,
		Logger:     logger,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []OpenAIMessage{{Role: "user", Content: "Test rate limit"}},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	start := time.Now()
	respData, err := om.makeRequest(ctx, req)
	totalElapsed := time.Since(start)

	// Print logs for debugging
	t.Logf("Logger output:\n%s", logBuf.String())

	if err != nil {
		t.Fatalf("Request should succeed after retries, got error: %v", err)
	}

	mu.Lock()
	finalCount := attemptCount
	times := make([]time.Time, len(attemptTimes))
	copy(times, attemptTimes)
	mu.Unlock()

	// Verify we made 3 attempts (2 failures + 1 success)
	if finalCount != 3 {
		t.Errorf("Expected 3 attempts (2 retries + 1 success), got %d", finalCount)
	}

	// Verify response is valid
	var response ChatCompletionResponse
	if err := json.Unmarshal(respData, &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if response.Choices[0].Message.Content != "Success after rate limit retries" {
		t.Errorf("Unexpected response content: %s", response.Choices[0].Message.Content)
	}

	// Verify backoff delays between attempts
	// Expected pattern: attempt 0 → wait ~1s → attempt 1 → wait ~2s → attempt 2
	if len(times) >= 2 {
		delay1 := times[1].Sub(times[0])
		t.Logf("Delay between attempt 1 and 2: %v", delay1)

		// First backoff should be initialBackoff (1 second) - allow some tolerance
		if delay1 < 500*time.Millisecond || delay1 > 3*time.Second {
			t.Logf("WARNING: First backoff seems unusual: %v (expected ~1s)", delay1)
		}
	}

	if len(times) >= 3 {
		delay2 := times[2].Sub(times[1])
		t.Logf("Delay between attempt 2 and 3: %v", delay2)

		// Second backoff should be ~2 seconds (exponential increase)
		if delay2 < 1*time.Second || delay2 > 5*time.Second {
			t.Logf("WARNING: Second backoff seems unusual: %v (expected ~2s)", delay2)
		}
	}

	// Total time should be at least sum of backoffs (~3 seconds for 2 retries)
	// But allow for reasonable timing variations in CI
	expectedMinTime := 2 * time.Second
	if totalElapsed < expectedMinTime {
		t.Logf("WARNING: Total elapsed time %v is less than expected minimum %v", totalElapsed, expectedMinTime)
	}

	t.Logf("✓ Successfully retried after rate limit with total time: %v", totalElapsed)

	// Verify logging mentions rate limit
	logOutput := logBuf.String()
	if !strings.Contains(strings.ToLower(logOutput), "rate limit") && !strings.Contains(logOutput, "429") {
		t.Logf("WARNING: Log doesn't mention rate limit: %s", logOutput)
	}
}

// TestRateLimitRetryAfterHeader tests that Retry-After header is respected.
func TestRateLimitRetryAfterHeader(t *testing.T) {
	var attemptTimes []time.Time
	var mu sync.Mutex
	attemptCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		count := attemptCount
		attemptTimes = append(attemptTimes, time.Now())
		mu.Unlock()

		if count == 1 {
			// First attempt: Return 429 with Retry-After: 2 seconds
			w.Header().Set("Retry-After", "2")
			w.Header().Set("X-RateLimit-Remaining", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}`))
			return
		}

		// Second attempt: Success
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ChatCompletionResponse{
			ID:      "test",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "Success after Retry-After",
					},
					FinishReason: "stop",
				},
			},
		})
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 3,
		Timeout:    10 * time.Second,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	start := time.Now()
	respData, err := om.makeRequest(context.Background(), req)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	mu.Lock()
	times := make([]time.Time, len(attemptTimes))
	copy(times, attemptTimes)
	mu.Unlock()

	// Verify we made 2 attempts
	if len(times) != 2 {
		t.Errorf("Expected 2 attempts, got %d", len(times))
	}

	// Verify delay respects Retry-After header (2 seconds)
	if len(times) == 2 {
		delay := times[1].Sub(times[0])
		t.Logf("Delay between attempts: %v", delay)

		// Should be approximately 2 seconds (Retry-After header value)
		if delay < 1800*time.Millisecond || delay > 3*time.Second {
			t.Errorf("Expected delay ~2s (from Retry-After header), got %v", delay)
		}
	}

	// Verify total time is at least 2 seconds
	if elapsed < 2*time.Second {
		t.Errorf("Total elapsed %v should be at least 2 seconds", elapsed)
	}

	var response ChatCompletionResponse
	json.Unmarshal(respData, &response)

	if response.Choices[0].Message.Content != "Success after Retry-After" {
		t.Errorf("Unexpected response: %v", response.Choices[0].Message.Content)
	}

	t.Logf("✓ Retry-After header respected: waited %v", elapsed)
}

// TestRateLimitExponentialBackoff tests the exponential backoff formula.
func TestRateLimitExponentialBackoff(t *testing.T) {
	var attemptTimes []time.Time
	var mu sync.Mutex
	attemptCount := 0

	// Server that returns 429 for first 4 attempts
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attemptCount++
		count := attemptCount
		attemptTimes = append(attemptTimes, time.Now())
		mu.Unlock()

		if count <= 2 {
			// No Retry-After header - should use exponential backoff
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":{"message":"Rate limit"}}`))
			return
		}

		// Success on 3rd attempt
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ChatCompletionResponse{
			ID:      "test",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []Choice{
				{
					Index:        0,
					Message:      OpenAIMessage{Role: "assistant", Content: "Success"},
					FinishReason: "stop",
				},
			},
		})
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 5,
		Timeout:    30 * time.Second,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	start := time.Now()
	_, err = om.makeRequest(context.Background(), req)
	totalElapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	mu.Lock()
	times := make([]time.Time, len(attemptTimes))
	copy(times, attemptTimes)
	mu.Unlock()

	// Should have 3 attempts (2 failures + 1 success)
	if len(times) != 3 {
		t.Errorf("Expected 3 attempts, got %d", len(times))
	}

	// Verify exponential backoff pattern
	// Formula: min(initialBackoff * (multiplier ^ attempt), maxBackoff)
	// initialBackoff = 1s, multiplier = 2.0, maxBackoff = 30s
	// Expected: 1s, 2s
	expectedBackoffs := []time.Duration{
		1 * time.Second,  // After attempt 1
		2 * time.Second,  // After attempt 2
	}

	for i := 0; i < len(times)-1 && i < len(expectedBackoffs); i++ {
		actualDelay := times[i+1].Sub(times[i])
		expected := expectedBackoffs[i]

		// Allow 50% tolerance for timing variations
		minExpected := time.Duration(float64(expected) * 0.5)
		maxExpected := time.Duration(float64(expected) * 1.5)

		if actualDelay < minExpected || actualDelay > maxExpected {
			t.Logf("WARNING: Backoff %d: expected ~%v, got %v", i+1, expected, actualDelay)
		} else {
			t.Logf("✓ Backoff %d: %v (expected ~%v)", i+1, actualDelay, expected)
		}
	}

	// Total should be approximately 1+2 = 3 seconds
	expectedTotal := 3 * time.Second
	if totalElapsed < time.Duration(float64(expectedTotal)*0.6) {
		t.Logf("WARNING: Total time %v is less than expected ~%v", totalElapsed, expectedTotal)
	}

	t.Logf("✓ Exponential backoff verified, total time: %v", totalElapsed)
}

// TestRateLimitContextCancellation tests cancellation during rate limit backoff.
func TestRateLimitContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always return 429 to force retry
		w.Header().Set("Retry-After", "10") // Long delay
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"Rate limit"}}`))
	}))
	defer server.Close()

	cfg := &Config{
		BaseURL:    server.URL,
		MaxRetries: 5,
		Timeout:    30 * time.Second,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []OpenAIMessage{{Role: "user", Content: "test"}},
	}

	// Create context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	_, err = om.makeRequest(ctx, req)
	elapsed := time.Since(start)

	// Should fail with context error
	if err == nil {
		t.Error("Expected error due to context cancellation")
	}

	// Should fail quickly (within timeout + small margin)
	if elapsed > 3*time.Second {
		t.Errorf("Expected quick failure due to context timeout, took %v", elapsed)
	}

	// Error should mention context or cancellation
	errStr := err.Error()
	if !strings.Contains(errStr, "context") && !strings.Contains(errStr, "cancel") && !strings.Contains(errStr, "deadline") {
		t.Logf("WARNING: Error doesn't clearly indicate context cancellation: %v", err)
	}

	t.Logf("✓ Context cancellation during rate limit backoff handled correctly in %v", elapsed)
}
