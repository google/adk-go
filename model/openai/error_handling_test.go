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
	"net/http"
	"strings"
	"testing"
	"time"
)

// Test JSON argument sanitization
func TestSanitizeJSONArgs(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		shouldErr bool
		errType   ErrorType
	}{
		{
			name:      "valid empty args",
			input:     "{}",
			shouldErr: false,
		},
		{
			name:      "valid simple args",
			input:     `{"location":"London","temp":20}`,
			shouldErr: false,
		},
		{
			name:      "empty string (valid)",
			input:     "",
			shouldErr: false,
		},
		{
			name:      "invalid JSON",
			input:     `{"invalid": }`,
			shouldErr: true,
			errType:   ErrorTypeInvalidJSON,
		},
		{
			name:      "too large JSON",
			input:     strings.Repeat("x", maxJSONArgSize+1),
			shouldErr: true,
			errType:   ErrorTypeValidation,
		},
		{
			name:      "valid nested JSON",
			input:     `{"outer":{"inner":{"value":123}}}`,
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := sanitizeJSONArgs(tt.input)

			if tt.shouldErr {
				if err == nil {
					t.Errorf("Expected error, got nil")
					return
				}
				if oaiErr, ok := err.(*OpenAIError); ok {
					if oaiErr.Type != tt.errType {
						t.Errorf("Expected error type %s, got %s", tt.errType, oaiErr.Type)
					}
				} else {
					t.Errorf("Expected OpenAIError, got %T", err)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
				if result == "" && tt.input != "" {
					t.Error("Result should not be empty for valid input")
				}
			}
		})
	}
}

// Test JSON depth checking
func TestCheckJSONDepth(t *testing.T) {
	// Create deeply nested JSON structure
	var buildNested func(depth int) map[string]any
	buildNested = func(depth int) map[string]any {
		if depth >= maxJSONDepth+5 {
			return map[string]any{"value": depth}
		}
		return map[string]any{
			"level":  depth,
			"nested": buildNested(depth + 1),
		}
	}

	deepStruct := buildNested(0)
	deepJSON, _ := json.Marshal(deepStruct)

	_, err := sanitizeJSONArgs(string(deepJSON))
	if err == nil {
		t.Error("Expected error for too deep JSON")
		return
	}

	if oaiErr, ok := err.(*OpenAIError); ok {
		if oaiErr.Type != ErrorTypeValidation {
			t.Errorf("Expected validation error, got %s", oaiErr.Type)
		}
		if !strings.Contains(oaiErr.Message, "depth") {
			t.Errorf("Error message should mention depth: %s", oaiErr.Message)
		}
	} else {
		t.Errorf("Expected OpenAIError, got %T: %v", err, err)
	}
}

// Test string length validation
func TestJSONStringLengthLimit(t *testing.T) {
	longString := strings.Repeat("a", maxStringLength+1)
	json := `{"text":"` + longString + `"}`

	_, err := sanitizeJSONArgs(json)
	if err == nil {
		t.Error("Expected error for too long string")
	}

	if oaiErr, ok := err.(*OpenAIError); ok {
		if oaiErr.Type != ErrorTypeValidation {
			t.Errorf("Expected validation error, got %s", oaiErr.Type)
		}
	}
}

// Test backoff calculation
func TestCalculateBackoff(t *testing.T) {
	initial := 1 * time.Second
	max := 30 * time.Second
	factor := 2.0

	tests := []struct {
		attempt int
		min     time.Duration
		max     time.Duration
	}{
		{0, 0, 2 * time.Second},                // ~1s with jitter (±20%)
		{1, 1 * time.Second, 3 * time.Second},  // ~2s with jitter
		{2, 2 * time.Second, 6 * time.Second},  // ~4s with jitter
		{10, 20 * time.Second, 36 * time.Second}, // capped at max (30s) + jitter (±6s)
	}

	for _, tt := range tests {
		backoff := calculateBackoff(tt.attempt, initial, max, factor)
		if backoff < tt.min || backoff > tt.max {
			t.Errorf("Attempt %d: backoff %v not in range [%v, %v]", tt.attempt, backoff, tt.min, tt.max)
		}
	}
}

// Test error type checking
func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		shouldRetry bool
	}{
		{
			name:        "nil error",
			err:         nil,
			shouldRetry: false,
		},
		{
			name:        "context deadline",
			err:         context.DeadlineExceeded,
			shouldRetry: false,
		},
		{
			name:        "context canceled",
			err:         context.Canceled,
			shouldRetry: false,
		},
		{
			name: "rate limit error",
			err: &OpenAIError{
				Type:    ErrorTypeRateLimit,
				Message: "rate limited",
			},
			shouldRetry: true,
		},
		{
			name: "network error",
			err: &OpenAIError{
				Type:    ErrorTypeNetwork,
				Message: "connection failed",
			},
			shouldRetry: true,
		},
		{
			name: "validation error",
			err: &OpenAIError{
				Type:    ErrorTypeValidation,
				Message: "invalid input",
			},
			shouldRetry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableError(tt.err)
			if result != tt.shouldRetry {
				t.Errorf("Expected retryable=%v, got %v for error: %v", tt.shouldRetry, result, tt.err)
			}
		})
	}
}

// Test partial tool call detection
func TestIsPartialToolCall(t *testing.T) {
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
					Name:      "test",
					Arguments: `{}`,
				},
			},
			isPartial: false,
		},
		{
			name: "missing ID",
			toolCall: &ToolCall{
				Type: "function",
				Function: FunctionCall{
					Name:      "test",
					Arguments: `{}`,
				},
			},
			isPartial: true,
		},
		{
			name: "missing type",
			toolCall: &ToolCall{
				ID: "call_123",
				Function: FunctionCall{
					Name:      "test",
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
			name: "empty arguments (valid)",
			toolCall: &ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: FunctionCall{
					Name:      "test",
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

// Test error response creation
func TestCreateErrorResponse(t *testing.T) {
	tests := []struct {
		name       string
		toolCallID string
		toolName   string
		err        error
	}{
		{
			name:       "OpenAIError",
			toolCallID: "call_123",
			toolName:   "get_weather",
			err: &OpenAIError{
				Type:    ErrorTypeToolNotFound,
				Message: "tool not found",
			},
		},
		{
			name:       "Generic error",
			toolCallID: "call_456",
			toolName:   "search",
			err:        fmt.Errorf("generic error"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := createErrorResponse(tt.toolCallID, tt.toolName, tt.err)

			if msg.Role != "tool" {
				t.Errorf("Expected role 'tool', got '%s'", msg.Role)
			}
			if msg.ToolCallID != tt.toolCallID {
				t.Errorf("Expected ToolCallID '%s', got '%s'", tt.toolCallID, msg.ToolCallID)
			}
			if msg.Content == nil {
				t.Error("Content should not be nil")
			}

			// Verify content is valid JSON with error structure
			content, ok := msg.Content.(string)
			if !ok {
				t.Errorf("Content should be string, got %T", msg.Content)
			}
			if !strings.Contains(content, "error") {
				t.Error("Content should contain 'error' key")
			}
		})
	}
}

// Test truncateString
func TestTruncateString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected int // expected result length
	}{
		{
			name:     "short string",
			input:    "hello",
			maxLen:   10,
			expected: 5,
		},
		{
			name:     "exact length",
			input:    "hello",
			maxLen:   5,
			expected: 5,
		},
		{
			name:     "too long",
			input:    "hello world",
			maxLen:   5,
			expected: 8, // 5 + "..."
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncateString(tt.input, tt.maxLen)
			if len(result) != tt.expected {
				t.Errorf("Expected length %d, got %d: %s", tt.expected, len(result), result)
			}
			if len(tt.input) > tt.maxLen && !strings.HasSuffix(result, "...") {
				t.Error("Truncated string should end with '...'")
			}
		})
	}
}

// Test error helper functions
func TestErrorHelpers(t *testing.T) {
	rateLimitErr := &OpenAIError{Type: ErrorTypeRateLimit}
	timeoutErr := &OpenAIError{Type: ErrorTypeTimeout}
	validationErr := &OpenAIError{Type: ErrorTypeValidation}
	httpRateLimitErr := &HTTPError{StatusCode: http.StatusTooManyRequests}

	// IsRateLimitError
	if !IsRateLimitError(rateLimitErr) {
		t.Error("Should detect rate limit error")
	}
	if !IsRateLimitError(httpRateLimitErr) {
		t.Error("Should detect HTTP 429 as rate limit")
	}
	if IsRateLimitError(timeoutErr) {
		t.Error("Should not detect timeout as rate limit")
	}

	// IsTimeoutError
	if !IsTimeoutError(timeoutErr) {
		t.Error("Should detect timeout error")
	}
	if !IsTimeoutError(context.DeadlineExceeded) {
		t.Error("Should detect context.DeadlineExceeded as timeout")
	}
	if IsTimeoutError(validationErr) {
		t.Error("Should not detect validation as timeout")
	}

	// IsValidationError
	if !IsValidationError(validationErr) {
		t.Error("Should detect validation error")
	}
	if IsValidationError(timeoutErr) {
		t.Error("Should not detect timeout as validation")
	}
}

// Test withTimeout
func TestWithTimeout(t *testing.T) {
	// Context without deadline
	ctx1 := context.Background()
	ctx1Timeout, cancel1 := withTimeout(ctx1, 5*time.Second)
	defer cancel1()

	deadline1, ok1 := ctx1Timeout.Deadline()
	if !ok1 {
		t.Error("Should have deadline after withTimeout")
	}
	if time.Until(deadline1) > 6*time.Second {
		t.Error("Deadline should be ~5 seconds from now")
	}

	// Context with existing sooner deadline
	ctx2, cancel2 := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel2()

	ctx2Timeout, cancel2Timeout := withTimeout(ctx2, 10*time.Second)
	defer cancel2Timeout()

	deadline2, _ := ctx2Timeout.Deadline()
	if time.Until(deadline2) > 2*time.Second {
		t.Error("Should keep existing sooner deadline")
	}
}

// Test OpenAIError formatting
func TestOpenAIError_Error(t *testing.T) {
	tests := []struct {
		name     string
		err      *OpenAIError
		contains []string
	}{
		{
			name: "with code",
			err: &OpenAIError{
				Type:    ErrorTypeRateLimit,
				Message: "rate limited",
				Code:    "429",
			},
			contains: []string{"rate_limit", "429", "rate limited"},
		},
		{
			name: "without code",
			err: &OpenAIError{
				Type:    ErrorTypeTimeout,
				Message: "request timeout",
			},
			contains: []string{"timeout", "request timeout"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errStr := tt.err.Error()
			for _, substr := range tt.contains {
				if !strings.Contains(errStr, substr) {
					t.Errorf("Error string should contain '%s': %s", substr, errStr)
				}
			}
		})
	}
}

// Test HTTPError formatting
func TestHTTPError_Error(t *testing.T) {
	err := &HTTPError{
		StatusCode: 500,
		Status:     "Internal Server Error",
		Body:       "server error details",
	}

	errStr := err.Error()
	if !strings.Contains(errStr, "500") {
		t.Errorf("Error should contain status code: %s", errStr)
	}
}
