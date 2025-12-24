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
	"math"
	"net/http"
	"time"
)

const (
	// JSON argument limits
	maxJSONArgSize  = 1 << 20 // 1MB
	maxJSONDepth    = 32
	maxStringLength = 10000

	// Retry configuration
	defaultInitialBackoff = 1 * time.Second
	defaultMaxBackoff     = 30 * time.Second
	defaultBackoffFactor  = 2.0

	// Rate limit headers
	headerRateLimitRemaining = "X-RateLimit-Remaining"
	headerRateLimitReset     = "X-RateLimit-Reset"
	headerRetryAfter         = "Retry-After"
)

// ErrorType represents the category of error
type ErrorType string

const (
	ErrorTypeInvalidJSON    ErrorType = "invalid_json"
	ErrorTypeToolNotFound   ErrorType = "tool_not_found"
	ErrorTypeMaxIterations  ErrorType = "max_iterations"
	ErrorTypeTimeout        ErrorType = "timeout"
	ErrorTypeRateLimit      ErrorType = "rate_limit"
	ErrorTypePartialCall    ErrorType = "partial_call"
	ErrorTypeNetwork        ErrorType = "network"
	ErrorTypeValidation     ErrorType = "validation"
	ErrorTypeUnknown        ErrorType = "unknown"
)

// OpenAIError represents a structured error from the OpenAI adapter
type OpenAIError struct {
	Type    ErrorType `json:"type"`
	Message string    `json:"message"`
	Code    string    `json:"code,omitempty"`
	Details any       `json:"details,omitempty"`
}

func (e *OpenAIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("[%s:%s] %s", e.Type, e.Code, e.Message)
	}
	return fmt.Sprintf("[%s] %s", e.Type, e.Message)
}

// isRetryableError determines if an error should trigger a retry
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Check for context errors (timeout/canceled) - not retryable
	if err == context.DeadlineExceeded || err == context.Canceled {
		return false
	}

	// Check for OpenAIError
	if oaiErr, ok := err.(*OpenAIError); ok {
		switch oaiErr.Type {
		case ErrorTypeRateLimit, ErrorTypeNetwork:
			return true
		default:
			return false
		}
	}

	return false
}

// sanitizeJSONArgs validates and sanitizes JSON arguments
func sanitizeJSONArgs(argsJSON string) (string, error) {
	// Check size limit
	if len(argsJSON) > maxJSONArgSize {
		return "", &OpenAIError{
			Type:    ErrorTypeValidation,
			Message: fmt.Sprintf("JSON arguments too large: %d bytes (max: %d)", len(argsJSON), maxJSONArgSize),
		}
	}

	// Empty or whitespace-only is valid (no args)
	if argsJSON == "" || argsJSON == "{}" {
		return "{}", nil
	}

	// Parse to validate JSON structure and check depth
	var parsed any
	if err := json.Unmarshal([]byte(argsJSON), &parsed); err != nil {
		return "", &OpenAIError{
			Type:    ErrorTypeInvalidJSON,
			Message: fmt.Sprintf("invalid JSON arguments: %v", err),
			Details: map[string]any{"raw": truncateString(argsJSON, 100)},
		}
	}

	// Check depth recursively
	if err := checkJSONDepth(parsed, 0); err != nil {
		return "", err
	}

	// Re-marshal to ensure consistent formatting
	cleaned, err := json.Marshal(parsed)
	if err != nil {
		return "", &OpenAIError{
			Type:    ErrorTypeInvalidJSON,
			Message: fmt.Sprintf("failed to re-marshal JSON: %v", err),
		}
	}

	return string(cleaned), nil
}

// checkJSONDepth recursively checks the depth of JSON structures
func checkJSONDepth(v any, depth int) error {
	if depth > maxJSONDepth {
		return &OpenAIError{
			Type:    ErrorTypeValidation,
			Message: fmt.Sprintf("JSON depth exceeds maximum: %d (max: %d)", depth, maxJSONDepth),
		}
	}

	switch val := v.(type) {
	case map[string]any:
		for _, item := range val {
			if err := checkJSONDepth(item, depth+1); err != nil {
				return err
			}
		}
	case []any:
		for _, item := range val {
			if err := checkJSONDepth(item, depth+1); err != nil {
				return err
			}
		}
	case string:
		if len(val) > maxStringLength {
			return &OpenAIError{
				Type:    ErrorTypeValidation,
				Message: fmt.Sprintf("string value too long: %d chars (max: %d)", len(val), maxStringLength),
			}
		}
	}

	return nil
}

// truncateString truncates a string to maxLen characters
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// calculateBackoff calculates exponential backoff with jitter
func calculateBackoff(attempt int, initial, max time.Duration, factor float64) time.Duration {
	backoff := float64(initial) * math.Pow(factor, float64(attempt))
	if backoff > float64(max) {
		backoff = float64(max)
	}

	// Add jitter (±20%)
	jitter := backoff * 0.2 * (2*float64(time.Now().UnixNano()%100)/100 - 1)
	backoff += jitter

	if backoff < 0 {
		backoff = float64(initial)
	}

	return time.Duration(backoff)
}

// extractRetryAfter extracts retry delay from HTTP response headers
func extractRetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}

	// Check Retry-After header
	if retryAfter := resp.Header.Get(headerRetryAfter); retryAfter != "" {
		// Try as seconds
		if seconds, err := time.ParseDuration(retryAfter + "s"); err == nil {
			return seconds
		}
		// Try as HTTP date (not implemented for simplicity)
	}

	// Check X-RateLimit-Reset
	if reset := resp.Header.Get(headerRateLimitReset); reset != "" {
		// Parse unix timestamp
		var resetTime int64
		if _, err := fmt.Sscanf(reset, "%d", &resetTime); err == nil {
			now := time.Now().Unix()
			if resetTime > now {
				return time.Duration(resetTime-now) * time.Second
			}
		}
	}

	return 0
}

// createErrorResponse creates a tool response for an error
func createErrorResponse(toolCallID, toolName string, err error) *OpenAIMessage {
	var errorMsg string
	var errorType string

	if oaiErr, ok := err.(*OpenAIError); ok {
		errorType = string(oaiErr.Type)
		errorMsg = oaiErr.Message
	} else {
		errorType = string(ErrorTypeUnknown)
		errorMsg = err.Error()
	}

	errorResponse := map[string]any{
		"error": map[string]any{
			"type":    errorType,
			"message": errorMsg,
			"tool":    toolName,
		},
	}

	responseJSON, _ := json.Marshal(errorResponse)

	return &OpenAIMessage{
		Role:       "tool",
		Content:    string(responseJSON),
		ToolCallID: toolCallID,
	}
}

// isPartialToolCall checks if a tool call is incomplete/malformed
func isPartialToolCall(tc *ToolCall) bool {
	if tc == nil {
		return true
	}
	if tc.ID == "" || tc.Type == "" {
		return true
	}
	if tc.Function.Name == "" {
		return true
	}
	// Arguments can be empty for parameterless functions
	return false
}

// recoverFromPanic recovers from panic and converts to error
func recoverFromPanic() error {
	if r := recover(); r != nil {
		return &OpenAIError{
			Type:    ErrorTypeUnknown,
			Message: fmt.Sprintf("panic recovered: %v", r),
		}
	}
	return nil
}

// withTimeout wraps a context with timeout if not already set
func withTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	// Check if context already has a deadline
	if deadline, ok := ctx.Deadline(); ok {
		// Use existing deadline if it's sooner
		remaining := time.Until(deadline)
		if remaining < timeout {
			return context.WithCancel(ctx)
		}
	}

	// Add new timeout
	return context.WithTimeout(ctx, timeout)
}

// HTTPError represents an HTTP error with status code
type HTTPError struct {
	StatusCode int
	Status     string
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Status)
}

// IsRateLimitError checks if error is a rate limit error
func IsRateLimitError(err error) bool {
	if httpErr, ok := err.(*HTTPError); ok {
		return httpErr.StatusCode == http.StatusTooManyRequests
	}
	if oaiErr, ok := err.(*OpenAIError); ok {
		return oaiErr.Type == ErrorTypeRateLimit
	}
	return false
}

// IsTimeoutError checks if error is a timeout
func IsTimeoutError(err error) bool {
	if err == context.DeadlineExceeded {
		return true
	}
	if oaiErr, ok := err.(*OpenAIError); ok {
		return oaiErr.Type == ErrorTypeTimeout
	}
	return false
}

// IsValidationError checks if error is a validation error
func IsValidationError(err error) bool {
	if oaiErr, ok := err.(*OpenAIError); ok {
		return oaiErr.Type == ErrorTypeValidation
	}
	return false
}
