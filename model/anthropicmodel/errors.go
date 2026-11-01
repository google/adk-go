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

package anthropicmodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
)

var (
	// ErrNoContent is returned when a request has no sendable content blocks.
	ErrNoContent = errors.New("anthropicmodel: request contains no sendable content")
	// ErrInvalidConfig is returned when ClientConfig or GenerateContentConfig is invalid.
	ErrInvalidConfig = errors.New("anthropicmodel: invalid configuration")
	// ErrUnsupportedToolType is returned when a genai.Tool contains a non-function tool.
	ErrUnsupportedToolType = errors.New("anthropicmodel: unsupported tool type")
	// ErrUnsupportedContent is returned when unsupported content causes a message to fail.
	ErrUnsupportedContent = errors.New("anthropicmodel: unsupported content")
	// ErrAuthentication is returned for authentication or permission failures (HTTP 401/403).
	ErrAuthentication = errors.New("anthropicmodel: authentication failed")
	// ErrBadRequest is returned for invalid request errors from the API (HTTP 400).
	ErrBadRequest = errors.New("anthropicmodel: bad request")
	// ErrRateLimited is returned when the API reports a rate limit (HTTP 429 or rate_limit_error).
	ErrRateLimited = errors.New("anthropicmodel: rate limited")
	// ErrOverloaded is returned when the API reports overload (HTTP 529 or overloaded_error).
	ErrOverloaded = errors.New("anthropicmodel: service overloaded")
	// ErrAPIStatus is returned for other non-2xx API responses.
	ErrAPIStatus = errors.New("anthropicmodel: unexpected API status")
)

// StatusOverloaded is the HTTP status code used by Anthropic when overloaded.
const StatusOverloaded = 529

// APIError represents a structured error returned by the Anthropic API or
// Vertex AI Claude endpoint, either as an HTTP status response or as a
// mid-stream SSE error event.
//
// Its Error method formats only status, error type, request ID, and a sanitized
// error description without echoing prompts, model output, URLs, or credentials.
type APIError struct {
	// StatusCode is the HTTP status code (for example, 400, 401, 429, 500, 529).
	StatusCode int
	// Type is the Anthropic error type (for example, "rate_limit_error" or "overloaded_error").
	Type string
	// RequestID is the request identifier returned by the server, when present.
	RequestID string
	// Message is a short, non-sensitive description of the API error.
	Message string
	// Err is the underlying sentinel error (such as ErrRateLimited or ErrOverloaded).
	Err error
}

// Error implements the error interface.
func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("anthropicmodel: api error")
	if e.StatusCode > 0 {
		fmt.Fprintf(&b, " (status %d)", e.StatusCode)
	}
	if e.Type != "" {
		fmt.Fprintf(&b, " [type=%s]", e.Type)
	}
	if e.RequestID != "" {
		fmt.Fprintf(&b, " [request_id=%s]", e.RequestID)
	}
	if e.Message != "" {
		fmt.Fprintf(&b, ": %s", e.Message)
	}
	return b.String()
}

// Unwrap returns the sentinel error for errors.Is classification.
func (e *APIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

type errorEnvelope struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id"`
	Error     struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// classifyError converts an SDK, HTTP, or streaming error into a sanitized
// error that preserves context cancellation and classifies rate-limit (429 /
// rate_limit_error) and overload (529 / overloaded_error) conditions.
func classifyError(err error) error {
	if err == nil {
		return nil
	}

	// Preserve context cancellation and deadline errors while stripping any *url.Error URL.
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}

	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr
	}

	var sdkErr *anthropic.Error
	if errors.As(err, &sdkErr) {
		return classifySDKError(sdkErr)
	}

	// Handle mid-stream SSE errors when ssestreamDecoder lacked a non-nil http.Response.Request.
	msg := err.Error()
	if idx := strings.Index(msg, "{"); idx >= 0 {
		rawJSON := strings.TrimSpace(msg[idx:])
		var env errorEnvelope
		if json.Unmarshal([]byte(rawJSON), &env) == nil && (env.Error.Type != "" || env.Type == "error") {
			return buildAPIError(0, env.Error.Type, env.RequestID, env.Error.Message)
		}
	}

	// Strip *url.Error wrapper so request URLs never leak in error strings.
	var urlErr *url.Error
	if errors.As(err, &urlErr) && urlErr.Err != nil {
		return fmt.Errorf("anthropicmodel: transport error: %w", urlErr.Err)
	}

	return fmt.Errorf("anthropicmodel: request failed: %w", err)
}

func classifySDKError(sdkErr *anthropic.Error) error {
	statusCode := sdkErr.StatusCode
	errType := string(sdkErr.Type())
	requestID := sdkErr.RequestID
	var errMsg string

	if raw := sdkErr.RawJSON(); raw != "" {
		var env errorEnvelope
		if json.Unmarshal([]byte(raw), &env) == nil {
			if errType == "" && env.Error.Type != "" {
				errType = env.Error.Type
			}
			if requestID == "" && env.RequestID != "" {
				requestID = env.RequestID
			}
			errMsg = env.Error.Message
		}
	}

	return buildAPIError(statusCode, errType, requestID, errMsg)
}

func buildAPIError(statusCode int, errType, requestID, rawMsg string) *APIError {
	errType = strings.TrimSpace(errType)
	var sentinel error

	switch {
	case statusCode == http.StatusTooManyRequests || errType == string(anthropic.ErrorTypeRateLimitError):
		sentinel = ErrRateLimited
		if statusCode == 0 || statusCode == http.StatusOK {
			statusCode = http.StatusTooManyRequests
		}
	case statusCode == StatusOverloaded || errType == string(anthropic.ErrorTypeOverloadedError):
		sentinel = ErrOverloaded
		if statusCode == 0 || statusCode == http.StatusOK {
			statusCode = StatusOverloaded
		}
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden ||
		errType == string(anthropic.ErrorTypeAuthenticationError) ||
		errType == string(anthropic.ErrorTypePermissionError):
		sentinel = ErrAuthentication
		if statusCode == 0 || statusCode == http.StatusOK {
			statusCode = http.StatusUnauthorized
		}
	case statusCode == http.StatusBadRequest || errType == string(anthropic.ErrorTypeInvalidRequestError):
		sentinel = ErrBadRequest
		if statusCode == 0 || statusCode == http.StatusOK {
			statusCode = http.StatusBadRequest
		}
	default:
		sentinel = ErrAPIStatus
		if statusCode == 0 || statusCode == http.StatusOK {
			switch errType {
			case string(anthropic.ErrorTypeNotFoundError):
				statusCode = http.StatusNotFound
			case string(anthropic.ErrorTypeTimeoutError):
				statusCode = http.StatusGatewayTimeout
			case string(anthropic.ErrorTypeBillingError):
				statusCode = http.StatusPaymentRequired
			default:
				statusCode = http.StatusInternalServerError
			}
		}
	}

	return &APIError{
		StatusCode: statusCode,
		Type:       errType,
		RequestID:  strings.TrimSpace(requestID),
		Message:    sanitizeErrorMessage(rawMsg),
		Err:        sentinel,
	}
}

// sanitizeErrorMessage removes any URLs or bearer/key tokens from an API error
// message while preserving the diagnostic reason (such as field names or budget limits).
func sanitizeErrorMessage(msg string) string {
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return ""
	}
	lower := strings.ToLower(msg)
	if strings.Contains(lower, "http://") || strings.Contains(lower, "https://") ||
		strings.Contains(lower, "gs://") || strings.Contains(lower, "sk-ant-") ||
		strings.Contains(lower, "bearer ") {
		return "api request rejected"
	}
	if len(msg) > 256 {
		msg = strings.ToValidUTF8(msg[:256], "")
	}
	return msg
}
