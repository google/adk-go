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

package llminternal

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// mockTool implements the toolinternal.FunctionTool interface for testing
type mockTool struct {
	name        string
	description string
	runFunc     func(ctx tool.Context, args any) (map[string]any, error)
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return m.description }
func (m *mockTool) IsLongRunning() bool { return false }

func (m *mockTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return nil
}

func (m *mockTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        m.name,
		Description: m.description,
	}
}

func (m *mockTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	if m.runFunc != nil {
		return m.runFunc(ctx, args)
	}
	return nil, nil
}

// Ensure mockTool implements toolinternal.FunctionTool
var _ toolinternal.FunctionTool = (*mockTool)(nil)

// TestCallTool_ErrorSerialization tests that errors in tool execution
// are properly serialized as strings in the FunctionResponse
func TestCallTool_ErrorSerialization(t *testing.T) {
	flow := &Flow{}

	testCases := []struct {
		name           string
		toolError      error
		expectedError  string
		expectedPrefix string
	}{
		{
			name:           "fmt.Errorf error",
			toolError:      fmt.Errorf("something went wrong with value %d", 42),
			expectedError:  "tool \"test-tool\" failed: something went wrong with value 42",
			expectedPrefix: "tool \"test-tool\" failed:",
		},
		{
			name:           "errors.New error",
			toolError:      errors.New("simple error message"),
			expectedError:  "tool \"test-tool\" failed: simple error message",
			expectedPrefix: "tool \"test-tool\" failed:",
		},
		{
			name:           "wrapped error",
			toolError:      fmt.Errorf("outer error: %w", errors.New("inner error")),
			expectedError:  "tool \"test-tool\" failed: outer error: inner error",
			expectedPrefix: "tool \"test-tool\" failed:",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a mock tool that returns the test error
			mockTool := &mockTool{
				name:        "test-tool",
				description: "A tool for testing error serialization",
				runFunc: func(ctx tool.Context, args any) (map[string]any, error) {
					return nil, tc.toolError
				},
			}

			// Call the tool through callTool
			result := flow.callTool(mockTool, map[string]any{}, nil)

			// Check that the result contains an error field
			errorValue, exists := result["error"]
			if !exists {
				t.Fatal("Expected 'error' field in result, but it was not present")
			}

			// Check that the error field is a string (not an error object)
			errorString, ok := errorValue.(string)
			if !ok {
				t.Fatalf("Expected error to be a string, got %T", errorValue)
			}

			// Check that the error message is correct
			if errorString != tc.expectedError {
				t.Errorf("Expected error message %q, got %q", tc.expectedError, errorString)
			}

			// Test that the error can be JSON marshaled properly (this was the original issue)
			jsonData, err := json.Marshal(result)
			if err != nil {
				t.Fatalf("Failed to marshal result to JSON: %v", err)
			}

			// Parse back to verify the error is preserved
			var unmarshaled map[string]any
			if err := json.Unmarshal(jsonData, &unmarshaled); err != nil {
				t.Fatalf("Failed to unmarshal JSON: %v", err)
			}

			unmarshaledError, exists := unmarshaled["error"]
			if !exists {
				t.Fatal("Error field lost during JSON marshaling/unmarshaling")
			}

			unmarshaledErrorString, ok := unmarshaledError.(string)
			if !ok {
				t.Fatalf("Error field is not a string after JSON roundtrip, got %T", unmarshaledError)
			}

			if unmarshaledErrorString != tc.expectedError {
				t.Errorf("Error message changed during JSON roundtrip: expected %q, got %q", tc.expectedError, unmarshaledErrorString)
			}
		})
	}
}

// TestCallTool_BeforeToolCallbackError tests error serialization for BeforeToolCallback errors
func TestCallTool_BeforeToolCallbackError(t *testing.T) {
	flow := &Flow{
		BeforeToolCallbacks: []BeforeToolCallback{
			func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error) {
				return nil, fmt.Errorf("before tool callback failed")
			},
		},
	}

	mockTool := &mockTool{
		name:        "test-tool",
		description: "A tool for testing",
	}

	result := flow.callTool(mockTool, map[string]any{}, nil)

	errorValue, exists := result["error"]
	if !exists {
		t.Fatal("Expected 'error' field in result")
	}

	errorString, ok := errorValue.(string)
	if !ok {
		t.Fatalf("Expected error to be a string, got %T", errorValue)
	}

	expectedError := "BeforeToolCallback failed: failed to execute callback: before tool callback failed"
	if errorString != expectedError {
		t.Errorf("Expected error message %q, got %q", expectedError, errorString)
	}
}

// TestCallTool_AfterToolCallbackError tests error serialization for AfterToolCallback errors  
func TestCallTool_AfterToolCallbackError(t *testing.T) {
	flow := &Flow{
		AfterToolCallbacks: []AfterToolCallback{
			func(ctx tool.Context, tool tool.Tool, args map[string]any, result map[string]any, err error) (map[string]any, error) {
				return nil, fmt.Errorf("after tool callback failed")
			},
		},
	}

	mockTool := &mockTool{
		name:        "test-tool",
		description: "A tool for testing",
		runFunc: func(ctx tool.Context, args any) (map[string]any, error) {
			return map[string]any{"success": true}, nil
		},
	}

	result := flow.callTool(mockTool, map[string]any{}, nil)

	errorValue, exists := result["error"]
	if !exists {
		t.Fatal("Expected 'error' field in result")
	}

	errorString, ok := errorValue.(string)
	if !ok {
		t.Fatalf("Expected error to be a string, got %T", errorValue)
	}

	expectedError := "AfterToolCallback failed: failed to execute callback: after tool callback failed"
	if errorString != expectedError {
		t.Errorf("Expected error message %q, got %q", expectedError, errorString)
	}
}