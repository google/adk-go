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

// Package retryandreflect provides a plugin that provides self-healing,
// concurrent-safe error recovery for tool failures.
//
// This is the Go version of the Python plugin.
// See https://github.com/google/adk-py/blob/main/google/adk/plugins/retry_and_reflect_plugin.py
package retryandreflect

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"google.golang.org/adk/plugin/plugin"
	"google.golang.org/adk/adk/tool/tool"
)

const (
	reflectAndRetryResponseType = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
	globalScopeKey              = "__global_reflect_and_retry_scope__"
)

// TrackingScope defines the lifecycle scope for tracking tool failure counts.
type TrackingScope string

const (
	// Invocation tracks failures per-invocation.
	Invocation TrackingScope = "invocation"
	// Global tracks failures globally across all turns and users.
	Global TrackingScope = "global"
)

type retryAndReflect struct {
	mu                    sync.Mutex
	maxRetries            int
	errorIfRetryExceeded  bool
	scope                 TrackingScope
	scopedFailureCounters map[string]map[string]int
}

// PluginOption is an option for configuring the ReflectAndRetryToolPlugin.
type PluginOption func(*retryAndReflect)

// WithMaxRetries sets the maximum number of retries for a tool.
func WithMaxRetries(maxRetries int) PluginOption {
	return func(r *retryAndReflect) {
		r.maxRetries = maxRetries
	}
}

// WithErrorIfRetryExceeded sets whether to throw an exception if the retry limit is exceeded.
func WithErrorIfRetryExceeded(errorIfRetryExceeded bool) PluginOption {
	return func(r *retryAndReflect) {
		r.errorIfRetryExceeded = errorIfRetryExceeded
	}
}

// WithTrackingScope sets the tracking scope for tool failures.
func WithTrackingScope(scope TrackingScope) PluginOption {
	return func(r *retryAndReflect) {
		r.scope = scope
	}
}

// New creates a new reflect and retry tool plugin.
func New(maxRetries int, errorIfRetryExceeded bool, scope TrackingScope) (*plugin.Plugin, error) {
	if maxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be a non-negative integer")
	}
	r := &retryAndReflect{
		maxRetries:            maxRetries,
		errorIfRetryExceeded:  errorIfRetryExceeded,
		scope:                 scope,
		scopedFailureCounters: make(map[string]map[string]int),
	}
	return plugin.New(plugin.Config{
		Name:                "ReflectAndRetryToolPlugin",
		AfterToolCallback:   r.afterTool,
		OnToolErrorCallback: r.onToolError,
	})
}

func (r *retryAndReflect) afterTool(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if err == nil {
		isReflectResponse := false
		if rt, ok := result["response_type"].(string); ok && rt == reflectAndRetryResponseType {
			isReflectResponse = true
		}
		// On success, reset the failure count for this specific tool within its scope.
		// But do not reset if OnToolErrorCallback just produced a reflection response.
		if !isReflectResponse {
			r.resetFailuresForTool(ctx, tool.Name())
		}
	}
	return nil, nil
}

func (r *retryAndReflect) onToolError(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	return r.handleToolError(ctx, tool, args, err)
}

func (r *retryAndReflect) handleToolError(ctx tool.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	if r.maxRetries == 0 {
		if r.errorIfRetryExceeded {
			return nil, err
		}
		return r.createToolRetryExceedMsg(tool, args, err), nil
	}

	scopeKey := r.scopeKey(ctx)
	r.mu.Lock()
	defer r.mu.Unlock()

	toolFailureCounter, ok := r.scopedFailureCounters[scopeKey]
	if !ok {
		toolFailureCounter = make(map[string]int)
		r.scopedFailureCounters[scopeKey] = toolFailureCounter
	}
	currentRetries := toolFailureCounter[tool.Name()] + 1
	toolFailureCounter[tool.Name()] = currentRetries

	if currentRetries <= r.maxRetries {
		return r.createToolReflectionResponse(tool, args, err, currentRetries), nil
	}

	// Max Retry exceeded
	if r.errorIfRetryExceeded {
		return nil, err
	}
	return r.createToolRetryExceedMsg(tool, args, err), nil
}

func (r *retryAndReflect) scopeKey(ctx tool.Context) string {
	if r.scope == Global {
		return globalScopeKey
	}
	return ctx.InvocationID()
}

func (r *retryAndReflect) resetFailuresForTool(ctx tool.Context, toolName string) {
	scopeKey := r.scopeKey(ctx)

	r.mu.Lock()
	defer r.mu.Unlock()
	if scope, ok := r.scopedFailureCounters[scopeKey]; ok {
		delete(scope, toolName)
	}
}

func (r *retryAndReflect) formatErrorDetails(err error) string {
	return fmt.Sprintf("%T: %v", err, err)
}

func (r *retryAndReflect) createToolReflectionResponse(tool tool.Tool, toolArgs map[string]any, toolErr error, retryCount int) map[string]any {
	argsBytes, err := json.MarshalIndent(toolArgs, "", "  ")
	argsSummary := string(argsBytes)
	if err != nil {
		argsSummary = fmt.Sprintf("%+v", toolArgs)
	}
	errorDetails := r.formatErrorDetails(toolErr)

	var msg strings.Builder
	fmt.Fprintf(&msg, "\nThe call to tool `%s` failed.\n\n", tool.Name())
	fmt.Fprintf(&msg, "**Error Details:**\n```\n%s\n```\n\n", errorDetails)
	fmt.Fprintf(&msg, "**Tool Arguments Used:**\n```json\n%s\n```\n\n", argsSummary)
	fmt.Fprintf(&msg, "**Reflection Guidance:**\n")
	fmt.Fprintf(&msg, "This is retry attempt **%d of %d**. Analyze the error and the arguments you provided. Do not repeat the exact same call. Consider the following before your next attempt:\n\n", retryCount, r.maxRetries)
	fmt.Fprintf(&msg, "1.  **Invalid Parameters**: Does the error suggest that one or more arguments are incorrect, badly formatted, or missing? Review the tool's schema and your arguments.\n")
	fmt.Fprintf(&msg, "2.  **State or Preconditions**: Did a previous step fail or not produce the necessary state/resource for this tool to succeed?\n")
	fmt.Fprintf(&msg, "3.  **Alternative Approach**: Is this the right tool for the job? Could another tool or a different sequence of steps achieve the goal?\n")
	fmt.Fprintf(&msg, "4.  **Simplify the Task**: Can you break the problem down into smaller, simpler steps?\n")
	fmt.Fprintf(&msg, "5.  **Wrong Function Name**: Does the error indicate the tool is not found? Please check again and only use available tools.\n\n")
	fmt.Fprintf(&msg, "Formulate a new plan based on your analysis and try a corrected or different approach.\n")

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          fmt.Sprintf("%T", toolErr),
		"error_details":       toolErr.Error(),
		"retry_count":         retryCount,
		"reflection_guidance": strings.TrimSpace(msg.String()),
	}
}

func (r *retryAndReflect) createToolRetryExceedMsg(tool tool.Tool, toolArgs map[string]any, toolErr error) map[string]any {
	argsBytes, err := json.MarshalIndent(toolArgs, "", "  ")
	argsSummary := string(argsBytes)
	if err != nil {
		argsSummary = fmt.Sprintf("%+v", toolArgs)
	}
	errorDetails := r.formatErrorDetails(toolErr)

	var msg strings.Builder
	fmt.Fprintf(&msg, "\nThe tool `%s` has failed consecutively %d times and the retry limit has been exceeded.\n\n", tool.Name(), r.maxRetries)
	fmt.Fprintf(&msg, "**Last Error:**\n```\n%s\n```\n\n", errorDetails)
	fmt.Fprintf(&msg, "**Last Arguments Used:**\n```json\n%s\n```\n\n", argsSummary)
	fmt.Fprintf(&msg, "**Final Instruction:**\n")
	fmt.Fprintf(&msg, "**Do not attempt to use the `%s` tool again for this task.** You must now try a different approach. Acknowledge the failure and devise a new strategy, potentially using other available tools or informing the user that the task cannot be completed.\n", tool.Name())

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          fmt.Sprintf("%T", toolErr),
		"error_details":       toolErr.Error(),
		"retry_count":         r.maxRetries,
		"reflection_guidance": strings.TrimSpace(msg.String()),
	}
}
