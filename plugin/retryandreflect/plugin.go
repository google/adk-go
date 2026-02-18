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
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"text/template"

	"google.golang.org/adk/plugin"
	"google.golang.org/adk/tool"

	_ "embed"
)

//go:embed reflection.md
var reflection string
var reflectionTemplate = template.Must(template.New("ReflectionTemplate").Parse(reflection))

//go:embed exceeded.md
var exceeded string
var exceededTemplate = template.Must(template.New("ExceededTemplate").Parse(exceeded))

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

// WithErrorIfRetryExceeded sets whether to return an error if the retry limit is exceeded.
// If set to true, then the original error is returned, otherwise instead of the original error,
// the plugin will return a new instruction "createToolRetryExceedMsg" telling LLM to stop using this tool for the current task.
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
func New(opts ...PluginOption) (*plugin.Plugin, error) {
	r := &retryAndReflect{
		maxRetries:            3, // A sensible default
		errorIfRetryExceeded:  false,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	for _, opt := range opts {
		opt(r)
	}

	if r.maxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be a non-negative integer")
	}

	return plugin.New(plugin.Config{
		Name:                "RetryAndReflectPlugin",
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

func (r *retryAndReflect) formatToolArgs(toolArgs map[string]any) string {
	argsBytes, err := json.MarshalIndent(toolArgs, "", "  ")
	if err != nil {
		return fmt.Sprintf("%+v", toolArgs)
	}
	return string(argsBytes)
}

// templateData represents the variables in the templates.
type templateData struct {
	ToolName     string
	ErrorDetails string
	ArgsSummary  string
	RetryCount   int
	MaxRetries   int
}

func (r *retryAndReflect) createToolReflectionResponse(tool tool.Tool, toolArgs map[string]any, toolErr error, retryCount int) map[string]any {
	argsSummary := r.formatToolArgs(toolArgs)
	errorDetails := r.formatErrorDetails(toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errorDetails,
		ArgsSummary:  argsSummary,
		RetryCount:   retryCount,
		MaxRetries:   r.maxRetries,
	}

	var buf bytes.Buffer
	err := reflectionTemplate.Execute(&buf, d)
	if err != nil {
		return nil
	}

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          fmt.Sprintf("%T", toolErr),
		"error_details":       toolErr.Error(),
		"retry_count":         retryCount,
		"reflection_guidance": strings.TrimSpace(buf.String()),
	}
}

func (r *retryAndReflect) createToolRetryExceedMsg(tool tool.Tool, toolArgs map[string]any, toolErr error) map[string]any {
	argsSummary := r.formatToolArgs(toolArgs)
	errorDetails := r.formatErrorDetails(toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errorDetails,
		ArgsSummary:  argsSummary,
	}

	var buf bytes.Buffer
	err := exceededTemplate.Execute(&buf, d)
	if err != nil {
		return nil
	}

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          fmt.Sprintf("%T", toolErr),
		"error_details":       toolErr.Error(),
		"retry_count":         r.maxRetries,
		"reflection_guidance": strings.TrimSpace(buf.String()),
	}
}
