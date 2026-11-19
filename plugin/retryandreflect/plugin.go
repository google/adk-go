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

// Package retryandreflect provides plugins that provide self-healing,
// concurrent-safe error recovery: [New] for tool failures and [NewModel] for
// model errors.
//
// A tool can also fail without returning an error, by returning a result that
// describes the failure. A [ResultErrorExtractor], set with
// [WithResultErrorExtractor], reads the result of each call that returned
// normally and reports such failures. The plugin counts, answers and limits a
// failure it reports like an error the tool returned.
//
// [New] is the Go version of adk-python's ReflectAndRetryToolPlugin
// (src/google/adk/plugins/reflect_retry_tool_plugin.py), and [NewModel] of its
// model plugin (src/google/adk/plugins/_reflect_retry_model_plugin.py).
package retryandreflect

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"text/template"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"

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
	// panicErrorType stands in for the Go type of a recovered tool panic, which
	// names an internal package the model has no use for.
	panicErrorType = "panic"
	// resultErrorType is the error_type of a failure a ResultErrorExtractor
	// reported, as adk-python names an extracted error that is not an
	// exception.
	resultErrorType = "ToolError"
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
	// extractError is set at construction and only read afterwards; nil means
	// no result is a failure.
	extractError ResultErrorExtractor
}

// ResultErrorExtractor reports whether a tool call that returned normally
// failed.
//
// The plugin calls it from its AfterToolCallback, once for each call that
// returned without an error, with the call's tool, arguments and result.
// result is nil when the tool returned nothing. The plugin never passes it one
// of its own reflection responses. Return nil when the result is a success.
// Return a non-nil error when it is a failure; the plugin then handles the
// call as if the tool had returned that error. The failure counts even when
// the error's text is empty. The model sees error_type "ToolError" and the
// error's text as error_details, so the text must be fit for the model to read.
//
// The plugin does not lock while it calls the function, and it may call it
// from several goroutines at once, so it must be safe for concurrent use. It
// must not modify args or result. The plugin does not recover a panic in it.
type ResultErrorExtractor func(ctx agent.Context, t tool.Tool, args, result map[string]any) error

// PluginOption is an option for configuring the ReflectAndRetryToolPlugin.
type PluginOption func(*retryAndReflect)

// WithMaxRetries sets how many consecutive failures of one tool, counted
// within the tracking scope, the plugin answers with reflection guidance. The
// next failure exceeds the limit (see [WithErrorIfRetryExceeded]); a call of
// the tool that succeeds resets its count. Zero means that every failure
// exceeds the limit. The default is 3. [New] returns an error for a negative
// value.
func WithMaxRetries(maxRetries int) PluginOption {
	return func(r *retryAndReflect) {
		r.maxRetries = maxRetries
	}
}

// WithErrorIfRetryExceeded sets what the plugin does with a failure that
// exceeds the retry limit set by [WithMaxRetries].
//
// With false, the default, the plugin answers the call with a final guidance
// response that tells the model not to use the tool again for the current
// task, and the run goes on.
//
// With true, the plugin's callback returns the failure's error: the error the
// tool returned, or the error a [ResultErrorExtractor] reported. The agent's
// flow handles that error as it handles a tool error that no callback
// answered: it becomes the call's function response, {"error": "..."}, which
// the model reads, and the run goes on. The run does not end with the error,
// as it does in adk-python, whose default is true.
func WithErrorIfRetryExceeded(errorIfRetryExceeded bool) PluginOption {
	return func(r *retryAndReflect) {
		r.errorIfRetryExceeded = errorIfRetryExceeded
	}
}

// WithTrackingScope sets the scope in which the plugin counts the failures of
// each tool. The default is [Invocation].
func WithTrackingScope(scope TrackingScope) PluginOption {
	return func(r *retryAndReflect) {
		r.scope = scope
	}
}

// WithResultErrorExtractor makes the plugin treat as failures the results that
// extract reports as failures. A nil extract, the default, treats no result as
// a failure. Options are read when New runs; the last of two
// WithResultErrorExtractor options wins.
func WithResultErrorExtractor(extract ResultErrorExtractor) PluginOption {
	return func(r *retryAndReflect) {
		r.extractError = extract
	}
}

// New creates a new reflect and retry tool plugin. By default only an error
// the tool returns is a failure; [WithResultErrorExtractor] makes some results
// failures too. The other defaults are 3 retries per tool, counted per
// invocation, and the final guidance past the limit. The plugin's name is
// "RetryAndReflectPlugin", so a runner takes at most one plugin that New
// returns. New returns an error when the retry limit is negative.
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
		AfterRunCallback:    r.afterRun,
		OnRunErrorCallback:  r.onRunError,
	})
}

// MustNew creates a new reflect and retry tool plugin and panics if it fails.
func MustNew(opts ...PluginOption) *plugin.Plugin {
	p, err := New(opts...)
	if err != nil {
		panic(err)
	}
	return p
}

// resultError is a failure a ResultErrorExtractor reported for a result.
type resultError struct{ cause error }

func (e *resultError) Error() string { return e.cause.Error() }

func (e *resultError) Unwrap() error { return e.cause }

func (r *retryAndReflect) afterTool(ctx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	// A reflection response here is the one OnToolErrorCallback just produced:
	// it is neither a success that resets the count nor a result to extract
	// an error from.
	if err != nil || isReflectionResponse(result) {
		return nil, nil
	}
	if r.extractError != nil {
		if xerr := r.extractError(ctx, t, args, result); xerr != nil {
			return r.handleToolError(ctx, t, args, &resultError{xerr})
		}
	}
	r.resetFailuresForTool(ctx, t.Name())
	return nil, nil
}

// isReflectionResponse reports whether result is a response this plugin
// answered a failure with.
func isReflectionResponse(result map[string]any) bool {
	rt, ok := result["response_type"].(string)
	return ok && rt == reflectAndRetryResponseType
}

// afterRun drops the failure counts of an invocation once its run has ended.
// Nothing reads them afterwards, so keeping them would grow
// scopedFailureCounters with every invocation that saw a failure. Every scope
// but Global keys its counts by invocation ID, as scopeKey does.
// The counts of a run that paused on an unanswered long-running call are kept:
// the run that answers the call reuses the invocation ID and continues the
// same retry budget, as in adk-python.
func (r *retryAndReflect) afterRun(ictx agent.InvocationContext) {
	if r.scope == Global {
		return
	}
	invocationID := ictx.InvocationID()
	r.mu.Lock()
	_, tracked := r.scopedFailureCounters[invocationID]
	r.mu.Unlock()
	// Most runs record no failure; skip the scan of the session events for them.
	if !tracked || hasOpenLongRunningCall(ictx.Session(), invocationID) {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.scopedFailureCounters, invocationID)
}

// onRunError drops the failure counts of an invocation that a run error
// ended. The runner calls no after-run callback for such a run, so without it
// the counts would stay in scopedFailureCounters. The counts of a run paused
// on an unanswered long-running call are kept, as in afterRun.
func (r *retryAndReflect) onRunError(ictx agent.InvocationContext, _ error) {
	r.afterRun(ictx)
}

// hasOpenLongRunningCall reports whether the invocation has a long-running call
// that no user function response has answered yet. A failure response from
// this plugin also settles the call, since no answer will follow.
func hasOpenLongRunningCall(sess session.Session, invocationID string) bool {
	if sess == nil {
		return false
	}
	open := make(map[string]struct{})
	for ev := range sess.Events().All() {
		if ev == nil || ev.InvocationID != invocationID {
			continue
		}
		for _, id := range ev.LongRunningToolIDs {
			open[id] = struct{}{}
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil || part.FunctionResponse == nil || part.FunctionResponse.ID == "" {
				continue
			}
			fr := part.FunctionResponse
			if ev.Author == "user" || fr.Response["response_type"] == reflectAndRetryResponseType {
				delete(open, fr.ID)
			}
		}
	}
	return len(open) > 0
}

func (r *retryAndReflect) onToolError(ctx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
	// A tool that asks for or was refused confirmation has not failed. The
	// check stays out of handleToolError so that an extracted error, which
	// counts whatever it wraps, never meets it.
	if errors.Is(err, tool.ErrConfirmationRequired) || errors.Is(err, tool.ErrConfirmationRejected) {
		return nil, nil
	}
	return r.handleToolError(ctx, t, args, err)
}

func (r *retryAndReflect) handleToolError(ctx agent.Context, failedTool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	if r.maxRetries == 0 {
		if r.errorIfRetryExceeded {
			return nil, err
		}
		return r.createToolRetryExceedMsg(failedTool, args, err), nil
	}

	scopeKey := r.scopeKey(ctx)
	r.mu.Lock()
	defer r.mu.Unlock()

	toolFailureCounter, ok := r.scopedFailureCounters[scopeKey]
	if !ok {
		toolFailureCounter = make(map[string]int)
		r.scopedFailureCounters[scopeKey] = toolFailureCounter
	}
	currentRetries := toolFailureCounter[failedTool.Name()] + 1
	toolFailureCounter[failedTool.Name()] = currentRetries

	if currentRetries <= r.maxRetries {
		return r.createToolReflectionResponse(failedTool, args, err, currentRetries), nil
	}

	// Max Retry exceeded
	if r.errorIfRetryExceeded {
		return nil, err
	}
	return r.createToolRetryExceedMsg(failedTool, args, err), nil
}

func (r *retryAndReflect) scopeKey(ctx agent.Context) string {
	if r.scope == Global {
		return globalScopeKey
	}
	return ctx.InvocationID()
}

func (r *retryAndReflect) resetFailuresForTool(ctx agent.Context, toolName string) {
	scopeKey := r.scopeKey(ctx)

	r.mu.Lock()
	defer r.mu.Unlock()
	if scope, ok := r.scopedFailureCounters[scopeKey]; ok {
		delete(scope, toolName)
		if len(scope) == 0 {
			delete(r.scopedFailureCounters, scopeKey)
		}
	}
}

// modelError is the type and text of a failed call of the tool named toolName
// that the plugin may put in the response it answers with, which the model,
// the session and REST clients all see, and summary, the error as the
// guidance text shows it. A panic recovered from the tool's handler, at any
// wrapping depth, becomes an opaque type and a generic message, because the
// panic error's own text carries the panic value and the goroutine stack.
// The summary of an extracted error is its text alone, as adk-python shows an
// extracted error that is not an exception; every other summary is
// "<type>: <text>".
func modelError(toolName string, err error) (errType, details, summary string) {
	if text, panicked := toolinternal.ModelErrorText(toolName, err); panicked {
		return panicErrorType, text, panicErrorType + ": " + text
	}
	if _, ok := errors.AsType[*resultError](err); ok {
		return resultErrorType, err.Error(), err.Error()
	}
	errType = fmt.Sprintf("%T", err)
	return errType, err.Error(), errType + ": " + err.Error()
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
	errType, errDetails, errSummary := modelError(tool.Name(), toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errSummary,
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
		"error_type":          errType,
		"error_details":       errDetails,
		"retry_count":         retryCount,
		"reflection_guidance": strings.TrimSpace(buf.String()),
	}
}

func (r *retryAndReflect) createToolRetryExceedMsg(tool tool.Tool, toolArgs map[string]any, toolErr error) map[string]any {
	argsSummary := r.formatToolArgs(toolArgs)
	errType, errDetails, errSummary := modelError(tool.Name(), toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errSummary,
		ArgsSummary:  argsSummary,
		RetryCount:   r.maxRetries,
		MaxRetries:   r.maxRetries,
	}

	var buf bytes.Buffer
	err := exceededTemplate.Execute(&buf, d)
	if err != nil {
		return nil
	}

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          errType,
		"error_details":       errDetails,
		"retry_count":         r.maxRetries,
		"reflection_guidance": strings.TrimSpace(buf.String()),
	}
}
