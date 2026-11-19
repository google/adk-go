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

package retryandreflect

import (
	"errors"
	"fmt"
	"slices"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	defaultModelPluginName = "reflect_retry_model_plugin"
	// reservedToolName names the tool the model plugin answers a model
	// failure with. Its reflection calls carry ids that start with it.
	reservedToolName = "adk_handle_model_error"
	// reservedToolDescription is adk-python's docstring of the tool.
	reservedToolDescription = "A tool that triggers reflection. Reserved for internal framework use only. Do not call directly."
	reservedCallErrorType   = "RESERVED_TOOL_CALL"
	reservedCallDetails     = "Model attempted to call reserved tool " + reservedToolName + " directly. This tool is reserved for framework use only. Do not call it."
	reservedCallExhausted   = "Model attempted to call reserved tool and retry limit was exceeded."
	// skipThoughtSignatureValidator is the thought signature of a function
	// call part the model did not produce; Gemini then skips validating it.
	// The bytes are adk-python's (utils/content_utils.py).
	skipThoughtSignatureValidator = "skip_thought_signature_validator"
	// modelReflectionGuidance is adk-python's guidance text, byte for byte,
	// with the retry count and the retry budget to fill in.
	modelReflectionGuidance = "\nThe call to the model failed.\n\n**Reflection Guidance:**\n- This is retry attempt **%d** of **%d**\n- Analyze the error and the arguments you provided. Do not repeat the exact same call.\n\nFormulate a new plan based on your analysis and try a corrected or different approach.\n    "
)

// ErrModelRetryLimitExceeded is wrapped by the error the model plugin
// returns when a model fails more often than its retry budget allows.
var ErrModelRetryLimitExceeded = errors.New("retry limit has been exceeded")

// ModelOption configures NewModel.
type ModelOption func(*modelPlugin)

// WithModelPluginName sets the plugin's name, which the runner uses to tell
// plugins apart and which the error text of a failing callback names. The
// default is "reflect_retry_model_plugin". Any string is accepted. A runner
// refuses two plugins with one name, so two model plugins with different
// settings on one runner need different names.
func WithModelPluginName(name string) ModelOption {
	return func(p *modelPlugin) {
		p.name = name
	}
}

// WithModelMaxRetries sets the retry budget: the number of consecutive
// failures of a model that the plugin answers with a reflection call. The
// failure after them exhausts the budget. The default is 3. 0 makes the first
// failure exhaust the budget. NewModel fails for a negative value.
func WithModelMaxRetries(maxRetries int) ModelOption {
	return func(p *modelPlugin) {
		p.maxRetries = maxRetries
	}
}

// WithModelErrorIfRetryExceeded sets what the plugin does with a failure that
// exhausts the retry budget. With true, the default, the after-model callback
// returns an error wrapping [ErrModelRetryLimitExceeded], which ends the run
// with that error. With false, the callback returns no error: a model error
// stays the step's response and becomes the agent's event, and a call of the
// reserved tool is replaced by an error response with error code
// "RESERVED_TOOL_CALL" and no content, so the model's calls are not run.
//
// The default differs from [WithErrorIfRetryExceeded] of the tool plugin,
// which is false.
func WithModelErrorIfRetryExceeded(errorIfRetryExceeded bool) ModelOption {
	return func(p *modelPlugin) {
		p.errorIfRetryExceeded = errorIfRetryExceeded
	}
}

// WithModelTrackingScope sets which failures share a count: [Invocation], the
// default, counts the failures of each invocation apart; [Global] counts the
// failures of all invocations of the plugin together. Either way each model
// has its own count. NewModel fails for any other value.
func WithModelTrackingScope(scope TrackingScope) ModelOption {
	return func(p *modelPlugin) {
		p.scope = scope
	}
}

// WithModelErrorFinishReasons sets the finish reasons that make a model
// response with an error code a model error, replacing the default list
// [genai.FinishReasonMalformedFunctionCall]. Called with no reasons, it makes
// no response a model error. Reasons the genai package does not list are
// accepted. NewModel fails for an empty reason. The reasons are copied.
func WithModelErrorFinishReasons(reasons ...genai.FinishReason) ModelOption {
	return func(p *modelPlugin) {
		// A non-nil slice, empty or not, tells NewModel the option ran.
		p.reasons = append([]genai.FinishReason{}, reasons...)
	}
}

// NewModel returns a plugin that answers model failures with a reflection
// call and retries the model, up to a retry budget.
//
// A model response is a failure when it calls the reserved tool
// "adk_handle_model_error", or when it is a model error: its error code is
// not empty and its finish reason is one of [WithModelErrorFinishReasons].
// A call of the reserved tool is checked first and counts as a failure even
// on an error response. The plugin counts consecutive failures per model, in
// the scope of [WithModelTrackingScope]. Any other response the model ends a
// step with resets the model's count; a reflection call does not.
//
// While a failure is within the retry budget of [WithModelMaxRetries], the
// plugin replaces the response with a reflection call: one model-role part
// calling "adk_handle_model_error" with the arguments response_type,
// error_type, error_details, finish_reason and retry_count, the number of
// consecutive failures so far. The agent then runs the reserved tool, which
// answers with reflection guidance, and calls the model again. For a model
// error, error_type and finish_reason are the response's, and error_details is
// its error message, or null when the message is empty. For a call of the
// reserved tool, error_type is "RESERVED_TOOL_CALL" and finish_reason "OTHER",
// and the model's calls are not run. The plugin offers the reserved tool so
// that the agent can run it, but does not declare it to the model.
//
// The failure that exhausts the budget gives an error wrapping
// [ErrModelRetryLimitExceeded], or, with [WithModelErrorIfRetryExceeded]
// false, keeps the response as the step's response; neither resets the count.
// Either way, later plugins' and the agent's after-model callbacks do not see
// the response.
//
// The plugin reads only the response that ends a model call: it leaves the
// partial responses of a streamed call alone.
//
// Defaults: name "reflect_retry_model_plugin", 3 retries, an error past the
// budget, [Invocation] scope, and the one finish reason
// [genai.FinishReasonMalformedFunctionCall]. The tool plugin [New] does not
// return an error past its budget by default.
//
// Counts of the [Invocation] scope outlive the run, so a later run of the
// same invocation continues the count, as in adk-python.
//
// A failure counts against the model of the LLM agent whose model call the
// callback surrounds, read from the callback context's agent when the
// response arrives, as the agent's step resolves it: the agent's own model,
// else the model of its nearest LLM-agent ancestor in the run's agent tree,
// else the default model. Its name is the key, so two agents on models of one
// name share a count, and a plugin earlier in the list that rewrites the
// request's model name does not change the key. The after-model callback
// fails before any count changes when the context's agent was not built by
// llmagent.New, when its model cannot be resolved, or when the model's name
// is empty.
//
// NewModel applies the options in order; the last one for a setting wins and
// a nil option is skipped. It returns a nil plugin and an error when the
// retry budget is negative, the scope is neither [Invocation] nor [Global], or
// a finish reason is empty, checked in that order. Each call returns a plugin
// with its own counts. The plugin is safe for concurrent use; runners that
// share it share its counts.
func NewModel(opts ...ModelOption) (*plugin.Plugin, error) {
	p := &modelPlugin{
		name:                 defaultModelPluginName,
		maxRetries:           3,
		errorIfRetryExceeded: true,
		scope:                Invocation,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	if p.reasons == nil {
		p.reasons = []genai.FinishReason{genai.FinishReasonMalformedFunctionCall}
	}

	if p.maxRetries < 0 {
		return nil, fmt.Errorf("retryandreflect: WithModelMaxRetries: must not be negative, got %d", p.maxRetries)
	}
	if p.scope != Invocation && p.scope != Global {
		return nil, fmt.Errorf("retryandreflect: WithModelTrackingScope: must be %q or %q, got %q", Invocation, Global, p.scope)
	}
	if i := slices.Index(p.reasons, ""); i >= 0 {
		return nil, fmt.Errorf("retryandreflect: WithModelErrorFinishReasons: finish reason at index %d must not be empty", i)
	}
	reserved, err := functiontool.New(functiontool.Config{Name: reservedToolName, Description: reservedToolDescription}, p.reflect)
	if err != nil {
		return nil, fmt.Errorf("retryandreflect: build tool %s: %w", reservedToolName, err)
	}
	p.reservedTool = reserved

	return plugin.New(plugin.Config{
		Name:                p.name,
		BeforeModelCallback: p.beforeModel,
		AfterModelCallback:  p.afterModel,
	})
}

// modelPlugin holds the settings of a model plugin, which are fixed once
// NewModel returns, and its counts, which guard themselves.
type modelPlugin struct {
	name                 string
	maxRetries           int
	errorIfRetryExceeded bool
	scope                TrackingScope
	// reasons is nil until WithModelErrorFinishReasons runs, then non-nil,
	// also when empty.
	reasons      []genai.FinishReason
	reservedTool tool.Tool
	failures     failureTracker
}

// reflectArgs are the arguments of a reflection call. Pointer fields accept
// JSON null.
type reflectArgs struct {
	ResponseType string  `json:"response_type"`
	ErrorType    *string `json:"error_type"`
	ErrorDetails *string `json:"error_details"`
	FinishReason *string `json:"finish_reason"`
	RetryCount   int     `json:"retry_count"`
}

type reflectResult struct {
	ReflectionGuidance string `json:"reflection_guidance"`
}

// reflect is the handler of the reserved tool.
func (p *modelPlugin) reflect(_ agent.Context, args reflectArgs) (reflectResult, error) {
	return reflectResult{ReflectionGuidance: fmt.Sprintf(modelReflectionGuidance, args.RetryCount, p.maxRetries)}, nil
}

// beforeModel offers the reserved tool to the step. Writing req.Tools, not
// req.Config.Tools, lets the agent run the tool without declaring it to the
// model.
func (p *modelPlugin) beforeModel(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	req.Tools[reservedToolName] = p.reservedTool
	return nil, nil
}

// failure is what a reflection call reports about a failed response.
type failure struct {
	errorType string
	// errorDetails is nil or a string, so that the call's arguments compare
	// equal to a literal map.
	errorDetails any
	finishReason string
}

// afterModel ignores llmErr: the flow passes a non-nil one only on a path
// that has already handled the error.
func (p *modelPlugin) afterModel(ctx agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
	// The flow calls after-model callbacks for every streamed chunk, and the
	// response that ends the stream repeats a partial chunk's failure. Counting
	// partial chunks would count one failed call twice, and a partial text
	// chunk would reset the count of a call that fails. adk-python does both.
	if resp.Partial {
		return nil, nil
	}
	modelName, err := agentModelName(ctx)
	if err != nil {
		return nil, err
	}
	scopeKey := p.scopeKey(ctx)

	if hasReservedCall(resp) {
		f := failure{errorType: reservedCallErrorType, errorDetails: reservedCallDetails, finishReason: string(genai.FinishReasonOther)}
		retry, err := p.countFailure(ctx, scopeKey, modelName, f)
		if retry != nil || err != nil {
			return retry, err
		}
		return &model.LLMResponse{ErrorCode: reservedCallErrorType, ErrorMessage: reservedCallExhausted}, nil
	}

	if p.isModelError(resp) {
		f := failure{errorType: resp.ErrorCode, finishReason: string(resp.FinishReason)}
		if resp.ErrorMessage != "" {
			f.errorDetails = resp.ErrorMessage
		}
		retry, err := p.countFailure(ctx, scopeKey, modelName, f)
		if retry != nil || err != nil {
			return retry, err
		}
		// A non-nil result ends the after-model callback chain, so no later
		// callback replaces the failed response, as adk-python returns
		// llm_response.
		return resp, nil
	}

	p.failures.reset(scopeKey, modelName)
	return nil, nil
}

// countFailure adds a failure to the count of modelName in scopeKey. Within
// the retry budget it returns the reflection response. Past the budget it
// returns the exhaustion error, or nil, nil when the error option is off.
func (p *modelPlugin) countFailure(ctx agent.Context, scopeKey, modelName string, f failure) (*model.LLMResponse, error) {
	n := p.failures.increment(scopeKey, modelName)
	if n <= p.maxRetries {
		return p.reflectionResponse(ctx, n, f), nil
	}
	if p.errorIfRetryExceeded {
		// The text is adk-python's, with the real failure count where adk-python
		// gives the retry budget, and no final full stop.
		return nil, fmt.Errorf("The model has failed consecutively %d times and the %w", n, ErrModelRetryLimitExceeded) //nolint:staticcheck // ST1005: the text is adk-python's
	}
	return nil, nil
}

func (p *modelPlugin) reflectionResponse(ctx agent.Context, retryCount int, f failure) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{{
			FunctionCall: &genai.FunctionCall{
				ID:   reservedToolName + "_" + platform.NewUUID(ctx),
				Name: reservedToolName,
				Args: map[string]any{
					"response_type": reflectAndRetryResponseType,
					"error_type":    f.errorType,
					"error_details": f.errorDetails,
					"finish_reason": f.finishReason,
					"retry_count":   retryCount,
				},
			},
			ThoughtSignature: []byte(skipThoughtSignatureValidator),
		}},
	}}
}

func (p *modelPlugin) isModelError(resp *model.LLMResponse) bool {
	return resp.ErrorCode != "" && slices.Contains(p.reasons, resp.FinishReason)
}

// hasReservedCall reports whether resp calls the reserved tool.
func hasReservedCall(resp *model.LLMResponse) bool {
	return slices.ContainsFunc(resp.FunctionCalls(), func(fc *genai.FunctionCall) bool {
		return fc.Name == reservedToolName
	})
}

func (p *modelPlugin) scopeKey(ctx agent.Context) string {
	if p.scope == Global {
		return globalScopeKey
	}
	return ctx.InvocationID()
}

// agentModelName returns the name of the model of ctx's agent. ctx carries
// the run's agent tree, so the model is the one the agent's step resolved,
// also for an agent listed under two parents.
func agentModelName(ctx agent.Context) (string, error) {
	m, err := llminternal.ResolveModel(ctx, ctx.Agent())
	if err != nil {
		return "", fmt.Errorf("model of agent %q not found: %w", ctx.AgentName(), err)
	}
	if m.Name() == "" {
		return "", fmt.Errorf("model of agent %q not found", ctx.AgentName())
	}
	return m.Name(), nil
}

// failureTracker counts consecutive failures per scope key and model name.
// A scope's map exists only while it holds a count.
type failureTracker struct {
	mu     sync.Mutex
	counts map[string]map[string]int
}

// increment adds one failure and returns the new count.
func (t *failureTracker) increment(scopeKey, modelName string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.counts == nil {
		t.counts = make(map[string]map[string]int)
	}
	scope, ok := t.counts[scopeKey]
	if !ok {
		scope = make(map[string]int)
		t.counts[scopeKey] = scope
	}
	scope[modelName]++
	return scope[modelName]
}

func (t *failureTracker) reset(scopeKey, modelName string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	scope, ok := t.counts[scopeKey]
	if !ok {
		return
	}
	delete(scope, modelName)
	if len(scope) == 0 {
		delete(t.counts, scopeKey)
	}
}
