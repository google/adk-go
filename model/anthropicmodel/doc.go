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

// Package anthropicmodel implements the [model.LLM] interface for Anthropic
// Claude models served by the first-party Anthropic API and by Google Cloud
// Vertex AI (Agent Platform / Model Garden).
//
// EXPERIMENTAL: This package is experimental; its API and behavior may change
// between minor releases before stabilizing.
//
// # Backends and Construction
//
// Call [NewModel] with a [context.Context], a model name, and a [ClientConfig].
// Set [ClientConfig.Backend] to select the serving backend:
//
//   - [BackendAnthropic] (the default when [ClientConfig.Backend] is empty)
//     targets the first-party Anthropic Messages API.
//   - [BackendVertex] targets Claude on Google Cloud Vertex AI.
//
// Every optional field on [ClientConfig] may be left at its zero value. When
// [ClientConfig.MaxTokens] is zero, [DefaultMaxTokens] (8192) is used unless
// overridden per request by [genai.GenerateContentConfig.MaxOutputTokens].
//
//	// First-party Anthropic API (uses ANTHROPIC_API_KEY when APIKey is empty):
//	llm, err := anthropicmodel.NewModel(ctx, "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
//		Backend: anthropicmodel.BackendAnthropic,
//	})
//
//	// Claude on Vertex AI (uses GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and ADC):
//	llm, err := anthropicmodel.NewModel(ctx, "claude-sonnet-4-5@20250929", anthropicmodel.ClientConfig{
//		Backend: anthropicmodel.BackendVertex,
//	})
//
// # Credential and Endpoint Resolution
//
// For [BackendAnthropic], credentials are resolved in the following order:
//
//  1. [ClientConfig.APIKey] (`x-api-key` header) or [ClientConfig.AuthToken]
//     (`Authorization: Bearer` header).
//  2. The `ANTHROPIC_API_KEY` environment variable.
//  3. The `ANTHROPIC_AUTH_TOKEN` environment variable.
//
// When [ClientConfig.BaseURL] is empty and no custom base URL is passed in
// [ClientConfig.RequestOptions], [BackendAnthropic] also honors the
// `ANTHROPIC_BASE_URL` environment variable when set. On [BackendAnthropic],
// when [ClientConfig.HTTPClient] is nil, the default HTTP client uses a
// 10-minute response-header timeout.
//
// For [BackendVertex], project, location (region), and credentials are resolved
// in the following order:
//
//  1. [ClientConfig.Project] and [ClientConfig.Location].
//  2. A full Vertex AI resource name passed to [NewModel]:
//     `projects/<project>/locations/<location>/publishers/anthropic/models/<model>`.
//  3. Environment variables: `GOOGLE_CLOUD_PROJECT` (falling back to
//     `CLOUD_ML_PROJECT_ID`) for project, and `GOOGLE_CLOUD_LOCATION` (falling
//     back to `CLOUD_ML_REGION`) for location.
//  4. Credentials from [ClientConfig.Credentials], then
//     [ClientConfig.TokenSource], then Application Default Credentials (ADC)
//     with the `https://www.googleapis.com/auth/cloud-platform` scope.
//
// On [BackendVertex], the model may be addressed as a bare model ID
// (`claude-*`), a publisher-prefixed ID (`anthropic/claude-*`), or a full
// Vertex AI resource name
// (`projects/<project>/locations/<location>/publishers/anthropic/models/<model>`).
// If a full resource name in [NewModel] or [model.LLMRequest.Model] specifies a
// project or location that does not match the configured project and location,
// the call fails with [ErrInvalidConfig] before making any network request.
//
// # Custom BaseURL Credential Guard
//
// When [ClientConfig.BaseURL] is set or `option.WithBaseURL` is passed in
// [ClientConfig.RequestOptions], credentials discovered automatically from
// environment variables (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`) or
// Application Default Credentials are never sent to that endpoint. Callers
// configuring a custom base URL must supply an explicit credential on
// [ClientConfig] (`APIKey` or `AuthToken` for [BackendAnthropic]; `Credentials`
// or `TokenSource` for [BackendVertex]); otherwise [NewModel] returns an error
// wrapping [ErrInvalidConfig].
//
// # GenerateContentConfig Support
//
// Every top-level field of [genai.GenerateContentConfig] is either honored,
// rejected with an error wrapping [ErrInvalidConfig] that names the field, or
// ignored by design:
//
//	| Field                        | Disposition |
//	|------------------------------|-------------|
//	| SystemInstruction            | Honored (text parts only; non-text parts rejected) |
//	| Temperature                  | Honored (mutually exclusive with TopP; omitted with a warning when thinking or Effort is active) |
//	| TopP                         | Honored (mutually exclusive with Temperature; omitted with a warning when thinking or Effort is active) |
//	| TopK                         | Honored (must be >= 0; omitted with a warning when thinking or Effort is active) |
//	| CandidateCount               | Honored when 0 or 1; rejected when < 0 or > 1 |
//	| MaxOutputTokens              | Honored when > 0; defaults to ClientConfig.MaxTokens (8192) |
//	| StopSequences                | Honored |
//	| ResponseModalities           | Honored when empty or ["TEXT"]; rejected otherwise |
//	| ResponseMIMEType             | Honored when "", "text/plain", or "application/json" (with ResponseSchema or ResponseJsonSchema); rejected otherwise |
//	| ResponseSchema               | Honored via output_config.format JSON schema (mutually exclusive with ResponseJsonSchema) |
//	| ResponseJsonSchema           | Honored via output_config.format JSON schema (mutually exclusive with ResponseSchema) |
//	| Tools                        | Honored for FunctionDeclarations; non-function tools rejected |
//	| ToolConfig                   | Honored for AUTO, ANY (0 or 1 AllowedFunctionNames), NONE; VALIDATED and unsupported combinations rejected |
//	| ThinkingConfig               | Honored (ThinkingBudget 0, > 0, < 0; nil ThinkingBudget and non-empty ThinkingLevel rejected; IncludeThoughts ignored) |
//	| HTTPOptions.Timeout          | Honored when > 0; rejected when < 0 |
//	| HTTPOptions.Headers          | Ignored by design (backend-specific headers are not forwarded) |
//	| HTTPOptions.BaseURL          | Rejected (use ClientConfig.BaseURL) |
//	| HTTPOptions.BaseURLResourceScope | Rejected |
//	| HTTPOptions.APIVersion       | Rejected |
//	| HTTPOptions.ExtraBody        | Rejected |
//	| HTTPOptions.ExtrasRequestProvider | Rejected |
//	| HTTPOptions.RetryOptions     | Rejected (use ClientConfig.MaxRetries) |
//	| AutomaticContinuation        | Ignored by design |
//	| ResponseLogprobs             | Rejected |
//	| Logprobs                     | Rejected |
//	| PresencePenalty              | Rejected |
//	| FrequencyPenalty             | Rejected |
//	| Seed                         | Rejected |
//	| SafetySettings               | Rejected |
//	| CachedContent                | Rejected (use ClientConfig.PromptCache) |
//	| Labels                       | Rejected |
//	| MediaResolution              | Rejected |
//	| SpeechConfig                 | Rejected |
//	| AudioTimestamp               | Rejected |
//	| ImageConfig                  | Rejected |
//	| RoutingConfig                | Rejected |
//	| ModelSelectionConfig         | Rejected |
//	| ModelArmorConfig             | Rejected |
//	| EnableEnhancedCivicAnswers   | Rejected |
//	| ServiceTier                  | Rejected |
//	| AudioTranscriptionConfig     | Rejected |
//	| ContinuationToken            | Rejected |
//
// Claude 4.5 and later reject requests that set both `temperature` and `top_p`;
// setting both on [genai.GenerateContentConfig] fails with [ErrInvalidConfig]
// before any network call. When thinking is enabled or [ClientConfig.Effort] is
// set, `Temperature`, `TopP`, and `TopK` are omitted from the request with a
// warning log, matching `adk-python`.
//
// # Multimodal Input Support
//
// Claude accepts images and documents on user turns and inside tool results,
// and produces text responses only:
//
//	| Input                                                                 | Anthropic API | Vertex AI   |
//	|-----------------------------------------------------------------------|---------------|-------------|
//	| Images (image/jpeg, image/png, image/gif, image/webp) inline bytes    | Sent          | Sent        |
//	| PDF documents (application/pdf) inline bytes                          | Sent          | Sent        |
//	| Plain-text documents (text/plain) inline bytes                        | Sent          | Sent        |
//	| Images or PDFs by https:// URL (FileData.FileURI)                     | Sent          | Unsupported |
//	| Plain-text documents by URL, gs:// URIs, or non-https:// URI schemes  | Unsupported   | Unsupported |
//	| Audio or video in any form                                            | Unsupported   | Unsupported |
//	| Any other MIME type or media on assistant turns                       | Unsupported   | Unsupported |
//
// The provider never downloads URLs itself. Unsupported content parts are
// omitted and logged at `slog.LevelWarn` through [ClientConfig.Logger] (or
// [slog.Default] when unset) with structured metadata (`backend`, `model`,
// `role`, `part_kind`, `mime_type`, `uri_scheme`) and no user data or URLs. If
// skipping unsupported content leaves the newest message in
// [model.LLMRequest.Contents] with no content blocks to send, the call fails
// before any network request with an error wrapping [ErrNoContent] and
// [ErrUnsupportedContent] that identifies the dropped part kind, MIME type, or
// URI scheme.
//
// # Extended Thinking and Effort
//
// [genai.ThinkingConfig] is mapped to the request parameters below:
//
//	| ThinkingConfig                | Request Parameter Mapping |
//	|-------------------------------|---------------------------|
//	| nil                           | Thinking parameter omitted |
//	| ThinkingBudget == nil         | Rejected with ErrInvalidConfig naming ThinkingBudget |
//	| *ThinkingBudget == 0          | Thinking disabled ({"type": "disabled"}) |
//	| *ThinkingBudget > 0           | Manual thinking enabled ({"type": "enabled", "budget_tokens": N}); never clamped |
//	| *ThinkingBudget < 0           | Adaptive thinking ({"type": "adaptive", "display": "summarized"}) |
//	| ThinkingLevel != ""           | Rejected with ErrInvalidConfig naming ThinkingLevel |
//	| IncludeThoughts               | Ignored; thought parts are always returned when thinking is enabled |
//
// Model support for individual thinking modes varies by model generation:
// `claude-haiku-4-5`, `claude-sonnet-4-6`, and `claude-opus-4-6` accept all
// three modes (`disabled`, `enabled`, and `adaptive`), whereas newer models
// such as `claude-sonnet-5-5` and `claude-opus-5-5` accept `adaptive` thinking
// (`*ThinkingBudget < 0`) or omitted `ThinkingConfig` (and may emit thought
// parts by default when `ThinkingConfig` is nil), while rejecting `disabled`
// and manual `enabled` budgets with HTTP 400.
//
// Reasoning effort (`low`, `medium`, `high`, `xhigh`, `max`) is configured via
// [ClientConfig.Effort] (`output_config.effort`). Claude `thinking` blocks are
// returned as `genai.Part{Thought: true, Text: ..., ThoughtSignature: ...}` and
// `redacted_thinking` blocks as
// `genai.Part{Thought: true, ThoughtSignature: ...}`. When replaying assistant
// history, signed `thinking` and `redacted_thinking` blocks are placed at the
// start of the assistant message as Claude requires, while unsigned thought
// parts and content-free foreign signatures are stripped.
//
// # Prompt Caching and Token Usage Metadata
//
// Setting [ClientConfig.PromptCache] enables ephemeral prompt caching with a
// 5-minute ([PromptCacheTTL5m], the default) or 1-hour ([PromptCacheTTL1h])
// TTL. Cache control breakpoints are placed on the last tool declaration, the
// system instruction block, and the last cacheable (non-thinking) content block
// in `messages`.
//
// Token usage is reported on [model.LLMResponse.UsageMetadata]:
//
//   - `PromptTokenCount`: `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
//   - `ThoughtsTokenCount`: `min(thinking_tokens, output_tokens)`
//   - `CandidatesTokenCount`: `output_tokens - ThoughtsTokenCount`
//   - `TotalTokenCount`: `PromptTokenCount + output_tokens`
//   - `CachedContentTokenCount`: `cache_read_input_tokens`
//
// When `cache_creation_input_tokens > 0`, the count is also stored as an
// `int64` in `LLMResponse.CustomMetadata[CacheCreationInputTokensMetadataKey]`.
//
// # Finish Reasons and CustomMetadata
//
// Claude `stop_reason` values map to [genai.FinishReason] as follows:
//
//   - `end_turn`, `stop_sequence`, `tool_use`, `pause_turn` -> [genai.FinishReasonStop]
//   - `max_tokens` -> [genai.FinishReasonMaxTokens]
//   - `model_context_window_exceeded` -> [genai.FinishReasonMaxTokens]
//   - `refusal` -> [genai.FinishReasonSafety]
//   - any other value -> [genai.FinishReasonOther]
//
// When a turn ends with `max_tokens` or `model_context_window_exceeded`, any
// truncated `tool_use` content block is omitted so callers never execute a tool
// with incomplete arguments.
//
// When a stop reason has no exact 1:1 [genai.FinishReason] equivalent
// (`pause_turn`, `refusal`, `model_context_window_exceeded`, or an unknown
// stop reason), the raw string value is preserved in
// `LLMResponse.CustomMetadata[StopReasonMetadataKey]`. `LLMResponse.ErrorCode`
// is never set on a content-bearing response because of its stop reason alone.
//
// # Errors and Retries
//
// API failures return an [*APIError] that unwraps to a sentinel error for use
// with [errors.Is] and [errors.As]:
//
//   - HTTP 429 or `rate_limit_error` (including mid-stream SSE `error` events)
//     -> [ErrRateLimited]
//   - HTTP 529 or `overloaded_error` (including mid-stream SSE `error` events)
//     -> [ErrOverloaded]
//   - HTTP 401/403 or `authentication_error` / `permission_error`
//     -> [ErrAuthentication]
//   - HTTP 400 or `invalid_request_error` -> [ErrBadRequest]
//   - Other HTTP status errors -> [ErrAPIStatus]
//
// By default, the underlying Anthropic Go SDK retries transient connection
// errors and HTTP 408, 409, 429, and >= 500 (including 529) responses up to 2
// times with exponential backoff (starting at 0.5s, capped at 8s, honoring
// `Retry-After-Ms` and `Retry-After` headers). Set [ClientConfig.MaxRetries] to
// override the retry limit (for example, `0` to disable retries).
//
// # Intentional Differences from adk-python
//
//   - Unified provider with explicit backend selection: Instead of two classes
//     (`AnthropicLlm` and `Claude`), this package provides [NewModel] with
//     [ClientConfig.Backend] ([BackendAnthropic] or [BackendVertex]).
//   - Ambient credential discovery on [BackendAnthropic]: [NewModel] reads
//     `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_BASE_URL`
//     from the environment when no custom `BaseURL` is set, and does not
//     autoload on-disk CLI profiles or ambient workload-federation environment
//     variables.
//   - Provider-level `Effort` and `PromptCache`: Because `model.LLMRequest` and
//     `genai.GenerateContentConfig` are shared across providers in `adk-go`,
//     Claude's `Effort` and `PromptCacheConfig` are configured on
//     [ClientConfig] rather than via a `GenerateContentConfig` subclass or
//     `LLMRequest.CacheConfig`.
//   - Structured output via `output_config.format`: `ResponseSchema` and
//     `ResponseJsonSchema` are sent through Claude's native
//     `output_config.format` JSON schema parameter rather than converted into a
//     synthetic tool declaration. Non-object root schemas are wrapped in a
//     single-property object for `output_config.format` and unwrapped on the
//     final response text (holding back wrapped partial text deltas during
//     streaming).
//   - Strict `ThinkingLevel` rejection: Setting
//     `ThinkingConfig.ThinkingLevel` returns an error wrapping
//     [ErrInvalidConfig] naming `ThinkingLevel`, whereas `adk-python` logs a
//     warning when `thinking_level` is set on the base `GenerateContentConfig`.
//   - Strict `GenerateContentConfig` validation: Unsupported
//     `GenerateContentConfig` fields, non-function `genai.Tool` types,
//     unsupported `ToolConfig` modes, and setting both `Temperature` and `TopP`
//     fail immediately with [ErrInvalidConfig] rather than being ignored or
//     silently dropped.
//   - Streaming tool calls: Partial streaming responses (`Partial: true`) are
//     emitted only for incremental text and thinking deltas; complete
//     `FunctionCall` parts and signed `thinking` / `redacted_thinking` parts
//     are emitted once on the final aggregated response (`Partial: false`,
//     `TurnComplete: true`) to match `adk-go`'s runner contract.
//   - `FinishReason` and `CustomMetadata`: `model_context_window_exceeded` maps
//     to [genai.FinishReasonMaxTokens] (with the raw `stop_reason` stored in
//     `CustomMetadata[StopReasonMetadataKey]`), whereas `adk-python` maps
//     `model_context_window_exceeded` to `FINISH_REASON_UNSPECIFIED`. Cache-write
//     tokens are reported in `CustomMetadata[CacheCreationInputTokensMetadataKey]`
//     because Go's `genai.GenerateContentResponseUsageMetadata` struct does not
//     allow arbitrary dynamic attributes.
package anthropicmodel
