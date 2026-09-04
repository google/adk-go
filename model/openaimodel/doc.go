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

// Package openaimodel provides a client for interacting with OpenAI's API.
//
// EXPERIMENTAL: This package is experimental and its behavior may change or be
// removed in the future.
//
// It implements the model.LLM interface, making it compatible with
// providers that expose the OpenAI Responses API surface. This package
// allows for easy integration of OpenAI's language models into applications.
//
// Every top-level field of genai.GenerateContentConfig is either translated to
// the Responses API or rejected with an error naming it. The long-standing
// rejections (TopK, StopSequences, multiple candidates, the penalties, Labels,
// SafetySettings) keep their own errors and are checked first, so an existing
// errors.Is call site is unaffected; everything else returns
// ErrUnsupportedConfigField.
//
// For a field with no equivalent at all, rejection keys on presence rather than
// value: setting the knob is itself the request. Presence is only observable
// where the zero value cannot be a setting — a pointer, slice, map or struct.
// A plain bool or string cannot tell its zero from unset, so AudioTimestamp
// false, and CachedContent or MediaResolution left empty, pass unremarked.
// A field that is translated can also be handed a value that is not — a
// negative MaxOutputTokens or CandidateCount, or Logprobs without
// ResponseLogprobs — and those are rejected too. What is never rejected is a
// setting that asks for nothing: an empty ThinkingConfig or HTTPOptions
// requests no behavior, and is satisfied by sending none.
//
// ServiceTier is translated; every tier genai names has a Responses
// equivalent, with genai's "standard" reaching OpenAI as "default".
//
// HTTPOptions is transport rather than generation, and only its Timeout
// crosses, bounding the request when positive. A non-positive one is rejected,
// since openai-go reads a zero timeout as no deadline at all and would lift the
// caller's bound instead of applying it.
//
// Headers do not cross, and that is a deliberate refusal rather than an
// omission. HTTPOptions describes a call to Gemini, so a credential in it —
// x-goog-api-key, say — would be sent verbatim to OpenAI, and an Authorization
// header would be worse than useless: openai-go treats one as an override and
// then declines to attach the configured API key, so the request would go out
// authenticated by the caller's header alone. Headers intended for OpenAI
// belong on ClientConfig.Options, which is scoped to the backend that receives
// them. The endpoint and credentials likewise come from ClientConfig, so
// BaseURL, BaseURLResourceScope and APIVersion are rejected rather than
// fighting it, as are ExtraBody and ExtrasRequestProvider, which shape a Gemini
// request body this package does not send, and RetryOptions, whose backoff has
// no faithful equivalent.
//
// ThinkingConfig is translated to the Responses API's effort-based reasoning.
// ThinkingLevel maps to the effort of the same name. A token budget, which
// Responses has no knob for, survives only as the distinction between none of
// it (zero, the none effort), some of it (positive, medium), and the model's
// own choice (-1, no effort sent). Models differ in which efforts they accept —
// gpt-5.4-nano takes none but not minimal, the o-series takes neither — so
// asking for one a model lacks draws a 400 naming reasoning.effort and listing
// what it does take. That is deliberate: quietly substituting a neighboring
// effort would bill the caller for thinking they asked not to do.
// A ThinkingConfig with nothing set asks for nothing and is accepted as such.
// IncludeThoughts asks for reasoning summaries, which arrive as parts with
// Thought set. It is off by default because summaries require a verified
// OpenAI organization, and that is enforced per model rather than per account:
// the same key that gets summaries from gpt-5.4-nano is refused by o4-mini. So
// asking for one unprompted would break reasoning on the older models for
// callers who never wanted summaries at all.
//
// Two of those differ from adk-python's OpenAI Responses support, which sends
// a summary unconditionally and maps a zero budget to minimal. Both were
// measured to fail against live models, so parity is broken deliberately here.
//
// The guarantee stops at the top level: within Tools, ToolConfig and the schema
// types, the parts this package does not read are still dropped quietly. It is
// also stricter than adk-python, which drops these settings in silence, so a
// config ported from Python may need fields removed before it is accepted.
//
// Clients construct a ClientConfig and pass it to NewModel:
//
//	ctx := context.Background()
//	cfg := &openaimodel.ClientConfig{APIKey: os.Getenv("OPENAI_API_KEY")}
//	llm, err := openaimodel.NewModel(ctx, openai.ChatModelGPT4oMini, cfg)
//	if err != nil {
//		log.Fatal(err)
//	}
package openaimodel
