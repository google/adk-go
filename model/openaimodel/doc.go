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
// value: setting the knob is itself the request. The exception is
// AudioTimestamp, a plain bool whose false cannot be told from unset, so it
// alone still passes unremarked. A field that is translated can also be handed
// a value that is not — a negative MaxOutputTokens or CandidateCount, or
// Logprobs without ResponseLogprobs — and those are rejected too.
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
// IncludeThoughts asks for reasoning summaries, which arrive as parts with
// Thought set; it is off by default because summaries require a verified
// OpenAI organization.
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
