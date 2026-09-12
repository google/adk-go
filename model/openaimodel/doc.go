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
// the Responses API or rejected with an error naming it, the pre-existing errors
// being checked first so existing errors.Is call sites are unaffected. Rejection
// keys on presence, which is only observable where the zero value cannot itself
// be a setting, so a plain bool or string carrying its zero — AudioTimestamp
// false, CachedContent or MediaResolution empty — passes unremarked, as does
// HTTPOptions.Headers, ignored by design because headers addressed to another
// backend must not reach OpenAI.
//
//	Translated  Temperature, TopP, MaxOutputTokens, SystemInstruction,
//	            ResponseMIMEType, ResponseSchema, ResponseJsonSchema,
//	            ResponseLogprobs with Logprobs, Tools, ToolConfig,
//	            ThinkingConfig, ServiceTier, HTTPOptions.Timeout
//	Rejected    TopK, StopSequences, CandidateCount above one, the penalties,
//	            Labels, SafetySettings, an unsupported ResponseMIMEType, Seed,
//	            CachedContent, ResponseModalities, MediaResolution,
//	            SpeechConfig, AudioTimestamp, ImageConfig, RoutingConfig,
//	            ModelSelectionConfig, ModelArmorConfig,
//	            EnableEnhancedCivicAnswers, AudioTranscriptionConfig,
//	            HTTPOptions apart from Timeout and Headers
//	Ignored     HTTPOptions.Headers
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
