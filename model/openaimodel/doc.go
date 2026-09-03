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
// SafetySettings) keep their own errors, and the rest return
// ErrUnsupportedConfigField. Rejection keys on presence rather than value, so
// setting a knob at all is enough — except for AudioTimestamp, a plain bool
// whose false is indistinguishable from unset and so still passes unremarked.
//
// ThinkingConfig is translated to the Responses API's effort-based reasoning:
// ThinkingLevel maps to the effort of the same name, and a token budget, which
// Responses has no knob for, collapses to none effort when zero and medium
// otherwise. Models differ in which efforts they accept — gpt-5.4-nano takes
// none but not minimal, the o-series takes neither — so asking for one a model
// lacks draws a 400 naming reasoning.effort and listing what it does take.
// That is deliberate: quietly substituting a neighbouring effort would bill the
// caller for thinking they asked not to do. IncludeThoughts asks for reasoning
// summaries, which arrive as parts with Thought set; it is off by default
// because summaries require a verified OpenAI organization.
//
// The guarantee stops at the top level: within Tools, ToolConfig and the schema
// types, the parts this package does not read are still dropped quietly.
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
