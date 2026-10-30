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

// Package classifier defines the experimental [Classifier] interface and the
// typed question and answer values shared across classifier implementations.
//
// This package is experimental; its API and behavior may change or be removed
// in a later version without notice.
//
// Unlike a [google.golang.org/adk/v2/model.LLM], a [Classifier] does not
// generate text, stream, or call tools. It takes an input and a named set of
// [Choice], [Binary], and [Score] questions, and returns a calibrated or
// estimated probability distribution over each question's possible answers.
// Dedicated decision models such as
// [google.golang.org/adk/v2/exp/classifier/typesafe.Classifier] implement
// [Classifier] directly, while [LLMClassifier] (constructed with [NewLLM])
// implements it on top of any [google.golang.org/adk/v2/model.LLM] via
// structured JSON output.
//
// Classifiers are stateless: context from earlier conversation turns must be
// included in the input passed to [Classifier.Classify], as text or as plain
// JSON values such as a slice of role and text maps. A []*genai.Content
// history is rejected with [ErrInvalidInput]; see [ValidateInput].
//
// To route a [google.golang.org/adk/v2/workflow.Workflow] graph on a
// classifier's answer to a [Choice] or [Binary] question, construct a [Node]
// with [NewNode].
package classifier
