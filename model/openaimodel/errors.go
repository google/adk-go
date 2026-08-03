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

package openaimodel

import "errors"

var (
	// ErrModelNameRequired is returned when a model name is not provided.
	ErrModelNameRequired = errors.New("openai: model name is required")
	// ErrRequestNil is returned when the provided request is nil.
	ErrRequestNil = errors.New("openai: request is nil")
	// ErrNoContents is returned when the LLM request has no contents.
	ErrNoContents = errors.New("openai: LLM request has no contents to convert")
	// ErrFunctionCallMissingName is returned when a function call is missing a name.
	ErrFunctionCallMissingName = errors.New("openai: function call missing name")
	// ErrTopKNotSupported is returned when TopK is used, which is not supported.
	ErrTopKNotSupported = errors.New("openai: topK is not supported by the Responses API")
	// ErrStopSequencesNotSupported is returned when stop sequences are used, which is not supported.
	ErrStopSequencesNotSupported = errors.New("openai: stop sequences are not supported")
	// ErrMultipleCandidatesNotSupported is returned when multiple candidates are requested, which is not supported.
	ErrMultipleCandidatesNotSupported = errors.New("openai: multiple candidates per request are not supported")
	// ErrPenaltiesNotSupported is returned when frequency/presence penalties are used, which is not supported.
	ErrPenaltiesNotSupported = errors.New("openai: frequency/presence penalties are not supported")
	// ErrLabelsNotSupported is returned when request labels are used, which is not supported.
	ErrLabelsNotSupported = errors.New("openai: request labels are not supported")
	// ErrSafetySettingsNotSupported is returned when Gemini safety settings are used, which is not supported.
	ErrSafetySettingsNotSupported = errors.New("openai: gemini safety settings are not supported")
	// ErrUnsupportedMIMEType is returned when an unsupported MIME type is used.
	ErrUnsupportedMIMEType = errors.New("openai: unsupported mime type")

	// ErrEmptyJSONSchema is returned when an empty JSON schema is provided.
	ErrEmptyJSONSchema = errors.New("openai: empty json schema")
	// ErrEmptyResponse is returned when the OpenAI API returns an empty response.
	ErrEmptyResponse = errors.New("openai: empty response")
	// ErrNoOutputItems is returned when the response contains no output items.
	ErrNoOutputItems = errors.New("openai: response included no output items")
	// ErrUnsupportedMessageContentType is returned when an unsupported message content type is used.
	ErrUnsupportedMessageContentType = errors.New("openai: unsupported message content type")
	// ErrUnsupportedOutputItemType is returned when an unsupported output item type is used.
	ErrUnsupportedOutputItemType = errors.New("openai: unsupported output item type")
	// ErrNoTextOrToolContent is returned when the response output does not contain text or tool content.
	ErrNoTextOrToolContent = errors.New("openai: response output did not contain text or tool content")
)
