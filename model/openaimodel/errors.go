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

import "google.golang.org/adk/v2/model/openaimodel/internal/shared"

// The sentinels are defined in the internal package the endpoint paths return
// them from, and aliased here. Assignment preserves identity, so errors.Is is
// unaffected.
var (
	// ErrModelNameRequired is returned when a model name is not provided.
	ErrModelNameRequired = shared.ErrModelNameRequired
	// ErrUnsupportedAPI is returned when ClientConfig.API names an API this package does not implement.
	ErrUnsupportedAPI = shared.ErrUnsupportedAPI
	// ErrNoChoices is returned when a Chat Completions response carries no choices.
	ErrNoChoices = shared.ErrNoChoices
	// ErrRequestNil is returned when the provided request is nil.
	ErrRequestNil = shared.ErrRequestNil
	// ErrNoContents is returned when the LLM request has no contents.
	ErrNoContents = shared.ErrNoContents
	// ErrFunctionCallMissingName is returned when a function call is missing a name.
	ErrFunctionCallMissingName = shared.ErrFunctionCallMissingName
	// ErrTopKNotSupported is returned when TopK is used, which is not supported.
	ErrTopKNotSupported = shared.ErrTopKNotSupported
	// ErrStopSequencesNotSupported is returned when stop sequences are used with the Responses API, which does not support them.
	ErrStopSequencesNotSupported = shared.ErrStopSequencesNotSupported
	// ErrMultipleCandidatesNotSupported is returned when multiple candidates are requested, which is not supported.
	ErrMultipleCandidatesNotSupported = shared.ErrMultipleCandidatesNotSupported
	// ErrPenaltiesNotSupported is returned when frequency/presence penalties are used with the Responses API, which does not support them.
	ErrPenaltiesNotSupported = shared.ErrPenaltiesNotSupported
	// ErrLabelsNotSupported is returned when request labels are used, which is not supported.
	ErrLabelsNotSupported = shared.ErrLabelsNotSupported
	// ErrSafetySettingsNotSupported is returned when Gemini safety settings are used, which is not supported.
	ErrSafetySettingsNotSupported = shared.ErrSafetySettingsNotSupported
	// ErrUnsupportedMIMEType is returned when an unsupported MIME type is used.
	ErrUnsupportedMIMEType = shared.ErrUnsupportedMIMEType
	// ErrUnsupportedConfigField is returned when a generation config field has no equivalent on the selected API; the message names it.
	ErrUnsupportedConfigField = shared.ErrUnsupportedConfigField
	// ErrEmptyJSONSchema is returned when an empty JSON schema is provided.
	ErrEmptyJSONSchema = shared.ErrEmptyJSONSchema
	// ErrEmptyResponse is returned when the OpenAI API returns an empty response.
	ErrEmptyResponse = shared.ErrEmptyResponse
	// ErrNoOutputItems is returned when the response contains no output items.
	ErrNoOutputItems = shared.ErrNoOutputItems
	// ErrUnsupportedMessageContentType is returned when an unsupported message content type is used.
	ErrUnsupportedMessageContentType = shared.ErrUnsupportedMessageContentType
	// ErrUnsupportedOutputItemType is returned when an unsupported output item type is used.
	ErrUnsupportedOutputItemType = shared.ErrUnsupportedOutputItemType
	// ErrFunctionCallArgs is returned when a function call's arguments are not a decodable JSON object.
	ErrFunctionCallArgs = shared.ErrFunctionCallArgs
	// ErrNoTextOrToolContent is returned when the response output does not contain text or tool content.
	ErrNoTextOrToolContent = shared.ErrNoTextOrToolContent
	// ErrResponseFailed is returned when the server reports the response itself
	// as failed, whatever output it came with. Such a failure arrives as HTTP
	// 200, and both a blocking call and a stream report it in place of the
	// output that would otherwise have read as a turn.
	ErrResponseFailed = shared.ErrResponseFailed
)
