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

package shared

import "errors"

var (
	// ErrModelNameRequired is returned when a model name is not provided.
	ErrModelNameRequired = errors.New("openai: model name is required")
	// ErrUnsupportedAPI is returned when ClientConfig.API names an API this package does not implement.
	ErrUnsupportedAPI = errors.New("openai: unsupported API")
	// ErrNoChoices is returned when a Chat Completions response carries no choices.
	ErrNoChoices = errors.New("openai: response included no choices")
	// ErrRequestNil is returned when the provided request is nil.
	ErrRequestNil = errors.New("openai: request is nil")
	// ErrNoContents is returned when the LLM request has no contents.
	ErrNoContents = errors.New("openai: LLM request has no contents to convert")
	// ErrFunctionCallMissingName is returned when a function call is missing a name.
	ErrFunctionCallMissingName = errors.New("openai: function call missing name")
	// ErrTopKNotSupported is returned when TopK is used, which is not supported.
	ErrTopKNotSupported = errors.New("openai: topK is not supported")
	// ErrStopSequencesNotSupported is returned when stop sequences are used with the Responses API, which does not support them.
	ErrStopSequencesNotSupported = errors.New("openai: stop sequences are not supported")
	// ErrMultipleCandidatesNotSupported is returned when multiple candidates are requested, which is not supported.
	ErrMultipleCandidatesNotSupported = errors.New("openai: multiple candidates per request are not supported")
	// ErrPenaltiesNotSupported is returned when frequency/presence penalties are used with the Responses API, which does not support them.
	ErrPenaltiesNotSupported = errors.New("openai: frequency/presence penalties are not supported")
	// ErrLabelsNotSupported is returned when request labels are used, which is not supported.
	ErrLabelsNotSupported = errors.New("openai: request labels are not supported")
	// ErrSafetySettingsNotSupported is returned when Gemini safety settings are used, which is not supported.
	ErrSafetySettingsNotSupported = errors.New("openai: gemini safety settings are not supported")
	// ErrUnsupportedMIMEType is returned when an unsupported MIME type is used.
	ErrUnsupportedMIMEType = errors.New("openai: unsupported mime type")
	// ErrMediaMIMETypeRequired is returned when a part carries media without a
	// MIME type. The type is what decides whether an endpoint has a field for
	// the media at all, so it is required rather than defaulted: calling
	// unlabelled bytes application/octet-stream told the server they were
	// opaque when in truth nobody had looked.
	ErrMediaMIMETypeRequired = errors.New("openai: media part is missing a mime type")
	// ErrMediaDataRequired is returned when a part carries inline media whose
	// bytes are empty, which would otherwise go out as a data URL with nothing
	// after the comma.
	ErrMediaDataRequired = errors.New("openai: inline media part carries no data")
	// ErrMediaURIRequired is returned when a part carries file data without a
	// URI, leaving nothing to name the media.
	ErrMediaURIRequired = errors.New("openai: file data part is missing a uri")
	// ErrMediaOnAssistantTurn is returned when a replayed assistant turn
	// carries media. Both endpoints replay a prior assistant turn as text —
	// "output_text" items on Responses, a string on Chat Completions — and
	// neither shape has a field for an image or a file, so media there is
	// refused rather than dropped on the way out.
	ErrMediaOnAssistantTurn = errors.New("openai: media on a replayed assistant turn is not supported")
	// ErrUnsupportedConfigField is returned when a generation config field has no equivalent on the selected API; the message names it.
	ErrUnsupportedConfigField = errors.New("openai: unsupported generation config field")

	// ErrEmptyJSONSchema is returned when an empty JSON schema is provided.
	ErrEmptyJSONSchema = errors.New("openai: empty json schema")
	// ErrEmptyResponse is returned when the OpenAI API returns an empty response.
	ErrEmptyResponse = errors.New("openai: empty response")
	// ErrNoOutputItems is returned when the response contains no output items.
	ErrNoOutputItems = errors.New("openai: response included no output items")
	// ErrResponseFailed is returned when the server reports the response itself
	// as failed, whatever output it came with. Such a failure arrives as HTTP
	// 200, and both a blocking call and a stream report it in place of the
	// output that would otherwise have read as a turn.
	ErrResponseFailed = errors.New("openai: response failed")
	// ErrUnsupportedMessageContentType is returned when an unsupported message content type is used.
	ErrUnsupportedMessageContentType = errors.New("openai: unsupported message content type")
	// ErrUnsupportedOutputItemType is returned when an unsupported output item type is used.
	ErrUnsupportedOutputItemType = errors.New("openai: unsupported output item type")
	// ErrFunctionCallArgs is returned when a function call's arguments are not a decodable JSON object.
	ErrFunctionCallArgs = errors.New("openai: parse function call args")
	// ErrNoTextOrToolContent is returned when the response output does not contain text or tool content.
	ErrNoTextOrToolContent = errors.New("openai: response output did not contain text or tool content")
)
