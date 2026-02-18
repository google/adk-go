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

package openai

const (
	// Event types
	responseOutputTextDelta            = "response.output_text.delta"
	responseReasoningTextDelta         = "response.reasoning_text.delta"
	responseReasoningSummaryTextDelta  = "response.reasoning_summary_text.delta"
	responseFunctionCallArgumentsDelta = "response.function_call_arguments.delta"
	responseFunctionCallArgumentsDone  = "response.function_call_arguments.done"
	responseOutputTextDone             = "response.output_text.done"
	responseReasoningTextDone          = "response.reasoning_text.done"
	responseReasoningSummaryTextDone   = "response.reasoning_summary_text.done"
	responseCompleted                  = "response.completed"
	responseInProgress                 = "response.in_progress"
	responseOutputItemAdded            = "response.output_item.added"
	responseOutputItemDone             = "response.output_item.done"
)
