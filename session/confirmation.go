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

package session

// ConfirmationRequest represents a request for confirmation that needs to be handled by the user/system.
type ConfirmationRequest struct {
	// ID uniquely identifies this confirmation request.
	ID string
	// ToolName is the name of the tool requesting confirmation.
	ToolName string
	// Hint provides context about why confirmation is needed.
	Hint string
	// Payload contains additional data related to the confirmation request.
	Payload map[string]any
}

// ConfirmationResponse represents a response to a confirmation request.
type ConfirmationResponse struct {
	// RequestID identifies which confirmation request this responds to.
	RequestID string
	// Approved indicates whether the action was approved or denied.
	Approved bool
	// Reason provides an optional explanation for the approval/denial.
	Reason string
}
