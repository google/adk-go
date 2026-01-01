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

package auth

// AuthToolArguments represents the arguments for the special long-running
// function tool that is used to request end user credentials.
// This matches Python ADK's AuthToolArguments in auth_tool.py:93
type AuthToolArguments struct {
	// FunctionCallID is the ID of the original function call that requested auth.
	FunctionCallID string `json:"function_call_id"`
	// AuthConfig is the auth configuration for the tool.
	AuthConfig *AuthConfig `json:"auth_config"`
}
