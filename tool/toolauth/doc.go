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

// Package toolauth is adk-python's auth data model, in its wire format: how a
// resource expects to be authenticated ([Scheme]), a credential a tool was
// configured with or obtained ([Credential]), and the two together with the
// key the credential is stored under ([Config]). When a tool needs the end
// user's consent, ADK sends a [Config] to the client in an
// adk_request_credential call, so these types write and read the same JSON as
// adk-python and ADK Web.
//
// The types are data that travels in events and to the client. They do not
// write themselves onto requests, which is what a
// [google.golang.org/adk/v2/auth.Credential] does. [Config.Key] is the string
// adk-python names a credential by, not a
// [google.golang.org/adk/v2/auth.CredentialKey].
package toolauth
