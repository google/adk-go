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

// Package credentialservice defines the service that stores the credentials
// tools obtain, so that a later invocation of the same user can use them
// without asking the user again.
//
// A credential is stored under three parts: the app name and user ID of the
// invocation that saves it, and the key of the auth config it belongs to
// ([google.golang.org/adk/v2/tool/toolauth.Config.Key]). Credentials of
// different apps or users never meet, and one user can hold one credential
// per key.
//
// A runner takes a [Service] in its CredentialService configuration field and
// makes it available to every invocation it starts. Tools and callbacks reach
// it through agent.LoadCredential and agent.SaveCredential, which pass the
// invocation's app name and user ID. A runner without a service has none:
// those calls then fail. [InMemoryService] keeps credentials in the memory of
// the process, and loses them when it ends.
//
// EXPERIMENTAL: This package is experimental; its API and behavior may change or be removed in a later version.
package credentialservice
