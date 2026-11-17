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

package credentialservice

import (
	"context"

	"google.golang.org/adk/v2/tool/toolauth"
)

// Service loads and saves the credentials of tools, keyed by app name, user
// ID and the credential key of an auth config ([toolauth.Config.Key]).
//
// Implementations must be safe for concurrent use: ADK runs the function
// calls of one model turn concurrently, and each may load or save a
// credential.
type Service interface {
	// LoadCredential returns the credential saved for appName and userID
	// under the key of config. It returns a nil credential and a nil error
	// when nothing is saved under that key.
	//
	// The returned credential is the caller's: changing it does not change
	// the stored credential, and a later save does not change it.
	//
	// A nil config is an error. When config.Key fails, LoadCredential returns
	// that error and a nil credential.
	LoadCredential(ctx context.Context, appName, userID string, config *toolauth.Config) (*toolauth.Credential, error)

	// SaveCredential saves config.ExchangedAuthCredential for appName and
	// userID under the key of config, replacing any credential saved under
	// that key. The service keeps its own copy, so the caller may change
	// config afterwards. A nil ExchangedAuthCredential erases the saved
	// credential: a later LoadCredential under that key returns none, as
	// adk-python's InMemoryCredentialService does.
	//
	// A nil config is an error. When config.Key fails, SaveCredential returns
	// that error and stores nothing.
	SaveCredential(ctx context.Context, appName, userID string, config *toolauth.Config) error
}
