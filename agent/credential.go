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

package agent

import (
	"errors"

	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/tool/toolauth"
)

var (
	errNoCredentialService = errors.New("credential service is not initialized")
	errNilCredentialConfig = errors.New("credential auth config is nil")
)

// LoadCredential loads a credential from the credential service of ctx's
// invocation, the one its runner was configured with (runner.Config's
// CredentialService). It passes the service the invocation's app name and
// user ID and config, so a tool or callback reads only the credentials of its
// own app and user. It returns a nil credential and a nil error when the
// service holds no credential under config's key.
//
// It returns the error "credential service is not initialized", without
// reading config, when the invocation has no credential service; that check
// comes before the check of config. A nil config is an error. Otherwise it
// returns the credential and the error of the service's LoadCredential
// unchanged.
func LoadCredential(ctx Context, config *toolauth.Config) (*toolauth.Credential, error) {
	svc := icredentialservice.FromContext(ctx)
	if svc == nil {
		return nil, errNoCredentialService
	}
	if config == nil {
		return nil, errNilCredentialConfig
	}
	return svc.LoadCredential(ctx, ctx.AppName(), ctx.UserID(), config)
}

// SaveCredential saves config.ExchangedAuthCredential to the credential
// service of ctx's invocation, the one its runner was configured with
// (runner.Config's CredentialService), under the invocation's app name and
// user ID and config's key. It replaces any credential saved under that key.
// A nil ExchangedAuthCredential erases the saved credential, so a later
// [LoadCredential] with the same key returns none.
//
// It returns the error "credential service is not initialized", without
// reading config, when the invocation has no credential service; that check
// comes before the check of config. A nil config is an error. Otherwise it
// returns the error of the service's SaveCredential unchanged.
func SaveCredential(ctx Context, config *toolauth.Config) error {
	svc := icredentialservice.FromContext(ctx)
	if svc == nil {
		return errNoCredentialService
	}
	if config == nil {
		return errNilCredentialConfig
	}
	return svc.SaveCredential(ctx, ctx.AppName(), ctx.UserID(), config)
}
