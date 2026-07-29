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

package auth

import (
	"context"
	"fmt"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/idtoken"
	"google.golang.org/api/option"
)

// CredentialProvider resolves a [Credential] for the current invocation. It is
// the Go analog of adk-python's BaseAuthProvider.get_auth_credential.
type CredentialProvider interface {
	// Credential resolves a ready-to-use credential. ctx carries cancellation
	// and deadlines.
	//
	// A provider that needs the acting user's identity (for example the GCP
	// provider, which keys on the user) recovers the ADK context from ctx via a
	// shared helper introduced together with the first provider that needs it —
	// so identity rides on the one ADK context rather than an auth-specific key.
	//
	// When interactive (3-legged) consent is required and cannot be completed
	// non-interactively, Credential returns a *ConsentRequiredError carrying the
	// authorization URI; the tool layer turns that into a human-in-the-loop
	// consent round-trip. Non-interactive providers never return it.
	Credential(ctx context.Context) (Credential, error)
}

// ProviderFunc adapts an ordinary function to a [CredentialProvider].
type ProviderFunc func(ctx context.Context) (Credential, error)

// Credential implements [CredentialProvider].
func (f ProviderFunc) Credential(ctx context.Context) (Credential, error) {
	return f(ctx)
}

// ConsentRequiredError signals that interactive, 3-legged OAuth consent is needed
// before a credential can be issued. Consumers detect it with errors.As.
type ConsentRequiredError struct {
	// AuthURI is the URL the end user must visit to grant consent.
	AuthURI string
	// Nonce is an opaque value echoed back to correlate the consent response.
	Nonce string
	// Key is the credential-store key to resume the flow under.
	Key string
}

// Error implements error.
func (e *ConsentRequiredError) Error() string {
	return fmt.Sprintf("auth: interactive consent required (auth_uri=%q)", e.AuthURI)
}

// StaticToken returns a provider that always yields the given bearer token.
func StaticToken(token string) CredentialProvider {
	return ProviderFunc(func(context.Context) (Credential, error) {
		return BearerCredential{Token: token}, nil
	})
}

// APIKey returns a provider that yields a header-based API key credential,
// where name is the header (for example "X-Api-Key").
func APIKey(name, value string) CredentialProvider {
	return ProviderFunc(func(context.Context) (Credential, error) {
		return APIKeyCredential{Name: name, Value: value}, nil
	})
}

// TokenSourceProvider wraps any [oauth2.TokenSource]. The resolved credential
// carries the source, so a fresh (auto-refreshed) token is minted at apply
// time. This covers client-credentials / 2-legged OAuth.
func TokenSourceProvider(ts oauth2.TokenSource) CredentialProvider {
	return ProviderFunc(func(context.Context) (Credential, error) {
		if ts == nil {
			return nil, fmt.Errorf("auth: nil token source")
		}
		return OAuth2Credential{TokenSource: ts}, nil
	})
}

// scopeCloudPlatform is the default ADC scope, matching adk-python and gcloud.
const scopeCloudPlatform = "https://www.googleapis.com/auth/cloud-platform"

func adcTokenSource(ctx context.Context, scopes []string) (oauth2.TokenSource, error) {
	if len(scopes) == 0 {
		scopes = []string{scopeCloudPlatform}
	}
	creds, err := google.FindDefaultCredentials(ctx, scopes...)
	if err != nil {
		return nil, fmt.Errorf("auth: find default credentials: %w", err)
	}
	return creds.TokenSource, nil
}

// ADC returns a provider backed by Google Application Default Credentials.
// Credentials are discovered lazily on first use and reused thereafter; with no
// scopes the cloud-platform scope is used.
func ADC(scopes ...string) CredentialProvider {
	return lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		return adcTokenSource(ctx, scopes)
	})
}

// ServiceAccountConfig configures a service-account credential provider.
//
// Spec: https://cloud.google.com/iam/docs/service-account-creds
type ServiceAccountConfig struct {
	// JSONKey is a service-account key in JSON form. When empty, Application
	// Default Credentials are used instead.
	JSONKey []byte
	// Scopes requested for an OAuth2 access token. Ignored when Audience is set.
	// Required with an explicit JSONKey; when falling back to ADC (JSONKey
	// empty), empty Scopes default to the cloud-platform scope.
	Scopes []string
	// Audience, when non-empty, requests a Google-signed ID token for that
	// audience (for example a Cloud Run or IAP URL) instead of an access token.
	Audience string
}

// ServiceAccount returns a provider that mints tokens for a service account,
// from an explicit key or from ADC. The token source is built lazily on first
// use and reused thereafter.
func ServiceAccount(cfg ServiceAccountConfig) CredentialProvider {
	return lazyTokenSource(func(ctx context.Context) (oauth2.TokenSource, error) {
		if cfg.Audience != "" {
			var opts []option.ClientOption
			if len(cfg.JSONKey) > 0 {
				opts = append(opts, option.WithAuthCredentialsJSON(option.ServiceAccount, cfg.JSONKey))
			}
			ts, err := idtoken.NewTokenSource(ctx, cfg.Audience, opts...)
			if err != nil {
				return nil, fmt.Errorf("auth: id token source: %w", err)
			}
			return ts, nil
		}
		if len(cfg.JSONKey) > 0 {
			// Stricter than adk-python (scopes optional there): an explicit-key
			// access token is scope-bound, so no scopes = unusable — fail fast.
			if len(cfg.Scopes) == 0 {
				return nil, fmt.Errorf("auth: scopes are required for a service-account access token")
			}
			creds, err := google.CredentialsFromJSONWithType(ctx, cfg.JSONKey, google.ServiceAccount, cfg.Scopes...)
			if err != nil {
				return nil, fmt.Errorf("auth: credentials from json: %w", err)
			}
			return creds.TokenSource, nil
		}
		return adcTokenSource(ctx, cfg.Scopes)
	})
}

// initTimeout bounds a hung init so it fails and is retried, not wedged forever.
var initTimeout = 30 * time.Second

// lazyTokenSource builds a provider that runs init once in a detached goroutine
// and reuses the source on success; a failed init is retried. Each caller waits
// on init or its own ctx, so a slow init never blocks a caller past its deadline.
func lazyTokenSource(init func(context.Context) (oauth2.TokenSource, error)) CredentialProvider {
	type attempt struct {
		done chan struct{}
		ts   oauth2.TokenSource
		err  error
	}
	var (
		mu  sync.Mutex
		ts  oauth2.TokenSource
		cur *attempt // in-flight init, for single-flight
	)
	return ProviderFunc(func(ctx context.Context) (Credential, error) {
		mu.Lock()
		if ts != nil {
			mu.Unlock()
			return OAuth2Credential{TokenSource: ts}, nil
		}
		a := cur
		if a == nil {
			a = &attempt{done: make(chan struct{})}
			cur = a
			// Detach from the caller's cancellation (keep values); initTimeout bounds it.
			go func() {
				dctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), initTimeout)
				defer cancel()
				a.ts, a.err = init(dctx)
				mu.Lock()
				if a.err == nil {
					ts = a.ts
				}
				cur = nil // reset so a failure is retried
				mu.Unlock()
				close(a.done) // publishes a.ts/a.err to waiters
			}()
		}
		mu.Unlock()

		select {
		case <-a.done:
			if a.err != nil {
				return nil, a.err
			}
			return OAuth2Credential{TokenSource: a.ts}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
}
