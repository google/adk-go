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

package authtool

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/authflow"
	"google.golang.org/adk/v2/internal/authhandler"
	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/tool/toolauth"
)

// CredentialManager orchestrates the credential lifecycle: it loads the
// credential a [toolauth.Config] describes, exchanges it for tokens,
// refreshes tokens that are about to expire, and saves the result in the
// runner's credential service.
//
// Create one with [NewCredentialManager]; the zero value is not usable. A
// CredentialManager is safe for concurrent use. Calls of
// [CredentialManager.GetAuthCredential] for the same app name, user ID and
// credential key run one at a time, across every CredentialManager of the
// process, so that two tools sharing one config redeem a code or request a
// token only once.
//
// This is adk-python's CredentialManager (auth/credential_manager.py), which
// adk-python marks experimental.
//
// EXPERIMENTAL: CredentialManager is experimental; its API and behavior may change or be removed in a later version.
type CredentialManager struct {
	cfg   *toolauth.Config
	clock func() time.Time

	mu         sync.RWMutex // guards exchangers
	exchangers map[toolauth.CredentialType]CredentialExchanger
}

// NewCredentialManager creates a new manager for the credential cfg
// describes.
//
// It returns an error when cfg is nil or its AuthScheme has no Type, as the
// zero Scheme has none. The manager keeps cfg and never writes to it, so cfg
// can be shared by many tools and users; the caller must not change it
// afterwards. The manager starts with an [OAuth2CredentialExchanger]
// registered for OAuth2 and OpenID Connect credentials. clock is the time the
// manager decides token expiry by; nil means [time.Now].
//
// This is the constructor of adk-python's CredentialManager
// (auth/credential_manager.py), which adk-python marks experimental.
//
// EXPERIMENTAL: NewCredentialManager is experimental; its API and behavior may change or be removed in a later version.
func NewCredentialManager(cfg *toolauth.Config, clock func() time.Time) (*CredentialManager, error) {
	if cfg == nil {
		return nil, errors.New("authtool: NewCredentialManager needs an auth config")
	}
	if cfg.AuthScheme.Type == "" {
		return nil, errors.New("authtool: NewCredentialManager needs an auth scheme with a type")
	}
	if clock == nil {
		clock = time.Now
	}
	oauth2Exchanger := OAuth2CredentialExchanger{}
	return &CredentialManager{
		cfg:   cfg,
		clock: clock,
		exchangers: map[toolauth.CredentialType]CredentialExchanger{
			toolauth.CredentialTypeOAuth2:        oauth2Exchanger,
			toolauth.CredentialTypeOpenIDConnect: oauth2Exchanger,
		},
	}, nil
}

// RegisterCredentialExchanger registers an exchanger for a credential type on this instance.
//
// exchanger replaces the one registered for credType on m, and no other
// manager sees it. It may be called while other goroutines call
// [CredentialManager.GetAuthCredential]; calls that start after it returns
// use exchanger.
//
// This is adk-python's CredentialManager.register_credential_exchanger
// (auth/credential_manager.py), which adk-python marks experimental.
//
// EXPERIMENTAL: RegisterCredentialExchanger is experimental; its API and behavior may change or be removed in a later version.
func (m *CredentialManager) RegisterCredentialExchanger(credType toolauth.CredentialType, exchanger CredentialExchanger) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.exchangers[credType] = exchanger
}

// GetAuthCredential loads, exchanges, and refreshes the credential.
//
// ctx must be a tool's context. An API-key or HTTP credential is returned as
// a deep copy of the config's raw credential. Any other credential comes from
// the first of: the runner's credential service, the user's answer to an
// earlier request ([authflow.GetAuthResponse], which redeems an OAuth2
// authorization code), and, for the client-credentials grant, a copy of the
// raw credential. An exchanger registered for the credential's type then
// exchanges it; a credential that was not exchanged is refreshed when its
// token expires less than 60 seconds from now. A credential that came from
// the user's answer, was exchanged, or needed a refresh (even one that
// failed, as in adk-python) is saved in the credential service, through a
// copy of the config, so the config itself never holds a user's credential.
// Without a credential service, the load and the save are skipped.
//
// GetAuthCredential returns (nil, nil) when no credential is ready and the
// user has to authorize; the caller then calls
// [CredentialManager.RequestCredential]. It returns (nil, err) when the
// config is invalid or names an unsupported scheme or credential (a custom
// scheme, OAuth2 endpoint discovery, a service account), when redeeming the
// user's authorization code fails, when an exchanger returns no credential,
// when the credential service fails, or when ctx ends while the call waits
// for another call for the same credential. It returns (cred, err) when an
// exchange or a refresh fails: cred is then the credential as it was before,
// unexchanged or stale, as in adk-python.
//
// This is adk-python's CredentialManager.get_auth_credential
// (auth/credential_manager.py), which adk-python marks experimental.
//
// EXPERIMENTAL: GetAuthCredential is experimental; its API and behavior may change or be removed in a later version.
func (m *CredentialManager) GetAuthCredential(ctx agent.Context) (*toolauth.Credential, error) {
	if !knownSchemeType(m.cfg.AuthScheme.Type) {
		return nil, &authhandler.ConfigError{Err: errors.New("custom auth scheme is not supported")}
	}
	key, err := m.cfg.Key()
	if err != nil {
		return nil, fmt.Errorf("authtool: credential key: %w", err)
	}
	// Held until the credential is saved, so a caller waiting for the same
	// credential loads what this one saved instead of exchanging again.
	unlock, err := lockCredential(ctx, credentialLockKey{ctx.AppName(), ctx.UserID(), key})
	if err != nil {
		return nil, fmt.Errorf("authtool: wait for credential: %w", err)
	}
	defer unlock()

	if err := validate(m.cfg); err != nil {
		return nil, err
	}
	raw := m.cfg.RawAuthCredential
	if isReady(raw) {
		return raw.Clone(), nil
	}

	hasService := icredentialservice.FromContext(ctx) != nil
	var cred *toolauth.Credential
	if hasService {
		loaded, err := agent.LoadCredential(ctx, m.cfg)
		if err != nil {
			return nil, fmt.Errorf("authtool: load credential: %w", err)
		}
		cred = loaded
	}
	fromAuthResponse := false
	if cred == nil {
		answered, err := authflow.GetAuthResponse(ctx, m.cfg)
		if err != nil {
			return nil, err
		}
		cred, fromAuthResponse = answered, answered != nil
	}
	if cred == nil {
		if raw == nil || !usesClientCredentials(&m.cfg.AuthScheme) {
			return nil, nil
		}
		cred = raw.Clone()
	}

	cred, exchanged, exchangeErr := m.exchange(ctx, cred)
	if cred == nil {
		return nil, exchangeErr
	}
	refreshed := false
	var refreshErr error
	if !exchanged {
		cred, refreshed, refreshErr = refresh(ctx, &m.cfg.AuthScheme, cred, m.clock())
	}
	softErr := errors.Join(exchangeErr, refreshErr)

	// adk-python also saves a raw copy when the auth response was empty
	// (credential_manager.py:284-287); a credential nobody exchanged, refreshed
	// or answered is not worth storing.
	if hasService && (fromAuthResponse || exchanged || refreshed) {
		if err := m.save(ctx, cred); err != nil {
			return nil, err
		}
	}
	return cred, softErr
}

// exchange runs the exchanger registered for cred's type. It returns a nil
// credential, with the error to return, when the exchanger fails the call.
func (m *CredentialManager) exchange(ctx agent.Context, cred *toolauth.Credential) (*toolauth.Credential, bool, error) {
	m.mu.RLock()
	ex := m.exchangers[cred.AuthType]
	m.mu.RUnlock()
	if ex == nil {
		return cred, false, nil
	}
	out, exchanged, err := ex.Exchange(ctx, m.cfg, cred)
	if out == nil {
		if err == nil {
			err = fmt.Errorf("authtool: exchanger for %s returned no credential", cred.AuthType)
		}
		return nil, false, err
	}
	return out, exchanged, err
}

// save stores cred in the credential service through a copy of the shared
// config, which never carries a user's credential.
func (m *CredentialManager) save(ctx agent.Context, cred *toolauth.Credential) error {
	cfg := *m.cfg
	cfg.ExchangedAuthCredential = cred
	if err := agent.SaveCredential(ctx, &cfg); err != nil {
		return fmt.Errorf("authtool: save credential: %w", err)
	}
	return nil
}

// RequestCredential records a request for the user to authenticate.
//
// It records the request on the response of the tool call ctx belongs to, as
// [authflow.RequestCredential] does, and returns an error when ctx has no
// function call ID.
//
// This is adk-python's CredentialManager.request_credential
// (auth/credential_manager.py), which adk-python marks experimental.
//
// EXPERIMENTAL: RequestCredential is experimental; its API and behavior may change or be removed in a later version.
func (m *CredentialManager) RequestCredential(ctx agent.Context) error {
	return authflow.RequestCredential(ctx, m.cfg)
}

func knownSchemeType(t toolauth.SchemeType) bool {
	switch t {
	case toolauth.SchemeTypeAPIKey, toolauth.SchemeTypeHTTP, toolauth.SchemeTypeOAuth2, toolauth.SchemeTypeOpenIDConnect:
		return true
	}
	return false
}

// validate checks cfg as adk-python's _validate_credential does, with its
// error texts, and refuses the inputs this package does not support. The
// errors adk-python raises as ValueError are *authhandler.ConfigError; the
// refusals of OAuth2 auto-discovery and service accounts, which adk-python
// supports, are not.
func validate(cfg *toolauth.Config) error {
	scheme, raw := &cfg.AuthScheme, cfg.RawAuthCredential
	if raw == nil && (scheme.Type == toolauth.SchemeTypeOAuth2 || scheme.Type == toolauth.SchemeTypeOpenIDConnect) {
		return &authhandler.ConfigError{Err: fmt.Errorf("raw_auth_credential is required for auth_scheme type %s", scheme.Type)}
	}
	if raw != nil && raw.OAuth2 == nil &&
		(raw.AuthType == toolauth.CredentialTypeOAuth2 || raw.AuthType == toolauth.CredentialTypeOpenIDConnect) {
		return &authhandler.ConfigError{Err: fmt.Errorf("auth_config.raw_credential.oauth2 required for credential type %s", raw.AuthType)}
	}
	if missingOAuthInfo(scheme) {
		if scheme.IssuerURL != "" {
			return errors.New("OAuth2 auto-discovery is not supported")
		}
		return &authhandler.ConfigError{Err: errors.New("OAuth scheme info is missing, and auto-discovery has failed to fill them in.")} //nolint:staticcheck // ST1005: the text is adk-python's
	}
	if raw != nil && raw.AuthType == toolauth.CredentialTypeServiceAccount {
		return errors.New("service account is not supported")
	}
	return nil
}

// missingOAuthInfo reports whether an OAuth2 scheme's flow lacks a URL it
// needs, as adk-python's _missing_oauth_info decides.
func missingOAuthInfo(s *toolauth.Scheme) bool {
	f := s.Flows
	if s.Type != toolauth.SchemeTypeOAuth2 || f == nil {
		return false
	}
	return f.Implicit != nil && f.Implicit.AuthorizationURL == "" ||
		f.Password != nil && f.Password.TokenURL == "" ||
		f.ClientCredentials != nil && f.ClientCredentials.TokenURL == "" ||
		f.AuthorizationCode != nil && (f.AuthorizationCode.AuthorizationURL == "" || f.AuthorizationCode.TokenURL == "")
}

// isReady reports whether raw is used as it is, with no exchange or refresh.
func isReady(raw *toolauth.Credential) bool {
	return raw != nil && (raw.AuthType == toolauth.CredentialTypeAPIKey || raw.AuthType == toolauth.CredentialTypeHTTP)
}

// usesClientCredentials reports whether s is redeemed with the
// client-credentials grant, as adk-python's _is_client_credentials_flow
// decides.
func usesClientCredentials(s *toolauth.Scheme) bool {
	switch s.Type {
	case toolauth.SchemeTypeOAuth2:
		return s.Flows != nil && s.Flows.ClientCredentials != nil
	case toolauth.SchemeTypeOpenIDConnect:
		return slices.Contains(s.GrantTypesSupported, string(toolauth.OAuthGrantTypeClientCredentials))
	}
	return false
}
