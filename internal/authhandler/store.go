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

package authhandler

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolauth"
)

type exchangeLock struct {
	mu   sync.Mutex
	refs int
}

var (
	exchangeLocksMu sync.Mutex
	exchangeLocks   = map[[sha256.Size]byte]*exchangeLock{}
)

func lockExchange(key string, o *toolauth.OAuth2Auth) func() {
	k := sha256.Sum256([]byte(key + "\x00" + o.AuthResponseURI + "\x00" + o.AuthCode))
	exchangeLocksMu.Lock()
	l := exchangeLocks[k]
	if l == nil {
		l = &exchangeLock{}
		exchangeLocks[k] = l
	}
	l.refs++
	exchangeLocksMu.Unlock()

	l.mu.Lock()
	return func() {
		l.mu.Unlock()
		exchangeLocksMu.Lock()
		l.refs--
		if l.refs == 0 {
			delete(exchangeLocks, k)
		}
		exchangeLocksMu.Unlock()
	}
}

// StoreAuthResponse stores the credential in resp, the client's answer to the
// request ADK sent, under "temp:" plus that request's key, and returns the key.
func StoreAuthResponse(state session.State, requested *toolauth.Config, resp map[string]any) (string, error) {
	answer, err := parseAuthResponse(resp)
	if err != nil {
		return "", err
	}
	pinned := pinAuthResponse(requested, answer)
	key, err := pinned.Key()
	if err != nil {
		return "", fmt.Errorf("authhandler: credential key: %w", err)
	}
	if err := state.Set(session.KeyPrefixTemp+key, storedCredential(pinned)); err != nil {
		return "", fmt.Errorf("authhandler: store auth response: %w", err)
	}
	return key, nil
}

// storedCredential pins the configured client ID, so the client cannot pick
// which app redeems its code, and drops the secret, since state is readable.
func storedCredential(pinned *toolauth.Config) toolauth.Credential {
	cred := *pinned.ExchangedAuthCredential
	if cred.OAuth2 == nil {
		return cred
	}
	o := *cred.OAuth2
	if raw := pinned.RawAuthCredential; raw != nil && raw.OAuth2 != nil {
		o.ClientID = raw.OAuth2.ClientID
	}
	o.ClientSecret = ""
	cred.OAuth2 = &o
	return cred
}

// readStoredCredential returns the credential stored for key and the state key
// holding it: "temp:"+key, where StoreAuthResponse writes, or else key itself,
// where an app may keep one. The credential is nil when neither holds one.
func readStoredCredential(state session.State, scheme *toolauth.Scheme, key string) (string, *toolauth.Credential, error) {
	for _, k := range []string{session.KeyPrefixTemp + key, key} {
		v, err := state.Get(k)
		if errors.Is(err, session.ErrStateKeyNotExist) {
			continue
		}
		if err != nil {
			return "", nil, fmt.Errorf("authhandler: read stored credential: %w", err)
		}
		cred, err := credentialFromValue(scheme, v)
		if err != nil {
			return "", nil, err
		}
		if cred != nil {
			return k, cred, nil
		}
	}
	return "", nil, nil
}

// credentialFromValue reads a state value as adk-python's
// _read_stored_credential does: a credential as it is, a JSON object decoded,
// a non-empty string as a token for scheme. Other values hold no credential.
func credentialFromValue(scheme *toolauth.Scheme, v any) (*toolauth.Credential, error) {
	switch v := v.(type) {
	case toolauth.Credential:
		return &v, nil
	case *toolauth.Credential:
		if v == nil {
			return nil, nil
		}
		c := *v
		return &c, nil
	case map[string]any:
		return decodeCredential(v)
	case string:
		if v == "" {
			return nil, nil
		}
		return tokenCredential(scheme, v)
	}
	return nil, nil
}

func decodeCredential(m map[string]any) (*toolauth.Credential, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("authhandler: encode stored credential: %w", err)
	}
	var cred toolauth.Credential
	if err := json.Unmarshal(b, &cred); err != nil {
		return nil, fmt.Errorf("authhandler: stored credential does not decode: %w", err)
	}
	return &cred, nil
}

// tokenCredential wraps a bare token in the credential scheme expects, as
// adk-python's _credential_from_string does. HTTP basic needs a username and
// password, so a token for it is an error.
func tokenCredential(scheme *toolauth.Scheme, token string) (*toolauth.Credential, error) {
	switch scheme.Type {
	case toolauth.SchemeTypeAPIKey:
		return &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: token}, nil
	case toolauth.SchemeTypeHTTP:
		if strings.EqualFold(scheme.Scheme, "basic") {
			return nil, errors.New("authhandler: HTTP basic auth needs a username and password, not a token")
		}
		return &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme:      cmp.Or(scheme.Scheme, "bearer"),
			Credentials: toolauth.HTTPCredentials{Token: token},
		}}, nil
	}
	return &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: token}}, nil
}

// GetAuthResponse returns the credential the user's answer produced for cfg,
// the tool's own config, or nil when no answer is stored. A code is exchanged
// for tokens, which go back to state without the client secret. Concurrent
// exchanges of the same stored code are serialized so tool calls sharing a
// credential key redeem it once.
func GetAuthResponse(ctx context.Context, state session.State, cfg *toolauth.Config) (*toolauth.Credential, error) {
	key, err := cfg.Key()
	if err != nil {
		return nil, fmt.Errorf("authhandler: credential key: %w", err)
	}
	_, cred, err := readConfiguredCredential(state, cfg, key)
	if !needsExchange(&cfg.AuthScheme, cred) {
		return cred, err
	}
	unlock := lockExchange(key, cred.OAuth2)
	defer unlock()
	stateKey, cred, err := readConfiguredCredential(state, cfg, key)
	if !needsExchange(&cfg.AuthScheme, cred) {
		return cred, err
	}
	exchanged, err := exchangeCode(ctx, &cfg.AuthScheme, cred)
	if err != nil {
		return nil, err
	}
	if err := state.Set(stateKey, withoutClientSecret(exchanged)); err != nil {
		return nil, fmt.Errorf("authhandler: store exchanged credential: %w", err)
	}
	return exchanged, nil
}

func readConfiguredCredential(state session.State, cfg *toolauth.Config, key string) (string, *toolauth.Credential, error) {
	stateKey, cred, err := readStoredCredential(state, &cfg.AuthScheme, key)
	if err != nil || cred == nil {
		return "", nil, err
	}
	return stateKey, withConfiguredClient(cred, cfg.RawAuthCredential), nil
}

// withConfiguredClient puts the tool's own client ID, secret and auth method
// on cred. adk-python lets an answer's method win; pinning it keeps a bare
// credential, written with client_secret_basic, from overriding the config.
func withConfiguredClient(cred, raw *toolauth.Credential) *toolauth.Credential {
	if raw == nil || raw.OAuth2 == nil || cred.OAuth2 == nil {
		return cred
	}
	o := *cred.OAuth2
	o.ClientID = raw.OAuth2.ClientID
	o.ClientSecret = raw.OAuth2.ClientSecret
	o.TokenEndpointAuthMethod = raw.OAuth2.TokenEndpointAuthMethod
	out := *cred
	out.OAuth2 = &o
	return &out
}

// needsExchange reports whether cred still holds a code that ADK redeems, as
// adk-python's _is_exchangeable decides with allow_public.
func needsExchange(scheme *toolauth.Scheme, cred *toolauth.Credential) bool {
	if cred == nil || (scheme.Type != toolauth.SchemeTypeOAuth2 && scheme.Type != toolauth.SchemeTypeOpenIDConnect) {
		return false
	}
	o := cred.OAuth2
	return o != nil && o.AccessToken == "" && o.ClientID != ""
}

func withoutClientSecret(cred *toolauth.Credential) toolauth.Credential {
	out := *cred
	if out.OAuth2 != nil {
		o := *out.OAuth2
		o.ClientSecret = ""
		out.OAuth2 = &o
	}
	return out
}
