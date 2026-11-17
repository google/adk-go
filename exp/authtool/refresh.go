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
	"context"
	"errors"
	"fmt"
	"time"

	"golang.org/x/oauth2"

	"google.golang.org/adk/v2/internal/authhandler"
	"google.golang.org/adk/v2/tool/toolauth"
)

// refreshLeeway is authlib's default leeway in OAuth2Token.is_expired: a token
// expiring sooner than this counts as expired.
const refreshLeeway = 60 * time.Second

// refreshNeeded reports whether o's access token expires less than a minute
// after now, as authlib's OAuth2Token.is_expired decides: a missing ExpiresAt
// is now plus ExpiresIn, cut to whole seconds, and neither means no expiry.
func refreshNeeded(o *toolauth.OAuth2Auth, now time.Time) bool {
	expiresAt := o.ExpiresAt
	if expiresAt == 0 {
		if o.ExpiresIn == 0 {
			return false
		}
		expiresAt = now.Unix() + o.ExpiresIn
	}
	return time.Unix(expiresAt, 0).Add(-refreshLeeway).Before(now)
}

// refresh refreshes an OAuth2 or OpenID Connect credential whose token
// expires within a minute of now, as adk-python's OAuth2CredentialRefresher
// does. It reports whether a refresh was needed. When the refresh fails, it
// returns cred unchanged, true and the error, as adk-python keeps the stale
// token.
func refresh(ctx context.Context, scheme *toolauth.Scheme, cred *toolauth.Credential, now time.Time) (*toolauth.Credential, bool, error) {
	if cred.AuthType != toolauth.CredentialTypeOAuth2 && cred.AuthType != toolauth.CredentialTypeOpenIDConnect {
		return cred, false, nil
	}
	o := cred.OAuth2
	if o == nil || !refreshNeeded(o, now) {
		return cred, false, nil
	}
	if o.RefreshToken == "" {
		return cred, true, errors.New("authtool: token expires and the credential has no refresh token")
	}
	conf, err := authhandler.TokenConfig(scheme, o)
	if err != nil {
		return cred, true, fmt.Errorf("authtool: refresh: %w", err)
	}
	ctx, cancel := context.WithTimeout(ctx, authhandler.TokenRequestTimeout)
	defer cancel()
	// A token without an access token is never valid, so the source always
	// asks the endpoint, whatever the stored expiry says.
	tok, err := conf.TokenSource(ctx, &oauth2.Token{RefreshToken: o.RefreshToken}).Token()
	if err != nil {
		return cred, true, fmt.Errorf("authtool: refresh: %w", authhandler.TokenError(err))
	}
	return authhandler.WithTokens(cred, tok), true, nil
}
