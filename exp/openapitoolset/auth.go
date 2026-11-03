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

package openapitoolset

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/tool/toolauth"
)

// resolvedAuth is the credential a tool applies to its request, placed where
// the auth scheme asks: as headers, query parameters or cookies.
type resolvedAuth struct {
	headers map[string]string
	query   map[string]string
	cookies map[string]string
}

func newResolvedAuth() *resolvedAuth {
	return &resolvedAuth{
		headers: map[string]string{},
		query:   map[string]string{},
		cookies: map[string]string{},
	}
}

// resolveAuth turns the tool's auth scheme and credential into what to put on
// the request. The second result is false when the scheme needs a credential
// the tool cannot produce without interaction — an OAuth2 or OpenID Connect
// flow with no access token — which the caller reports to the model, since the
// interactive flow is not supported yet. A nil result with true means no auth
// is needed. It mirrors the static part of adk-python's prepare_auth_credentials
// followed by credential_to_param.
func (t *RESTAPITool) resolveAuth(ctx context.Context) (*resolvedAuth, bool, error) {
	if t.authScheme == nil {
		return nil, true, nil
	}
	cred := t.authCredential
	if cred == nil {
		return nil, false, nil
	}

	out := newResolvedAuth()
	// Headers the credential carries alongside the secret, such as
	// x-goog-user-project, are always set. adk-python applies these in
	// _prepare_request_params from the configured credential.
	if cred.HTTP != nil {
		for k, v := range cred.HTTP.AdditionalHeaders {
			out.headers[k] = v
		}
	}

	switch {
	case cred.AuthType == toolauth.CredentialTypeServiceAccount && cred.ServiceAccount != nil:
		authz, err := serviceAccountAuthHeader(ctx, cred.ServiceAccount)
		if err != nil {
			return nil, false, err
		}
		out.headers["Authorization"] = authz
		return out, true, nil

	case t.authScheme.Type == toolauth.SchemeTypeOAuth2 || t.authScheme.Type == toolauth.SchemeTypeOpenIDConnect:
		// After an interactive flow the access token is an OAuth2 field; a tool
		// configured with a ready bearer token carries it under HTTP. Either is
		// sent as a bearer token, as adk-python's OAuth2CredentialExchanger
		// produces. With neither, the token must still be obtained interactively.
		token := ""
		if cred.OAuth2 != nil {
			token = cred.OAuth2.AccessToken
		}
		if token == "" && cred.HTTP != nil {
			token = cred.HTTP.Credentials.Token
		}
		if token == "" {
			return nil, false, nil
		}
		out.headers["Authorization"] = "Bearer " + token
		return out, true, nil
	}

	if err := credentialToParams(t.authScheme, cred, out); err != nil {
		return nil, false, err
	}
	return out, true, nil
}

// applyAuth merges resolved auth onto a prepared request. A header, query
// parameter or cookie set by auth overrides one of the same name built from the
// arguments, so a credential cannot be shadowed by a model-supplied value.
func applyAuth(req *preparedRequest, ra *resolvedAuth) {
	if ra == nil {
		return
	}
	for k, v := range ra.headers {
		req.headers[k] = v
	}
	for k, v := range ra.query {
		req.query.Set(k, v)
	}
	for k, v := range ra.cookies {
		req.cookies[k] = v
	}
}

// credentialToParams places an API key or an HTTP basic/bearer credential on
// out. It is the apiKey and http part of adk-python's credential_to_param.
func credentialToParams(scheme *toolauth.Scheme, cred *toolauth.Credential, out *resolvedAuth) error {
	if scheme.Type == toolauth.SchemeTypeAPIKey && cred.APIKey != "" {
		switch scheme.In {
		case "header":
			out.headers[scheme.Name] = cred.APIKey
		case "query":
			out.query[scheme.Name] = cred.APIKey
		case "cookie":
			out.cookies[scheme.Name] = cred.APIKey
		default:
			return fmt.Errorf("invalid API key location %q", scheme.In)
		}
		return nil
	}

	if cred.AuthType == toolauth.CredentialTypeHTTP && cred.HTTP != nil {
		c := cred.HTTP
		switch {
		case c.Credentials.Token != "":
			out.headers["Authorization"] = "Bearer " + c.Credentials.Token
			return nil
		case strings.EqualFold(c.Scheme, "basic") && (c.Credentials.Username != "" || c.Credentials.Password != ""):
			if strings.Contains(c.Credentials.Username, ":") {
				return fmt.Errorf("invalid HTTP auth credentials: username cannot contain colons")
			}
			raw := base64.StdEncoding.EncodeToString([]byte(c.Credentials.Username + ":" + c.Credentials.Password))
			out.headers["Authorization"] = "Basic " + raw
			return nil
		default:
			return fmt.Errorf("invalid HTTP auth credentials")
		}
	}

	return fmt.Errorf("invalid security scheme and credential combination")
}

// serviceAccountAuthHeader mints a token for a Google service account and
// returns the Authorization header value. It delegates the token exchange to
// the auth package, which handles a key, Application Default Credentials and ID
// tokens.
func serviceAccountAuthHeader(ctx context.Context, sa *toolauth.ServiceAccount) (string, error) {
	cfg := auth.ServiceAccountConfig{Scopes: sa.Scopes}
	if sa.UseIDToken {
		cfg.Audience = sa.Audience
	}
	if !sa.UseDefaultCredential && sa.ServiceAccountCredential != nil {
		key, err := serviceAccountKeyJSON(sa.ServiceAccountCredential)
		if err != nil {
			return "", err
		}
		cfg.JSONKey = key
	}
	cred, err := auth.ServiceAccount(cfg).Credential(ctx)
	if err != nil {
		return "", err
	}
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		return "", err
	}
	authz := h.Get("Authorization")
	if authz == "" {
		return "", fmt.Errorf("service account produced no Authorization header")
	}
	return authz, nil
}

// serviceAccountKeyJSON renders a service account credential as the Google key
// file JSON the auth package reads, whose fields are snake_case.
func serviceAccountKeyJSON(sac *toolauth.ServiceAccountCredential) ([]byte, error) {
	typ := sac.Type
	if typ == "" {
		typ = "service_account"
	}
	key := map[string]string{
		"type":                        typ,
		"project_id":                  sac.ProjectID,
		"private_key_id":              sac.PrivateKeyID,
		"private_key":                 sac.PrivateKey,
		"client_email":                sac.ClientEmail,
		"client_id":                   sac.ClientID,
		"auth_uri":                    sac.AuthURI,
		"token_uri":                   sac.TokenURI,
		"auth_provider_x509_cert_url": sac.AuthProviderX509CertURL,
		"client_x509_cert_url":        sac.ClientX509CertURL,
		"universe_domain":             sac.UniverseDomain,
	}
	data, err := json.Marshal(key)
	if err != nil {
		return nil, fmt.Errorf("render service account key: %w", err)
	}
	return data, nil
}
