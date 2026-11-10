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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/v2/tool/toolauth"
)

func apiKeyScheme(in, name string) *toolauth.Scheme {
	return &toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: in, Name: name}
}

func TestCredentialToParamsAPIKey(t *testing.T) {
	tests := []struct {
		in       string
		where    func(*resolvedAuth) map[string]string
		wantName string
	}{
		{"header", func(r *resolvedAuth) map[string]string { return r.headers }, "X-API-Key"},
		{"query", func(r *resolvedAuth) map[string]string { return r.query }, "api_key"},
		{"cookie", func(r *resolvedAuth) map[string]string { return r.cookies }, "session"},
	}
	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			out := newResolvedAuth()
			scheme := apiKeyScheme(tc.in, tc.wantName)
			cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "secret-key"}
			if err := credentialToParams(scheme, cred, out); err != nil {
				t.Fatal(err)
			}
			if got := tc.where(out)[tc.wantName]; got != "secret-key" {
				t.Errorf("%s[%s] = %q, want secret-key", tc.in, tc.wantName, got)
			}
		})
	}
}

func TestCredentialToParamsAPIKeyInvalidLocation(t *testing.T) {
	out := newResolvedAuth()
	err := credentialToParams(apiKeyScheme("body", "X"), &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"}, out)
	if err == nil {
		t.Fatal("expected an error for an invalid API key location")
	}
}

func TestCredentialToParamsBearer(t *testing.T) {
	out := newResolvedAuth()
	scheme := &toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"}
	cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
		Scheme: "bearer", Credentials: toolauth.HTTPCredentials{Token: "tok123"},
	}}
	if err := credentialToParams(scheme, cred, out); err != nil {
		t.Fatal(err)
	}
	if out.headers["Authorization"] != "Bearer tok123" {
		t.Errorf("Authorization = %q, want Bearer tok123", out.headers["Authorization"])
	}
}

func TestCredentialToParamsBasic(t *testing.T) {
	out := newResolvedAuth()
	scheme := &toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "basic"}
	cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
		Scheme: "basic", Credentials: toolauth.HTTPCredentials{Username: "user", Password: "pass"},
	}}
	if err := credentialToParams(scheme, cred, out); err != nil {
		t.Fatal(err)
	}
	want := "Basic " + base64.StdEncoding.EncodeToString([]byte("user:pass"))
	if out.headers["Authorization"] != want {
		t.Errorf("Authorization = %q, want %q", out.headers["Authorization"], want)
	}
}

func TestCredentialToParamsBasicRejectsColonUsername(t *testing.T) {
	out := newResolvedAuth()
	scheme := &toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "basic"}
	cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
		Scheme: "basic", Credentials: toolauth.HTTPCredentials{Username: "user:name", Password: "pass"},
	}}
	if err := credentialToParams(scheme, cred, out); err == nil {
		t.Fatal("expected an error for a username containing a colon")
	}
}

func TestCredentialToParamsInvalidCombo(t *testing.T) {
	out := newResolvedAuth()
	// An apiKey scheme with no API key and no HTTP credential is not a valid
	// combination.
	err := credentialToParams(apiKeyScheme("header", "X"), &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey}, out)
	if err == nil {
		t.Fatal("expected an error for an invalid scheme/credential combination")
	}
}

func TestResolveAuthNoScheme(t *testing.T) {
	tool := &RESTAPITool{name: "t"}
	ra, ok, err := tool.resolveAuth(context.Background())
	if err != nil || !ok || ra != nil {
		t.Fatalf("resolveAuth = %v, %v, %v; want nil, true, nil", ra, ok, err)
	}
}

func TestResolveAuthSchemeNoCredential(t *testing.T) {
	tool := &RESTAPITool{name: "t", authScheme: apiKeyScheme("header", "X-API-Key")}
	ra, ok, err := tool.resolveAuth(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if ok || ra != nil {
		t.Fatalf("resolveAuth = %v, %v; want nil, false (needs a credential)", ra, ok)
	}
}

func TestResolveAuthOAuth2WithToken(t *testing.T) {
	tool := &RESTAPITool{
		name:           "t",
		authScheme:     &toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		authCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: "at-1"}},
	}
	ra, ok, err := tool.resolveAuth(context.Background())
	if err != nil || !ok {
		t.Fatalf("resolveAuth = %v, %v, %v; want ok", ra, ok, err)
	}
	if ra.headers["Authorization"] != "Bearer at-1" {
		t.Errorf("Authorization = %q, want Bearer at-1", ra.headers["Authorization"])
	}
}

func TestResolveAuthOAuth2WithoutTokenNeedsInteractive(t *testing.T) {
	tool := &RESTAPITool{
		name:           "t",
		authScheme:     &toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		authCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{ClientID: "c", ClientSecret: "s"}},
	}
	ra, ok, err := tool.resolveAuth(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if ok || ra != nil {
		t.Fatalf("resolveAuth = %v, %v; want nil, false (interactive needed)", ra, ok)
	}
}

func TestResolveAuthAdditionalHeaders(t *testing.T) {
	tool := &RESTAPITool{
		name:       "t",
		authScheme: &toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
		authCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme:            "bearer",
			Credentials:       toolauth.HTTPCredentials{Token: "tok"},
			AdditionalHeaders: map[string]string{"x-goog-user-project": "proj-1"},
		}},
	}
	ra, ok, err := tool.resolveAuth(context.Background())
	if err != nil || !ok {
		t.Fatalf("resolveAuth: %v, %v, %v", ra, ok, err)
	}
	if ra.headers["x-goog-user-project"] != "proj-1" {
		t.Errorf("additional header missing: %v", ra.headers)
	}
	if ra.headers["Authorization"] != "Bearer tok" {
		t.Errorf("Authorization = %q", ra.headers["Authorization"])
	}
}

func TestResolveAuthServiceAccountBadKeyErrors(t *testing.T) {
	// A bogus service-account key must surface as an error, not a panic, with no
	// network access needed.
	tool := &RESTAPITool{
		name:       "t",
		authScheme: &toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		authCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeServiceAccount, ServiceAccount: &toolauth.ServiceAccount{
			ServiceAccountCredential: &toolauth.ServiceAccountCredential{
				Type: "service_account", ClientEmail: "x@y.iam.gserviceaccount.com", PrivateKey: "not-a-key",
			},
			Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
		}},
	}
	_, _, err := tool.resolveAuth(context.Background())
	if err == nil {
		t.Fatal("expected an error for a bogus service account key")
	}
}

func TestServiceAccountKeyJSON(t *testing.T) {
	sac := &toolauth.ServiceAccountCredential{
		ProjectID: "proj", PrivateKeyID: "kid", PrivateKey: "pk", ClientEmail: "e@x.iam", ClientID: "cid",
		TokenURI: "https://oauth2.googleapis.com/token",
	}
	data, err := serviceAccountKeyJSON(sac)
	if err != nil {
		t.Fatal(err)
	}
	var key map[string]string
	if err := json.Unmarshal(data, &key); err != nil {
		t.Fatal(err)
	}
	if key["type"] != "service_account" {
		t.Errorf("type = %q, want service_account (default)", key["type"])
	}
	for field, want := range map[string]string{
		"project_id": "proj", "private_key_id": "kid", "private_key": "pk",
		"client_email": "e@x.iam", "client_id": "cid", "token_uri": "https://oauth2.googleapis.com/token",
	} {
		if key[field] != want {
			t.Errorf("key[%q] = %q, want %q", field, key[field], want)
		}
	}
}

// authServerTool builds a tool that targets a loopback server, with the given
// auth, and returns the request the server received.
func authServerTool(t *testing.T, scheme *toolauth.Scheme, cred *toolauth.Credential) (*RESTAPITool, *http.Request) {
	t.Helper()
	var captured *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Clone(context.Background())
		_, _ = io.WriteString(w, `{"ok": true}`)
	}))
	t.Cleanup(srv.Close)
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:           "test_tool",
		Endpoint:       Endpoint{BaseURL: srv.URL, Path: "/api", Method: "get"},
		AuthScheme:     scheme,
		AuthCredential: cred,
	})
	if err != nil {
		t.Fatal(err)
	}
	runTool(t, tool, map[string]any{})
	if captured == nil {
		t.Fatal("server received no request")
	}
	return tool, captured
}

func TestRunAppliesAPIKeyHeader(t *testing.T) {
	_, req := authServerTool(t,
		apiKeyScheme("header", "X-API-Key"),
		&toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "secret-key"})
	if got := req.Header.Get("X-API-Key"); got != "secret-key" {
		t.Errorf("X-API-Key header = %q, want secret-key", got)
	}
}

func TestRunAppliesAPIKeyQuery(t *testing.T) {
	_, req := authServerTool(t,
		apiKeyScheme("query", "api_key"),
		&toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "secret-key"})
	if got := req.URL.Query().Get("api_key"); got != "secret-key" {
		t.Errorf("api_key query = %q, want secret-key", got)
	}
}

func TestRunAppliesAPIKeyCookie(t *testing.T) {
	_, req := authServerTool(t,
		apiKeyScheme("cookie", "session"),
		&toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "secret-key"})
	c, err := req.Cookie("session")
	if err != nil || c.Value != "secret-key" {
		t.Errorf("session cookie = %v (err %v), want secret-key", c, err)
	}
}

func TestRunAppliesBearer(t *testing.T) {
	_, req := authServerTool(t,
		&toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
		&toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme: "bearer", Credentials: toolauth.HTTPCredentials{Token: "tok123"},
		}})
	if got := req.Header.Get("Authorization"); got != "Bearer tok123" {
		t.Errorf("Authorization = %q, want Bearer tok123", got)
	}
}

func TestRunAppliesBasic(t *testing.T) {
	_, req := authServerTool(t,
		&toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "basic"},
		&toolauth.Credential{AuthType: toolauth.CredentialTypeHTTP, HTTP: &toolauth.HTTPAuth{
			Scheme: "basic", Credentials: toolauth.HTTPCredentials{Username: "u", Password: "p"},
		}})
	user, pass, ok := req.BasicAuth()
	if !ok || user != "u" || pass != "p" {
		t.Errorf("basic auth = %q/%q (ok %v), want u/p", user, pass, ok)
	}
}

func TestToolsetAppliesConfigAuthToAllTools(t *testing.T) {
	scheme := apiKeyScheme("header", "X-API-Key")
	cred := &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k"}
	ts, err := New(Config{Spec: loadFixture(t), AuthScheme: scheme, AuthCredential: cred})
	if err != nil {
		t.Fatal(err)
	}
	tools, err := ts.Tools(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tl := range tools {
		rt := tl.(*RESTAPITool)
		if rt.authScheme != scheme {
			t.Errorf("tool %q scheme = %v, want the config override", rt.name, rt.authScheme)
		}
		if rt.authCredential != cred {
			t.Errorf("tool %q credential not set from config", rt.name)
		}
	}
}

func TestRunNoAuthWhenSchemeNil(t *testing.T) {
	// Without a scheme, no Authorization header is added.
	tool, _ := newServerTool(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Errorf("unexpected Authorization header: %q", r.Header.Get("Authorization"))
		}
		_, _ = io.WriteString(w, `{}`)
	}, nil)
	runTool(t, tool, map[string]any{})
}

func TestRunNeedsInteractiveAuthErrors(t *testing.T) {
	tool, err := NewRESTAPITool(RESTAPIToolConfig{
		Name:           "test_tool",
		Endpoint:       Endpoint{BaseURL: "https://example.com", Path: "/api", Method: "get"},
		AuthScheme:     &toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		AuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{ClientID: "c"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	res := runTool(t, tool, map[string]any{})
	if errMsg, _ := res["error"].(string); !strings.Contains(errMsg, "authentication") {
		t.Errorf("error = %q, want it to mention authentication", errMsg)
	}
}
