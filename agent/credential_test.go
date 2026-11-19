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

package agent_test

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/adk/v2/agent"
	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/tool/toolauth"
)

type credentialCall struct {
	method          string
	appName, userID string
	config          *toolauth.Config
}

// fakeCredentialService records its calls and answers each with cred and err.
type fakeCredentialService struct {
	calls []credentialCall
	cred  *toolauth.Credential
	err   error
}

func (s *fakeCredentialService) LoadCredential(_ context.Context, appName, userID string, config *toolauth.Config) (*toolauth.Credential, error) {
	s.calls = append(s.calls, credentialCall{"load", appName, userID, config})
	return s.cred, s.err
}

func (s *fakeCredentialService) SaveCredential(_ context.Context, appName, userID string, config *toolauth.Config) error {
	s.calls = append(s.calls, credentialCall{"save", appName, userID, config})
	return s.err
}

func credentialConfig() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2},
		CredentialKey: "key-a",
		ExchangedAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "client-a", ClientSecret: "secret-a"},
		},
	}
}

// credentialOps calls LoadCredential or SaveCredential and returns the
// credential (nil for save) and the error.
var credentialOps = []struct {
	name string
	call func(agent.Context, *toolauth.Config) (*toolauth.Credential, error)
}{
	{name: "load", call: agent.LoadCredential},
	{name: "save", call: func(ctx agent.Context, cfg *toolauth.Config) (*toolauth.Credential, error) {
		return nil, agent.SaveCredential(ctx, cfg)
	}},
}

func TestCredentialCallsWithoutService(t *testing.T) {
	sess := newTestSession(t)
	configs := []struct {
		name   string
		config *toolauth.Config
	}{
		{name: "config", config: credentialConfig()},
		{name: "nil config", config: nil},
	}
	forEachContext(t, sess, services{}, func(t *testing.T, c namedContext) {
		for _, op := range credentialOps {
			for _, cfg := range configs {
				t.Run(op.name+"/"+cfg.name, func(t *testing.T) {
					got, err := op.call(c.ctx, cfg.config)
					if err == nil || err.Error() != "credential service is not initialized" {
						t.Errorf("%s error = %v, want %q", op.name, err, "credential service is not initialized")
					}
					if got != nil {
						t.Errorf("%s credential = %+v, want nil", op.name, got)
					}
				})
			}
		}
	})
}

func TestCredentialCallsNilConfig(t *testing.T) {
	sess := newTestSession(t)
	svc := &fakeCredentialService{}
	goCtx := icredentialservice.WithService(t.Context(), svc)
	forEachContext(t, sess, services{goCtx: goCtx}, func(t *testing.T, c namedContext) {
		for _, op := range credentialOps {
			t.Run(op.name, func(t *testing.T) {
				svc.calls = nil
				got, err := op.call(c.ctx, nil)
				if err == nil {
					t.Errorf("%s(nil config) error = nil, want an error", op.name)
				}
				if got != nil {
					t.Errorf("%s(nil config) credential = %+v, want nil", op.name, got)
				}
				if len(svc.calls) != 0 {
					t.Errorf("service received %d calls for a nil config, want none", len(svc.calls))
				}
			})
		}
	})
}

func TestCredentialCallsPassInvocationIdentity(t *testing.T) {
	sess := newTestSession(t) // app-1, user-1, sess-1
	stored := &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{ClientID: "client-x"}}
	svc := &fakeCredentialService{cred: stored}
	goCtx := icredentialservice.WithService(t.Context(), svc)
	forEachContext(t, sess, services{goCtx: goCtx}, func(t *testing.T, c namedContext) {
		for _, op := range credentialOps {
			t.Run(op.name, func(t *testing.T) {
				svc.calls = nil
				cfg := credentialConfig()
				got, err := op.call(c.ctx, cfg)
				if err != nil {
					t.Fatalf("%s error = %v", op.name, err)
				}
				if len(svc.calls) != 1 {
					t.Fatalf("service received %d calls, want 1", len(svc.calls))
				}
				call := svc.calls[0]
				if call.method != op.name || call.appName != "app-1" || call.userID != "user-1" {
					t.Errorf("service call = %s(%q, %q), want %s(%q, %q)", call.method, call.appName, call.userID, op.name, "app-1", "user-1")
				}
				if call.config != cfg {
					t.Error("service received a different config, want the caller's")
				}
				if op.name == "load" && got != stored {
					t.Errorf("load credential = %+v, want the service's credential %+v", got, stored)
				}
			})
		}
	})
}

func TestCredentialCallsReturnServiceErrorUnchanged(t *testing.T) {
	sess := newTestSession(t)
	errBoom := errors.New("boom")
	svc := &fakeCredentialService{err: errBoom}
	goCtx := icredentialservice.WithService(t.Context(), svc)
	forEachContext(t, sess, services{goCtx: goCtx}, func(t *testing.T, c namedContext) {
		for _, op := range credentialOps {
			t.Run(op.name, func(t *testing.T) {
				got, err := op.call(c.ctx, credentialConfig())
				if err != errBoom {
					t.Errorf("%s error = %v, want the service's error %v unchanged", op.name, err, errBoom)
				}
				if got != nil {
					t.Errorf("%s credential = %+v, want nil", op.name, got)
				}
			})
		}
	})
}
